"""
Market data: Yahoo Finance chart API (v8) first, then optional yfinance fallback.
Yahoo often blocks yfinance's quote/crumb endpoints (401 Invalid Crumb) while the
public chart endpoint still works with a browser-like User-Agent.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Nate Silver confidence ratings for different data sources
CONFIDENCE_RATINGS = {
    "analyst_ratings": 0.31,
    "prediction_markets": 0.73,
    "fundamental_data": 0.85,
    "news_sentiment": 0.38,
    "insider_trading": 0.62,
    "price_history": 0.90,
}

SP500_SYMBOL = "^GSPC"


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except (ValueError, TypeError):
        return None


def _yahoo_chart_url(symbol: str, range_param: str, interval: str = "1d") -> str:
    enc = urllib.parse.quote(symbol, safe="")
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
        f"?interval={interval}&range={range_param}"
    )


def _chart_json_ok(data: Optional[dict]) -> bool:
    if not data or not isinstance(data, dict):
        return False
    ch = data.get("chart") or {}
    if ch.get("error"):
        return False
    return bool(ch.get("result"))


def _fetch_yahoo_chart_urllib(symbol: str, range_param: str, interval: str = "1d") -> Optional[dict]:
    url = _yahoo_chart_url(symbol, range_param, interval)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _CHROME_UA,
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        logger.debug("Yahoo chart HTTP %s for %s", e.code, symbol)
        return None
    except Exception as e:
        logger.debug("Yahoo chart (urllib) failed for %s: %s", symbol, e)
        return None


def _fetch_yahoo_chart_cffi(symbol: str, range_param: str, interval: str = "1d") -> Optional[dict]:
    try:
        from curl_cffi import requests as creq
    except ImportError:
        return None
    url = _yahoo_chart_url(symbol, range_param, interval)
    try:
        r = creq.get(url, impersonate="chrome", timeout=25)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        logger.debug("Yahoo chart (curl_cffi) failed for %s: %s", symbol, e)
        return None


def _fetch_yahoo_chart(symbol: str, range_param: str, interval: str = "1d") -> Optional[dict]:
    d = _fetch_yahoo_chart_urllib(symbol, range_param, interval)
    if _chart_json_ok(d):
        return d
    d = _fetch_yahoo_chart_cffi(symbol, range_param, interval)
    if _chart_json_ok(d):
        return d
    return None


def _range_for_history_days(days: int) -> str:
    if days <= 5:
        return "5d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 182:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"


def _quote_from_chart_json(data: Optional[dict], symbol: str) -> Optional[Dict[str, Any]]:
    if not _chart_json_ok(data):
        return None
    res = (data.get("chart") or {}).get("result")[0]
    meta = res.get("meta") or {}
    price = _safe_float(meta.get("regularMarketPrice"))
    prev_close = _safe_float(meta.get("previousClose") or meta.get("chartPreviousClose"))

    quotes = (res.get("indicators") or {}).get("quote") or []
    closes: List[float] = []
    if quotes:
        raw = quotes[0].get("close") or []
        for c in raw:
            v = _safe_float(c)
            if v is not None:
                closes.append(v)
    if price is None and closes:
        price = closes[-1]
    if prev_close is None and len(closes) >= 2:
        prev_close = closes[-2]

    change_pct = None
    if price is not None and prev_close is not None and prev_close != 0:
        change_pct = ((price - prev_close) / prev_close) * 100

    if price is None:
        return None
    return {
        "symbol": symbol,
        "price": price,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "confidence": CONFIDENCE_RATINGS["price_history"],
        "source": "yahoo_chart",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _history_from_chart_json(data: Optional[dict]) -> List[Dict[str, Any]]:
    if not _chart_json_ok(data):
        return []
    res = (data.get("chart") or {}).get("result")[0]
    ts = res.get("timestamp") or []
    quotes = (res.get("indicators") or {}).get("quote") or []
    if not quotes or not ts:
        return []
    closes = quotes[0].get("close") or []
    rows: List[Dict[str, Any]] = []
    last_close: Optional[float] = None
    for t, c in zip(ts, closes):
        v = _safe_float(c)
        if v is None:
            v = last_close
        if v is None:
            continue
        last_close = v
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({"date": dt, "close": float(v)})
    return rows


class MarketDataService:
    """Live quotes and history via Yahoo chart API, with yfinance fallback."""

    def __init__(self):
        self._yf = None

    def _ensure_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf

                self._yf = yf
            except ImportError:
                raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    def _quote_yfinance_fallback(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Use only history + dict-safe info (avoids crumb .info failures that raise inside yfinance)."""
        self._ensure_yf()
        try:
            ticker = self._yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            price = None
            prev_close = None
            if hist is not None and not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    prev_close = float(hist["Close"].iloc[-2])
            info = getattr(ticker, "info", None)
            if isinstance(info, dict):
                cp = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
                if cp is not None:
                    price = cp
                pc = _safe_float(info.get("previousClose"))
                if pc is not None:
                    prev_close = pc
            if price is None:
                return None
            change_pct = None
            if prev_close is not None and prev_close != 0:
                change_pct = ((price - prev_close) / prev_close) * 100
            return {
                "symbol": symbol,
                "price": price,
                "prev_close": prev_close,
                "change_pct": change_pct,
                "confidence": CONFIDENCE_RATINGS["price_history"],
                "source": "yfinance",
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        except Exception as e:
            logger.debug("yfinance fallback quote failed for %s: %s", symbol, e)
            return None

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol with confidence metadata."""
        data = _fetch_yahoo_chart(symbol, "5d", "1d")
        q = _quote_from_chart_json(data, symbol)
        if q:
            return q

        q = self._quote_yfinance_fallback(symbol)
        if q:
            return q

        return {
            "symbol": symbol,
            "price": None,
            "prev_close": None,
            "change_pct": None,
            "error": "Could not fetch quote (Yahoo unreachable or blocked).",
            "confidence": 0.0,
            "source": None,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        return {s: self.get_quote(s) for s in symbols}

    def get_sp500_value(self) -> Optional[float]:
        """Current S&P 500 index value for baseline comparison."""
        q = self.get_quote(SP500_SYMBOL)
        return q.get("price")

    def _history_yfinance(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        self._ensure_yf()
        try:
            ticker = self._yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            if hist is None or hist.empty:
                return []
            rows = []
            for dt, row in hist.iterrows():
                rows.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]),
                })
            return rows
        except Exception as e:
            logger.debug("yfinance history failed for %s: %s", symbol, e)
            return []

    def get_sp500_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Historical S&P 500 for charting."""
        rng = _range_for_history_days(days)
        data = _fetch_yahoo_chart(SP500_SYMBOL, rng, "1d")
        rows = _history_from_chart_json(data)
        if rows:
            return rows[-max(days, 5) :] if len(rows) > max(days, 5) else rows
        return self._history_yfinance(SP500_SYMBOL, days)

    def get_stock_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Historical prices for a stock."""
        rng = _range_for_history_days(days)
        data = _fetch_yahoo_chart(symbol, rng, "1d")
        rows = _history_from_chart_json(data)
        if rows:
            return rows[-max(days, 5) :] if len(rows) > max(days, 5) else rows
        return self._history_yfinance(symbol, days)


market_data_service = MarketDataService()
