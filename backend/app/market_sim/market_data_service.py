"""
Market data service using yfinance for live prices.
Provides real-time quotes, S&P 500 baseline, and confidence-weighted data (Nate Silver approach).
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Nate Silver confidence ratings for different data sources
CONFIDENCE_RATINGS = {
    "analyst_ratings": 0.31,
    "prediction_markets": 0.73,
    "fundamental_data": 0.85,
    "news_sentiment": 0.38,
    "insider_trading": 0.62,  # for buys
    "price_history": 0.90,   # raw price/volume
}

SP500_SYMBOL = "^GSPC"


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


class MarketDataService:
    """Fetch live market data via yfinance."""

    def __init__(self):
        self._yf = None

    def _ensure_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol with confidence metadata."""
        self._ensure_yf()
        try:
            ticker = self._yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            if price is None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
            prev_close = _safe_float(info.get("previousClose"))
            if prev_close is None and len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
            change_pct = None
            if price and prev_close and prev_close != 0:
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
            logger.warning("yfinance quote failed for %s: %s", symbol, e)
            return {"symbol": symbol, "price": None, "error": str(e), "confidence": 0.0}

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        out = {}
        for s in symbols:
            out[s] = self.get_quote(s)
        return out

    def get_sp500_value(self) -> Optional[float]:
        """Current S&P 500 index value for baseline comparison."""
        q = self.get_quote(SP500_SYMBOL)
        return q.get("price")

    def get_sp500_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Historical S&P 500 for charting."""
        self._ensure_yf()
        try:
            ticker = self._yf.Ticker(SP500_SYMBOL)
            hist = ticker.history(period=f"{days}d")
            if hist.empty:
                return []
            rows = []
            for dt, row in hist.iterrows():
                rows.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]),
                })
            return rows
        except Exception as e:
            logger.warning("S&P 500 history failed: %s", e)
            return []

    def get_stock_history(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Historical prices for a stock."""
        self._ensure_yf()
        try:
            ticker = self._yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            if hist.empty:
                return []
            rows = []
            for dt, row in hist.iterrows():
                rows.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]),
                })
            return rows
        except Exception as e:
            logger.warning("History failed for %s: %s", symbol, e)
            return []


market_data_service = MarketDataService()
