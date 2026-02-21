"""
Market Simulator SQLite database layer.
Tracks portfolio state, trades, strategy performance, and Monte Carlo results.
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "market_sim.db"
INITIAL_CASH = 10_000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class MarketDB:
    """SQLite database for trades, portfolio snapshots, and strategy performance."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL DEFAULT 10000.0,
                    positions_json TEXT NOT NULL DEFAULT '{}',
                    total_value REAL NOT NULL,
                    sp500_baseline REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    total REAL NOT NULL,
                    strategy_id TEXT,
                    strategy_name TEXT,
                    ai_reasoning TEXT,
                    confidence_scores_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_tournaments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    strategies_json TEXT NOT NULL,
                    scenarios_json TEXT,
                    results_json TEXT NOT NULL,
                    winner_id TEXT,
                    winner_reasoning TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    sp500_value REAL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at);
                CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_created ON portfolio_snapshots(created_at);
            """)
        logger.info("Market sim DB initialized at %s", self.path)

    def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get current portfolio state. Returns None if never initialized."""
        with self._conn() as c:
            row = c.execute(
                "SELECT id, cash, positions_json, total_value, sp500_baseline, created_at FROM portfolio ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        positions = json.loads(row["positions_json"] or "{}")
        return {
            "id": row["id"],
            "cash": float(row["cash"]),
            "positions": positions,
            "total_value": float(row["total_value"]),
            "sp500_baseline": float(row["sp500_baseline"]) if row["sp500_baseline"] is not None else None,
            "created_at": row["created_at"],
        }

    def init_portfolio_if_needed(self, sp500_value: Optional[float] = None) -> Dict[str, Any]:
        """Initialize portfolio with $10,000 if not exists."""
        p = self.get_portfolio()
        if p:
            return p
        with self._conn() as c:
            c.execute(
                "INSERT INTO portfolio (cash, positions_json, total_value, sp500_baseline, created_at) VALUES (?, ?, ?, ?, ?)",
                (INITIAL_CASH, "{}", INITIAL_CASH, sp500_value, _now_iso()),
            )
        return self.get_portfolio()

    def update_portfolio(
        self,
        cash: float,
        positions: Dict[str, float],
        total_value: float,
        sp500_value: Optional[float] = None,
    ) -> None:
        positions_json = json.dumps(positions)
        with self._conn() as c:
            c.execute(
                "INSERT INTO portfolio (cash, positions_json, total_value, sp500_baseline, created_at) VALUES (?, ?, ?, ?, ?)",
                (cash, positions_json, total_value, sp500_value, _now_iso()),
            )

    def record_trade(
        self,
        symbol: str,
        side: str,
        shares: float,
        price: float,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        ai_reasoning: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
    ) -> int:
        total = shares * price
        conf_json = json.dumps(confidence_scores or {})
        with self._conn() as c:
            c.execute(
                """INSERT INTO trades (symbol, side, shares, price, total, strategy_id, strategy_name, ai_reasoning, confidence_scores_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, side, shares, price, total, strategy_id, strategy_name, ai_reasoning, conf_json, _now_iso()),
            )
            return c.lastrowid

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            conf = json.loads(r["confidence_scores_json"] or "{}")
            out.append({
                "id": r["id"],
                "symbol": r["symbol"],
                "side": r["side"],
                "shares": float(r["shares"]),
                "price": float(r["price"]),
                "total": float(r["total"]),
                "strategy_id": r["strategy_id"],
                "strategy_name": r["strategy_name"],
                "ai_reasoning": r["ai_reasoning"],
                "confidence_scores": conf,
                "created_at": r["created_at"],
            })
        return out

    def save_tournament(
        self,
        run_id: str,
        strategies: List[Dict],
        results: Dict,
        scenarios_preview: Optional[Dict] = None,
        winner_id: Optional[str] = None,
        winner_reasoning: Optional[str] = None,
    ) -> None:
        strategies_json = json.dumps(strategies)
        results_json = json.dumps(results)
        scenarios_json = json.dumps(scenarios_preview or {}) if scenarios_preview else None
        with self._conn() as c:
            c.execute(
                """INSERT INTO strategy_tournaments (run_id, strategies_json, scenarios_json, results_json, winner_id, winner_reasoning, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, strategies_json, scenarios_json, results_json, winner_id, winner_reasoning, _now_iso()),
            )

    def get_latest_tournament(self) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM strategy_tournaments ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        results = json.loads(row["results_json"] or "{}")
        regime_performance = results.pop("_regime_performance", None)
        return {
            "run_id": row["run_id"],
            "strategies": json.loads(row["strategies_json"] or "[]"),
            "scenarios_preview": json.loads(row["scenarios_json"] or "{}") if row["scenarios_json"] else None,
            "results": results,
            "regime_performance": regime_performance,
            "winner_id": row["winner_id"],
            "winner_reasoning": row["winner_reasoning"],
            "created_at": row["created_at"],
        }

    def save_snapshot(self, total_value: float, cash: float, positions: Dict[str, float], sp500_value: Optional[float] = None) -> None:
        positions_json = json.dumps(positions)
        with self._conn() as c:
            c.execute(
                "INSERT INTO portfolio_snapshots (total_value, cash, positions_json, sp500_value, created_at) VALUES (?, ?, ?, ?, ?)",
                (total_value, cash, positions_json, sp500_value, _now_iso()),
            )

    def get_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "total_value": float(r["total_value"]),
                "cash": float(r["cash"]),
                "positions": json.loads(r["positions_json"] or "{}"),
                "sp500_value": float(r["sp500_value"]) if r["sp500_value"] is not None else None,
                "created_at": r["created_at"],
            })
        return out


market_db = MarketDB()
