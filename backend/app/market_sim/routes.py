"""FastAPI routes for Market Simulator."""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body, Request, Query

from .market_db import INITIAL_CASH, market_db
from .market_data_service import market_data_service
from .monte_carlo_engine import run_tournament
from .trading_agent import generate_strategies
from .trading_agent import run_ai_tournament_and_decide

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-sim", tags=["Market Simulator"])


@router.get("/status")
async def get_status():
    """Health check and feature availability."""
    try:
        sp = market_data_service.get_sp500_value()
        data_ok = sp is not None and sp > 0
    except Exception:
        data_ok = False
    return {
        "ok": True,
        "market_data": data_ok,
        "portfolio_initialized": market_db.get_portfolio() is not None,
    }


@router.get("/portfolio")
async def get_portfolio():
    """Current portfolio state."""
    p = market_db.init_portfolio_if_needed()
    sp500 = market_data_service.get_sp500_value()
    return {"portfolio": p, "sp500_current": sp500}


@router.post("/portfolio/reset")
async def reset_portfolio_to_starting_capital():
    """Append a new state: full cash at INITIAL_CASH, empty positions. Does not delete trade/tournament history."""
    market_db.init_portfolio_if_needed()
    sp500 = market_data_service.get_sp500_value()
    p = market_db.reset_to_starting_capital(sp500_value=sp500)
    return {"portfolio": p, "sp500_current": sp500, "starting_capital": INITIAL_CASH}


@router.get("/quotes")
async def get_quotes(symbols: str = Query("SPY,AAPL,^GSPC", description="Comma-separated symbols")):
    """Batch quotes for symbols."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    quotes = market_data_service.get_quotes_batch(sym_list)
    return {"quotes": quotes}


@router.get("/sp500/history")
async def get_sp500_history(days: int = Query(30, ge=5, le=365)):
    """S&P 500 historical for chart."""
    rows = market_data_service.get_sp500_history(days)
    return {"data": rows}


@router.get("/trades")
async def get_trades(limit: int = Query(100, ge=1, le=500)):
    """Trade history."""
    trades = market_db.get_trades(limit)
    return {"trades": trades}


@router.get("/tournament/latest")
async def get_latest_tournament():
    """Latest Monte Carlo tournament results."""
    t = market_db.get_latest_tournament()
    if not t:
        return {"tournament": None}
    return {"tournament": t}


@router.post("/tournament/run")
async def run_tournament_endpoint(
    request: Request,
    payload: dict = Body(default={}),
):
    """
    Run Monte Carlo tournament: generate 10 strategies, 10k scenarios, AI selects winner.
    Optionally execute the winning strategy (simulated trade).
    """
    model_id = payload.get("model")
    execute_trade = payload.get("execute_trade", False)

    # Build market context
    sp500 = market_data_service.get_sp500_value()
    quotes = market_data_service.get_quotes_batch(["SPY", "AAPL", "^GSPC"])
    p = market_db.get_portfolio()
    if not p:
        p = market_db.init_portfolio_if_needed(sp500)
    recent_trades = market_db.get_trades(10)
    market_context = {
        "sp500": sp500,
        "quotes": quotes,
        "portfolio": p,
        "advisor_session": {
            "tournaments_completed_prior": market_db.count_tournaments(),
            "recent_trades": [
                {
                    "symbol": t.get("symbol"),
                    "side": t.get("side"),
                    "shares": t.get("shares"),
                    "total": t.get("total"),
                    "created_at": t.get("created_at"),
                    "strategy_id": t.get("strategy_id"),
                }
                for t in recent_trades
            ],
            "note": (
                "Paper capital for training, but you must treat it as a live fiduciary mandate. "
                "Decisions compound over time across tournaments and trades."
            ),
        },
    }

    model_manager = getattr(request.app.state, "model_manager", None)
    primary_model = getattr(request.app.state, "primary_model", None)
    if not primary_model and hasattr(request.app.state, "active_model"):
        primary_model = getattr(request.app.state, "active_model", None)

    result = await run_ai_tournament_and_decide(
        market_context=market_context,
        model_id=model_id,
        model_manager=model_manager,
        primary_model=primary_model,
    )
    tournament = result["tournament"]

    # Save tournament to DB
    strategies = [r["strategy"] for r in tournament["strategy_results"].values()]
    results_summary = {
        k: {"analysis": v["analysis"]}
        for k, v in tournament["strategy_results"].items()
    }
    results_with_regime = {
        **results_summary,
        "_regime_performance": tournament.get("regime_performance"),
        "_simulation_calibration": tournament.get("simulation_calibration"),
    }
    market_db.save_tournament(
        run_id=tournament["run_id"],
        strategies=strategies,
        results=results_with_regime,
        scenarios_preview={"sample": tournament.get("scenarios_sample", [])[:5]},
        winner_id=result.get("winner_id"),
        winner_reasoning=result.get("reasoning"),
    )

    # Save portfolio snapshot after tournament
    try:
        p = market_db.get_portfolio()
        if p:
            sp500 = market_data_service.get_sp500_value()
            market_db.save_snapshot(
                total_value=p["total_value"],
                cash=p["cash"],
                positions=p.get("positions", {}),
                sp500_value=sp500,
            )
    except Exception as ex:
        logger.warning("Failed to save snapshot after tournament: %s", ex)

    # Optionally execute (simulated) trade
    trade_executed = None
    if execute_trade and result.get("trade_action", {}).get("action") == "buy":
        action = result["trade_action"]
        symbol = action.get("symbol", "SPY")
        q = quotes.get(symbol) or market_data_service.get_quote(symbol)
        price = q.get("price")
        p = market_db.get_portfolio()
        if price and price > 0 and p:
            cash = p.get("cash", INITIAL_CASH)
            shares = cash * 0.95 / price  # Up to 95% of cash (no arbitrary share cap)
            if shares >= 0.01:
                total = round(shares * price, 2)
                tid = market_db.record_trade(
                    symbol=symbol,
                    side="buy",
                    shares=round(shares, 2),
                    price=price,
                    strategy_id=result.get("winner_id"),
                    strategy_name=action.get("description", "AI Strategy"),
                    ai_reasoning=result.get("reasoning"),
                    confidence_scores={"tournament": result.get("confidence", 0.5)},
                )
                # Update portfolio: subtract cash, add position
                new_cash = round(cash - total, 2)
                positions = dict(p.get("positions", {}))
                positions[symbol] = round(positions.get(symbol, 0) + shares, 4)
                new_total = new_cash
                for sym, sh in positions.items():
                    qt = market_data_service.get_quote(sym)
                    if qt.get("price"):
                        new_total += sh * qt["price"]
                market_db.update_portfolio(new_cash, positions, round(new_total, 2), sp500)
                trade_executed = {"id": tid, "symbol": symbol, "shares": shares, "price": price}

    return {
        "run_id": tournament["run_id"],
        "winner_id": result.get("winner_id"),
        "reasoning": result.get("reasoning"),
        "confidence": result.get("confidence"),
        "trade_action": result.get("trade_action"),
        "tournament": {
            "strategies": strategies,
            "results": results_summary,
            "regime_performance": tournament.get("regime_performance"),
            "simulation_calibration": tournament.get("simulation_calibration"),
        },
        "trade_executed": trade_executed,
    }


@router.get("/snapshots")
async def get_snapshots(limit: int = Query(100, ge=1, le=500)):
    """Portfolio value snapshots for chart."""
    snaps = market_db.get_snapshots(limit)
    return {"snapshots": snaps}


@router.post("/snapshot")
async def save_snapshot():
    """Manually save a portfolio snapshot (e.g. after trade)."""
    p = market_db.get_portfolio()
    if not p:
        p = market_db.init_portfolio_if_needed()
    sp500 = market_data_service.get_sp500_value()
    market_db.save_snapshot(
        total_value=p["total_value"],
        cash=p["cash"],
        positions=p.get("positions", {}),
        sp500_value=sp500,
    )
    return {"ok": True}
