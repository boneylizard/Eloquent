"""
Monte Carlo market simulation engine.
Generates 10,000 shared scenarios (bull/bear/sideways/crash, volatility, sector rotation, black swan)
and simulates each strategy through all scenarios for fair comparison.
"""
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NUM_SCENARIOS = 10_000
BLACK_SWAN_PROB = 0.02
HORIZON_DAYS = 252  # 1 trading year

# Market regime types
REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_CRASH = "crash"

# Strategy types (AI will generate variations)
STRATEGY_TYPES = [
    "aggressive_growth",
    "conservative_value",
    "index_fund",
    "cash",
    "momentum_trade",
    "contrarian",
    "sector_rotation",
    "dca",
    "hedged_position",
    "dividend_focus",
]


def _generate_market_regime(rng: np.random.Generator) -> str:
    probs = (0.40, 0.25, 0.25, 0.10)
    regimes = [REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR, REGIME_CRASH]
    return str(rng.choice(regimes, p=probs))


def _generate_returns_for_regime(regime: str, rng: np.random.Generator) -> float:
    """Annualized return (log) for regime."""
    mu_map = {
        REGIME_BULL: 0.15,
        REGIME_SIDEWAYS: 0.00,
        REGIME_BEAR: -0.20,
        REGIME_CRASH: -0.40,
    }
    sigma_map = {
        REGIME_BULL: 0.12,
        REGIME_SIDEWAYS: 0.08,
        REGIME_BEAR: 0.20,
        REGIME_CRASH: 0.35,
    }
    mu = mu_map.get(regime, 0.0)
    sigma = sigma_map.get(regime, 0.15)
    return float(rng.normal(mu / 252, sigma / np.sqrt(252), 1)[0])


def _apply_black_swan(return_: float, rng: np.random.Generator) -> float:
    if rng.random() < BLACK_SWAN_PROB:
        return return_ - 0.15  # large negative shock
    return return_


def generate_scenarios(
    seed: Optional[int] = None,
    n: int = NUM_SCENARIOS,
    fed_rate_base: float = 0.05,
    recession_prob: float = 0.15,
) -> List[Dict[str, Any]]:
    """Generate n shared market scenarios for fair strategy comparison."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        regime = _generate_market_regime(rng)
        daily_return = _generate_returns_for_regime(regime, rng)
        daily_return = _apply_black_swan(daily_return, rng)
        vol = float(rng.uniform(0.01, 0.04))
        fed_shock = float(rng.normal(0, 0.005))
        recession = rng.random() < recession_prob
        scenarios.append({
            "id": f"s{i}",
            "regime": regime,
            "daily_return": daily_return,
            "volatility": vol,
            "fed_rate": fed_rate_base + fed_shock,
            "recession": recession,
        })
    return scenarios


def simulate_strategy(
    strategy: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    initial_value: float = 10_000.0,
) -> List[float]:
    """
    Simulate a strategy through all scenarios.
    Returns list of final portfolio values (one per scenario).
    """
    strategy_type = strategy.get("type", "index_fund")
    # Map strategy type to exposure multiplier (0=all cash, 1=full market, >1=leveraged)
    exposure_map = {
        "aggressive_growth": 1.2,
        "conservative_value": 0.7,
        "index_fund": 1.0,
        "cash": 0.0,
        "momentum_trade": 1.1,
        "contrarian": 0.9,
        "sector_rotation": 1.0,
        "dca": 0.85,
        "hedged_position": 0.6,
        "dividend_focus": 0.8,
    }
    exposure = exposure_map.get(strategy_type, 1.0)
    # Contrarian: invert regime effect slightly
    if strategy_type == "contrarian":
        exposure = -0.3 + 0.6 * exposure  # dampen
    # Cash: no market exposure
    if strategy_type == "cash":
        return [initial_value] * len(scenarios)
    values = []
    for s in scenarios:
        r = s["daily_return"]
        vol = s["volatility"]
        adj_r = r * exposure + float(np.random.default_rng(hash(s["id"]) % 2**32).normal(0, vol * 0.5, 1)[0])
        # Compound over horizon
        final = initial_value * np.exp(np.sum([adj_r] * HORIZON_DAYS))
        values.append(max(0.0, float(final)))
    return values


def analyze_strategy_results(
    strategy_id: str,
    strategy_name: str,
    values: List[float],
    initial_value: float,
    sp500_scenario_returns: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute metrics: expected value, median, std, Sharpe, max drawdown, VaR 95%, win rate, profit factor."""
    arr = np.array(values)
    returns = (arr / initial_value) - 1.0
    mean_ret = float(np.mean(returns))
    median_ret = float(np.median(returns))
    std_ret = float(np.std(returns)) if len(returns) > 1 else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    # Max drawdown (simplified: from peak)
    cum = np.maximum.accumulate(arr)
    drawdowns = (arr - cum) / np.where(cum > 0, cum, 1)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    var95 = float(np.percentile(returns, 5))
    win_rate = float(np.mean(returns > 0))
    wins = np.sum(returns > 0)
    losses = np.sum(returns <= 0)
    avg_win = float(np.mean(returns[returns > 0])) if wins > 0 else 0.0
    avg_loss = float(np.mean(np.abs(returns[returns <= 0]))) if losses > 0 else 0.0
    profit_factor = (avg_win * wins) / (avg_loss * losses) if (losses > 0 and avg_loss > 0) else float("inf")
    beat_sp500 = None
    if sp500_scenario_returns is not None:
        beat_sp500 = float(np.mean(returns > np.array(sp500_scenario_returns)))

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "expected_value": float(np.mean(arr)),
        "median_value": float(np.median(arr)),
        "std_value": float(np.std(arr)),
        "mean_return": mean_ret,
        "median_return": median_ret,
        "std_return": std_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "var_95": var95,
        "win_rate": win_rate,
        "profit_factor": min(profit_factor, 10.0),
        "beat_sp500_prob": beat_sp500,
    }


def run_tournament(
    strategies: List[Dict[str, Any]],
    seed: Optional[int] = None,
    initial_value: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Run Monte Carlo tournament: generate scenarios, simulate each strategy, analyze.
    Returns full results for AI to pick winner.
    """
    scenarios = generate_scenarios(seed=seed)
    # S&P 500 proxy: index_fund style
    sp500_values = simulate_strategy({"type": "index_fund"}, scenarios, initial_value)
    sp500_returns = [(v / initial_value) - 1.0 for v in sp500_values]

    results = {}
    for s in strategies:
        sid = s.get("id", "unknown")
        name = s.get("name", s.get("type", sid))
        vals = simulate_strategy(s, scenarios, initial_value)
        results[sid] = {
            "strategy": s,
            "values": vals,
            "analysis": analyze_strategy_results(sid, name, vals, initial_value, sp500_returns),
        }

    # Regime breakdown
    regime_results = {}
    for regime in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_CRASH]:
        idx = [i for i, sc in enumerate(scenarios) if sc["regime"] == regime]
        if not idx:
            regime_results[regime] = {}
            continue
        regime_results[regime] = {}
        for sid, r in results.items():
            sub_vals = [r["values"][i] for i in idx]
            regime_results[regime][sid] = float(np.mean(sub_vals))

    return {
        "run_id": str(uuid.uuid4())[:8],
        "num_scenarios": len(scenarios),
        "scenarios_sample": scenarios[:20],
        "strategy_results": {k: {
            "strategy": v["strategy"],
            "analysis": v["analysis"],
        } for k, v in results.items()},
        "regime_performance": regime_results,
        "sp500_scenario_returns": sp500_returns[:100],
    }
