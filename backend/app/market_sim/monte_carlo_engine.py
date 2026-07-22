"""
Monte Carlo market simulation engine.

Shared market paths per scenario; strategies differ by rules (beta, DCA ramp, momentum tilt, etc.).
Calibrated so the unconditional simulated equity index has a modest positive risk premium (~5-7%),
making the tournament actionable for training an AI that still faces real stress (bear/crash slices).
"""
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .market_db import INITIAL_CASH

logger = logging.getLogger(__name__)

NUM_SCENARIOS = 10_000
BLACK_SWAN_PROB = 0.012
HORIZON_DAYS = 252
RISK_FREE_ANNUAL = 0.04

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_CRASH = "crash"

REGIME_ORDER = [REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR, REGIME_CRASH]
# Stationary weights: positive unconditional equity premium for the index (~5-6% annual drift)
REGIME_STATIONARY_P = np.array([0.50, 0.30, 0.15, 0.05], dtype=np.float64)

MU_ANNUAL = {
    REGIME_BULL: 0.16,
    REGIME_SIDEWAYS: 0.02,
    REGIME_BEAR: -0.12,
    REGIME_CRASH: -0.28,
}
SIGMA_ANNUAL = {
    REGIME_BULL: 0.12,
    REGIME_SIDEWAYS: 0.08,
    REGIME_BEAR: 0.20,
    REGIME_CRASH: 0.35,
}

BLACK_SWAN_ANNUAL_SHOCK = 0.08
HALF_YEAR_PERSIST_PROB = 0.70

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

_EXPOSURE_MAP = {
    "aggressive_growth": 1.2,
    "conservative_value": 0.7,
    "index_fund": 1.0,
    "cash": 0.0,
    "momentum_trade": 1.1,
    "contrarian": 0.88,
    "sector_rotation": 1.0,
    "dca": 0.85,
    "hedged_position": 0.6,
    "dividend_focus": 0.8,
}

_MU_VEC = np.array([MU_ANNUAL[r] for r in REGIME_ORDER], dtype=np.float64)
_SIG_VEC = np.array([SIGMA_ANNUAL[r] for r in REGIME_ORDER], dtype=np.float64)


def _regime_to_idx(regime: str) -> int:
    return REGIME_ORDER.index(regime) if regime in REGIME_ORDER else 0


def _generate_market_regime(rng: np.random.Generator) -> str:
    idx = int(rng.choice(4, p=REGIME_STATIONARY_P))
    return REGIME_ORDER[idx]


def generate_scenarios(
    seed: Optional[int] = None,
    n: int = NUM_SCENARIOS,
    fed_rate_base: float = 0.05,
    recession_prob: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Each scenario: starting regime, macro shocks (black swan / recession / tight Fed),
    and idiosyncratic vol scaler. Path builder adds a second-half regime (often persistent).
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        regime_h1 = _generate_market_regime(rng)
        macro_mu_annual = 0.0
        if rng.random() < BLACK_SWAN_PROB:
            macro_mu_annual -= BLACK_SWAN_ANNUAL_SHOCK

        fed_shock = float(rng.normal(0, 0.004))
        fed_rate = float(fed_rate_base + fed_shock)
        recession = bool(rng.random() < recession_prob)

        if recession:
            macro_mu_annual -= 0.035
        if fed_rate > 0.055:
            macro_mu_annual -= float(min(0.032, (fed_rate - 0.055) * 1.15))

        vol_scalar = float(rng.uniform(0.01, 0.04))
        scenarios.append({
            "id": f"s{i}",
            "regime": regime_h1,
            "regime_h1": regime_h1,
            "macro_mu_annual": float(macro_mu_annual),
            "vol_scalar": vol_scalar,
            "fed_rate": fed_rate,
            "recession": recession,
            "daily_return": (MU_ANNUAL[regime_h1] + macro_mu_annual) / 252.0,
            "volatility": vol_scalar,
        })
    return scenarios


def strategy_exposure(strategy_type: str) -> float:
    if strategy_type == "cash":
        return 0.0
    return float(_EXPOSURE_MAP.get(strategy_type, 1.0))


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum of x over trailing `window` days (inclusive)."""
    c = np.cumsum(x, axis=1)
    n, T = x.shape
    out = np.zeros_like(x)
    out[:, window:] = c[:, window:] - c[:, : T - window]
    out[:, :window] = c[:, :window]
    return out


def build_market_paths(
    scenarios: List[Dict[str, Any]],
    seed: Optional[int] = None,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """
    Two halves per scenario: regime may persist or resample (~stationary) for second half.
    Same path shared by all strategies in that scenario.
    """
    rng = np.random.default_rng(seed)
    n = len(scenarios)
    mid = horizon // 2

    reg1_idx = np.array([_regime_to_idx(s["regime_h1"]) for s in scenarios], dtype=np.int64)
    macro = np.array([float(s["macro_mu_annual"]) for s in scenarios], dtype=np.float64)
    vol_idio = np.array([float(s["vol_scalar"]) for s in scenarios], dtype=np.float64)

    persist = rng.random(n) < HALF_YEAR_PERSIST_PROB
    reg2_rand = rng.choice(4, size=n, p=REGIME_STATIONARY_P)
    reg2_idx = np.where(persist, reg1_idx, reg2_rand)

    mu_a1 = _MU_VEC[reg1_idx] + macro
    mu_a2 = _MU_VEC[reg2_idx] + macro
    sig_a1 = _SIG_VEC[reg1_idx]
    sig_a2 = _SIG_VEC[reg2_idx]

    sig_d1 = np.sqrt((sig_a1 / np.sqrt(252.0)) ** 2 + (vol_idio * 0.5) ** 2)
    sig_d2 = np.sqrt((sig_a2 / np.sqrt(252.0)) ** 2 + (vol_idio * 0.5) ** 2)
    mu_d1 = mu_a1 / 252.0
    mu_d2 = mu_a2 / 252.0

    paths = np.empty((n, horizon), dtype=np.float64)
    paths[:, :mid] = mu_d1[:, np.newaxis] + rng.standard_normal((n, mid)) * sig_d1[:, np.newaxis]
    paths[:, mid:] = mu_d2[:, np.newaxis] + rng.standard_normal((n, horizon - mid)) * sig_d2[:, np.newaxis]
    np.clip(paths, -0.40, 0.40, out=paths)
    return paths


def _finalize_wealth(strat_daily: np.ndarray, initial_value: float) -> Tuple[np.ndarray, np.ndarray]:
    log_cum = np.cumsum(strat_daily, axis=1)
    wealth = initial_value * np.exp(log_cum)
    finals = wealth[:, -1]
    peak = np.maximum.accumulate(wealth, axis=1)
    safe_peak = np.maximum(peak, 1e-12)
    dd = (wealth - safe_peak) / safe_peak
    max_dd = np.min(dd, axis=1)
    return finals, max_dd


def simulate_strategy_paths(
    strategy: Dict[str, Any],
    market_paths: np.ndarray,
    initial_value: float = INITIAL_CASH,
) -> Tuple[np.ndarray, np.ndarray]:
    strategy_type = strategy.get("type", "index_fund")
    m = market_paths
    n, T = m.shape

    if strategy_type == "cash":
        return np.full(n, initial_value, dtype=np.float64), np.zeros(n, dtype=np.float64)

    exp = strategy_exposure(strategy_type)
    tbill_log_daily = 0.038 / 252.0
    div_log_daily = 0.014 / 252.0

    if strategy_type == "dca":
        ramp = np.minimum(1.0, (np.arange(T, dtype=np.float64) + 1.0) / 126.0)
        strat_daily = m * exp * ramp[np.newaxis, :]
    elif strategy_type == "momentum_trade":
        roll = _rolling_sum(m, 10)
        mult = np.where(roll > 0, 1.11, 0.89)
        strat_daily = m * exp * mult
    elif strategy_type == "contrarian":
        roll = _rolling_sum(m, 10)
        mult = np.where(roll < 0, 1.09, 0.93)
        strat_daily = m * exp * mult
    elif strategy_type == "hedged_position":
        strat_daily = 0.55 * m * exp + 0.45 * tbill_log_daily
    elif strategy_type == "dividend_focus":
        strat_daily = m * exp + div_log_daily
    elif strategy_type == "sector_rotation":
        strat_daily = m * exp * 0.96 + np.full((n, T), 0.04 * (0.028 / 252.0), dtype=np.float64)
    else:
        strat_daily = m * exp

    return _finalize_wealth(strat_daily, initial_value)


def analyze_strategy_results(
    strategy_id: str,
    strategy_name: str,
    values: List[float],
    initial_value: float,
    sp500_scenario_returns: Optional[List[float]] = None,
    max_drawdown_per_path: Optional[List[float]] = None,
    strategy_type: Optional[str] = None,
) -> Dict[str, Any]:
    arr = np.array(values, dtype=np.float64)
    returns = (arr / initial_value) - 1.0
    mean_ret = float(np.mean(returns))
    median_ret = float(np.median(returns))
    std_ret = float(np.std(returns)) if len(returns) > 1 else 0.0
    sharpe_scenario = (mean_ret / std_ret) if std_ret > 1e-12 else 0.0
    excess_mean = mean_ret - RISK_FREE_ANNUAL
    sharpe_vs_rf = (excess_mean / std_ret) if std_ret > 1e-12 else 0.0

    if max_drawdown_per_path is not None and len(max_drawdown_per_path) == len(values):
        max_dd = float(np.median(np.array(max_drawdown_per_path, dtype=np.float64)))
    else:
        max_dd = 0.0

    var95 = float(np.percentile(returns, 5))
    win_rate = float(np.mean(returns > 0))
    wins = int(np.sum(returns > 0))
    losses = int(np.sum(returns <= 0))
    avg_win = float(np.mean(returns[returns > 0])) if wins > 0 else 0.0
    avg_loss = float(np.mean(np.abs(returns[returns <= 0]))) if losses > 0 else 0.0
    profit_factor = (avg_win * wins) / (avg_loss * losses) if (losses > 0 and avg_loss > 0) else float("inf")

    beat_sp500: Optional[float] = None
    beat_sp500_note: Optional[str] = None
    if sp500_scenario_returns is not None:
        sp_arr = np.array(sp500_scenario_returns, dtype=np.float64)
        if strategy_type == "index_fund":
            beat_sp500 = None
            beat_sp500_note = "Not applicable: identical 1.0× market path to the benchmark."
        else:
            beat_sp500 = float(np.mean((returns - sp_arr) > 1e-7))

    win_rate_note: Optional[str] = None
    if strategy_type == "cash":
        win_rate_note = (
            "Cash is modeled as 0% return every scenario, so win rate is 0% by definition—not a performance score."
        )

    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type or "",
        "expected_value": float(np.mean(arr)),
        "median_value": float(np.median(arr)),
        "std_value": float(np.std(arr)),
        "mean_return": mean_ret,
        "median_return": median_ret,
        "std_return": std_ret,
        "sharpe_ratio": sharpe_scenario,
        "sharpe_excess_vs_4pct_rf": sharpe_vs_rf,
        "max_drawdown": max_dd,
        "var_95": var95,
        "win_rate": win_rate,
        "profit_factor": min(profit_factor, 10.0),
        "beat_sp500_prob": beat_sp500,
        "beat_sp500_note": beat_sp500_note,
        "win_rate_note": win_rate_note,
        "simulation_note": (
            "Two-regime year (half-year segments), macro shocks, strategy-specific rules (DCA/momentum/etc.). "
            "max_drawdown = median intrayear max DD."
        ),
    }


def run_tournament(
    strategies: List[Dict[str, Any]],
    seed: Optional[int] = None,
    initial_value: float = INITIAL_CASH,
) -> Dict[str, Any]:
    scenarios = generate_scenarios(seed=seed)
    path_seed = (seed + 7919) if seed is not None else None
    market_paths = build_market_paths(scenarios, seed=path_seed)

    sp500_finals, sp500_mdd = simulate_strategy_paths({"type": "index_fund"}, market_paths, initial_value)
    sp500_values = sp500_finals.tolist()
    sp500_returns = [(v / initial_value) - 1.0 for v in sp500_values]
    sp500_arr = np.array(sp500_returns, dtype=np.float64)

    results = {}
    for s in strategies:
        sid = s.get("id", "unknown")
        name = s.get("name", s.get("type", sid))
        finals, mdd = simulate_strategy_paths(s, market_paths, initial_value)
        vals = finals.tolist()
        results[sid] = {
            "strategy": s,
            "values": vals,
            "analysis": analyze_strategy_results(
                sid,
                name,
                vals,
                initial_value,
                sp500_returns,
                max_drawdown_per_path=mdd.tolist(),
                strategy_type=s.get("type"),
            ),
        }

    regime_results = {}
    for regime in REGIME_ORDER:
        idx = [i for i, sc in enumerate(scenarios) if sc["regime"] == regime]
        if not idx:
            regime_results[regime] = {}
            continue
        regime_results[regime] = {}
        for sid, r in results.items():
            sub_vals = [r["values"][i] for i in idx]
            regime_results[regime][sid] = float(np.mean(sub_vals))

    # Theoretical unconditional drift (before noise) for transparency
    e_mu = float(np.dot(REGIME_STATIONARY_P, _MU_VEC))

    simulation_calibration = {
        "design_intent": (
            "Equity index has positive unconditional drift (~5-7% annual) in regime mix; "
            "recession/tight Fed and rare shocks trim some paths. Second-half regime can differ."
        ),
        "stationary_regime_weights": {REGIME_ORDER[i]: float(REGIME_STATIONARY_P[i]) for i in range(4)},
        "unconditional_regime_mu_annual_approx": round(e_mu, 4),
        "simulated_index_mean_return": round(float(np.mean(sp500_arr)), 4),
        "simulated_index_median_return": round(float(np.median(sp500_arr)), 4),
        "simulated_index_pct_positive_scenarios": round(float(np.mean(sp500_arr > 0)), 4),
    }

    return {
        "run_id": str(uuid.uuid4())[:8],
        "num_scenarios": len(scenarios),
        "horizon_days": HORIZON_DAYS,
        "scenarios_sample": scenarios[:20],
        "strategy_results": {k: {
            "strategy": v["strategy"],
            "analysis": v["analysis"],
        } for k, v in results.items()},
        "regime_performance": regime_results,
        "sp500_scenario_returns": sp500_returns[:100],
        "simulation_calibration": simulation_calibration,
    }
