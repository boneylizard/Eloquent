"""
Monte Carlo election simulation with correlated polling error.
Uses national + regional + race-specific error so similar states swing together.
Supports calibration from special/off-year election results with custom weights.
"""
import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Calibration: special/off-year results stored here
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "election_calibration_results.json"
STATE_LEG_2024_MARGINS_PATH = Path(__file__).resolve().parent.parent / "data" / "state_leg_2024_presidential_margins.json"


def _load_district_2024_margins() -> Tuple[Dict[int, float], Dict[int, float]]:
    """Read VA_House and NJ_Assembly 2024 Trump margin by district from state_leg_2024_presidential_margins.json (no cache)."""
    va: Dict[int, float] = {}
    nj: Dict[int, float] = {}
    if not STATE_LEG_2024_MARGINS_PATH.exists():
        return (va, nj)
    try:
        data = json.loads(STATE_LEG_2024_MARGINS_PATH.read_text(encoding="utf-8"))
        for k, v in (data.get("VA_House") or {}).items():
            try:
                va[int(k)] = float(v)
            except (ValueError, TypeError):
                pass
        for k, v in (data.get("NJ_Assembly") or {}).items():
            try:
                nj[int(k)] = float(v)
            except (ValueError, TypeError):
                pass
    except Exception as e:
        logger.debug("Could not load state_leg_2024_presidential_margins.json: %s", e)
    return (va, nj)


# VA House districts to skip for 2024 margin (VPAP 2024 used old boundaries; e.g. D4 mismatch).
VA_HOUSE_2024_SKIP_DISTRICTS = {4}


def _va_house_district_from_label(label: str) -> Optional[int]:
    """Parse 'VA House D96 2025 general' -> 96."""
    if not label:
        return None
    m = re.search(r"VA House D(\d+)", label, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _nj_assembly_district_from_label(label: str) -> Optional[int]:
    """Parse 'NJ Assembly D5 2025 general' -> 5."""
    if not label:
        return None
    m = re.search(r"NJ Assembly D(\d+)", label, re.IGNORECASE)
    return int(m.group(1)) if m else None

# Regional groupings for correlated error (MVP: fixed)
REGIONS: Dict[str, List[str]] = {
    "rust_belt": ["WI", "MI", "PA", "OH"],
    "sun_belt": ["AZ", "NV", "GA", "NC", "TX", "FL"],
    "northeast": ["NY", "NJ", "CT", "MA", "NH", "ME", "VT", "RI"],
    "plains_midwest": ["IA", "MN", "NE", "KS", "MO", "SD", "ND"],
    "west": ["CA", "WA", "OR", "CO", "NM"],
    "south": ["SC", "AL", "MS", "LA", "AR", "TN", "KY", "WV", "OK"],
}
# All other states (ID, MT, WY, UT, etc.) map to "other"
STATE_TO_REGION: Dict[str, str] = {}
for region, states in REGIONS.items():
    for s in states:
        STATE_TO_REGION[s] = region

# Default uncertainty (points). Phase 2: vary by poll count / race type.
SIGMA_NATIONAL = 3.0
SIGMA_REGIONAL = 1.5
SIGMA_RACE_BY_TYPE = {"senate": 2.2, "governor": 2.0, "house": 2.5}
# Weights: national + regional + race-specific
W_NAT = 0.60
W_REG = 0.25
W_RACE = 0.15

# Systematic polling error: 40% unbiased, 30% polls underestimated R (add to GOP), 30% underestimated D
SYS_BIAS_PROBS = (0.40, 0.30, 0.30)  # none, pro-R (polls missed R), pro-D (polls missed D)
SYS_BIAS_RANGE = (2.0, 4.0)  # points; applied as shift to Dem share (negative = R gains)


def _get_region(state: str) -> str:
    return STATE_TO_REGION.get(state.upper(), "other")


# --- Calibration: special / off-year results with custom weights ---


def load_calibration() -> List[Dict[str, Any]]:
    """Load calibration results from JSON. Each entry: id, label, type, state, date, dem_actual_pct, poll_avg_pct, weight, region?, note?
    VA/NJ state_house entries are sanitized: trump_2024_margin and swing_toward_d are cleared (we never use state-level margin for state leg districts)."""
    if not CALIBRATION_PATH.exists():
        return []
    try:
        raw = CALIBRATION_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        entries = list(data) if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Failed to load calibration: %s", e)
        return []
    # Fill or clear 2024 margin / swing for state_house: VA/NJ from district file when present; never use state-level
    va_house_2024, nj_assembly_2024 = _load_district_2024_margins()
    # Governor: always refill 2024 margin from current state file (avoids stale/wrong stored values e.g. VA)
    try:
        from .pres_2024_county_loader import load_pres_2024_state_margins
        _governor_margins = load_pres_2024_state_margins()
    except Exception:
        _governor_margins = {}
    out = []
    for e in entries:
        e = dict(e)
        if e.get("type") == "governor":
            state = (e.get("state") or "").upper()
            dem_pct = e.get("dem_actual_pct")
            if state and state in _governor_margins and dem_pct is not None:
                trump_margin = float(_governor_margins[state])
                election_margin_r = 100.0 - 2.0 * float(dem_pct)
                e["trump_2024_margin"] = round(trump_margin, 1)
                e["swing_toward_d"] = round(trump_margin - election_margin_r, 1)
            else:
                e["trump_2024_margin"] = None
                e["swing_toward_d"] = None
            out.append(e)
            continue
        if e.get("type") != "state_house":
            out.append(e)
            continue
        state = (e.get("state") or "").upper()
        dem_pct = e.get("dem_actual_pct")
        trump_margin = None
        if state == "VA":
            dist = _va_house_district_from_label(e.get("label") or "")
            if dist is not None and dist not in VA_HOUSE_2024_SKIP_DISTRICTS and dist in va_house_2024:
                trump_margin = va_house_2024[dist]
        elif state == "NJ":
            dist = _nj_assembly_district_from_label(e.get("label") or "")
            if dist is not None and dist in nj_assembly_2024:
                trump_margin = nj_assembly_2024[dist]
        if trump_margin is not None and dem_pct is not None:
            election_margin_r = 100.0 - 2.0 * float(dem_pct)
            e["trump_2024_margin"] = round(trump_margin, 1)
            e["swing_toward_d"] = round(trump_margin - election_margin_r, 1)
        else:
            e["trump_2024_margin"] = None
            e["swing_toward_d"] = None
        out.append(e)
    return out


def load_calibration_for_analysis() -> List[Dict[str, Any]]:
    """Calibration entries that have 2024 R margin and swing data only. Use for API list and for simulation/forecast; entries without swing are not usable for analysis."""
    return [e for e in load_calibration() if e.get("swing_toward_d") is not None]


def save_calibration(entries: List[Dict[str, Any]]) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def add_calibration_entry(
    label: str,
    entry_type: str,
    state: str,
    date: str,
    dem_actual_pct: float,
    poll_avg_pct: Optional[float] = None,
    weight: float = 1.0,
    region: Optional[str] = None,
    note: Optional[str] = None,
    trump_2024_margin: Optional[float] = None,
    swing_toward_d: Optional[float] = None,
    rep_actual_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Add one calibration result. Overperformance = dem_actual_pct - poll_avg_pct when poll_avg_pct is set (positive = D beat polls).
    Only entries with poll_avg_pct contribute to calibration shift. trump_2024_margin: 2024 pres margin (positive = Trump won)."""
    entries = load_calibration()
    entry: Dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "label": (label or "").strip() or f"{state} {date}",
        "type": (entry_type or "special").strip().lower() or "special",
        "state": (state or "").strip().upper()[:2],
        "date": (date or "").strip(),
        "dem_actual_pct": float(dem_actual_pct),
        "poll_avg_pct": float(poll_avg_pct) if poll_avg_pct is not None else None,
        "weight": max(0.0, min(2.0, float(weight))),
        "region": (region or "").strip() or None,
        "note": (note or "").strip() or None,
    }
    if trump_2024_margin is not None:
        entry["trump_2024_margin"] = float(trump_2024_margin)
    if swing_toward_d is not None:
        entry["swing_toward_d"] = float(swing_toward_d)
    if rep_actual_pct is not None:
        entry["rep_actual_pct"] = float(rep_actual_pct)
    entries.append(entry)
    save_calibration(entries)
    return entry


def delete_calibration_entry(entry_id: str) -> bool:
    entries = load_calibration()
    new_list = [e for e in entries if str(e.get("id")) != str(entry_id)]
    if len(new_list) == len(entries):
        return False
    save_calibration(new_list)
    return True


def compute_calibration_shift(entries: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    From calibration entries, compute weighted mean and weighted std of D overperformance.
    Overperformance = dem_actual_pct - poll_avg_pct (positive = D did better than polls).
    Returns (mean_shift, std). If no entries or zero total weight, returns (0.0, SIGMA_NATIONAL).
    """
    if not entries:
        return 0.0, SIGMA_NATIONAL
    overperformances = []
    weights = []
    for e in entries:
        try:
            poll = e.get("poll_avg_pct")
            if poll is None:
                continue
            actual = float(e.get("dem_actual_pct", 0))
            poll = float(poll)
            # Treat 50 as placeholder "no poll" from legacy imports
            if abs(poll - 50.0) < 0.01:
                continue
            w = float(e.get("weight", 1.0))
            if w <= 0:
                continue
            overperformances.append(actual - poll)
            weights.append(w)
        except (TypeError, ValueError):
            continue
    if not overperformances or not weights:
        return 0.0, SIGMA_NATIONAL
    weights_arr = np.array(weights)
    over_arr = np.array(overperformances)
    total_w = weights_arr.sum()
    if total_w <= 0:
        return 0.0, SIGMA_NATIONAL
    mean_shift = float(np.average(over_arr, weights=weights_arr))
    # Weighted variance, then sqrt for std
    var = np.average((over_arr - mean_shift) ** 2, weights=weights_arr)
    std = float(np.sqrt(var)) if var > 0 else SIGMA_NATIONAL
    return mean_shift, max(1.0, min(6.0, std))


def _draw_systematic_bias(n_simulations: int) -> np.ndarray:
    """Per-sim systematic bias in Dem-share points: 40% 0, 30% -2 to -4 (pro-R), 30% +2 to +4 (pro-D)."""
    u = np.random.random(n_simulations)
    low, high = SYS_BIAS_RANGE
    bias = np.zeros(n_simulations, dtype=np.float64)
    # 0–0.4: 0; 0.4–0.7: pro-R (negative); 0.7–1.0: pro-D (positive)
    pro_r = (u >= SYS_BIAS_PROBS[0]) & (u < SYS_BIAS_PROBS[0] + SYS_BIAS_PROBS[1])
    pro_d = u >= SYS_BIAS_PROBS[0] + SYS_BIAS_PROBS[1]
    bias[pro_r] = -np.random.uniform(low, high, size=pro_r.sum())
    bias[pro_d] = np.random.uniform(low, high, size=pro_d.sum())
    return bias


def run_simulation(
    state_averages: Dict[str, Dict[str, Any]],
    race_type: str = "senate",
    n_simulations: int = 10_000,
    sigma_national: float = SIGMA_NATIONAL,
    sigma_regional: float = SIGMA_REGIONAL,
    sigma_race: Optional[float] = None,
    seed: Optional[int] = None,
    calibration_entries: Optional[List[Dict[str, Any]]] = None,
    use_systematic_error: bool = False,
    quality_sigma_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation with correlated error.

    state_averages: { state_abbr: { "dem_avg": float, "gop_avg": float, "poll_count": int? } }
    calibration_entries: optional list from load_calibration(); used to shift national error mean (D overperformance).
    use_systematic_error: if True, draw 40/30/30 bias (unbiased / polls missed R / polls missed D) per sim.
    quality_sigma_multiplier: optional scale for sigma (e.g. from sigma_from_data_quality); <1 = tighter uncertainty.
    Returns: { state_win_probs, state_stats, n_simulations, elapsed_sec, races_included, calibration_shift? }
    """
    t0 = time.perf_counter()
    logger.info("run_simulation: starting race_type=%s n_simulations=%s states=%s", race_type, n_simulations, len(state_averages))
    if seed is not None:
        np.random.seed(seed)

    national_shift = 0.0
    sigma_national_use = sigma_national
    if calibration_entries:
        national_shift, sigma_from_cal = compute_calibration_shift(calibration_entries)
        sigma_national_use = sigma_from_cal

    sigma_race = sigma_race or SIGMA_RACE_BY_TYPE.get(race_type, 2.0)
    if quality_sigma_multiplier is not None and quality_sigma_multiplier > 0:
        sigma_national_use = sigma_national_use * quality_sigma_multiplier
        sigma_race = sigma_race * quality_sigma_multiplier
    races: List[Tuple[str, float]] = []  # (state, dem_two_way_share)
    for state, avg in state_averages.items():
        state = (state or "").strip().upper()
        if len(state) != 2 or not state.isalpha():
            continue
        dem = avg.get("dem_avg")
        gop = avg.get("gop_avg") or avg.get("rep_avg")
        if dem is None or gop is None:
            continue
        try:
            dem_f = float(dem)
            gop_f = float(gop)
        except (TypeError, ValueError):
            continue
        total = dem_f + gop_f
        if total <= 0:
            continue
        dem_share = dem_f / total * 100.0  # two-way share
        races.append((state, dem_share))

    if not races:
        return {
            "state_win_probs": {},
            "state_stats": {},
            "n_simulations": n_simulations,
            "elapsed_sec": 0,
            "races_included": 0,
            "error": "No state averages with dem_avg/gop_avg",
        }

    states = [r[0] for r in races]
    dem_shares = np.array([r[1] for r in races])
    regions = [_get_region(s) for s in states]
    unique_regions = list(dict.fromkeys(regions))

    # Predraw all random numbers for speed (national mean = calibration shift when provided)
    national_errors = np.random.normal(national_shift, sigma_national_use, n_simulations)
    if use_systematic_error:
        national_errors += _draw_systematic_bias(n_simulations)
    regional_draws = {
        reg: np.random.normal(0, sigma_regional, n_simulations) for reg in unique_regions
    }
    race_noise = np.random.normal(0, sigma_race, (n_simulations, len(states)))

    # (n_sims, n_races): final Dem two-way share per sim per race
    # total_error in points: positive = shift toward D
    total_error = np.zeros((n_simulations, len(states)))
    for i, reg in enumerate(regions):
        total_error[:, i] = (
            W_NAT * national_errors
            + W_REG * regional_draws[reg]
            + W_RACE * race_noise[:, i]
        )
    # dem_share is already in 0-100 scale (two-way %)
    final_shares = dem_shares + total_error
    winners = (final_shares > 50).astype(np.int32)  # 1 = D, 0 = R

    # Aggregate
    dem_win_pct = winners.mean(axis=0) * 100.0
    dem_share_mean = final_shares.mean(axis=0)
    dem_share_p5 = np.percentile(final_shares, 5, axis=0)
    dem_share_p95 = np.percentile(final_shares, 95, axis=0)

    state_win_probs: Dict[str, float] = {}
    state_stats: Dict[str, Dict[str, float]] = {}
    for i, state in enumerate(states):
        state_win_probs[state] = float(round(dem_win_pct[i], 2))
        state_stats[state] = {
            "dem_win_pct": round(dem_win_pct[i], 2),
            "dem_share_mean": round(float(dem_share_mean[i]), 2),
            "dem_share_p5": round(float(dem_share_p5[i]), 2),
            "dem_share_p95": round(float(dem_share_p95[i]), 2),
        }

    # "D wins majority of simulated races" (among these states)
    races_per_sim = winners.sum(axis=1)
    n_races = len(states)
    d_majority = (races_per_sim > n_races / 2).mean() * 100.0

    elapsed_sec = time.perf_counter() - t0
    logger.info("run_simulation: done race_type=%s elapsed_sec=%.2f races_included=%s", race_type, elapsed_sec, len(states))

    out = {
        "state_win_probs": state_win_probs,
        "state_stats": state_stats,
        "d_majority_pct": round(float(d_majority), 2),
        "n_simulations": n_simulations,
        "races_included": len(states),
        "elapsed_sec": round(elapsed_sec, 2),
    }
    if calibration_entries:
        out["calibration_shift"] = round(national_shift, 2)
    return out


def run_and_cache(
    state_averages: Dict[str, Dict[str, Any]],
    race_type: str,
    n_simulations: int = 10_000,
    cache: Optional[Dict[str, Any]] = None,
    cache_key: Optional[str] = None,
    use_calibration: bool = True,
    use_systematic_error: bool = False,
    quality_sigma_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    """Run simulation and optionally store in cache. If use_calibration, loads calibration and shifts national error.
    If use_systematic_error, applies 40/30/30 systematic bias draw per sim.
    quality_sigma_multiplier: optional scale for uncertainty from poll count/time (e.g. from election_forecast.sigma_from_data_quality)."""
    calibration_entries = load_calibration_for_analysis() if use_calibration else None
    out = run_simulation(
        state_averages,
        race_type=race_type,
        n_simulations=n_simulations,
        calibration_entries=calibration_entries,
        use_systematic_error=use_systematic_error,
        quality_sigma_multiplier=quality_sigma_multiplier,
    )
    if cache is not None and cache_key is not None:
        cache[cache_key] = {
            "result": out,
            "race_type": race_type,
            "n_simulations": n_simulations,
            "ts": time.time(),
        }
    return out


def cache_key_for(race_type: str, state_averages: Dict[str, Dict[str, Any]]) -> str:
    """Stable key for caching by race type and state averages (dem_avg, gop_avg per state)."""
    payload = {
        "race_type": race_type,
        "states": {
            s: {"dem": v.get("dem_avg"), "gop": v.get("gop_avg")}
            for s, v in sorted((state_averages or {}).items())
        },
    }
    return "sim_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
