"""
Fundamentals-based forecast + time-varying blend with polls.
Polls = primary signal; fundamentals = subtle structural prior (gravitational nudge).
See FORECAST_DESIGN.md for formulas and data sources.
"""
import math
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default 2026 general election (Senate/Governor/House)
DEFAULT_ELECTION_DATE = date(2026, 11, 3)

# Midterm: president's party (R) typically loses ~2 pts; we add +2/100 to Dem share as baseline
MIDTERM_PENALTY_D_PTS = 2.0

# When generic ballot is missing, proxy from net approval: G ≈ 0.5 + 0.012 * net_approval (rough)
APPROVAL_TO_GENERIC_SLOPE = 0.012

# --- Prior adjustment parameters (exposed for tuning; do not mutate at runtime) ---
# Baseline influence of fundamentals on state mean (e.g. 0.05–0.20 = 5–20% nudge).
FUNDAMENTAL_WEIGHT_BASE_DEFAULT = 0.10
# Time curve: "decay" = prior influence decays as election nears; "flat" = no time scaling.
TIME_DECAY_CURVE_DEFAULT = "decay"
# Scale for state partisan lean (0 = ignore lean, 1 = full lean). State fundamentals = national + lean_mult * state_lean_pts.
STATE_LEAN_MULTIPLIER_DEFAULT = 1.0

# State partisan lean (Dem two-party share minus 50). Positive = more D than national. 2020 presidential reference.
STATE_LEAN_PTS: Dict[str, float] = {
    "AL": -26.0, "AK": -10.0, "AZ": -3.0, "AR": -27.0, "CA": 29.0, "CO": 14.0, "CT": 20.0, "DE": 23.0,
    "FL": -4.0, "GA": -1.0, "HI": 32.0, "ID": -31.0, "IL": 17.0, "IN": -16.0, "IA": -8.0, "KS": -21.0,
    "KY": -26.0, "LA": -20.0, "ME": 9.0, "MD": 35.0, "MA": 34.0, "MI": 3.0, "MN": 7.0, "MS": -17.0,
    "MO": -16.0, "MT": -16.0, "NE": -20.0, "NV": 3.0, "NH": 1.0, "NJ": 17.0, "NM": 11.0, "NY": 24.0,
    "NC": -1.0, "ND": -34.0, "OH": -8.0, "OK": -34.0, "OR": 17.0, "PA": 2.0, "RI": 21.0, "SC": -12.0,
    "SD": -26.0, "TN": -24.0, "TX": -6.0, "UT": -21.0, "VT": 36.0, "VA": 11.0, "WA": 20.0, "WV": -37.0,
    "WI": 1.0, "WY": -44.0, "DC": 42.0,
}


def compute_fundamentals_national_share(
    approval_data: Optional[Dict[str, Any]] = None,
    generic_ballot_data: Optional[Dict[str, Any]] = None,
    calibration_entries: Optional[List[Dict[str, Any]]] = None,
    calibration_swing_weight: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute national Dem two-party share from fundamentals (no polls).
    Uses: generic ballot > approval proxy > calibration average. Then adds special-election
    swing (weighted) and midterm penalty.
    Returns (dem_share_0_to_100, metadata).
    """
    metadata: Dict[str, Any] = {"source": "fundamentals", "components": {}}
    dem_share = 50.0  # neutral prior

    # 1) Generic ballot (best predictor)
    if generic_ballot_data and isinstance(generic_ballot_data, dict):
        polls_list = generic_ballot_data.get("polls") or generic_ballot_data.get("data") or []
        dem_sum, gop_sum, n = 0.0, 0.0, 0
        for p in polls_list[:30]:
            dem_val = p.get("dem") or p.get("Democrat") or p.get("Candidate 1")
            gop_val = p.get("gop") or p.get("Republican") or p.get("Candidate 2")
            if dem_val is not None and gop_val is not None:
                try:
                    d, g = float(str(dem_val).replace("%", "").strip()), float(str(gop_val).replace("%", "").strip())
                    if d + g > 0:
                        dem_sum += d
                        gop_sum += g
                        n += 1
                except (ValueError, TypeError):
                    pass
        if n > 0 and (dem_sum + gop_sum) > 0:
            dem_share = 100.0 * dem_sum / (dem_sum + gop_sum)
            metadata["components"]["generic_ballot"] = round(dem_share, 2)
            metadata["generic_ballot_n"] = n

    # 2) Approval proxy if no generic ballot
    if "generic_ballot" not in metadata.get("components", {}) and approval_data and isinstance(approval_data, dict):
        polls_list = approval_data.get("polls") or approval_data.get("data") or []
        approve_sum, disapprove_sum, n = 0.0, 0.0, 0
        for p in polls_list[:20]:
            app = p.get("approve") or p.get("Approve")
            dis = p.get("disapprove") or p.get("Disapprove")
            if app is not None and dis is not None:
                try:
                    a, d = float(str(app).replace("%", "").strip()), float(str(dis).replace("%", "").strip())
                    approve_sum += a
                    disapprove_sum += d
                    n += 1
                except (ValueError, TypeError):
                    pass
        if n > 0:
            net = approve_sum - disapprove_sum
            dem_share = 50.0 + APPROVAL_TO_GENERIC_SLOPE * net
            dem_share = max(35.0, min(65.0, dem_share))
            metadata["components"]["approval_proxy"] = round(dem_share, 2)
            metadata["approval_net"] = round(net / n, 1)

    # 3) Special election swing (weighted mean of swing_toward_d) — heavy weight
    swing_pts = 0.0
    swing_weight_total = 0.0
    if calibration_entries:
        for e in calibration_entries:
            s = e.get("swing_toward_d")
            w = float(e.get("weight", 1.0))
            if s is not None and w > 0:
                try:
                    swing_pts += float(s) * w
                    swing_weight_total += w
                except (TypeError, ValueError):
                    pass
    if swing_weight_total > 0:
        swing_pts = swing_pts / swing_weight_total
        effective_swing = swing_pts * max(0.0, min(2.0, float(calibration_swing_weight)))
        # Interpret: positive swing_toward_d = D gained vs 2024. Add to Dem share (in points).
        dem_share += effective_swing
        metadata["components"]["special_election_swing_pts_raw"] = round(swing_pts, 2)
        metadata["components"]["special_election_swing_pts"] = round(effective_swing, 2)
        metadata["calibration_swing_weight"] = calibration_swing_weight
        metadata["calibration_n"] = len([e for e in calibration_entries if e.get("swing_toward_d") is not None])

    # 4) Midterm penalty (president's party loses; we're in midterm cycle)
    dem_share += MIDTERM_PENALTY_D_PTS
    metadata["components"]["midterm_penalty_D_pts"] = MIDTERM_PENALTY_D_PTS

    dem_share = max(30.0, min(70.0, dem_share))
    metadata["dem_share"] = round(dem_share, 2)
    return round(dem_share, 2), metadata


# Weights for combined calibration (generic ballot, approval, special elections). Sum to 1.0.
CALIBRATION_WEIGHT_GENERIC_BALLOT = 0.50
CALIBRATION_WEIGHT_APPROVAL = 0.30
CALIBRATION_WEIGHT_SPECIAL_ELECTION = 0.20

# How much generic ballot D+5 shows up in each race type (impact coefficients).
# House: D+5 → ~D+4.5; Senate: D+5 → ~D+2.25; Governor: D+5 → ~D+1.5.
GENERIC_BALLOT_IMPACT_BY_RACE_TYPE: Dict[str, float] = {
    "house": 0.90,
    "senate": 0.45,
    "governor": 0.30,
}

# How much presidential approval (net) impacts each race type (multiplier on base slope).
# House: most referendum-like; Senate: baseline; Governor: more candidate-specific.
APPROVAL_IMPACT_BY_RACE_TYPE: Dict[str, float] = {
    "house": 1.2,
    "senate": 1.0,
    "governor": 0.8,
}


def _generic_ballot_impact_coefficient(race_type: Optional[str]) -> float:
    """Return the generic ballot impact coefficient for this race type (default 0.45 if unknown)."""
    if not race_type:
        return 0.45
    return GENERIC_BALLOT_IMPACT_BY_RACE_TYPE.get((race_type or "").strip().lower(), 0.45)


def _approval_impact_coefficient(race_type: Optional[str]) -> float:
    """Return the approval impact multiplier for this race type (default 1.0 if unknown)."""
    if not race_type:
        return 1.0
    return APPROVAL_IMPACT_BY_RACE_TYPE.get((race_type or "").strip().lower(), 1.0)


def compute_combined_calibration_shift(
    generic_ballot_dem_share: Optional[float] = None,
    approval_net: Optional[float] = None,
    special_election_swing_pts: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    calibration_weight: float = 1.0,
    president_party: str = "R",
    race_type: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Combine generic ballot, presidential approval, and special-election swing into one calibration shift (points toward D).
    Each signal is converted to "points toward D", then weighted and scaled by calibration_weight (0–1 slider).
    president_party: 'R' or 'D'. For R president, low approval -> positive shift toward D.
    race_type: 'house', 'senate', or 'governor'. Generic ballot impact is scaled by race type (House ~0.9, Senate ~0.55, Governor ~0.35).
    Returns (effective_shift_pts, metadata).
    """
    w_gb = (weights or {}).get("generic_ballot", CALIBRATION_WEIGHT_GENERIC_BALLOT)
    w_app = (weights or {}).get("approval", CALIBRATION_WEIGHT_APPROVAL)
    w_spec = (weights or {}).get("special_election", CALIBRATION_WEIGHT_SPECIAL_ELECTION)
    # Normalize to sum 1
    total = w_gb + w_app + w_spec
    if total <= 0:
        return (0.0, {"combined_shift": 0.0, "effective_shift": 0.0, "components": {}})
    w_gb, w_app, w_spec = w_gb / total, w_app / total, w_spec / total

    gb_impact = _generic_ballot_impact_coefficient(race_type)
    app_impact = _approval_impact_coefficient(race_type)

    components: Dict[str, Any] = {}
    # Generic ballot: Dem share - 50 = points toward D; scale by race-type impact (House > Senate > Governor)
    gb_shift = (float(generic_ballot_dem_share) - 50.0) if generic_ballot_dem_share is not None else None
    if gb_shift is not None:
        components["generic_ballot_dem_share"] = round(generic_ballot_dem_share, 2)
        components["generic_ballot_shift"] = round(gb_shift, 2)
        components["generic_ballot_impact_coefficient"] = round(gb_impact, 2)
    else:
        w_gb = 0.0

    # Approval: for R president, low approval -> D gain. Scale by race type (House > Senate > Governor).
    if approval_net is not None:
        app_shift = -APPROVAL_TO_GENERIC_SLOPE * float(approval_net)
        if president_party.upper() == "D":
            app_shift = -app_shift
        app_shift = app_shift * app_impact
        components["approval_net"] = round(approval_net, 2)
        components["approval_shift"] = round(app_shift, 2)
        components["approval_impact_coefficient"] = round(app_impact, 2)
    else:
        app_shift = 0.0
        w_app = 0.0

    # Special election swing: already in points toward D
    spec_shift = float(special_election_swing_pts) if special_election_swing_pts is not None else None
    if spec_shift is not None:
        components["special_election_swing_pts"] = round(spec_shift, 2)
    else:
        spec_shift = 0.0
        w_spec = 0.0

    # Renormalize weights over available signals
    total_w = w_gb + w_app + w_spec
    if total_w <= 0:
        return (0.0, {"combined_shift": 0.0, "effective_shift": 0.0, "components": components})
    w_gb, w_app, w_spec = w_gb / total_w, w_app / total_w, w_spec / total_w

    # Apply race-type impact to generic ballot: House 0.9, Senate 0.55, Governor 0.35
    effective_gb = (gb_shift * gb_impact) if gb_shift is not None else 0.0
    combined = (w_gb * effective_gb +
                w_app * app_shift +
                w_spec * spec_shift)
    effective = combined * max(0.0, min(2.0, calibration_weight))
    metadata: Dict[str, Any] = {
        "combined_shift": round(combined, 2),
        "effective_shift": round(effective, 2),
        "components": components,
        "weights_used": {"generic_ballot": round(w_gb, 2), "approval": round(w_app, 2), "special_election": round(w_spec, 2)},
        "calibration_weight": calibration_weight,
        "generic_ballot_impact_coefficient": round(gb_impact, 2),
        "approval_impact_coefficient": round(app_impact, 2),
    }
    return (round(effective, 2), metadata)


def blend_weight_polls(
    days_to_election: float,
    inflection_days: float = 90.0,
    steepness: float = 0.08,
) -> float:
    """
    Weight for polls in [0, 1]. (1 - this) = weight for fundamentals.
    At 0 days out -> ~1 (all polls); at 180 days out -> ~0 (all fundamentals).
    Uses logistic: w_polls = 1 / (1 + exp(-k * (inflection_days - days_to_election))).
    """
    if days_to_election <= 0:
        return 1.0
    x = inflection_days - days_to_election
    import math
    w = 1.0 / (1.0 + math.exp(-steepness * x))
    return max(0.0, min(1.0, w))


def blend_forecast(
    fundamentals_dem_share: float,
    poll_dem_share: Optional[float],
    days_to_election: float,
    election_date: Optional[date] = None,
) -> Tuple[float, float]:
    """
    Blend fundamentals and poll average. If poll_dem_share is None, return fundamentals only with full uncertainty.
    Returns (blended_dem_share, w_polls_used).
    """
    if poll_dem_share is None:
        return (fundamentals_dem_share, 0.0)
    w_polls = blend_weight_polls(days_to_election)
    blended = w_polls * poll_dem_share + (1.0 - w_polls) * fundamentals_dem_share
    return (round(blended, 2), w_polls)


def days_to_election(election_date: Optional[date] = None) -> float:
    """Days from today to election. Uses DEFAULT_ELECTION_DATE if not provided."""
    target = election_date or DEFAULT_ELECTION_DATE
    if hasattr(target, "date"):
        target = target.date()
    return (target - date.today()).days


def get_state_lean_pts(
    state_abbr: str,
    state_lean_override: Optional[Dict[str, float]] = None,
) -> float:
    """State partisan lean in points (Dem share - 50). Positive = more D than national. Override takes precedence."""
    state = (state_abbr or "").strip().upper()
    if not state or len(state) != 2:
        return 0.0
    if state_lean_override is not None and state in state_lean_override:
        return float(state_lean_override[state])
    return float(STATE_LEAN_PTS.get(state, 0.0))


def time_factor_for_prior(
    days_to_election: float,
    time_decay_curve: str = "decay",
) -> float:
    """
    Multiplier for prior weight in [0, 1]. Used so prior influence can decay as election nears.
    - "decay": far from election = full weight, close to election = 0 (polls dominate). Linear in days, cap 365.
    - "flat": always 1.0 (no time scaling).
    """
    if time_decay_curve == "flat":
        return 1.0
    days = max(0.0, float(days_to_election))
    # Prior matters more when far out; decays to 0 as we approach (e.g. last 365 days)
    if days >= 365:
        return 1.0
    return min(1.0, days / 365.0)


def compute_fundamentals_prior_adjustment(
    poll_dem_share: float,
    national_fundamentals_share: float,
    state_abbr: str,
    fundamental_weight_base: float = FUNDAMENTAL_WEIGHT_BASE_DEFAULT,
    days_to_election: Optional[float] = None,
    state_lean_multiplier: float = STATE_LEAN_MULTIPLIER_DEFAULT,
    time_decay_curve: str = TIME_DECAY_CURVE_DEFAULT,
    state_lean_override: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Subtle prior nudge: adjustment in points to add to poll_dem_share.
    Final state mean = poll_dem_share + adjustment (polls remain primary; no flattening).
    State fundamentals = national_fundamentals_share + state_lean_multiplier * state_lean_pts.
    adjustment = effective_weight * (state_fundamentals_share - poll_dem_share).
    Returns (adjustment_pts, metadata dict for debugging).
    """
    days = days_to_election if days_to_election is not None else days_to_election()
    time_factor = time_factor_for_prior(days, time_decay_curve)
    effective_weight = fundamental_weight_base * time_factor
    lean_pts = get_state_lean_pts(state_abbr, state_lean_override)
    state_fundamentals_share = national_fundamentals_share + state_lean_multiplier * lean_pts
    # Nudge poll mean toward state fundamentals by a fraction
    adjustment = effective_weight * (state_fundamentals_share - poll_dem_share)
    metadata: Dict[str, Any] = {
        "effective_weight": round(effective_weight, 4),
        "state_lean_pts": round(lean_pts, 2),
        "state_fundamentals_share": round(state_fundamentals_share, 2),
        "adjustment_pts": round(adjustment, 3),
    }
    return (round(adjustment, 3), metadata)


def sigma_from_data_quality(
    poll_count: int,
    pollster_confidence: Optional[float] = None,
    days_to_election: float = 0.0,
    base_sigma: float = 2.0,
) -> float:
    """
    Scale uncertainty by number of polls, quality, and time to election.
    More polls + higher quality + closer to election -> lower sigma.
    """
    # Poll count: 0 -> ~3x, 10+ -> ~1x
    n_mult = 1.0 + 2.0 / (1.0 + max(0, poll_count))
    # Quality: 0–1 confidence -> multiply sigma by 1.5 down to 1.0
    q_mult = 1.5 - 0.5 * (pollster_confidence if pollster_confidence is not None else 0.5)
    # Time: 180 days out -> 1.5x, 0 days -> 1x
    t_mult = 1.0 + min(1.0, max(0.0, days_to_election) / 180.0) * 0.5
    sigma = base_sigma * n_mult * q_mult * t_mult
    return max(1.0, min(6.0, round(sigma, 2)))
