"""
Build 2024 presidential state-level margins from county-level data.

Uses [tonmcg/US_County_Level_Election_Results_08-24](https://github.com/tonmcg/US_County_Level_Election_Results_08-24)
CSV: 2024_US_County_Level_Presidential_Results.csv. Aggregates by state (sum votes_gop, votes_dem),
computes two-party Trump margin in points (positive = Trump won state by X). Used for governor races
(same electorate as state). Not used for state legislative districts (county ≠ district).
"""

import csv
import io
import logging
from pathlib import Path
from typing import Dict

import httpx

logger = logging.getLogger(__name__)

URL_2024_COUNTY_CSV = "https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/master/2024_US_County_Level_Presidential_Results.csv"
PRES_2024_STATE_MARGINS_JSON = Path(__file__).resolve().parent.parent / "data" / "pres_2024_state_margins.json"

# CSV state_name -> 2-letter abbreviation (for aggregation output)
STATE_NAME_TO_ABBR: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}


def fetch_and_aggregate_2024_state_margins() -> Dict[str, float]:
    """
    Fetch 2024 county-level CSV from tonmcg repo, aggregate by state, return Trump margin
    (R - D in two-party) in points per state. Positive = Trump won that state by X.
    """
    margins: Dict[str, float] = {}
    state_gop: Dict[str, int] = {}
    state_dem: Dict[str, int] = {}
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        r = client.get(URL_2024_COUNTY_CSV)
        r.raise_for_status()
        text = r.text
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        state_name = (row.get("state_name") or "").strip()
        if not state_name:
            continue
        try:
            gop = int(row.get("votes_gop") or 0)
            dem = int(row.get("votes_dem") or 0)
        except (ValueError, TypeError):
            continue
        state_gop[state_name] = state_gop.get(state_name, 0) + gop
        state_dem[state_name] = state_dem.get(state_name, 0) + dem
    for state_name, gop in state_gop.items():
        dem = state_dem.get(state_name, 0)
        total = gop + dem
        if total <= 0:
            continue
        # Two-party Trump margin in points: (R - D) / (R + D) * 100
        margin_pts = round(100.0 * (gop - dem) / total, 2)
        abbr = STATE_NAME_TO_ABBR.get(state_name)
        if abbr:
            margins[abbr] = margin_pts
    logger.info("Aggregated 2024 state margins from county CSV: %d states", len(margins))
    return margins


def write_pres_2024_state_margins_json() -> Path:
    """Fetch county data, aggregate by state, write pres_2024_state_margins.json. Returns path."""
    margins = fetch_and_aggregate_2024_state_margins()
    PRES_2024_STATE_MARGINS_JSON.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(PRES_2024_STATE_MARGINS_JSON, "w", encoding="utf-8") as f:
        json.dump(margins, f, indent=2)
    logger.info("Wrote %s", PRES_2024_STATE_MARGINS_JSON)
    return PRES_2024_STATE_MARGINS_JSON


def load_pres_2024_state_margins() -> Dict[str, float]:
    """
    Load state-level 2024 Trump margins. Uses pres_2024_state_margins.json if present
    (built from tonmcg county data); otherwise returns empty dict (caller uses fallback).
    """
    if not PRES_2024_STATE_MARGINS_JSON.exists():
        return {}
    try:
        import json
        with open(PRES_2024_STATE_MARGINS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception as e:
        logger.warning("Could not load %s: %s", PRES_2024_STATE_MARGINS_JSON, e)
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = write_pres_2024_state_margins_json()
    print("Wrote", path)
    margins = load_pres_2024_state_margins()
    print("Sample (VA, NJ):", {k: margins[k] for k in ["VA", "NJ"] if k in margins})
