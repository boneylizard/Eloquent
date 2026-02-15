"""
Fetch VPAP 2024 presidential results by VA House of Delegates district and write
Trump margin per district to state_leg_2024_presidential_margins.json (VA_House).
Then VA House calibration rows get 2024 R margin and Swing (D) from district-level data.
"""

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

VPAP_VA_HOUSE_CSV = "https://vpap-production.s3.amazonaws.com/media/visuals/custom/static/2024/11/presidential-results-by-district/assets/2024_Presidential_results_by_house_of_delegates_district.csv"
STATE_LEG_MARGINS_PATH = Path(__file__).resolve().parent.parent / "data" / "state_leg_2024_presidential_margins.json"

# VA House districts to omit from 2024 margins (boundary change or bad match). VPAP 2024 is by old map;
# e.g. D4 in VPAP was D+50.6 but 2025 D4 went 37 D / 63 R — not comparable.
VA_HOUSE_2024_EXCLUDE_DISTRICTS = {4}


def _district_code_to_num(code: str) -> Optional[int]:
    """HD001 -> 1, HD100 -> 100."""
    if not code:
        return None
    m = re.match(r"^HD(\d+)$", code.strip().upper())
    return int(m.group(1)) if m else None


def fetch_va_house_2024_margins() -> Dict[int, float]:
    """Fetch VPAP CSV and return dict district_number -> Trump margin in points (R - D, two-party)."""
    with httpx.Client(follow_redirects=True, timeout=20.0) as client:
        r = client.get(VPAP_VA_HOUSE_CSV)
        r.raise_for_status()
        text = r.text
    margins: Dict[int, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        code = row.get("district_code") or ""
        num = _district_code_to_num(code)
        if num is None:
            continue
        try:
            votes_dem = float(row.get("votes_dem") or 0)
            votes_rep = float(row.get("votes_rep") or 0)
        except (ValueError, TypeError):
            continue
        total_2p = votes_dem + votes_rep
        if total_2p <= 0:
            continue
        if num in VA_HOUSE_2024_EXCLUDE_DISTRICTS:
            continue
        # Trump margin in points: (R - D) two-party
        margin_pts = round(100.0 * (votes_rep - votes_dem) / total_2p, 1)
        margins[num] = margin_pts
    logger.info("VPAP VA House 2024: %d districts", len(margins))
    return margins


def write_va_house_to_state_leg_margins() -> Path:
    """Fetch VPAP VA House 2024, merge into state_leg_2024_presidential_margins.json (VA_House only), save. Returns path."""
    va_margins = fetch_va_house_2024_margins()
    # Load existing so we don't wipe NJ_Assembly
    data: Dict = {}
    if STATE_LEG_MARGINS_PATH.exists():
        try:
            with open(STATE_LEG_MARGINS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Could not load existing %s: %s", STATE_LEG_MARGINS_PATH, e)
    data["VA_House"] = {str(k): v for k, v in va_margins.items()}
    STATE_LEG_MARGINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_LEG_MARGINS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote VA_House (%d districts) to %s", len(va_margins), STATE_LEG_MARGINS_PATH)
    # Clear ballotpedia_scraper cache so next scrape sees the new data
    try:
        from . import ballotpedia_scraper
        ballotpedia_scraper._VA_HOUSE_2024.clear()
        ballotpedia_scraper._NJ_ASSEMBLY_2024.clear()
    except Exception:
        pass
    return STATE_LEG_MARGINS_PATH


def fetch_nj_assembly_2024_if_available() -> bool:
    """If a public CSV/API for NJ Assembly 2024 presidential by district exists, fetch and merge into state_leg_2024_presidential_margins.json (NJ_Assembly). Returns True if NJ data was loaded."""
    # No public CSV found for NJ Assembly 2024 pres by district (VPAP is Virginia-only; NJ Globe has maps not data).
    # When a source is available, add URL and parsing here, merge into data["NJ_Assembly"], save, return True.
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = write_va_house_to_state_leg_margins()
    print("Wrote", path)
