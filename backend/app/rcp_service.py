"""
RealClearPolitics / RealClearPolling integration.
Scrapes HTML tables (no Playwright). Falls back to direct httpx + BeautifulSoup.
"""
import hashlib
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup

from .election_db import make_poll_id

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Default year for RCP dates (no year in "Wednesday, February 11")
RCP_DEFAULT_YEAR = 2026
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

# Ingest all polls from the page; get_polls(days_back=70) filters to last 10 weeks for the tab
RCP_MAX_DAYS_BACK = 70  # used only for reference; we no longer filter at scrape time

# Latest-polls pages: one table, tbody rows = date row or single-cell poll row
RCP_URLS = {
    "senate": "https://www.realclearpolling.com/latest-polls/senate",
    "governor": "https://www.realclearpolling.com/latest-polls/governor",
    "house": "https://www.realclearpolling.com/latest-polls/house",
}

# State name (or abbreviation) -> 2-letter code for display
STATE_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
# Add abbreviations as self-mapping
for _abbr in list(STATE_TO_ABBR.values()):
    STATE_TO_ABBR[_abbr.lower()] = _abbr


def _state_from_race_name(race_name: str) -> Optional[str]:
    """Extract state abbreviation from RCP race_name e.g. '2026 Minnesota Senate - ...' -> 'MN'."""
    if not race_name:
        return None
    # Skip year prefix: "2026 Minnesota Senate" or "2026 Texas Republican Primary"
    rest = re.sub(r"^\d{4}\s+", "", race_name.strip(), count=1)
    parts = rest.split()
    if not parts:
        return None
    # First word might be state name (Minnesota) or abbreviation (MN)
    first = parts[0].lower()
    if first in STATE_TO_ABBR:
        return STATE_TO_ABBR[first]
    # Try first two words for "New York", "North Carolina", etc.
    if len(parts) >= 2:
        two = f"{parts[0].lower()} {parts[1].lower()}"
        if two in STATE_TO_ABBR:
            return STATE_TO_ABBR[two]
    return None


def _rcp_date_to_iso(date_str: str, default_year: int = RCP_DEFAULT_YEAR) -> str:
    """Convert 'Wednesday, February 11' or 'Monday, November 3, 2025' to ISO 'YYYY-MM-DD'."""
    if not date_str or not date_str.strip():
        return ""
    # Explicit 4-digit year in string (e.g. "November 3, 2025")
    year_match = re.search(r"\b(20[12]\d)\b", date_str)
    year = int(year_match.group(1)) if year_match else default_year
    # Strip weekday prefix: "Wednesday, February 11" -> "February 11"
    parts = date_str.strip().split(",", 1)
    month_day = parts[-1].strip() if len(parts) > 1 else date_str.strip()
    for i, month in enumerate(MONTH_NAMES):
        if month in month_day.lower():
            day_match = re.search(r"\d+", month_day)
            day = int(day_match.group()) if day_match else 1
            day = min(day, 31)
            month_num = i + 1
            try:
                parsed = date(year, month_num, day)
            except ValueError:
                parsed = date(year, month_num, min(day, 28))
            # If no year in string and parsed date is in the future, use previous year (RCP shows past polls)
            if not year_match and parsed > date.today():
                try:
                    parsed = date(year - 1, month_num, day)
                except ValueError:
                    parsed = date(year - 1, month_num, min(day, 28))
            return parsed.isoformat()
    return date_str  # fallback: return as-is


def _parse_rcp_poll_cell(text: str, race_type: str) -> Optional[Tuple[str, str, Dict[str, str], str]]:
    """
    Parse RCP single-cell format. Returns (pollster, race_name, results, margin) or None.
    Examples:
      House: "2026 Generic...PollCygnalResultsDemocrats48Republicans44SpreadDemocrats+4"
      Senate: "2026 MN Senate - A vs. BPollEmersonResultsA47B41SpreadA+6"
    """
    if not text or "Poll" not in text or "Results" not in text:
        return None
    poll_match = re.search(r"Poll(.+?)Results", text, re.DOTALL)
    pollster = poll_match.group(1).strip() if poll_match else ""
    if not pollster:
        return None
    race_name = text.split("Poll", 1)[0].strip()
    after_results = text.split("Results", 1)[-1]
    spread_match = re.search(r"Spread(.+)$", after_results)
    margin = spread_match.group(1).strip() if spread_match else ""
    before_spread = after_results.split("Spread")[0] if "Spread" in after_results else after_results
    # Two percentages: "Democrats48Republicans44" or "Flanagan47Tafoya41"
    parts = re.findall(r"([A-Za-z][A-Za-z\s\.\-/]+?)(\d+)", before_spread)
    if len(parts) >= 2:
        k1, v1 = parts[0][0].strip(), parts[0][1]
        k2, v2 = parts[1][0].strip(), parts[1][1]
        # Normalize party labels for trend charts (Dem/GOP)
        if "democrat" in k1.lower():
            k1, k2 = "Dem", "GOP"
        elif "republican" in k1.lower():
            k1, k2 = "GOP", "Dem"
        results = {k1: v1, k2: v2}
    else:
        nums = re.findall(r"\d+", before_spread)
        results = {"Dem": nums[0], "GOP": nums[1]} if len(nums) >= 2 else {}
    return (pollster, race_name, results, margin)


async def fetch_rcp_polls(race_type: str) -> List[Dict[str, Any]]:
    """Scrape RCP latest-polls page. RCP uses one table; tbody rows are date rows or single-cell poll rows."""
    url = RCP_URLS.get(race_type)
    if not url:
        return []

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        logger.warning("RCP fetch failed for %s: %s", race_type, e)
        return []

    soup = BeautifulSoup(html, "html.parser")
    polls: List[Dict[str, Any]] = []
    seen: set = set()  # (pollster, race_name, margin, date_iso) to avoid duplicate rows from same poll
    now = datetime.now(timezone.utc).isoformat()
    current_date = ""

    # Use the table with the most rows (main poll list); first table is often nav/filters
    tables = soup.find_all("table")
    if not tables:
        return polls
    tbody = None
    rows = []
    for table in tables:
        tb = table.find("tbody")
        r = tb.find_all("tr") if tb else table.find_all("tr")
        if len(r) > len(rows):
            rows = r
            tbody = tb

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) != 1:
            continue
        text = cells[0].get_text(strip=True)
        if not text:
            continue
        if re.match(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),", text):
            current_date = text
            continue
        parsed = _parse_rcp_poll_cell(text, race_type)
        if not parsed:
            continue
        pollster, race_name, results, margin = parsed
        if not results:
            continue
        # House tab: exclude all generic/congressional ballot (belongs in Generic Ballot tab)
        if race_type == "house":
            rn_lower = race_name.lower()
            if "generic" in rn_lower or "congressional ballot" in rn_lower or "generic ballot" in rn_lower:
                continue
        date_iso = _rcp_date_to_iso(current_date)
        # Store all polls from the page; get_polls(days_back=70) filters to last 10 weeks for the tab
        dedupe_key = (pollster, (race_name or "").strip(), (margin or "").strip(), date_iso or "")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        # Include matchup in race_key so same pollster+date but different races (e.g. Lindell vs Klobuchar, Demuth vs Klobuchar) get separate rows
        race_slug = hashlib.sha256((race_name or "").strip().encode()).hexdigest()[:8]
        race_key = f"RCP-{race_type}-{pollster}-{date_iso}-{race_slug}"
        state_abbr = _state_from_race_name(race_name)
        p = {
            "id": make_poll_id("rcp", pollster, date_iso, race_key),
            "source": "rcp",
            "race_type": race_type,
            "race_key": race_key,
            "state": state_abbr,
            "pollster": pollster,
            "poll_url": url,
            "start_date": date_iso,
            "end_date": date_iso,
            "date_added": current_date,  # keep human-readable for display
            "sample_size": "",
            "population": "",
            "grade": "",
            "margin": margin,
            "results": results,
            "raw_data": {"race_name": race_name, "cell": text[:200]},
            "fetched_at": now,
        }
        polls.append(p)

    if polls:
        logger.info("RCP parsed %d polls for %s from %s", len(polls), race_type, url)
    return polls


async def refresh_rcp_all() -> Dict[str, int]:
    """Fetch RCP for senate, governor, house and upsert to DB."""
    from .election_db import election_db

    counts: Dict[str, int] = {}
    for race_type in ("senate", "governor", "house"):
        polls = await fetch_rcp_polls(race_type)
        if polls:
            n = await election_db.upsert_polls(polls)
            counts[race_type] = n
            await election_db.log_fetch("rcp", race_type, n, status="success" if n else "partial")
    return counts
