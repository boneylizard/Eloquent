"""
Scrape Ballotpedia for off-year and special election results (Nov 2025–present).
Federal: House/Senate special elections from 119th Congress page (includes 2024 Pres MOV).
State: Gubernatorial from VA/NJ 2025 race pages (state 2024 margin used — same electorate).
State legislative: specials + VA House/NJ Assembly 2025 general. trump_2024_margin and
  swing_toward_d are set only when district-level 2024 pres data exists (state_leg_2024_
  presidential_margins.json or overrides). State-level margins are never used for districts
  (would bias simulation: e.g. solidly blue district vs state margin is meaningless).
Output: dem_actual_pct, rep_actual_pct; trump_2024_margin/swing_toward_d when same-area data.
"""
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Progress: (callback, counter_list, total_estimate). Call _report_progress(progress, message) before each fetch.
ProgressT = Optional[Tuple[Callable[[int, int, str], None], List[int], int]]

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Optional: 2024 presidential margin by district (Trump margin: + = R won). Load from data file when present.
_STATE_LEG_2024_MARGINS_PATH = Path(__file__).resolve().parent.parent / "data" / "state_leg_2024_presidential_margins.json"
_VA_HOUSE_2024: Dict[int, float] = {}
_NJ_ASSEMBLY_2024: Dict[int, float] = {}
# VA House districts to skip for 2024 margin (VPAP 2024 old boundaries; D4 mismatch vs 2025 result).
_VA_HOUSE_2024_EXCLUDE: frozenset = frozenset({4})


def _load_2024_district_margins() -> Tuple[Dict[int, float], Dict[int, float]]:
    """Load VA House and NJ Assembly 2024 presidential margins by district from optional JSON file."""
    global _VA_HOUSE_2024, _NJ_ASSEMBLY_2024
    if _VA_HOUSE_2024 or _NJ_ASSEMBLY_2024:
        return (_VA_HOUSE_2024, _NJ_ASSEMBLY_2024)
    try:
        if _STATE_LEG_2024_MARGINS_PATH.exists():
            with open(_STATE_LEG_2024_MARGINS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in (data.get("VA_House") or {}).items():
                try:
                    num = int(k)
                    if num in _VA_HOUSE_2024_EXCLUDE:
                        continue
                    _VA_HOUSE_2024[num] = float(v)
                except (ValueError, TypeError):
                    pass
            for k, v in (data.get("NJ_Assembly") or {}).items():
                try:
                    _NJ_ASSEMBLY_2024[int(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            logger.info("Loaded 2024 district margins: VA House %d, NJ Assembly %d", len(_VA_HOUSE_2024), len(_NJ_ASSEMBLY_2024))
    except Exception as e:
        logger.debug("Could not load state_leg_2024_presidential_margins.json: %s", e)
    return (_VA_HOUSE_2024, _NJ_ASSEMBLY_2024)

# Ballotpedia URLs
URL_119TH_SPECIAL = "https://ballotpedia.org/Special_elections_to_the_119th_United_States_Congress_(2025-2026)"
URL_VA_GOV_2025 = "https://ballotpedia.org/Virginia_gubernatorial_election,_2025"
URL_NJ_GOV_2025 = "https://ballotpedia.org/New_Jersey_gubernatorial_and_lieutenant_gubernatorial_election,_2025"
URL_STATE_LEG_2025 = "https://ballotpedia.org/State_legislative_special_elections,_2025"
URL_STATE_LEG_2026 = "https://ballotpedia.org/State_legislative_special_elections,_2026"
# Virginia House of Delegates general election Nov 2025 (100 districts); district pages have vote %
VA_HOUSE_DISTRICT_URL = "https://ballotpedia.org/Virginia_House_of_Delegates_District_{}"
VA_HOUSE_2025_ELECTION_DATE = datetime(2025, 11, 4).date()
# New Jersey General Assembly Nov 2025 (40 districts, 2 seats each); district pages have D/R %
NJ_ASSEMBLY_DISTRICT_URL = "https://ballotpedia.org/New_Jersey_General_Assembly_District_{}"
NJ_ASSEMBLY_2025_ELECTION_DATE = datetime(2025, 11, 4).date()

# Only include elections on or after this date (user: "since November 2025")
CUTOFF_DATE = datetime(2025, 11, 1).date()

# State name/abbrev from race text (e.g. "Tennessee's 7th" -> TN)
STATE_FROM_RACE: Dict[str, str] = {
    "arizona": "AZ", "california": "CA", "florida": "FL", "georgia": "GA",
    "new jersey": "NJ", "ohio": "OH", "tennessee": "TN", "texas": "TX",
    "virginia": "VA",
}

# 2024 presidential margin by state: Trump margin (positive = Trump won by X, negative = Trump lost by X).
# Used ONLY for statewide races (e.g. governor). Never use for state legislative districts — would bias swing.
# Prefer pres_2024_state_margins.json (from tonmcg county data) when present; this is fallback.
# VA/NJ: Harris won (negative = D+). Confirmed: VA ~D+5.8, NJ ~D+5.9 (2024 results).
PRES_2024_STATE_MARGIN: Dict[str, float] = {
    "VA": -5.78,
    "NJ": -5.88,
    "TX": 9.5,
    "GA": 7.4,
    "KY": 26.2,
    "SC": 12.0,
    "IA": 9.4,
    "TN": 22.0,
}


def _get_2024_state_margin(state_abbr: str) -> Optional[float]:
    """State-level 2024 Trump margin for governor races. Prefers tonmcg county-derived JSON."""
    from .pres_2024_county_loader import load_pres_2024_state_margins
    margins = load_pres_2024_state_margins()
    if margins:
        m = margins.get(state_abbr.upper())
        if m is not None:
            return float(m)
    return PRES_2024_STATE_MARGIN.get(state_abbr.upper())

# 2024 Trump margin for specific state legislative districts (when known). Key: "ST_Senate_9" or "ST_House_22".
# TN-07: from Ballotpedia 119th specials table "2024 Presidential MOV" column. TX SD9: estimated/override (D flipped Jan 2026).
TRUMP_2024_STATE_LEG_OVERRIDES: Dict[str, float] = {
    "TX_Senate_9": 17.0,  # Trump margin in TX SD9 (approx); verify with district-level 2024 pres data if available
}

# Approximate max links we fetch (federal 1 + gov 2 + state leg index 2 + state leg races 45 + VA 100 + NJ 40)
PROGRESS_TOTAL_ESTIMATE = 200


def _report_progress(progress: ProgressT, message: str) -> None:
    if not progress:
        return
    callback, counter, total = progress
    counter[0] += 1
    callback(counter[0], total, message)


def _parse_date_cell(text: str) -> Optional[datetime]:
    """Parse 'September 23, 2025' or 'November 4, 2025' -> date."""
    text = (text or "").strip()
    if not text or "TBD" in text:
        return None
    try:
        return datetime.strptime(text, "%B %d, %Y").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%b. %d, %Y").date()
    except ValueError:
        pass
    return None


def _mov_to_dem_pct(mov_text: str) -> Optional[float]:
    """Convert 'D+40' or 'R+9' to two-party dem share. Returns None if TBD or unparseable."""
    mov_text = (mov_text or "").strip().upper()
    if not mov_text or "TBD" in mov_text:
        return None
    m = re.match(r"(D|R)\+(\d+(?:\.\d+)?)", mov_text)
    if not m:
        return None
    party, pts = m.group(1), float(m.group(2))
    if party == "D":
        return 50.0 + pts / 2.0
    return 50.0 - pts / 2.0


def _mov_to_signed(mov_text: str) -> Optional[float]:
    """Convert 'D+40' or 'R+9' to signed margin: positive = R won by X, negative = D won by X (Trump margin convention)."""
    mov_text = (mov_text or "").strip().upper()
    if not mov_text or "TBD" in mov_text:
        return None
    m = re.match(r"(D|R)\+(\d+(?:\.\d+)?)", mov_text)
    if not m:
        return None
    party, pts = m.group(1), float(m.group(2))
    return -pts if party == "D" else pts


def _state_from_race_name(race_name: str) -> str:
    """Extract state abbreviation from race label (e.g. Tennessee's 7th -> TN)."""
    race_lower = (race_name or "").lower()
    for name, abbr in STATE_FROM_RACE.items():
        if name in race_lower:
            return abbr
    return ""


# State names for state legislative table (e.g. "Texas State Senate 9" -> TX)
_STATE_ABBR: Dict[str, str] = {
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
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def _parse_dates_in_cell(text: str) -> List[datetime]:
    """Extract all dates like 'January 31, 2026' or 'November 4, 2025' from text. Returns list of date objects."""
    out: List[datetime] = []
    for m in re.finditer(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", text, re.IGNORECASE):
        try:
            dt = datetime.strptime(m.group(0).replace(",", ""), "%B %d %Y").date()
            out.append(dt)
        except ValueError:
            pass
    return out


def _state_leg_office_to_state_chamber_district(office_text: str) -> tuple:
    """From 'Texas State Senate 9' or 'Mississippi State House 22' return (state_abbr, 'state_senate'|'state_house', district_label)."""
    text = (office_text or "").strip().lower()
    state_abbr = ""
    for name, abbr in _STATE_ABBR.items():
        if name in text:
            state_abbr = abbr
            break
    if "senate" in text:
        chamber = "state_senate"
    elif "house" in text or "assembly" in text or "delegate" in text:
        chamber = "state_house"
    else:
        chamber = "state_senate"
    # District: try "District 9" or "9" or "40B" etc.
    district_label = office_text.strip() if office_text else ""
    return (state_abbr, chamber, district_label)


def _fetch(url: str, client: httpx.Client) -> str:
    r = client.get(
        url,
        timeout=25.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; BallotpediaScraper/1.0)"},
    )
    r.raise_for_status()
    return r.text


def scrape_federal_special_elections(client: httpx.Client, progress: ProgressT = None) -> List[Dict[str, Any]]:
    """Parse 119th Congress special elections page; return completed races since CUTOFF_DATE."""
    out: List[Dict[str, Any]] = []
    _report_progress(progress, "Federal specials index")
    html = _fetch(URL_119TH_SPECIAL, client)
    soup = BeautifulSoup(html, "html.parser")

    # Find tables: "Results of special elections to the 119th Congress (House)" and "(Senate)"
    for table in soup.find_all("table"):
        caption = table.find("caption") or table.find_previous("caption")
        cap_text = (caption.get_text() if caption else "") or ""
        if "Results of special elections" not in cap_text and "119th" not in cap_text:
            continue
        is_senate = "Senate" in cap_text
        race_type = "senate" if is_senate else "house"

        headers = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th"):
                headers.append((th.get_text() or "").strip().lower())
        rows = table.find("tbody")
        if not rows:
            rows = table
        for tr in (rows.find_all("tr") if rows else []):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 3:
                continue
            # Columns: Race | Election date | Incumbent | Winner | Election MOV | Previous MOV | 2024 Presidential MOV
            texts = [(c.get_text() or "").strip() for c in cells]
            race_name = texts[0] if texts else ""
            date_text = texts[1] if len(texts) > 1 else ""
            winner_text = texts[3] if len(texts) > 3 else texts[-3]
            mov_text = texts[4] if len(texts) > 4 else texts[-2]
            pres_2024_mov_text = texts[6] if len(texts) > 6 else ""

            if "TBD" in winner_text and "TBD" in mov_text:
                continue
            dt = _parse_date_cell(date_text)
            if not dt or dt < CUTOFF_DATE:
                continue
            state = _state_from_race_name(race_name)
            if not state:
                continue
            dem_pct = _mov_to_dem_pct(mov_text)
            if dem_pct is None and "TBD" not in winner_text:
                if "(D)" in winner_text or "[D]" in winner_text:
                    dem_pct = 55.0
                elif "(R)" in winner_text or "[R]" in winner_text:
                    dem_pct = 45.0
            if dem_pct is None:
                continue
            # 2024 presidential margin in this district (Trump margin: + = R won)
            trump_2024_margin = _mov_to_signed(pres_2024_mov_text) if pres_2024_mov_text else None
            election_margin_r = _mov_to_signed(mov_text)
            # Swing toward D: positive = D did better than 2024
            swing_toward_d = None
            if trump_2024_margin is not None and election_margin_r is not None:
                swing_toward_d = round(trump_2024_margin - election_margin_r, 1)
            label = race_name.replace("'s ", " ").replace("' ", " ").split("(")[0].strip()
            if not label:
                label = f"{state} {race_type} {date_text}"
            entry = {
                "label": label[:80],
                "type": race_type,
                "state": state,
                "date": dt.strftime("%Y-%m-%d"),
                "dem_actual_pct": round(dem_pct, 1),
                "rep_actual_pct": round(100.0 - dem_pct, 1),
                "poll_avg_pct": None,
                "note": "Ballotpedia 119th Congress special (Election MOV)",
            }
            if trump_2024_margin is not None:
                entry["trump_2024_margin"] = round(trump_2024_margin, 1)
            if swing_toward_d is not None:
                entry["swing_toward_d"] = swing_toward_d
            out.append(entry)
    return out


def _scrape_governor_page(url: str, state: str, client: httpx.Client, progress: ProgressT = None) -> Optional[Dict[str, Any]]:
    """Parse a gubernatorial race page for general election D/R percentages. Returns one result or None."""
    try:
        _report_progress(progress, f"Governor {state}")
        html = _fetch(url, client)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None
    soup = BeautifulSoup(html, "html.parser")

    dem_pct: Optional[float] = None
    rep_pct: Optional[float] = None
    date_str = "2025-11-04"

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        # Ballotpedia often has a row of percentages then rows with Candidate (D)/(R) and votes
        pct_row: Optional[List[float]] = None
        for tr in rows:
            cells = tr.find_all(["th", "td"])
            texts = [(c.get_text() or "").strip() for c in cells]
            # Collect numbers that look like percentages (e.g. 57.6, 42.2)
            pcts = []
            for t in texts:
                t_clean = t.replace("%", "").replace(",", "")
                if re.match(r"^\d{1,2}\.\d$", t_clean):
                    try:
                        p = float(t_clean)
                        if 0 <= p <= 100:
                            pcts.append(p)
                    except ValueError:
                        pass
            if len(pcts) >= 2 and not any("(d)" in " ".join(texts).lower() or "(r)" in " ".join(texts).lower() for _ in [1]):
                pct_row = pcts
            row_lower = " ".join(texts).lower()
            if "(d)" in row_lower or "(r)" in row_lower:
                for i, t in enumerate(texts):
                    t_clean = (t or "").replace("%", "").replace(",", "").strip()
                    if re.match(r"^\d{1,2}\.\d$", t_clean):
                        try:
                            p = float(t_clean)
                            if 0 <= p <= 100:
                                if "(d)" in row_lower:
                                    dem_pct = p
                                elif "(r)" in row_lower:
                                    rep_pct = p
                        except ValueError:
                            pass
                if dem_pct is not None and rep_pct is not None:
                    break
        if pct_row and (dem_pct is None or rep_pct is None):
            idx = 0
            for tr in rows:
                row_lower = " ".join((c.get_text() or "").lower() for c in tr.find_all(["th", "td"]))
                if "(d)" in row_lower and dem_pct is None and idx < len(pct_row):
                    dem_pct = pct_row[idx]
                    idx += 1
                elif "(r)" in row_lower and rep_pct is None and idx < len(pct_row):
                    rep_pct = pct_row[idx]
                    idx += 1
        if dem_pct is not None or rep_pct is not None:
            break

    if dem_pct is None and rep_pct is not None:
        dem_pct = round(100.0 - rep_pct, 1)
    elif dem_pct is not None and rep_pct is not None and dem_pct + rep_pct > 0:
        dem_pct = round(100.0 * dem_pct / (dem_pct + rep_pct), 1)
    if dem_pct is None:
        return None

    # 2024 presidential margin in this state (Trump margin) and swing vs 2024
    trump_2024_margin = _get_2024_state_margin(state)
    election_margin_r = 100.0 - 2.0 * dem_pct  # R% - D% in two-party
    swing_toward_d = None
    if trump_2024_margin is not None:
        swing_toward_d = round(trump_2024_margin - election_margin_r, 1)

    entry = {
        "label": f"{state} Governor 2025",
        "type": "governor",
        "state": state,
        "date": date_str,
        "dem_actual_pct": dem_pct,
        "rep_actual_pct": round(100.0 - dem_pct, 1),
        "poll_avg_pct": None,
        "note": "Ballotpedia gubernatorial general election",
    }
    if trump_2024_margin is not None:
        entry["trump_2024_margin"] = round(trump_2024_margin, 1)
    if swing_toward_d is not None:
        entry["swing_toward_d"] = swing_toward_d
    return entry


def _state_leg_index_races(client: httpx.Client, since_date: datetime.date, progress: ProgressT = None) -> List[Dict[str, Any]]:
    """Parse state leg special election index pages; return list of {url, state, chamber, label, election_date} for races with election date >= since_date."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for i, url in enumerate((URL_STATE_LEG_2025, URL_STATE_LEG_2026)):
        try:
            _report_progress(progress, f"State leg index {i + 1}/2")
            html = _fetch(url, client)
        except Exception as e:
            logger.warning("State leg index %s failed: %s", url, e)
            continue
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            cap = (table.find("caption") or table.find_previous("caption")) and (table.find("caption") or table.find_previous("caption")).get_text() or ""
            if "state legislative special" not in cap.lower() and "special elections" not in cap.lower():
                continue
            rows = table.find("tbody") or table
            for tr in (rows.find_all("tr") if hasattr(rows, "find_all") else []):
                cells = tr.find_all(["td", "th"])
                if len(cells) < 5:
                    continue
                office_cell = cells[0]
                office_link = office_cell.find("a", href=True)
                office_text = (office_cell.get_text() or "").strip()
                if not office_link or not office_text:
                    continue
                href = office_link.get("href", "")
                if href.startswith("/"):
                    race_url = "https://ballotpedia.org" + href
                else:
                    race_url = href
                date_cell = cells[4].get_text() if len(cells) > 4 else ""
                dates = _parse_dates_in_cell(date_cell)
                election_date = None
                for d in dates:
                    if d >= since_date:
                        if election_date is None or d > election_date:
                            election_date = d
                if election_date is None:
                    continue
                state_abbr, chamber, _ = _state_leg_office_to_state_chamber_district(office_text)
                if not state_abbr:
                    continue
                key = (race_url, election_date.isoformat())
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "url": race_url,
                    "state": state_abbr,
                    "chamber": chamber,
                    "label": office_text[:80],
                    "election_date": election_date,
                })
    return out


def _scrape_state_leg_race_page(
    race_url: str, state: str, chamber: str, label: str, election_date: datetime.date, client: httpx.Client,
    progress: ProgressT = None,
) -> Optional[Dict[str, Any]]:
    """Fetch one state leg district page and parse special/runoff general election result (D/R %) for the given election_date."""
    try:
        _report_progress(progress, f"State leg: {state} {chamber}")
        html = _fetch(race_url, client)
    except Exception as e:
        logger.warning("State leg race %s failed: %s", race_url, e)
        return None
    soup = BeautifulSoup(html, "html.parser")
    date_str_verbose = election_date.strftime("%B %d, %Y")
    dem_pct: Optional[float] = None
    rep_pct: Optional[float] = None
    # Find any table that has (D) and (R) rows with percentage cells and is in a section that mentions our date or "special"
    for table in soup.find_all("table"):
        table_text = (table.get_text() or "")
        if "(d)" not in table_text.lower() or "(r)" not in table_text.lower():
            continue
        # Only use table if it's in a section that mentions this election date (avoid 2022/2018 tables)
        parent = table.parent
        section_ok = False
        for _ in range(20):
            if parent is None:
                break
            ptext = (parent.get_text() or "") if hasattr(parent, "get_text") else ""
            if date_str_verbose in ptext:
                section_ok = True
                break
            parent = getattr(parent, "parent", None)
        if not section_ok:
            continue
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row_lower = " ".join((c.get_text() or "").lower() for c in cells)
            if "(d)" not in row_lower and "(r)" not in row_lower:
                continue
            for c in cells:
                t = (c.get_text() or "").strip().replace("%", "")
                if re.match(r"^\d{1,2}\.\d$", t):
                    try:
                        p = float(t)
                        if 0 <= p <= 100:
                            if "(d)" in row_lower:
                                dem_pct = p
                            elif "(r)" in row_lower:
                                rep_pct = p
                    except ValueError:
                        pass
        if dem_pct is not None or rep_pct is not None:
            break
    if dem_pct is None and rep_pct is not None:
        dem_pct = round(100.0 - rep_pct, 1)
    elif dem_pct is not None and rep_pct is not None and dem_pct + rep_pct > 0:
        dem_pct = round(100.0 * dem_pct / (dem_pct + rep_pct), 1)
    if dem_pct is None:
        return None
    # 2024 margin and swing only when we have district/race-specific data (never state-level for state leg).
    parts = (label or "").split()
    dist_part = parts[-1] if parts else ""
    override_key = f"{state}_{'Senate' if chamber == 'state_senate' else 'House'}_{dist_part}" if dist_part else None
    trump_2024_margin = TRUMP_2024_STATE_LEG_OVERRIDES.get(override_key) if override_key else None
    election_margin_r = 100.0 - 2.0 * dem_pct
    swing_toward_d = None
    if trump_2024_margin is not None:
        swing_toward_d = round(trump_2024_margin - election_margin_r, 1)
    chamber_label = "Senate" if chamber == "state_senate" else "House"
    short_label = f"{state} State {chamber_label} {dist_part}" if dist_part else f"{state} {label}"
    entry = {
        "label": short_label[:80],
        "type": chamber,
        "state": state,
        "date": election_date.strftime("%Y-%m-%d"),
        "dem_actual_pct": dem_pct,
        "rep_actual_pct": round(100.0 - dem_pct, 1),
        "poll_avg_pct": None,
        "note": "Ballotpedia state legislative special",
    }
    if trump_2024_margin is not None:
        entry["trump_2024_margin"] = round(trump_2024_margin, 1)
    if swing_toward_d is not None:
        entry["swing_toward_d"] = swing_toward_d
    return entry


def scrape_state_leg_specials(client: httpx.Client, since_nov_2025: bool = True, progress: ProgressT = None) -> List[Dict[str, Any]]:
    """Scrape state house/senate special elections with election date on or after 2025-11-01."""
    since_date = CUTOFF_DATE if since_nov_2025 else datetime(2020, 1, 1).date()
    races = _state_leg_index_races(client, since_date, progress=progress)
    results: List[Dict[str, Any]] = []
    for r in races[:45]:
        entry = _scrape_state_leg_race_page(
            r["url"], r["state"], r["chamber"], r["label"], r["election_date"], client, progress=progress
        )
        if entry:
            results.append(entry)
    return results


def _parse_va_district_2025_general(html: str, district_num: int) -> Optional[float]:
    """Parse one Virginia House district page for 2025 general election; return two-party Dem share or None."""
    soup = BeautifulSoup(html, "html.parser")
    dem_pct: Optional[float] = None
    rep_pct: Optional[float] = None
    # Find table that has (D) and (R) and is in a section mentioning 2025 and this district's general election
    for table in soup.find_all("table"):
        table_text = (table.get_text() or "")
        if "(d)" not in table_text.lower() or "(r)" not in table_text.lower():
            continue
        parent = table.parent
        section_ok = False
        for _ in range(25):
            if parent is None:
                break
            ptext = (parent.get_text() or "") if hasattr(parent, "get_text") else ""
            ptext_lower = ptext.lower()
            if "2025" in ptext and "general election" in ptext_lower and f"district {district_num}" in ptext_lower:
                section_ok = True
                break
            parent = getattr(parent, "parent", None)
        if not section_ok:
            continue
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row_lower = " ".join((c.get_text() or "").lower() for c in cells)
            if "(d)" not in row_lower and "(r)" not in row_lower:
                continue
            for c in cells:
                t = (c.get_text() or "").strip().replace("%", "")
                if re.match(r"^\d{1,3}\.\d$", t):
                    try:
                        p = float(t)
                        if 0 <= p <= 100:
                            if "(d)" in row_lower:
                                dem_pct = p
                            elif "(r)" in row_lower:
                                rep_pct = p
                    except ValueError:
                        pass
        if dem_pct is not None or rep_pct is not None:
            break
    if dem_pct is None and rep_pct is not None:
        dem_pct = round(100.0 - rep_pct, 1)
    elif dem_pct is not None and rep_pct is not None and dem_pct + rep_pct > 0:
        dem_pct = round(100.0 * dem_pct / (dem_pct + rep_pct), 1)
    return dem_pct


def scrape_va_house_general_2025(
    client: httpx.Client,
    max_districts: int = 100,
    progress: ProgressT = None,
) -> List[Dict[str, Any]]:
    """Scrape Virginia House of Delegates Nov 2025 general. trump_2024_margin and swing_toward_d only when district-level 2024 pres data exists (state_leg_2024_presidential_margins.json or overrides); never state-level (would bias simulation)."""
    results: List[Dict[str, Any]] = []
    va_2024, _ = _load_2024_district_margins()
    for n in range(1, max_districts + 1):
        if n > 1:
            time.sleep(0.35)
        url = VA_HOUSE_DISTRICT_URL.format(n)
        try:
            _report_progress(progress, f"VA House district {n}/{max_districts}")
            html = _fetch(url, client)
        except Exception as e:
            logger.warning("VA House district %d fetch failed: %s", n, e)
            continue
        dem_pct = _parse_va_district_2025_general(html, n)
        if dem_pct is None:
            continue
        trump_2024_margin = va_2024.get(n) if va_2024 is not None else None
        if trump_2024_margin is None:
            trump_2024_margin = TRUMP_2024_STATE_LEG_OVERRIDES.get(f"VA_House_{n}")
        election_margin_r = 100.0 - 2.0 * dem_pct
        swing_toward_d = round(trump_2024_margin - election_margin_r, 1) if trump_2024_margin is not None else None
        note = "Ballotpedia VA House 2025 general"
        if va_2024 and n in va_2024:
            note += "; 2024 pres by district"
        entry: Dict[str, Any] = {
            "label": f"VA House D{n} 2025 general",
            "type": "state_house",
            "state": "VA",
            "date": VA_HOUSE_2025_ELECTION_DATE.strftime("%Y-%m-%d"),
            "dem_actual_pct": dem_pct,
            "rep_actual_pct": round(100.0 - dem_pct, 1),
            "poll_avg_pct": None,
            "note": note,
        }
        if trump_2024_margin is not None:
            entry["trump_2024_margin"] = round(trump_2024_margin, 1)
        if swing_toward_d is not None:
            entry["swing_toward_d"] = swing_toward_d
        results.append(entry)
    return results


def _parse_nj_assembly_district_2025_general(html: str, district_num: int) -> Optional[float]:
    """Parse one NJ Assembly district page for 2025 general (2 seats). Sum D and R vote shares for two-party dem_pct."""
    soup = BeautifulSoup(html, "html.parser")
    dem_sum: float = 0.0
    rep_sum: float = 0.0
    for table in soup.find_all("table"):
        table_text = (table.get_text() or "")
        if "(d)" not in table_text.lower() or "(r)" not in table_text.lower():
            continue
        parent = table.parent
        section_ok = False
        for _ in range(25):
            if parent is None:
                break
            ptext = (parent.get_text() or "") if hasattr(parent, "get_text") else ""
            ptext_lower = ptext.lower()
            if "2025" in ptext and "general election" in ptext_lower and f"district {district_num}" in ptext_lower:
                section_ok = True
                break
            parent = getattr(parent, "parent", None)
        if not section_ok:
            continue
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row_lower = " ".join((c.get_text() or "").lower() for c in cells)
            if "(d)" not in row_lower and "(r)" not in row_lower:
                continue
            for c in cells:
                t = (c.get_text() or "").strip().replace("%", "")
                if re.match(r"^\d{1,3}\.\d$", t):
                    try:
                        p = float(t)
                        if 0 <= p <= 100:
                            if "(d)" in row_lower:
                                dem_sum += p
                            elif "(r)" in row_lower:
                                rep_sum += p
                    except ValueError:
                        pass
        if dem_sum > 0 or rep_sum > 0:
            break
    if dem_sum <= 0 and rep_sum <= 0:
        return None
    total = dem_sum + rep_sum
    if total <= 0:
        return None
    return round(100.0 * dem_sum / total, 1)


def scrape_nj_assembly_general_2025(
    client: httpx.Client,
    max_districts: int = 40,
    progress: ProgressT = None,
) -> List[Dict[str, Any]]:
    """Scrape NJ General Assembly Nov 2025 general (40 districts). trump_2024_margin and swing only when district-level 2024 pres data exists; never state-level (would bias simulation)."""
    results: List[Dict[str, Any]] = []
    _, nj_2024 = _load_2024_district_margins()
    for n in range(1, max_districts + 1):
        if n > 1:
            time.sleep(0.35)
        url = NJ_ASSEMBLY_DISTRICT_URL.format(n)
        try:
            _report_progress(progress, f"NJ Assembly district {n}/{max_districts}")
            html = _fetch(url, client)
        except Exception as e:
            logger.warning("NJ Assembly district %d fetch failed: %s", n, e)
            continue
        dem_pct = _parse_nj_assembly_district_2025_general(html, n)
        if dem_pct is None:
            continue
        trump_2024_margin = nj_2024.get(n) if nj_2024 is not None else None
        if trump_2024_margin is None:
            trump_2024_margin = TRUMP_2024_STATE_LEG_OVERRIDES.get(f"NJ_Assembly_{n}")
        election_margin_r = 100.0 - 2.0 * dem_pct
        swing_toward_d = round(trump_2024_margin - election_margin_r, 1) if trump_2024_margin is not None else None
        note = "Ballotpedia NJ Assembly 2025 general"
        if nj_2024 and n in nj_2024:
            note += "; 2024 pres by district"
        entry: Dict[str, Any] = {
            "label": f"NJ Assembly D{n} 2025 general",
            "type": "state_house",
            "state": "NJ",
            "date": NJ_ASSEMBLY_2025_ELECTION_DATE.strftime("%Y-%m-%d"),
            "dem_actual_pct": dem_pct,
            "rep_actual_pct": round(100.0 - dem_pct, 1),
            "poll_avg_pct": None,
            "note": note,
        }
        if trump_2024_margin is not None:
            entry["trump_2024_margin"] = round(trump_2024_margin, 1)
        if swing_toward_d is not None:
            entry["swing_toward_d"] = swing_toward_d
        results.append(entry)
    return results


def scrape_state_leg_generals(
    client: httpx.Client,
    since_nov_2025: bool = True,
    progress: ProgressT = None,
) -> List[Dict[str, Any]]:
    """Scrape state legislative general elections (VA House, NJ Assembly) with election on or after 2025-11-01. Includes 2024 pres margin by district when in state_leg_2024_presidential_margins.json."""
    out: List[Dict[str, Any]] = []
    if not since_nov_2025:
        return out
    try:
        va = scrape_va_house_general_2025(client, progress=progress)
        out.extend(va)
        logger.info("Scraped %d VA House 2025 general election result(s)", len(va))
    except Exception as e:
        logger.exception("VA House 2025 general scrape failed: %s", e)
    try:
        nj = scrape_nj_assembly_general_2025(client, progress=progress)
        out.extend(nj)
        logger.info("Scraped %d NJ Assembly 2025 general election result(s)", len(nj))
    except Exception as e:
        logger.exception("NJ Assembly 2025 general scrape failed: %s", e)
    return out


def scrape_all(
    since_nov_2025: bool = True,
    include_governors: bool = True,
    include_federal: bool = True,
    include_state_leg: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Scrape Ballotpedia for special/off-year results.
    If since_nov_2025, only elections on or after 2025-11-01.
    include_state_leg: state house/senate specials (e.g. TX Senate 9 flip).
    progress_callback(current, total, message) is called before each link fetch.
    Returns list of dicts with dem_actual_pct, trump_2024_margin, swing_toward_d when available.
    """
    results: List[Dict[str, Any]] = []
    counter: List[int] = [0]
    total_est = PROGRESS_TOTAL_ESTIMATE
    progress: ProgressT = (progress_callback, counter, total_est) if progress_callback else None

    with httpx.Client(follow_redirects=True) as client:
        if include_federal:
            try:
                federal = scrape_federal_special_elections(client, progress=progress)
                results.extend(federal)
                logger.info("Scraped %d federal special election(s)", len(federal))
            except Exception as e:
                logger.exception("Federal special elections scrape failed: %s", e)
        if include_governors:
            for url, state in [(URL_VA_GOV_2025, "VA"), (URL_NJ_GOV_2025, "NJ")]:
                try:
                    entry = _scrape_governor_page(url, state, client, progress=progress)
                    if entry and (not since_nov_2025 or entry["date"] >= "2025-11-01"):
                        results.append(entry)
                except Exception as e:
                    logger.warning("Governor scrape %s failed: %s", state, e)
        if include_state_leg:
            try:
                state_leg = scrape_state_leg_specials(client, since_nov_2025=since_nov_2025, progress=progress)
                results.extend(state_leg)
                logger.info("Scraped %d state legislative special(s)", len(state_leg))
            except Exception as e:
                logger.exception("State leg specials scrape failed: %s", e)
            try:
                state_leg_generals = scrape_state_leg_generals(client, since_nov_2025=since_nov_2025, progress=progress)
                results.extend(state_leg_generals)
                logger.info("Scraped %d state legislative general(s) (e.g. VA House 2025)", len(state_leg_generals))
            except Exception as e:
                logger.exception("State leg generals scrape failed: %s", e)
    if progress_callback and counter[0] > 0:
        progress_callback(counter[0], counter[0], "Done")
    return results


def run_and_import_to_calibration(
    since_nov_2025: bool = True,
    merge: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run scraper and optionally merge into calibration JSON.
    If merge=True, appends new entries (by label+date) that are not already in calibration.
    progress_callback(current, total, message) is invoked before each link fetch during scrape.
    Returns { "scraped": list, "added": list, "skipped": list, "calibration_path": str }.
    """
    from .election_simulation import add_calibration_entry, load_calibration, CALIBRATION_PATH

    scraped = scrape_all(
        since_nov_2025=since_nov_2025,
        include_governors=True,
        include_federal=True,
        include_state_leg=True,
        progress_callback=progress_callback,
    )
    existing = load_calibration()
    existing_keys = {(e.get("label", "").strip(), e.get("date", "").strip()) for e in existing}
    added = []
    skipped = []
    for s in scraped:
        key = (s.get("label", "").strip(), s.get("date", "").strip())
        if key in existing_keys:
            skipped.append(s)
            continue
        entry = add_calibration_entry(
            label=s["label"],
            entry_type=s["type"],
            state=s["state"],
            date=s["date"],
            dem_actual_pct=s["dem_actual_pct"],
            poll_avg_pct=s.get("poll_avg_pct"),
            weight=1.0,
            note=s.get("note"),
            trump_2024_margin=s.get("trump_2024_margin"),
            swing_toward_d=s.get("swing_toward_d"),
            rep_actual_pct=s.get("rep_actual_pct"),
        )
        added.append(entry)
        existing_keys.add(key)
    return {
        "scraped": scraped,
        "added": added,
        "skipped": skipped,
        "calibration_path": str(CALIBRATION_PATH),
    }
