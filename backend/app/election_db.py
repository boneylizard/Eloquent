"""
Election SQLite database layer.
Polls from VoteHub, RCP, and RaceToTheWH are stored here for instant API serving.
"""
import hashlib
import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# Full and abbreviated month names for parsing any stored date format
_MONTH_NAMES_FULL = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
# abbr -> month number 1-12 (sep/sept both 9)
_MONTH_ABBR_TO_NUM = {a: i + 1 for i, a in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MONTH_ABBR_TO_NUM["sept"] = 9

# State name -> 2-letter code (for deriving state from RCP race_name when serving)
_STATE_TO_ABBR = {
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
for _ab in list(_STATE_TO_ABBR.values()):
    _STATE_TO_ABBR[_ab.lower()] = _ab


def _state_from_race_name(race_name: str) -> Optional[str]:
    """Extract state abbreviation from RCP-style race_name e.g. '2026 Minnesota Senate - ...' -> 'MN'."""
    if not race_name:
        return None
    rest = re.sub(r"^\d{4}\s+", "", race_name.strip(), count=1)
    parts = rest.split()
    if not parts:
        return None
    first = parts[0].lower()
    if first in _STATE_TO_ABBR:
        return _STATE_TO_ABBR[first]
    if len(parts) >= 2:
        two = f"{parts[0].lower()} {parts[1].lower()}"
        if two in _STATE_TO_ABBR:
            return _STATE_TO_ABBR[two]
    return None


def _state_from_race_text(race: str) -> Optional[str]:
    """
    Extract state from Race or Candidate text (for RTWH polls). Handles:
    - (TX-R), (TX-D), (MI-D), (AL-R) etc. -> 2-letter code in parentheses
    - 'Alabama GOP Primary', 'Texas Republican Primary' -> state name at start
    - 'OH - Marshall v. Rogers' -> prefix before ' - '
    """
    if not race or not str(race).strip():
        return None
    s = str(race).strip()

    # Pattern: (XX-D) or (XX-R) or (XX-I) or (XX-P) anywhere in string
    paren_match = re.search(r"\(([A-Za-z]{2})-[DRIP]\)", s)
    if paren_match:
        abbr = paren_match.group(1).upper()
        if len(abbr) == 2 and abbr.isalpha():
            return abbr

    # State name at start: "Alabama GOP Primary", "Texas ...", "New York ..."
    sl = s.lower()
    for name, abbr in _STATE_TO_ABBR.items():
        if name in ("al", "ak", "az", "ca", "co", "ct", "de", "fl", "ga", "hi", "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "va", "vt", "wa", "wi", "wv", "wy", "dc"):
            continue  # skip abbreviations in this loop
        if sl == name or sl.startswith(name + " ") or sl.startswith(name + ","):
            return abbr

    # Two-letter prefix before " - ": "OH - Marshall v. Rogers"
    if " - " in s:
        prefix = s.split(" - ", 1)[0].strip()
        if len(prefix) == 2 and prefix.isalpha():
            return prefix.upper()
        if prefix.lower() in _STATE_TO_ABBR:
            return _STATE_TO_ABBR[prefix.lower()]

    # District-style: "TX-15", "AZ-1", "CA-27" -> state abbr
    district_match = re.match(r"^([A-Za-z]{2})-\d+", s, re.IGNORECASE)
    if district_match:
        abbr = district_match.group(1).upper()
        if abbr in ("AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"):
            return abbr

    return None


def _parse_date_to_iso(s: str) -> Optional[str]:
    """
    Parse a single date string to YYYY-MM-DD. Uses today for year when missing;
    if the parsed date would be in the future, uses previous year.
    Handles: ISO, 'Weekday, Month Day', 'Feb 9', 'Feb 9, 2026', '2/9/2026', '2/9/26', '2/9'.
    """
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    today = date.today()
    year = today.year

    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    # Explicit 4-digit year in string
    year_match = re.search(r"\b(19|20)\d{2}\b", s)
    if year_match:
        year = int(year_match.group(0))

    # Numeric: 2/9/2026, 2/9/26, 2/9, 02/09/2026
    num_match = re.match(r"^(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?$", s)
    if num_match:
        m, d = int(num_match.group(1)), int(num_match.group(2))
        y = num_match.group(3)
        if y:
            y = int(y)
            if y < 100:
                y = 2000 + y if y >= 30 else 2000 + y
            year = y
        if 1 <= m <= 12 and 1 <= d <= 31:
            try:
                parsed = date(year, m, d)
                if parsed > today:
                    parsed = date(year - 1, m, d)
                return parsed.isoformat()
            except ValueError:
                pass

    sl = s.lower()

    # Full month: "Wednesday, February 11", "February 11"
    for i, month in enumerate(_MONTH_NAMES_FULL):
        if month in sl:
            day_m = re.search(r"\d+", s)
            day = int(day_m.group()) if day_m else 1
            month_num = i + 1
            try:
                parsed = date(year, month_num, day)
                if parsed > today:
                    parsed = date(year - 1, month_num, day)
                return parsed.isoformat()
            except ValueError:
                parsed = date(year, month_num, min(day, 28))
                return parsed.isoformat()

    # Abbreviated month: "Feb 9", "Feb 9, 2026", "Feb. 9"
    for abbr, month_num in _MONTH_ABBR_TO_NUM.items():
        if abbr in sl or (abbr + ".") in sl:
            day_m = re.search(r"\d+", s)
            day = int(day_m.group()) if day_m else 1
            try:
                parsed = date(year, month_num, day)
                if parsed > today:
                    parsed = date(year - 1, month_num, day)
                return parsed.isoformat()
            except ValueError:
                return f"{year}-{month_num:02d}-{min(day, 28):02d}"
    return None


def _end_date_sort_key(end_date: Any) -> str:
    """Return YYYY-MM-DD for sorting; 0000-00-00 if unparseable."""
    out = _parse_date_to_iso(str(end_date).strip() if end_date else "")
    return out or "0000-00-00"


def _best_date_sort_key(end_date: Any, date_added: Any) -> str:
    """Best sort key from end_date or date_added so we sort by the most meaningful date."""
    key = _parse_date_to_iso(str(end_date).strip() if end_date else "")
    if key:
        return key
    key = _parse_date_to_iso(str(date_added).strip() if date_added else "")
    return key or "0000-00-00"

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "election.db"


def make_poll_id(source: str, pollster: str, start_date: str, race_key: str) -> str:
    raw = f"{source}|{pollster}|{start_date}|{race_key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ElectionDB:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or DB_PATH

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS polls (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    race_type TEXT NOT NULL,
                    race_key TEXT,
                    state TEXT,
                    pollster TEXT,
                    poll_url TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    date_added TEXT,
                    sample_size TEXT,
                    population TEXT,
                    grade TEXT,
                    margin TEXT,
                    results_json TEXT,
                    raw_data_json TEXT,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(source, pollster, start_date, race_key)
                );
                CREATE INDEX IF NOT EXISTS idx_polls_race ON polls(race_type, race_key);
                CREATE INDEX IF NOT EXISTS idx_polls_date ON polls(end_date DESC);
                CREATE INDEX IF NOT EXISTS idx_polls_state ON polls(state);

                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    race_type TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    poll_count INTEGER,
                    status TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS user_map_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    race_type TEXT NOT NULL,
                    candidate_1_name TEXT,
                    candidate_1_party TEXT,
                    candidate_1_pct REAL,
                    candidate_2_name TEXT,
                    candidate_2_party TEXT,
                    candidate_2_pct REAL,
                    margin REAL,
                    source_note TEXT,
                    updated_at TEXT NOT NULL,
                    UNIQUE(state, race_type)
                );

                CREATE TABLE IF NOT EXISTS ai_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_type TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    UNIQUE(race_type, context_hash)
                );
            """)
            await db.commit()
        logger.info("Election DB initialized at %s", self.path)

    async def upsert_polls(self, polls: List[Dict[str, Any]]) -> int:
        """Insert or replace polls. Each poll dict must include id or be computable. Returns count upserted.
        Never deletes: refresh only adds/updates rows (ON CONFLICT DO UPDATE), so historical data is retained
        for map averages and trends."""
        if not polls:
            return 0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        async with aiosqlite.connect(self.path) as db:
            for p in polls:
                pid = p.get("id")
                if not pid:
                    pid = make_poll_id(
                        p.get("source", ""),
                        p.get("pollster", ""),
                        p.get("start_date", ""),
                        p.get("race_key", ""),
                    )
                results_json = json.dumps(p.get("results") or p.get("results_json") or {})
                raw_json = json.dumps(p.get("raw_data") or p.get("raw_data_json") or {})
                try:
                    await db.execute(
                        """
                        INSERT INTO polls (
                            id, source, race_type, race_key, state, pollster, poll_url,
                            start_date, end_date, date_added, sample_size, population,
                            grade, margin, results_json, raw_data_json, fetched_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(source, pollster, start_date, race_key) DO UPDATE SET
                            end_date=excluded.end_date, date_added=excluded.date_added,
                            sample_size=excluded.sample_size, population=excluded.population,
                            grade=excluded.grade, margin=excluded.margin,
                            results_json=excluded.results_json, raw_data_json=excluded.raw_data_json,
                            fetched_at=excluded.fetched_at
                        """,
                        (
                            pid,
                            p.get("source", ""),
                            p.get("race_type", ""),
                            p.get("race_key") or "",
                            p.get("state"),
                            p.get("pollster") or "",
                            p.get("poll_url") or "",
                            p.get("start_date") or "",
                            p.get("end_date") or "",
                            p.get("date_added") or "",
                            p.get("sample_size") or "",
                            p.get("population") or "",
                            p.get("grade") or "",
                            p.get("margin") or "",
                            results_json,
                            raw_json,
                            p.get("fetched_at") or now,
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Upsert poll failed for id=%s: %s", pid, e)
            await db.commit()
        return count

    async def get_polls(
        self,
        race_type: str,
        state: Optional[str] = None,
        limit: int = 100,
        sources: Optional[List[str]] = None,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get polls for a race, optionally filtered by state, source list, and/or time window.
        When days_back is set, only polls with end_date/date_added within the last N days are returned
        (so map/trends can use a full window of data without losing older polls as new ones arrive)."""
        # When using a time window, fetch more rows then filter by date (end_date is stored in various formats).
        internal_limit = limit
        if days_back is not None and days_back > 0:
            internal_limit = min(5000, max(limit, 2000))
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            where_parts = ["race_type = ?"]
            params: List[Any] = [race_type]
            if state:
                where_parts.append("(state IS NULL OR state = ? OR state = ?)")
                params.extend([state, state.upper()])
            if sources:
                placeholders = ",".join("?" * len(sources))
                where_parts.append(f"source IN ({placeholders})")
                params.extend(sources)
            params.append(internal_limit)
            where_sql = " AND ".join(where_parts)
            cursor = await db.execute(
                f"""
                SELECT id, source, race_type, race_key, state, pollster, poll_url,
                       start_date, end_date, date_added, sample_size, population,
                       grade, margin, results_json, raw_data_json, fetched_at
                FROM polls
                WHERE {where_sql}
                ORDER BY end_date DESC
                LIMIT ?
                """,
                params,
            )
            rows = await cursor.fetchall()
        out = []
        cutoff_iso = None
        today_iso = date.today().isoformat()
        if days_back is not None and days_back > 0:
            cutoff_iso = (date.today() - timedelta(days=days_back)).isoformat()
        for row in rows:
            r = dict(row)
            if r.get("results_json"):
                try:
                    r["results"] = json.loads(r["results_json"])
                except Exception:
                    r["results"] = {}
            r.pop("results_json", None)
            # RCP: use human-readable race name from raw_data; derive state from race_name when missing
            raw_data_json = r.pop("raw_data_json", None)
            if raw_data_json and r.get("source") == "rcp":
                try:
                    raw_data = json.loads(raw_data_json)
                    if isinstance(raw_data, dict) and raw_data.get("race_name"):
                        r["race"] = raw_data["race_name"]
                        if not r.get("state"):
                            r["state"] = _state_from_race_name(raw_data["race_name"])
                    else:
                        r["race"] = r.get("race_key") or ""
                except Exception:
                    r["race"] = r.get("race_key") or ""
            else:
                r["race"] = r.get("race_key") or ""
            # RTWH: derive state from Race or Candidate text when missing (e.g. "Alabama GOP Primary", "J. Cornyn Fav. (TX-R)")
            if r.get("source") == "racetothewh" and not r.get("state"):
                derived = _state_from_race_text(r.get("race") or "")
                if derived:
                    r["state"] = derived
            # Frontend expects margin/lead, added
            r["margin"] = r.get("margin") or ""
            r["lead"] = r.get("margin") or ""
            r["added"] = r.get("date_added") or r.get("end_date") or ""
            r["date_added"] = r.get("added")
            if cutoff_iso:
                poll_iso = _parse_date_to_iso(str(r.get("end_date") or r.get("date_added") or "").strip())
                if not poll_iso or poll_iso < cutoff_iso:
                    continue
                # Exclude future-dated polls (e.g. misparsed 2025 race stored as 2026)
                if poll_iso > today_iso:
                    continue
                r["date_iso"] = poll_iso
            out.append(r)
        # Ensure every row has a parsed date for sort and display (so UI can show year: Dec 22, 2025)
        for p in out:
            if not p.get("date_iso"):
                p["date_iso"] = _best_date_sort_key(p.get("end_date"), p.get("date_added"))
        # Sort by calendar date, newest first (date-aware; impossible for Feb 2026 to show "Dec 2026" as current)
        out.sort(key=lambda p: p.get("date_iso") or "0000-00-00", reverse=True)
        # Deduplicate: same logical poll can appear twice (e.g. RCP lists under two date rows). Keep first by (source, pollster, state, race, margin, date_iso).
        seen_keys = set()
        deduped = []
        for p in out:
            race = (p.get("race") or p.get("race_key") or "").strip()
            state = (p.get("state") or "").strip().upper()
            key = (
                p.get("source") or "",
                (p.get("pollster") or "").strip(),
                state,
                race,
                (p.get("margin") or "").strip(),
                p.get("date_iso") or "",
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(p)
        return deduped[:limit]

    async def get_race_metadata(self, race_type: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Simple metadata: candidates list from latest polls."""
        polls = await self.get_polls(race_type, state=state, limit=20)
        candidates = []
        for p in polls:
            res = p.get("results") or {}
            for k in res:
                if k and k not in candidates:
                    candidates.append(k)
        return {"candidates": candidates, "title": f"{race_type.replace('_', ' ').title()} Polls"}

    async def get_last_fetch_time(self, race_type: str, sources: Optional[List[str]] = None) -> Optional[str]:
        """Latest fetched_at for this race_type, optionally limited to given sources."""
        async with aiosqlite.connect(self.path) as db:
            if sources:
                placeholders = ",".join("?" * len(sources))
                cursor = await db.execute(
                    f"SELECT fetched_at FROM polls WHERE race_type = ? AND source IN ({placeholders}) ORDER BY fetched_at DESC LIMIT 1",
                    (race_type, *sources),
                )
            else:
                cursor = await db.execute(
                    "SELECT fetched_at FROM polls WHERE race_type = ? ORDER BY fetched_at DESC LIMIT 1",
                    (race_type,),
                )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def get_sources_for_race(self, race_type: str) -> List[str]:
        """Distinct sources that have polls for this race."""
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT DISTINCT source FROM polls WHERE race_type = ?",
                (race_type,),
            )
            rows = await cursor.fetchall()
        return [r[0] for r in rows] if rows else []

    async def log_fetch(self, source: str, race_type: str, poll_count: int, status: str = "success", error_message: Optional[str] = None) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO fetch_log (source, race_type, fetched_at, poll_count, status, error_message) VALUES (?, ?, ?, ?, ?, ?)",
                (source, race_type, now, poll_count, status, error_message),
            )
            await db.commit()

    # --- Trend aggregation ---
    async def get_trend_data(self, race_type: str, days: int = 90) -> List[Dict[str, Any]]:
        """Time-series averages by week (Sunday of each week) for charting. Dem/GOP/Approve/Disapprove.
        results_json keys vary: Dem/GOP, Democrat/Republican, Candidate 1/Candidate 2, Result1/Result2, etc."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            # First value: use COALESCE over all known keys so generic_ballot (Candidate 1/2) and Dem/GOP both work
            dem_val = "COALESCE(json_extract(results_json, '$.Dem'), json_extract(results_json, '$.Democrat'), json_extract(results_json, '$.Candidate 1'), json_extract(results_json, '$.Result1'))"
            gop_val = "COALESCE(json_extract(results_json, '$.GOP'), json_extract(results_json, '$.Republican'), json_extract(results_json, '$.Candidate 2'), json_extract(results_json, '$.Result2'), json_extract(results_json, '$.Rep'))"
            approve_val = "COALESCE(json_extract(results_json, '$.Approve'), json_extract(results_json, '$.Favorable'))"
            disapprove_val = "COALESCE(json_extract(results_json, '$.Disapprove'), json_extract(results_json, '$.Unfavorable'))"
            cursor = await db.execute(
                f"""
                SELECT
                    date(end_date, '-' || strftime('%w', end_date) || ' days') AS date,
                    AVG(CASE WHEN {dem_val} IS NOT NULL AND CAST({dem_val} AS TEXT) != '' THEN CAST({dem_val} AS REAL) END) AS dem_avg,
                    AVG(CASE WHEN {gop_val} IS NOT NULL AND CAST({gop_val} AS TEXT) != '' THEN CAST({gop_val} AS REAL) END) AS gop_avg,
                    AVG(CASE WHEN json_extract(results_json, '$.Rep') IS NOT NULL THEN CAST(json_extract(results_json, '$.Rep') AS REAL) END) AS rep_avg,
                    AVG(CASE WHEN {approve_val} IS NOT NULL AND CAST({approve_val} AS TEXT) != '' THEN CAST({approve_val} AS REAL) END) AS approve_avg,
                    AVG(CASE WHEN {disapprove_val} IS NOT NULL AND CAST({disapprove_val} AS TEXT) != '' THEN CAST({disapprove_val} AS REAL) END) AS disapprove_avg,
                    COUNT(*) AS poll_count
                FROM polls
                WHERE race_type = ?
                  AND end_date >= date('now', '-' || ? || ' days')
                  AND end_date != ''
                  AND date(end_date, '-' || strftime('%w', end_date) || ' days') IS NOT NULL
                GROUP BY date(end_date, '-' || strftime('%w', end_date) || ' days')
                ORDER BY date ASC
                """,
                (race_type, days),
            )
            rows = await cursor.fetchall()
        out = []
        for row in rows:
            d = dict(row)
            # Use rep_avg if gop_avg is null (VoteHub uses "Rep")
            if d.get("gop_avg") is None and d.get("rep_avg") is not None:
                d["gop_avg"] = d["rep_avg"]
            out.append(d)
        return out

    # --- User map data ---
    async def get_all_map_data(self, race_type: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT state, race_type, candidate_1_name, candidate_1_party, candidate_1_pct, "
                "candidate_2_name, candidate_2_party, candidate_2_pct, margin, source_note, updated_at "
                "FROM user_map_data WHERE race_type = ?",
                (race_type,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_state_level_averages(self, race_type: str) -> Dict[str, Dict[str, Any]]:
        """Aggregate polls by state for map. Uses same result keys as trends (Dem/GOP, Candidate 1/2, etc.). State normalized to 2-letter upper."""
        dem_val = "COALESCE(json_extract(results_json, '$.Dem'), json_extract(results_json, '$.Democrat'), json_extract(results_json, '$.Candidate 1'), json_extract(results_json, '$.Result1'))"
        gop_val = "COALESCE(json_extract(results_json, '$.GOP'), json_extract(results_json, '$.Republican'), json_extract(results_json, '$.Candidate 2'), json_extract(results_json, '$.Result2'), json_extract(results_json, '$.Rep'))"
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT UPPER(TRIM(state)) AS state,
                       AVG(CASE WHEN {dem_val} IS NOT NULL AND CAST({dem_val} AS TEXT) != '' THEN CAST({dem_val} AS REAL) END) AS dem_avg,
                       AVG(CASE WHEN {gop_val} IS NOT NULL AND CAST({gop_val} AS TEXT) != '' THEN CAST({gop_val} AS REAL) END) AS gop_avg,
                       COUNT(*) AS poll_count
                FROM polls
                WHERE race_type = ? AND state IS NOT NULL AND TRIM(state) != ''
                GROUP BY UPPER(TRIM(state))
                """,
                (race_type,),
            )
            rows = await cursor.fetchall()
        return {r["state"]: dict(r) for r in rows} if rows else {}

    async def upsert_map_data(
        self,
        state: str,
        race_type: str,
        candidate_1_name: Optional[str] = None,
        candidate_1_party: Optional[str] = None,
        candidate_1_pct: Optional[float] = None,
        candidate_2_name: Optional[str] = None,
        candidate_2_party: Optional[str] = None,
        candidate_2_pct: Optional[float] = None,
        margin: Optional[float] = None,
        source_note: Optional[str] = None,
    ) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO user_map_data (
                    state, race_type, candidate_1_name, candidate_1_party, candidate_1_pct,
                    candidate_2_name, candidate_2_party, candidate_2_pct, margin, source_note, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state, race_type) DO UPDATE SET
                    candidate_1_name=excluded.candidate_1_name, candidate_1_party=excluded.candidate_1_party,
                    candidate_1_pct=excluded.candidate_1_pct, candidate_2_name=excluded.candidate_2_name,
                    candidate_2_party=excluded.candidate_2_party, candidate_2_pct=excluded.candidate_2_pct,
                    margin=excluded.margin, source_note=excluded.source_note, updated_at=excluded.updated_at
                """,
                (
                    state.upper(),
                    race_type,
                    candidate_1_name,
                    candidate_1_party,
                    candidate_1_pct,
                    candidate_2_name,
                    candidate_2_party,
                    candidate_2_pct,
                    margin,
                    source_note,
                    now,
                ),
            )
            await db.commit()

    async def delete_map_data(self, state: str, race_type: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "DELETE FROM user_map_data WHERE state = ? AND race_type = ?",
                (state.upper(), race_type),
            )
            await db.commit()

    # --- AI questions cache ---
    async def get_cached_questions(self, race_type: str, context_hash: str) -> Optional[List[str]]:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT questions_json FROM ai_questions WHERE race_type = ? AND context_hash = ?",
                (race_type, context_hash),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    async def cache_questions(self, race_type: str, context_hash: str, questions: List[str]) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO ai_questions (race_type, context_hash, questions_json, generated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(race_type, context_hash) DO UPDATE SET
                    questions_json=excluded.questions_json, generated_at=excluded.generated_at
                """,
                (race_type, context_hash, json.dumps(questions), now),
            )
            await db.commit()


election_db = ElectionDB()
