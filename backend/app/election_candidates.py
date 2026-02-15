"""
2026 candidate roster: load from 2026_candidates.json and resolve name -> party (D/R/I).
Used by map state averages, poll display, and AI context.
To refresh the roster from the datasheet: run `python -m app.build_candidates_from_jsonl` from backend (reads candidates.jsonl, writes data/2026_candidates.json).
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "data" / "2026_candidates.json"
_roster: Optional[List[Dict]] = None
_name_to_party: Optional[Dict[str, str]] = None


def _normalize_name(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip()).lower()


def get_candidates() -> List[Dict]:
    """Return the full candidate roster for API/frontend."""
    return _load_roster()


def _load_roster() -> List[Dict]:
    global _roster
    if _roster is not None:
        return _roster
    if not _CANDIDATES_PATH.exists():
        _roster = []
        return _roster
    try:
        with open(_CANDIDATES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _roster = data.get("candidates") or []
        return _roster
    except Exception:
        _roster = []
        return _roster


def _build_name_to_party() -> Dict[str, str]:
    """Map normalized name and aliases -> party. First match wins; prefer longer keys for specificity."""
    global _name_to_party
    if _name_to_party is not None:
        return _name_to_party
    _name_to_party = {}
    for c in _load_roster():
        party = (c.get("party") or "").strip().upper()
        if party not in ("D", "R", "I", "G", "L", "F", "C"):
            continue
        name = _normalize_name(c.get("name") or "")
        if name and name not in _name_to_party:
            _name_to_party[name] = party
        for alias in c.get("aliases") or []:
            a = _normalize_name(alias)
            if a and a not in _name_to_party:
                _name_to_party[a] = party
    return _name_to_party


def get_party_for_candidate(
    name: str,
    state: Optional[str] = None,
    office: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve candidate name (or alias) to party: D, R, I, G, L.
    state/office are optional hints to disambiguate (e.g. same last name in different races).
    """
    if not name or not isinstance(name, str):
        return None
    lookup = _build_name_to_party()
    key = _normalize_name(name)
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    roster = _load_roster()
    for c in roster:
        if state and (c.get("state") or "").upper() != (state or "").upper():
            continue
        if office and (c.get("office") or "").lower() != (office or "").lower():
            continue
        if _normalize_name(c.get("name")) == key:
            return (c.get("party") or "").strip().upper()
        for alias in c.get("aliases") or []:
            if _normalize_name(alias) == key:
                return (c.get("party") or "").strip().upper()
    return None


def get_dem_gop_from_candidate_names(
    names_and_values: List[Tuple[str, float]],
    state: Optional[str] = None,
    office: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """
    Given [(candidate_name, pct), ...], return (dem_pct, gop_pct).
    If both candidates are in the roster, use both. If only one is known, assign the other
    to the opposite party (two-way race). Returns None only when neither candidate is known.
    """
    if not names_and_values or len(names_and_values) != 2:
        return None
    (n1, v1), (n2, v2) = names_and_values[0], names_and_values[1]
    p1 = get_party_for_candidate(n1, state=state, office=office)
    p2 = get_party_for_candidate(n2, state=state, office=office)
    dem = None
    gop = None
    if p1 in ("D", "G", "L"):
        dem = v1
    elif p1 == "R":
        gop = v1
    if p2 in ("D", "G", "L"):
        dem = v2
    elif p2 == "R":
        gop = v2
    if dem is not None and gop is not None:
        return (dem, gop)
    if p1 and not p2:
        if p1 == "R":
            return (v2, v1)
        return (v1, v2)
    if p2 and not p1:
        if p2 == "R":
            return (v1, v2)
        return (v2, v1)
    return None


def get_roster_for_ai() -> str:
    """Return a short, readable summary of the candidate roster for AI context (names and parties by state/office)."""
    roster = _load_roster()
    if not roster:
        return ""
    lines = ["2026 candidate roster (name -> party):"]
    by_office = {}
    for c in roster:
        office = (c.get("office") or "other").lower()
        if office not in by_office:
            by_office[office] = []
        by_office[office].append(c)
    for office in ["senate", "governor", "house"]:
        if office not in by_office:
            continue
        lines.append(f"\n{office.upper()}:")
        for c in by_office[office]:
            name = c.get("name") or "?"
            party = c.get("party") or "?"
            st = (c.get("state") or "").upper()
            lines.append(f"  {name} ({party}) {st}")
    return "\n".join(lines)
