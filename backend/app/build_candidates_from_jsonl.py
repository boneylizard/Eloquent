"""
Build backend/data/2026_candidates.json from backend/app/candidates.jsonl.

Run from backend directory:
  python -m app.build_candidates_from_jsonl

The jsonl is the source of truth. This script normalizes office/party, generates
aliases (last name, first initial + last name, accent variants like Luján->Lujan)
so polls showing "Cornyn", "J. Cornyn", or "Lujan" still match. Covers everyone
in the datasheet so map and Polls tab can resolve D/R for all candidates.

Nicknames: To add extra aliases for specific candidates (e.g. "MTG" for Marjorie
Taylor Greene), add entries to _MANUAL_ALIASES in this file, keyed by
(state_abbr, office, normalized_full_name). Re-run the script after editing
candidates.jsonl or _MANUAL_ALIASES.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Script lives in backend/app; data and jsonl paths relative to it
_APP_DIR = Path(__file__).resolve().parent
_JSONL_PATH = _APP_DIR / "candidates.jsonl"
_OUT_PATH = _APP_DIR.parent / "data" / "2026_candidates.json"

# Party label in jsonl -> single letter for roster
_PARTY_MAP = {
    "democratic": "D",
    "republican": "R",
    "dfl": "D",
    "independent": "I",
    "green": "G",
    "libertarian": "L",
    "freedom and unity party": "F",
    "potential": None,  # skip or use same as announced; we keep and map by context
}

# Optional: add nicknames / alternate spellings so polls match. Key = normalized (lower), value = canonical name to resolve to (we still emit by canonical).
# Example: "mtg" -> "Marjorie Taylor Greene", "j. cornyn" -> "John Cornyn". Here we only add aliases to the emitted entry; for true nickname resolution we'd need to map nickname -> existing candidate.
# Manual aliases to add for specific candidates (name_key -> list of extra aliases). name_key = (state_abbr, office, normalized_full_name).
_MANUAL_ALIASES: Dict[Tuple[str, str, str], List[str]] = {
    # ("TX", "senate", "john cornyn"): ["J. Cornyn", "Cornyn"],
    # Add more as you find poll formats that miss: e.g. ("OH", "senate", "sherrod brown"): ["S. Brown"],
}


def _normalize_party(party: str) -> str | None:
    if not party:
        return None
    key = party.strip().lower()
    return _PARTY_MAP.get(key)


def _normalize_office(office: str) -> str:
    if not office:
        return "senate"
    o = office.strip().lower()
    if "governor" in o:
        return "governor"
    if "senate" in o:
        return "senate"
    if "house" in o:
        return "house"
    return "senate"


def _alias_strip_accents(s: str) -> str:
    """Return ASCII-friendly version for alias (e.g. Luján -> Lujan)."""
    if not s:
        return s
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _aliases_for_name(full_name: str) -> List[str]:
    """Generate aliases so polls matching 'Cornyn', 'J. Cornyn', or 'Lujan' still resolve."""
    if not full_name or not full_name.strip():
        return []
    name = full_name.strip()
    parts = [p for p in re.split(r"\s+", name) if p]
    if not parts:
        return []
    seen: Set[str] = set()
    out: List[str] = []

    # Last word (last name)
    last = parts[-1]
    if last not in seen:
        seen.add(last)
        out.append(last)
    # First initial + last name (e.g. J. Cornyn)
    if len(parts) >= 2:
        first_initial = parts[0][0] if parts[0] else ""
        fl = f"{first_initial} {last}".strip()
        if fl and fl not in seen:
            seen.add(fl)
            out.append(fl)
    # Accent-stripped last name (e.g. Lujan from Luján)
    last_ascii = _alias_strip_accents(last)
    if last_ascii != last and last_ascii not in seen:
        seen.add(last_ascii)
        out.append(last_ascii)
    # Compound last name (e.g. "Moore Capito", "Lance Bottoms")
    if len(parts) >= 3:
        compound = " ".join(parts[-2:])
        if compound not in seen:
            seen.add(compound)
            out.append(compound)
    return out


def _normalize_name_for_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def run() -> None:
    if not _JSONL_PATH.exists():
        print(f"Missing {_JSONL_PATH}")
        return
    seen: Set[Tuple[str, str, str]] = set()  # (state_abbr, office, normalized_name)
    candidates: List[Dict[str, Any]] = []

    with open(_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping bad line: {e}")
                continue
            state_abbr = (row.get("state_abbr") or "").strip().upper()
            if len(state_abbr) != 2:
                continue
            office = _normalize_office(row.get("office") or "")
            for c in row.get("candidates") or []:
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                party_label = (c.get("party") or "").strip()
                party = _normalize_party(party_label)
                if party is None and party_label:
                    party = "I"  # unknown party -> I for now
                elif party is None:
                    continue
                name_key = _normalize_name_for_key(name)
                dedupe = (state_abbr, office, name_key)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                aliases = _aliases_for_name(name)
                manual = _MANUAL_ALIASES.get((state_abbr, office, name_key))
                if manual:
                    for a in manual:
                        if a and _normalize_name_for_key(a) not in {_normalize_name_for_key(x) for x in aliases}:
                            aliases.append(a)
                candidates.append({
                    "name": name,
                    "aliases": list(dict.fromkeys(aliases)),  # preserve order, no dupes
                    "party": party,
                    "office": office,
                    "state": state_abbr,
                })

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "meta": {
            "source": "candidates.jsonl",
            "description": "Parseable candidate roster for 2026: name/alias -> party (D/R/I). Built from candidates.jsonl. Used for poll display, map averages, and AI context.",
            "party_values": ["D", "R", "I", "G", "L", "F", "C"],
        },
        "candidates": candidates,
    }
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(candidates)} candidates to {_OUT_PATH}")


if __name__ == "__main__":
    run()
