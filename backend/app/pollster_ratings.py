"""
FiveThirtyEight pollster ratings for the Elections Map confidence layer.

Uses pollster-ratings-combined.csv from the FiveThirtyEight public data repository as the
authoritative source. Ratings are used directly—no new rating system or derivation.

Source: https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/pollster-ratings-combined.csv
Columns used: pollster, numeric_grade, bias_ppm, wtd_avg_transparency (inactive pollsters have NA grade).
"""
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Any

_PATH = Path(__file__).resolve().parent.parent / "data" / "pollster-ratings-combined.csv"
_FTE_CSV_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/pollster-ratings/pollster-ratings-combined.csv"
_cache: Optional[Dict[str, Dict[str, Any]]] = None

# Map our polling data pollster names to FTE dataset names when they differ (e.g. abbreviations).
# Add entries as we find mismatches. Keys are normalized (lower, single spaces).
_POLLSTER_ALIASES: Dict[str, str] = {
    "unh": "University of New Hampshire Survey Center",
    "unh survey center": "University of New Hampshire Survey Center",
}


def _normalize(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    # Strip trailing grade in parens (e.g. "Emerson (B)" from RCP/Infogram)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name.strip())
    return re.sub(r"\s+", " ", name.strip()).lower()


def _load_ratings() -> Dict[str, Dict[str, Any]]:
    """Load FTE CSV and return normalized name -> {numeric_grade, bias_ppm, transparency}."""
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    path = _PATH
    csv_content = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                csv_content = f.read()
        except Exception:
            pass
    if not csv_content:
        try:
            import urllib.request
            with urllib.request.urlopen(_FTE_CSV_URL, timeout=15) as resp:
                csv_content = resp.read().decode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(csv_content)
        except Exception:
            return _cache
    try:
        from io import StringIO
        reader = csv.DictReader(StringIO(csv_content))
        for row in reader:
                pollster = (row.get("pollster") or "").strip()
                if not pollster:
                    continue
                raw_grade = (row.get("numeric_grade") or "").strip()
                if raw_grade.upper() == "NA" or raw_grade == "":
                    continue
                try:
                    numeric_grade = float(raw_grade)
                except ValueError:
                    continue
                bias_raw = (row.get("bias_ppm") or "").strip()
                bias_ppm = float(bias_raw) if bias_raw and bias_raw.upper() != "NA" else None
                trans_raw = (row.get("wtd_avg_transparency") or "").strip()
                transparency = float(trans_raw) if trans_raw and trans_raw.upper() != "NA" else None
                key = _normalize(pollster)
                if key and key not in _cache:
                    _cache[key] = {
                        "numeric_grade": numeric_grade,
                        "bias_ppm": bias_ppm,
                        "transparency": transparency,
                        "pollster": pollster,
                    }
    except Exception:
        _cache = {}
    return _cache


def get_rating(pollster_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the FTE rating for a pollster, or None if not found.
    Uses exact normalized match first, then alias map, then prefix/substring match
    so RCP short names (e.g. "Emerson", "Siena") match FTE full names ("Emerson College", "Siena College").
    """
    if not pollster_name or not isinstance(pollster_name, str):
        return None
    ratings = _load_ratings()
    key = _normalize(pollster_name)
    if not key:
        return None
    if key in ratings:
        return ratings[key]
    canonical = _POLLSTER_ALIASES.get(key)
    if canonical:
        return ratings.get(_normalize(canonical))
    # Fallback: our name may be a shortened form (e.g. "Emerson" -> "Emerson College")
    if len(key) >= 3:
        best = None
        best_len = 0
        for fte_key, data in ratings.items():
            if fte_key == key:
                return data
            # FTE name starts with our name as a word (e.g. "emerson college" for "emerson")
            if fte_key.startswith(key + " ") or fte_key.startswith(key + "/"):
                if len(fte_key) > best_len:
                    best, best_len = data, len(fte_key)
        if best is not None:
            return best
    return None


def get_state_confidence_from_pollsters(pollster_names: list) -> tuple:
    """
    Compute a 0–1 confidence score and a short explanation from a list of pollster names.
    Uses FTE numeric_grade (1–3) directly: higher grade = higher confidence.
    Unrated pollsters contribute a neutral 0.5 so they don't dominate.

    Returns (confidence_0_to_1, explanation_string).
    """
    if not pollster_names:
        return (0.0, "No polls")
    grades = []
    rated = 0
    for name in pollster_names:
        r = get_rating(name)
        if r is not None and r.get("numeric_grade") is not None:
            g = r["numeric_grade"]
            if 0 <= g <= 3:
                grades.append(g)
                rated += 1
    n = len(pollster_names)
    if not grades:
        return (0.5, f"Average based on {n} poll(s); no FiveThirtyEight rating")
    # Scale grade 1–3 to 0–1: (grade - 1) / 2. Then blend with neutral for unrated.
    avg_grade = sum(grades) / len(grades)
    score_rated = (avg_grade - 1.0) / 2.0  # 1 -> 0, 2 -> 0.5, 3 -> 1
    confidence = (rated / n) * score_rated + (1 - rated / n) * 0.5
    confidence = max(0.0, min(1.0, confidence))
    note = f"FiveThirtyEight avg grade {avg_grade:.1f}/3 ({rated} of {n} polls rated)"
    return (round(confidence, 3), note)
