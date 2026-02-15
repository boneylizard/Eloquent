"""
VoteHub API integration. Fetches structured poll data (approval, generic ballot, etc.).
No API key required. CC BY 4.0.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .election_db import make_poll_id

logger = logging.getLogger(__name__)

# Try documented base; plan used votehub.com/polls/api
VOTEHUB_BASE = "https://api.votehub.com"
VOTEHUB_POLLS_URL = f"{VOTEHUB_BASE}/polls"


def _calculate_margin(results: Dict[str, str]) -> str:
    """Compute margin string from results dict. Dem/GOP or Approve/Disapprove."""
    if not results:
        return ""
    keys = list(results.keys())
    if len(keys) < 2:
        return ""
    try:
        v1 = float(str(results.get(keys[0], "0")).replace("%", ""))
        v2 = float(str(results.get(keys[1], "0")).replace("%", ""))
        diff = v1 - v2
        if abs(diff) < 0.1:
            return "Tie"
        return f"{keys[0][:1]}+{abs(diff):.1f}" if diff > 0 else f"{keys[1][:1]}+{abs(diff):.1f}"
    except (ValueError, TypeError):
        return ""


def map_votehub_type(vt: str) -> str:
    return {
        "generic-ballot": "generic_ballot",
        "approval": "approval",
        "favorability": "favorability",
    }.get(vt, vt.replace("-", "_"))


async def fetch_votehub_polls(
    poll_type: str,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch polls from VoteHub API and normalize to our schema."""
    params: Dict[str, str] = {"poll_type": poll_type}
    if subject:
        params["subject"] = subject

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(VOTEHUB_POLLS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("VoteHub fetch failed for %s: %s", poll_type, e)
        return []

    # API may return {"polls": [...]} or a raw list
    if isinstance(data, list):
        raw_polls = data
    else:
        raw_polls = data.get("polls", []) if isinstance(data, dict) else []

    polls: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for p in raw_polls:
        if not isinstance(p, dict):
            continue
        results = {}
        for answer in p.get("answers", []):
            if not isinstance(answer, dict):
                continue
            choice = answer.get("choice", "")
            pct = answer.get("pct")
            if pct is not None:
                results[choice] = str(pct)

        subject_val = p.get("subject", poll_type)
        race_key = f"{subject_val}-{poll_type}"
        race_type = map_votehub_type(p.get("poll_type", poll_type))

        poll = {
            "source": "votehub",
            "race_type": race_type,
            "race_key": race_key,
            "state": None,
            "pollster": p.get("pollster", ""),
            "poll_url": p.get("url", ""),
            "start_date": p.get("start_date", ""),
            "end_date": p.get("end_date", ""),
            "date_added": p.get("created_at", ""),
            "sample_size": str(p.get("sample_size", "")),
            "population": p.get("population", ""),
            "grade": "",
            "margin": _calculate_margin(results),
            "results": results,
            "raw_data": p,
            "fetched_at": now,
        }
        poll["id"] = make_poll_id(
            poll["source"],
            poll["pollster"],
            poll["start_date"],
            poll["race_key"],
        )
        polls.append(poll)

    return polls


async def refresh_votehub_all() -> Dict[str, int]:
    """Fetch all VoteHub poll types we care about and return counts by race_type."""
    from .election_db import election_db

    counts: Dict[str, int] = {}
    # VoteHub: approval, generic-ballot, favorability (subject for approval/favorability)
    tasks = [
        ("approval", "approval", "Donald-Trump"),
        ("generic_ballot", "generic-ballot", "2026"),
        ("generic_ballot", "generic-ballot", None),
    ]
    for race_type, poll_type, subject in tasks:
        polls = await fetch_votehub_polls(poll_type, subject)
        if polls:
            n = await election_db.upsert_polls(polls)
            counts[race_type] = counts.get(race_type, 0) + n
            await election_db.log_fetch("votehub", race_type, n)
    return counts
