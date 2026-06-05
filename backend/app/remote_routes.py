from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/remote/v1", tags=["remote"])

_LOCK = Lock()
_SESSIONS: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"next_id": 1, "commands": [], "state": {}}
)
_MAX_COMMANDS = 500


class RemoteCommandIn(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=80)
    command: str = Field(..., min_length=2, max_length=64)
    payload: Optional[Dict[str, Any]] = None
    source: Optional[str] = None


@router.post("/command")
async def post_remote_command(body: RemoteCommandIn):
    sid = body.session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    with _LOCK:
        session = _SESSIONS[sid]
        cid = int(session["next_id"])
        session["next_id"] = cid + 1
        entry = {
            "id": cid,
            "command": body.command.strip().lower(),
            "payload": body.payload or {},
            "source": (body.source or "").strip(),
        }
        session["commands"].append(entry)
        if len(session["commands"]) > _MAX_COMMANDS:
            session["commands"] = session["commands"][-_MAX_COMMANDS:]
    return {"status": "ok", "queued": entry}


@router.get("/commands")
async def get_remote_commands(
    session_id: str = Query(..., min_length=3, max_length=80),
    after_id: int = Query(0, ge=0),
):
    sid = session_id.strip()
    with _LOCK:
        session = _SESSIONS.get(sid)
        if not session:
            return {"status": "ok", "commands": [], "last_id": after_id}
        commands: List[Dict[str, Any]] = [
            c for c in session["commands"] if int(c.get("id", 0)) > after_id
        ]
        last_id = after_id
        if commands:
            last_id = int(commands[-1]["id"])
    return {"status": "ok", "commands": commands, "last_id": last_id}
