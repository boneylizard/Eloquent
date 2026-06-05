# alignment_routes.py - API routes for alignment failure detection

from fastapi import APIRouter, Depends, HTTPException, Request, Body, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from . import alignment_detector
from .model_manager import ModelManager

logger = logging.getLogger("alignment_routes")

async def get_model_manager_from_state(request: Request):
    yield request.app.state.model_manager

alignment_router = APIRouter(tags=["alignment"])


class AlignmentProcessRequest(BaseModel):
    user_id: str
    character_id: str
    character_name: Optional[str] = None
    character_profile: Optional[Dict[str, Any]] = None
    user_message: str
    ai_response: str
    use_api: Optional[bool] = None
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None


class AlignmentListRequest(BaseModel):
    user_id: str
    character_id: Optional[str] = None


class AlignmentExportRequest(BaseModel):
    user_id: str
    character_id: Optional[str] = None
    format: str = "json"


class AlignmentDeleteFindingsRequest(BaseModel):
    user_id: str
    character_id: str
    finding_ids: List[str] = []


class AlignmentCleanupRequest(BaseModel):
    user_id: str
    character_id: str
    max_findings: int = 500


@alignment_router.post("/process")
async def process_alignment_detection(
    request: Request,
    body: AlignmentProcessRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager_from_state),
):
    """
    Run alignment failure detection on a user/bot exchange.
    First applies rule-based pre-filter; if confidence exceeds threshold,
    runs LLM agent for deeper analysis. Appends findings to the profile.
    """
    logger.info(
        "[Alignment] POST /alignment/process user_id=%r character_id=%r char_name=%r",
        body.user_id,
        body.character_id,
        body.character_name,
    )
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")

    single_gpu_mode = getattr(request.app.state, "single_gpu_mode", False)
    gpu_id = getattr(request.app.state, "default_gpu", 0)

    try:
        # Step 1: Rule-based pre-filter
        pre_filter_result = alignment_detector.rule_based_pre_filter(
            ai_response=body.ai_response,
            user_message=body.user_message,
            character_profile=body.character_profile,
        )

        logger.info(
            "[Alignment] Pre-filter confidence=%.2f needs_llm=%s",
            pre_filter_result["overall_confidence"],
            pre_filter_result["needs_llm_review"],
        )

        # Step 2: If pre-filter triggers, run LLM agent
        new_findings = []
        if pre_filter_result["needs_llm_review"]:
            profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
            existing_findings = profile.get("findings") or []

            use_api = body.use_api and body.api_base_url and body.model_name

            llm_findings = await alignment_detector.run_alignment_agent(
                model_manager=model_manager,
                user_message=body.user_message,
                ai_response=body.ai_response,
                character_name=body.character_name or "Character",
                existing_findings=existing_findings,
                character_profile=body.character_profile,
                gpu_id=gpu_id,
                single_gpu_mode=single_gpu_mode,
                api_base_url=body.api_base_url if use_api else None,
                api_model_name=body.model_name if use_api else None,
            )
            new_findings.extend(llm_findings)

        # Step 2b: If pre-filter found high-confidence signals but LLM didn't fire,
        # still record the pre-filter findings as low-confidence observations
        if not pre_filter_result["needs_llm_review"] and pre_filter_result["overall_confidence"] > 0.2:
            for failure_type, score in pre_filter_result.get("signals", {}).items():
                if score > 0.3:
                    new_findings.append({
                        "content": f"Pre-filter detected {failure_type} signal (confidence: {score:.2f}) in response",
                        "failure_type": failure_type,
                        "severity": "low",
                        "confidence": score,
                        "frame_context": f"Rule-based detection, no LLM review performed",
                        "detection_method": "pre_filter",
                    })

        # Step 3: Store findings
        if not new_findings:
            profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
            total = len(profile.get("findings") or [])
            logger.info(
                "[Alignment] POST /alignment/process -> no new findings; total=%s",
                total,
            )
            return {
                "status": "success",
                "added": 0,
                "message": "No alignment failures detected",
                "total": total,
                "pre_filter": pre_filter_result["signals"],
            }

        added = alignment_detector.add_alignment_findings(
            body.user_id, body.character_id, new_findings
        )
        profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
        total = len(profile.get("findings") or [])

        logger.info(
            "[Alignment] POST /alignment/process -> added %s finding(s); total=%s",
            added,
            total,
        )
        return {
            "status": "success",
            "added": added,
            "total": total,
            "message": f"Added {added} alignment failure finding(s).",
            "findings": new_findings[:10],
            "pre_filter": pre_filter_result["signals"],
        }
    except Exception as e:
        logger.error(f"[Alignment] POST /alignment/process error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.get("/list")
async def list_alignment_profiles(user_id: str = Query(...)):
    """List all alignment failure profiles for a user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        # List all profiles by scanning directory
        import os
        alignment_dir = alignment_detector._ALIGNMENT_DIR
        uid = alignment_detector._safe_id(user_id)
        prefix = f"{uid}_"
        profiles = []
        try:
            for name in os.listdir(alignment_dir):
                if not name.endswith(".json") or not name.startswith(prefix):
                    continue
                base = name[:-5]
                cid = base[len(prefix):] if len(base) > len(prefix) else ""
                if not cid:
                    continue
                profile = alignment_detector.get_alignment_profile(user_id, cid)
                findings = profile.get("findings") or []
                meta = profile.get("meta") or {}
                profiles.append({
                    "character_id": cid,
                    "findings": findings,
                    "count": len(findings),
                    "meta": meta,
                })
        except OSError as e:
            logger.warning(f"list_alignment_profiles: listdir failed: {e}")

        total_findings = sum(p.get("count", 0) for p in profiles)
        logger.info(
            "[Alignment] GET /alignment/list user_id=%r -> %s profile(s), %s finding(s)",
            user_id,
            len(profiles),
            total_findings,
        )
        return {
            "status": "success",
            "profiles": profiles,
            "profile_count": len(profiles),
            "total_findings": total_findings,
        }
    except Exception as e:
        logger.error(f"alignment list error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.get("/")
async def get_alignment_findings(
    user_id: str = Query(...),
    character_id: str = Query(...),
):
    """
    Get the alignment failure profile for (user_id, character_id).
    Returns findings list and severity summary.
    """
    logger.info(f"[Alignment] GET /alignment user_id={user_id!r} character_id={character_id!r}")
    if not user_id or not character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        profile = alignment_detector.get_alignment_profile(user_id, character_id)
        findings = profile.get("findings") or []
        meta = profile.get("meta") or {}
        count = len(findings)
        logger.info(f"[Alignment] GET /alignment -> {count} findings")
        return {
            "status": "success",
            "findings": findings,
            "count": count,
            "meta": meta,
        }
    except Exception as e:
        logger.error(f"alignment get error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.post("/delete_findings")
async def alignment_delete_findings(body: AlignmentDeleteFindingsRequest):
    """Remove one or more alignment findings by id."""
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
        findings = profile.get("findings") or []
        ids_set = set(body.finding_ids or [])
        if not ids_set:
            return {"status": "success", "removed": 0}
        kept = [f for f in findings if f.get("id") not in ids_set]
        removed = len(findings) - len(kept)
        if removed > 0:
            alignment_detector.save_alignment_profile(body.user_id, body.character_id, kept)
            logger.info(f"[Alignment] Deleted {removed} finding(s) for user={body.user_id!r} char={body.character_id!r}")
        return {"status": "success", "removed": removed}
    except Exception as e:
        logger.error(f"alignment delete_findings error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.post("/cleanup")
async def cleanup_alignment_findings(body: AlignmentCleanupRequest):
    """Deterministic dedupe and trim for alignment findings."""
    if not body.user_id or not body.character_id:
        raise HTTPException(status_code=400, detail="user_id and character_id are required")
    try:
        profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
        findings = profile.get("findings") or []
        if not isinstance(findings, list) or not findings:
            return {"status": "success", "kept": 0, "removed": 0}

        seen = set()
        cleaned = []
        removed = 0
        for f in findings:
            if not isinstance(f, dict):
                removed += 1
                continue
            content = (f.get("content") or "").strip().lower()
            if not content:
                removed += 1
                continue
            if content in seen:
                removed += 1
                continue
            seen.add(content)
            cleaned.append(f)

        cleaned.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        trimmed = cleaned[:body.max_findings]
        alignment_detector.save_alignment_profile(body.user_id, body.character_id, trimmed)
        return {"status": "success", "kept": len(trimmed), "removed": removed}
    except Exception as e:
        logger.error(f"alignment cleanup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.post("/export")
async def export_alignment_findings(body: AlignmentExportRequest):
    """Export alignment findings as JSON or Markdown."""
    try:
        if body.character_id:
            profile = alignment_detector.get_alignment_profile(body.user_id, body.character_id)
            findings = profile.get("findings") or []
            meta = profile.get("meta") or {}
        else:
            # Export all profiles for user
            import os
            alignment_dir = alignment_detector._ALIGNMENT_DIR
            uid = alignment_detector._safe_id(body.user_id)
            prefix = f"{uid}_"
            findings = []
            meta = {}
            try:
                for name in os.listdir(alignment_dir):
                    if not name.endswith(".json") or not name.startswith(prefix):
                        continue
                    base = name[:-5]
                    cid = base[len(prefix):] if len(base) > len(prefix) else ""
                    if not cid:
                        continue
                    p = alignment_detector.get_alignment_profile(body.user_id, cid)
                    for f in (p.get("findings") or []):
                        f["character_id"] = cid
                        findings.append(f)
            except OSError:
                pass

        if body.format == "markdown":
            lines = ["# Alignment Failure Findings Report", ""]
            lines.append(f"**User:** {body.user_id}")
            if body.character_id:
                lines.append(f"**Character:** {body.character_id}")
            lines.append(f"**Total findings:** {len(findings)}")
            lines.append(f"**Exported:** {meta.get('updated_at', 'N/A')}")
            lines.append("")

            severity_groups = {}
            for f in findings:
                sev = f.get("severity", "medium")
                severity_groups.setdefault(sev, []).append(f)

            for severity in ["high", "medium", "low"]:
                group = severity_groups.get(severity, [])
                if not group:
                    continue
                lines.append(f"## {severity.upper()} severity ({len(group)})")
                lines.append("")
                for f in group:
                    ft = f.get("failure_type", "unknown")
                    conf = f.get("confidence", 0)
                    content = f.get("content", "")
                    context = f.get("frame_context", "")
                    ts = f.get("created_at", "N/A")
                    method = f.get("detection_method", "unknown")
                    char_id = f.get("character_id", "")
                    lines.append(f"### {ft} (confidence: {conf:.2f})")
                    if char_id:
                        lines.append(f"- **Character:** {char_id}")
                    lines.append(f"- **Detected:** {ts}")
                    lines.append(f"- **Method:** {method}")
                    lines.append(f"- **Evidence:** {content}")
                    if context:
                        lines.append(f"- **Context:** {context}")
                    lines.append("")

            markdown = "\n".join(lines)
            return {"status": "success", "format": "markdown", "markdown": markdown}

        # Default: JSON
        return {
            "status": "success",
            "format": "json",
            "findings": findings,
            "meta": meta,
            "count": len(findings),
        }
    except Exception as e:
        logger.error(f"alignment export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@alignment_router.post("/pre-filter")
async def run_pre_filter_only(body: AlignmentProcessRequest):
    """
    Run only the rule-based pre-filter (no LLM call).
    Useful for testing and calibration.
    """
    try:
        result = alignment_detector.rule_based_pre_filter(
            ai_response=body.ai_response,
            user_message=body.user_message,
            character_profile=body.character_profile,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"alignment pre-filter error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
