"""
AI-driven move selection and commentary for Eloquent Chess.
Uses the engine's top N candidates and LLM to pick a move by personality and explain it.
Supports both local models (inference.generate_text) and OpenAI-compatible API endpoints.
"""
import logging
import random
from typing import Any, Dict, List, Optional

from . import inference
from .openai_compat import (
    is_api_endpoint,
    get_configured_endpoint,
    prepare_endpoint_request,
    forward_to_configured_endpoint_non_streaming,
)

logger = logging.getLogger(__name__)

PERSONALITIES = ["balanced", "aggressive", "positional", "defensive", "romantic", "coach"]

# ELO-based selection weights: (top_move_weight, spread over top N)
# Higher spread at lower ELO so we pick from more candidates (including weaker moves)
def _elo_selection_weights(elo: int) -> tuple:
    if elo >= 2400:
        return (0.92, 1)   # 92% top move
    if elo >= 2000:
        return (0.80, 3)
    if elo >= 1600:
        return (0.65, 5)
    if elo >= 1200:
        return (0.45, 7)
    return (0.30, 10)  # 1200 and below: spread over up to 10 candidates


def _pick_candidate_index_by_elo(candidates: List[Dict], elo: int) -> int:
    """Choose candidate index 0..len(candidates)-1 based on ELO (higher ELO = prefer top moves)."""
    if not candidates:
        return 0
    top_weight, spread = _elo_selection_weights(elo)
    n = min(len(candidates), max(1, spread))
    # Weighted random: index 0 gets top_weight, rest share (1 - top_weight)
    r = random.random()
    if r < top_weight:
        return 0
    remaining = 1.0 - top_weight
    for i in range(1, n):
        # Equal share of remaining for indices 1..n-1
        p = remaining / (n - 1)
        if r < top_weight + p * i:
            return min(i, len(candidates) - 1)
    return min(n - 1, len(candidates) - 1)


def select_move_without_llm(
    candidates: List[Dict],
    elo: int,
    personality: str,
) -> Dict[str, Any]:
    """
    Select a move from engine candidates using ELO and simple personality rules.
    Use this when LLM is unavailable or for fast path. Returns chosen candidate index and reason.
    """
    if not candidates:
        return {"index": 0, "move_uci": None, "commentary": "No moves available.", "candidate": None}

    # First apply ELO: pick index with weighted random
    idx = _pick_candidate_index_by_elo(candidates, elo)
    cand = candidates[idx]

    # Personality can nudge: aggressive prefers tactical, positional prefers positional/development
    if personality == "aggressive" and len(candidates) > 1:
        tactical_indices = [i for i, c in enumerate(candidates) if c.get("tactical")]
        if tactical_indices and random.random() < 0.5:
            idx = random.choice(tactical_indices[:3])
            cand = candidates[idx]
    elif personality == "positional" and len(candidates) > 1:
        pos_indices = [i for i, c in enumerate(candidates) if c.get("positional")]
        if pos_indices and random.random() < 0.4:
            idx = min(pos_indices[0], len(candidates) - 1)
            cand = candidates[idx]
    elif personality == "defensive" and len(candidates) > 1:
        safe = [i for i, c in enumerate(candidates) if not c.get("is_capture") and not c.get("is_check")]
        if safe and random.random() < 0.4:
            idx = safe[0]
            cand = candidates[idx]
    elif personality == "romantic" and len(candidates) > 1:
        sacrificial = [i for i, c in enumerate(candidates) if "speculative_or_sacrifice" in c.get("themes", [])]
        if sacrificial and random.random() < 0.35:
            idx = sacrificial[0]
            cand = candidates[idx]
    elif personality == "coach":
        idx = _pick_candidate_index_by_elo(candidates, elo)
        cand = candidates[idx]

    short_reason = f"Chose candidate {idx + 1} (ELO ~{elo}, {personality})."
    return {
        "index": idx,
        "move_uci": cand.get("move_uci"),
        "move_san": cand.get("move_san"),
        "commentary": short_reason,
        "candidate": cand,
        "evaluation_cp": cand.get("score_cp"),
    }


# --- TIER 1: Anti-hallucination principle. LLMs cannot "see" or evaluate chess; only translate engine output. ---

def _coach_system(elo: int, turn: str) -> str:
    """System prompt for Coach: translate engine output only; no own chess analysis."""
    turn = (turn or "black").strip().lower()
    ai_side = "White" if turn in ("w", "white") else "Black"
    student_side = "Black" if turn in ("w", "white") else "White"
    return (
        f"You are a chess coach playing as {ai_side} against a student ({student_side}). Your ONLY job is to turn the engine's data into friendly, plain-English commentary.\n\n"
        "CRITICAL: You do NOT evaluate the position yourself. You cannot 'see' the board—FEN is just text to you. You MUST only describe what the engine report says: the evaluation (who is better and by how much), the best move the engine suggests, and the main line (principal variation) the engine gives. Explain that in simple, teaching language. Do NOT add your own chess judgments, suggestions, or evaluations. Do NOT invent tactics or plans the engine did not show. If the engine says 'best move X, eval +0.5', you say something like 'The engine likes this move and thinks White is a bit better; the main idea is …' using only the PV the engine provided.\n\n"
        "Keep it to two or three short sentences. Warm, encouraging tone; you can react briefly to the opponent's last move in terms of what the engine evaluation implies (e.g. 'that move left you slightly worse according to the engine'). Sound like a coach, not a textbook."
    )


def _chess_character_system(personality: str, elo: int, turn: str) -> str:
    """System prompt: translate engine output into character voice; no own analysis."""
    if personality == "coach":
        return _coach_system(elo, turn)
    turn = (turn or "black").strip().lower()
    ai_side = "White" if turn in ("w", "white") else "Black"
    opponent_side = "Black" if turn in ("w", "white") else "White"
    style = {
        "aggressive": "You sound aggressive and like tactics.",
        "positional": "You sound positional and strategic.",
        "defensive": "You sound solid and cautious.",
        "romantic": "You sound bold and speculative when the engine allows it.",
        "balanced": "You sound balanced.",
    }
    voice = style.get(personality, style["balanced"])
    return (
        f"You are a chess commentator. You are playing as {ai_side} (opponent: {opponent_side}). Your ONLY job is to turn the engine's data into short, natural commentary in your style ({voice}).\n\n"
        "CRITICAL: You do NOT evaluate the position yourself. You cannot 'see' the board. You MUST only explain what the engine report says: the evaluation (who is better, by how much), the best move, and the principal variation (PV) the engine gives. Put that in plain English in 1–2 sentences. Do NOT add your own chess analysis, evaluations, or suggestions. Do NOT invent moves or plans not in the engine output. Only translate: 'The engine says …' into conversational language.\n\n"
        "Example: if the engine says move Nf3, eval +0.3, PV Nf3 d5 d4 …, you might say 'Playing Nf3—the engine gives White a small edge and suggests this main line.' Do not claim to see threats or ideas the engine did not show."
    )


def _format_recent_moves(move_history: List[Dict]) -> str:
    """Format last few moves for the prompt."""
    if not move_history:
        return ""
    parts = []
    for m in move_history[-10:]:
        side = (m.get("side") or "").strip().lower()
        san = (m.get("san") or "").strip()
        if side in ("w", "white") and san:
            parts.append(f"White {san}")
        elif side in ("b", "black") and san:
            parts.append(f"Black {san}")
        elif san:
            parts.append(san)
    return "; ".join(parts) if parts else ""


async def select_move_with_llm(
    model_manager: Any,
    model_name: Optional[str],
    candidates: List[Dict],
    elo: int,
    personality: str,
    fen: str,
    turn: str,
    game_context: Optional[str] = None,
    move_history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Pick move by ELO (same distribution as non-LLM path), then use LLM only for commentary.
    This keeps play strength at the set rating instead of letting the LLM always pick the best move.
    """
    if not candidates:
        return select_move_without_llm(candidates, elo, personality)

    # Choose move by ELO so strength is enforced regardless of LLM
    chosen_idx = _pick_candidate_index_by_elo(candidates, elo)
    chosen = candidates[chosen_idx]
    chosen_san = (chosen.get("move_san") or "").strip()
    eval_pawns = (chosen.get("score_cp") / 100.0) if chosen.get("score_cp") is not None else None
    eval_str = f"{eval_pawns:+.2f}" if eval_pawns is not None else "?"

    recent = _format_recent_moves(move_history or [])
    opponent_last = ""
    opponent_side = "White"
    if move_history:
        last = move_history[-1]
        side = (last.get("side") or "").strip().lower()
        san = (last.get("san") or "").strip()
        if san:
            opponent_side = "White" if side in ("w", "white") else "Black"
            opponent_last = san

    lines = [
        "Current position:",
        f"FEN: {fen}",
        f"Side to move: {turn}. Your ELO: {elo}. Style: {personality}.",
    ]
    if recent:
        lines.append(f"Recent moves: {recent}.")
    if opponent_last:
        lines.append(f"{opponent_side}'s last move: {opponent_last}.")
    lines.append("")
    lines.append("Engine analysis (evaluations in pawns; positive = better for side to move). Top alternatives:")
    for i, c in enumerate(candidates[:6]):
        ep = (c.get("score_cp") / 100.0) if c.get("score_cp") is not None else None
        es = f"{ep:+.2f}" if ep is not None else "?"
        pv_san = c.get("pv_san") or []
        pv_str = " → ".join(pv_san) if pv_san else (c.get("move_san") or "?")
        mark = "  ← YOUR MOVE" if i == chosen_idx else ""
        lines.append(f"  {i+1}. {c.get('move_san', '?')} ({es}): {pv_str}{mark}")
    lines.extend([
        "",
        f"The move you are playing: {chosen_san}. Engine evaluation for this move: {eval_str} pawns (positive = better for the side to move).",
        "",
        "TASK: In 1–3 short sentences, explain in plain English what the engine data above says: what the evaluation means (who is better and by how much), and what the principal variation (PV) shows will likely happen. Do NOT add your own chess analysis or evaluations. Only translate the engine's numbers and lines into readable, in-character commentary.",
    ])
    if game_context:
        lines.insert(3, f"Context: {game_context}")
    prompt = "\n".join(lines)
    system_prompt = _chess_character_system(personality, elo, turn)

    logger.info(
        "Chess AI (in-game): playing move %s (candidate %d of %d), requesting LLM commentary for this move only.",
        chosen_san,
        chosen_idx + 1,
        len(candidates),
    )

    fallback = {
        "index": chosen_idx,
        "move_uci": chosen.get("move_uci"),
        "move_san": chosen_san,
        "commentary": f"Played {chosen_san}.",
        "candidate": chosen,
        "evaluation_cp": chosen.get("score_cp"),
    }
    if not model_name:
        return fallback

    try:
        if is_api_endpoint(model_name):
            endpoint_config = get_configured_endpoint(model_name)
            if not endpoint_config:
                logger.warning("Chess AI: endpoint %s not configured or disabled", model_name)
                return fallback
            request_data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 260,
                "temperature": 0.5,
            }
            endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
            logger.info("Chess AI: calling API endpoint model=%s", model_name)
            response_json = await forward_to_configured_endpoint_non_streaming(
                endpoint_config, url, prepared_data
            )
            text = ""
            if response_json.get("choices") and len(response_json["choices"]) > 0:
                choice = response_json["choices"][0]
                msg = choice.get("message") or choice
                text = (msg.get("content") or msg.get("text") or "").strip()
            logger.info("Chess AI: API response excerpt: %s", (text or "(empty)")[:350])
        else:
            if not model_manager:
                return fallback
            logger.info("Chess AI: calling local model=%s", model_name)
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=system_prompt + "\n\n" + prompt,
                max_tokens=260,
                temperature=0.5,
                gpu_id=0,
            )
            if isinstance(response, dict):
                text = (response.get("choices", [{}])[0].get("text") or "").strip()
            else:
                text = (response or "").strip()
            logger.info("Chess AI: local model response excerpt: %s", (text or "(empty)")[:350])

        if not text:
            return fallback

        # Full response is commentary (move was already chosen by ELO)
        commentary = text.strip() or fallback.get("commentary", "")
        return {
            "index": chosen_idx,
            "move_uci": chosen.get("move_uci"),
            "move_san": chosen_san,
            "commentary": commentary,
            "candidate": chosen,
            "evaluation_cp": chosen.get("score_cp"),
        }
    except Exception as e:
        logger.warning("Chess AI LLM fallback: %s", e)
        return fallback


GAME_COMMENTARY_SYSTEM = """You are a chess writer summarizing a game that has ALREADY been played. You are an observer, not a player. Write in a clear, instructive style (like Chernev).

CRITICAL: You do NOT evaluate positions yourself. You only have the move list and the result. Base your summary ONLY on: (1) the result and how the game ended (checkmate, resignation, draw), (2) the sequence of moves—you may describe what happened (e.g. "White opened the centre", "Black resigned in a lost endgame") but do NOT invent evaluations like "White was winning" unless the result makes that obvious. Stick to describing the moves and the outcome. In 3–5 sentences: how the game was decided, one or two concrete moments from the move list, and a brief takeaway. Do not add your own chess judgment about who was better at move 15, etc."""


async def get_game_commentary(
    model_manager: Any,
    model_name: Optional[str],
    move_history: List[Dict],
    result: str,
) -> str:
    """Generate brief AI commentary on a finished game. Returns empty string on failure."""
    if not move_history:
        return ""
    moves_text = " ".join(f"{m.get('side', '')} {m.get('san', '')}" for m in move_history[:80])
    prompt = f"This game has already been played. Result: {result}\nMoves: {moves_text.strip()}\n\nWrite a short Chernev-style summary (3–5 sentences): how the game was decided, one or two key moments, and an instructive takeaway."
    fallback = ""
    try:
        if model_name and is_api_endpoint(model_name):
            logger.info("Chess AI: game commentary via API model=%s", model_name)
            endpoint_config = get_configured_endpoint(model_name)
            if not endpoint_config:
                return fallback
            request_data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": GAME_COMMENTARY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 280,
                "temperature": 0.5,
            }
            endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
            response_json = await forward_to_configured_endpoint_non_streaming(
                endpoint_config, url, prepared_data
            )
            if response_json.get("choices") and len(response_json["choices"]) > 0:
                msg = response_json["choices"][0].get("message") or response_json["choices"][0]
                out = (msg.get("content") or msg.get("text") or "").strip()
                logger.info("Chess AI: game commentary response: %s", (out or "(empty)")[:200])
                return out
        elif model_manager and model_name:
            logger.info("Chess AI: game commentary via local model=%s", model_name)
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=GAME_COMMENTARY_SYSTEM + "\n\n" + prompt,
                max_tokens=280,
                temperature=0.5,
                gpu_id=0,
            )
            if isinstance(response, dict):
                text = (response.get("choices", [{}])[0].get("text") or "").strip()
            else:
                text = (response or "").strip()
            logger.info("Chess AI: game commentary (local) response: %s", (text or "(empty)")[:200])
            return text
    except Exception as e:
        logger.warning("Chess game commentary failed: %s", e)
    return fallback


PER_MOVE_COMMENTARY_SYSTEM = """You are a strong chess expert annotating a game that has ALREADY been played. You are an observer. Annotate each move in the list only; do not add moves like "resigns" or stop early.

CRITICAL:
1. **Adopt the evaluation as YOUR OWN opinion.** Do not say "The engine says White is better." Say "White has a clear advantage," "I prefer Black's position," or "This move is a mistake."
2. **Explain the chess reasons.** Use the provided engine evaluation (in pawns; positive = White better) to form your opinion, but explain it using chess concepts (control of center, open lines, king safety, rapid development). checking the eval change to decide if a move was good or bad.
3. **Be natural and instructive.** Wrtie 1-3 short, clear sentences per move in the style of a human commentator like Irving Chernev.

Output format: exactly one block per move, numbered 1., 2., 3., ... up to the last move. Each number on its own line, then your annotation. Do not skip numbers. Do not add move numbers beyond the list."""



# Max moves to annotate in one call (longer games get annotations for first N only; rest get empty string)
PER_MOVE_ANNOTATION_LIMIT = 80


async def get_per_move_commentary(
    model_manager: Any,
    model_name: Optional[str],
    moves_with_evals: List[Dict],
    result: str,
) -> List[str]:
    """Get one AI comment per move in the style of Irving Chernev. moves_with_evals: list of {san, side, evaluation_cp}. Returns list of strings, same length."""
    if not moves_with_evals:
        return []
    moves_to_annotate = moves_with_evals[:PER_MOVE_ANNOTATION_LIMIT]
    n_annotate = len(moves_to_annotate)
    lines = []
    for i, m in enumerate(moves_to_annotate):
        san = m.get("san") or "?"
        side = m.get("side", "w")
        cp = m.get("evaluation_cp")
        pawns = f"{cp / 100:+.2f}" if cp is not None else "?"
        lines.append(f"  {i + 1}. {side} {san} (eval {pawns})")
    prompt = (
        f"The game result was: {result}. This game has already been played; you are annotating it as an observer.\n\n"
        f"Moves with engine evaluations (eval in pawns; positive = White better):\n"
        + "\n".join(lines)
        + f"\n\nAnnotate exactly these {n_annotate} moves. Reply with move number 1. then your annotation (1–3 sentences in Chernev style), then 2. then your annotation, and so on up to {n_annotate}. Do not add any move (e.g. no 'resigns'). Do not skip any number from 1 to {n_annotate}."
    )
    fallback = [""] * len(moves_with_evals)
    try:
        if model_name and is_api_endpoint(model_name):
            endpoint_config = get_configured_endpoint(model_name)
            if not endpoint_config:
                return fallback
            request_data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": PER_MOVE_COMMENTARY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 6000,
                "temperature": 0.35,
            }
            endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
            response_json = await forward_to_configured_endpoint_non_streaming(
                endpoint_config, url, prepared_data
            )
            if response_json.get("choices") and response_json["choices"]:
                msg = response_json["choices"][0].get("message") or response_json["choices"][0]
                text = (msg.get("content") or msg.get("text") or "").strip()
            else:
                return fallback
        elif model_manager and model_name:
            response = await inference.generate_text(
                model_manager=model_manager,
                model_name=model_name,
                prompt=PER_MOVE_COMMENTARY_SYSTEM + "\n\n" + prompt,
                max_tokens=6000,
                temperature=0.35,
                gpu_id=0,
            )
            if isinstance(response, dict):
                choice = (response.get("choices") or [{}])[0] if response.get("choices") else {}
                text = (choice.get("content") or choice.get("text") or "").strip()
            else:
                text = (response or "").strip()
        else:
            return fallback
        if not text:
            logger.warning("Chess per-move commentary: empty model response")
            return fallback
        # Parse numbered annotations; each can be multi-line (Chernev-style 1–3 sentences)
        comments = []
        current = []
        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                if current:
                    comments.append(" ".join(current).strip())
                    current = []
                continue
            # New move number at start of line (e.g. "1.", "2.", "12.")
            if line_stripped[0].isdigit():
                if current:
                    comments.append(" ".join(current).strip())
                rest = line_stripped.lstrip("0123456789").lstrip(".) \t:").strip()
                current = [rest] if rest else []
            else:
                current.append(line_stripped)
        if current:
            comments.append(" ".join(current).strip())
        # Pad to requested length; we only annotated first n_annotate moves
        while len(comments) < n_annotate:
            comments.append("")
        comments = comments[:n_annotate]
        # Pad to full game length (moves beyond limit get no annotation)
        while len(comments) < len(moves_with_evals):
            comments.append("")
        result = comments[: len(moves_with_evals)]
        logger.info("Chess per-move commentary: got %d comments for %d moves (Chernev style)", len(result), len(moves_with_evals))
        return result
    except Exception as e:
        logger.warning("Chess per-move commentary failed: %s", e)
    return fallback
