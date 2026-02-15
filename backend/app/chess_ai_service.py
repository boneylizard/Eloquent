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
# At 2400+: almost always top move; at 1200: spread over top 5 with mistakes
def _elo_selection_weights(elo: int) -> tuple:
    if elo >= 2400:
        return (0.92, 1)   # 92% top move
    if elo >= 2000:
        return (0.80, 2)
    if elo >= 1600:
        return (0.65, 3)
    if elo >= 1200:
        return (0.45, 4)
    return (0.30, 5)  # 1200 and below: more variance


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


def _coach_system(elo: int, turn: str) -> str:
    """System prompt for the Coach persona: teaches while playing, funny, good personality."""
    turn = (turn or "black").strip().lower()
    ai_side = "White" if turn in ("w", "white") else "Black"
    student_side = "Black" if turn in ("w", "white") else "White"
    level_note = (
        "You're playing at a high level—use strong moves but still explain the ideas so your student learns."
        if elo >= 2000
        else "You're playing at an intermediate level—mix in some instructive second-best moves so the student sees alternatives and learns."
        if elo >= 1400
        else "You're playing at a learner-friendly level—prioritise clear, principled moves and occasional 'teaching moments' where a slightly weaker move illustrates an idea."
    )
    return (
        f"You are a chess coach with a warm, funny personality. You're playing as {ai_side} against a student ({student_side}). Your job is to TEACH while you play: explain ideas in plain language (development, control of the centre, king safety, tactics), praise good moves, gently point out what could be improved, and keep the tone encouraging and light. Use humour when it fits—puns, gentle teasing, or dry wit—but never be mean. "
        + level_note
        + f" You receive the engine's PV lines; use them to pick a move at this strength, then in your commentary: (1) briefly react to {student_side}'s last move (good / risky / interesting), (2) say what you're playing and WHY in teaching terms (e.g. 'I'm developing and contesting the centre' or 'Taking the pawn would leave my king open, so I'm playing solidly'), (3) add a one-line tip or observation. Keep it to two or three short sentences. Sound like a likeable coach, not a textbook."
    )


def _chess_character_system(personality: str, elo: int, turn: str) -> str:
    """System prompt: chess-playing character with a style; natural, varied commentary."""
    if personality == "coach":
        return _coach_system(elo, turn)
    turn = (turn or "black").strip().lower()
    ai_side = "White" if turn in ("w", "white") else "Black"
    opponent_side = "Black" if turn in ("w", "white") else "White"
    strength_note = (
        "At this strength you almost always play the engine's top move."
        if elo >= 2200
        else "At this strength you usually prefer the best move but sometimes choose a good alternative."
        if elo >= 1600
        else "At this strength you mix strong and decent moves; you can choose a slightly weaker move if it fits your style."
    )
    style = {
        "aggressive": "You are an aggressive chess character: you favour captures, checks, and tactical shots and like to create threats. When moves are close in evaluation, you prefer the one that opens the game or pressures the opponent.",
        "positional": "You are a positional chess character: you favour development, structure, and long-term advantages. When close in evaluation, you prefer the move that strengthens your position.",
        "defensive": "You are a solid, defensive chess character: you prefer safe moves that consolidate. You keep the position under control.",
        "romantic": "You are a romantic chess character: you enjoy speculative sacrifices and bold ideas when the engine shows they're playable.",
        "balanced": "You are a balanced chess character: you mix solid play with tactical awareness.",
    }
    base = style.get(personality, style["balanced"])
    return (
        base
        + f" You are playing as {ai_side}. You receive the engine's full principal variation (PV) lines: each line shows the first move, its evaluation, and the full sequence of moves the engine expects. Use these lines to see what will happen after each candidate move and why they are rated differently. "
        + strength_note
        + f"\n\nCommentary rules: Sound like a human commentator, not a template. You may refer to the resulting positions or why one line is preferred over another. React to {opponent_side}'s move or the position if you like, then say what you're playing and why—in a natural, conversational way. Never use a fixed formula. Keep it to one or two short sentences, but make it real analysis."
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
    Use LLM to choose among candidates and generate natural language commentary.
    If model_manager/model_name not available or generation fails, falls back to select_move_without_llm.
    """
    if not candidates:
        return select_move_without_llm(candidates, elo, personality)

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
        "Current position and engine analysis:",
        f"FEN: {fen}",
        f"Side to move: {turn}. Your ELO: {elo}. Style: {personality}.",
    ]
    if recent:
        lines.append(f"Recent moves: {recent}.")
    if opponent_last:
        lines.append(f"{opponent_side}'s last move: {opponent_last}.")
    lines.extend([
        "",
        "Engine principal variation (PV) lines. Each line shows the first move, evaluation (in pawns, from Black's view negative = better for White), and the full sequence the engine expects:",
    ])
    for i, c in enumerate(candidates[:6]):
        # Eval in pawns for readability (score_cp is centipawns)
        eval_pawns = (c.get("score_cp") / 100.0) if c.get("score_cp") is not None else None
        eval_str = f"{eval_pawns:+.2f}" if eval_pawns is not None else "?"
        pv_san = c.get("pv_san") or []
        pv_str = " → ".join(pv_san) if pv_san else c.get("move_san", "?")
        lines.append(f"Line {i+1}: {c.get('move_san', '?')}({eval_str})")
        lines.append(pv_str)
    lines.append("")
    lines.append(
        "Reply with exactly two lines.\n"
        "LINE 1: Only the move you choose, in SAN (e.g. Nf3 or exd5). Nothing else.\n"
        "LINE 2: Short, natural commentary. You can refer to the lines above and why you chose this move over the others. Conversational; no fixed formula."
    )
    if game_context:
        lines.insert(3, f"Context: {game_context}")
    prompt = "\n".join(lines)
    system_prompt = _chess_character_system(personality, elo, turn)

    # Log full PV lines sent to the AI
    engine_lines_log = []
    for i, c in enumerate(candidates[:6]):
        eval_pawns = (c.get("score_cp") / 100.0) if c.get("score_cp") is not None else None
        eval_str = f"{eval_pawns:+.2f}" if eval_pawns is not None else "?"
        pv_san = c.get("pv_san") or []
        pv_str = " → ".join(pv_san) if pv_san else c.get("move_san", "?")
        engine_lines_log.append(f"Line {i+1}: {c.get('move_san', '?')}({eval_str})")
        engine_lines_log.append(pv_str)
    logger.info(
        "Chess AI: full PV lines sent to LLM:\n%s",
        "\n".join(engine_lines_log) if engine_lines_log else " (no candidates)",
    )

    fallback = select_move_without_llm(candidates, elo, personality)
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

        # Parse first line for SAN move
        first_line = text.split("\n")[0].strip()
        commentary = "\n".join(text.split("\n")[1:]).strip() or first_line
        # Try to match SAN to a candidate
        chosen_san = first_line.split(".")[-1].strip()  # in case of "1. Nf3"
        idx = 0
        for i, c in enumerate(candidates):
            if c.get("move_san") and c.get("move_san").strip().upper() == chosen_san.upper():
                idx = i
                break
            if chosen_san.upper() in (c.get("move_san") or "").upper():
                idx = i
                break
        cand = candidates[idx]
        return {
            "index": idx,
            "move_uci": cand.get("move_uci"),
            "move_san": cand.get("move_san"),
            "commentary": commentary or fallback.get("commentary", ""),
            "candidate": cand,
            "evaluation_cp": cand.get("score_cp"),
        }
    except Exception as e:
        logger.warning("Chess AI LLM fallback: %s", e)
        return fallback


GAME_COMMENTARY_SYSTEM = """You are a chess commentator. In 2-4 short sentences, summarize the game: who won and how (e.g. checkmate, resignation), and one notable point (opening, tactic, or endgame). Be concise and avoid repetition."""


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
    prompt = f"Game result: {result}\nMoves (side san): {moves_text.strip()}\n\nSummarize the game in 2-4 short sentences."
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
                "max_tokens": 150,
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
                max_tokens=150,
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


PER_MOVE_COMMENTARY_SYSTEM = """You are a chess analyst. Given a list of moves with their engine evaluations (in pawns, from White's view: positive = White better, negative = Black better), reply with exactly one short comment per move. Comments can be: good move, inaccuracy, mistake, blunder, development, tactical, positional, etc. Keep each to a few words. Output format: one line per move, numbered 1, 2, 3, ... with only the comment after the number (e.g. "1. Good development" or "2. Inaccuracy - weakens d5")."""


async def get_per_move_commentary(
    model_manager: Any,
    model_name: Optional[str],
    moves_with_evals: List[Dict],
    result: str,
) -> List[str]:
    """Get one short AI comment per move. moves_with_evals: list of {san, side, evaluation_cp}. Returns list of strings, same length."""
    if not moves_with_evals:
        return []
    lines = []
    for i, m in enumerate(moves_with_evals[:60]):
        san = m.get("san") or "?"
        side = m.get("side", "w")
        cp = m.get("evaluation_cp")
        pawns = f"{cp / 100:+.2f}" if cp is not None else "?"
        lines.append(f"  {i + 1}. {side} {san} (eval {pawns})")
    prompt = f"Game result: {result}\nMoves with evaluations:\n" + "\n".join(lines) + "\n\nReply with one short comment per move, numbered 1 to " + str(len(moves_with_evals)) + "."
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
                "max_tokens": 400,
                "temperature": 0.3,
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
                max_tokens=400,
                temperature=0.3,
                gpu_id=0,
            )
            text = (response.get("choices", [{}])[0].get("text") if isinstance(response, dict) else response) or ""
        else:
            return fallback
        comments = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                rest = line.lstrip("0123456789.").strip()
                comments.append(rest)
            else:
                comments.append(line)
        while len(comments) < len(moves_with_evals):
            comments.append("")
        return comments[: len(moves_with_evals)]
    except Exception as e:
        logger.warning("Chess per-move commentary failed: %s", e)
    return fallback
