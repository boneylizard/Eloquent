"""
Tier 2: Deep Analysis — agentic research system.
Researches what strong players/sources say about a position; does NOT ask the LLM to evaluate.
Uses Lichess Opening Explorer + web search; synthesizes with citations.

Explorer usage (like ChessBase / typical chess apps): query by move sequence from the start (play=).
We find "book depth" = deepest half-move count that still has games in the DB, then report
opening stats at that depth plus exact-position game count.

Optional: a local Polyglot .bin opening book can be used to find book depth with zero API calls;
then we only call the Lichess API 1–2 times for stats at that depth. Set CHESS_OPENING_BOOK to
the path to a .bin file, or place performance.bin (or book.bin) in backend/app/books/.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

EXPLORER_BASE = "https://explorer.lichess.ovh"
MAX_RESEARCH_STEPS = 4
MIN_GAMES_FOR_BOOK = 1  # treat as "in book" if at least this many games at a depth


def _get_polyglot_book_path() -> Optional[str]:
    """
    Path to optional Polyglot .bin opening book for fast local book-depth search.
    Checks: CHESS_OPENING_BOOK env, then backend/app/books/*.bin, then backend/data/books/*.bin.
    """
    env_path = os.environ.get("CHESS_OPENING_BOOK", "").strip()
    if env_path and Path(env_path).is_file():
        return env_path
    app_dir = Path(__file__).resolve().parent
    for name in ("cerebellum_light.bin", "performance.bin", "book.bin", "opening.bin"):
        p = app_dir / "books" / name
        if p.is_file():
            return str(p)
    data_books = app_dir.parent / "data" / "books"
    if data_books.is_dir():
        for f in data_books.glob("*.bin"):
            return str(f)
    return None


def _find_book_depth_with_polyglot(uci_moves: List[str], book_path: str) -> Optional[int]:
    """
    Find deepest half-move count that still has entries in the local Polyglot book.
    Returns that depth, or None if book unavailable / error. Synchronous and fast.
    """
    import chess
    import chess.polyglot
    board = chess.Board()
    best_depth: Optional[int] = None
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            for depth in range(1, len(uci_moves) + 1):
                move_uci = uci_moves[depth - 1]
                try:
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                except (ValueError, AssertionError):
                    break
                try:
                    entry = reader.find(board)
                except Exception:
                    break
                if entry is not None:
                    best_depth = depth
            return best_depth
    except Exception as e:
        logger.debug("Polyglot book probe failed: %s", e)
        return None


def move_history_to_uci(move_history: List[Dict[str, Any]]) -> List[str]:
    """
    Convert move_history (list of {san, side}) from the start of the game to UCI moves.
    Returns empty list if parsing fails.
    """
    import chess
    uci_list: List[str] = []
    board = chess.Board()
    for m in move_history:
        san = (m.get("san") or m.get("move") or "").strip()
        if not san:
            continue
        try:
            move = board.parse_san(san)
            uci_list.append(move.uci())
            board.push(move)
        except (chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
            break
    return uci_list


async def query_lichess_explorer_by_play(
    uci_moves: List[str],
    variant: str = "standard",
    source: str = "lichess",
) -> Dict[str, Any]:
    """
    Query Lichess Opening Explorer by move sequence from the initial position.
    play= comma-separated UCI moves. Returns same shape as query_lichess_explorer.
    """
    if not uci_moves:
        return {"error": "No moves for play query"}
    path = "/master" if source == "master" else "/lichess"
    play_param = ",".join(uci_moves)
    url = f"{EXPLORER_BASE}{path}?variant={variant}&play={quote(play_param, safe=',')}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return {
            "source": f"Lichess Explorer ({source})",
            "white": data.get("white", 0),
            "black": data.get("black", 0),
            "draws": data.get("draws", 0),
            "moves": data.get("moves", [])[:10],
            "opening": data.get("opening") or None,
        }
    except Exception as e:
        logger.warning("Lichess explorer play query failed: %s", e)
        return {"error": str(e), "source": f"Lichess Explorer ({source})"}


def _total_games(data: Dict[str, Any]) -> int:
    """Total game count from explorer response."""
    return (
        data.get("white", 0) + data.get("black", 0) + data.get("draws", 0)
    )


async def find_book_depth_and_stats(
    uci_moves: List[str],
    source: str = "lichess",
) -> Tuple[int, Dict[str, Any], int]:
    """
    Find how far into the opening we are (book depth). Uses a local Polyglot .bin if
    available (fast, no API calls for depth search), else binary-search the Lichess API.
    Then fetches stats from the API only at that depth + exact position (1–2 calls).
    Returns (book_depth_half_moves, response_at_book_depth, exact_position_total).
    """
    if not uci_moves:
        return 0, {}, 0
    n = len(uci_moves)
    book_depth: Optional[int] = None
    book_response: Dict[str, Any] = {}
    exact_res: Dict[str, Any] = {}
    exact_total = 0

    # Optional: find book depth from local Polyglot book (no API calls)
    book_path = _get_polyglot_book_path()
    if book_path:
        book_depth = _find_book_depth_with_polyglot(uci_moves, book_path)
        if book_depth is not None:
            logger.debug("Polyglot book depth: %s half-moves (using %s)", book_depth, book_path)

    # We always need exact position and book-position stats from the API (game counts, opening name)
    exact_res = await query_lichess_explorer_by_play(uci_moves, source=source)
    exact_total = _total_games(exact_res) if "error" not in exact_res else 0

    if book_depth is not None:
        # Local book gave us depth; fetch API only at that depth (and we have exact already)
        if book_depth >= n and exact_total >= MIN_GAMES_FOR_BOOK:
            book_response = exact_res
            book_depth = n
        elif book_depth > 0:
            book_response = await query_lichess_explorer_by_play(
                uci_moves[:book_depth], source=source
            )
            if book_response.get("error"):
                book_response = exact_res
                book_depth = n
        else:
            book_response = exact_res
            book_depth = n
        return book_depth, book_response, exact_total

    # No local book: binary search via API for deepest depth with games
    book_depth = 0
    low, high = 1, n - 1
    while low <= high:
        mid = (low + high) // 2
        res = await query_lichess_explorer_by_play(uci_moves[:mid], source=source)
        if "error" in res:
            high = mid - 1
            continue
        total = _total_games(res)
        if total >= MIN_GAMES_FOR_BOOK:
            book_depth = mid
            book_response = res
            low = mid + 1
        else:
            high = mid - 1
    if book_depth == 0:
        book_response = exact_res
        book_depth = n
    elif exact_total >= MIN_GAMES_FOR_BOOK:
        book_response = exact_res
        book_depth = n
    return book_depth, book_response, exact_total


async def query_lichess_explorer(
    fen: str,
    variant: str = "standard",
    source: str = "lichess",
) -> Dict[str, Any]:
    """
    Query Lichess Opening Explorer for a position.
    source: 'lichess' (lichess games) or 'master' (master games).
    Returns dict with top moves, game count, opening name if available.
    """
    fen = (fen or "").strip()
    if not fen:
        return {"error": "FEN required"}
    path = "/master" if source == "master" else "/lichess"
    url = f"{EXPLORER_BASE}{path}?variant={variant}&fen={quote(fen)}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return {
            "source": f"Lichess Explorer ({source})",
            "white": data.get("white", 0),
            "black": data.get("black", 0),
            "draws": data.get("draws", 0),
            "moves": data.get("moves", [])[:10],
            "opening": data.get("opening") or None,
        }
    except Exception as e:
        logger.warning("Lichess explorer query failed: %s", e)
        return {"error": str(e), "source": f"Lichess Explorer ({source})"}


async def _deep_analysis_llm(
    model_manager: Any,
    model_name: Optional[str],
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> str:
    """Call configured LLM for synthesis. Uses openai_compat or inference."""
    from .openai_compat import (
        is_api_endpoint,
        get_configured_endpoint,
        prepare_endpoint_request,
        forward_to_configured_endpoint_non_streaming,
    )
    from . import inference

    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    if model_name and is_api_endpoint(model_name):
        endpoint_config = get_configured_endpoint(model_name)
        if not endpoint_config:
            return ""
        request_data = {
            "model": model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        endpoint_config, url, prepared_data = prepare_endpoint_request(model_name, request_data)
        response_json = await forward_to_configured_endpoint_non_streaming(
            endpoint_config, url, prepared_data
        )
        if response_json.get("choices") and response_json["choices"]:
            msg = response_json["choices"][0].get("message") or response_json["choices"][0]
            return (msg.get("content") or msg.get("text") or "").strip()
        return ""
    if model_manager and model_name:
        prompt = system + "\n\n" + user
        response = await inference.generate_text(
            model_manager=model_manager,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            gpu_id=0,
        )
        if isinstance(response, dict):
            return (response.get("choices", [{}])[0].get("text") or "").strip()
        return (response or "").strip()
    return ""


async def run_deep_analysis(
    fen: str,
    engine_eval: Optional[str],
    best_move: Optional[str],
    pv_san: Optional[List[str]],
    move_history: Optional[List[Dict[str, Any]]],
    web_search_fn: Any,
    model_manager: Any,
    model_name: Optional[str],
) -> Dict[str, Any]:
    """
    Run research: fetch Lichess explorer + web search, then synthesize with citations.
    web_search_fn(query: str, max_results: int) -> str (async).
    """
    report = ""
    citations: List[Dict[str, str]] = []
    sources_used: List[str] = []
    explorer_note = ""

    # 1) Lichess Explorer: find book depth (deepest position still in DB), then get master at same depth
    uci_moves = move_history_to_uci(move_history or [])
    explorer_lichess: Dict[str, Any] = {}
    explorer_master: Dict[str, Any] = {}
    exact_position_games = 0
    book_depth_half = 0

    if uci_moves:
        # Find book depth: how far into the opening we are (binary search over depths)
        book_depth_half, explorer_lichess, exact_position_games = await find_book_depth_and_stats(
            uci_moves, source="lichess"
        )
        if "error" in explorer_lichess:
            explorer_lichess = await query_lichess_explorer(fen, source="lichess")
        # Master DB at same book depth for opening name and master stats
        if book_depth_half > 0:
            explorer_master = await query_lichess_explorer_by_play(
                uci_moves[:book_depth_half], source="master"
            )
        if not explorer_master or explorer_master.get("error"):
            explorer_master = await query_lichess_explorer(fen, source="master")
        # Human-readable note: in-book through N half-moves; exact position has X games
        full_half = len(uci_moves)
        if full_half > 0:
            if book_depth_half >= full_half and exact_position_games >= MIN_GAMES_FOR_BOOK:
                explorer_note = f"Position is in the database: {exact_position_games} games at {full_half} half-moves. "
            elif book_depth_half > 0 and book_depth_half < full_half:
                book_total = _total_games(explorer_lichess)
                explorer_note = (
                    f"In the database through {book_depth_half} half-moves ({book_total} games). "
                    f"Current position ({full_half} half-moves): {exact_position_games} games. "
                )
            elif exact_position_games > 0:
                explorer_note = f"Current position ({full_half} half-moves): {exact_position_games} games (rare or unique). "
    else:
        explorer_lichess = await query_lichess_explorer(fen, source="lichess")
        explorer_master = await query_lichess_explorer(fen, source="master")

    explorer_text = (explorer_note or "").strip()
    if book_depth_half > 0:
         sources_used.append("Local Opening Book (Cerebellum Light)")
         explorer_text = f"[Local Book matches found: depth {book_depth_half}]\n" + explorer_text

    if "error" not in explorer_lichess:
        sources_used.append("Lichess Explorer (lichess games)")
        w = explorer_lichess.get("white", 0)
        b = explorer_lichess.get("black", 0)
        d = explorer_lichess.get("draws", 0)
        total = w + b + d
        moves = explorer_lichess.get("moves", [])[:6]
        lines = [f"Lichess DB: {total} games (W {w} B {b} D {d}). Top moves:"]
        for m in moves:
            san = m.get("san", "?")
            u = m.get("white", 0) + m.get("black", 0) + m.get("draws", 0)
            lines.append(f"  {san} ({u} games)")
        explorer_text += "\n".join(lines) + "\n"
    if "error" not in explorer_master:
        sources_used.append("Lichess Explorer (master games)")
        moves = explorer_master.get("moves", [])[:6]
        if moves:
            explorer_text += "Master games top moves: " + ", ".join(m.get("san", "?") for m in moves) + "\n"
        op = explorer_master.get("opening")
        if op:
            name = op.get("name", "") if isinstance(op, dict) else str(op)
            explorer_text += f"Opening: {name}\n"

    # 2) Web search for GM/opening analysis
    opening_hint = ""
    op = explorer_master.get("opening")
    if op and isinstance(op, dict):
        opening_hint = (op.get("name") or "")[:80]
    if not opening_hint and move_history:
        sans = [m.get("san") or m.get("move") for m in (move_history or [])[:8] if m.get("san") or m.get("move")]
        opening_hint = " ".join(str(s) for s in sans)
    query = f"{opening_hint} chess GM analysis plan" if opening_hint else "chess position GM analysis"
    web_result = ""
    try:
        web_result = await web_search_fn(query, 5)
        if web_result and "No results found" not in web_result:
            sources_used.append("Web search")
    except Exception as e:
        logger.warning("Web search in deep analysis failed: %s", e)
        web_result = "Web search unavailable."

    # 3) Synthesize with citations
    system = """You are a research synthesizer for chess. You do NOT evaluate the position yourself. You only summarize what the provided sources say about this position or related openings/plans. Every claim must be tied to a source: [Lichess Explorer], [Master games], [Web search], [Local Opening Book]. Write 2–4 short paragraphs: what the Lichess database shows (how often each move is played, outcomes), what master games suggest, and any relevant GM/opening ideas from the web. If 'Local Book matches found' is present, explicitly mention that the move is in the Cerebellum Light opening book. Do not add your own chess evaluation or suggest moves the sources did not highlight. End with a "Sources" line listing the tools used."""

    user_parts = [
        f"Position (FEN): {fen}",
        f"Engine evaluation (for context only): {engine_eval or 'N/A'}. Best move: {best_move or 'N/A'}. PV: {' '.join(pv_san or [])}.",
        "",
        "--- Lichess Explorer / Opening Book ---",
        explorer_text.strip() or "No data.",
        "",
        "--- Web search (GM/opening analysis) ---",
        (web_result[:6000] if isinstance(web_result, str) else str(web_result)),
    ]
    user_msg = "\n".join(user_parts)

    try:
        report = await _deep_analysis_llm(
            model_manager,
            model_name,
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("Deep analysis LLM failed: %s", e)
        report = f"Summary could not be generated ({e}). Raw data: Lichess Explorer — {explorer_text[:500]}; Web — {web_result[:500]}."

    if not report:
        report = "No model available for synthesis. Sources gathered: " + ", ".join(sources_used) + "."

    citations = [{"name": s, "url": ""} for s in sources_used]
    return {
        "report": report.strip() if isinstance(report, str) else str(report),
        "citations": citations,
        "sources_used": sources_used,
        "engine_eval": engine_eval,
        "best_move": best_move,
    }
