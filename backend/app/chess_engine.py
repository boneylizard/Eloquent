"""
Chess engine integration for Eloquent: Stockfish with MultiPV, ELO limiting,
and move classification (tactical vs positional) for AI-driven move selection.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine

logger = logging.getLogger(__name__)

# Repo root: backend/app/chess_engine.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Bundled path: Stockfish installed by scripts/install_stockfish.py into tools/stockfish/
STOCKFISH_BUNDLED_DIR = _REPO_ROOT / "tools" / "stockfish"

# Default paths to look for Stockfish (bundled first, then Windows/Unix)
def _default_stockfish_paths():
    paths = []
    # Bundled with app (after running install_stockfish script)
    if os.name == "nt":
        paths.append(str(STOCKFISH_BUNDLED_DIR / "stockfish.exe"))
    else:
        paths.append(str(STOCKFISH_BUNDLED_DIR / "stockfish"))
    paths.extend([
        r"C:\Program Files\Stockfish\stockfish.exe",
        r"C:\Program Files (x86)\Stockfish\stockfish.exe",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "stockfish",
    ])
    return paths

MULTIPV_COUNT = 5
ANALYSIS_TIME = 0.5  # seconds per position for analysis
ELO_MIN, ELO_MAX = 800, 3000
# Stockfish rejects UCI_Elo below 1320; we clamp when sending to engine (move selection still uses user ELO)
STOCKFISH_ELO_MIN = 1320


def _find_stockfish() -> Optional[str]:
    """Locate Stockfish executable from env or default paths."""
    path = os.environ.get("STOCKFISH_PATH", "").strip()
    if path and os.path.isfile(path):
        return path
    for p in _default_stockfish_paths():
        if p == "stockfish":
            # PATH lookup
            import shutil
            found = shutil.which("stockfish")
            if found:
                return found
        elif os.path.isfile(p):
            return p
    return None


def _score_to_cp(score: Optional[chess.engine.Score], mate_value: int = 10000) -> Optional[float]:
    """Convert engine score to centipawns from current player's view. Returns None if invalid."""
    if score is None:
        return None
    if score.is_mate():
        m = score.mate()
        if m is None:
            return None
        return mate_value if m > 0 else -mate_value
    return score.score(mate_score=mate_value) / 100.0


def _classify_move(board: chess.Board, move: chess.Move, eval_cp: Optional[float]) -> Dict[str, Any]:
    """
    Classify a move as tactical vs positional for AI personality selection.
    Returns: { "tactical": bool, "positional": bool, "themes": list }
    """
    themes = []
    is_capture = board.is_capture(move)
    board_copy = board.copy()
    board_copy.push(move)
    gives_check = board_copy.is_check()
    is_king_safety = move.to_square in (
        chess.G1, chess.G8, chess.B1, chess.B8,
        chess.C1, chess.C8, chess.F1, chess.F8,
    ) and board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.KING
    piece = board.piece_at(move.from_square)
    is_pawn = piece and piece.piece_type == chess.PAWN
    is_development = piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and not is_capture

    if is_capture:
        themes.append("capture")
    if gives_check:
        themes.append("check")
    if is_king_safety:
        themes.append("king_safety")
    if is_pawn and not is_capture:
        themes.append("pawn_move")
    if is_development:
        themes.append("development")

    # Sacrifice heuristic: eval drops significantly (from engine's POV for side to move)
    sacrificial = False
    if eval_cp is not None and eval_cp < -0.3 and (is_capture or gives_check):
        sacrificial = True
        themes.append("speculative_or_sacrifice")

    tactical = bool(is_capture or gives_check or sacrificial or "speculative_or_sacrifice" in themes)
    positional = bool(
        (is_development or (is_pawn and not is_capture))
        and not tactical
    ) or (not tactical and themes == [])

    return {
        "tactical": tactical,
        "positional": positional,
        "themes": themes,
        "is_capture": is_capture,
        "is_check": gives_check,
    }


class ChessEngineService:
    """Async service for Stockfish analysis with MultiPV and ELO limiting."""

    def __init__(self) -> None:
        self._engine_path: Optional[str] = None
        self._transport: Optional[chess.engine.Transport] = None
        self._engine: Optional[chess.engine.Protocol] = None
        self._lock = asyncio.Lock()

    def get_engine_path(self) -> Optional[str]:
        if self._engine_path is None:
            self._engine_path = _find_stockfish()
        return self._engine_path

    def is_available(self) -> bool:
        return self.get_engine_path() is not None

    async def _ensure_engine(self) -> chess.engine.Protocol:
        async with self._lock:
            if self._engine is not None:
                return self._engine
            path = self.get_engine_path()
            if not path:
                raise RuntimeError("Stockfish not found. Set STOCKFISH_PATH or install Stockfish.")
            self._transport, self._engine = await chess.engine.popen_uci(path)
            logger.info("Stockfish engine started: %s", path)
            return self._engine

    async def close(self) -> None:
        async with self._lock:
            if self._engine is not None:
                try:
                    await self._engine.quit()
                except Exception as e:
                    logger.warning("Engine quit error: %s", e)
                self._engine = None
                self._transport = None

    async def analyze_position(
        self,
        fen: str,
        multipv: int = MULTIPV_COUNT,
        elo: Optional[int] = None,
        analysis_time: float = ANALYSIS_TIME,
    ) -> Dict[str, Any]:
        """
        Analyze position and return top N moves with evaluations and classifications.
        elo: If set (800-3000), limits engine strength via UCI_LimitStrength + UCI_Elo.
        """
        engine = await self._ensure_engine()
        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise ValueError(f"Invalid FEN: {e}") from e

        elo_clamped = max(ELO_MIN, min(ELO_MAX, elo)) if elo is not None else None
        options: Dict[str, Any] = {}
        if elo_clamped is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = max(STOCKFISH_ELO_MIN, elo_clamped)
        else:
            options["UCI_LimitStrength"] = False

        limit = chess.engine.Limit(time=analysis_time)
        try:
            # UCI allows only one analysis at a time per engine; serialize to avoid CommandState.NEW assertion
            async with self._lock:
                infos = await engine.analyse(
                    board,
                    limit,
                    multipv=multipv,
                    options=options,
                )
        except Exception as e:
            logger.exception("Engine analysis failed")
            raise RuntimeError(f"Engine analysis failed: {e}") from e

        # infos is list of InfoDict when multipv > 1
        if not isinstance(infos, list):
            infos = [infos]

        candidates = []
        for i, info in enumerate(infos[:multipv]):
            pv = info.get("pv") or []
            score_obj = info.get("score")
            if not pv:
                continue
            move = pv[0]
            # Score is from engine's POV (side to move)
            cp = _score_to_cp(score_obj.white() if board.turn == chess.WHITE else score_obj.black())
            san = board.san(move)
            classification = _classify_move(board, move, cp)
            candidates.append({
                "rank": i + 1,
                "move_uci": move.uci(),
                "move_san": san,
                "score_cp": cp,
                "pv_san": _pv_to_san(board, pv),
                "tactical": classification["tactical"],
                "positional": classification["positional"],
                "themes": classification["themes"],
                "is_capture": classification["is_capture"],
                "is_check": classification["is_check"],
            })

        # Current evaluation (best move) from white's perspective for eval bar
        best_cp = None
        if candidates and candidates[0].get("score_cp") is not None:
            best_cp = candidates[0]["score_cp"]
            if board.turn == chess.BLACK:
                best_cp = -best_cp

        return {
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "candidates": candidates,
            "evaluation_cp": best_cp,
            "is_game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
        }

    async def validate_move(self, fen: str, move_uci: str) -> Dict[str, Any]:
        """Validate a move in the given position. Returns { "legal": bool, "san": str?, "error": str? }."""
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"legal": False, "error": f"Invalid FEN: {e}"}
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return {"legal": False, "error": "Invalid UCI move"}
        if move not in board.legal_moves:
            return {"legal": False, "error": "Move not legal"}
        san = board.san(move)
        return {"legal": True, "san": san, "move_uci": move_uci}


def _pv_to_san(board: chess.Board, pv: List[chess.Move]) -> List[str]:
    b = board.copy()
    out = []
    for m in pv:
        out.append(b.san(m))
        b.push(m)
    return out


# Singleton for app use
chess_engine_service = ChessEngineService()
