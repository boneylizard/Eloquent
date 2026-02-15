import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Chess } from 'chess.js';
import Chessboard from 'chessboardjsx';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Switch } from './ui/switch';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const CHESS_SAVED_GAME_KEY = 'chess-saved-game';

/** chess.js has no .result(); compute result string (1-0, 0-1, 1/2-1/2, *) */
function getGameResult(game) {
  if (!game.isGameOver()) return '*';
  if (game.isCheckmate()) return game.turn() === 'b' ? '1-0' : '0-1';
  if (game.isStalemate() || game.isDraw() || game.isThreefoldRepetition() || game.isInsufficientMaterial()) return '1/2-1/2';
  return '*';
}

/** Human-readable game-over message */
function getGameOverLabel(game) {
  if (!game.isGameOver()) return null;
  if (game.isCheckmate()) return game.turn() === 'b' ? 'Checkmate — White wins' : 'Checkmate — Black wins';
  if (game.isStalemate()) return 'Stalemate — Draw';
  if (game.isDraw()) return 'Draw (50-move rule or insufficient material)';
  if (game.isThreefoldRepetition()) return 'Draw — Threefold repetition';
  if (game.isInsufficientMaterial()) return 'Draw — Insufficient material';
  return 'Game over';
}

/** Convert FEN to chessboardjsx position object: { a1: 'wR', ... } */
function fenToPosition(fen) {
  const game = new Chess(fen);
  const board = game.board();
  const pos = {};
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file];
      if (piece) {
        const square = String.fromCharCode(97 + file) + (8 - rank);
        pos[square] = (piece.color === 'w' ? 'w' : 'b') + piece.type.toUpperCase();
      }
    }
  }
  return pos;
}

function getBackendUrl() {
  const port = parseInt(localStorage.getItem('Eloquent-backend-port') || '8000', 10);
  return `http://localhost:${port}`;
}

export default function ChessTab() {
  const { PRIMARY_API_URL, primaryModel, playTTS } = useApp();
  const baseUrl = PRIMARY_API_URL || getBackendUrl();
  const BOARD_SIZE = 360;

  const [game, setGame] = useState(() => new Chess());
  const [fen, setFen] = useState(STARTING_FEN);
  const [engineAvailable, setEngineAvailable] = useState(false);
  const [elo, setElo] = useState(1600);
  const [personality, setPersonality] = useState('balanced');
  const [commentary, setCommentary] = useState('');
  const [evaluationCp, setEvaluationCp] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [chosenIndex, setChosenIndex] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState('');
  const [moveHistory, setMoveHistory] = useState([]);
  // Toggleable analysis (default off for fair play – no eval/top moves visible)
  const [showEvalBar, setShowEvalBar] = useState(false);
  const [showCommentary, setShowCommentary] = useState(false);
  const [showTopMoves, setShowTopMoves] = useState(false);
  const [showEngineThinking, setShowEngineThinking] = useState(false);
  const [gameOverCommentary, setGameOverCommentary] = useState('');
  const [gameAnalysis, setGameAnalysis] = useState(null);
  const [gameAnalysisFull, setGameAnalysisFull] = useState(null);
  const [analysisReplayIndex, setAnalysisReplayIndex] = useState(0);
  const [loadingCommentary, setLoadingCommentary] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [speakCommentary, setSpeakCommentary] = useState(false);
  const [coachMode, setCoachMode] = useState(false);
  const [userPlaysWhite, setUserPlaysWhite] = useState(true);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const autoCommentaryRequested = useRef(false);

  const isUserTurn = userPlaysWhite ? game.turn() === 'w' : game.turn() === 'b';
  const isGameOver = game.isGameOver();
  const gameResult = getGameResult(game);
  const gameOverLabel = getGameOverLabel(game);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/chess/status`);
      const data = await res.json();
      setEngineAvailable(data.available === true);
      if (!data.available) setError('Stockfish not found. Set STOCKFISH_PATH or install Stockfish.');
      else setError('');
    } catch (e) {
      setEngineAvailable(false);
      setError('Cannot reach chess backend.');
    }
  }, [baseUrl]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const requestAiMove = useCallback(async () => {
    if (!engineAvailable || isUserTurn || isGameOver) return;
    const currentFen = fen;
    setThinking(true);
    setError('');
    try {
      const res = await fetch(`${baseUrl}/chess/ai-move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fen: currentFen,
          elo,
          personality: coachMode ? 'coach' : personality,
          use_llm: true,
          model_name: primaryModel || undefined,
          move_history: moveHistory.slice(-12).map((m) => ({ side: m.side, san: m.san })),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      if (data.game_over) {
        setCommentary(data.commentary || 'Game over.');
        setThinking(false);
        return;
      }
      if (data.move_uci) {
        const gameCopy = new Chess(currentFen);
        gameCopy.move({ from: data.move_uci.slice(0, 2), to: data.move_uci.slice(2, 4), promotion: data.move_uci.length > 4 ? data.move_uci[4] : undefined });
        setGame(new Chess(gameCopy.fen()));
        setFen(gameCopy.fen());
        const comment = data.commentary || '';
        setCommentary(comment);
        setEvaluationCp(data.evaluation_cp);
        setCandidates(data.candidates || []);
        setChosenIndex(data.chosen_index ?? null);
        setMoveHistory((prev) => [...prev, { side: userPlaysWhite ? 'b' : 'w', san: data.move_san, commentary: data.commentary }]);
        if (speakCommentary && comment && typeof playTTS === 'function') {
          playTTS(`chess-${Date.now()}`, comment);
        }
      } else {
        setCommentary(data.commentary || 'No move.');
      }
    } catch (e) {
      setError(e.message || 'AI move failed');
      setCommentary('');
    } finally {
      setThinking(false);
    }
  }, [baseUrl, engineAvailable, isUserTurn, isGameOver, fen, elo, personality, coachMode, primaryModel, moveHistory, speakCommentary, playTTS, userPlaysWhite]);

  useEffect(() => {
    if (!isUserTurn && !isGameOver && engineAvailable && !thinking) {
      requestAiMove();
    }
  }, [fen, isUserTurn, isGameOver, engineAvailable, thinking, requestAiMove]);

  const boardPosition = useMemo(() => fenToPosition(fen), [fen]);

  const onSquareClick = useCallback((square) => {
    if (!isUserTurn || isGameOver || thinking) {
      setSelectedSquare(null);
      return;
    }
    const piece = boardPosition[square];
    const userColor = userPlaysWhite ? 'w' : 'b';
    const isUserPiece = piece && piece[0] === userColor;
    if (selectedSquare) {
      if (square === selectedSquare) {
        setSelectedSquare(null);
        return;
      }
      const gameCopy = new Chess(fen);
      const moveOpts = { from: selectedSquare, to: square };
      const isPawnToLastRank = gameCopy.get(selectedSquare)?.type === 'p' && (square[1] === '8' || square[1] === '1');
      if (isPawnToLastRank) moveOpts.promotion = 'q';
      try {
        const m = gameCopy.move(moveOpts);
        if (m) {
          const newFen = gameCopy.fen();
          const san = m.san;
          const side = m.color;
          requestAnimationFrame(() => {
            setGame(new Chess(newFen));
            setFen(newFen);
            setMoveHistory((prev) => [...prev, { side, san, commentary: null }]);
            setCommentary('');
            setError('');
            setSelectedSquare(null);
          });
        } else {
          if (isUserPiece) setSelectedSquare(square);
          else setSelectedSquare(null);
        }
      } catch {
        if (isUserPiece) setSelectedSquare(square);
        else setSelectedSquare(null);
      }
      return;
    }
    if (isUserPiece) setSelectedSquare(square);
  }, [fen, isUserTurn, isGameOver, thinking, selectedSquare, boardPosition, userPlaysWhite]);

  const onDrop = useCallback(({ sourceSquare, targetSquare }) => {
    if (!targetSquare || !isUserTurn || isGameOver) return false;
    if (sourceSquare === targetSquare) return true;
    const gameCopy = new Chess(fen);
    const moveOpts = { from: sourceSquare, to: targetSquare };
    const isPawnToLastRank = gameCopy.get(sourceSquare)?.type === 'p' && (targetSquare[1] === '8' || targetSquare[1] === '1');
    if (isPawnToLastRank) moveOpts.promotion = 'q';
    let m;
    try {
      m = gameCopy.move(moveOpts);
    } catch {
      return false;
    }
    if (!m) return false;
    const newFen = gameCopy.fen();
    const san = m.san;
    const side = m.color;
    requestAnimationFrame(() => {
      setGame(new Chess(newFen));
      setFen(newFen);
      setMoveHistory((prev) => [...prev, { side, san, commentary: null }]);
      setCommentary('');
      setError('');
      setSelectedSquare(null);
    });
    return true;
  }, [fen, isUserTurn, isGameOver]);

  const newGame = () => {
    setGame(new Chess());
    setFen(STARTING_FEN);
    setCommentary('');
    setEvaluationCp(null);
    setCandidates([]);
    setChosenIndex(null);
    setMoveHistory([]);
    setError('');
    setGameOverCommentary('');
    setGameAnalysis(null);
    setGameAnalysisFull(null);
    setAnalysisReplayIndex(0);
    setSelectedSquare(null);
    autoCommentaryRequested.current = false;
  };

  const saveGame = () => {
    try {
      const payload = {
        fen,
        moveHistory,
        elo,
        personality,
        userPlaysWhite,
        savedAt: new Date().toISOString(),
      };
      new Chess(fen); // validate FEN
      localStorage.setItem(CHESS_SAVED_GAME_KEY, JSON.stringify(payload));
      setError('');
    } catch (e) {
      setError('Could not save game.');
    }
  };

  const loadGame = () => {
    try {
      const raw = localStorage.getItem(CHESS_SAVED_GAME_KEY);
      if (!raw) {
        setError('No saved game to load.');
        return;
      }
      const data = JSON.parse(raw);
      const loadedFen = data.fen || STARTING_FEN;
      new Chess(loadedFen); // validate
      setGame(new Chess(loadedFen));
      setFen(loadedFen);
      setMoveHistory(Array.isArray(data.moveHistory) ? data.moveHistory : []);
      if (typeof data.elo === 'number') setElo(Math.max(800, Math.min(3000, data.elo)));
      if (data.personality && ['balanced', 'aggressive', 'positional', 'defensive', 'romantic'].includes(data.personality)) {
        setPersonality(data.personality);
      }
      if (typeof data.userPlaysWhite === 'boolean') setUserPlaysWhite(data.userPlaysWhite);
      setCommentary('');
      setEvaluationCp(null);
      setCandidates([]);
      setChosenIndex(null);
      setThinking(false);
      setError('');
      setGameOverCommentary('');
      setGameAnalysis(null);
      autoCommentaryRequested.current = false;
    } catch (e) {
      setError('Invalid or missing saved game.');
    }
  };

  const requestGameCommentary = useCallback(async () => {
    if (!isGameOver || moveHistory.length === 0) return;
    setLoadingCommentary(true);
    try {
      const res = await fetch(`${baseUrl}/chess/game-commentary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          move_history: moveHistory.map((m) => ({ side: m.side, san: m.san })),
          result: gameResult,
          model_name: primaryModel || undefined,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setGameOverCommentary(data.commentary || '');
      }
    } catch (e) {
      setGameOverCommentary('');
    } finally {
      setLoadingCommentary(false);
    }
  }, [baseUrl, isGameOver, moveHistory, gameResult, primaryModel]);

  const requestGameAnalysis = useCallback(async () => {
    if (moveHistory.length === 0) return;
    setLoadingAnalysis(true);
    setGameAnalysisFull(null);
    try {
      const res = await fetch(`${baseUrl}/chess/analyze-game-full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          move_history: moveHistory.map((m) => ({ side: m.side, san: m.san })),
          result: gameResult,
          model_name: primaryModel || undefined,
          add_commentary: true,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setGameAnalysisFull(data);
        setAnalysisReplayIndex(0);
        setGameAnalysis({ summary: null, final_eval: data.moves?.length ? data.moves[data.moves.length - 1]?.evaluation_cp : null });
      }
    } catch (e) {
      setGameAnalysisFull(null);
      setGameAnalysis(null);
    } finally {
      setLoadingAnalysis(false);
    }
  }, [baseUrl, moveHistory, gameResult, primaryModel]);

  // Auto-request AI commentary once when the game ends
  useEffect(() => {
    if (isGameOver && moveHistory.length > 0 && primaryModel && !autoCommentaryRequested.current) {
      autoCommentaryRequested.current = true;
      requestGameCommentary();
    }
  }, [isGameOver, moveHistory.length, primaryModel, requestGameCommentary]);

  const evalPercent = evaluationCp == null ? 50 : Math.max(0, Math.min(100, userPlaysWhite ? 50 + (evaluationCp || 0) * 0.25 : 50 - (evaluationCp || 0) * 0.25));

  const inReplayMode = gameAnalysisFull?.moves?.length > 0;
  const replayFen = inReplayMode ? gameAnalysisFull.moves[analysisReplayIndex]?.fen_after : null;
  const displayFen = inReplayMode ? replayFen : fen;
  const displayBoardPosition = useMemo(() => fenToPosition(displayFen || STARTING_FEN), [displayFen]);
  const replayEval = inReplayMode ? gameAnalysisFull.moves[analysisReplayIndex]?.evaluation_cp : null;
  const replayEvalPawns = replayEval != null ? replayEval / 100 : null;
  const replayEvalPercent = replayEvalPawns == null ? 50 : Math.max(0, Math.min(100, userPlaysWhite ? 50 + (replayEvalPawns || 0) * 25 : 50 - (replayEvalPawns || 0) * 25));
  const currentStepCommentary = inReplayMode && gameAnalysisFull.moves[analysisReplayIndex] ? gameAnalysisFull.moves[analysisReplayIndex].commentary : null;
  const replayMoveCount = gameAnalysisFull?.moves?.length ?? 0;

  return (
    <div className="chess-tab flex flex-col h-full max-w-5xl mx-auto">
      <div className="chess-toolbar flex flex-wrap items-center gap-4 mb-4">
        <h2 className="text-xl font-semibold">Chess</h2>
        <label className="flex items-center gap-2 cursor-pointer rounded-md px-2 py-1.5 border border-primary/40 bg-primary/5 hover:bg-primary/10">
          <Switch checked={coachMode} onCheckedChange={setCoachMode} />
          <span className="text-sm font-medium">Play against coach!</span>
        </label>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{coachMode ? 'Coach level' : 'ELO'}</span>
          <Slider
            value={[elo]}
            onValueChange={([v]) => setElo(v)}
            min={800}
            max={3000}
            step={100}
            className="w-32"
          />
          <span className="text-sm font-medium w-10">{elo}</span>
        </div>
        {!coachMode && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Style</span>
            <Select value={personality} onValueChange={setPersonality}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="balanced">Balanced</SelectItem>
                <SelectItem value="aggressive">Aggressive</SelectItem>
                <SelectItem value="positional">Positional</SelectItem>
                <SelectItem value="defensive">Defensive</SelectItem>
                <SelectItem value="romantic">Romantic</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Play as</span>
          <Button variant={userPlaysWhite ? 'default' : 'outline'} size="sm" onClick={() => setUserPlaysWhite(true)}>White</Button>
          <Button variant={!userPlaysWhite ? 'default' : 'outline'} size="sm" onClick={() => setUserPlaysWhite(false)}>Black</Button>
        </div>
        <Button variant="default" size="sm" onClick={newGame}>New game</Button>
        <Button variant="secondary" size="sm" onClick={saveGame}>Save game</Button>
        <Button variant="outline" size="sm" onClick={loadGame}>Load game</Button>
        {!engineAvailable && (
          <span className="text-amber-600 text-sm">Engine unavailable</span>
        )}
        <div className="flex flex-wrap items-center gap-4 ml-2 border-l border-border/60 pl-4">
          <span className="text-xs text-muted-foreground font-medium">Show:</span>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showEvalBar} onCheckedChange={setShowEvalBar} />
            <span className="text-sm">Eval bar</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showCommentary} onCheckedChange={setShowCommentary} />
            <span className="text-sm">Commentary</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showTopMoves} onCheckedChange={setShowTopMoves} />
            <span className="text-sm">Top moves</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={showEngineThinking} onCheckedChange={setShowEngineThinking} />
            <span className="text-sm">Engine lines</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={speakCommentary} onCheckedChange={setSpeakCommentary} />
            <span className="text-sm">Speak commentary (TTS)</span>
          </label>
        </div>
      </div>

      {error && (
        <div className="mb-2 text-sm text-red-600 bg-red-50 dark:bg-red-950/30 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {isGameOver && gameOverLabel && (
        <div className="mb-4 chess-panel p-4 text-center">
          <p className="text-lg font-semibold text-foreground">{gameOverLabel}</p>
          <p className="text-sm text-muted-foreground mt-1">Result: {gameResult}</p>
          <div className="flex flex-wrap justify-center gap-2 mt-3">
            <Button variant="default" size="sm" onClick={requestGameCommentary} disabled={loadingCommentary}>
              {loadingCommentary ? 'Loading…' : 'Get AI game commentary'}
            </Button>
            <Button variant="secondary" size="sm" onClick={requestGameAnalysis} disabled={loadingAnalysis}>
              {loadingAnalysis ? 'Analyzing…' : 'Analyse game (engine + AI)'}
            </Button>
          </div>
          {gameOverCommentary && (
            <div className="mt-3 text-sm text-left bg-muted/50 rounded-lg p-3 border border-border">{gameOverCommentary}</div>
          )}
          {gameAnalysis && (
            <div className="mt-3 text-sm text-left bg-muted/50 rounded-lg p-3 border border-border space-y-1">
              {gameAnalysis.summary && <p>{gameAnalysis.summary}</p>}
              {gameAnalysis.final_eval != null && (
                <p className="text-muted-foreground">Final position eval: {(gameAnalysis.final_eval > 0 ? '+' : '') + gameAnalysis.final_eval.toFixed(2)}</p>
              )}
            </div>
          )}
        </div>
      )}

      <div
        className="flex flex-col lg:flex-row gap-3 flex-1 min-h-0 items-start"
        style={{ '--chess-board-size': `${BOARD_SIZE}px` }}
      >
        <div className="flex flex-col gap-3 shrink-0">
          <div className="flex items-start gap-1">
            {/* Vertical eval bar left of board */}
        {showEvalBar && (
          <div className="relative flex flex-col items-center shrink-0">
            <span className="chess-eval-label chess-eval-label-top">{userPlaysWhite ? 'W' : 'B'}</span>
            <div className="chess-eval-track">
              <div
                className="chess-eval-segment chess-eval-segment--white transition-all duration-300"
                style={{ height: `${userPlaysWhite ? (inReplayMode ? replayEvalPercent : evalPercent) : 100 - (inReplayMode ? replayEvalPercent : evalPercent)}%` }}
              />
              <div
                className="chess-eval-segment chess-eval-segment--black transition-all duration-300"
                style={{ height: `${userPlaysWhite ? 100 - (inReplayMode ? replayEvalPercent : evalPercent) : (inReplayMode ? replayEvalPercent : evalPercent)}%` }}
              />
            </div>
            <span className="chess-eval-label chess-eval-label-bottom">{userPlaysWhite ? 'B' : 'W'}</span>
            <div className="chess-eval-score">
              {inReplayMode
                ? (replayEval != null
                    ? (userPlaysWhite ? `${replayEval > 0 ? '+' : ''}${(replayEval / 100).toFixed(2)}` : `${replayEval < 0 ? '+' : ''}${(-replayEval / 100).toFixed(2)}`)
                    : '—')
                : (evaluationCp != null
                    ? (userPlaysWhite ? `${evaluationCp > 0 ? '+' : ''}${(evaluationCp / 100).toFixed(2)}` : `${evaluationCp < 0 ? '+' : ''}${(-evaluationCp / 100).toFixed(2)}`)
                    : '—')}
            </div>
          </div>
        )}

        {/* Board + Engine lines below */}
        <div className="flex flex-col items-center gap-2">
          {inReplayMode && (
            <div className="flex items-center gap-2 w-full max-w-[var(--chess-board-size)]">
              <Button variant="outline" size="sm" onClick={() => setGameAnalysisFull(null)}>Exit analysis</Button>
              <span className="text-xs text-muted-foreground">
                Step {analysisReplayIndex + 1} of {replayMoveCount}
              </span>
              <Button variant="ghost" size="sm" onClick={() => setAnalysisReplayIndex((i) => Math.max(0, i - 1))} disabled={analysisReplayIndex <= 0}>← Prev</Button>
              <Button variant="ghost" size="sm" onClick={() => setAnalysisReplayIndex((i) => Math.min(replayMoveCount - 1, i + 1))} disabled={analysisReplayIndex >= replayMoveCount - 1}>Next →</Button>
            </div>
          )}
          <div className="chess-board-frame">
            <Chessboard
              key={displayFen || fen}
              position={displayBoardPosition}
              onDrop={inReplayMode ? undefined : onDrop}
              onSquareClick={inReplayMode ? undefined : onSquareClick}
              squareStyles={inReplayMode ? {} : (selectedSquare ? { [selectedSquare]: { backgroundColor: 'rgba(255, 200, 100, 0.6)' } } : {})}
              orientation={userPlaysWhite ? 'white' : 'black'}
              draggable={!inReplayMode && isUserTurn && !isGameOver && !thinking}
              width={BOARD_SIZE}
              showNotation
              lightSquareStyle={{ backgroundColor: 'var(--chess-board-light)' }}
              darkSquareStyle={{ backgroundColor: 'var(--chess-board-dark)' }}
            />
          </div>
          {inReplayMode && currentStepCommentary && (
            <div className="chess-panel w-[var(--chess-board-size)] max-w-full p-2 text-xs text-muted-foreground border border-border rounded">
              {currentStepCommentary}
            </div>
          )}
          {thinking && !inReplayMode && (
            <p className="text-sm text-muted-foreground mt-1">AI is thinking...</p>
          )}

          {showEngineThinking && candidates.length > 0 && (
            <section className="chess-panel w-[var(--chess-board-size)] max-w-full p-3">
              <h3 className="chess-panel-title mb-2">Engine lines</h3>
              <div className="space-y-2 max-h-44 overflow-y-auto">
                {candidates.map((c, i) => (
                  <div key={i} className="rounded-md bg-muted/40 p-2 border border-border/50">
                    <div className="font-medium text-foreground text-xs mb-1">
                      Line {i + 1}: {c.move_san}
                      {c.score_cp != null && (
                        <span className="text-muted-foreground font-normal ml-1">
                          ({c.score_cp > 0 ? '+' : ''}{c.score_cp.toFixed(2)})
                        </span>
                      )}
                    </div>
                    {c.pv_san && c.pv_san.length > 0 && (
                      <div className="text-muted-foreground text-xs leading-relaxed break-words">
                        {c.pv_san.map((move, j) => (
                          <span key={j}>
                            {j > 0 && ' → '}{move}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
          </div>
        </div>

        {/* Commentary, Top moves, Moves in right column */}
        <div className="flex flex-col gap-3 w-full lg:w-[300px] lg:shrink-0">
          {inReplayMode && (
            <section className="chess-panel p-3">
              <h3 className="chess-panel-title mb-2">Analysis — move list</h3>
              <p className="text-xs text-muted-foreground mb-2">Click a move to jump to that position.</p>
              <div className="text-xs max-h-64 overflow-y-auto space-y-0.5">
                {gameAnalysisFull.moves.map((m, i) => {
                  const moveNum = m.move_index === 0 ? 'Start' : `${Math.ceil(m.move_index / 2)}${m.move_index % 2 === 1 ? '.' : '...'}`;
                  const label = m.move_index === 0 ? 'Start' : `${moveNum} ${m.san}`;
                  const evalPawns = m.evaluation_cp != null ? (m.evaluation_cp > 0 ? '+' : '') + (m.evaluation_cp / 100).toFixed(2) : '—';
                  const active = i === analysisReplayIndex;
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setAnalysisReplayIndex(i)}
                      className={`w-full text-left px-2 py-1 rounded border transition-colors ${active ? 'bg-primary/20 border-primary' : 'border-transparent hover:bg-muted/50'}`}
                    >
                      <span className="font-medium">{label}</span>
                      <span className="text-muted-foreground ml-2">{evalPawns}</span>
                    </button>
                  );
                })}
              </div>
            </section>
          )}
          {showCommentary && !inReplayMode && (
            <section className="chess-panel p-3">
              <h3 className="chess-panel-title mb-2">Commentary</h3>
              <p className="text-sm text-foreground leading-relaxed min-h-[2.5rem]">
                {commentary || (isGameOver ? gameResult : '—')}
              </p>
            </section>
          )}

          {showTopMoves && candidates.length > 0 && !inReplayMode && (
            <section className="chess-panel p-3">
              <h3 className="chess-panel-title mb-2">Engine top moves</h3>
              <ul className="text-xs space-y-1">
                {candidates.slice(0, 6).map((c, i) => (
                  <li key={i} className="flex items-baseline gap-2">
                    <span className="font-medium text-primary w-5">{i + 1}.</span>
                    <span>{c.move_san}</span>
                    <span className="text-muted-foreground">({c.score_cp != null ? (c.score_cp > 0 ? '+' : '') + c.score_cp.toFixed(2) : '?'})</span>
                    <span className="text-muted-foreground">— {c.themes?.join(', ') || 'quiet'}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {moveHistory.length > 0 && !inReplayMode && (
            <section className="chess-panel p-3">
              <h3 className="chess-panel-title mb-2">Moves</h3>
              <div className="text-xs max-h-28 overflow-y-auto space-y-1">
                {moveHistory.map((m, i) => (
                  <div key={i} className="flex flex-wrap gap-x-1">
                    <span className="font-medium">{m.side === 'w' ? 'White' : 'Black'}:</span>
                    <span>{m.san}</span>
                    {showCommentary && m.commentary && (
                      <span className="text-muted-foreground">— {m.commentary}</span>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
