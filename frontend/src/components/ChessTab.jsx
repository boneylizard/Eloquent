import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Chess } from 'chess.js';
import Chessboard from 'chessboardjsx';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Switch } from './ui/switch';
import SimpleChatImageButton from './SimpleChatImageButton';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const CHESS_SAVED_GAME_KEY = 'chess-saved-game';
const CHESS_ELOQUENT_USER_KEY = 'chess-eloquent-user-id';

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

/** Build minimal game analysis for replay (no engine/AI yet). history = [{ side, san }, ...]. */
function buildMinimalGameAnalysis(history) {
  if (!history?.length) return null;
  const g = new Chess();
  const moves = [{ move_index: 0, san: null, side: null, fen_after: g.fen(), evaluation_cp: null, commentary: null }];
  for (let i = 0; i < history.length; i++) {
    const m = history[i];
    if (!g.move(m.san)) break;
    moves.push({ move_index: i + 1, san: m.san, side: m.side, fen_after: g.fen(), evaluation_cp: null, commentary: null });
  }
  return { moves, result: getGameResult(g) };
}

/** Parse PGN header tags. Returns { whiteName, blackName, whiteElo, blackElo } (empty string if missing). */
function parsePgnHeaders(headersStr) {
  const s = (headersStr || '').trim();
  const tag = (name) => {
    const m = s.match(new RegExp('\\[' + name + '\\s+"([^"]*)"\\]', 'i'));
    return m ? m[1].trim() : '';
  };
  return {
    whiteName: tag('White'),
    blackName: tag('Black'),
    whiteElo: tag('WhiteElo'),
    blackElo: tag('BlackElo'),
  };
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
  const { PRIMARY_API_URL, primaryModel, playTTS, settings, updateSettings } = useApp();
  const baseUrl = PRIMARY_API_URL || getBackendUrl();
  const historianAvatarSize = settings?.characterAvatarSize ?? 40;
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
  const [takeOverMode, setTakeOverMode] = useState(false);
  const [branchFen, setBranchFen] = useState(null);
  const [branchEval, setBranchEval] = useState(null);
  const [branchCandidates, setBranchCandidates] = useState([]);
  const [branchSelectedSquare, setBranchSelectedSquare] = useState(null);
  const [loadingCommentary, setLoadingCommentary] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [speakCommentary, setSpeakCommentary] = useState(false);
  const [coachMode, setCoachMode] = useState(false);
  const [userPlaysWhite, setUserPlaysWhite] = useState(true);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const autoCommentaryRequested = useRef(false);
  // Chess OAuth / linked accounts
  const [linkedAccounts, setLinkedAccounts] = useState([]);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');
  const [importedGames, setImportedGames] = useState([]);
  const [importLoading, setImportLoading] = useState(false);
  const [importSuccess, setImportSuccess] = useState(null);
  const [accountsPanelOpen, setAccountsPanelOpen] = useState(false);
  const [deepAnalysisLoading, setDeepAnalysisLoading] = useState(false);
  const [deepAnalysisReport, setDeepAnalysisReport] = useState(null);
  const [myGamesPanelOpen, setMyGamesPanelOpen] = useState(false);
  /** When viewing an imported Lichess game: { whiteName, blackName, whiteElo, blackElo }. Cleared on new/saved load. */
  const [loadedGameMeta, setLoadedGameMeta] = useState(null);
  const [historianOpen, setHistorianOpen] = useState(true);
  const [historianMessages, setHistorianMessages] = useState([]);
  const [historianCurrentFact, setHistorianCurrentFact] = useState(null);
  const [historianInput, setHistorianInput] = useState('');
  const [historianLoading, setHistorianLoading] = useState(false);
  const historianLastActivityRef = useRef(Date.now());
  const historianFactsIntervalRef = useRef(null);
  const historianRecentFactsRef = useRef([]);
  const [historianAvatarPickerOpen, setHistorianAvatarPickerOpen] = useState(false);
  const [historianAvatarZoomOpen, setHistorianAvatarZoomOpen] = useState(false);
  const [historianPersonaLoading, setHistorianPersonaLoading] = useState(false);
  const historianAvatarInputRef = useRef(null);
  const FALLBACK_FACT = 'Chess has been played for over a millennium.';

  const isUserTurn = userPlaysWhite ? game.turn() === 'w' : game.turn() === 'b';
  const isGameOver = game.isGameOver();
  const gameResult = getGameResult(game);
  const gameOverLabel = getGameOverLabel(game);

  const eloquentUserId = useMemo(() => {
    let id = localStorage.getItem(CHESS_ELOQUENT_USER_KEY);
    if (!id) {
      id = 'eloquent-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem(CHESS_ELOQUENT_USER_KEY, id);
    }
    return id;
  }, []);

  const authHeaders = useMemo(() => ({ 'X-Eloquent-User-Id': eloquentUserId }), [eloquentUserId]);

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

  /** Normalize engine eval to centipawns (engine may return pawns, e.g. 0.5). */
  const toCentipawns = useCallback((v) => {
    if (v == null) return null;
    if (Math.abs(v) <= 20) return Math.round(v * 100);
    return v;
  }, []);

  const requestPositionEval = useCallback(async (fenToAnalyze) => {
    if (!engineAvailable || !fenToAnalyze) return;
    try {
      const res = await fetch(`${baseUrl}/chess/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: fenToAnalyze, multipv: 1, analysis_time: 0.2 }),
      });
      if (!res.ok) return;
      const data = await res.json();
      setEvaluationCp(toCentipawns(data.evaluation_cp));
    } catch {
      setEvaluationCp(null);
    }
  }, [baseUrl, engineAvailable, toCentipawns]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const fetchAuthMe = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/auth/me`, { headers: authHeaders });
      if (res.ok) {
        const data = await res.json();
        setLinkedAccounts(data.accounts || []);
      }
    } catch {
      setLinkedAccounts([]);
    }
  }, [baseUrl, authHeaders]);

  const fetchImportedGames = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/auth/games?limit=50`, { headers: authHeaders });
      if (res.ok) {
        const data = await res.json();
        setImportedGames(data.games || []);
      }
    } catch {
      setImportedGames([]);
    }
  }, [baseUrl, authHeaders]);

  useEffect(() => {
    fetchAuthMe();
  }, [fetchAuthMe]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const linked = params.get('chess_linked');
    const err = params.get('chess_error');
    if (linked || err) {
      const u = new URL(window.location.href);
      u.searchParams.delete('chess_linked');
      u.searchParams.delete('chess_error');
      window.history.replaceState({}, '', u.toString());
      if (linked) {
        setAuthError('');
        fetchAuthMe();
        setAccountsPanelOpen(true);
      }
      if (err) setAuthError(err === 'access_denied' ? 'You denied access.' : err);
    }
  }, []);

  const loginLichess = useCallback(async () => {
    setAuthLoading(true);
    setAuthError('');
    try {
      const res = await fetch(`${baseUrl}/auth/lichess/authorize`, { headers: authHeaders });
      const data = await res.json();
      if (data?.url) window.location.href = data.url;
      else setAuthError('Could not get login URL.');
    } catch (e) {
      setAuthError(e.message || 'Login failed');
    } finally {
      setAuthLoading(false);
    }
  }, [baseUrl, authHeaders]);

  const unlinkAccount = useCallback(async (platform) => {
    try {
      const res = await fetch(`${baseUrl}/auth/unlink/${platform}`, { method: 'POST', headers: authHeaders });
      if (res.ok) await fetchAuthMe();
    } catch {}
  }, [baseUrl, authHeaders, fetchAuthMe]);

  const importGames = useCallback(async () => {
    setImportLoading(true);
    setAuthError('');
    setImportSuccess(null);
    try {
      const res = await fetch(`${baseUrl}/auth/import-games?max_games=50`, { method: 'POST', headers: authHeaders });
      const data = await res.json();
      if (!res.ok) {
        const detail = data?.detail || res.statusText;
        setAuthError(typeof detail === 'string' ? detail : 'Import failed');
        return;
      }
      if (data?.errors?.length) {
        setAuthError(data.errors.join('; '));
        if (data.imported > 0) setImportSuccess(`Imported ${data.imported} games.`);
      } else {
        setImportSuccess(data.imported > 0 ? `Imported ${data.imported} games. See "My games" below.` : 'No new games (you may already have the last 50).');
      }
      await fetchImportedGames();
      if ((data.imported || 0) > 0) setMyGamesPanelOpen(true);
    } catch (e) {
      setAuthError(e.message || 'Import failed');
    } finally {
      setImportLoading(false);
    }
  }, [baseUrl, authHeaders, fetchImportedGames]);

  useEffect(() => {
    if (accountsPanelOpen && linkedAccounts.length > 0) fetchImportedGames();
  }, [accountsPanelOpen, linkedAccounts.length, fetchImportedGames]);

  useEffect(() => {
    if (linkedAccounts.length > 0) fetchImportedGames();
  }, [linkedAccounts.length, fetchImportedGames]);

  useEffect(() => {
    if (linkedAccounts.length > 0 && !accountsPanelOpen) setAccountsPanelOpen(true);
  }, [linkedAccounts.length]);

  const loadImportedGame = useCallback((gameOrPgn) => {
    const DEBUG_PGN = true;
    const log = (msg, ...args) => { if (DEBUG_PGN) console.warn('[Chess PGN]', msg, ...args); };
    const isGameObject = typeof gameOrPgn === 'object' && gameOrPgn != null && gameOrPgn.pgn_text;
    const pgnText = isGameObject ? gameOrPgn.pgn_text : (typeof gameOrPgn === 'string' ? gameOrPgn : '');
    const linkedUsername = isGameObject ? (gameOrPgn.username || '').trim() : null;
    let raw = pgnText.trim().replace(/\r\n/g, '\n') || '';
    if (!raw) return;
    raw = raw.replace(/\b0-0-0\b/g, 'O-O-O').replace(/\b0-0\b/g, 'O-O');
    log('raw length:', raw.length, '| first 300 chars:', JSON.stringify(raw.slice(0, 300)));
    const stripMoveComments = (s) => {
      if (!s || !s.trim()) return s;
      let t = s
        .replace(/\{[^}]*\}/g, ' ')
        .replace(/;[^\n]*/g, ' ');
      while (t.includes('(') && t.includes(')')) {
        const next = t.replace(/\([^()]*\)/g, ' ');
        if (next === t) break;
        t = next;
      }
      t = t.replace(/\$\d+/g, ' ').replace(/\[\s*%clk[^\]]*\]/gi, ' ').replace(/\[\s*%eval[^\]]*\]/gi, ' ');
      t = t.replace(/[()]/g, ' ');
      return t.replace(/\s+/g, ' ').trim();
    };
    const blankLine = raw.search(/\n\s*\n/);
    const headersPart = blankLine >= 0 ? raw.slice(0, blankLine).trim() : '';
    const movePart = blankLine >= 0 ? raw.slice(blankLine).trim() : raw;
    log('blankLine index:', blankLine, '| headersPart length:', headersPart.length, '| movePart length:', movePart.length);
    const meta = parsePgnHeaders(headersPart);
    const applyLoadedGameMeta = () => {
      let userSide = null;
      if (linkedUsername) {
        if (meta.whiteName === linkedUsername) { setUserPlaysWhite(true); userSide = 'white'; }
        else if (meta.blackName === linkedUsername) { setUserPlaysWhite(false); userSide = 'black'; }
      }
      setLoadedGameMeta({ whiteName: meta.whiteName, blackName: meta.blackName, whiteElo: meta.whiteElo, blackElo: meta.blackElo, userSide });
    };
    log('movePart first 400:', JSON.stringify(movePart.slice(0, 400)));
    let strippedMoves = stripMoveComments(movePart);
    strippedMoves = strippedMoves.replace(/[!?]+/g, '').replace(/[\u00A0\u200B-\u200D\uFEFF]/g, ' ').replace(/\s+/g, ' ').trim();
    const hasResult = /\s(1-0|0-1|1\/2-1\/2|\*)\s*$/.test(strippedMoves);
    if (!hasResult && strippedMoves.trim()) {
      const resultMatch = headersPart.match(/\[Result\s+"([^"]+)"\]/i);
      const resultStr = resultMatch ? resultMatch[1].trim() : '*';
      if (['1-0', '0-1', '1/2-1/2', '*'].includes(resultStr)) strippedMoves = strippedMoves.trim() + ' ' + resultStr;
      else strippedMoves = strippedMoves.trim() + ' *';
    }
    log('strippedMoves length:', strippedMoves.length, '| first 400:', JSON.stringify(strippedMoves.slice(0, 400)));
    const pgnWithHeaders = headersPart ? `${headersPart}\n\n${strippedMoves}` : strippedMoves;
    setError('');
    try {
      const g = new Chess();
      let loaded = g.loadPgn(pgnWithHeaders, { sloppy: true });
      if (!loaded && strippedMoves) {
        loaded = g.loadPgn(strippedMoves, { sloppy: true });
      }
      if (!loaded && strippedMoves) {
        const movesOneLine = strippedMoves.replace(/\s+/g, ' ').trim();
        log('trying moves as single line, length:', movesOneLine.length);
        loaded = g.loadPgn(movesOneLine, { sloppy: true });
      }
      if (!loaded && strippedMoves) {
        const tokens = strippedMoves.replace(/\s*(1-0|0-1|1\/2-1\/2|\*)\s*$/i, '').trim().split(/\s+/).filter((t) => /^(O-O-O|O-O|[PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?[+#]?)$/.test(t));
        const replay = new Chess();
        let replayOk = true;
        for (const san of tokens) {
          if (!replay.move(san)) { replayOk = false; break; }
        }
        if (replayOk && tokens.length > 0) {
          applyLoadedGameMeta();
          const history = replay.history({ verbose: true });
          const moveList = history.map((m) => ({ side: m.color === 'w' ? 'w' : 'b', san: m.san }));
          setMoveHistory(moveList);
          setGame(new Chess(replay.fen()));
          setFen(replay.fen());
          setCommentary('');
          const minimal = buildMinimalGameAnalysis(moveList);
          setGameAnalysisFull(minimal);
          setAnalysisReplayIndex(minimal?.moves?.length ? minimal.moves.length - 1 : 0);
          setTakeOverMode(false);
          setMyGamesPanelOpen(false);
          setError('');
          log('loaded via replay fallback (loadPgn returned false), moves:', history.length);
          return;
        }
      }
      if (!loaded) {
        log('loadPgn FAILED (all attempts). pgnWithHeaders first 500:', JSON.stringify(pgnWithHeaders.slice(0, 500)));
        log('strippedMoves only first 500:', JSON.stringify(strippedMoves.slice(0, 500)));
        setError('Could not parse game (invalid PGN). See console for [Chess PGN] debug.');
        return;
      }
      applyLoadedGameMeta();
      const history = g.history({ verbose: true });
      const moveList = history.map((m) => ({ side: m.color === 'w' ? 'w' : 'b', san: m.san }));
      setMoveHistory(moveList);
      setGame(new Chess(g.fen()));
      setFen(g.fen());
      setCommentary('');
      const minimal = buildMinimalGameAnalysis(moveList);
      setGameAnalysisFull(minimal);
      setAnalysisReplayIndex(minimal?.moves?.length ? minimal.moves.length - 1 : 0);
      setTakeOverMode(false);
      setMyGamesPanelOpen(false);
    } catch (e) {
      log('loadImportedGame exception:', e?.message, '| raw length:', raw?.length, '| raw first 600:', JSON.stringify(raw?.slice(0, 600)));
      if (e?.expected) log('parser expected:', e.expected);
      if (e?.found !== undefined) log('parser found:', e.found);
      const replayFallback = () => {
        const tokens = strippedMoves.replace(/\s*(1-0|0-1|1\/2-1\/2|\*)\s*$/i, '').trim().split(/\s+/).filter((t) => /^(O-O-O|O-O|[PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?[+#]?)$/.test(t));
        const g = new Chess();
        for (const san of tokens) {
          const move = g.move(san);
          if (!move) return null;
        }
        return g;
      };
      const g = replayFallback();
      if (g) {
        applyLoadedGameMeta();
        const history = g.history({ verbose: true });
        const moveList = history.map((m) => ({ side: m.color === 'w' ? 'w' : 'b', san: m.san }));
        setMoveHistory(moveList);
        setGame(new Chess(g.fen()));
        setFen(g.fen());
        setCommentary('');
        const minimal = buildMinimalGameAnalysis(moveList);
        setGameAnalysisFull(minimal);
        setAnalysisReplayIndex(minimal?.moves?.length ? minimal.moves.length - 1 : 0);
        setTakeOverMode(false);
        setMyGamesPanelOpen(false);
        setError('');
        log('loaded via replay fallback, moves:', history.length);
        return;
      }
      setError('Could not load game: ' + (e?.message || 'invalid PGN') + '. See console for [Chess PGN] debug.');
    }
  }, []);

  const HISTORIAN_ERROR_MESSAGE = 'Something went wrong. Please check your connection and try again.';

  const sendHistorianMessage = useCallback(async (content) => {
    if (!content?.trim() || historianLoading) return;
    const userMsg = { role: 'user', content: content.trim() };
    setHistorianMessages((prev) => [...prev, userMsg]);
    setHistorianInput('');
    historianLastActivityRef.current = Date.now();
    setHistorianLoading(true);
    try {
      const res = await fetch(`${baseUrl}/chess/historian/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...historianMessages, userMsg].map((m) => ({ role: m.role, content: m.content })),
          model_name: primaryModel || undefined,
          persona_prompt: settings?.chessHistorianPersonaPrompt || undefined,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setHistorianMessages((prev) => [...prev, { role: 'assistant', content: data.reply || '', pgn: data.pgn || null }]);
    } catch (e) {
      setHistorianMessages((prev) => [...prev, { role: 'assistant', content: HISTORIAN_ERROR_MESSAGE, pgn: null, isError: true }]);
    } finally {
      setHistorianLoading(false);
    }
  }, [baseUrl, historianMessages, historianLoading, primaryModel, settings?.chessHistorianPersonaPrompt]);

  const retryHistorianAt = useCallback(async (index, currentMessages) => {
    if (historianLoading || index >= currentMessages.length) return;
    const messagesToSend = currentMessages.slice(0, index).map((m) => ({ role: m.role, content: m.content }));
    setHistorianLoading(true);
    try {
      const res = await fetch(`${baseUrl}/chess/historian/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messagesToSend,
          model_name: primaryModel || undefined,
          persona_prompt: settings?.chessHistorianPersonaPrompt || undefined,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setHistorianMessages((prev) => [...prev.slice(0, index), { role: 'assistant', content: data.reply || '', pgn: data.pgn || null }, ...prev.slice(index + 1)]);
    } catch (e) {
      setHistorianMessages((prev) => [...prev.slice(0, index), { role: 'assistant', content: HISTORIAN_ERROR_MESSAGE, pgn: null, isError: true }, ...prev.slice(index + 1)]);
    } finally {
      setHistorianLoading(false);
    }
  }, [baseUrl, historianLoading, primaryModel, settings?.chessHistorianPersonaPrompt]);

  const requestHistorianFact = useCallback(async () => {
    if (historianLoading) return;
    historianLastActivityRef.current = Date.now();
    setHistorianLoading(true);
    try {
      const recentFacts = historianRecentFactsRef.current.slice(-16);
      const res = await fetch(`${baseUrl}/chess/historian/fact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: primaryModel || undefined,
          recent_facts: recentFacts,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      const fact = (data.fact || '').trim();
      const norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
      const isFallbackOrEmpty = !fact || fact.length < 15 || norm(fact) === norm(FALLBACK_FACT) || /^\.+$/.test(fact);
      if (isFallbackOrEmpty) return;
      const factNorm = norm(fact);
      const isDup = recentFacts.some((r) => norm(r) === factNorm || (r.length > 30 && fact.length > 30 && (norm(r).includes(factNorm) || factNorm.includes(norm(r)))));
      if (!isDup) {
        historianRecentFactsRef.current = [...historianRecentFactsRef.current.slice(-19), fact];
        setHistorianCurrentFact(fact);
      }
    } catch {
      // Keep previous fact; do not set fallback to avoid repetition
    } finally {
      setHistorianLoading(false);
    }
  }, [baseUrl, historianLoading, primaryModel]);

  useEffect(() => {
    if (!historianOpen) return;
    const interval = setInterval(() => {
      const elapsed = (Date.now() - historianLastActivityRef.current) / 1000;
      if (elapsed >= 30 && !historianLoading && !loadingAnalysis && !deepAnalysisLoading) {
        historianLastActivityRef.current = Date.now();
        requestHistorianFact();
      }
    }, 10000);
    return () => clearInterval(interval);
  }, [historianOpen, historianLoading, loadingAnalysis, deepAnalysisLoading, requestHistorianFact]);

  const handleHistorianAvatarUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const allowed = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
    if (!allowed.includes(file.type)) {
      setError('Invalid image type. Use PNG, JPEG, GIF, or WebP.');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${baseUrl}/upload_avatar`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data?.file_url) {
        updateSettings({ chessHistorianAvatar: data.file_url });
        setError('');
      } else throw new Error(data?.detail || 'Upload failed');
    } catch (err) {
      setError('Avatar upload failed: ' + (err?.message || ''));
    }
    e.target.value = null;
  }, [baseUrl, updateSettings]);

  const requestHistorianPersona = useCallback(async () => {
    setHistorianPersonaLoading(true);
    try {
      const res = await fetch(`${baseUrl}/chess/historian/persona`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: primaryModel || undefined }),
      });
      const data = await res.json();
      const persona = data?.persona?.trim();
      if (persona) updateSettings({ chessHistorianPersonaPrompt: persona });
    } catch {
      setError('Could not generate persona.');
    } finally {
      setHistorianPersonaLoading(false);
    }
  }, [baseUrl, primaryModel, updateSettings]);

  const requestDeepAnalysis = useCallback(async (fenToAnalyze) => {
    const f = (fenToAnalyze || fen || '').trim();
    if (!f) return;
    setDeepAnalysisLoading(true);
    setDeepAnalysisReport(null);
    try {
      const res = await fetch(`${baseUrl}/chess/deep-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fen: f,
          move_history: moveHistory.map((m) => ({ side: m.side, san: m.san })),
          model_name: primaryModel || undefined,
        }),
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setDeepAnalysisReport(data);
    } catch (e) {
      setDeepAnalysisReport({ report: `Deep analysis failed: ${e.message}`, citations: [], sources_used: [] });
    } finally {
      setDeepAnalysisLoading(false);
    }
  }, [baseUrl, fen, moveHistory, primaryModel]);

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
        setEvaluationCp(toCentipawns(data.evaluation_cp));
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
  }, [baseUrl, engineAvailable, isUserTurn, isGameOver, fen, elo, personality, coachMode, primaryModel, moveHistory, speakCommentary, playTTS, userPlaysWhite, toCentipawns]);

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
          requestPositionEval(newFen);
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
  }, [fen, isUserTurn, isGameOver, thinking, selectedSquare, boardPosition, userPlaysWhite, requestPositionEval]);

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
    requestPositionEval(newFen);
    return true;
  }, [fen, isUserTurn, isGameOver, requestPositionEval]);

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
    setTakeOverMode(false);
    setBranchFen(null);
    setBranchEval(null);
    setBranchCandidates([]);
    setBranchSelectedSquare(null);
    setSelectedSquare(null);
    setLoadedGameMeta(null);
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
      setLoadedGameMeta(null);
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

  const requestBranchAnalysis = useCallback(async (fenToAnalyze) => {
    if (!engineAvailable || !fenToAnalyze) return;
    try {
      const res = await fetch(`${baseUrl}/chess/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: fenToAnalyze, multipv: 5, analysis_time: 0.4 }),
      });
      if (!res.ok) return;
      const data = await res.json();
      let ev = data.evaluation_cp;
      if (ev != null && Math.abs(ev) <= 100) ev = Math.round(ev * 100);
      setBranchEval(ev);
      setBranchCandidates(data.candidates || []);
    } catch {
      setBranchEval(null);
      setBranchCandidates([]);
    }
  }, [baseUrl, engineAvailable]);

  // When we have moveHistory but no gameAnalysisFull (e.g. loaded game), use minimal analysis so replay UI is identical to post-game analysis
  const analysisFromMoveHistory = useMemo(
    () => (moveHistory.length > 0 ? buildMinimalGameAnalysis(moveHistory) : null),
    [moveHistory]
  );
  const effectiveAnalysis =
    gameAnalysisFull?.moves?.length > 0
      ? gameAnalysisFull
      : analysisFromMoveHistory?.moves?.length > 0
        ? analysisFromMoveHistory
        : null;
  const inReplayMode = effectiveAnalysis != null;
  const replayMoveCountRaw = effectiveAnalysis?.moves?.length ?? 0;
  const clampedReplayIndex = replayMoveCountRaw > 0 ? Math.min(analysisReplayIndex, replayMoveCountRaw - 1) : 0;
  const replayFen = inReplayMode ? effectiveAnalysis.moves[clampedReplayIndex]?.fen_after : null;

  const applyReplayMove = useCallback((fromSquare, toSquare, promotion) => {
    const currentFen = (takeOverMode && branchFen) ? branchFen : (effectiveAnalysis?.moves?.[clampedReplayIndex]?.fen_after || STARTING_FEN);
    const gameCopy = new Chess(currentFen);
    const opts = { from: fromSquare, to: toSquare };
    if (promotion) opts.promotion = promotion;
    const m = gameCopy.move(opts);
    if (!m) return false;
    const newFen = gameCopy.fen();
    setBranchFen(newFen);
    setBranchSelectedSquare(null);
    requestBranchAnalysis(newFen);
    return true;
  }, [takeOverMode, branchFen, effectiveAnalysis, clampedReplayIndex, requestBranchAnalysis]);

  const onReplaySquareClick = useCallback((square) => {
    if (!inReplayMode || !takeOverMode) return;
    const currentFen = branchFen ?? effectiveAnalysis?.moves?.[clampedReplayIndex]?.fen_after;
    if (!currentFen) return;
    const pos = fenToPosition(currentFen);
    const piece = pos[square];
    const turn = new Chess(currentFen).turn();
    const turnChar = turn === 'w' ? 'w' : 'b';
    const isTurnPiece = piece && piece[0] === turnChar;
    if (branchSelectedSquare) {
      if (square === branchSelectedSquare) {
        setBranchSelectedSquare(null);
        return;
      }
      const sourcePiece = pos[branchSelectedSquare];
      const promoted = (square[1] === '8' || square[1] === '1') && sourcePiece && sourcePiece[1] === 'P';
      if (applyReplayMove(branchSelectedSquare, square, promoted ? 'q' : undefined)) return;
      if (isTurnPiece) setBranchSelectedSquare(square);
      else setBranchSelectedSquare(null);
      return;
    }
    if (isTurnPiece) setBranchSelectedSquare(square);
  }, [inReplayMode, takeOverMode, branchFen, effectiveAnalysis, clampedReplayIndex, branchSelectedSquare, applyReplayMove]);

  const onReplayDrop = useCallback(({ sourceSquare, targetSquare }) => {
    if (!inReplayMode || !takeOverMode || sourceSquare === targetSquare) return false;
    const currentFen = branchFen ?? effectiveAnalysis?.moves?.[clampedReplayIndex]?.fen_after;
    if (!currentFen) return false;
    const gameCopy = new Chess(currentFen);
    const piece = gameCopy.get(sourceSquare);
    if (!piece) return false;
    const needPromo = (targetSquare[1] === '8' || targetSquare[1] === '1') && piece.type === 'p';
    return applyReplayMove(sourceSquare, targetSquare, needPromo ? 'q' : undefined);
  }, [inReplayMode, takeOverMode, branchFen, effectiveAnalysis, clampedReplayIndex, applyReplayMove]);

  // Auto-request AI commentary once when the game ends
  useEffect(() => {
    if (isGameOver && moveHistory.length > 0 && primaryModel && !autoCommentaryRequested.current) {
      autoCommentaryRequested.current = true;
      requestGameCommentary();
    }
  }, [isGameOver, moveHistory.length, primaryModel, requestGameCommentary]);

  const evalPercent = evaluationCp == null ? 50 : Math.max(0, Math.min(100, userPlaysWhite ? 50 + (evaluationCp || 0) * 0.25 : 50 - (evaluationCp || 0) * 0.25));

  const displayFen = inReplayMode ? (takeOverMode && branchFen ? branchFen : replayFen) : fen;
  const displayBoardPosition = useMemo(() => fenToPosition(displayFen || STARTING_FEN), [displayFen]);
  const replayEval = inReplayMode ? effectiveAnalysis?.moves?.[clampedReplayIndex]?.evaluation_cp : null;
  const displayEvalCp = inReplayMode ? (takeOverMode ? (branchEval != null ? branchEval : replayEval) : replayEval) : evaluationCp;
  const displayEvalPawns = displayEvalCp != null ? displayEvalCp / 100 : null;
  const displayEvalPercent = displayEvalPawns == null ? 50 : Math.max(0, Math.min(100, userPlaysWhite ? 50 + (displayEvalPawns || 0) * 25 : 50 - (displayEvalPawns || 0) * 25));
  const displayCandidates = inReplayMode && takeOverMode ? branchCandidates : candidates;
  const replayEvalPercent = displayEvalPercent;
  const currentStepCommentary = inReplayMode && effectiveAnalysis?.moves?.[clampedReplayIndex] ? effectiveAnalysis.moves[clampedReplayIndex].commentary : null;
  const currentStepMove = inReplayMode && effectiveAnalysis?.moves?.[clampedReplayIndex] ? effectiveAnalysis.moves[clampedReplayIndex] : null;
  const replayMoveCount = replayMoveCountRaw;
  const hasMinimalAnalysis = replayMoveCount > 0 && effectiveAnalysis?.moves?.every((m) => m.evaluation_cp == null);

  // When entering take-over, analyze current position so eval and lines show immediately
  useEffect(() => {
    if (inReplayMode && takeOverMode && replayFen && !branchFen) {
      requestBranchAnalysis(replayFen);
    }
  }, [inReplayMode, takeOverMode, replayFen, branchFen, requestBranchAnalysis]);

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
        <Button variant="outline" size="sm" onClick={() => setAccountsPanelOpen((o) => !o)}>
          {accountsPanelOpen ? 'Hide accounts' : 'Link Lichess / Chess.com'}
        </Button>
        <Button variant={myGamesPanelOpen ? 'secondary' : 'outline'} size="sm" onClick={() => setMyGamesPanelOpen((o) => !o)} title="View games imported from Lichess">
          {importedGames.length > 0 ? `My games (${importedGames.length})` : 'My games'}
        </Button>
        <Button variant="outline" size="sm" onClick={() => requestDeepAnalysis(displayFen || fen)} disabled={deepAnalysisLoading} title="Research this position: Lichess Opening Explorer + web search; AI summarizes with citations. Different from full-game analysis (use replay + Generate engine + AI analysis for that).">
          {deepAnalysisLoading ? 'Researching…' : 'Deep Analysis'}
        </Button>
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

      {myGamesPanelOpen && (
        <section className="chess-panel p-4 mb-4 border border-primary/30">
          <h3 className="chess-panel-title mb-2">My games</h3>
          <p className="text-sm text-muted-foreground mb-2">Games imported from Lichess. Click Load to open on the board.</p>
          {importedGames.length > 0 ? (
            <ul className="text-sm max-h-56 overflow-y-auto space-y-1.5 border border-border rounded p-3 bg-muted/20">
              {importedGames.map((g) => (
                <li key={g.id} className="flex items-center gap-2">
                  <span className="truncate flex-1 font-medium">{g.platform} · {g.platform_game_id}</span>
                  <Button variant="default" size="sm" className="shrink-0" onClick={() => loadImportedGame(g)}>Load</Button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground italic mb-2">
              No games yet. Click &quot;Link Lichess / Chess.com&quot; above to connect your account, then use &quot;Import last 50 games&quot; in that panel.
            </p>
          )}
          <div className="flex gap-2 mt-2">
            <Button variant="outline" size="sm" onClick={() => { setMyGamesPanelOpen(false); setAccountsPanelOpen(true); }}>Link account &amp; import</Button>
            <Button variant="ghost" size="sm" onClick={() => setMyGamesPanelOpen(false)}>Close</Button>
          </div>
        </section>
      )}

      {accountsPanelOpen && (
        <section className="chess-panel p-4 mb-4">
          <h3 className="chess-panel-title mb-3">Chess accounts</h3>
          {authError && (
            <p className="text-sm text-amber-600 dark:text-amber-400 mb-2">{authError}</p>
          )}
          <div className="flex flex-wrap gap-2 mb-3">
            <Button variant="default" size="sm" onClick={loginLichess} disabled={authLoading}>
              {authLoading ? 'Redirecting…' : 'Login with Lichess'}
            </Button>
            <Button variant="outline" size="sm" disabled title="Chess.com requires developer approval">
              Login with Chess.com (coming soon)
            </Button>
          </div>
          {linkedAccounts.length > 0 && (
            <>
              <p className="text-sm text-muted-foreground mb-1">Linked:</p>
              <ul className="text-sm space-y-1 mb-3">
                {linkedAccounts.map((acc) => (
                  <li key={acc.platform} className="flex items-center gap-2">
                    <span className="font-medium">{acc.platform === 'lichess' ? 'Lichess' : 'Chess.com'}: @{acc.username}</span>
                    <Button variant="ghost" size="sm" className="h-6 px-1 text-muted-foreground" onClick={() => unlinkAccount(acc.platform)}>Unlink</Button>
                  </li>
                ))}
              </ul>
              <Button variant="secondary" size="sm" onClick={importGames} disabled={importLoading} className="mb-2">
                {importLoading ? 'Importing…' : 'Import last 50 games'}
              </Button>
              {importSuccess && (
                <p className="text-sm text-green-600 dark:text-green-400 mb-2">{importSuccess}</p>
              )}
            </>
          )}
          <p className="text-sm text-muted-foreground mt-2">
            {linkedAccounts.length > 0
              ? 'Use "My games" (toolbar) to view and load imported games.'
              : 'Link Lichess above, then use "My games" to import and load games.'}
          </p>
          {null /* removed duplicate list
            <p className="text-sm text-muted-foreground italic">
              {linkedAccounts.length > 0
                ? 'No games imported yet. Click “Import last 10 games” above to fetch your recent Lichess games.'
                : 'Link Lichess above, then import games.'}
            </p>
          */}
        </section>
      )}

      {deepAnalysisReport && (
        <section className="chess-panel p-4 mb-4">
          <h3 className="chess-panel-title mb-1">Deep Analysis (research this position)</h3>
          <p className="text-xs text-muted-foreground mb-2">Lichess Explorer + web search, summarized with citations. For per-move annotation of a whole game, use replay then &quot;Generate engine + AI analysis&quot;.</p>
          <div className="text-sm prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap mb-2">
            {deepAnalysisReport.report}
          </div>
          {deepAnalysisReport.sources_used?.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Sources: {deepAnalysisReport.sources_used.join(', ')}
            </p>
          )}
          <Button variant="ghost" size="sm" className="mt-2" onClick={() => setDeepAnalysisReport(null)}>Dismiss</Button>
        </section>
      )}

      <section className="chess-panel p-4 mb-4 border border-border/50">
        <button type="button" className="flex items-center gap-2 w-full text-left font-semibold mb-2" onClick={() => setHistorianOpen((o) => !o)}>
          <span>{historianOpen ? '▼' : '▶'}</span>
          <div
            className="rounded-full border border-border bg-muted flex-shrink-0 overflow-hidden cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary"
            style={{ width: historianAvatarSize, height: historianAvatarSize }}
            title="Chess Historian avatar (click to zoom). Saved in settings."
            onClick={(e) => { e.stopPropagation(); if (settings?.chessHistorianAvatar) setHistorianAvatarZoomOpen(true); }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && settings?.chessHistorianAvatar) { e.stopPropagation(); setHistorianAvatarZoomOpen(true); } }}
          >
            {settings?.chessHistorianAvatar ? (
              <img src={settings.chessHistorianAvatar} alt="Historian" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-muted-foreground text-xs">♔</div>
            )}
          </div>
          <span>Chess Historian</span>
          <span className="text-xs font-normal text-muted-foreground">Ask about players, games, rivalries — I’ll look it up</span>
        </button>
        {historianOpen && (
          <>
            <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
              <Button variant="outline" size="sm" onClick={() => setHistorianAvatarPickerOpen(true)} title="Generate avatar with AI">
                Generate avatar
              </Button>
              <Button variant="outline" size="sm" onClick={() => historianAvatarInputRef.current?.click()} title="Upload image as avatar">
                Upload avatar
              </Button>
              <input ref={historianAvatarInputRef} type="file" accept="image/png,image/jpeg,image/gif,image/webp" className="hidden" onChange={handleHistorianAvatarUpload} />
              <Button variant="outline" size="sm" onClick={requestHistorianPersona} disabled={historianPersonaLoading} title="AI describes its own historian persona">
                {historianPersonaLoading ? '…' : 'Regenerate persona'}
              </Button>
            </div>
            {settings?.chessHistorianPersonaPrompt && (
              <p className="text-xs text-muted-foreground italic mb-2 border-l-2 border-primary/50 pl-2">{settings.chessHistorianPersonaPrompt}</p>
            )}
            <div className="mb-2 text-sm border border-border/50 rounded p-3 bg-muted/20">
              <span className="font-medium text-muted-foreground text-xs uppercase tracking-wide">Chess fact</span>
              <p className="mt-1 text-foreground min-h-[2.5rem]">
                {historianLoading && !historianCurrentFact ? '…' : (historianCurrentFact || 'Every 30s I’ll share a chess history fact when you’re not chatting.')}
              </p>
            </div>
            <div className="max-h-48 overflow-y-auto space-y-2 mb-3 text-sm border border-border/50 rounded p-3 bg-muted/10">
              {historianMessages.length === 0 && (
                <p className="text-muted-foreground italic">Ask about players, games, or any chess history.</p>
              )}
              {historianMessages.map((msg, i) => (
                <div key={i} className={msg.role === 'user' ? 'text-right' : ''}>
                  <span className="font-medium text-muted-foreground">{msg.role === 'user' ? 'You' : 'Historian'}: </span>
                  {msg.role === 'assistant' ? (
                    msg.isError ? (
                      <div className="inline-block text-left">
                        <p className="text-muted-foreground">{msg.content}</p>
                        <Button variant="outline" size="sm" className="mt-1" onClick={() => retryHistorianAt(i, historianMessages)} disabled={historianLoading}>
                          Retry
                        </Button>
                      </div>
                    ) : (
                      <div className="prose prose-sm dark:prose-invert max-w-none inline-block text-left">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content || ''}</ReactMarkdown>
                      </div>
                    )
                  ) : (
                    <span className="whitespace-pre-wrap">{msg.content}</span>
                  )}
                  {msg.pgn && (
                    <div className="mt-1">
                      <Button variant="default" size="sm" onClick={() => loadImportedGame(msg.pgn)}>Load this game on board</Button>
                    </div>
                  )}
                </div>
              ))}
              {historianLoading && historianMessages.length > 0 && <p className="text-muted-foreground italic">…</p>}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 rounded border border-border bg-background px-3 py-2 text-sm"
                placeholder="Ask about players, games, or history…"
                value={historianInput}
                onChange={(e) => setHistorianInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') sendHistorianMessage(historianInput); }}
              />
              <Button size="sm" onClick={() => sendHistorianMessage(historianInput)} disabled={historianLoading}>Send</Button>
            </div>
          </>
        )}
      </section>

      {historianAvatarPickerOpen && (
        <SimpleChatImageButton
          defaultOpen={true}
          onImageGenerated={(url) => {
            updateSettings({ chessHistorianAvatar: url });
            setHistorianAvatarPickerOpen(false);
          }}
          onClose={() => setHistorianAvatarPickerOpen(false)}
        />
      )}

      {historianAvatarZoomOpen && settings?.chessHistorianAvatar && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setHistorianAvatarZoomOpen(false)}
          role="dialog"
          aria-label="Historian avatar zoomed"
        >
          <div className="relative max-w-2xl max-h-[85vh] flex items-center justify-center" onClick={(e) => e.stopPropagation()}>
            <img src={settings.chessHistorianAvatar} alt="Chess Historian avatar" className="max-w-full max-h-[85vh] w-auto h-auto object-contain rounded-lg shadow-xl" />
            <Button variant="secondary" size="sm" className="absolute top-2 right-2" onClick={() => setHistorianAvatarZoomOpen(false)}>Close</Button>
          </div>
        </div>
      )}

      {isGameOver && gameOverLabel && !inReplayMode && (
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
            <div className="mt-3 text-sm text-left bg-muted/50 rounded-lg p-3 border border-border prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:first:mt-0 prose-p:last:mb-0">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{gameOverCommentary}</ReactMarkdown>
            </div>
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
        {(showEvalBar || (inReplayMode && takeOverMode)) && (
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
                ? (displayEvalCp != null
                    ? (userPlaysWhite ? `${displayEvalCp > 0 ? '+' : ''}${(displayEvalCp / 100).toFixed(2)}` : `${displayEvalCp < 0 ? '+' : ''}${(-displayEvalCp / 100).toFixed(2)}`)
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
            <div className="flex flex-wrap items-center gap-2 w-full max-w-[var(--chess-board-size)]">
              <Button variant="outline" size="sm" onClick={() => { setGameAnalysisFull(null); setTakeOverMode(false); setBranchFen(null); setBranchEval(null); setBranchCandidates([]); }}>Exit analysis</Button>
              {!takeOverMode ? (
                <Button variant="default" size="sm" onClick={() => setTakeOverMode(true)}>Take over</Button>
              ) : (
                <>
                  <Button variant="secondary" size="sm" onClick={() => { setBranchFen(null); setBranchEval(null); setBranchCandidates([]); setBranchSelectedSquare(null); }}>Back to position</Button>
                  <Button variant="ghost" size="sm" onClick={() => { setTakeOverMode(false); setBranchFen(null); setBranchEval(null); setBranchCandidates([]); }}>Exit take over</Button>
                </>
              )}
              <span className="text-xs text-muted-foreground font-medium">
                Position {clampedReplayIndex + 1} of {replayMoveCount}
              </span>
              <Button variant="ghost" size="sm" onClick={() => { setAnalysisReplayIndex(0); setBranchFen(null); }} disabled={clampedReplayIndex <= 0} title="First position">First</Button>
              <Button variant="ghost" size="sm" onClick={() => { setAnalysisReplayIndex((i) => Math.max(0, i - 1)); setBranchFen(null); }} disabled={clampedReplayIndex <= 0}>← Prev</Button>
              <Button variant="ghost" size="sm" onClick={() => { setAnalysisReplayIndex((i) => Math.min(replayMoveCount - 1, i + 1)); setBranchFen(null); }} disabled={clampedReplayIndex >= replayMoveCount - 1}>Next →</Button>
              <Button variant="ghost" size="sm" onClick={() => { setAnalysisReplayIndex(replayMoveCount - 1); setBranchFen(null); }} disabled={clampedReplayIndex >= replayMoveCount - 1} title="Last position">Last</Button>
              {hasMinimalAnalysis && (
                <Button variant="default" size="sm" onClick={() => requestGameAnalysis()} disabled={loadingAnalysis} title="Engine evals for every position + AI annotation per move (one request per game; may take a moment).">
                  {loadingAnalysis ? 'Analyzing…' : 'Generate engine + AI analysis'}
                </Button>
              )}
            </div>
          )}
          {loadedGameMeta && (loadedGameMeta.whiteName || loadedGameMeta.blackName) && (
            <div className="w-full max-w-[var(--chess-board-size)] text-xs text-muted-foreground py-0.5 text-center">
              {userPlaysWhite ? (loadedGameMeta.blackName || '—') : (loadedGameMeta.whiteName || '—')}
              {userPlaysWhite ? (loadedGameMeta.blackElo ? ` (${loadedGameMeta.blackElo})` : '') : (loadedGameMeta.whiteElo ? ` (${loadedGameMeta.whiteElo})` : '')}
            </div>
          )}
          <div className="chess-board-frame">
            <Chessboard
              key={displayFen || fen}
              position={displayBoardPosition}
              onDrop={inReplayMode && takeOverMode ? onReplayDrop : (inReplayMode ? undefined : onDrop)}
              onSquareClick={inReplayMode && takeOverMode ? onReplaySquareClick : (inReplayMode ? undefined : onSquareClick)}
              squareStyles={inReplayMode && takeOverMode ? (branchSelectedSquare ? { [branchSelectedSquare]: { backgroundColor: 'rgba(255, 200, 100, 0.6)' } } : {}) : (inReplayMode ? {} : (selectedSquare ? { [selectedSquare]: { backgroundColor: 'rgba(255, 200, 100, 0.6)' } } : {}))}
              orientation={userPlaysWhite ? 'white' : 'black'}
              draggable={(inReplayMode && takeOverMode) || (!inReplayMode && isUserTurn && !isGameOver && !thinking)}
              width={BOARD_SIZE}
              showNotation
              lightSquareStyle={{ backgroundColor: 'var(--chess-board-light)' }}
              darkSquareStyle={{ backgroundColor: 'var(--chess-board-dark)' }}
            />
          </div>
          {loadedGameMeta && (loadedGameMeta.whiteName || loadedGameMeta.blackName) && (
            <div className="w-full max-w-[var(--chess-board-size)] text-xs text-muted-foreground py-0.5 text-center">
              {userPlaysWhite ? (loadedGameMeta.whiteName || '—') : (loadedGameMeta.blackName || '—')}
              {userPlaysWhite ? (loadedGameMeta.whiteElo ? ` (${loadedGameMeta.whiteElo})` : '') : (loadedGameMeta.blackElo ? ` (${loadedGameMeta.blackElo})` : '')}
              {loadedGameMeta.userSide != null ? ' (You)' : ''}
            </div>
          )}
          {inReplayMode && (
            <div className="chess-panel w-[var(--chess-board-size)] max-w-full p-2 text-xs text-muted-foreground border border-border rounded space-y-2">
              <div className="font-medium text-foreground">Annotation</div>
              <div className="min-h-[1.5rem] prose prose-sm dark:prose-invert max-w-none prose-p:my-0.5 prose-p:first:mt-0 prose-p:last:mb-0">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{currentStepCommentary || '—'}</ReactMarkdown>
              </div>
              {currentStepMove?.judgment && currentStepMove.judgment !== 'best' && (
                <div className="space-y-1 pt-1 border-t border-border">
                  <div className="font-medium text-foreground">{currentStepMove.judgment}</div>
                  {currentStepMove.best_move_san && (
                    <div>
                      <span className="text-muted-foreground">Best: </span>
                      <span className="text-foreground">{currentStepMove.best_move_san}</span>
                      {currentStepMove.best_move_pv_san?.length > 0 && (
                        <div className="text-muted-foreground mt-0.5">→ {currentStepMove.best_move_pv_san.join(' → ')}</div>
                      )}
                    </div>
                  )}
                  {currentStepMove.continuation_pv_san?.length > 0 && (
                    <div>
                      <span className="text-muted-foreground">After your move: </span>
                      <span className="text-foreground">{currentStepMove.continuation_pv_san.join(' → ')}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          {thinking && !inReplayMode && (
            <p className="text-sm text-muted-foreground mt-1">AI is thinking...</p>
          )}

          {((showEngineThinking && candidates.length > 0) || (inReplayMode && takeOverMode && displayCandidates.length > 0)) && (
            <section className="chess-panel w-[var(--chess-board-size)] max-w-full p-3">
              <h3 className="chess-panel-title mb-2">Engine lines</h3>
              <div className="space-y-2 max-h-44 overflow-y-auto">
                {(inReplayMode && takeOverMode ? displayCandidates : candidates).map((c, i) => (
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
                {effectiveAnalysis.moves.map((m, i) => {
                  const moveNum = m.move_index === 0 ? 'Start' : `${Math.ceil(m.move_index / 2)}${m.move_index % 2 === 1 ? '.' : '...'}`;
                  const label = m.move_index === 0 ? 'Start' : `${moveNum} ${m.san}`;
                  const evalPawns = m.evaluation_cp != null ? (m.evaluation_cp > 0 ? '+' : '') + (m.evaluation_cp / 100).toFixed(2) : '—';
                  const active = i === clampedReplayIndex;
                  const annotation = m.commentary && String(m.commentary).trim() ? m.commentary : null;
                  const judgment = m.judgment && m.judgment !== 'best' ? m.judgment : null;
                  return (
                    <button
                      key={i}
                      type="button"
                      onClick={() => { setAnalysisReplayIndex(i); setBranchFen(null); }}
                      className={`w-full text-left px-2 py-1.5 rounded border transition-colors ${active ? 'bg-primary/20 border-primary' : 'border-transparent hover:bg-muted/50'}`}
                    >
                      <div className="flex flex-wrap items-baseline gap-x-2">
                        <span className="font-medium">{label}</span>
                        <span className="text-muted-foreground">{evalPawns}</span>
                        {judgment && (
                          <span className={`text-xs px-1.5 py-0 rounded ${judgment === 'blunder' ? 'bg-red-500/20 text-red-700 dark:text-red-400' : judgment === 'mistake' ? 'bg-orange-500/20 text-orange-700 dark:text-orange-400' : 'bg-amber-500/20 text-amber-700 dark:text-amber-400'}`}>
                            {judgment}
                          </span>
                        )}
                      </div>
                      {annotation && (
                        <div className="text-muted-foreground mt-0.5 leading-snug prose prose-sm dark:prose-invert max-w-none prose-p:my-0 prose-p:first:mt-0 prose-p:last:mb-0">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{annotation}</ReactMarkdown>
                        </div>
                      )}
                      {judgment && m.best_move_san && (
                        <div className="text-muted-foreground mt-0.5 text-[11px]">
                          Best: {m.best_move_san}
                          {m.best_move_pv_san?.length > 0 && ` → ${m.best_move_pv_san.slice(0, 5).join(' → ')}${(m.best_move_pv_san.length > 5 ? '…' : '')}`}
                        </div>
                      )}
                      {judgment && m.continuation_pv_san?.length > 0 && (
                        <div className="text-muted-foreground mt-0.5 text-[11px]">
                          After your move: {m.continuation_pv_san.slice(0, 5).join(' → ')}{m.continuation_pv_san.length > 5 ? '…' : ''}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </section>
          )}
          {showCommentary && !inReplayMode && (
            <section className="chess-panel p-3">
              <h3 className="chess-panel-title mb-2">Commentary</h3>
              <div className="text-sm text-foreground leading-relaxed min-h-[2.5rem] prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:first:mt-0 prose-p:last:mb-0">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{commentary || (isGameOver ? gameResult : '—')}</ReactMarkdown>
              </div>
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
                      <span className="text-muted-foreground">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: ({ children }) => <span className="inline">{children}</span> }}>{`— ${m.commentary}`}</ReactMarkdown>
                      </span>
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
