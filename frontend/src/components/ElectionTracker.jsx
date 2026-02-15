import { useEffect, useRef, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import './ElectionTracker.css';
import ElectionMap from './ElectionMap';
import PollTrends from './PollTrends';
import PollTable from './PollTable';
import ElectionNews from './ElectionNews';
import { Vote, TrendingUp, Table, Newspaper, MessageSquare, X, BarChart3 } from 'lucide-react';

const ELECTION_RACE_TYPE_KEY = 'election_race_type';
const VALID_RACE_TYPES = ['senate', 'governor', 'house', 'approval', 'generic_ballot'];
const MAP_RACE_TYPES = ['senate', 'governor', 'house'];

function getInitialRaceType() {
    if (typeof window === 'undefined') return 'senate';
    try {
        const saved = window.localStorage.getItem(ELECTION_RACE_TYPE_KEY);
        if (saved && VALID_RACE_TYPES.includes(saved)) return saved;
    } catch (_) { }
    return 'senate';
}

function getInitialMapRaceType() {
    if (typeof window === 'undefined') return 'senate';
    try {
        const saved = window.localStorage.getItem(ELECTION_RACE_TYPE_KEY);
        if (saved && MAP_RACE_TYPES.includes(saved)) return saved;
    } catch (_) { }
    return 'senate';
}

const ElectionTracker = () => {
    const { PRIMARY_API_URL, primaryModel } = useApp();
    const [activeTab, setActiveTab] = useState('polls');
    const [raceType, setRaceType] = useState(getInitialRaceType);
    const [mapRaceType, setMapRaceType] = useState(getInitialMapRaceType);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [trendData, setTrendData] = useState([]);
    const [trendLoading, setTrendLoading] = useState(false);
    const [trendError, setTrendError] = useState(null);
    const [aiOpen, setAiOpen] = useState(false);
    const [aiMessages, setAiMessages] = useState([]);
    const [aiInput, setAiInput] = useState('');
    const [aiLoading, setAiLoading] = useState(false);
    const [aiError, setAiError] = useState(null);
    const [suggestedQuestions, setSuggestedQuestions] = useState([]);
    const [questionsLoading, setQuestionsLoading] = useState(false);
    const [mapData, setMapData] = useState({ user_data: [], scraped_averages: {} });
    const [mapLoading, setMapLoading] = useState(false);
    const [mapError, setMapError] = useState(null);
    const [refreshing, setRefreshing] = useState(false);
    const [simResult, setSimResult] = useState(null);
    const [simLoading, setSimLoading] = useState(false);
    const [simError, setSimError] = useState(null);
    const [calibrationEntries, setCalibrationEntries] = useState([]);
    const [calibrationOpen, setCalibrationOpen] = useState(false);
    const [useCalibration, setUseCalibration] = useState(true);
    const [calibrationWeightPct, setCalibrationWeightPct] = useState(25); // 0–100: how much swing influences baseline (25 = light touch default)
    const [simRunMeta, setSimRunMeta] = useState(null); // { use_calibration, calibration_n, ... } from last run
    const [simCompletedAt, setSimCompletedAt] = useState(null); // ISO string or null; when last run finished
    const [simJustUpdated, setSimJustUpdated] = useState(false); // true briefly after run so we can show "Results updated"
    const [simRunCount, setSimRunCount] = useState(0); // increments each successful run so user sees Run #1, #2, ...
    const [simRunId, setSimRunId] = useState(null); // server-generated run_id so user can verify backend actually ran
    const [savedSimRuns, setSavedSimRuns] = useState([]); // { id, label, savedAt, race_type, run_id, result, meta }[]
    const [simNCount, setSimNCount] = useState(10000); // 10000 | 25000 | 50000
    const [calibrationForm, setCalibrationForm] = useState({ label: '', type: 'special', state: '', date: '', dem_actual_pct: '', poll_avg_pct: '', weight: '1', note: '' });
    const [scrapeProgress, setScrapeProgress] = useState(null); // { current, total, message } while scraping
    const requestIdRef = useRef(0);
    const abortRef = useRef(null);

    const SAVED_RUNS_KEY = 'election_sim_saved_runs';
    const LAST_RUN_KEY = 'election_sim_last_run';
    const MAX_SAVED_RUNS = 20;

    const loadSavedRuns = () => {
        try {
            const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(SAVED_RUNS_KEY) : null;
            const list = raw ? JSON.parse(raw) : [];
            setSavedSimRuns(Array.isArray(list) ? list : []);
        } catch {
            setSavedSimRuns([]);
        }
    };

    const saveCurrentRun = (label = '') => {
        const r = simResult;
        if (!r || !r.state_win_probs) return;
        const entry = {
            id: `saved-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
            label: (label || `Run ${mapRaceType} ${new Date().toLocaleDateString()}`).trim(),
            savedAt: new Date().toISOString(),
            race_type: mapRaceType,
            run_id: simRunId,
            result: { state_win_probs: r.state_win_probs, state_stats: r.state_stats, d_majority_pct: r.d_majority_pct, n_simulations: r.n_simulations, races_included: r.races_included, elapsed_sec: r.elapsed_sec },
            meta: simRunMeta ? { ...simRunMeta } : null,
        };
        const next = [entry, ...savedSimRuns].slice(0, MAX_SAVED_RUNS);
        setSavedSimRuns(next);
        try {
            localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
        } catch (_) {}
    };

    const loadSavedRun = (entry) => {
        setSimResult(entry.result ? { ...entry.result } : null);
        setSimRunMeta(entry.meta || null);
        setSimRunId(entry.run_id || null);
        setSimCompletedAt(entry.savedAt || null);
    };

    const deleteSavedRun = (id) => {
        const next = savedSimRuns.filter((e) => e.id !== id);
        setSavedSimRuns(next);
        try {
            localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(next));
        } catch (_) {}
    };

    // Load saved runs and optionally restore last run on mount
    useEffect(() => {
        loadSavedRuns();
        try {
            const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(LAST_RUN_KEY) : null;
            if (raw) {
                const last = JSON.parse(raw);
                if (last && last.result && last.result.state_win_probs) {
                    setSimResult(last.result);
                    setSimRunMeta(last.meta || null);
                    setSimRunId(last.run_id || null);
                    setSimCompletedAt(last.completed_at || null);
                }
            }
        } catch (_) {}
    }, []);

    const fetchPollingData = async (signal, requestId) => {
        setLoading(true);
        setError(null);
        try {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const url = `${baseUrl}/election/polls?race_type=${raceType}`;

            const response = await fetch(url, { signal });
            if (!response.ok) {
                throw new Error('Failed to fetch polling data');
            }
            const result = await response.json();

            if (requestId !== requestIdRef.current) return;

            if (result.error) {
                setError(result.error);
                setData(null);
            } else {
                setData(result);
            }
        } catch (err) {
            if (err?.name === 'AbortError') return;
            console.error(err);
            setError('Could not load election data.');
            setData(null);
        } finally {
            if (requestId === requestIdRef.current) {
                setLoading(false);
            }
        }
    };

    const fetchTrendData = async (signal, requestId) => {
        setTrendLoading(true);
        setTrendError(null);
        try {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const url = `${baseUrl}/election/trends?race_type=${raceType}&days=90`;
            const response = await fetch(url, { signal });
            if (!response.ok) throw new Error('Failed to fetch trends');
            const result = await response.json();
            if (requestId !== requestIdRef.current) return;
            setTrendData(result.trend || []);
        } catch (err) {
            if (err?.name === 'AbortError') return;
            console.error(err);
            setTrendError('Could not load trend data.');
            setTrendData([]);
        } finally {
            if (requestId === requestIdRef.current) setTrendLoading(false);
        }
    };

    useEffect(() => {
        if (activeTab === 'news') return;

        if (activeTab === 'map') {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            setMapLoading(true);
            setMapError(null);
            fetch(`${baseUrl}/election/map?race_type=${mapRaceType}`)
                .then((r) => {
                    if (!r.ok) throw new Error('Map request failed');
                    return r.json();
                })
                .then((d) => {
                    setMapData(d);
                    setMapError(null);
                })
                .catch((err) => {
                    console.error('Map fetch failed', err);
                    setMapData({ user_data: [], scraped_averages: {} });
                    setMapError('Could not load map data. Check console.');
                })
                .finally(() => setMapLoading(false));
            return;
        }

        // Polls or Trends tab
        if (abortRef.current) {
            abortRef.current.abort();
        }
        const controller = new AbortController();
        abortRef.current = controller;
        const requestId = ++requestIdRef.current;
        if (activeTab === 'polls') {
            fetchPollingData(controller.signal, requestId);
        } else if (activeTab === 'trends') {
            fetchTrendData(controller.signal, requestId);
        } else if (activeTab === 'simulations' && ['senate', 'governor', 'house'].includes(mapRaceType)) {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            fetch(`${baseUrl}/election/map?race_type=${mapRaceType}`)
                .then((r) => r.json())
                .then((d) => { setMapData(d); setMapError(null); })
                .catch(() => setMapError('Could not load map data.'));
            fetch(`${baseUrl}/election/simulation/results?race_type=${mapRaceType}`)
                .then((r) => r.json())
                .then((d) => {
                    if (d.result) setSimResult(d.result);
                    else setSimResult(null);
                    setSimError(d.error || null);
                })
                .catch(() => { setSimResult(null); setSimError('Could not load simulation results.'); });
            fetch(`${baseUrl}/election/simulation/calibration`)
                .then((r) => r.json())
                .then((d) => setCalibrationEntries(d.entries || []))
                .catch(() => setCalibrationEntries([]));
        }

        return () => {
            controller.abort();
        };
    }, [raceType, mapRaceType, activeTab]);

    const generateQuestions = async (regenerate = false) => {
        if (!data?.polls?.length || !primaryModel || questionsLoading) return;
        setQuestionsLoading(true);
        try {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const res = await fetch(`${baseUrl}/election/questions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    race_type: raceType,
                    polls: (data.polls || []).slice(0, 20),
                    metadata: data.metadata || {},
                    model: primaryModel,
                    regenerate
                })
            });
            const result = await res.json();
            setSuggestedQuestions(result.questions || []);
        } catch (err) {
            console.error('Failed to generate questions:', err);
        } finally {
            setQuestionsLoading(false);
        }
    };

    useEffect(() => {
        if (data?.polls?.length > 0 && primaryModel) generateQuestions(false);
    }, [data?.polls?.length, raceType]);

    const sendAiMessage = async (messageText) => {
        const trimmed = messageText.trim();
        if (!trimmed || aiLoading) return;

        if (!primaryModel) {
            setAiError('No model selected. Choose a model in Settings before using AI insights.');
            return;
        }

        setAiError(null);
        setAiLoading(true);
        setAiInput('');

        setAiMessages((prev) => [...prev, { role: 'user', content: trimmed }]);

        try {
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const response = await fetch(`${baseUrl}/election/assistant`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: trimmed,
                    race_type: raceType,
                    metadata: data?.metadata || null,
                    polls: data?.polls || null,
                    model: primaryModel,
                    temperature: 0.2,
                    max_tokens: 900
                })
            });

            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || 'Failed to fetch AI response');
            }

            const result = await response.json();
            setAiMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: result.answer || 'No response.',
                    toolSteps: result.tool_steps || []
                }
            ]);
        } catch (err) {
            console.error(err);
            setAiError('Could not load AI response. Check that the AI endpoint is configured.');
        } finally {
            setAiLoading(false);
        }
    };

    const handleAskAI = () => {
        setAiOpen((prev) => !prev);
        setAiError(null);
    };

    const handleQuickInsight = () => {
        sendAiMessage('Give me 3 concise insights about the latest polling and who is leading.');
    };

    const formatTimeAgo = (iso) => {
        if (!iso) return '';
        try {
            const d = new Date(iso);
            const sec = Math.floor((Date.now() - d) / 1000);
            if (sec < 60) return 'just now';
            if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
            if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
            return `${Math.floor(sec / 86400)}d ago`;
        } catch { return ''; }
    };

    const formatCompletedAt = (iso) => {
        if (!iso) return '';
        try {
            return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch { return ''; }
    };

    const renderContent = () => {
        if (activeTab === 'news') return <ElectionNews />;
        if (activeTab === 'simulations') {
            if (!['senate', 'governor', 'house'].includes(mapRaceType)) {
                return (
                    <div style={{ padding: '1.5rem', color: 'var(--text-secondary)' }}>
                        Simulations are available for Senate, Governor, and House. Use the &quot;Race type&quot; dropdown above to choose one, then click Run.
                    </div>
                );
            }
            const baseUrl = PRIMARY_API_URL || getBackendUrl();
            const r = simResult || {};
            const winProbs = r.state_win_probs || {};
            const raceLabel = mapRaceType === 'senate' ? '2026 Senate' : mapRaceType === 'governor' ? '2026 Governor' : '2026 House';
            const statesIncluded = Object.keys(winProbs || {}).sort();
            return (
                <div className="simulations-panel">
                    <div className="simulations-header">
                        <h3>{raceLabel} races — simulation</h3>
                        <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                            Runs a Monte Carlo over <strong>all states that have polls</strong> for this race type. Use the &quot;Race type&quot; dropdown above to choose Senate, Governor, or House, then click Run.
                        </p>
                        {statesIncluded.length > 0 && (
                            <p style={{ margin: '0.5rem 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                States in this run ({statesIncluded.length}): {statesIncluded.join(', ')}
                            </p>
                        )}
                    </div>
                    <div style={{ marginBottom: '1rem' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center', marginBottom: '0.75rem' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', userSelect: 'none' }}>
                                <input
                                    type="checkbox"
                                    checked={useCalibration}
                                    onChange={(e) => setUseCalibration(e.target.checked)}
                                    style={{ width: '1rem', height: '1rem', accentColor: 'var(--primary)' }}
                                />
                                <span>Use calibration (swing from special/off-year races)</span>
                            </label>
                            {useCalibration && (
                                <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem 1rem', fontSize: '0.9rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <label style={{ whiteSpace: 'nowrap' }}>Swing weight:</label>
                                        <input
                                            type="range"
                                            min={0}
                                            max={100}
                                            value={calibrationWeightPct}
                                            onChange={(e) => setCalibrationWeightPct(Number(e.target.value))}
                                            style={{ width: '100px', accentColor: 'var(--primary)' }}
                                            title="0 = off, 25% = light, 50% = half, 100% = full"
                                        />
                                        <span style={{ minWidth: '2.5rem', fontWeight: 600 }}>{calibrationWeightPct}%</span>
                                    </div>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', maxWidth: '320px' }}>
                                        {calibrationWeightPct === 0 ? 'Swing ignored; no shift applied.' : calibrationWeightPct <= 25 ? 'Light: raw swing scaled to 25%, applied to every state baseline.' : calibrationWeightPct <= 50 ? 'Moderate: 50% of swing applied to state baselines.' : 'Full: 100% of swing applied to state baselines (results change clearly).'}
                                    </span>
                                </div>
                            )}
                            <button
                                type="button"
                                className="election-sim-run-btn"
                                disabled={simLoading}
                                onClick={async () => {
                                    const runStartedAt = Date.now();
                                    const MIN_LOADING_MS = 1800;
                                    setSimLoading(true);
                                    setSimError(null);
                                    const calWeight = calibrationWeightPct / 100;
                                    const url = `${baseUrl}/election/simulation/run?race_type=${mapRaceType}&n_simulations=${simNCount}&use_calibration=${useCalibration}&calibration_weight=${calWeight}`;
                                    const applyResultAndHideLoading = () => {
                                        const elapsed = Date.now() - runStartedAt;
                                        const wait = Math.max(0, MIN_LOADING_MS - elapsed);
                                        if (wait > 0) {
                                            setTimeout(() => setSimLoading(false), wait);
                                        } else {
                                            setSimLoading(false);
                                        }
                                    };
                                    try {
                                        const res = await fetch(url, { method: 'POST', cache: 'no-store', headers: { Accept: 'application/json' } });
                                        const text = await res.text();
                                        const json = text ? (() => { try { return JSON.parse(text); } catch { return {}; } })() : {};
                                        if (!res.ok) {
                                            setSimError(json.detail || json.error || `Request failed (${res.status})`);
                                            applyResultAndHideLoading();
                                        } else if (json.result != null) {
                                            setSimResult({ ...json.result });
                                            setSimCompletedAt(json.completed_at || new Date().toISOString());
                                            setSimRunId(json.run_id ?? null);
                                            if (json.use_calibration != null) {
                                                setSimRunMeta({
                                                    use_calibration: json.use_calibration,
                                                    calibration_n: json.calibration_n,
                                                    calibration_swing_pts: json.calibration_swing_pts,
                                                    calibration_weight: json.calibration_weight,
                                                    calibration_swing_effective: json.calibration_swing_effective,
                                                    calibration_combined_shift: json.calibration_combined_shift,
                                                    calibration_components: json.calibration_components || {},
                                                    calibration_weights_used: json.calibration_weights_used || {},
                                                });
                                            }
                                            setSimRunCount((c) => c + 1);
                                            setSimJustUpdated(true);
                                            setTimeout(() => setSimJustUpdated(false), 4000);
                                            try {
                                                const lastRun = { race_type: mapRaceType, run_id: json.run_id, result: json.result, meta: json.use_calibration != null ? { use_calibration: json.use_calibration, calibration_n: json.calibration_n, calibration_swing_pts: json.calibration_swing_pts, calibration_weight: json.calibration_weight, calibration_swing_effective: json.calibration_swing_effective, calibration_components: json.calibration_components || {}, calibration_weights_used: json.calibration_weights_used || {} } : null, completed_at: json.completed_at };
                                                localStorage.setItem(LAST_RUN_KEY, JSON.stringify(lastRun));
                                            } catch (_) {}
                                            applyResultAndHideLoading();
                                        } else {
                                            setSimError(json.error || 'No result in response.');
                                            applyResultAndHideLoading();
                                        }
                                    } catch (e) {
                                        setSimError(e?.message || 'Run failed.');
                                        applyResultAndHideLoading();
                                    }
                                }}
                            >
                                {simLoading ? (
                                    <>
                                        <span className="sim-spinner" style={{ display: 'inline-block', width: '1rem', height: '1rem', marginRight: '0.5rem', verticalAlign: 'middle', border: '2px solid var(--border-color)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'sim-spin 0.8s linear infinite' }} aria-hidden />
                                        Running simulation…
                                    </>
                                ) : (
                                    `Run simulation (${(r.n_simulations || simNCount).toLocaleString()})`
                                )}
                            </button>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                <span>Sims:</span>
                                <select
                                    value={simNCount}
                                    onChange={(e) => setSimNCount(Number(e.target.value))}
                                    style={{ padding: '0.25rem 0.5rem', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)' }}
                                >
                                    <option value={10000}>10,000</option>
                                    <option value={25000}>25,000</option>
                                    <option value={50000}>50,000</option>
                                </select>
                            </label>
                            {r.state_win_probs && Object.keys(r.state_win_probs).length > 0 && (
                                <button
                                    type="button"
                                    onClick={() => {
                                        const label = window.prompt('Label this run (optional)', `Run ${mapRaceType} ${new Date().toLocaleDateString()}`);
                                        if (label !== null) saveCurrentRun(label);
                                    }}
                                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.85rem', borderRadius: 8, border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)', cursor: 'pointer' }}
                                >
                                    Save this run
                                </button>
                            )}
                            <p style={{ margin: '0.35rem 0 0', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                First run after server start may take 10–15s (cold start); later runs are fast. Each run runs the full simulation.
                            </p>
                        </div>
                        {/* One results block: loading XOR result. Same flow every run. */}
                        {simLoading ? (
                            <div style={{ marginTop: '1rem', padding: '1.25rem', background: 'var(--bg-tertiary)', borderRadius: '12px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <span style={{ display: 'inline-block', width: '28px', height: '28px', flexShrink: 0, border: '3px solid var(--border-color)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'sim-spin 0.8s linear infinite' }} aria-hidden />
                                <div>
                                    <p style={{ margin: 0, fontWeight: 600, fontSize: '1rem' }}>Running simulation…</p>
                                    <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Results will appear below when this run finishes.</p>
                                </div>
                            </div>
                        ) : (
                            <>
                                {simJustUpdated && (
                                    <p style={{ margin: '0.5rem 0 0', fontSize: '0.9rem', fontWeight: 600, color: 'var(--primary)' }}>
                                        Run #{simRunCount} complete
                                        {r.elapsed_sec != null && (
                                            <span style={{ fontWeight: 500, color: 'var(--text-secondary)', marginLeft: '0.35rem' }}>
                                                (backend {r.elapsed_sec}s)
                                            </span>
                                        )}
                                        {simCompletedAt && (
                                            <span style={{ fontWeight: 400, color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>
                                                at {formatCompletedAt(simCompletedAt)}
                                            </span>
                                        )}
                                        . Table below is from this run.
                                        {simRunId && (
                                            <span style={{ display: 'block', marginTop: '0.35rem', fontSize: '0.8rem', fontWeight: 400, color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                                                Server run ID: {simRunId}
                                            </span>
                                        )}
                                    </p>
                                )}
                                {simCompletedAt && !simJustUpdated && (
                                    <p style={{ margin: '0.5rem 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                        Run #{simRunCount} · {formatTimeAgo(simCompletedAt)}
                                        {simRunId && <span style={{ marginLeft: '0.5rem', fontFamily: 'monospace' }}>· {simRunId}</span>}
                                    </p>
                                )}
                                {r.n_simulations ? (
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem 1rem', alignItems: 'center', fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                        <span className="simulations-stats">
                                            <strong>{r.n_simulations.toLocaleString()} sims</strong> · {r.races_included} states · {r.elapsed_sec}s
                                            {r.d_majority_pct != null && ` · D majority ${r.d_majority_pct}%`}
                                        </span>
                                        {simRunMeta?.use_calibration && (simRunMeta.calibration_components && Object.keys(simRunMeta.calibration_components).length > 0 || simRunMeta.calibration_n > 0 || simRunMeta.calibration_swing_effective != null) ? (
                                            <span style={{ padding: '0.35rem 0.6rem', background: 'var(--bg-tertiary)', borderRadius: '6px', fontSize: '0.85rem' }}>
                                                {(simRunMeta.calibration_components && Object.keys(simRunMeta.calibration_components).length > 0) ? (
                                                    <>
                                                        <strong>
                                                            {simRunMeta.calibration_swing_effective != null && simRunMeta.calibration_swing_effective !== 0
                                                                ? (simRunMeta.calibration_swing_effective >= 0 ? `D+${simRunMeta.calibration_swing_effective}` : `R+${Math.abs(simRunMeta.calibration_swing_effective)}`) + ' pts applied'
                                                                : '0 pts applied'}
                                                        </strong>
                                                        {typeof simRunMeta.calibration_weight === 'number' && simRunMeta.calibration_weight < 1 && (
                                                            <span style={{ color: 'var(--text-secondary)' }}> at {Math.round((simRunMeta.calibration_weight ?? 0) * 100)}% weight</span>
                                                        )}
                                                        <span style={{ color: 'var(--text-secondary)' }}>
                                                            {' · From: '}
                                                            {simRunMeta.calibration_components.generic_ballot_shift != null && (
                                                                <>GB {simRunMeta.calibration_components.generic_ballot_shift >= 0 ? `D+${simRunMeta.calibration_components.generic_ballot_shift}` : `R+${Math.abs(simRunMeta.calibration_components.generic_ballot_shift)}`}</>
                                                            )}
                                                            {simRunMeta.calibration_components.approval_shift != null && (
                                                                <>, approval {simRunMeta.calibration_components.approval_net != null ? `${simRunMeta.calibration_components.approval_net >= 0 ? '+' : ''}${simRunMeta.calibration_components.approval_net} net` : ''} → {simRunMeta.calibration_components.approval_shift >= 0 ? `D+${simRunMeta.calibration_components.approval_shift}` : `R+${Math.abs(simRunMeta.calibration_components.approval_shift)}`}</>
                                                            )}
                                                            {simRunMeta.calibration_components.special_election_swing_pts != null && (
                                                                <>, special {simRunMeta.calibration_components.special_election_swing_pts >= 0 ? `D+${simRunMeta.calibration_components.special_election_swing_pts}` : `R+${Math.abs(simRunMeta.calibration_components.special_election_swing_pts)}`}</>
                                                            )}
                                                        </span>
                                                    </>
                                                ) : (
                                                    <>
                                                        <strong>{simRunMeta.calibration_swing_effective >= 0 ? `D+${simRunMeta.calibration_swing_effective}` : `R+${Math.abs(simRunMeta.calibration_swing_effective)}`} pts applied</strong>
                                                        {typeof simRunMeta.calibration_weight === 'number' && simRunMeta.calibration_weight < 1 && (
                                                            <span style={{ color: 'var(--text-secondary)' }}> at {Math.round((simRunMeta.calibration_weight ?? 0) * 100)}% weight</span>
                                                        )}
                                                    </>
                                                )}
                                            </span>
                                        ) : simRunMeta?.use_calibration && calibrationEntries.length === 0 && !(simRunMeta.calibration_components && Object.keys(simRunMeta.calibration_components).length > 0) ? (
                                            <span style={{ fontStyle: 'italic' }}>Calibration on but no data — add/scrape below.</span>
                                        ) : simRunMeta?.use_calibration === false ? (
                                            <span style={{ fontStyle: 'italic' }}>Calibration off</span>
                                        ) : null}
                                    </div>
                                ) : (
                                    <p style={{ margin: '1rem 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>No results yet. Click Run simulation above.</p>
                                )}
                            </>
                        )}
                    </div>
                    {simError && <div className="error-message" style={{ marginBottom: '1rem' }}>{simError}</div>}

                    {/* State-by-state table: only when we have a result and not loading (same rule every time) */}
                    {!simLoading && r.state_stats && Object.keys(r.state_stats).length > 0 && (
                        <div className={`simulations-state-table-wrap${simJustUpdated ? ' just-updated' : ''}`}>
                            <h4>State-by-state results ({r.n_simulations?.toLocaleString?.() || '10,000'} sims per state)</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0.5rem' }}>
                                Two-party only. R win % = 100 − D win %; GOP share = 100 − Dem share. Same simulation — shown both ways.
                            </p>
                            <div style={{ overflowX: 'auto', borderRadius: '0 0 12px 12px' }}>
                                <table style={{ width: '100%', minWidth: 520, fontSize: '0.9rem', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ background: 'var(--bg-tertiary)', borderBottom: '1px solid var(--border-color)' }}>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left' }}>State</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>D win %</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>R win %</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Dem share (mean)</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>GOP share (mean)</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Dem 5th %ile</th>
                                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Dem 95th %ile</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(r.state_stats)
                                            .sort(([a], [b]) => a.localeCompare(b))
                                            .map(([state, stats]) => {
                                                const dWin = stats.dem_win_pct != null ? Number(stats.dem_win_pct) : (winProbs[state] != null ? Number(winProbs[state]) : null);
                                                const rWin = dWin != null ? 100 - dWin : null;
                                                const demMean = stats.dem_share_mean != null ? Number(stats.dem_share_mean) : null;
                                                const gopMean = demMean != null ? 100 - demMean : null;
                                                return (
                                                    <tr key={state} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                        <td style={{ padding: '0.5rem 0.75rem', fontWeight: 600 }}>{state}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{dWin != null ? `${dWin.toFixed(1)}%` : '—'}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{rWin != null ? `${rWin.toFixed(1)}%` : '—'}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{demMean != null ? `${demMean.toFixed(1)}%` : '—'}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{gopMean != null ? `${gopMean.toFixed(1)}%` : '—'}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{stats.dem_share_p5 != null ? `${Number(stats.dem_share_p5).toFixed(1)}%` : '—'}</td>
                                                        <td style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>{stats.dem_share_p95 != null ? `${Number(stats.dem_share_p95).toFixed(1)}%` : '—'}</td>
                                                    </tr>
                                                );
                                            })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {savedSimRuns.length > 0 && (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--bg-tertiary)', borderRadius: 12, border: '1px solid var(--border-color)' }}>
                            <h4 style={{ margin: '0 0 0.75rem', fontSize: '1rem' }}>Saved runs ({savedSimRuns.length})</h4>
                            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
                                {savedSimRuns.map((entry) => (
                                    <li key={entry.id} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 0', borderBottom: '1px solid var(--border-color)' }}>
                                        <span style={{ flex: 1, fontSize: '0.9rem' }}>
                                            <strong>{entry.label}</strong>
                                            <span style={{ color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>
                                                {entry.race_type} · {(entry.result?.n_simulations || 0).toLocaleString()} sims
                                                {entry.result?.d_majority_pct != null && ` · D majority ${entry.result.d_majority_pct}%`}
                                            </span>
                                            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '0.5rem' }}>
                                                {entry.savedAt ? new Date(entry.savedAt).toLocaleString() : ''}
                                            </span>
                                        </span>
                                        <button type="button" onClick={() => loadSavedRun(entry)} style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-secondary)', cursor: 'pointer' }}>Load</button>
                                        <button type="button" onClick={() => deleteSavedRun(entry.id)} style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem', borderRadius: 6, border: '1px solid var(--border-color)', background: 'var(--bg-secondary)', color: 'var(--text-secondary)', cursor: 'pointer' }}>Delete</button>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    <div className="calibration-accordion">
                        <button
                            type="button"
                            className="calibration-toggle"
                            onClick={() => setCalibrationOpen((o) => !o)}
                        >
                            Calibration (2024→now swing) — {calibrationEntries.length} race{calibrationEntries.length !== 1 ? 's' : ''} with swing data
                        </button>
                        {calibrationOpen && (
                            <div className="calibration-body">
                                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                    When <strong>Use calibration</strong> is on, the weighted swing from these races is added to a national “fundamentals” baseline; that baseline then gently pulls state poll means (polls stay primary). <strong>Swing weight</strong> above controls how much: <strong>0%</strong> = ignore, <strong>25%</strong> = light touch (default, ~0.2 pt nudge per state), <strong>50%</strong> = about half that effect, <strong>100%</strong> = full swing (~0.5–1 pt nudge). Scrape or add races below, then run.
                                </p>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                                    Only entries with 2024 margin and swing are shown and used. Governor races use state 2024 data; VA House uses VPAP district data. Re-run after loading VA/NJ 2024 districts or adding entries.
                                </p>
                                <p style={{ fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                                    <button
                                        type="button"
                                        className="election-tab election-tab-primary"
                                        style={{ marginRight: '0.5rem' }}
                                        disabled={scrapeProgress !== null}
                                        onClick={async () => {
                                            setScrapeProgress({ current: 0, total: 200, message: 'Starting…' });
                                            try {
                                                const res = await fetch(`${baseUrl}/election/simulation/calibration/ballotpedia-import-stream?since_nov_2025=true`, { method: 'POST' });
                                                if (!res.ok || !res.body) {
                                                    setScrapeProgress(null);
                                                    alert('Ballotpedia import failed.');
                                                    return;
                                                }
                                                const reader = res.body.getReader();
                                                const decoder = new TextDecoder();
                                                let buffer = '';
                                                while (true) {
                                                    const { value, done } = await reader.read();
                                                    if (done) break;
                                                    buffer += decoder.decode(value, { stream: true });
                                                    const lines = buffer.split('\n');
                                                    buffer = lines.pop() || '';
                                                    for (const line of lines) {
                                                        if (line.startsWith('data: ')) {
                                                            try {
                                                                const data = JSON.parse(line.slice(6));
                                                                if (data.type === 'progress') {
                                                                    setScrapeProgress({ current: data.current, total: data.total, message: data.message || '' });
                                                                } else if (data.type === 'done') {
                                                                    setScrapeProgress(null);
                                                                    const listRes = await fetch(`${baseUrl}/election/simulation/calibration`);
                                                                    const listData = await listRes.json();
                                                                    setCalibrationEntries(listData.entries || []);
                                                                    if (data.error) alert(`Import error: ${data.error}`);
                                                                    else if ((data.added || 0) > 0) alert(`Imported ${data.added} result(s) from Ballotpedia (with 2024 margin & swing).`);
                                                                    return;
                                                                }
                                                            } catch (_) { }
                                                        }
                                                    }
                                                }
                                                setScrapeProgress(null);
                                                const listRes = await fetch(`${baseUrl}/election/simulation/calibration`);
                                                const listData = await listRes.json();
                                                setCalibrationEntries(listData.entries || []);
                                            } catch (e) {
                                                setScrapeProgress(null);
                                                alert('Ballotpedia import failed. Check console.');
                                            }
                                        }}
                                    >
                                        {scrapeProgress !== null ? 'Scraping…' : 'Scrape Ballotpedia & import (Nov 2025+)'}
                                    </button>
                                    <button
                                        type="button"
                                        className="election-tab election-tab-primary"
                                        style={{ marginLeft: '0.5rem' }}
                                        onClick={async () => {
                                            try {
                                                const res = await fetch(`${baseUrl}/election/simulation/calibration/refresh-va-nj-2024-districts`, { method: 'POST' });
                                                const data = await res.json().catch(() => ({}));
                                                if (res.ok && data.status === 'ok') {
                                                    const listRes = await fetch(`${baseUrl}/election/simulation/calibration`);
                                                    const listData = await listRes.json();
                                                    setCalibrationEntries(listData.entries || []);
                                                    alert(data.message || 'VA & NJ 2024 district data updated. Reload list to see 2024 margin and swing.');
                                                } else alert(data.error || data.detail || 'Failed.');
                                            } catch (e) {
                                                alert('Failed. Check console.');
                                            }
                                        }}
                                    >
                                        Load VA & NJ 2024 districts
                                    </button>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', display: 'block', marginTop: '0.35rem' }}>VA House from VPAP; NJ Assembly when a district-level source is available.</span>
                                </p>
                                {scrapeProgress !== null && (
                                    <div className="calibration-scrape-progress" style={{ marginBottom: '0.75rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                                            <span>Link {scrapeProgress.current} of {scrapeProgress.total}</span>
                                            <span>{scrapeProgress.total > 0 ? Math.round((100 * scrapeProgress.current) / scrapeProgress.total) : 0}%</span>
                                        </div>
                                        <div style={{ height: '8px', background: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                                            <div
                                                style={{
                                                    height: '100%',
                                                    width: `${scrapeProgress.total > 0 ? Math.min(100, (100 * scrapeProgress.current) / scrapeProgress.total) : 0}%`,
                                                    background: 'var(--primary-gradient)',
                                                    borderRadius: '4px',
                                                    transition: 'width 0.2s ease',
                                                }}
                                            />
                                        </div>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{scrapeProgress.message}</div>
                                    </div>
                                )}
                                <div className="calibration-table-container">
                                    <table style={{ width: '100%', fontSize: '0.85rem', marginBottom: '1rem' }} className="calibration-table">
                                        <thead>
                                            <tr>
                                                <th>Label</th>
                                                <th>State</th>
                                                <th>Date</th>
                                                <th>Actual D%</th>
                                                <th>Actual R%</th>
                                                <th>Poll avg%</th>
                                                <th>D over</th>
                                                <th title="2024 presidential margin in this region (+ = Trump won by X)">2024 R margin</th>
                                                <th title="Midterm swing vs 2024 (+ = D gained)">Swing (D)</th>
                                                <th>Weight</th>
                                                <th></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {calibrationEntries.map((e) => {
                                                const over = (e.dem_actual_pct != null && e.poll_avg_pct != null) ? (Number(e.dem_actual_pct) - Number(e.poll_avg_pct)).toFixed(1) : '—';
                                                const fmt2024 = e.trump_2024_margin != null ? (Number(e.trump_2024_margin) >= 0 ? `R+${Number(e.trump_2024_margin)}` : `D+${Math.abs(Number(e.trump_2024_margin))}`) : '—';
                                                const fmtSwing = e.swing_toward_d != null ? (Number(e.swing_toward_d) >= 0 ? `D+${Number(e.swing_toward_d)}` : `R+${Math.abs(Number(e.swing_toward_d))}`) : '—';
                                                const fmtPct = (v) => (v != null && v !== '') ? `${Number(v).toFixed(1)}%` : '—';
                                                return (
                                                    <tr key={e.id}>
                                                        <td>{e.label || e.id}</td>
                                                        <td>{e.state}</td>
                                                        <td>{e.date}</td>
                                                        <td>{fmtPct(e.dem_actual_pct)}</td>
                                                        <td>{fmtPct(e.rep_actual_pct)}</td>
                                                        <td>{e.poll_avg_pct != null && e.poll_avg_pct !== '' ? `${Number(e.poll_avg_pct).toFixed(1)}%` : '—'}</td>
                                                        <td>{over}</td>
                                                        <td>{fmt2024}</td>
                                                        <td>{fmtSwing}</td>
                                                        <td>{e.weight}</td>
                                                        <td>
                                                            <button
                                                                type="button"
                                                                className="election-tab"
                                                                style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
                                                                onClick={async () => {
                                                                    try {
                                                                        await fetch(`${baseUrl}/election/simulation/calibration/${e.id}`, { method: 'DELETE' });
                                                                        setCalibrationEntries((prev) => prev.filter((x) => x.id !== e.id));
                                                                    } catch (_) { }
                                                                }}
                                                            >
                                                                Remove
                                                            </button>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                                {calibrationEntries.length === 0 && <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>No calibration entries. Add one below.</p>}
                                <form
                                    onSubmit={async (ev) => {
                                        ev.preventDefault();
                                        const f = calibrationForm;
                                        const dem = parseFloat(f.dem_actual_pct);
                                        const poll = parseFloat(f.poll_avg_pct);
                                        if (Number.isNaN(dem) || Number.isNaN(poll) || !f.state?.trim() || !f.date?.trim()) return;
                                        try {
                                            const res = await fetch(`${baseUrl}/election/simulation/calibration`, {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    label: f.label || `${f.state} ${f.date}`,
                                                    type: f.type || 'special',
                                                    state: f.state.trim().toUpperCase().slice(0, 2),
                                                    date: f.date.trim(),
                                                    dem_actual_pct: dem,
                                                    poll_avg_pct: poll,
                                                    weight: Math.max(0, Math.min(2, parseFloat(f.weight) || 1)),
                                                    note: f.note || null
                                                })
                                            });
                                            const data = await res.json();
                                            if (data.entry) setCalibrationEntries((prev) => [...prev, data.entry]);
                                            setCalibrationForm({ label: '', type: 'special', state: '', date: '', dem_actual_pct: '', poll_avg_pct: '', weight: '1', note: '' });
                                        } catch (_) { }
                                    }}
                                    className="calibration-form"
                                >
                                    <input placeholder="Label" value={calibrationForm.label} onChange={(e) => setCalibrationForm((f) => ({ ...f, label: e.target.value }))} style={{ width: '100px' }} />
                                    <select value={calibrationForm.type} onChange={(e) => setCalibrationForm((f) => ({ ...f, type: e.target.value }))}>
                                        <option value="special">Special</option>
                                        <option value="off_year">Off-year</option>
                                        <option value="midterm">Midterm</option>
                                        <option value="state_senate">State Senate</option>
                                        <option value="state_house">State House</option>
                                    </select>
                                    <input placeholder="State (e.g. VA)" value={calibrationForm.state} onChange={(e) => setCalibrationForm((f) => ({ ...f, state: e.target.value }))} style={{ width: '50px' }} />
                                    <input placeholder="Date (e.g. 2025-11)" value={calibrationForm.date} onChange={(e) => setCalibrationForm((f) => ({ ...f, date: e.target.value }))} style={{ width: '90px' }} />
                                    <input type="number" placeholder="Actual D%" value={calibrationForm.dem_actual_pct} onChange={(e) => setCalibrationForm((f) => ({ ...f, dem_actual_pct: e.target.value }))} style={{ width: '70px' }} />
                                    <input type="number" placeholder="Poll avg%" value={calibrationForm.poll_avg_pct} onChange={(e) => setCalibrationForm((f) => ({ ...f, poll_avg_pct: e.target.value }))} style={{ width: '70px' }} />
                                    <input type="number" step="0.1" min="0" max="2" placeholder="Weight" value={calibrationForm.weight} onChange={(e) => setCalibrationForm((f) => ({ ...f, weight: e.target.value }))} style={{ width: '55px' }} />
                                    <input placeholder="Note" value={calibrationForm.note} onChange={(e) => setCalibrationForm((f) => ({ ...f, note: e.target.value }))} style={{ width: '120px' }} />
                                    <button type="submit" className="election-tab election-tab-primary">Add</button>
                                </form>
                            </div>
                        )}
                    </div>

                    <ElectionMap
                        raceType={mapRaceType}
                        userData={mapData?.user_data || []}
                        scrapedAverages={mapData?.scraped_averages || {}}
                        winProbByState={Object.keys(winProbs).length ? winProbs : null}
                    />
                </div>
            );
        }
        if (activeTab === 'map') {
            if (mapLoading) {
                return (
                    <div className="loading-spinner">
                        <div style={{ marginBottom: '1rem' }}>Loading map data from polls…</div>
                    </div>
                );
            }
            if (mapError) {
                return (
                    <div className="error-message">
                        {mapError}
                        <button
                            type="button"
                            className="election-tab"
                            style={{ marginLeft: '0.5rem' }}
                            onClick={() => {
                                setMapError(null);
                                setMapLoading(true);
                                const baseUrl = PRIMARY_API_URL || getBackendUrl();
                                fetch(`${baseUrl}/election/map?race_type=${mapRaceType}`)
                                    .then((r) => r.json())
                                    .then((d) => { setMapData(d); setMapError(null); })
                                    .catch(() => setMapError('Retry failed.'))
                                    .finally(() => setMapLoading(false));
                            }}
                        >
                            Retry
                        </button>
                    </div>
                );
            }
            return (
                <ElectionMap
                    raceType={mapRaceType}
                    userData={mapData.user_data || []}
                    scrapedAverages={mapData.scraped_averages || {}}
                    onRefresh={() => {
                        setMapLoading(true);
                        setMapError(null);
                        const baseUrl = PRIMARY_API_URL || getBackendUrl();
                        fetch(`${baseUrl}/election/map?race_type=${mapRaceType}`)
                            .then((r) => r.json())
                            .then((d) => { setMapData(d); setMapError(null); })
                            .catch(() => setMapError('Refresh failed.'))
                            .finally(() => setMapLoading(false));
                    }}
                />
            );
        }

        // For tabs dependent on polling data
        if (activeTab === 'trends' && trendLoading) {
            return (
                <div className="loading-spinner">
                    <div style={{ marginBottom: '1rem' }}>Loading trend data...</div>
                </div>
            );
        }

        if (activeTab !== 'trends' && loading) {
            return (
                <div className="loading-spinner">
                    <div style={{ marginBottom: '1rem' }}>Loading polling data...</div>
                </div>
            );
        }

        if (activeTab === 'trends' && trendError) return <div className="error-message">{trendError}</div>;
        if (activeTab !== 'trends' && error) return <div className="error-message">{error}</div>;

        switch (activeTab) {
            case 'trends':
                return <PollTrends data={trendData} raceType={raceType} />;
            case 'polls': {
                const isRefreshing = data?.status === 'refreshing';
                return (
                    <div>
                        {isRefreshing && (
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                {data?.message || 'Fetching data...'}
                            </p>
                        )}
                        <PollTable polls={data?.polls || []} metadata={data?.metadata || {}} raceType={raceType} />
                    </div>
                );
            }
            default:
                return null;
        }
    };

    return (
        <div className="election-tracker-container">
            {/* Header */}
            <div className="election-header">
                <div className="election-tabs">
                    <button
                        className={`election-tab ${activeTab === 'polls' ? 'active' : ''}`}
                        onClick={() => setActiveTab('polls')}
                    >
                        <Table size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        Polls
                    </button>
                    <button
                        className={`election-tab ${activeTab === 'trends' ? 'active' : ''}`}
                        onClick={() => setActiveTab('trends')}
                    >
                        <TrendingUp size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        Trends
                    </button>
                    <button
                        className={`election-tab ${activeTab === 'news' ? 'active' : ''}`}
                        onClick={() => setActiveTab('news')}
                    >
                        <Newspaper size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        News
                    </button>
                    <button
                        className={`election-tab ${activeTab === 'map' ? 'active' : ''}`}
                        onClick={() => setActiveTab('map')}
                    >
                        <Vote size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        Map
                    </button>
                    <button
                        className={`election-tab ${activeTab === 'simulations' ? 'active' : ''}`}
                        onClick={() => setActiveTab('simulations')}
                    >
                        <BarChart3 size={16} style={{ marginRight: '8px', verticalAlign: 'text-bottom' }} />
                        Simulations
                    </button>
                </div>
            </div>

            {/* Controls - Only show for relevant tabs */}
            {['polls', 'trends', 'map', 'simulations'].includes(activeTab) && (
                <div className="election-controls">
                    <div className="control-group">
                        <label>{activeTab === 'map' ? 'Race (for Polls/Trends):' : 'Race:'}</label>
                        <select
                            className="control-select"
                            value={raceType}
                            onChange={(e) => {
                                const v = e.target.value;
                                setRaceType(v);
                                try {
                                    window.localStorage.setItem(ELECTION_RACE_TYPE_KEY, v);
                                } catch (_) { }
                                if (MAP_RACE_TYPES.includes(v)) setMapRaceType(v);
                            }}
                            style={{ color: '#111827', background: '#f8fafc' }}
                        >
                            <option value="approval">Presidential Approval</option>
                            <option value="generic_ballot">Generic Ballot (National)</option>
                            <option value="senate">2026 Senate - General</option>
                            <option value="house">2026 House - Individual Races</option>
                            <option value="governor">2026 Governor - General</option>
                        </select>
                    </div>
                    {(activeTab === 'map' || activeTab === 'simulations') && (
                        <div className="control-group">
                            <label>{activeTab === 'simulations' ? 'Race type:' : 'Map:'}</label>
                            <select
                                className="control-select"
                                value={mapRaceType}
                                onChange={(e) => setMapRaceType(e.target.value)}
                                style={{ color: '#111827', background: '#f8fafc' }}
                            >
                                {activeTab === 'simulations' ? (
                                    <>
                                        <option value="senate">2026 Senate (all states with polls)</option>
                                        <option value="governor">2026 Governor (all states with polls)</option>
                                        <option value="house">2026 House (states with polls; seat projections later)</option>
                                    </>
                                ) : (
                                    <>
                                        <option value="senate">Senate</option>
                                        <option value="house">House</option>
                                        <option value="governor">Governor</option>
                                    </>
                                )}
                            </select>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                {activeTab === 'simulations' ? 'Choose what to simulate' : 'State-level only'}
                            </span>
                        </div>
                    )}
                    {['polls', 'trends'].includes(activeTab) && (
                        <button
                            className="election-tab"
                            disabled={refreshing}
                            onClick={async () => {
                                setRefreshing(true);
                                const baseUrl = PRIMARY_API_URL || getBackendUrl();
                                try {
                                    const res = await fetch(`${baseUrl}/election/polls/refresh?race_type=${raceType}`, { method: 'POST' });
                                    const json = await res.json().catch(() => ({}));
                                    const ctrl = new AbortController();
                                    const rid = ++requestIdRef.current;
                                    if (activeTab === 'polls') fetchPollingData(ctrl.signal, rid);
                                    else if (activeTab === 'trends') fetchTrendData(ctrl.signal, rid);
                                    if (json.message) {
                                        const msg = json.status === 'error' ? `Refresh failed: ${json.message}` : json.message;
                                        if (typeof window !== 'undefined' && window.alert) window.alert(msg);
                                    }
                                } finally {
                                    setRefreshing(false);
                                }
                            }}
                            style={{ border: '1px solid var(--border-color)', fontSize: '0.85rem' }}
                        >
                            {refreshing ? 'Refreshing…' : '↻ Refresh Data'}
                        </button>
                    )}
                    {data?.sources?.length > 0 && (
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', alignSelf: 'center' }}>
                            Sources: {data.sources.join(', ')} | Last updated: {formatTimeAgo(data.last_updated)}
                        </span>
                    )}
                    <button className="ask-ai-btn" onClick={handleAskAI}>
                        <MessageSquare size={16} />
                        {aiOpen ? 'Close AI' : 'Ask AI about this'}
                    </button>
                </div>
            )}

            {aiOpen && (
                <div className="election-ai-panel">
                    <div className="election-ai-header">
                        <div>
                            <div className="election-ai-title">Election AI Insights</div>
                            <div className="election-ai-subtitle">
                                Uses live web search + current polls for context.
                            </div>
                        </div>
                        <button className="election-ai-close" onClick={handleAskAI} aria-label="Close AI panel">
                            <X size={16} />
                        </button>
                    </div>

                    <div className="election-ai-actions">
                        <button
                            className="election-ai-quick"
                            onClick={handleQuickInsight}
                            disabled={aiLoading}
                        >
                            Quick Insight
                        </button>
                        <div className="election-ai-status">
                            {aiLoading ? 'Thinking...' : (primaryModel ? `Model: ${primaryModel}` : 'No model selected')}
                        </div>
                    </div>

                    {aiError && <div className="election-ai-error">{aiError}</div>}

                    <div className="election-ai-messages">
                        {aiMessages.length === 0 && (
                            <div className="election-ai-empty">
                                Ask about trends, key races, or surprising shifts in the latest polls.
                            </div>
                        )}
                        {aiMessages.map((msg, idx) => (
                            <div key={`${msg.role}-${idx}`} className={`election-ai-message ${msg.role}`}>
                                <div className="election-ai-role">
                                    {msg.role === 'user' ? 'You' : 'Assistant'}
                                </div>
                                <div className="election-ai-content">{msg.content}</div>
                                {msg.toolSteps && msg.toolSteps.length > 0 && (
                                    <div className="election-ai-tools">
                                        Searches: {msg.toolSteps.map((step) => step.query || step.tool || 'web_search').join(', ')}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    <form
                        className="election-ai-input"
                        onSubmit={(e) => {
                            e.preventDefault();
                            sendAiMessage(aiInput);
                        }}
                    >
                        <input
                            type="text"
                            placeholder="Ask about this race, a pollster, or why a lead matters..."
                            value={aiInput}
                            onChange={(e) => setAiInput(e.target.value)}
                            disabled={aiLoading}
                        />
                        <button type="submit" disabled={aiLoading || !aiInput.trim()}>
                            Send
                        </button>
                    </form>
                </div>
            )}

            {/* Suggested questions */}
            {suggestedQuestions.length > 0 && (activeTab === 'polls' || aiOpen) && (
                <div style={{
                    display: 'flex',
                    gap: '0.5rem',
                    flexWrap: 'wrap',
                    padding: '0.75rem 1rem',
                    borderBottom: '1px solid var(--border-color)'
                }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', alignSelf: 'center' }}>Ask AI:</span>
                    {suggestedQuestions.map((q, idx) => (
                        <button
                            key={idx}
                            className="election-ai-quick"
                            onClick={() => {
                                if (!aiOpen) setAiOpen(true);
                                sendAiMessage(q);
                            }}
                            disabled={aiLoading}
                            style={{ fontSize: '0.8rem' }}
                        >
                            {q}
                        </button>
                    ))}
                    <button
                        className="election-ai-quick"
                        onClick={() => generateQuestions(true)}
                        disabled={questionsLoading}
                        style={{ fontSize: '0.75rem', opacity: 0.7 }}
                    >
                        ↻ {questionsLoading ? '...' : 'New questions'}
                    </button>
                </div>
            )}

            {/* Main Content */}
            <div className="election-content">
                {renderContent()}
            </div>
        </div>
    );
};

export default ElectionTracker;
