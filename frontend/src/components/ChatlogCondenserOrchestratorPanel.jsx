import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Button } from './ui/button';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Checkbox } from './ui/checkbox';
import { Loader2, MessageSquare, Pause, Play, Square, Zap } from 'lucide-react';

import {
  formatApiError,
  normalizeEndpointModelId,
  pickLongerDraft,
  readOrchestratorStream,
} from '../utils/chatlogCondenserUtils';

const RUN_ID_STORAGE_KEY = 'chatlogCondenserOrchestratorRunId';

/**
 * Autonomous condenser run: ordered API failover, load-share rotation, checkpoints, boss chat.
 */
export default function ChatlogCondenserOrchestratorPanel({
  apiUrl,
  apiReady,
  inputText,
  includeFullLogContext,
  settings = {},
  useRag = false,
  ragDocs = [],
}) {
  const endpointOptions = useMemo(() => {
    const opts = [];
    for (const ep of settings.customApiEndpoints || []) {
      if (!ep?.enabled || !ep?.id) continue;
      opts.push({
        id: normalizeEndpointModelId(ep.id),
        label: ep.name || ep.id,
      });
    }
    return opts;
  }, [settings.customApiEndpoints]);

  const globalRoundRobin = settings.apiEndpointRoundRobinEnabled === true;

  const [selectedEndpoints, setSelectedEndpoints] = useState([]);
  const [chunkTurns, setChunkTurns] = useState(20);
  const [autoRun, setAutoRun] = useState(true);
  const [alternateApis, setAlternateApis] = useState(true);
  const [useGlobalRoundRobin, setUseGlobalRoundRobin] = useState(globalRoundRobin);
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [cursorTurn, setCursorTurn] = useState(-1);
  const [totalTurns, setTotalTurns] = useState(0);
  const [partialCondensed, setPartialCondensed] = useState('');
  const [stepDraft, setStepDraft] = useState('');
  const [logs, setLogs] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [activeEndpoint, setActiveEndpoint] = useState(null);
  const [activeSlot, setActiveSlot] = useState(null);
  const [tokensEst, setTokensEst] = useState(null);
  const [stepCount, setStepCount] = useState(0);
  const [supervisorDraft, setSupervisorDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);
  const pollRef = useRef(null);

  useEffect(() => {
    setUseGlobalRoundRobin(globalRoundRobin);
  }, [globalRoundRobin]);

  useEffect(() => {
    if (selectedEndpoints.length || !endpointOptions.length) return;
    setSelectedEndpoints(endpointOptions.slice(0, 2).map((o) => o.id));
  }, [endpointOptions, selectedEndpoints.length]);

  const progressPct = useMemo(() => {
    if (totalTurns <= 0) return 0;
    const done = Math.max(0, cursorTurn + 1);
    return Math.min(100, Math.round((done / totalTurns) * 100));
  }, [cursorTurn, totalTurns]);

  const endpointLabelById = useMemo(() => {
    const m = {};
    for (const o of endpointOptions) m[o.id] = o.label;
    return m;
  }, [endpointOptions]);

  const applyRun = useCallback((run) => {
    if (!run) return;
    setRunStatus(run.status);
    setCursorTurn(run.cursor_turn ?? -1);
    setTotalTurns(run.total_turns ?? 0);
    setPartialCondensed((prev) => pickLongerDraft(prev, run.partial_condensed));
    if (Array.isArray(run.logs)) setLogs(run.logs);
    setStepCount(run.step_count ?? 0);
    const slot = run.current_endpoint_slot ?? run.active_endpoint_id
      ? selectedEndpoints.indexOf(run.active_endpoint_id) + 1
      : null;
    setActiveSlot(run.current_endpoint_slot ?? slot);
    setActiveEndpoint({
      id: run.active_endpoint_id || run.current_endpoint_id,
      name:
        run.active_endpoint_name ||
        endpointLabelById[run.active_endpoint_id || run.current_endpoint_id] ||
        run.current_endpoint_id,
    });
    if (typeof run.last_step_tokens_est === 'number') {
      setTokensEst(run.last_step_tokens_est);
    }
  }, [endpointLabelById, selectedEndpoints]);

  const refreshRun = useCallback(
    async (id) => {
      const res = await fetch(`${apiUrl}/memory/chatlog-condenser/orchestrator/${id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      applyRun(data.run);
      return data.run;
    },
    [apiUrl, applyRun]
  );

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (id) => {
      stopPolling();
      pollRef.current = setInterval(() => {
        refreshRun(id).catch(() => {});
      }, 2500);
    },
    [refreshRun, stopPolling]
  );

  const connectStream = useCallback(
    async (id) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      const res = await fetch(
        `${apiUrl}/memory/chatlog-condenser/orchestrator/${id}/stream`,
        { signal: ac.signal }
      );
      await readOrchestratorStream(
        res,
        {
          onEvent: (ev) => {
            if (ev.type === 'status' && ev.run) applyRun(ev.run);
            if (ev.type === 'step_timeline') {
              setTimeline((prev) => [
                ...prev.slice(-40),
                {
                  at: new Date().toISOString(),
                  phase: ev.phase,
                  attempt: ev.attempt,
                  endpoint: ev.endpoint_name || ev.endpoint_id,
                  slot: ev.endpoint_slot,
                  tokens: ev.tokens_est,
                },
              ]);
              if (ev.tokens_est != null) setTokensEst(ev.tokens_est);
            }
            if (ev.type === 'step_start') {
              setActiveEndpoint({
                id: ev.endpoint_id,
                name: ev.endpoint_name || endpointLabelById[ev.endpoint_id] || ev.endpoint_id,
              });
              setActiveSlot(ev.endpoint_slot);
            }
            if (ev.type === 'token' && ev.text) {
              setStepDraft(ev.text);
              if (ev.aggregated) {
                setPartialCondensed((prev) => pickLongerDraft(prev, ev.text));
              }
            }
            if (ev.type === 'step_done') {
              setStepDraft('');
              if (ev.partial_condensed) setPartialCondensed(ev.partial_condensed);
              if (typeof ev.cursor_turn === 'number') setCursorTurn(ev.cursor_turn);
              if (typeof ev.step === 'number') setStepCount(ev.step);
              if (ev.tokens_est != null) setTokensEst(ev.tokens_est);
              setTimeline((prev) => [
                ...prev,
                {
                  at: new Date().toISOString(),
                  phase: 'done',
                  marker: ev.marker,
                  endpoint: ev.endpoint_name || ev.endpoint_id,
                  checkpoint: ev.checkpoint,
                },
              ]);
            }
            if (ev.type === 'completed' || ev.type === 'stopped') {
              if (ev.partial_condensed) setPartialCondensed(ev.partial_condensed);
              if (ev.type === 'completed') setRunStatus('completed');
              if (ev.type === 'stopped') setRunStatus('stopped');
              setBusy(false);
              stopPolling();
            }
            if (ev.type === 'failover') {
              setLogs((prev) => [
                ...prev,
                {
                  level: 'warn',
                  message: `Failover ${ev.from_endpoint_id} → ${ev.to_endpoint_id}: ${ev.reason || ''}`,
                  created_at: new Date().toISOString(),
                },
              ]);
            }
            if (ev.type === 'error') {
              setError(ev.detail || 'Stream error');
              setBusy(false);
            }
          },
        },
        ac.signal
      );
    },
    [apiUrl, applyRun, endpointLabelById, stopPolling]
  );

  useEffect(() => () => {
    abortRef.current?.abort();
    stopPolling();
  }, [stopPolling]);

  useEffect(() => {
    const saved = localStorage.getItem(RUN_ID_STORAGE_KEY);
    if (!saved || runId) return;
    refreshRun(saved)
      .then((run) => {
        if (run && ['paused', 'stopped', 'running'].includes(run.status)) {
          setRunId(saved);
          if (run.status === 'running') {
            setBusy(true);
            startPolling(saved);
            connectStream(saved).catch(() => {});
          }
        }
      })
      .catch(() => localStorage.removeItem(RUN_ID_STORAGE_KEY));
  }, [apiUrl, connectStream, refreshRun, runId, startPolling]);

  const toggleEndpoint = (id) => {
    setSelectedEndpoints((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      return [...prev, id];
    });
  };

  const moveEndpoint = (id, dir) => {
    setSelectedEndpoints((prev) => {
      const i = prev.indexOf(id);
      if (i < 0) return prev;
      const j = i + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = [...prev];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  };

  const handleStart = async (resume = false) => {
    if (!resume && !inputText.trim()) {
      setError('Paste or load a chatlog first.');
      return;
    }
    if (!resume && !selectedEndpoints.length) {
      setError('Select at least one API endpoint (failover order).');
      return;
    }
    setError(null);
    setBusy(true);
    if (!resume) {
      setPartialCondensed('');
      setStepDraft('');
      setLogs([]);
      setTimeline([]);
      setCursorTurn(-1);
      setStepCount(0);
    }
    try {
      const body = resume
        ? { text: inputText || ' ', endpoint_ids: selectedEndpoints, resume_run_id: runId }
        : {
            text: inputText,
            endpoint_ids: selectedEndpoints,
            chunk_turns: chunkTurns,
            include_full_log_context: includeFullLogContext,
            auto_run: autoRun,
            target_ratio: 0.4,
            alternate_apis_every_step: alternateApis,
            use_global_round_robin: useGlobalRoundRobin && !alternateApis,
            use_rag: useRag && ragDocs.length > 0,
            rag_docs: ragDocs,
          };
      const res = await fetch(`${apiUrl}/memory/chatlog-condenser/orchestrator/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      const id = data.run?.run_id;
      setRunId(id);
      localStorage.setItem(RUN_ID_STORAGE_KEY, id);
      applyRun(data.run);
      startPolling(id);
      connectStream(id).catch((e) => {
        if (e?.name !== 'AbortError') setError(e.message);
      });
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  const postControl = async (action) => {
    if (!runId) return;
    setError(null);
    try {
      const path =
        action === 'stop' ? 'stop' : action;
      const res = await fetch(
        `${apiUrl}/memory/chatlog-condenser/orchestrator/${runId}/${path}`,
        { method: 'POST' }
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      applyRun(data.run);
      if (action === 'resume' && autoRun) {
        setBusy(true);
        connectStream(runId).catch(() => {});
        startPolling(runId);
      }
      if (action === 'pause' || action === 'stop' || action === 'cancel') {
        setBusy(false);
        abortRef.current?.abort();
        stopPolling();
      }
    } catch (e) {
      setError(e.message);
    }
  };

  const sendSupervisor = async () => {
    if (!runId || !supervisorDraft.trim()) return;
    setError(null);
    try {
      const res = await fetch(
        `${apiUrl}/memory/chatlog-condenser/orchestrator/${runId}/supervisor`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: supervisorDraft.trim() }),
        }
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      setSupervisorDraft('');
      setLogs((prev) => [
        ...prev,
        {
          level: 'info',
          message: `Supervisor queued: ${data.run?.supervisor_instruction || '(next step)'}`,
          created_at: new Date().toISOString(),
        },
      ]);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleProceed = async () => {
    if (!runId) return;
    setError(null);
    setBusy(true);
    try {
      const res = await fetch(
        `${apiUrl}/memory/chatlog-condenser/orchestrator/${runId}/tick`,
        { method: 'POST' }
      );
      const data = await res.json();
      if (!res.ok) throw new Error(formatApiError(data, res.statusText));
      applyRun(data.run);
      if (data.run?.status === 'completed') setBusy(false);
    } catch (e) {
      setError(e.message);
    } finally {
      if (runStatus !== 'running') setBusy(false);
    }
  };

  const isPaused = runStatus === 'paused';
  const isStopped = runStatus === 'stopped';
  const isRunning = runStatus === 'running' || busy;
  const isDone = runStatus === 'completed';

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground leading-relaxed">
        Autonomous run condenses the full log turn-by-turn. Enable load-share to alternate API #1
        and #2 every step; checkpoints land under{' '}
        <code className="px-1 rounded bg-muted">~/.LiangLocal/condenser_runs/</code>. Stop keeps
        partial draft and resume later.
      </p>

      {!apiReady && (
        <Alert>
          <AlertTitle>Waiting for backend</AlertTitle>
          <AlertDescription>Start Mirid and wait for storage hydration.</AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        <Label>API endpoints (failover order = #1, #2, …)</Label>
        {endpointOptions.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Add enabled custom API endpoints in Settings.
          </p>
        ) : (
          <ul className="space-y-1 rounded-lg border border-border/80 p-2 text-sm">
            {endpointOptions.map((ep) => {
              const sel = selectedEndpoints.includes(ep.id);
              const idx = selectedEndpoints.indexOf(ep.id);
              return (
                <li key={ep.id} className="flex flex-wrap items-center gap-2">
                  <Checkbox
                    id={`orch-ep-${ep.id}`}
                    checked={sel}
                    onCheckedChange={() => toggleEndpoint(ep.id)}
                  />
                  <Label htmlFor={`orch-ep-${ep.id}`} className="font-normal cursor-pointer flex-1">
                    {ep.label}
                    {sel ? ` (#${idx + 1})` : ''}
                  </Label>
                  {sel && (
                    <>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        disabled={idx <= 0}
                        onClick={() => moveEndpoint(ep.id, -1)}
                      >
                        ↑
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        disabled={idx >= selectedEndpoints.length - 1}
                        onClick={() => moveEndpoint(ep.id, 1)}
                      >
                        ↓
                      </Button>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-2">
          <Label>Turns per step</Label>
          <Input
            type="number"
            min={1}
            max={80}
            value={chunkTurns}
            onChange={(e) => setChunkTurns(Number(e.target.value) || 20)}
          />
        </div>
        <div className="flex flex-col gap-2 pb-1">
          <div className="flex items-center gap-2">
            <Checkbox
              id="orch-auto"
              checked={autoRun}
              onCheckedChange={(c) => setAutoRun(!!c)}
            />
            <Label htmlFor="orch-auto" className="font-normal cursor-pointer">
              Auto-run in background
            </Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="orch-alternate"
              checked={alternateApis}
              onCheckedChange={(c) => setAlternateApis(!!c)}
            />
            <Label htmlFor="orch-alternate" className="font-normal cursor-pointer">
              Alternate APIs every step (load share)
            </Label>
          </div>
          {globalRoundRobin && !alternateApis && (
            <div className="flex items-center gap-2">
              <Checkbox
                id="orch-global-rr"
                checked={useGlobalRoundRobin}
                onCheckedChange={(c) => setUseGlobalRoundRobin(!!c)}
              />
              <Label htmlFor="orch-global-rr" className="font-normal cursor-pointer text-xs">
                Use Settings global round-robin among selected endpoints
              </Label>
            </div>
          )}
        </div>
      </div>

      {runId && (
        <div className="rounded-lg border border-border/80 bg-muted/30 px-3 py-2 text-sm flex flex-wrap gap-x-4 gap-y-1">
          <span>
            Active API:{' '}
            <strong>
              {activeSlot ? `#${activeSlot} ` : ''}
              {activeEndpoint?.name || '—'}
            </strong>
          </span>
          <span>Steps completed: {stepCount}</span>
          {tokensEst != null && <span>Last step ~{tokensEst.toLocaleString()} tok est.</span>}
          {runId && (
            <span className="text-xs text-muted-foreground truncate max-w-full">
              Run {runId.slice(0, 8)}…
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <Button type="button" onClick={() => handleStart(false)} disabled={!apiReady || isRunning || isDone}>
          {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Zap className="mr-2 h-4 w-4" />}
          Start autonomous run
        </Button>
        {runId && (isStopped || isPaused) && (
          <Button type="button" variant="secondary" onClick={() => handleStart(true)}>
            <Play className="mr-2 h-4 w-4" />
            Resume saved run
          </Button>
        )}
        {runId && isRunning && (
          <Button type="button" variant="outline" onClick={() => postControl('pause')}>
            <Pause className="mr-2 h-4 w-4" />
            Pause
          </Button>
        )}
        {runId && isPaused && (
          <Button type="button" variant="outline" onClick={() => postControl('resume')}>
            <Play className="mr-2 h-4 w-4" />
            Resume
          </Button>
        )}
        {runId && !isDone && (
          <Button type="button" variant="destructive" onClick={() => postControl('stop')}>
            <Square className="mr-2 h-4 w-4" />
            Stop (keep checkpoints)
          </Button>
        )}
        {runId && isPaused && !autoRun && (
          <Button type="button" variant="secondary" onClick={handleProceed}>
            Proceed (one step)
          </Button>
        )}
      </div>

      {totalTurns > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Turn {Math.max(0, cursorTurn + 1)} of {totalTurns}
              {runStatus ? ` · ${runStatus}` : ''}
            </span>
            <span>{progressPct}%</span>
          </div>
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {runId && (
        <div className="space-y-2 rounded-lg border border-border/80 p-3">
          <Label className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Boss / supervisor (next step only)
          </Label>
          <div className="flex gap-2">
            <Input
              placeholder='e.g. "slow down", "skip to turn 50", "use endpoint 2 only"'
              value={supervisorDraft}
              onChange={(e) => setSupervisorDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') sendSupervisor();
              }}
            />
            <Button type="button" variant="secondary" onClick={sendSupervisor} disabled={!supervisorDraft.trim()}>
              Send
            </Button>
          </div>
        </div>
      )}

      <div className="grid gap-3 lg:grid-cols-3">
        <div className="space-y-2 lg:col-span-1">
          <Label>Step timeline</Label>
          <Textarea
            className="min-h-[140px] font-mono text-xs"
            readOnly
            value={
              timeline.length
                ? timeline
                    .map((t) => {
                      const bits = [t.phase, t.endpoint, t.marker, t.checkpoint, t.tokens != null ? `~${t.tokens}tok` : '']
                        .filter(Boolean)
                        .join(' · ');
                      return `[${t.at.slice(11, 19)}] ${bits}`;
                    })
                    .join('\n')
                : 'Steps appear here…'
            }
          />
        </div>
        <div className="space-y-2">
          <Label>Monitor log</Label>
          <Textarea
            className="min-h-[140px] font-mono text-xs"
            readOnly
            value={
              logs.length
                ? logs
                    .map(
                      (l) =>
                        `[${l.level}] ${l.endpoint_id ? `(${l.endpoint_id}) ` : ''}${l.message}`
                    )
                    .join('\n')
                : 'Logs appear after start…'
            }
          />
        </div>
        <div className="space-y-2">
          <Label>Condensed draft (streaming)</Label>
          <Textarea
            className="min-h-[140px] font-mono text-xs"
            readOnly
            value={pickLongerDraft(partialCondensed, stepDraft)}
          />
        </div>
      </div>
    </div>
  );
}
