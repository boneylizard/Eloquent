import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import {
  ChevronDown,
  ChevronUp,
  ListOrdered,
  Loader2,
  Play,
  Plus,
  Square,
  Trash2,
} from 'lucide-react';
import {
  createQueueJob,
  jobLabel,
  loadVoiceMergeQueue,
  saveVoiceMergeQueue,
} from '../utils/voiceMergeQueue';
import { runVoiceSculptStream, sculptBodyFromQueueJob } from '../utils/voiceSculptStream';
import { safeErrorMessage } from '../config/api';

function statusBadge(status) {
  switch (status) {
    case 'running':
      return 'Running';
    case 'done':
      return 'Done';
    case 'error':
      return 'Failed';
    case 'cancelled':
      return 'Cancelled';
    default:
      return 'Pending';
  }
}

export default function VoiceMergeQueuePanel({
  apiUrl,
  disabled = false,
  sculptRunning = false,
  onQueueRunningChange,
  getCurrentJobSnapshot,
  onVoiceReady,
}) {
  const [jobs, setJobs] = useState(() => loadVoiceMergeQueue());
  const [queueRunning, setQueueRunning] = useState(false);
  const [queueIndex, setQueueIndex] = useState(-1);
  const [queueMessage, setQueueMessage] = useState('');
  const [queueError, setQueueError] = useState(null);
  const abortRef = useRef(null);

  useEffect(() => {
    saveVoiceMergeQueue(jobs);
  }, [jobs]);

  useEffect(() => {
    onQueueRunningChange?.(queueRunning);
  }, [queueRunning, onQueueRunningChange]);

  const updateJob = useCallback((id, patch) => {
    setJobs((prev) => prev.map((j) => (j.id === id ? { ...j, ...patch } : j)));
  }, []);

  const removeJob = useCallback((id) => {
    setJobs((prev) => prev.filter((j) => j.id !== id));
  }, []);

  const moveJob = useCallback((id, dir) => {
    setJobs((prev) => {
      const idx = prev.findIndex((j) => j.id === id);
      if (idx < 0) return prev;
      const next = idx + dir;
      if (next < 0 || next >= prev.length) return prev;
      const copy = [...prev];
      [copy[idx], copy[next]] = [copy[next], copy[idx]];
      return copy;
    });
  }, []);

  const addFromCurrent = useCallback(() => {
    const snap = getCurrentJobSnapshot?.();
    if (!snap) return;
    setJobs((prev) => [...prev, createQueueJob({ ...snap, status: 'pending' })]);
  }, [getCurrentJobSnapshot]);

  const addEmpty = useCallback(() => {
    setJobs((prev) => [...prev, createQueueJob()]);
  }, []);

  const cancelQueue = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const runQueue = useCallback(async () => {
    if (!apiUrl || queueRunning || sculptRunning) return;
    const runnable = jobs.filter((j) => (j.source || '').trim());
    if (runnable.length === 0) {
      setQueueError('Add at least one job with source paths.');
      return;
    }

    setQueueError(null);
    setQueueRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const results = [];
    try {
      for (let i = 0; i < runnable.length; i += 1) {
        if (controller.signal.aborted) break;
        const job = runnable[i];
        setQueueIndex(i);
        setQueueMessage(`Job ${i + 1} of ${runnable.length}: ${jobLabel(job)}`);
        updateJob(job.id, { status: 'running', statusMessage: 'Starting…', error: undefined });

        const body = sculptBodyFromQueueJob(job);
        delete body._sourceCount;

        try {
          const done = await runVoiceSculptStream(apiUrl, body, {
            signal: controller.signal,
            onProgress: (ev) => {
              updateJob(job.id, { statusMessage: ev.message || '' });
              setQueueMessage(`Job ${i + 1}/${runnable.length}: ${ev.message || 'Working…'}`);
            },
          });
          updateJob(job.id, {
            status: 'done',
            statusMessage: 'Complete',
            resultVoiceId: done.voice_id,
            resultPath: done.path,
          });
          results.push(done);
          if (onVoiceReady) onVoiceReady(done);
        } catch (e) {
          if (e.name === 'AbortError') {
            updateJob(job.id, { status: 'cancelled', statusMessage: 'Cancelled' });
            break;
          }
          const msg = safeErrorMessage(e, 'Merge failed');
          updateJob(job.id, { status: 'error', error: msg, statusMessage: msg });
          setQueueError(`Job ${i + 1} failed: ${msg}`);
          break;
        }
      }
    } finally {
      setQueueRunning(false);
      setQueueIndex(-1);
      abortRef.current = null;
      if (!controller.signal.aborted && results.length === runnable.length) {
        setQueueMessage(`Finished ${results.length} merge(s).`);
      } else if (controller.signal.aborted) {
        setQueueMessage('Queue cancelled.');
      }
    }
  }, [apiUrl, jobs, queueRunning, sculptRunning, updateJob, onVoiceReady]);

  const busy = disabled || sculptRunning || queueRunning;
  const pendingCount = jobs.filter((j) => (j.source || '').trim()).length;

  return (
    <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
            <ListOrdered className="h-4 w-4 shrink-0" />
            Merge queue
          </p>
          <p className="text-[11px] text-muted-foreground mt-1 leading-relaxed">
            Queue multiple merges and run them one after another. Saved in this browser.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Button type="button" variant="outline" size="sm" className="h-8 text-xs" disabled={busy} onClick={addFromCurrent}>
            <Plus className="h-3 w-3 mr-1" />
            Add current
          </Button>
          <Button type="button" variant="ghost" size="sm" className="h-8 text-xs" disabled={busy} onClick={addEmpty}>
            Add blank
          </Button>
        </div>
      </div>

      {jobs.length === 0 ? (
        <p className="text-xs text-muted-foreground">No queued merges. Configure voices above, then &quot;Add current&quot;.</p>
      ) : (
        <ul className="space-y-2">
          {jobs.map((job, idx) => (
            <li
              key={job.id}
              className="rounded-md border border-border/50 bg-card/50 px-2 py-2 text-xs space-y-2"
            >
              <div className="flex flex-wrap items-center gap-2 justify-between">
                <span className="font-medium text-foreground truncate max-w-[70%]" title={jobLabel(job)}>
                  {idx + 1}. {jobLabel(job)}
                </span>
                <span
                  className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    job.status === 'done'
                      ? 'bg-emerald-500/15 text-emerald-700'
                      : job.status === 'error'
                        ? 'bg-destructive/15 text-destructive'
                        : job.status === 'running'
                          ? 'bg-primary/15 text-primary'
                          : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {statusBadge(job.status)}
                </span>
              </div>
              {job.statusMessage && job.status === 'running' ? (
                <p className="text-[10px] text-muted-foreground truncate">{job.statusMessage}</p>
              ) : null}
              {job.error ? <p className="text-[10px] text-destructive">{job.error}</p> : null}
              {job.resultVoiceId ? (
                <p className="text-[10px] text-muted-foreground break-all">
                  → {job.resultVoiceId}
                </p>
              ) : null}
              <div className="grid gap-2 sm:grid-cols-2">
                <div className="space-y-1 sm:col-span-2">
                  <Label className="text-[10px]">Sources (one path per line)</Label>
                  <textarea
                    className="w-full min-h-[56px] rounded-md border border-input bg-background px-2 py-1.5 font-mono text-[11px]"
                    value={job.source}
                    onChange={(e) => updateJob(job.id, { source: e.target.value, status: 'pending' })}
                    disabled={busy}
                    placeholder={'path_a.wav\npath_b.wav'}
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Output name</Label>
                  <Input
                    className="h-8 text-xs"
                    value={job.outputName}
                    onChange={(e) => updateJob(job.id, { outputName: e.target.value })}
                    disabled={busy}
                    placeholder="merged_voice"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px]">Morph balance % (2 clips)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    step={0.1}
                    className="h-8 text-xs font-mono"
                    value={job.morphBalance}
                    onChange={(e) => {
                      const n = parseFloat(e.target.value);
                      if (!Number.isFinite(n)) return;
                      updateJob(job.id, { morphBalance: Math.min(100, Math.max(0, n)) });
                    }}
                    disabled={busy}
                  />
                </div>
              </div>
              <div className="flex flex-wrap gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2"
                  disabled={busy || idx === 0}
                  onClick={() => moveJob(job.id, -1)}
                  aria-label="Move up"
                >
                  <ChevronUp className="h-3.5 w-3.5" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2"
                  disabled={busy || idx === jobs.length - 1}
                  onClick={() => moveJob(job.id, 1)}
                  aria-label="Move down"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-destructive hover:text-destructive"
                  disabled={busy}
                  onClick={() => removeJob(job.id)}
                  aria-label="Remove"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {queueMessage ? (
        <p className="text-xs text-muted-foreground flex items-center gap-2">
          {queueRunning ? <Loader2 className="h-3 w-3 animate-spin shrink-0" /> : null}
          {queueMessage}
          {queueRunning && queueIndex >= 0 ? ` (${queueIndex + 1}/${pendingCount})` : null}
        </p>
      ) : null}

      {queueError ? (
        <Alert variant="destructive">
          <AlertTitle className="text-sm">Queue stopped</AlertTitle>
          <AlertDescription className="text-xs">{queueError}</AlertDescription>
        </Alert>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          size="sm"
          disabled={busy || pendingCount === 0}
          onClick={runQueue}
        >
          {queueRunning ? (
            <>
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              Running queue…
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 mr-1.5" />
              Run queue ({pendingCount})
            </>
          )}
        </Button>
        {queueRunning ? (
          <Button type="button" size="sm" variant="outline" onClick={cancelQueue}>
            <Square className="h-3.5 w-3.5 mr-1.5" />
            Cancel
          </Button>
        ) : null}
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="text-xs"
          disabled={busy || jobs.length === 0}
          onClick={() => {
            if (!window.confirm('Clear all queued merges?')) return;
            setJobs([]);
            setQueueMessage('');
            setQueueError(null);
          }}
        >
          Clear queue
        </Button>
      </div>
    </div>
  );
}
