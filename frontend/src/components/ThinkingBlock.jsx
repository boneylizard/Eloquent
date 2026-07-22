import React, { useEffect, useMemo, useState, useRef } from 'react';
import { ChevronDown, ChevronRight, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

function formatSeconds(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${s}s`;
}

export default function ThinkingBlock({
  reasoningText,
  enabled = false,
  streaming = false,
  startedAtMs = null,
  finishedSeconds = null,
  className,
  onOpenChange,
}) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const wasStreamingRef = useRef(false);

  const setOpenAndNotify = (nextOpen) => {
    setOpen((prev) => {
      const value = typeof nextOpen === 'function' ? nextOpen(prev) : nextOpen;
      if (value !== prev && typeof onOpenChange === 'function') {
        onOpenChange(value);
      }
      return value;
    });
  };

  useEffect(() => {
    if (!enabled) return;
    if (streaming && !wasStreamingRef.current) {
      setOpen(true);
    }
    wasStreamingRef.current = streaming;
  }, [enabled, streaming]);

  useEffect(() => {
    if (!enabled) return;
    if (!streaming) return;
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [enabled, streaming]);

  const elapsedLabel = useMemo(() => {
    if (!enabled) return null;
    if (typeof finishedSeconds === 'number') return `${Math.max(0, Math.round(finishedSeconds))}s`;
    if (!startedAtMs) return '0s';
    return formatSeconds(now - startedAtMs);
  }, [enabled, finishedSeconds, startedAtMs, now]);

  if (!enabled) return null;

  const hasReasoning = typeof reasoningText === 'string' && reasoningText.trim().length > 0;
  if (!streaming && !hasReasoning) return null;

  const detailsId = 'thinking-block-details';

  if (streaming && !hasReasoning) {
    return (
      <div
        className={cn(
          'mb-2 w-full rounded-lg border animate-pulse',
          className
        )}
        style={{
          borderColor: 'var(--chat-thinking-border)',
          backgroundColor: 'var(--chat-thinking-bg)'
        }}
      >
        <div className="flex items-center justify-between gap-2 px-3 py-2 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-2">
            <Zap size={12} className="animate-pulse text-primary" />
            <span className="uppercase tracking-[0.18em]">Reasoning</span>
            <span className="opacity-50">·</span>
            <span className="font-medium tabular-nums">{elapsedLabel || '0s'}</span>
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] text-primary border-primary/40 bg-primary/10">
            <Zap size={10} className="animate-pulse" />
            thinking…
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        'mb-3 w-full rounded-xl border-2 transition-all duration-300',
        streaming && 'shadow-lg',
        className
      )}
      style={{
        borderColor: streaming ? 'var(--primary)' : 'var(--chat-thinking-border)',
        backgroundColor: 'var(--chat-thinking-bg)'
      }}
    >
      <button
        type="button"
        onClick={() => setOpenAndNotify((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-4 py-2.5 text-left text-xs transition-colors hover:bg-foreground/5 rounded-t-xl text-muted-foreground"
        aria-expanded={open}
        aria-controls={detailsId}
      >
        <span className="inline-flex items-center gap-2">
          {open ? (
            <ChevronDown size={14} className="transition-transform duration-200 text-primary" />
          ) : (
            <ChevronRight size={14} className="transition-transform duration-200 text-primary" />
          )}
          <span className="font-semibold uppercase tracking-[0.2em] text-primary">
            Reasoning
          </span>
          <span className="opacity-50">·</span>
          <span className="font-medium tabular-nums text-foreground">{elapsedLabel}</span>
        </span>
        {streaming && (
          <span className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-primary border-primary/40 bg-primary/10">
            <Zap size={10} className="animate-pulse" />
            thinking…
          </span>
        )}
      </button>
      {open && (
        <div id={detailsId} className={cn('px-4 pb-3', streaming && 'max-h-60 overflow-y-auto')}>
          <pre className="whitespace-pre-wrap break-words text-xs leading-relaxed reasoning-block font-mono"
            style={{ color: 'var(--chat-thinking-text)' }}
          >
            {reasoningText || ''}
          </pre>
        </div>
      )}
    </div>
  );
}
