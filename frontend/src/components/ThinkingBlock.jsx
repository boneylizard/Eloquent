import React, { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
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

  return (
    <div
      className={cn(
        'mb-2 w-full rounded-lg border border-[rgba(120,170,220,0.28)] bg-[rgba(16,17,19,0.85)]',
        className
      )}
    >
      <button
        type="button"
        onClick={() => setOpenAndNotify((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs text-[rgba(148,163,184,0.95)] hover:text-slate-100"
        aria-expanded={open}
      >
        <span className="inline-flex items-center gap-2">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="uppercase tracking-[0.18em]">Reasoning</span>
          <span className="text-[rgba(148,163,184,0.7)]">·</span>
          <span className="font-medium">{elapsedLabel}</span>
        </span>
        {streaming && (
          <span className="rounded-full border border-[rgba(120,170,220,0.35)] bg-[#101113] px-2 py-0.5 text-[10px] uppercase tracking-[0.18em]">
            thinking…
          </span>
        )}
      </button>
      {open && (
        <div className="px-3 pb-3">
          <pre className="whitespace-pre-wrap break-words text-xs text-slate-200/90 leading-relaxed">
            {reasoningText || ''}
          </pre>
        </div>
      )}
    </div>
  );
}

