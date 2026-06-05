import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  X,
  RefreshCw,
  Sparkles,
  Loader2,
  RotateCcw,
  GripVertical,
  Volume2,
  ChevronDown,
  ChevronRight,
  FileText,
  Layers,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { getBackendUrl } from '../config/api';
import { parseCallModeAboutResponse, splitUnstructuredAboutText } from '../utils/callModeCharacterAbout';
import {
  loadIntelTapMode,
  saveIntelTapMode,
} from '../utils/callModeIntelTts';

const LAYOUT_STORAGE_KEY = 'LiangLocal-call-about-card-layout';

const KOKORO_VOICE_FALLBACK = [
  { id: 'af_heart', name: 'Am. English Female (Heart)' },
  { id: 'af_sarah', name: 'Am. English Female (Sarah)' },
  { id: 'af_nova', name: 'Am. English Female (Nova)' },
  { id: 'am_adam', name: 'Am. English Male (Adam)' },
];

const CHIP_ACCENTS = {
  essence: 'border-cyan-400/35 bg-cyan-500/15 text-cyan-100 shadow-[0_0_24px_rgba(34,211,238,0.12)]',
  presence: 'border-violet-400/35 bg-violet-500/15 text-violet-100 shadow-[0_0_24px_rgba(167,139,250,0.12)]',
  relationship: 'border-fuchsia-400/35 bg-fuchsia-500/15 text-fuchsia-100 shadow-[0_0_24px_rgba(232,121,249,0.12)]',
  current_state: 'border-amber-400/35 bg-amber-500/15 text-amber-100 shadow-[0_0_24px_rgba(251,191,36,0.12)]',
  story_so_far: 'border-blue-400/35 bg-blue-500/15 text-blue-100 shadow-[0_0_24px_rgba(96,165,250,0.12)]',
  watch_for: 'border-rose-400/35 bg-rose-500/15 text-rose-100 shadow-[0_0_24px_rgba(251,113,133,0.12)]',
  voice_note: 'border-emerald-400/35 bg-emerald-500/15 text-emerald-100 shadow-[0_0_24px_rgba(52,211,153,0.12)]',
  headline: 'border-white/25 bg-white/10 text-white/90 shadow-[0_0_20px_rgba(255,255,255,0.06)]',
};

const PANEL_ACCENTS = {
  essence: 'border-cyan-400/25 bg-cyan-500/10',
  presence: 'border-violet-400/25 bg-violet-500/10',
  relationship: 'border-fuchsia-400/25 bg-fuchsia-500/10',
  current_state: 'border-amber-400/25 bg-amber-500/10',
  story_so_far: 'border-blue-400/25 bg-blue-500/10',
  watch_for: 'border-rose-400/25 bg-rose-500/10',
  voice_note: 'border-emerald-400/25 bg-emerald-500/10',
  headline: 'border-white/20 bg-white/8',
};

/** Default scatter positions (% of viewport). Margins clear the centered portrait. */
const SCATTER_SLOTS = [
  {
    id: 'essence',
    label: 'Essence',
    field: 'essence',
    defaultPosition: { x: 2, y: 12 },
    align: 'start',
    maxWidth: 'min(148px, 14vw)',
    expandedMaxWidth: 'min(300px, 28vw)',
  },
  {
    id: 'presence',
    label: 'On this call',
    field: 'presence',
    defaultPosition: { x: 98, y: 12 },
    align: 'end',
    maxWidth: 'min(148px, 14vw)',
    expandedMaxWidth: 'min(300px, 28vw)',
  },
  {
    id: 'relationship',
    label: 'With you',
    field: 'relationship',
    defaultPosition: { x: 2, y: 28 },
    align: 'start',
    maxWidth: 'min(148px, 14vw)',
    expandedMaxWidth: 'min(300px, 28vw)',
  },
  {
    id: 'current_state',
    label: 'Right now',
    field: 'current_state',
    defaultPosition: { x: 98, y: 28 },
    align: 'end',
    maxWidth: 'min(148px, 14vw)',
    expandedMaxWidth: 'min(300px, 28vw)',
  },
  {
    id: 'story_so_far',
    label: 'Story so far',
    field: 'story_so_far',
    defaultPosition: { x: 2, y: 48 },
    align: 'start',
    maxWidth: 'min(160px, 15vw)',
    expandedMaxWidth: 'min(320px, 30vw)',
  },
  {
    id: 'watch_for',
    label: 'Watch for',
    field: 'watch_for',
    defaultPosition: { x: 98, y: 48 },
    align: 'end',
    maxWidth: 'min(148px, 14vw)',
    expandedMaxWidth: 'min(300px, 28vw)',
  },
  {
    id: 'voice_note',
    label: 'Voice note',
    field: 'voice_note',
    defaultPosition: { x: 50, y: 72 },
    align: 'center',
    maxWidth: 'min(160px, 15vw)',
    expandedMaxWidth: 'min(340px, 32vw)',
    centered: true,
  },
];

const ALIGN_TRANSFORM = {
  start: 'none',
  end: 'translateX(-100%)',
  center: 'translateX(-50%)',
};

const TAP_MODE_OPTIONS = [
  { id: 'listen', label: 'Listen', short: 'TTS', icon: Volume2, title: 'Tap headings to play narrator audio' },
  { id: 'read', label: 'Read', short: 'Text', icon: FileText, title: 'Tap headings to expand full text' },
  { id: 'both', label: 'Both', short: 'Both', icon: Layers, title: 'Tap headings to expand and play audio' },
];

function loadCustomLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return {};
    const out = {};
    for (const [id, pos] of Object.entries(parsed)) {
      if (pos && typeof pos.x === 'number' && typeof pos.y === 'number') {
        out[id] = { x: pos.x, y: pos.y };
      }
    }
    return out;
  } catch {
    return {};
  }
}

function saveCustomLayout(layout) {
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch {
    /* ignore quota / private mode */
  }
}

function clampPercent(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function resolvePosition(slot, customLayout) {
  const custom = customLayout[slot.id];
  if (custom) return custom;
  return slot.defaultPosition;
}

function scatterStyle(slot, position, isExpanded) {
  return {
    left: `${position.x}%`,
    top: `${position.y}%`,
    maxWidth: isExpanded ? (slot.expandedMaxWidth || slot.maxWidth) : slot.maxWidth,
    transform: ALIGN_TRANSFORM[slot.align] || 'none',
  };
}

function IntelTapModeToggle({ value, onChange }) {
  return (
    <div
      className="flex h-10 items-center rounded-full border border-white/15 bg-black/70 p-0.5 backdrop-blur-md"
      role="group"
      aria-label="Insight tap behavior"
    >
      {TAP_MODE_OPTIONS.map((opt) => {
        const Icon = opt.icon;
        const active = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            title={opt.title}
            aria-pressed={active}
            onClick={() => onChange(opt.id)}
            className={cn(
              'flex h-[calc(100%-2px)] items-center gap-1 rounded-full px-2 text-[9px] font-semibold uppercase tracking-[0.1em] transition-colors sm:px-2.5',
              active ? 'bg-white/15 text-white' : 'text-white/50 hover:text-white/80'
            )}
          >
            <Icon className="h-3 w-3 shrink-0" />
            <span className="hidden sm:inline">{opt.label}</span>
            <span className="sm:hidden">{opt.short}</span>
          </button>
        );
      })}
    </div>
  );
}

function IntelExpandedBody({ id, label, text, centered, onClose }) {
  return (
    <article
      className={cn(
        'pointer-events-auto mt-1.5 rounded-xl border px-3 py-2.5 backdrop-blur-md',
        PANEL_ACCENTS[id] || 'border-white/15 bg-black/70',
        centered && 'text-center'
      )}
    >
      <div className={cn('flex items-start gap-2', centered && 'justify-center')}>
        <p className="min-w-0 flex-1 text-sm leading-relaxed text-white/92">{text}</p>
        <button
          type="button"
          className="shrink-0 rounded-full p-1 text-white/40 hover:bg-white/10 hover:text-white/80"
          title="Collapse"
          aria-label={`Collapse ${label}`}
          onClick={(e) => {
            e.stopPropagation();
            onClose?.();
          }}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </article>
  );
}

function IntelInsightChip({
  id,
  label,
  isPlaying,
  isExpanded,
  playDisabled,
  onHeadingClick,
  onToggleExpand,
  centered,
  className,
}) {
  return (
    <div className={cn('flex items-center gap-0.5', centered && 'justify-center', className)}>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onToggleExpand?.();
        }}
        title="Expand or collapse full text"
        className={cn(
          'pointer-events-auto flex h-7 w-6 shrink-0 items-center justify-center rounded-full border border-white/10 bg-black/50 text-white/45 backdrop-blur-sm hover:bg-white/10 hover:text-white/80',
          isExpanded && 'bg-white/15 text-white/90'
        )}
        aria-expanded={isExpanded}
      >
        {isExpanded ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>
      <button
        type="button"
        onClick={onHeadingClick}
        title={`Insight: ${label}`}
        className={cn(
          'pointer-events-auto rounded-full border px-2.5 py-1 text-[9px] font-semibold uppercase tracking-[0.16em] backdrop-blur-md transition-all',
          CHIP_ACCENTS[id] || 'border-white/20 bg-black/60 text-white/85',
          isPlaying && 'ring-2 ring-white/50 scale-[1.04]',
          isExpanded && 'ring-1 ring-white/35',
          'hover:scale-[1.03] hover:ring-1 hover:ring-white/30 cursor-pointer'
        )}
      >
        {label}
      </button>
    </div>
  );
}

function DraggableIntelChip({
  slot,
  text,
  isPlaying,
  isExpanded,
  playDisabled,
  onHeadingClick,
  onToggleExpand,
  customLayout,
  onLayoutChange,
}) {
  const position = resolvePosition(slot, customLayout);
  const dragRef = useRef(null);
  const draggingRef = useRef(false);
  const movedRef = useRef(false);
  const [isDragging, setIsDragging] = useState(false);
  const grabOffsetRef = useRef({ x: 0, y: 0 });

  const endDrag = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setIsDragging(false);
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    window.setTimeout(() => {
      movedRef.current = false;
    }, 0);
  }, []);

  useEffect(() => {
    const onPointerMove = (e) => {
      if (!draggingRef.current) return;
      movedRef.current = true;
      const x = ((e.clientX - grabOffsetRef.current.x) / window.innerWidth) * 100;
      const y = ((e.clientY - grabOffsetRef.current.y) / window.innerHeight) * 100;
      onLayoutChange(slot.id, {
        x: clampPercent(x, 0, 96),
        y: clampPercent(y, 6, 88),
      });
    };
    const onPointerUp = () => endDrag();

    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);
    document.addEventListener('pointercancel', onPointerUp);
    return () => {
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerup', onPointerUp);
      document.removeEventListener('pointercancel', onPointerUp);
    };
  }, [slot.id, onLayoutChange, endDrag]);

  const onGripPointerDown = useCallback((e) => {
    if (e.button !== 0) return;
    const el = dragRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    grabOffsetRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    draggingRef.current = true;
    movedRef.current = false;
    setIsDragging(true);
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'grabbing';
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleHeadingClick = useCallback(() => {
    if (movedRef.current) return;
    onHeadingClick?.();
  }, [onHeadingClick]);

  return (
    <div
      ref={dragRef}
      className="absolute z-[1] flex flex-col"
      style={scatterStyle(slot, position, isExpanded)}
      data-about-card={slot.id}
    >
      <div className={cn('flex items-start gap-0.5', slot.centered && 'justify-center')}>
        <span
          className={cn(
            'pointer-events-auto flex h-7 w-5 shrink-0 cursor-grab items-center justify-center rounded-full border border-white/10 bg-black/50 text-white/30 backdrop-blur-sm active:cursor-grabbing touch-none select-none',
            isDragging && 'ring-1 ring-white/25'
          )}
          onPointerDown={onGripPointerDown}
          title="Drag to reposition"
          aria-hidden
        >
          <GripVertical className="h-3 w-3" />
        </span>
        <IntelInsightChip
          id={slot.id}
          label={slot.label}
          isPlaying={isPlaying}
          isExpanded={isExpanded}
          playDisabled={playDisabled}
          onHeadingClick={handleHeadingClick}
          onToggleExpand={onToggleExpand}
          centered={slot.centered}
          className={isDragging ? 'shadow-lg' : undefined}
        />
      </div>
      {isExpanded && (
        <IntelExpandedBody
          id={slot.id}
          label={slot.label}
          text={text}
          centered={slot.centered}
          onClose={onToggleExpand}
        />
      )}
    </div>
  );
}

function SkeletonChip({ slot, customLayout }) {
  const position = resolvePosition(slot, customLayout);
  return (
    <div
      className="pointer-events-none absolute h-7 w-24 animate-pulse rounded-full border border-white/10 bg-black/55 backdrop-blur-md"
      style={scatterStyle(slot, position, false)}
      aria-hidden
    />
  );
}

function IntelVoicePicker({
  open,
  primaryApiUrl,
  settings,
  intelVoice,
  onIntelVoiceChange,
  characterVoiceLabel,
}) {
  const [expanded, setExpanded] = useState(false);
  const [availableVoices, setAvailableVoices] = useState({ chatterbox_voices: [], kokoro_voices: [] });
  const [isFetching, setIsFetching] = useState(false);

  const ttsEngine = settings?.ttsEngine || 'kokoro';
  const isChatterbox = ttsEngine === 'chatterbox' || ttsEngine === 'chatterbox_turbo';
  const isKokoro = ttsEngine === 'kokoro';

  useEffect(() => {
    if (!open) setExpanded(false);
  }, [open]);

  useEffect(() => {
    if (!expanded || (!isChatterbox && !isKokoro)) return;
    let cancelled = false;
    (async () => {
      setIsFetching(true);
      try {
        const base = primaryApiUrl || getBackendUrl();
        const res = await fetch(`${String(base).replace(/\/+$/, '')}/tts/voices`);
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (!cancelled) setAvailableVoices(data || { chatterbox_voices: [], kokoro_voices: [] });
      } catch (e) {
        console.warn('[CallModeAboutPanel] voices fetch failed', e);
        if (!cancelled) setAvailableVoices({ chatterbox_voices: [], kokoro_voices: [] });
      } finally {
        if (!cancelled) setIsFetching(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [expanded, isChatterbox, isKokoro, primaryApiUrl]);

  const voiceOptions = useMemo(() => {
    if (isChatterbox) return availableVoices?.chatterbox_voices || [];
    if (isKokoro) {
      if (availableVoices?.kokoro_voices?.length) return availableVoices.kokoro_voices;
      return KOKORO_VOICE_FALLBACK;
    }
    return [];
  }, [availableVoices, isChatterbox, isKokoro]);

  if (!isChatterbox && !isKokoro) {
    return (
      <span className="hidden sm:inline text-[10px] text-white/40" title="Insight voice uses global TTS engine">
        Narrator: global engine
      </span>
    );
  }

  const currentLabel =
    voiceOptions.find((v) => (v.id || v.voice_id) === intelVoice)?.name
    || voiceOptions.find((v) => (v.id || v.voice_id) === intelVoice)?.id
    || intelVoice
    || 'Default';

  return (
    <div className="relative">
      <button
        type="button"
        className="flex h-10 max-w-[min(220px,42vw)] items-center gap-1.5 rounded-full border border-white/15 bg-black/70 px-2.5 text-[10px] font-medium uppercase tracking-[0.12em] text-white/85 backdrop-blur-md hover:bg-white/10 sm:max-w-none sm:px-3"
        title="Voice for reading insight headings (separate from character speech)"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <Volume2 className="h-3.5 w-3.5 shrink-0" />
        <span className="hidden truncate sm:inline">Narrator</span>
        <span className="truncate text-white/60 normal-case tracking-normal">{currentLabel}</span>
        <ChevronDown className={cn('h-3 w-3 shrink-0 transition-transform', expanded && 'rotate-180')} />
      </button>
      {expanded && (
        <>
          <button
            type="button"
            className="fixed inset-0 z-[10001] cursor-default bg-transparent"
            aria-label="Close narrator voice menu"
            onClick={() => setExpanded(false)}
          />
          <div className="absolute right-0 top-full z-[10002] mt-2 w-[min(280px,88vw)] rounded-xl border border-white/15 bg-zinc-950/95 p-3 shadow-2xl backdrop-blur-md">
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/50">
              Insight narrator voice
            </p>
            <p className="mt-1 text-[11px] leading-snug text-white/55">
              Used when you tap insight headings. Character call voice
              {characterVoiceLabel ? ` (“${characterVoiceLabel}”)` : ''} is unchanged.
            </p>
            {isFetching ? (
              <p className="mt-3 flex items-center gap-2 text-xs text-white/50">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Loading voices…
              </p>
            ) : (
              <select
                className="mt-3 w-full rounded-lg border border-white/15 bg-black/60 px-2 py-2 text-sm text-white"
                value={intelVoice || ''}
                onChange={(e) => onIntelVoiceChange?.(e.target.value)}
              >
                {isChatterbox && (
                  <option value="default">Default reference</option>
                )}
                {voiceOptions.map((v) => {
                  const id = v.id || v.voice_id || v.name;
                  const name = v.name || id;
                  return (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  );
                })}
              </select>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function resolveDisplayData(result, partialText) {
  if (result?.structured && result.data) return result.data;
  const raw = (result?.rawText || partialText || '').trim();
  if (!raw) return null;
  const parsed = parseCallModeAboutResponse(raw);
  if (parsed.structured && parsed.data) return parsed.data;
  return splitUnstructuredAboutText(raw);
}

function tapModeHint(tapMode, intelDisabled) {
  if (intelDisabled) {
    return 'Insight playback pauses while the character is speaking · expand arrow still works';
  }
  if (tapMode === 'listen') {
    return 'Tap heading for narrator audio · arrow expands text (one at a time)';
  }
  if (tapMode === 'read') {
    return 'Tap heading or arrow to expand full text · one section at a time';
  }
  return 'Tap heading to expand and play narrator audio · arrow toggles text';
}

export default function CallModeAboutPanel({
  open,
  characterName,
  loading,
  partialText,
  result,
  error,
  onClose,
  onRefresh,
  primaryApiUrl,
  settings,
  intelVoice,
  onIntelVoiceChange,
  characterVoiceLabel,
  autoTtsBusy,
  playingIntelSlotId,
  onPlayIntel,
}) {
  const [customLayout, setCustomLayout] = useState(() => loadCustomLayout());
  const [tapMode, setTapMode] = useState(() => loadIntelTapMode('both'));
  const [expandedSlotId, setExpandedSlotId] = useState(null);

  useEffect(() => {
    if (open) setCustomLayout(loadCustomLayout());
    else setExpandedSlotId(null);
  }, [open]);

  useEffect(() => {
    saveIntelTapMode(tapMode);
  }, [tapMode]);

  const handleLayoutChange = useCallback((cardId, pos) => {
    setCustomLayout((prev) => {
      const next = { ...prev, [cardId]: pos };
      saveCustomLayout(next);
      return next;
    });
  }, []);

  const handleResetLayout = useCallback(() => {
    setCustomLayout({});
    try {
      localStorage.removeItem(LAYOUT_STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const displayData = useMemo(
    () => (open ? resolveDisplayData(result, partialText) : null),
    [open, result, partialText]
  );

  const cards = useMemo(() => {
    const list = SCATTER_SLOTS.map((slot) => ({
      ...slot,
      text: displayData?.[slot.field],
    })).filter((c) => c.text && String(c.text).trim());

    const headline = displayData?.headline?.trim();
    if (headline) {
      list.unshift({
        id: 'headline',
        label: 'Summary',
        field: 'headline',
        text: headline,
        defaultPosition: { x: 50, y: 8 },
        align: 'center',
        maxWidth: 'min(120px, 12vw)',
        expandedMaxWidth: 'min(320px, 30vw)',
        centered: true,
      });
    }
    return list;
  }, [displayData]);

  const intelPlayDisabled = Boolean(autoTtsBusy);

  const toggleExpand = useCallback((slotId) => {
    setExpandedSlotId((prev) => (prev === slotId ? null : slotId));
  }, []);

  const handleChipActivate = useCallback(
    (card, { expandOnly = false } = {}) => {
      const id = card.id;
      const text = String(card.text);

      if (expandOnly) {
        toggleExpand(id);
        return;
      }

      if (tapMode === 'read' || tapMode === 'both') {
        setExpandedSlotId((prev) => (prev === id ? null : id));
      }
      if ((tapMode === 'listen' || tapMode === 'both') && !intelPlayDisabled) {
        onPlayIntel?.(id, text);
      }
    },
    [tapMode, intelPlayDisabled, onPlayIntel, toggleExpand]
  );

  if (!open) return null;

  const hasContent = cards.length > 0 || displayData?.themes?.length;

  return (
    <div
      data-character-about-panel
      className="fixed inset-0 z-[10000] pointer-events-none overflow-hidden"
      role="dialog"
      aria-label={`About ${characterName || 'character'}`}
    >
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_38%_55%_at_50%_42%,transparent_0%,transparent_48%,rgba(0,0,0,0.72)_100%)]"
        aria-hidden
      />

      <div className="pointer-events-auto absolute top-6 left-3 right-3 z-10 flex flex-wrap items-center justify-end gap-2 sm:left-auto sm:right-[5.5rem] sm:max-w-[min(100%,48rem)]">
        <IntelTapModeToggle value={tapMode} onChange={setTapMode} />
        <IntelVoicePicker
          open={open}
          primaryApiUrl={primaryApiUrl}
          settings={settings}
          intelVoice={intelVoice}
          onIntelVoiceChange={onIntelVoiceChange}
          characterVoiceLabel={characterVoiceLabel}
        />
        <button
          type="button"
          className="hidden md:flex h-10 items-center gap-2 rounded-full border border-white/15 bg-black/70 px-3 text-[11px] font-medium uppercase tracking-[0.14em] text-white/85 backdrop-blur-md hover:bg-white/10"
          title="Reset heading positions to default"
          onClick={handleResetLayout}
        >
          <RotateCcw className="h-4 w-4" />
          <span className="hidden lg:inline">Reset layout</span>
        </button>
        <button
          type="button"
          className="flex h-10 items-center gap-2 rounded-full border border-white/15 bg-black/70 px-3 text-[11px] font-medium uppercase tracking-[0.14em] text-white/85 backdrop-blur-md hover:bg-white/10 disabled:opacity-40"
          title="Refresh insight"
          disabled={loading}
          onClick={onRefresh}
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          <span className="hidden sm:inline">Refresh</span>
        </button>
        <button
          type="button"
          className="flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-black/70 text-white/85 backdrop-blur-md hover:bg-white/10"
          title="Close"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="pointer-events-none absolute left-1/2 top-[5.25rem] z-10 w-full max-w-md -translate-x-1/2 px-4 text-center">
        <div className="flex items-center justify-center gap-2 text-cyan-300/85">
          <Sparkles className="h-4 w-4 shrink-0" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.32em]">Character insight</span>
        </div>
        {characterName && (
          <h2 className="mt-1 text-base font-semibold text-white sm:text-lg">{characterName}</h2>
        )}
        {displayData?.themes?.length > 0 && (
          <div className="pointer-events-auto mt-2 flex flex-wrap justify-center gap-1">
            {displayData.themes.map((theme) => (
              <span
                key={theme}
                className="rounded-full border border-violet-400/30 bg-violet-500/15 px-2 py-0.5 text-[9px] text-violet-100/90"
              >
                {theme}
              </span>
            ))}
          </div>
        )}
        {cards.length > 0 && (
          <p className="pointer-events-none mt-2 text-[10px] leading-snug text-white/40">
            {tapModeHint(tapMode, intelPlayDisabled)}
          </p>
        )}
      </div>

      {error && (
        <div className="pointer-events-auto absolute left-1/2 top-[8.5rem] z-20 max-w-md -translate-x-1/2 rounded-xl border border-red-400/35 bg-red-950/70 px-4 py-2 text-sm text-red-100 backdrop-blur-md">
          {error}
        </div>
      )}

      {loading && !hasContent && (
        <>
          <div className="hidden md:block">
            {SCATTER_SLOTS.map((slot) => (
              <SkeletonChip key={slot.id} slot={slot} customLayout={customLayout} />
            ))}
          </div>
          <div className="pointer-events-none absolute left-1/2 top-[42%] -translate-x-1/2 text-xs text-white/45">
            Reading character, user, and conversation…
          </div>
          <div className="md:hidden absolute inset-x-3 bottom-28 top-[42%] flex flex-wrap content-start justify-center gap-2 overflow-y-auto overscroll-contain pointer-events-none">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="h-7 w-24 animate-pulse rounded-full border border-white/10 bg-black/55 backdrop-blur-md"
              />
            ))}
          </div>
        </>
      )}

      {cards.length > 0 && (
        <>
          <div className="md:hidden absolute inset-x-3 bottom-24 top-[38%] overflow-y-auto overscroll-contain pointer-events-auto">
            <div className="flex flex-col items-stretch gap-3 pb-4">
              {cards.map((card) => {
                const isExpanded = expandedSlotId === card.id;
                return (
                  <div key={card.id} className="flex flex-col items-center gap-1.5">
                    <IntelInsightChip
                      id={card.id}
                      label={card.label}
                      isPlaying={playingIntelSlotId === card.id}
                      isExpanded={isExpanded}
                      playDisabled={intelPlayDisabled}
                      onHeadingClick={() => handleChipActivate(card)}
                      onToggleExpand={() => handleChipActivate(card, { expandOnly: true })}
                    />
                    {isExpanded && (
                      <IntelExpandedBody
                        id={card.id}
                        label={card.label}
                        text={String(card.text)}
                        centered
                        onClose={() => toggleExpand(card.id)}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="hidden md:block">
            {cards.map((card) => (
              <DraggableIntelChip
                key={card.id}
                slot={card}
                text={String(card.text)}
                isPlaying={playingIntelSlotId === card.id}
                isExpanded={expandedSlotId === card.id}
                playDisabled={intelPlayDisabled}
                onHeadingClick={() => handleChipActivate(card)}
                onToggleExpand={() => handleChipActivate(card, { expandOnly: true })}
                customLayout={customLayout}
                onLayoutChange={handleLayoutChange}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
