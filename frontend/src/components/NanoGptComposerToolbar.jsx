import React, { useState } from 'react';
import { cn } from '@/lib/utils';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';
import { ChevronDown, ChevronUp } from 'lucide-react';

export const NANO_GPT_QUICK_ACTIONS = [
  {
    label: 'Learn',
    description: 'Study a topic or paper',
    prompt: 'Teach me this step by step and quiz me as we go:',
  },
  {
    label: 'Create',
    description: 'Draft, outline, or code',
    prompt: 'Help me draft or outline the following:',
  },
  {
    label: 'Audio',
    description: 'Voice-first call-style chat',
    prompt: 'Let’s talk through this out loud. Summarize each answer briefly and ask clarifying questions:',
  },
  {
    label: 'Explore',
    description: 'Browse documents or transcripts',
    prompt: 'Help me explore and summarise my recent work or transcripts:',
  },
  {
    label: 'API',
    description: 'Test a NanoGPT endpoint',
    prompt: 'Help me design and test a NanoGPT API call with sensible defaults and safety checks:',
  },
];

const MODE_OPTIONS = [
  { id: 'chat', label: 'Chat' },
  { id: 'call', label: 'Call' },
  { id: 'focus', label: 'Focus' },
  { id: 'image', label: 'Image' },
];

function ModeChips({ activeMode, onModeChange, compact }) {
  return (
    <div className="inline-flex items-center gap-1 rounded-full bg-[#181a1f] px-1 py-1 border border-[rgba(120,170,220,0.28)]">
      {MODE_OPTIONS.map((mode) => {
        const isActive = activeMode === mode.id;
        return (
          <button
            key={mode.id}
            type="button"
            onClick={() => onModeChange?.(mode.id)}
            className={cn(
              compact ? 'px-2.5 py-1 text-[11px]' : 'px-3 py-1.5 text-[12px]',
              'rounded-full transition-colors',
              isActive
                ? 'bg-[#78aadc] text-[#050608] shadow-[0_10px_30px_rgba(15,23,42,0.95)]'
                : 'text-[rgba(148,163,184,0.95)] hover:text-slate-100 hover:bg-[#111827]',
            )}
          >
            {mode.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Shared NanoGPT landing controls: mode chips, model picker, quick-action pills.
 * Used on the empty-state landing (variant="landing") and above the composer when chatting.
 */
export default function NanoGptComposerToolbar({
  variant = 'compact',
  embedded = false,
  modelPickerOpen,
  onModelPickerOpenChange,
  currentModelId,
  primaryApiUrl,
  onCapabilities,
  onQuickAction,
  activeMode = 'chat',
  onModeChange,
  className,
}) {
  const isLanding = variant === 'landing';
  const isCompact = variant === 'compact';
  const [quickActionsCollapsed, setQuickActionsCollapsed] = useState(isCompact);

  const shellClass = embedded
    ? 'px-3 pt-2.5 pb-2'
    : isLanding
      ? 'rounded-2xl border border-[rgba(120,170,220,0.35)] bg-[radial-gradient(circle_at_top,_rgba(120,170,220,0.16),transparent_55%),_#101113] shadow-[0_22px_60px_rgba(3,4,10,0.95)] p-3 md:p-4'
      : 'rounded-xl border border-[rgba(120,170,220,0.35)] bg-[#101113]/95 px-3 py-2.5';

  return (
    <div className={cn(shellClass, className)}>
      <div
        className={cn(
          'flex items-center justify-between gap-2',
          isCompact ? 'flex-wrap' : 'mb-3',
        )}
      >
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <span className="text-[11px] uppercase tracking-[0.18em] text-[rgba(148,163,184,0.9)]">
            Mode
          </span>
          <ModeChips activeMode={activeMode} onModeChange={onModeChange} compact={isCompact} />
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[11px] uppercase tracking-[0.18em] text-[rgba(148,163,184,0.9)]">
            Model
          </span>
          <NanoGptModelSelectorPopover
            open={modelPickerOpen}
            onOpenChange={onModelPickerOpenChange}
            currentModelId={currentModelId}
            primaryApiUrl={primaryApiUrl}
            className="relative"
            compact={isCompact}
            onCapabilities={onCapabilities}
            trigger={({ setOpen, display }) => (
              <button
                type="button"
                onClick={() => setOpen(true)}
                className={cn(
                  'inline-flex items-center gap-2 rounded-full bg-[#181a1f] border border-[rgba(120,170,220,0.4)] text-slate-100 hover:border-[rgba(120,170,220,0.8)] transition-colors',
                  isCompact ? 'px-2.5 py-1 text-[11px]' : 'px-3 py-1.5 text-[12px]',
                )}
              >
                <span className="text-sm leading-none">{display?.icon || '⬜'}</span>
                <span className={cn('truncate', isCompact ? 'max-w-[120px]' : 'max-w-[160px]')}>
                  {display?.shortLabel || 'Select model'}
                </span>
              </button>
            )}
          />
        </div>
      </div>

      {isLanding && (
        <div className="rounded-xl border border-[rgba(148,163,184,0.25)] bg-[#0b0c0f]/80 px-3 py-3 text-sm text-slate-200 text-left mb-0">
          <p className="text-[13px]">
            Start typing in the composer below to begin a new conversation.
          </p>
          <p className="mt-1 text-[11px] text-[rgba(148,163,184,0.95)]">
            Press <span className="px-1 py-0.5 rounded bg-[#111827] border border-[rgba(148,163,184,0.5)] text-[10px]">Enter</span> to send ·{' '}
            <span className="px-1 py-0.5 rounded bg-[#111827] border border-[rgba(148,163,184,0.5)] text-[10px]">Shift + Enter</span> for a new line.
          </p>
        </div>
      )}

      {isCompact && (
        <div className="mt-2 flex items-center justify-end">
          <button
            type="button"
            onClick={() => setQuickActionsCollapsed((prev) => !prev)}
            className="inline-flex items-center gap-1 rounded-full border border-[rgba(120,170,220,0.35)] bg-[#101113] px-2.5 py-1 text-[10px] text-[rgba(148,163,184,0.95)] hover:text-slate-100 hover:border-[rgba(120,170,220,0.7)]"
            title={quickActionsCollapsed ? 'Show quick presets' : 'Hide quick presets'}
          >
            {quickActionsCollapsed ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
            {quickActionsCollapsed ? 'Show presets' : 'Hide presets'}
          </button>
        </div>
      )}

      {(!isCompact || !quickActionsCollapsed) && (
        <div
          className={cn(
            'flex gap-2 overflow-x-auto no-scrollbar',
            isLanding ? 'flex-wrap mt-4 -mx-1 px-1 md:gap-3' : 'mt-2.5',
            isCompact && 'md:flex-wrap',
          )}
        >
          {NANO_GPT_QUICK_ACTIONS.map((action) => (
            <button
              key={action.label}
              type="button"
              onClick={() => onQuickAction?.(`${action.prompt}\n\n`)}
              className={cn(
                'group rounded-full border border-[rgba(120,170,220,0.35)] bg-[#101113] text-left text-slate-200 hover:border-[rgba(120,170,220,0.85)] hover:bg-[#14151a] transition-colors flex-shrink-0',
                isLanding
                  ? 'flex-1 min-w-[140px] px-3 py-2.5 text-xs'
                  : 'min-w-[100px] px-2.5 py-1.5 text-[11px]',
              )}
            >
              <div className={cn('flex gap-2', isLanding ? 'items-center justify-between' : 'flex-col items-start')}>
                <span className={cn('font-semibold', isLanding ? 'text-[12px]' : 'text-[11px]')}>
                  {action.label}
                </span>
                <span
                  className={cn(
                    'text-[rgba(148,163,184,0.95)] group-hover:text-slate-200',
                    isLanding ? 'text-[10px]' : 'text-[10px] truncate max-w-full',
                  )}
                >
                  {action.description}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
