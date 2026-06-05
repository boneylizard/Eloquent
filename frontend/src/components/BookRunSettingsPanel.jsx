import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  API_CONTEXT_WINDOW_MIN,
  API_CONTEXT_WINDOW_MAX,
  API_CONTEXT_WINDOW_SLIDER_STEP,
  formatApiContextWindowShort,
  clampApiContextWindowTokens,
} from '../config/apiContextLimits';

/**
 * Book run budgets, refusal floor, preamble, and quick-prompt buttons.
 * Persisted via AppContext `updateSettings` (same keys as before under LLM Settings).
 */
export default function BookRunSettingsPanel({ settings, updateSettings, disabled }) {
  const rows = Array.isArray(settings.bookQuickPromptButtons) ? settings.bookQuickPromptButtons : [];

  return (
    <div
      className={`relative mx-auto max-w-3xl space-y-6 text-foreground ${disabled ? 'pointer-events-none' : ''}`}
    >
      {disabled ? (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm font-medium text-amber-950 dark:text-amber-100">
          Run in progress — settings are read-only until it finishes or stops.
        </p>
      ) : null}

      <div>
        <h3 className="text-base font-semibold tracking-tight text-foreground">Run packing</h3>
        <p className="mt-1 text-sm leading-relaxed text-foreground/80">
          Used only while a book queue or overlay quick prompt is active. Rolling memory is forced for that window.
        </p>
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-sm">
        <Label className="text-sm font-semibold text-foreground">
          Book run context window ({formatApiContextWindowShort(settings.bookWritingApiContextTokens ?? 262144)})
        </Label>
        <Slider
          value={[clampApiContextWindowTokens(settings.bookWritingApiContextTokens ?? 262144)]}
          min={API_CONTEXT_WINDOW_MIN}
          max={API_CONTEXT_WINDOW_MAX}
          step={API_CONTEXT_WINDOW_SLIDER_STEP}
          disabled={disabled}
          onValueChange={([v]) =>
            updateSettings({ bookWritingApiContextTokens: clampApiContextWindowTokens(Number(v) || 262144) })
          }
        />
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-sm">
        <Label htmlFor="book-verbatim-overlay" className="text-sm font-semibold text-foreground">
          Verbatim dialogue budget (tokens)
        </Label>
        <Input
          id="book-verbatim-overlay"
          type="number"
          min={2048}
          max={999999}
          step={1024}
          disabled={disabled}
          value={settings.bookWritingVerbatimTokenBudget ?? 98304}
          onChange={(e) =>
            updateSettings({
              bookWritingVerbatimTokenBudget: Math.max(2048, parseInt(e.target.value, 10) || 98304),
            })
          }
          className="max-w-xs"
        />
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-sm">
        <Label htmlFor="book-refusal-overlay" className="text-sm font-semibold text-foreground">
          Treat reply shorter than (characters) as refusal → retry
        </Label>
        <Input
          id="book-refusal-overlay"
          type="number"
          min={200}
          max={20000}
          step={100}
          disabled={disabled}
          value={settings.bookRefusalMaxChars ?? 2200}
          onChange={(e) =>
            updateSettings({ bookRefusalMaxChars: Math.max(200, parseInt(e.target.value, 10) || 2200) })
          }
          className="max-w-xs"
        />
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-card p-4 shadow-sm">
        <Label htmlFor="book-preamble-overlay" className="text-sm font-semibold text-foreground">
          Word-floor preamble (first chapter of each run only)
        </Label>
        <Textarea
          id="book-preamble-overlay"
          className="text-sm min-h-[100px]"
          disabled={disabled}
          value={settings.bookWordFloorPreamble ?? ''}
          onChange={(e) => updateSettings({ bookWordFloorPreamble: e.target.value })}
        />
      </div>

      <div className="space-y-4 rounded-lg border border-border bg-card p-4 shadow-sm">
        <Label className="text-sm font-semibold text-foreground">Quick prompt buttons</Label>
        <p className="text-sm leading-relaxed text-foreground/80">
          One tap sends this text as a user message (same pipeline as chapters). Shown on the Chapters tab when at least one has a label or body.
        </p>
        {rows.map((row, idx) => (
          <div key={row.id || idx} className="flex flex-col gap-2 rounded-md border border-border bg-muted p-3">
            <div className="flex flex-col sm:flex-row gap-2">
              <Input
                placeholder="Button label"
                className="text-sm sm:w-40"
                disabled={disabled}
                value={row.label || ''}
                onChange={(e) => {
                  const next = [...rows];
                  next[idx] = { ...row, label: e.target.value };
                  updateSettings({ bookQuickPromptButtons: next });
                }}
              />
              <Textarea
                placeholder="Full prompt text"
                className="text-sm flex-1 min-h-[56px]"
                disabled={disabled}
                value={row.text || ''}
                onChange={(e) => {
                  const next = [...rows];
                  next[idx] = { ...row, text: e.target.value };
                  updateSettings({ bookQuickPromptButtons: next });
                }}
              />
            </div>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="self-start"
              disabled={disabled}
              onClick={() => {
                const next = rows.filter((_, i) => i !== idx);
                updateSettings({ bookQuickPromptButtons: next });
              }}
            >
              Remove
            </Button>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          onClick={() => {
            const next = [
              ...rows,
              { id: `qb_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, label: '', text: '' },
            ];
            updateSettings({ bookQuickPromptButtons: next });
          }}
        >
          Add quick prompt
        </Button>
      </div>
    </div>
  );
}
