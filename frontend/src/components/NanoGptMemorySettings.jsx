import React from 'react';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { BookOpen, Sparkles } from 'lucide-react';

/**
 * Settings → NanoGPT memory — Context Memory (NanoGPT-hosted), not Eloquent's local memory files.
 */
export default function NanoGptMemorySettings({ settings, updateSettings }) {
  const enabled = settings.nanoGptContextMemoryEnabled === true;
  const mode = settings.nanoGptContextMemoryMode === 'suffix' ? 'suffix' : 'header';
  const days = Math.min(
    365,
    Math.max(1, parseInt(settings.nanoGptContextMemoryExpirationDays, 10) || 30)
  );

  return (
    <div className="space-y-6">
      <Alert className="border-primary/30 bg-muted/20">
        <Sparkles className="h-4 w-4" />
        <AlertTitle className="text-sm">What this does</AlertTitle>
        <AlertDescription className="text-xs leading-relaxed space-y-2 mt-2">
          <p>
            When your <strong>LLM Settings → Custom API endpoint</strong> points at{' '}
            <code className="text-[11px] px-1 rounded bg-muted">nano-gpt.com</code>, Eloquent can ask NanoGPT to use{' '}
            <strong>Context Memory</strong> on each chat completion (long-thread compression on their side). This is{' '}
            <em>separate</em> from Eloquent&apos;s profile / agentic / rolling memory.
          </p>
          <p className="text-muted-foreground">
            Official docs:{' '}
            <a
              className="underline font-medium"
              href="https://docs.nano-gpt.com/api-reference/miscellaneous/context-memory"
              target="_blank"
              rel="noreferrer"
            >
              Context Memory
            </a>
            . Billing may include extra memory tokens — check your NanoGPT dashboard.
          </p>
        </AlertDescription>
      </Alert>

      <div className="rounded-xl border border-border/70 bg-card/50 p-4 space-y-4">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <Label className="text-base font-semibold">Enable NanoGPT Context Memory</Label>
            <p className="text-xs text-muted-foreground max-w-xl">
              Off by default. Turn on only when your primary model routes through a NanoGPT API endpoint.
            </p>
          </div>
          <Switch
            checked={enabled}
            onCheckedChange={(v) => updateSettings({ nanoGptContextMemoryEnabled: !!v })}
          />
        </div>

        {enabled ? (
          <>
            <div className="space-y-2 max-w-md">
              <Label className="text-sm">How to enable</Label>
              <Select
                value={mode}
                onValueChange={(v) => updateSettings({ nanoGptContextMemoryMode: v === 'suffix' ? 'suffix' : 'header' })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="header">HTTP headers — memory: true (recommended)</SelectItem>
                  <SelectItem value="suffix">Model suffix — :memory-{'{days}'}</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Headers leave your model id unchanged; suffix appends NanoGPT&apos;s memory token to the model name sent upstream.
              </p>
            </div>

            <div className="space-y-2 max-w-xs">
              <Label htmlFor="ngm-days">Retention (days)</Label>
              <Input
                id="ngm-days"
                type="number"
                min={1}
                max={365}
                value={days}
                onChange={(e) =>
                  updateSettings({
                    nanoGptContextMemoryExpirationDays: Math.min(365, Math.max(1, parseInt(e.target.value, 10) || 30)),
                  })
                }
              />
              <p className="text-[11px] text-muted-foreground">
                NanoGPT default is 30 days; you can set 1–365 per their docs.
              </p>
            </div>
          </>
        ) : null}
      </div>

      <Alert>
        <BookOpen className="h-4 w-4" />
        <AlertTitle className="text-sm">Quick checklist</AlertTitle>
        <AlertDescription>
          <ol className="list-decimal pl-5 text-xs space-y-1 text-muted-foreground">
            <li>Configure your NanoGPT endpoint under <strong>LLM Settings → Custom API Endpoints</strong> (URL must include nano-gpt.com).</li>
            <li>Turn this toggle on and choose header vs suffix.</li>
            <li>Send a normal chat; Eloquent adds the memory flags only on that endpoint.</li>
          </ol>
        </AlertDescription>
      </Alert>
    </div>
  );
}
