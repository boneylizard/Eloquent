import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Cable,
  CheckCircle2,
  AlertCircle,
  Copy,
  ExternalLink,
  Image,
  Mic2,
  RefreshCw,
  Server,
  Volume2,
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';

const BRIDGE_REPOSITORY = 'https://github.com/boneylizard/mirid-sillytavern-bridge';

async function readJson(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.message || `${response.status} ${response.statusText}`);
  return body;
}

function countModels(body) {
  if (Array.isArray(body?.data)) return body.data.length;
  if (Array.isArray(body?.models)) return body.models.length;
  return 0;
}

function countVoices(body) {
  return ['kokoro_voices', 'chatterbox_voices', 'voices']
    .reduce((total, key) => total + (Array.isArray(body?.[key]) ? body[key].length : 0), 0);
}

function StatusCard({ icon: Icon, title, state, detail }) {
  const ready = state === 'ready';
  return (
    <article className="rounded-2xl border border-border/70 bg-card/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <Icon className="h-5 w-5 text-primary" />
        <Badge variant={ready ? 'default' : 'outline'}>{ready ? 'Ready' : state === 'checking' ? 'Checking' : 'Needs attention'}</Badge>
      </div>
      <h2 className="mt-4 font-semibold">{title}</h2>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{detail}</p>
    </article>
  );
}

function CopyRow({ label, value, onCopy }) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-xl border border-border/70 bg-background/60 p-2">
      <div className="min-w-0 flex-1">
        <p className="text-[10px] uppercase tracking-[.16em] text-muted-foreground">{label}</p>
        <code className="block truncate text-xs text-foreground">{value}</code>
      </div>
      <Button type="button" variant="ghost" size="icon" onClick={() => onCopy(value, label)} aria-label={`Copy ${label}`}>
        <Copy className="h-4 w-4" />
      </Button>
    </div>
  );
}

export default function SillyTavernSetupPage() {
  const {
    PRIMARY_API_URL,
    primaryModel,
    settings,
    setActiveTab,
    openSettingsTab,
    updateSettings,
  } = useApp();
  const [checks, setChecks] = useState({});
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState('');

  const addresses = useMemo(() => ({
    base: String(PRIMARY_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, ''),
    text: `${String(PRIMARY_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')}/v1`,
  }), [PRIMARY_API_URL]);

  const copy = useCallback(async (value, label) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(label);
      window.setTimeout(() => setCopied(''), 1800);
    } catch {
      setError(`Mirid could not copy ${label.toLowerCase()}. Select the text and copy it manually.`);
    }
  }, []);

  const runChecks = useCallback(async () => {
    setChecking(true);
    setError('');
    const requests = {
      core: readJson(`${addresses.base}/integrations/sillytavern/capabilities`),
      text: readJson(`${addresses.base}/v1/models`),
      speech: readJson(`${addresses.base}/tts/voices`),
      transcription: readJson(`${addresses.base}/stt/available-engines`),
      images: readJson(`${addresses.base}/sdapi/v1/sd-models`),
    };
    const entries = await Promise.all(Object.entries(requests).map(async ([key, promise]) => {
      try {
        return [key, { state: 'ready', data: await promise }];
      } catch (reason) {
        return [key, { state: 'error', error: reason?.message || String(reason) }];
      }
    }));
    const next = Object.fromEntries(entries);
    setChecks(next);
    if (next.core?.state !== 'ready') setError(next.core?.error || 'Mirid could not test its SillyTavern endpoints.');
    setChecking(false);
  }, [addresses.base]);

  useEffect(() => {
    void runChecks();
  }, [runChecks]);

  const modelCount = countModels(checks.text?.data);
  const voiceCount = countVoices(checks.speech?.data);
  const transcriptionCount = Array.isArray(checks.transcription?.data?.available_engines)
    ? checks.transcription.data.available_engines.length
    : 0;
  const imageCount = Array.isArray(checks.images?.data) ? checks.images.data.length : 0;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-5 pb-14">
      <header className="rounded-3xl border border-border/70 bg-card/70 p-6 md:p-8">
        <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
          <div className="max-w-3xl">
            <p className="text-[11px] uppercase tracking-[.24em] text-muted-foreground">SillyTavern engine</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight md:text-4xl">Connect SillyTavern to Mirid</h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground md:text-base">
              SillyTavern remains your interface. Mirid supplies the model connection, voices, transcription and local image engine.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={runChecks} disabled={checking}>
              <RefreshCw className={`mr-2 h-4 w-4 ${checking ? 'animate-spin' : ''}`} />Test Mirid
            </Button>
            <Button variant="ghost" onClick={() => setActiveTab('chat')}>Use Mirid normally</Button>
          </div>
        </div>
      </header>

      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Keep port 8000 free for Mirid</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>If SillyTavern uses port 8000, close it, open its <code>config.yaml</code>, set <code>port: 8001</code>, then restart it. Mirid’s main engine must remain on 8000.</p>
          <Button variant="outline" size="sm" onClick={() => copy('port: 8001', 'SillyTavern port setting')}>
            <Copy className="mr-2 h-3.5 w-3.5" />{copied === 'SillyTavern port setting' ? 'Copied' : 'Copy setting'}
          </Button>
        </AlertDescription>
      </Alert>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Mirid is not ready to serve SillyTavern</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatusCard
          icon={Server}
          title="Text"
          state={checking ? 'checking' : checks.text?.state}
          detail={modelCount ? `${modelCount} model${modelCount === 1 ? '' : 's'} visible to SillyTavern.` : primaryModel ? 'The selected model will appear when the server catalogue refreshes.' : 'Choose a model before connecting SillyTavern.'}
        />
        <StatusCard
          icon={Volume2}
          title="Speech"
          state={checking ? 'checking' : checks.speech?.state}
          detail={voiceCount ? `${voiceCount} voice${voiceCount === 1 ? '' : 's'} available.` : 'The speech endpoint answered, but no voices were listed.'}
        />
        <StatusCard
          icon={Mic2}
          title="Transcription"
          state={checking ? 'checking' : checks.transcription?.state}
          detail={transcriptionCount ? `${transcriptionCount} transcription engine${transcriptionCount === 1 ? '' : 's'} available.` : 'Install or configure a transcription engine to use the microphone bridge.'}
        />
        <StatusCard
          icon={Image}
          title="Images"
          state={checking ? 'checking' : checks.images?.state}
          detail={imageCount ? `${imageCount} image model${imageCount === 1 ? '' : 's'} available.` : 'The image API is present; install an image model before generating.'}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-3xl border border-border/70 bg-card/60 p-5 md:p-6">
          <div className="flex items-center gap-2">
            <Cable className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">1. Install Mirid Bridge</h2>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            In SillyTavern, open Extensions, choose Install Extension, and paste the repository address.
          </p>
          <div className="mt-4">
            <CopyRow label="Extension repository" value={BRIDGE_REPOSITORY} onCopy={copy} />
          </div>
          <a className="mt-4 inline-flex items-center gap-2 text-sm text-primary hover:underline" href={BRIDGE_REPOSITORY} target="_blank" rel="noreferrer">
            View Bridge source <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </article>

        <article className="rounded-3xl border border-border/70 bg-card/60 p-5 md:p-6">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">2. Connect the services</h2>
          </div>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            Open Mirid Bridge in SillyTavern, paste the current Mirid address below, then select Test connection. Use the same address for SillyTavern’s manual image settings.
          </p>
          <div className="mt-4 space-y-2">
            <CopyRow label="Mirid address" value={addresses.base} onCopy={copy} />
            <CopyRow label="Custom OpenAI-compatible URL" value={addresses.text} onCopy={copy} />
            <CopyRow label="Automatic1111 URL" value={addresses.base} onCopy={copy} />
          </div>
        </article>
      </section>

      <section className="flex flex-col gap-3 rounded-2xl border border-border/70 bg-card/60 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-semibold">Choose what Mirid will serve</h2>
          <p className="mt-1 text-xs text-muted-foreground">A model is required for text. Speech, transcription and images remain optional.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => openSettingsTab('models', { forceWindow: false })}>{primaryModel ? 'Change model' : 'Choose a model'}</Button>
          <Button
            disabled={settings?.sillyTavernSetupCompleted === true}
            onClick={() => updateSettings({ sillyTavernSetupCompleted: true })}
          >
            {settings?.sillyTavernSetupCompleted === true ? 'Guide complete' : 'Mark guide complete'}
          </Button>
        </div>
      </section>
    </div>
  );
}
