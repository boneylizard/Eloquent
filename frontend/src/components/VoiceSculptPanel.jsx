import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { fetchWithTimeout } from '../config/api';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Textarea } from './ui/textarea';
import { Slider } from './ui/slider';
import {
  Loader2,
  Wand2,
  CheckCircle2,
  FolderOpen,
  AlertTriangle,
  RefreshCw,
  HelpCircle,
  XCircle,
} from 'lucide-react';
import VoiceMergeQueuePanel from './VoiceMergeQueuePanel';
import { safeErrorMessage } from '../config/api';
import { runVoiceSculptStream } from '../utils/voiceSculptStream';

const STEPS = [
  { id: 1, label: 'Clean (optional)', detail: 'UVR only if clips are not already isolated' },
  { id: 2, label: 'Merge', detail: 'Morph timbre like a celebrity face blend — both voices in one hybrid' },
  { id: 3, label: 'Normalize', detail: 'Write one voice reference WAV to voice_references/' },
];

function StepDot({ step, activeStep, label, done }) {
  const isActive = activeStep === step;
  const isDone = done || activeStep > step;
  return (
    <div className="flex flex-col items-center gap-1 flex-1 min-w-0">
      <div
        className={[
          'flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold border-2 transition-colors',
          isDone ? 'bg-primary border-primary text-primary-foreground' : '',
          isActive && !isDone ? 'border-primary text-primary bg-primary/10' : '',
          !isActive && !isDone ? 'border-border text-muted-foreground bg-muted/30' : '',
        ].filter(Boolean).join(' ')}
      >
        {isDone ? <CheckCircle2 className="h-4 w-4" /> : step}
      </div>
      <span className={`text-xs truncate max-w-full text-center ${isActive ? 'text-foreground font-medium' : 'text-muted-foreground'}`}>
        {label}
      </span>
    </div>
  );
}

function StatusPill({ ok, label, hint }) {
  return (
    <div
      className={`rounded-lg border px-3 py-2 text-xs ${ok ? 'border-emerald-500/40 bg-emerald-500/10' : 'border-amber-500/40 bg-amber-500/10'}`}
      title={hint || label}
    >
      <div className="flex items-center gap-1.5 font-medium">
        {ok ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" /> : <XCircle className="h-3.5 w-3.5 text-amber-600 shrink-0" />}
        <span className="truncate">{label}</span>
      </div>
      {hint ? <p className="text-muted-foreground mt-1 leading-snug">{hint}</p> : null}
    </div>
  );
}

function GuidanceDetails({ title, children, defaultOpen = false }) {
  return (
    <details className="group rounded-lg border border-border/60 bg-muted/10" open={defaultOpen}>
      <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted/30 rounded-lg">
        <HelpCircle className="h-4 w-4 text-muted-foreground shrink-0" />
        {title}
      </summary>
      <div className="px-3 pb-3 pt-1 text-xs text-muted-foreground space-y-2 leading-relaxed">
        {children}
      </div>
    </details>
  );
}

export default function VoiceSculptPanel({ onVoiceReady, disabled = false }) {
  const { PRIMARY_API_URL } = useApp();
  const [source, setSource] = useState('');
  const [outputName, setOutputName] = useState('');
  const [accentModel, setAccentModel] = useState('default');
  const [skipRvc, setSkipRvc] = useState(true);
  const [skipUvr, setSkipUvr] = useState(true);
  const [morphBalance, setMorphBalance] = useState(50);
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.75);
  const [protect, setProtect] = useState(0.33);
  const [voicePrompt, setVoicePrompt] = useState('');

  const [setupRunning, setSetupRunning] = useState(false);
  const [setupInfo, setSetupInfo] = useState(null);
  const [discoverInfo, setDiscoverInfo] = useState(null);
  const [hfUrl, setHfUrl] = useState('');
  const [hfUser, setHfUser] = useState('');
  const [hfRepo, setHfRepo] = useState('');
  const [hfInstalling, setHfInstalling] = useState(false);
  const autoSetupRan = useRef(false);

  const [preflight, setPreflight] = useState(null);
  const [preflightLoading, setPreflightLoading] = useState(true);

  const [running, setRunning] = useState(false);
  const [queueRunning, setQueueRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const abortRef = useRef(null);
  const guidance = discoverInfo?.guidance || setupInfo?.guidance || null;
  const voiceModels = discoverInfo?.models || [];

  const loadDiscover = useCallback(async () => {
    if (!PRIMARY_API_URL) return null;
    try {
      const res = await fetchWithTimeout(`${PRIMARY_API_URL}/voice-sculpt/discover`, {}, 15000);
      if (res.ok) {
        const data = await res.json();
        setDiscoverInfo(data);
        return data;
      }
    } catch {
      /* non-fatal */
    }
    return null;
  }, [PRIMARY_API_URL]);

  const sourceLineCount = source.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).length;

  const loadPreflight = useCallback(async () => {
    if (!PRIMARY_API_URL) return;
    setPreflightLoading(true);
    try {
      const params = new URLSearchParams({
        for_youtube: 'false',
        for_uvr: skipUvr ? 'false' : 'true',
        for_rvc: skipRvc ? 'false' : 'true',
        for_morph: sourceLineCount > 1 ? 'true' : 'false',
      });
      if (!skipRvc && accentModel && accentModel !== 'default') {
        params.set('accent_model', accentModel);
      }
      const res = await fetchWithTimeout(
        `${PRIMARY_API_URL}/voice-sculpt/preflight?${params}`,
        {},
        15000,
      );
      if (res.ok) {
        setPreflight(await res.json());
      } else {
        setPreflight({ ready: false, missing: [{ detail: 'Preflight request failed' }] });
      }
    } catch (e) {
      setPreflight({ ready: false, missing: [{ detail: safeErrorMessage(e, 'Could not reach backend') }] });
    } finally {
      setPreflightLoading(false);
    }
  }, [PRIMARY_API_URL, skipRvc, skipUvr, accentModel, sourceLineCount]);

  const runAutoSetup = useCallback(async ({ cloneApplio = false } = {}) => {
    if (!PRIMARY_API_URL || setupRunning) return null;
    setSetupRunning(true);
    setError(null);
    try {
      const res = await fetchWithTimeout(
        `${PRIMARY_API_URL}/voice-sculpt/auto-setup`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            clone_applio: cloneApplio,
            install_uvr: true,
            write_env_file: true,
          }),
        },
        cloneApplio ? 600000 : 120000,
      );
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Auto-setup failed');
      }
      setSetupInfo(data);
      if (data.guidance) {
        setDiscoverInfo((prev) => ({ ...(prev || {}), guidance: data.guidance }));
      }
      await Promise.all([loadPreflight(), loadDiscover()]);
      return data;
    } catch (e) {
      setError(safeErrorMessage(e));
      return null;
    } finally {
      setSetupRunning(false);
    }
  }, [PRIMARY_API_URL, setupRunning, loadPreflight, loadDiscover]);

  const installFromHuggingFace = useCallback(async () => {
    const builtUrl = hfUrl.trim()
      || (hfUser.trim() && hfRepo.trim()
        ? `https://huggingface.co/${hfUser.trim()}/${hfRepo.trim()}`
        : '');
    if (!PRIMARY_API_URL || !builtUrl || hfInstalling) return;
    setHfInstalling(true);
    setError(null);
    try {
      const res = await fetchWithTimeout(
        `${PRIMARY_API_URL}/voice-sculpt/install-hf-model`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: builtUrl }),
        },
        600000,
      );
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Hugging Face install failed');
      }
      setSetupInfo(data);
      setHfUrl('');
      setHfUser('');
      setHfRepo('');
      if (data.install?.model_name) {
        setAccentModel(data.install.model_name);
      }
      await Promise.all([loadPreflight(), loadDiscover()]);
    } catch (e) {
      setError(safeErrorMessage(e));
    } finally {
      setHfInstalling(false);
    }
  }, [PRIMARY_API_URL, hfUrl, hfUser, hfRepo, hfInstalling, loadPreflight, loadDiscover]);

  useEffect(() => {
    if (!PRIMARY_API_URL || autoSetupRan.current) return;
    autoSetupRan.current = true;
    (async () => {
      await runAutoSetup({ cloneApplio: false });
      await loadDiscover();
    })();
  }, [PRIMARY_API_URL, runAutoSetup, loadDiscover]);

  useEffect(() => {
    if (!autoSetupRan.current) return;
    loadPreflight();
  }, [loadPreflight, skipRvc, accentModel]);

  useEffect(() => {
    if (voiceModels.length === 0) return;
    const names = voiceModels.map((m) => m.name);
    if (accentModel === 'default' || !names.includes(accentModel)) {
      setAccentModel(voiceModels[0].name);
    }
  }, [voiceModels, accentModel]);

  const pickAudioFiles = async () => {
    try {
      const res = await fetchWithTimeout(
        `${PRIMARY_API_URL}/system/select-file`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: 'Select speech clips or voice references (any number)',
            initial_directory: source.split(/\r?\n/).map((l) => l.trim()).find(Boolean) || undefined,
            multiple: true,
          }),
        },
        120000,
      );
      const data = await res.json();
      if (data.status === 'success' && Array.isArray(data.files) && data.files.length > 0) {
        setSource((prev) => {
          const existing = prev.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
          const merged = [...existing];
          for (const f of data.files) {
            if (!merged.includes(f)) merged.push(f);
          }
          return merged.join('\n');
        });
      } else if (data.status === 'success' && data.file) {
        setSource((prev) => (prev.trim() ? `${prev.trim()}\n${data.file}` : data.file));
      }
    } catch (e) {
      setError(safeErrorMessage(e));
    }
  };

  const getCurrentJobSnapshot = useCallback(
    () => ({
      source: source.trim(),
      outputName: outputName.trim(),
      morphBalance,
      skipRvc,
      skipUvr,
      accentModel,
      pitch,
      indexRate,
      protect,
      voicePrompt: voicePrompt.trim(),
    }),
    [source, outputName, morphBalance, skipRvc, skipUvr, accentModel, pitch, indexRate, protect, voicePrompt],
  );

  const runSculpt = async () => {
    if (sourceLineCount === 0 || running || queueRunning) return;
    setRunning(true);
    setError(null);
    setResult(null);
    setActiveStep(0);
    setProgressMessage('Starting pipeline…');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data = await runVoiceSculptStream(
        PRIMARY_API_URL,
        {
          source: source.trim(),
          source_type: 'local_path',
          output_name: outputName.trim() || undefined,
          accent_model: accentModel,
          skip_rvc: skipRvc,
          skip_uvr: skipUvr,
          combine_mode: 'morph',
          morph_balance: morphBalance / 100,
          pitch,
          index_rate: indexRate,
          protect,
          voice_prompt: voicePrompt.trim() || undefined,
        },
        {
          signal: controller.signal,
          onProgress: (ev) => {
            setActiveStep(ev.step || 0);
            setProgressMessage(ev.message || '');
          },
        },
      );
      setActiveStep(4);
      setResult(data);
      setProgressMessage('Done — voice reference ready.');
      if (onVoiceReady) onVoiceReady(data);
    } catch (e) {
      if (e.name !== 'AbortError') {
        setError(safeErrorMessage(e));
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const modeReady = preflight?.merge_ready ?? preflight?.uvr_ready;
  const canSculpt =
    !disabled && !running && !queueRunning && !setupRunning && !preflightLoading && modeReady && sourceLineCount > 0;

  const toolsReady = preflight?.available_tools || [];
  const hasUvr = toolsReady.includes('audio-separator');
  const hasFfmpeg = toolsReady.includes('ffmpeg');
  const hasApplio = toolsReady.includes('applio');
  const hasPyworld = toolsReady.includes('pyworld');
  const hasVoiceModel = voiceModels.length > 0;

  const applioMissing = preflight?.missing?.some((m) =>
    ['applio', 'applio-python', 'applio-model'].includes(m.missing_tool),
  );

  const showPretrainedWarning = guidance?.has_pretrained_only;

  return (
    <section className="rounded-xl border border-border/80 bg-card/60 shadow-sm overflow-hidden mt-4">
      <header className="flex gap-3 items-start px-4 py-3 border-b border-border/60 bg-muted/25">
        <span className="flex h-9 min-w-[2.25rem] shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Wand2 className="h-4 w-4" />
        </span>
        <div className="min-w-0 pt-0.5 flex-1">
          <h3 className="text-sm font-semibold leading-snug">Voice Merge</h3>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            Morph two or more voice clips into one hybrid reference — like celebrity face merges, but for timbre.
          </p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="shrink-0 h-8 text-xs"
          disabled={setupRunning}
          onClick={() => runAutoSetup({ cloneApplio: false })}
        >
          {setupRunning ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
          <span className="ml-1.5">Refresh setup</span>
        </Button>
      </header>

      <div className="p-4 space-y-4">
        <GuidanceDetails title="How this works" defaultOpen>
          <ol className="list-decimal pl-4 space-y-1.5">
            {STEPS.map((s) => (
              <li key={s.id}>
                <strong className="text-foreground">{s.label}</strong> — {s.detail}
              </li>
            ))}
          </ol>
          {(guidance?.quick_start || []).map((line, i) => (
            <p key={i} className={i === 0 ? 'pt-1' : ''}>{line}</p>
          ))}
        </GuidanceDetails>

        {!preflightLoading && !setupRunning ? (
          <div className="grid gap-2 sm:grid-cols-2">
            {skipUvr ? null : (
              <StatusPill ok={hasUvr} label="UVR / audio-separator" hint={hasUvr ? 'Vocal isolation ready' : 'Install via Auto-setup'} />
            )}
            <StatusPill
              ok={hasFfmpeg}
              label="ffmpeg"
              hint={
                hasFfmpeg
                  ? 'Normalization ready'
                  : 'ffmpeg not found — install (winget install Gyan.FFmpeg) or set path in Settings → Audio → FFmpeg path, then Refresh setup'
              }
            />
            {sourceLineCount > 1 ? (
              <StatusPill ok={hasPyworld} label="pyworld (voice morph)" hint={hasPyworld ? 'Timbre merge ready' : 'pip install pyworld'} />
            ) : null}
            <StatusPill
              ok={!skipRvc ? hasApplio : true}
              label="Applio Python"
              hint={
                skipRvc
                  ? 'Not needed in Quick mode'
                  : hasApplio
                    ? discoverInfo?.discovered?.applio_python || 'Applio env found'
                    : 'Run tools/Applio/run-install.bat or complete-install.bat'
              }
            />
            <StatusPill
              ok={!skipRvc ? hasVoiceModel : true}
              label="RVC voice model"
              hint={
                skipRvc
                  ? 'Not needed in Quick mode'
                  : hasVoiceModel
                    ? `${voiceModels.length} voice model(s) in logs/ — .index optional`
                    : 'Install a voice model via Hugging Face below'
              }
            />
          </div>
        ) : (
          <p className="text-xs text-muted-foreground flex items-center gap-2">
            <Loader2 className="h-3 w-3 animate-spin" />
            {setupRunning ? 'Auto-configuring tools…' : 'Checking tools…'}
          </p>
        )}

        {showPretrainedWarning ? (
          <Alert className="border-amber-500/50 bg-amber-500/5">
            <AlertTriangle className="h-4 w-4 text-amber-600" />
            <AlertTitle className="text-sm">Pretrained trainers detected (not voice models)</AlertTitle>
            <AlertDescription className="text-xs mt-1">
              You have <strong>{guidance.pretrained_training_count} pretrained training model(s)</strong> in{' '}
              <code className="text-[11px]">rvc/models/pretraineds/</code> — those are for training, not sculpting.
            </AlertDescription>
          </Alert>
        ) : null}

        <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-3">
          <div>
            <p className="text-sm font-medium text-foreground">Install voice models from Hugging Face</p>
            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
              Uses <code className="text-[11px]">huggingface_hub</code>. Install as many as you want — pick one below for sculpting.
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="space-y-1">
              <Label className="text-xs">HF user / org</Label>
              <Input value={hfUser} onChange={(e) => setHfUser(e.target.value)} placeholder="SomeAuthor" disabled={hfInstalling || running} className="text-xs" />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Repo name</Label>
              <Input value={hfRepo} onChange={(e) => setHfRepo(e.target.value)} placeholder="SomeVoiceModel" disabled={hfInstalling || running} className="text-xs" />
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Or full URL (optional)</Label>
            <Input value={hfUrl} onChange={(e) => setHfUrl(e.target.value)} placeholder="https://huggingface.co/Author/VoiceModelName" disabled={hfInstalling || running} className="text-xs" />
          </div>
          <Button type="button" size="sm" variant="secondary" disabled={(!hfUrl.trim() && !(hfUser.trim() && hfRepo.trim())) || hfInstalling || running} onClick={installFromHuggingFace}>
            {hfInstalling ? (<><Loader2 className="h-3 w-3 animate-spin mr-1.5" /> Installing…</>) : 'Install voice model'}
          </Button>
        </div>

        {voiceModels.length > 0 ? (
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertTitle className="text-sm">{voiceModels.length} voice model(s) installed</AlertTitle>
            <AlertDescription className="text-xs mt-1">
              <ul className="list-disc pl-4 space-y-0.5">
                {voiceModels.map((m) => (
                  <li key={m.pth}>
                    <strong>{m.name}</strong>
                    {m.has_index ? ' (.pth + .index)' : ' (.pth only — .index optional, improves quality)'}
                  </li>
                ))}
              </ul>
              <p className="text-muted-foreground mt-2">Choose which voice to use via &quot;Voice model for sculpt&quot; below.</p>
            </AlertDescription>
          </Alert>
        ) : null}

        {preflight?.warnings?.length > 0 ? (
          <Alert className="border-amber-500/50 bg-amber-500/5">
            <AlertTriangle className="h-4 w-4 text-amber-600" />
            <AlertTitle className="text-sm">Notes (sculpt can still run)</AlertTitle>
            <AlertDescription className="text-xs mt-1 space-y-1">
              {preflight.warnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </AlertDescription>
          </Alert>
        ) : null}

        {preflight && !modeReady ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Merge tools missing</AlertTitle>
            <AlertDescription className="text-xs space-y-2 mt-1">
              {(preflight.missing || [])
                .filter((m) => {
                  if (skipRvc && ['applio', 'applio-python', 'applio-model'].includes(m.missing_tool)) return false;
                  if (skipUvr && m.missing_tool === 'audio-separator') return false;
                  if (sourceLineCount <= 1 && m.missing_tool === 'pyworld') return false;
                  return true;
                })
                .map((m, i) => (
                  <div key={i}>
                    <strong>{m.missing_tool || 'tool'}:</strong> {m.detail}
                    {m.install_hint ? (
                      <span className="block text-muted-foreground">{m.install_hint}</span>
                    ) : null}
                  </div>
                ))}
              {setupInfo?.next_steps?.length ? (
                <ul className="list-disc pl-4 space-y-1 text-muted-foreground">
                  {setupInfo.next_steps.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              ) : null}
              {applioMissing && !discoverInfo?.discovered?.applio_root ? (
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className="mt-2"
                  disabled={setupRunning}
                  onClick={() => runAutoSetup({ cloneApplio: true })}
                >
                  Install Applio (git clone)
                </Button>
              ) : null}
            </AlertDescription>
          </Alert>
        ) : preflight && modeReady ? (
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertTitle className="text-sm">Ready to merge</AlertTitle>
            <AlertDescription className="text-xs">
              {sourceLineCount > 1
                ? 'Will morph timbre from your clips into one hybrid voice reference.'
                : 'Will normalize your clip into a voice reference (add a second clip to morph).'}
            </AlertDescription>
          </Alert>
        ) : null}

        <div className="space-y-2">
          <Label htmlFor="sculpt-source" className="text-xs">
            Voice clips to merge (one path per line — order matters for balance)
          </Label>
          <Textarea
            id="sculpt-source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder={'C:\\clips\\speech_a.wav\nC:\\clips\\speech_b.wav'}
            disabled={running || queueRunning}
            className="min-h-[96px] text-xs font-mono"
          />
          <div className="flex flex-wrap gap-2 items-center">
            <Button type="button" variant="outline" size="sm" onClick={pickAudioFiles} disabled={running || queueRunning}>
              <FolderOpen className="h-3.5 w-3.5 mr-1.5" />
              Add files…
            </Button>
            <span className="text-[11px] text-muted-foreground">
              {sourceLineCount} file{sourceLineCount === 1 ? '' : 's'} queued
            </span>
          </div>
        </div>

        {sourceLineCount === 2 ? (
          <div className="space-y-2">
            <Label className="text-xs">
              Morph balance ({Number(morphBalance).toFixed(1)}% toward second clip)
            </Label>
            <p className="text-[11px] text-muted-foreground">
              0.0 = first clip only, 100.0 = second only. Step 0.1% — type an exact value (e.g. 95.3).
            </p>
            <Slider
              min={0}
              max={100}
              step={0.1}
              value={[morphBalance]}
              onValueChange={([v]) => setMorphBalance(Math.round(v * 10) / 10)}
              disabled={running || queueRunning}
            />
            <div className="flex items-center gap-2 max-w-[12rem]">
              <Input
                type="number"
                min={0}
                max={100}
                step={0.1}
                value={morphBalance}
                onChange={(e) => {
                  const raw = e.target.value;
                  if (raw === '') return;
                  const n = parseFloat(raw);
                  if (!Number.isFinite(n)) return;
                  setMorphBalance(Math.min(100, Math.max(0, Math.round(n * 10) / 10)));
                }}
                disabled={running || queueRunning}
                className="text-xs h-8 font-mono"
              />
              <span className="text-xs text-muted-foreground shrink-0">% → clip 2</span>
            </div>
          </div>
        ) : sourceLineCount > 2 ? (
          <p className="text-[11px] text-muted-foreground">Three or more clips: equal-weight morph.</p>
        ) : null}

        <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-3">
          <p className="text-xs font-medium text-foreground">Optional</p>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={!skipUvr} onChange={(e) => setSkipUvr(!e.target.checked)} disabled={running || queueRunning} />
            Clean clips first (UVR)
          </label>
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={!skipRvc} onChange={(e) => setSkipRvc(!e.target.checked)} disabled={running || queueRunning} />
            RVC polish after merge
          </label>
        </div>

        {skipRvc ? null : (
          <div className="space-y-2">
            <Label className="text-xs">RVC model (optional polish)</Label>
            <Select value={accentModel} onValueChange={setAccentModel} disabled={running || queueRunning || voiceModels.length === 0}>
              <SelectTrigger>
                <SelectValue placeholder={voiceModels.length ? 'Select voice' : 'Install a voice model first'} />
              </SelectTrigger>
              <SelectContent>
                {voiceModels.length > 0 ? (
                  voiceModels.map((m) => (
                    <SelectItem key={m.pth} value={m.name}>
                      {m.name}{m.has_index ? '' : ' (no index)'}
                    </SelectItem>
                  ))
                ) : (
                  <SelectItem value="default" disabled>No models installed</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>
        )}

        {!skipRvc ? (
          <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-3">
            <p className="text-xs font-medium text-foreground">RVC controls (Applio infer)</p>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Applied after morph, on the hybrid stem only.
            </p>
            <div className="space-y-1">
              <Label className="text-xs">Pitch ({pitch})</Label>
              <p className="text-[11px] text-muted-foreground">Semitones up/down. 0 = keep source pitch.</p>
              <Slider min={-12} max={12} step={1} value={[pitch]} onValueChange={([v]) => setPitch(v)} disabled={running || queueRunning} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Index rate ({indexRate.toFixed(2)})</Label>
              <p className="text-[11px] text-muted-foreground">0 = ignore .index (keep more source). 1 = strongest model timbre. Try 0.2–0.4 when blending sources.</p>
              <Slider min={0} max={1} step={0.05} value={[indexRate]} onValueChange={([v]) => setIndexRate(v)} disabled={running || queueRunning} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Protect ({protect.toFixed(2)})</Label>
              <p className="text-[11px] text-muted-foreground">Preserves consonants/plosives from source. Higher = less robotic swap on hard sounds.</p>
              <Slider min={0} max={0.5} step={0.01} value={[protect]} onValueChange={([v]) => setProtect(v)} disabled={running || queueRunning} />
            </div>
          </div>
        ) : null}

        <div className="space-y-2">
          <Label htmlFor="sculpt-output-name" className="text-xs">Output filename (optional)</Label>
          <Input
            id="sculpt-output-name"
            value={outputName}
            onChange={(e) => setOutputName(e.target.value)}
            placeholder="my_merged_voice"
            disabled={running || queueRunning}
            className="text-xs"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="voice-prompt" className="text-xs">Voice note / prompt (optional)</Label>
          <Textarea
            id="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            placeholder="Saved as .prompt.txt next to the WAV — no engine reads this yet."
            disabled={running || queueRunning}
            className="min-h-[64px] text-xs"
          />
        </div>



        {(running || activeStep > 0) && (
          <div className="space-y-3 pt-1">
            <div className="flex gap-2 items-start w-full">
              {STEPS.map((s) => (
                <StepDot
                  key={s.id}
                  step={s.id}
                  activeStep={activeStep}
                  label={s.label}
                  done={!!result}
                />
              ))}
            </div>
            {progressMessage ? (
              <p className="text-xs text-muted-foreground">{progressMessage}</p>
            ) : null}
          </div>
        )}

        {error ? (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription className="text-xs">{error}</AlertDescription>
          </Alert>
        ) : null}

        {result ? (
          <Alert>
            <CheckCircle2 className="h-4 w-4" />
            <AlertTitle>Voice ready</AlertTitle>
            <AlertDescription className="text-xs">
              <strong>{result.voice_id}</strong><br /><span className="break-all text-muted-foreground">{result.path}</span>
            </AlertDescription>
          </Alert>
        ) : null}

        <Button type="button" onClick={runSculpt} disabled={!canSculpt} className="w-full sm:w-auto">
          {running ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" /> Merging…
            </>
          ) : (
            <>
              <Wand2 className="h-4 w-4 mr-2" /> Merge voices
            </>
          )}
        </Button>

        <VoiceMergeQueuePanel
          apiUrl={PRIMARY_API_URL}
          disabled={disabled}
          sculptRunning={running}
          onQueueRunningChange={setQueueRunning}
          getCurrentJobSnapshot={getCurrentJobSnapshot}
          onVoiceReady={onVoiceReady}
        />
      </div>
    </section>
  );
}
