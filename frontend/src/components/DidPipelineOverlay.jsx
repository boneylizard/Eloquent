import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, X, Video, Sparkles, Image as ImageIcon, CheckCircle2, AlertTriangle } from 'lucide-react';

export const DID_AVATAR_SD_PREFIX =
  'Frontal facing portrait, medium shot — face, neck, and top of shoulders only, nothing lower. ' +
  'Mouth closed, neutral or slight smile. Even soft lighting, no harsh shadows. ' +
  'No jewellery near mouth or jaw. No hair falling over lower face. ' +
  'High resolution, photorealistic. No heavy stylisation, no anime, no cartoon.\n\n';

const emptySdState = () => ({
  open: false,
  mode: 'avatar',
  prompt: '',
  previewUrl: '',
  busy: false,
  error: '',
});

const DidPipelineOverlay = ({ open, onClose }) => {
  const { settings } = useApp();
  const base = getBackendUrl();

  const [avatarUrl, setAvatarUrl] = useState('');
  const [backgroundUrl, setBackgroundUrl] = useState('');
  const [emotion, setEmotion] = useState('neutral');
  const [movement, setMovement] = useState('active');
  const [concurrency, setConcurrency] = useState(2);
  const [requireVision, setRequireVision] = useState(true);
  const [wavFiles, setWavFiles] = useState([]);
  const [logLines, setLogLines] = useState([]);
  const [running, setRunning] = useState(false);
  const [lastOutput, setLastOutput] = useState('');
  const [visionResult, setVisionResult] = useState(null);
  const [savedAssets, setSavedAssets] = useState([]);
  const [sd, setSd] = useState(emptySdState);
  const folderInputRef = useRef(null);
  const fileInputRef = useRef(null);

  const quickButtons = useMemo(
    () =>
      (Array.isArray(settings.didQuickPromptButtons) ? settings.didQuickPromptButtons : []).filter(
        (b) => b && (b.label || b.text)
      ),
    [settings.didQuickPromptButtons]
  );

  const appendLog = useCallback((line) => {
    setLogLines((prev) => [...prev.slice(-400), line]);
  }, []);

  const refreshSaved = useCallback(async () => {
    try {
      const r = await fetch(`${base}/d-id/saved-assets`);
      const j = await r.json();
      if (j.items) setSavedAssets(j.items);
    } catch {
      /* noop */
    }
  }, [base]);

  React.useEffect(() => {
    if (open) refreshSaved();
  }, [open, refreshSaved]);

  const runVisionOnAvatarUrl = useCallback(async () => {
    if (!avatarUrl.trim()) {
      appendLog('Set avatar URL first.');
      return;
    }
    appendLog('Running vision screen on avatar URL…');
    try {
      const r = await fetch(`${base}/d-id/vision-screen-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: avatarUrl.trim() }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
      setVisionResult(j.result);
      appendLog(
        j.result?.pass
          ? `Vision pass (score ${j.result?.overall_score ?? '?'})`
          : `Vision fail (score ${j.result?.overall_score ?? '?'}): ${(j.result?.failure_reasons || []).join('; ')}`
      );
    } catch (e) {
      appendLog(`Vision error: ${e.message || e}`);
    }
  }, [appendLog, avatarUrl, base]);

  const pickWavFolder = useCallback((e) => {
    const files = Array.from(e.target.files || []).filter((f) => /\.wav$/i.test(f.name));
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
    setWavFiles(files);
    appendLog(`Selected ${files.length} WAV file(s).`);
    e.target.value = '';
  }, [appendLog]);

  const runBatch = useCallback(async () => {
    if (!avatarUrl.trim()) {
      alert('Avatar image URL is required (must be reachable by D-ID; set D_ID_PUBLIC_BASE_URL if you use /static/…).');
      return;
    }
    if (!wavFiles.length) {
      alert('Choose a folder of WAV files (or multiple WAVs).');
      return;
    }
    setRunning(true);
    setLastOutput('');
    setLogLines([]);
    appendLog('Starting batch…');
    const fd = new FormData();
    wavFiles.forEach((f) => fd.append('segments', f));
    fd.append('avatar_source_url', avatarUrl.trim());
    fd.append('emotion', emotion);
    fd.append('movement', movement);
    if (backgroundUrl.trim()) fd.append('background_url', backgroundUrl.trim());
    fd.append('concurrency', String(Math.max(1, Math.min(4, concurrency))));
    fd.append('require_vision', requireVision ? 'true' : 'false');
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 45 * 60 * 1000);
      const res = await fetch(`${base}/d-id/batch-run`, { method: 'POST', body: fd, signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\n');
        buf = parts.pop() || '';
        for (const line of parts) {
          const s = line.trim();
          if (!s) continue;
          let evt;
          try {
            evt = JSON.parse(s);
          } catch {
            appendLog(s);
            continue;
          }
          if (evt.event === 'vision_screen') {
            setVisionResult(evt.result);
          }
          if (evt.event === 'segment_done') {
            appendLog(`Segment ${evt.index + 1}: ${evt.mp4_path}`);
          } else if (evt.event === 'complete') {
            setLastOutput(evt.output_path || '');
            appendLog(`DONE → ${evt.output_path}`);
          } else if (evt.event === 'error') {
            appendLog(`ERROR: ${evt.message || JSON.stringify(evt)}`);
          } else {
            appendLog(s);
          }
        }
      }
    } catch (e) {
      appendLog(`Batch failed: ${e.message || e}`);
    } finally {
      setRunning(false);
    }
  }, [
    appendLog,
    avatarUrl,
    backgroundUrl,
    base,
    concurrency,
    emotion,
    movement,
    requireVision,
    wavFiles,
  ]);

  const openSd = useCallback((mode, initialPrompt) => {
    const def =
      mode === 'avatar'
        ? DID_AVATAR_SD_PREFIX
        : 'Wide establishing shot, photorealistic environment, soft even lighting, no people, no text, high detail.\n\n';
    setSd({
      ...emptySdState(),
      open: true,
      mode,
      prompt: initialPrompt != null ? initialPrompt : def,
    });
  }, []);

  const runSd = useCallback(async () => {
    const prompt = (sd.prompt || '').trim();
    if (!prompt) return;
    setSd((prev) => ({ ...prev, busy: true, error: '' }));
    try {
      const gpuId = settings.main_gpu_id ?? 0;
      const r = await fetch(`${base}/sd-local/txt2img`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          gpu_id: gpuId,
          width: 768,
          height: 512,
          steps: settings.sdSteps ?? 20,
          guidance_scale: settings.sdCfgScale ?? 7,
        }),
      });
      const j = await r.json();
      if (!r.ok || j.status !== 'success') throw new Error(j.detail || j.error || JSON.stringify(j));
      const rel = (j.image_urls && j.image_urls[0]) || '';
      const previewUrl = rel.startsWith('http') ? rel : `${base}${rel}`;
      setSd((prev) => ({ ...prev, previewUrl, busy: false }));
    } catch (e) {
      setSd((prev) => ({ ...prev, busy: false, error: e.message || String(e) }));
    }
  }, [base, sd.prompt, settings.main_gpu_id, settings.sdCfgScale, settings.sdSteps]);

  const acceptSd = useCallback(() => {
    if (!sd.previewUrl) return;
    const pathOrUrl = sd.previewUrl.startsWith(base) ? sd.previewUrl.slice(base.length) : sd.previewUrl;
    if (sd.mode === 'avatar') setAvatarUrl(pathOrUrl.startsWith('http') ? sd.previewUrl : `${base}${pathOrUrl}`);
    else setBackgroundUrl(pathOrUrl.startsWith('http') ? sd.previewUrl : `${base}${pathOrUrl}`);
    setSd(emptySdState());
  }, [base, sd.mode, sd.previewUrl]);

  const saveCurrentAsset = useCallback(
    async (kind) => {
      const url = kind === 'avatar' ? avatarUrl.trim() : backgroundUrl.trim();
      if (!url) return;
      const label = window.prompt('Label for saved preset', kind === 'avatar' ? 'Avatar' : 'Background');
      if (label === null) return;
      await fetch(`${base}/d-id/saved-assets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind, label: label || '', url }),
      });
      refreshSaved();
    },
    [avatarUrl, backgroundUrl, base, refreshSaved]
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 p-4">
      <div className="bg-card border rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-2 font-semibold">
            <Video className="h-5 w-5" />
            D-ID batch pipeline
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onClose} disabled={running}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0 p-4">
          <div className="space-y-4 text-sm">
            <p className="text-muted-foreground text-xs">
              D-ID needs a <strong>public https</strong> image URL. For local SD output, set backend env{' '}
              <code className="text-xs">D_ID_PUBLIC_BASE_URL</code> to your reachable origin so{' '}
              <code className="text-xs">/static/generated_images/…</code> resolves. Vision screening uses{' '}
              <code className="text-xs">D_ID_VISION_SCREEN_MODEL</code> (your Kimi <code className="text-xs">endpoint-…</code> id).
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label>Avatar image URL</Label>
                <Input value={avatarUrl} onChange={(e) => setAvatarUrl(e.target.value)} placeholder="https://…" />
                <div className="flex flex-wrap gap-1">
                  <Button type="button" size="sm" variant="outline" onClick={() => openSd('avatar')}>
                    <Sparkles className="h-3 w-3 mr-1" />
                    Create avatar (SD)
                  </Button>
                  <Button type="button" size="sm" variant="secondary" onClick={runVisionOnAvatarUrl}>
                    Screen avatar
                  </Button>
                  <Button type="button" size="sm" variant="ghost" onClick={() => saveCurrentAsset('avatar')}>
                    Save preset
                  </Button>
                </div>
              </div>
              <div className="space-y-1">
                <Label>Background URL (optional)</Label>
                <Input value={backgroundUrl} onChange={(e) => setBackgroundUrl(e.target.value)} placeholder="https://…" />
                <div className="flex flex-wrap gap-1">
                  <Button type="button" size="sm" variant="outline" onClick={() => openSd('background')}>
                    <ImageIcon className="h-3 w-3 mr-1" />
                    Create background (SD)
                  </Button>
                  <Button type="button" size="sm" variant="ghost" onClick={() => saveCurrentAsset('background')}>
                    Save preset
                  </Button>
                </div>
              </div>
            </div>

            {visionResult && (
              <div
                className={`rounded border p-2 text-xs flex gap-2 items-start ${
                  visionResult.pass ? 'border-green-700/50 bg-green-950/20' : 'border-amber-700/50 bg-amber-950/20'
                }`}
              >
                {visionResult.pass ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" />
                )}
                <div>
                  <div className="font-medium">
                    Vision {visionResult.pass ? 'passed' : 'failed'} — overall {visionResult.overall_score ?? '?'}
                    /10
                  </div>
                  {visionResult && !visionResult.pass && (visionResult.failure_reasons || []).length > 0 && (
                    <ul className="list-disc pl-4 mt-1">
                      {visionResult.failure_reasons.map((x, i) => (
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  )}
                  {visionResult && !visionResult.pass && (
                    <Button
                      type="button"
                      className="mt-2"
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const crit = (visionResult.failure_reasons || []).join('; ');
                        openSd('avatar', `${DID_AVATAR_SD_PREFIX}Address these issues: ${crit}\n\n`);
                      }}
                    >
                      Regenerate SD with critique
                    </Button>
                  )}
                </div>
              </div>
            )}

            {savedAssets.length > 0 && (
              <div className="space-y-1">
                <Label className="text-xs">Saved presets</Label>
                <div className="flex flex-wrap gap-1">
                  {savedAssets.map((a) => (
                    <Button
                      key={a.id}
                      type="button"
                      size="sm"
                      variant="secondary"
                      className="text-xs h-7"
                      onClick={() => {
                        if (a.kind === 'avatar') setAvatarUrl(a.url);
                        else setBackgroundUrl(a.url);
                      }}
                    >
                      {a.kind}: {a.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1">
                <Label>Emotion</Label>
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                  value={emotion}
                  onChange={(e) => setEmotion(e.target.value)}
                >
                  <option value="happy">happy</option>
                  <option value="neutral">neutral</option>
                  <option value="surprised">surprised</option>
                  <option value="serious">serious</option>
                </select>
              </div>
              <div className="space-y-1">
                <Label>Movement</Label>
                <select
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                  value={movement}
                  onChange={(e) => setMovement(e.target.value)}
                >
                  <option value="active">active</option>
                  <option value="still">still</option>
                </select>
              </div>
              <div className="space-y-1">
                <Label>Concurrency (1–4)</Label>
                <Input
                  type="number"
                  min={1}
                  max={4}
                  value={concurrency}
                  onChange={(e) => setConcurrency(Math.max(1, Math.min(4, parseInt(e.target.value, 10) || 2)))}
                />
              </div>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              <input
                ref={folderInputRef}
                type="file"
                multiple
                accept=".wav,audio/wav,audio/x-wav"
                className="hidden"
                onChange={pickWavFolder}
                webkitdirectory=""
                directory=""
              />
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".wav,audio/wav,audio/x-wav"
                className="hidden"
                onChange={pickWavFolder}
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => folderInputRef.current?.click()}
                disabled={running}
              >
                Choose WAV folder
              </Button>
              <Button type="button" variant="outline" onClick={() => fileInputRef.current?.click()} disabled={running}>
                Choose WAV files
              </Button>
              <label className="flex items-center gap-2 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={requireVision}
                  onChange={(e) => setRequireVision(e.target.checked)}
                  disabled={running}
                />
                Require vision pass before D-ID
              </label>
            </div>
            {wavFiles.length > 0 && (
              <p className="text-xs text-muted-foreground">
                {wavFiles.length} file(s): {wavFiles[0]?.name}
                {wavFiles.length > 1 ? ` … ${wavFiles[wavFiles.length - 1]?.name}` : ''}
              </p>
            )}

            {quickButtons.length > 0 && (
              <div className="space-y-1">
                <Label className="text-xs">Quick actions (from Settings)</Label>
                <div className="flex flex-wrap gap-1">
                  {quickButtons.map((b) => (
                    <Button
                      key={b.id || b.label}
                      type="button"
                      size="sm"
                      variant="outline"
                      className="text-xs h-8"
                      disabled={running}
                      onClick={() => {
                        appendLog(`[quick] ${b.label || 'prompt'}`);
                        if (b.text && /https?:\/\//i.test(b.text.trim())) {
                          setAvatarUrl(b.text.trim());
                        } else if (b.text) {
                          appendLog(b.text);
                        }
                      }}
                    >
                      {b.label || 'Button'}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {lastOutput && (
              <div className="text-xs">
                <span className="text-muted-foreground">Final output: </span>
                <code className="break-all">{lastOutput}</code>
              </div>
            )}

            <div className="space-y-1">
              <Label className="text-xs">Progress log</Label>
              <pre className="text-[11px] bg-muted/40 rounded p-2 max-h-40 overflow-auto whitespace-pre-wrap">
                {logLines.join('\n') || '—'}
              </pre>
            </div>
          </div>
        </ScrollArea>

        <div className="border-t px-4 py-3 flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={running}>
            Close
          </Button>
          <Button type="button" onClick={runBatch} disabled={running}>
            {running ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running…
              </>
            ) : (
              'Run batch'
            )}
          </Button>
        </div>
      </div>

      {sd.open && (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-4">
          <div className="bg-card border rounded-lg max-w-lg w-full p-4 space-y-3">
            <div className="font-medium">{sd.mode === 'avatar' ? 'Create avatar (SD)' : 'Create background (SD)'}</div>
            <Textarea
              className="min-h-[120px] text-sm"
              value={sd.prompt}
              onChange={(e) => setSd((p) => ({ ...p, prompt: e.target.value }))}
            />
            {sd.error && <p className="text-xs text-destructive">{sd.error}</p>}
            {sd.previewUrl && (
              <img src={sd.previewUrl} alt="preview" className="max-h-48 rounded border mx-auto" />
            )}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="ghost" onClick={() => setSd(emptySdState())}>
                Cancel
              </Button>
              <Button type="button" variant="secondary" onClick={runSd} disabled={sd.busy}>
                {sd.busy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Generate'}
              </Button>
              {sd.previewUrl && (
                <Button type="button" onClick={acceptSd}>
                  Accept
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DidPipelineOverlay;
