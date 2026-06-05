import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { fetchWithTimeout, formatFetchError } from '../config/api';
import { useApp } from '../contexts/AppContext';
import { mergeNanoGptMemoryIntoPayload } from '../utils/nanoGptMemoryPayload';
import { cleanModelOutput } from '../utils/cleanOutput';
import { createRouteTraceId, extractRouteMetaFromGenerateResult, logRouteTrace, resolveUnifiedRequestRoute } from '../utils/requestRouting';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Loader2, Copy, Sparkles, CheckCircle2, Play } from 'lucide-react';

const CURATOR_GENERATE_TIMEOUT_MS = 1_200_000;

function toCharacterCard(character) {
  if (!character || typeof character !== 'object') return null;
  const pick = (k) => (character[k] != null ? String(character[k]).trim() : '');
  const card = {
    name: pick('name') || undefined,
    description: pick('description') || undefined,
    personality: pick('personality') || undefined,
    scenario: pick('scenario') || undefined,
    speech_style: pick('speech_style') || undefined,
    background: pick('background') || undefined,
    model_instructions: pick('model_instructions') || undefined,
    ethics_justification: pick('ethics_justification') || undefined,
  };
  return Object.fromEntries(Object.entries(card).filter(([, v]) => v));
}

/**
 * In-character memory curator for Settings Memory Browser (profile + agentic tabs).
 *
 * @param {'profile'|'agentic'} scope
 */
export default function MemoryCuratorPanel({
  apiUrl,
  apiReady,
  activeProfileId,
  userProfile,
  characters = [],
  scope,
  onApplied,
}) {
  const {
    PRIMARY_API_URL,
    primaryModel,
    primaryIsAPI,
    activeModel,
    loadedModels = [],
    settings,
  } = useApp();
  const [runModelBusy, setRunModelBusy] = useState(false);

  const autoRouterEnabled = settings?.apiEndpointRoundRobinEnabled === true;

  /** Same resolution order as chat: navbar primary → active → GPU0 local load. */
  const curatorSelectedModel = useMemo(() => {
    if (primaryModel) return primaryModel;
    if (typeof activeModel === 'string' && activeModel.trim()) return activeModel.trim();
    const gpu0 = loadedModels.find((m) => m.gpu_id === 0);
    return gpu0?.name || null;
  }, [primaryModel, activeModel, loadedModels]);

  const curatorRoute = useMemo(
    () =>
      resolveUnifiedRequestRoute({
        primaryModel: curatorSelectedModel,
        primaryIsAPI,
        settings,
        requestPurpose: 'memory_curation',
      }),
    [curatorSelectedModel, primaryIsAPI, settings]
  );

  const canRunCuratorModel = Boolean(PRIMARY_API_URL && curatorRoute.effectiveModel);

  const npcCharacters = useMemo(
    () => (characters || []).filter((c) => c?.id && c.chat_role !== 'user'),
    [characters]
  );

  const [curatorCharacterId, setCuratorCharacterId] = useState('');
  const [agenticTargetId, setAgenticTargetId] = useState('');
  const [extraNotes, setExtraNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [pack, setPack] = useState(null);
  const [pastedRaw, setPastedRaw] = useState('');
  const [parsedPreview, setParsedPreview] = useState(null);
  const [parseError, setParseError] = useState(null);
  const [applyBusy, setApplyBusy] = useState(false);
  const [savePreviewToDisk, setSavePreviewToDisk] = useState(false);

  useEffect(() => {
    if (!curatorCharacterId && npcCharacters.length) {
      setCuratorCharacterId(npcCharacters[0].id);
    }
  }, [curatorCharacterId, npcCharacters]);

  useEffect(() => {
    if (scope !== 'agentic') return;
    if (!agenticTargetId && npcCharacters.length) {
      setAgenticTargetId(npcCharacters[0].id);
    }
  }, [scope, agenticTargetId, npcCharacters]);

  const curatorCharacter = useMemo(
    () => npcCharacters.find((c) => c.id === curatorCharacterId),
    [npcCharacters, curatorCharacterId]
  );

  const targetCharacter = useMemo(
    () => npcCharacters.find((c) => c.id === agenticTargetId),
    [npcCharacters, agenticTargetId]
  );

  const userProfileSummary = useMemo(() => {
    if (!userProfile || typeof userProfile !== 'object') return '';
    try {
      const slice = {
        name: userProfile.name,
        username: userProfile.username,
        preferences: userProfile.preferences,
        activeContextSummary: userProfile.activeContextSummary,
      };
      return JSON.stringify(slice, null, 0).slice(0, 8000);
    } catch {
      return '';
    }
  }, [userProfile]);

  const buildPack = useCallback(async () => {
    if (!activeProfileId || !curatorCharacterId) {
      setError('Select a profile and a curator character.');
      return;
    }
    if (scope === 'agentic' && !agenticTargetId) {
      setError('Select which character’s agentic memory file to curate.');
      return;
    }

    setBusy(true);
    setError(null);
    setPack(null);
    setParsedPreview(null);
    setParseError(null);

    const card = toCharacterCard(curatorCharacter);
    const body = {
      mode: scope,
      user_id: activeProfileId,
      user_display_name: (userProfile?.name || userProfile?.username || '').trim() || undefined,
      user_profile_summary: userProfileSummary || undefined,
      curator_character_name: curatorCharacter?.name || undefined,
      curator_character_card: card || undefined,
      extra_notes: extraNotes.trim() || undefined,
      save_preview_to_disk: savePreviewToDisk,
    };
    if (scope === 'agentic') {
      body.target_character_id = agenticTargetId;
      body.target_character_name = targetCharacter?.name || undefined;
    }

    try {
      const res = await fetchWithTimeout(
        `${apiUrl}/memory/curator/prompt_pack`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
        120000
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      if (data.status !== 'success' || !data.combined) throw new Error('Unexpected response');
      setPack(data);
    } catch (e) {
      setError(formatFetchError(e, { timeoutMs: 120000 }));
    } finally {
      setBusy(false);
    }
  }, [
    activeProfileId,
    curatorCharacterId,
    curatorCharacter,
    userProfile,
    userProfileSummary,
    extraNotes,
    scope,
    agenticTargetId,
    targetCharacter,
    apiUrl,
    savePreviewToDisk,
  ]);

  const copyCombined = async () => {
    if (!pack?.combined) return;
    try {
      await navigator.clipboard.writeText(pack.combined);
    } catch {
      /* ignore */
    }
  };

  const runModelOnCuratorPrompt = useCallback(async () => {
    if (!pack?.combined?.trim()) {
      setError('Build the curator prompt first.');
      return;
    }
    if (!canRunCuratorModel) {
      setError(
        autoRouterEnabled
          ? 'No model in the auto-routing pool. Add enabled API endpoints under Settings → LLM Settings → Custom API Endpoints, or pick a model from the navbar pill.'
          : 'No model available. Pick a model from the navbar pill (top) or load a local model on GPU 0.'
      );
      return;
    }
    setRunModelBusy(true);
    setError(null);
    setParsedPreview(null);
    setParseError(null);
    try {
      const route = curatorRoute;
      const traceId = createRouteTraceId();
      logRouteTrace({
        action: 'memory_curation',
        route,
        requestPurpose: 'memory_curation',
        traceId,
      });
      const mt = settings?.max_tokens;
      const maxTok = typeof mt === 'number' && mt > 0 ? Math.min(mt, 262144) : 65536;
      const payload = mergeNanoGptMemoryIntoPayload(
        {
          prompt: pack.combined,
          model_name: route.effectiveModel || curatorSelectedModel,
          selected_model: route.selectedModel || undefined,
          round_robin_enabled: route.autoEnabled,
          max_tokens: maxTok,
          temperature: typeof settings?.temperature === 'number' ? settings.temperature : 0.35,
          top_p: settings?.top_p ?? 0.9,
          top_k: settings?.top_k ?? 40,
          repetition_penalty: settings?.repetition_penalty ?? 1.05,
          frequency_penalty: settings?.frequencyPenalty ?? 0,
          presence_penalty: settings?.presencePenalty ?? 0,
          memoryEnabled: false,
          directProfileInjection: false,
          stream: false,
          use_rag: false,
          use_web_search: false,
          gpu_id: 0,
          request_purpose: 'continuation',
          memory_curation: true,
          skip_openai_message_pruning: true,
        },
        settings
      );
      const res = await fetchWithTimeout(
        `${PRIMARY_API_URL}/generate`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Router-Trace-Id': traceId },
          body: JSON.stringify(payload),
        },
        CURATOR_GENERATE_TIMEOUT_MS
      );
      const data = await res.json().catch(() => ({}));
      extractRouteMetaFromGenerateResult(data, res.headers);
      if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      const raw =
        data.text ??
        data.response ??
        data?.choices?.[0]?.message?.content ??
        data?.choices?.[0]?.text ??
        '';
      if (!String(raw).trim()) {
        throw new Error('Model returned empty text.');
      }
      setPastedRaw(cleanModelOutput(String(raw)));
    } catch (e) {
      let msg = e?.message || String(e);
      if (e?.name === 'AbortError') {
        msg = `Run model timed out after ${CURATOR_GENERATE_TIMEOUT_MS / 60000} minutes. Try a smaller pack or lower max_tokens.`;
      } else if (/Failed to fetch|NetworkError|Load failed/i.test(msg)) {
        msg = `${msg}\n\nPrimary API: ${PRIMARY_API_URL || '(not set)'}`;
      }
      setError(msg);
    } finally {
      setRunModelBusy(false);
    }
  }, [pack, PRIMARY_API_URL, curatorRoute, curatorSelectedModel, canRunCuratorModel, autoRouterEnabled, settings]);

  const parsePasted = async () => {
    setParseError(null);
    setParsedPreview(null);
    if (!pastedRaw.trim()) {
      setParseError('Paste the model JSON response first.');
      return;
    }
    try {
      const res = await fetchWithTimeout(
        `${apiUrl}/memory/curator/parse_response`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            raw_text: pastedRaw,
            mode: scope,
          }),
        },
        60000
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setParsedPreview(data);
    } catch (e) {
      setParseError(e?.message || String(e));
    }
  };

  const applyCurated = async () => {
    if (!parsedPreview || parsedPreview.status !== 'success') {
      alert('Parse the response first.');
      return;
    }
    const noun = scope === 'profile' ? 'profile memories' : 'agentic insights';
    if (
      !window.confirm(
        `Replace all ${noun} on disk with this curated list (${parsedPreview.memories?.length ?? parsedPreview.insights?.length ?? 0} rows)? This cannot be undone automatically.`
      )
    ) {
      return;
    }

    setApplyBusy(true);
    setError(null);
    try {
      if (scope === 'profile') {
        const memories = parsedPreview.memories;
        if (!Array.isArray(memories)) throw new Error('Invalid parse payload');
        const res = await fetchWithTimeout(
          `${apiUrl}/memory/curator/apply_profile`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: activeProfileId, memories }),
          },
          60000
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      } else {
        const insights = parsedPreview.insights;
        if (!Array.isArray(insights)) throw new Error('Invalid parse payload');
        const res = await fetchWithTimeout(
          `${apiUrl}/memory/curator/apply_agentic`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              user_id: activeProfileId,
              character_id: agenticTargetId,
              insights,
            }),
          },
          60000
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setPastedRaw('');
      setParsedPreview(null);
      onApplied?.();
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setApplyBusy(false);
    }
  };

  if (!apiReady) return null;

  return (
    <Alert className="mb-6 border-primary/25 bg-muted/30">
      <Sparkles className="h-4 w-4" />
      <AlertTitle>In-character memory curator</AlertTitle>
      <AlertDescription className="mt-3 space-y-4 text-sm">
        <p className="text-muted-foreground">
          Build a prompt where a character you choose reviews{' '}
          {scope === 'profile'
            ? 'your backend profile memories'
            : 'the agentic insight file for the character you pick'}
          . Use <strong>Run model</strong> with the same model as chat (navbar pill or ⟳ auto-routing), or paste JSON
          from another client. Then <strong>Parse JSON</strong> and <strong>Apply to disk</strong>.
        </p>

        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <Label>Review as (persona)</Label>
            <Select value={curatorCharacterId} onValueChange={setCuratorCharacterId}>
              <SelectTrigger>
                <SelectValue placeholder="Character" />
              </SelectTrigger>
              <SelectContent>
                {npcCharacters.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {c.name || c.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {scope === 'agentic' && (
            <div className="space-y-2">
              <Label>Agentic memory file (character)</Label>
              <Select value={agenticTargetId} onValueChange={setAgenticTargetId}>
                <SelectTrigger>
                  <SelectValue placeholder="Character" />
                </SelectTrigger>
                <SelectContent>
                  {npcCharacters.map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      {c.name || c.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                This is whose JSON insight store gets replaced when you apply — independent of “review as”.
              </p>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Label>Notes for this pass (optional)</Label>
          <Textarea
            value={extraNotes}
            onChange={(e) => setExtraNotes(e.target.value)}
            placeholder="e.g. prioritize emotional boundaries, drop stale tech prefs…"
            className="min-h-[72px] text-sm"
          />
        </div>

        <label className="flex items-start gap-2 text-sm cursor-pointer rounded-md border border-border/50 bg-muted/15 px-3 py-2">
          <Checkbox checked={savePreviewToDisk} onCheckedChange={(v) => setSavePreviewToDisk(!!v)} className="mt-0.5" />
          <span>
            <span className="font-medium">Test pass:</span> save the full prompt to{' '}
            <code className="text-[11px] px-1 rounded bg-muted">backend/data/preview_prompts/</code> on the server (no LLM call).
          </span>
        </label>

        <div className="flex flex-wrap gap-2 items-center">
          <Button type="button" size="sm" onClick={buildPack} disabled={busy || !activeProfileId}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Sparkles className="h-4 w-4 mr-2" />}
            Build curator prompt
          </Button>
          {pack?.combined && (
            <Button type="button" size="sm" variant="outline" onClick={copyCombined}>
              <Copy className="h-4 w-4 mr-2" />
              Copy prompt
            </Button>
          )}
          {pack?.combined && (
            <Button
              type="button"
              size="sm"
              variant="default"
              onClick={runModelOnCuratorPrompt}
              disabled={runModelBusy || !canRunCuratorModel}
            >
              {runModelBusy ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
              Run model
            </Button>
          )}
          {pack?.stats && (
            <span className="text-xs text-muted-foreground">
              ~{pack.stats.bundle_chars?.toLocaleString?.() ?? pack.stats.bundle_chars} chars ·{' '}
              {pack.stats.indexed_rows} rows indexed
            </span>
          )}
        </div>

        {pack?.combined && !canRunCuratorModel && (
          <p className="text-xs text-amber-800 dark:text-amber-300">
            {autoRouterEnabled ? (
              <>
                <strong>Run model</strong> needs at least one enabled endpoint in your auto-routing pool (Settings →
                LLM Settings → Custom API Endpoints). Or turn off ⟳ Auto-routing and pick a model from the{' '}
                <strong>navbar</strong> pill.
              </>
            ) : (
              <>
                Pick a model from the <strong>navbar</strong> pill at the top of the app, or load a local model on GPU
                0, to use <strong>Run model</strong> (otherwise copy the prompt and run it elsewhere).
              </>
            )}
          </p>
        )}

        {error && (
          <div className="text-sm text-red-600 bg-red-50 dark:bg-red-950/30 rounded px-3 py-2 border border-red-200/60">
            {error}
          </div>
        )}

        {pack?.preview_saved && (
          <div className="text-xs rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-emerald-900 dark:text-emerald-200 space-y-1">
            <p className="font-medium">Saved preview on disk</p>
            <p className="font-mono break-all opacity-90">{pack.preview_saved.absolute_path}</p>
            <p className="text-muted-foreground">
              {pack.preview_saved.bytes_written?.toLocaleString?.() ?? pack.preview_saved.bytes_written} bytes
            </p>
          </div>
        )}

        {pack?.combined && (
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground">Output schema (what the model must return)</summary>
            <pre className="mt-2 max-h-40 overflow-auto rounded border bg-background/80 p-2 whitespace-pre-wrap">
              {pack.output_spec}
            </pre>
          </details>
        )}

        <div className="space-y-2 border-t border-border/60 pt-4">
          <Label className="text-xs">Model JSON response (filled by Run model, or paste)</Label>
          <Textarea
            value={pastedRaw}
            onChange={(e) => setPastedRaw(e.target.value)}
            className="min-h-[120px] text-xs font-mono"
            spellCheck={false}
            placeholder='Paste the JSON object (starts with "{") … or use Run model above.'
          />
          <div className="flex flex-wrap gap-2">
            <Button type="button" size="sm" variant="secondary" onClick={parsePasted} disabled={!pastedRaw.trim()}>
              Parse JSON
            </Button>
            <Button
              type="button"
              size="sm"
              variant="default"
              onClick={applyCurated}
              disabled={applyBusy || !parsedPreview || parsedPreview.status !== 'success'}
            >
              {applyBusy ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <CheckCircle2 className="h-4 w-4 mr-2" />}
              Apply to disk
            </Button>
          </div>
          {parseError && <p className="text-xs text-red-600">{parseError}</p>}
          {parsedPreview?.status === 'success' && (
            <p className="text-xs text-green-700 dark:text-green-400">
              Parsed{' '}
              {(scope === 'profile' ? parsedPreview.memories?.length : parsedPreview.insights?.length) ?? 0} rows
              {parsedPreview.summary ? ` — ${parsedPreview.summary}` : ''}
            </p>
          )}
        </div>
      </AlertDescription>
    </Alert>
  );
}
