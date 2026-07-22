import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ChevronRight, Copy, Loader2, Sparkles } from 'lucide-react';
import { fetchWithTimeout, formatFetchError } from '../config/api';
import { useApp } from '../contexts/AppContext';
import { mergeNanoGptMemoryIntoPayload } from '../utils/nanoGptMemoryPayload';
import { cleanModelOutput } from '../utils/cleanOutput';
import {
  createRouteTraceId,
  extractRouteMetaFromGenerateResult,
  logRouteTrace,
  resolveUnifiedRequestRoute,
} from '../utils/requestRouting';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Textarea } from './ui/textarea';

const CURATOR_GENERATE_TIMEOUT_MS = 1_200_000;
const NEUTRAL_REVIEWER = '__neutral__';

function toCharacterCard(character) {
  if (!character || typeof character !== 'object') return null;
  const pick = (key) => (character[key] != null ? String(character[key]).trim() : '');
  return Object.fromEntries(Object.entries({
    name: pick('name') || undefined,
    description: pick('description') || undefined,
    personality: pick('personality') || undefined,
    scenario: pick('scenario') || undefined,
    speech_style: pick('speech_style') || undefined,
    background: pick('background') || undefined,
    model_instructions: pick('model_instructions') || undefined,
    ethics_justification: pick('ethics_justification') || undefined,
  }).filter(([, value]) => value));
}

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
  const [reviewerCharacterId, setReviewerCharacterId] = useState(NEUTRAL_REVIEWER);
  const [targetCharacterId, setTargetCharacterId] = useState('');
  const [extraNotes, setExtraNotes] = useState('');
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [promptPack, setPromptPack] = useState(null);
  const [rawResponse, setRawResponse] = useState('');
  const [parsedPreview, setParsedPreview] = useState(null);

  const npcCharacters = useMemo(
    () => (characters || []).filter((character) => character?.id && character.chat_role !== 'user'),
    [characters]
  );

  useEffect(() => {
    if (scope !== 'agentic' || targetCharacterId || npcCharacters.length === 0) return;
    setTargetCharacterId(npcCharacters[0].id);
  }, [npcCharacters, scope, targetCharacterId]);

  const reviewerCharacter = useMemo(
    () => npcCharacters.find((character) => character.id === reviewerCharacterId) || null,
    [npcCharacters, reviewerCharacterId]
  );
  const targetCharacter = useMemo(
    () => npcCharacters.find((character) => character.id === targetCharacterId) || null,
    [npcCharacters, targetCharacterId]
  );

  const userProfileSummary = useMemo(() => {
    if (!userProfile || typeof userProfile !== 'object') return '';
    try {
      return JSON.stringify({
        name: userProfile.name,
        username: userProfile.username,
        preferences: userProfile.preferences,
        activeContextSummary: userProfile.activeContextSummary,
      }).slice(0, 8000);
    } catch {
      return '';
    }
  }, [userProfile]);

  const selectedModel = useMemo(() => {
    if (primaryModel) return primaryModel;
    if (typeof activeModel === 'string' && activeModel.trim()) return activeModel.trim();
    return loadedModels.find((model) => model.gpu_id === 0)?.name || null;
  }, [activeModel, loadedModels, primaryModel]);

  const requestRoute = useMemo(
    () => resolveUnifiedRequestRoute({
      primaryModel: selectedModel,
      primaryIsAPI,
      settings,
      requestPurpose: 'memory_curation',
    }),
    [primaryIsAPI, selectedModel, settings]
  );
  const canRunModel = Boolean(PRIMARY_API_URL && requestRoute.effectiveModel);

  const requestPromptPack = useCallback(async () => {
    if (!activeProfileId) throw new Error('Choose a user profile first.');
    if (scope === 'agentic' && !targetCharacterId) throw new Error('Choose whose memories should be cleaned.');
    const body = {
      mode: scope,
      user_id: activeProfileId,
      user_display_name: (userProfile?.name || userProfile?.username || '').trim() || 'User',
      user_profile_summary: userProfileSummary || undefined,
      curator_character_name: reviewerCharacter?.name || 'Neutral reviewer',
      curator_character_card: toCharacterCard(reviewerCharacter) || undefined,
      extra_notes: extraNotes.trim() || undefined,
    };
    if (scope === 'agentic') {
      body.target_character_id = targetCharacterId;
      body.target_character_name = targetCharacter?.name || targetCharacterId;
    }
    const response = await fetchWithTimeout(
      `${apiUrl}/memory/curator/prompt_pack`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      120000
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status !== 'success' || !data.prompt_pack) {
      throw new Error(data.detail || data.message || 'Mirid could not gather these memories.');
    }
    setPromptPack(data.prompt_pack);
    return data.prompt_pack;
  }, [
    activeProfileId,
    apiUrl,
    extraNotes,
    reviewerCharacter,
    scope,
    targetCharacter,
    targetCharacterId,
    userProfile,
    userProfileSummary,
  ]);

  const runSelectedModel = useCallback(async (pack) => {
    if (!canRunModel) throw new Error('Choose or load a text model before reviewing memories.');
    const traceId = createRouteTraceId();
    logRouteTrace({ action: 'memory_curation', route: requestRoute, requestPurpose: 'memory_curation', traceId });
    const configuredMaxTokens = settings?.max_tokens;
    const maxTokens = typeof configuredMaxTokens === 'number' && configuredMaxTokens > 0
      ? Math.min(configuredMaxTokens, 262144)
      : 65536;
    const payload = mergeNanoGptMemoryIntoPayload({
      prompt: typeof pack === 'string' ? pack : JSON.stringify(pack),
      model_name: requestRoute.effectiveModel || selectedModel,
      selected_model: requestRoute.selectedModel || undefined,
      round_robin_enabled: requestRoute.autoEnabled,
      max_tokens: maxTokens,
      temperature: typeof settings?.temperature === 'number' ? settings.temperature : 0.3,
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
      request_purpose: 'model_testing',
      memory_curation: true,
      skip_openai_message_pruning: true,
    }, settings);
    const response = await fetchWithTimeout(
      `${PRIMARY_API_URL}/generate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Router-Trace-Id': traceId },
        body: JSON.stringify(payload),
      },
      CURATOR_GENERATE_TIMEOUT_MS
    );
    const data = await response.json().catch(() => ({}));
    extractRouteMetaFromGenerateResult(data, response.headers);
    if (!response.ok) throw new Error(data.detail || data.message || `Status ${response.status}`);
    const raw = data.text ?? data.response ?? data?.choices?.[0]?.message?.content ?? data?.choices?.[0]?.text ?? '';
    if (!String(raw).trim()) throw new Error('The selected model returned no review.');
    const cleaned = cleanModelOutput(String(raw));
    setRawResponse(cleaned);
    return cleaned;
  }, [PRIMARY_API_URL, canRunModel, requestRoute, selectedModel, settings]);

  const parseResponse = useCallback(async (raw) => {
    const response = await fetchWithTimeout(
      `${apiUrl}/memory/curator/parse_response`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_response: raw, mode: scope }),
      },
      60000
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.detail || 'The model replied, but Mirid could not read the reviewed memory list.');
    }
    setParsedPreview(data);
    return data;
  }, [apiUrl, scope]);

  const reviewMemories = useCallback(async () => {
    setError('');
    setSuccess('');
    setParsedPreview(null);
    try {
      setStage('gathering');
      const pack = await requestPromptPack();
      setStage('reviewing');
      const raw = await runSelectedModel(pack);
      setStage('reading');
      await parseResponse(raw);
      setSuccess('Review complete. Nothing has been changed yet.');
    } catch (reviewError) {
      setError(formatFetchError(reviewError, { timeoutMs: CURATOR_GENERATE_TIMEOUT_MS }));
    } finally {
      setStage('');
    }
  }, [parseResponse, requestPromptPack, runSelectedModel]);

  const buildForAnotherModel = useCallback(async () => {
    setError('');
    setSuccess('');
    try {
      setStage('gathering');
      await requestPromptPack();
    } catch (buildError) {
      setError(formatFetchError(buildError, { timeoutMs: 120000 }));
    } finally {
      setStage('');
    }
  }, [requestPromptPack]);

  const parseEditedResponse = useCallback(async () => {
    if (!rawResponse.trim()) return;
    setError('');
    setSuccess('');
    try {
      setStage('reading');
      await parseResponse(rawResponse);
      setSuccess('Response read successfully. Nothing has been changed yet.');
    } catch (parseError) {
      setError(formatFetchError(parseError, { timeoutMs: 60000 }));
    } finally {
      setStage('');
    }
  }, [parseResponse, rawResponse]);

  const applyCurated = useCallback(async () => {
    if (!parsedPreview || parsedPreview.status !== 'success') return;
    const rows = scope === 'profile' ? parsedPreview.memories : parsedPreview.insights;
    if (!Array.isArray(rows)) return;
    const targetName = scope === 'profile' ? 'your profile memories' : `${targetCharacter?.name || 'this character'}’s memories`;
    if (!window.confirm(`Replace ${targetName} with these ${rows.length} reviewed entries? This cannot be undone automatically.`)) return;
    setError('');
    setSuccess('');
    try {
      setStage('saving');
      const endpoint = scope === 'profile' ? 'apply_profile' : 'apply_agentic';
      const body = scope === 'profile'
        ? { user_id: activeProfileId, memories: rows }
        : { user_id: activeProfileId, character_id: targetCharacterId, insights: rows };
      const response = await fetchWithTimeout(
        `${apiUrl}/memory/curator/${endpoint}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
        60000
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Status ${response.status}`);
      setSuccess(`${rows.length} reviewed ${rows.length === 1 ? 'memory was' : 'memories were'} saved.`);
      onApplied?.();
    } catch (applyError) {
      setError(formatFetchError(applyError, { timeoutMs: 60000 }));
    } finally {
      setStage('');
    }
  }, [activeProfileId, apiUrl, onApplied, parsedPreview, scope, targetCharacter, targetCharacterId]);

  const copyPrompt = useCallback(async () => {
    if (!promptPack?.combined) return;
    try {
      await navigator.clipboard.writeText(promptPack.combined);
      setSuccess('Prompt copied.');
    } catch {
      setError('Mirid could not copy the prompt to the clipboard.');
    }
  }, [promptPack]);

  if (!apiReady) return <p className="text-sm text-muted-foreground">Connecting to memory storage…</p>;

  const reviewedRows = scope === 'profile' ? parsedPreview?.memories : parsedPreview?.insights;
  const stageText = {
    gathering: 'Gathering saved memories…',
    reviewing: `Asking ${requestRoute.effectiveModel || 'the selected model'} to review them…`,
    reading: 'Reading the reviewed list…',
    saving: 'Saving reviewed memories…',
  }[stage];

  return (
    <div className="space-y-5 rounded-xl border bg-card/70 p-5">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label>Review in the voice of</Label>
          <Select value={reviewerCharacterId} onValueChange={setReviewerCharacterId}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value={NEUTRAL_REVIEWER}>Neutral reviewer</SelectItem>
              {npcCharacters.map((character) => (
                <SelectItem key={character.id} value={character.id}>{character.name || character.id}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs leading-relaxed text-muted-foreground">
            This keeps the review in a familiar voice. It does not permit invented memories.
          </p>
        </div>

        {scope === 'agentic' ? (
          <div className="space-y-2">
            <Label>Whose memories should change?</Label>
            <Select value={targetCharacterId} onValueChange={setTargetCharacterId}>
              <SelectTrigger><SelectValue placeholder="Choose a character" /></SelectTrigger>
              <SelectContent>
                {npcCharacters.map((character) => (
                  <SelectItem key={character.id} value={character.id}>{character.name || character.id}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Only this character’s memory list can be replaced when you approve the result.
            </p>
          </div>
        ) : (
          <div className="rounded-lg border bg-muted/15 p-3 text-sm">
            <p className="font-medium">Memories being reviewed</p>
            <p className="mt-1 text-muted-foreground">Facts saved with {userProfile?.name || userProfile?.username || 'the current user profile'}.</p>
          </div>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor={`memory-review-notes-${scope}`}>What should the review preserve or correct? <span className="font-normal text-muted-foreground">Optional</span></Label>
        <Textarea
          id={`memory-review-notes-${scope}`}
          value={extraNotes}
          onChange={(event) => setExtraNotes(event.target.value)}
          placeholder="For example: preserve emotional boundaries; remove outdated preferences."
          className="min-h-[72px]"
        />
      </div>

      <div className="rounded-lg border bg-muted/15 p-3 text-sm">
        <p className="font-medium">Model used for the review</p>
        <p className="mt-1 text-muted-foreground">{canRunModel ? requestRoute.effectiveModel : 'No text model selected'}</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" onClick={reviewMemories} disabled={Boolean(stage) || !activeProfileId || !canRunModel || (scope === 'agentic' && !targetCharacterId)}>
          {stage && stage !== 'saving' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
          Review memories
        </Button>
        {stageText ? <span className="text-sm text-muted-foreground">{stageText}</span> : null}
      </div>

      {error ? <p className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</p> : null}
      {success ? <p className="flex items-center gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-800 dark:text-emerald-200"><CheckCircle2 className="h-4 w-4" />{success}</p> : null}

      {Array.isArray(reviewedRows) ? (
        <div className="space-y-3 rounded-xl border border-primary/25 bg-primary/5 p-4">
          <div>
            <h3 className="font-semibold">Reviewed memory list</h3>
            <p className="mt-1 text-xs text-muted-foreground">{reviewedRows.length} entries proposed. Nothing changes until you save them.</p>
          </div>
          {parsedPreview?.summary ? <p className="text-sm text-muted-foreground">{parsedPreview.summary}</p> : null}
          <details className="rounded-md border bg-background/60 px-3 py-2 text-xs">
            <summary className="cursor-pointer font-medium">Preview reviewed entries</summary>
            <ul className="mt-3 space-y-2">
              {reviewedRows.slice(0, 12).map((row, index) => <li key={row.id || `${index}-${row.content}`}>{row.content}</li>)}
            </ul>
          </details>
          <Button type="button" onClick={applyCurated} disabled={stage === 'saving'}>
            {stage === 'saving' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <CheckCircle2 className="mr-2 h-4 w-4" />}
            Save reviewed memories
          </Button>
        </div>
      ) : null}

      <details className="group rounded-lg border bg-muted/10">
        <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium hover:bg-muted/30">
          <ChevronRight className="h-4 w-4 transition-transform group-open:rotate-90" />
          Run elsewhere or inspect the generated prompt
        </summary>
        <div className="space-y-3 border-t px-3 py-4">
          <p className="text-xs leading-relaxed text-muted-foreground">Use this only when you want to run the review in another application or repair a model response manually.</p>
          <div className="flex flex-wrap gap-2">
            <Button type="button" size="sm" variant="outline" onClick={buildForAnotherModel} disabled={Boolean(stage) || !activeProfileId}>Build prompt</Button>
            <Button type="button" size="sm" variant="ghost" onClick={copyPrompt} disabled={!promptPack?.combined}><Copy className="mr-2 h-4 w-4" />Copy prompt</Button>
          </div>
          {promptPack?.combined ? <Textarea readOnly value={promptPack.combined} className="min-h-[140px] font-mono text-xs" spellCheck={false} /> : null}
          <Textarea value={rawResponse} onChange={(event) => setRawResponse(event.target.value)} className="min-h-[120px] font-mono text-xs" placeholder="Paste a model response here" spellCheck={false} />
          <Button type="button" size="sm" variant="secondary" onClick={parseEditedResponse} disabled={Boolean(stage) || !rawResponse.trim()}>Read pasted response</Button>
        </div>
      </details>
    </div>
  );
}
