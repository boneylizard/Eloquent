import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, ChevronRight, Copy, Loader2, Sparkles } from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import { fetchWithTimeout, formatFetchError } from '../config/api';
import { mergeNanoGptMemoryIntoPayload } from '../utils/nanoGptMemoryPayload';
import { cleanModelOutput } from '../utils/cleanOutput';
import {
  createRouteTraceId,
  extractRouteMetaFromGenerateResult,
  logRouteTrace,
  resolveUnifiedRequestRoute,
} from '../utils/requestRouting';
import { Button } from './ui/button';
import { Checkbox } from './ui/checkbox';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Textarea } from './ui/textarea';

const REALIGNMENT_TIMEOUT_MS = 1_200_000;

function speakerLabel(message) {
  if (message?.role === 'user') return 'User';
  if (message?.role === 'assistant' || message?.role === 'bot') return 'Assistant';
  return message?.role ? String(message.role) : 'Message';
}

function transcriptFromMessages(messages, maxChars) {
  if (!Array.isArray(messages) || !messages.length) return '';
  const parts = [];
  let total = 0;
  for (const message of messages) {
    const body = String(message?.content ?? '').trim();
    if (!body) continue;
    const line = `${speakerLabel(message)}: ${body}\n\n`;
    if (total + line.length > maxChars) break;
    parts.push(line);
    total += line.length;
  }
  return parts.join('').trim();
}

function toCharacterCard(character) {
  if (!character || typeof character !== 'object') return null;
  const pick = (key) => (character[key] != null ? String(character[key]).trim() : '');
  return {
    name: pick('name') || undefined,
    description: pick('description') || undefined,
    personality: pick('personality') || undefined,
    scenario: pick('scenario') || undefined,
    speech_style: pick('speech_style') || undefined,
    background: pick('background') || undefined,
    model_instructions: pick('model_instructions') || undefined,
    ethics_justification: pick('ethics_justification') || undefined,
    example_dialogue: Array.isArray(character.example_dialogue) ? character.example_dialogue : undefined,
  };
}

export default function PersonaRealignmentPanel() {
  const { activeProfileId } = useMemory();
  const {
    MEMORY_API_URL,
    PRIMARY_API_URL,
    portsReady,
    storageHydrated,
    characters = [],
    conversations = [],
    activeConversation,
    activeCharacter,
    userProfile,
    buildSystemPrompt,
    primaryModel,
    primaryIsAPI,
    activeModel,
    loadedModels = [],
    settings,
    saveCharacter,
  } = useApp();

  const apiReady = portsReady && storageHydrated;
  const [characterId, setCharacterId] = useState('');
  const [includeTranscriptActive, setIncludeTranscriptActive] = useState(true);
  const [includeRollingActive, setIncludeRollingActive] = useState(true);
  const [includeBackendMemories, setIncludeBackendMemories] = useState(true);
  const [includeCharacterMemories, setIncludeCharacterMemories] = useState(true);
  const [alsoRewriteUserProfile, setAlsoRewriteUserProfile] = useState(false);
  const [profileRewriteMode, setProfileRewriteMode] = useState('merge');
  const [extraConvId, setExtraConvId] = useState('');
  const [includeRollingExtra, setIncludeRollingExtra] = useState(false);
  const [transcriptMaxChars, setTranscriptMaxChars] = useState(120000);
  const [agenticMode, setAgenticMode] = useState('ranked');
  const [agenticMaxChars, setAgenticMaxChars] = useState(48000);
  const [ragQuery, setRagQuery] = useState('');
  const [extraNotes, setExtraNotes] = useState('');
  const [currentInstructions, setCurrentInstructions] = useState('');
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [promptPack, setPromptPack] = useState(null);
  const [rawResponse, setRawResponse] = useState('');
  const [parsedResult, setParsedResult] = useState(null);
  const [proposedInstructions, setProposedInstructions] = useState('');

  const selectableCharacters = useMemo(
    () => (characters || []).filter((character) => character?.id && character.chat_role !== 'user'),
    [characters]
  );

  useEffect(() => {
    if (characterId || selectableCharacters.length === 0) return;
    const preferred = selectableCharacters.find((character) => character.id === activeCharacter?.id);
    setCharacterId(preferred?.id || selectableCharacters[0].id);
  }, [activeCharacter?.id, characterId, selectableCharacters]);

  const selectedCharacter = useMemo(
    () => selectableCharacters.find((character) => character.id === characterId) || null,
    [characterId, selectableCharacters]
  );

  useEffect(() => {
    if (!selectedCharacter) {
      setCurrentInstructions('');
      return;
    }
    try {
      setCurrentInstructions(buildSystemPrompt(selectedCharacter) || selectedCharacter.model_instructions || '');
    } catch {
      setCurrentInstructions(selectedCharacter.model_instructions || '');
    }
    setPromptPack(null);
    setRawResponse('');
    setParsedResult(null);
    setProposedInstructions('');
    setError('');
    setSuccess('');
  }, [buildSystemPrompt, selectedCharacter]);

  const activeChat = useMemo(
    () => (conversations || []).find((conversation) => conversation.id === activeConversation),
    [activeConversation, conversations]
  );
  const extraChat = useMemo(
    () => (conversations || []).find((conversation) => conversation.id === extraConvId),
    [conversations, extraConvId]
  );

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

  const buildPayload = useCallback(() => {
    const transcripts = [];
    if (includeTranscriptActive && activeChat?.messages?.length) {
      const transcript = transcriptFromMessages(activeChat.messages, Number(transcriptMaxChars) || 120000);
      if (transcript) transcripts.push(`[Transcript: ${activeChat.title || 'Active chat'}]\n\n${transcript}`);
    }
    if (extraConvId && extraChat?.messages?.length) {
      const transcript = transcriptFromMessages(
        extraChat.messages,
        Math.floor((Number(transcriptMaxChars) || 120000) / 2)
      );
      if (transcript) transcripts.push(`[Transcript: ${extraChat.title || 'Additional chat'}]\n\n${transcript}`);
    }

    const rollingPacks = [];
    if (includeRollingActive && activeChat?.rollingMemoryPack?.trim()) {
      rollingPacks.push(`[Rolling summary: ${activeChat.title || 'Active chat'}]\n\n${activeChat.rollingMemoryPack.trim()}`);
    }
    if (includeRollingExtra && extraChat?.rollingMemoryPack?.trim()) {
      rollingPacks.push(`[Rolling summary: ${extraChat.title || 'Additional chat'}]\n\n${extraChat.rollingMemoryPack.trim()}`);
    }

    const displayName = (userProfile?.name || userProfile?.username || userProfile?.displayName || '').trim();
    return {
      user_id: activeProfileId,
      character_id: characterId,
      character_name: selectedCharacter?.name || undefined,
      user_display_name: displayName || undefined,
      character_card: toCharacterCard(selectedCharacter),
      current_character_instructions: currentInstructions,
      rolling_packs: rollingPacks.length ? rollingPacks : undefined,
      transcripts: transcripts.length ? transcripts : undefined,
      include_backend_memories: includeBackendMemories,
      agentic_mode: includeCharacterMemories ? agenticMode : 'none',
      agentic_max_chars: Math.min(500000, Math.max(4000, Number(agenticMaxChars) || 48000)),
      agentic_rag_query: includeCharacterMemories && agenticMode === 'rag' && ragQuery.trim() ? ragQuery.trim() : undefined,
      extra_notes: extraNotes.trim() || undefined,
      also_rewrite_user_profile: alsoRewriteUserProfile,
      user_profile_rewrite_mode: profileRewriteMode === 'from_scratch' ? 'from_scratch' : 'merge',
    };
  }, [
    activeChat,
    activeProfileId,
    agenticMaxChars,
    agenticMode,
    alsoRewriteUserProfile,
    characterId,
    currentInstructions,
    extraChat,
    extraConvId,
    extraNotes,
    includeBackendMemories,
    includeCharacterMemories,
    includeRollingActive,
    includeRollingExtra,
    includeTranscriptActive,
    profileRewriteMode,
    ragQuery,
    selectedCharacter,
    transcriptMaxChars,
    userProfile,
  ]);

  const requestPromptPack = useCallback(async () => {
    const response = await fetchWithTimeout(
      `${MEMORY_API_URL}/memory/persona_realignment/prompt_pack`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      },
      120000
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status !== 'success' || !data.combined) {
      throw new Error(data.detail || data.message || 'Mirid could not gather the selected memory context.');
    }
    setPromptPack(data);
    return data;
  }, [MEMORY_API_URL, buildPayload]);

  const runSelectedModel = useCallback(async (pack) => {
    const traceId = createRouteTraceId();
    logRouteTrace({ action: 'persona_realignment', route: requestRoute, requestPurpose: 'memory_curation', traceId });
    const configuredMaxTokens = settings?.max_tokens;
    const maxTokens = typeof configuredMaxTokens === 'number' && configuredMaxTokens > 0
      ? Math.min(configuredMaxTokens, 262144)
      : 65536;
    const payload = mergeNanoGptMemoryIntoPayload({
      prompt: pack.combined,
      model_name: requestRoute.effectiveModel || selectedModel,
      selected_model: requestRoute.selectedModel || undefined,
      round_robin_enabled: requestRoute.autoEnabled,
      max_tokens: maxTokens,
      temperature: typeof settings?.temperature === 'number' ? settings.temperature : 0.25,
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
      REALIGNMENT_TIMEOUT_MS
    );
    const data = await response.json().catch(() => ({}));
    extractRouteMetaFromGenerateResult(data, response.headers);
    if (!response.ok) throw new Error(data.detail || data.message || `Status ${response.status}`);
    const raw = data.text ?? data.response ?? data?.choices?.[0]?.message?.content ?? data?.choices?.[0]?.text ?? '';
    if (!String(raw).trim()) throw new Error('The selected model returned no review.');
    const cleaned = cleanModelOutput(String(raw));
    setRawResponse(cleaned);
    return cleaned;
  }, [PRIMARY_API_URL, requestRoute, selectedModel, settings]);

  const parseResponse = useCallback(async (raw) => {
    const response = await fetchWithTimeout(
      `${MEMORY_API_URL}/memory/persona_realignment/parse_response`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_text: raw, character_id: characterId, user_id: activeProfileId }),
      },
      60000
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.status !== 'success') {
      throw new Error(data.detail || 'The model replied, but Mirid could not read the proposed changes.');
    }
    setParsedResult(data);
    setProposedInstructions(String(data.revised_model_instructions || '').trim());
    return data;
  }, [MEMORY_API_URL, activeProfileId, characterId]);

  const runAutomatedReview = useCallback(async () => {
    if (!activeProfileId || !selectedCharacter) {
      setError('Choose a user profile and character first.');
      return;
    }
    if (!canRunModel) {
      setError('Choose or load a text model before reviewing this character.');
      return;
    }
    setError('');
    setSuccess('');
    setParsedResult(null);
    setProposedInstructions('');
    try {
      setStage('gathering');
      const pack = await requestPromptPack();
      setStage('reviewing');
      const raw = await runSelectedModel(pack);
      setStage('reading');
      await parseResponse(raw);
      setSuccess('Review complete. Nothing has been changed yet.');
    } catch (reviewError) {
      setError(formatFetchError(reviewError, { timeoutMs: REALIGNMENT_TIMEOUT_MS }));
    } finally {
      setStage('');
    }
  }, [activeProfileId, canRunModel, parseResponse, requestPromptPack, runSelectedModel, selectedCharacter]);

  const buildForAnotherModel = useCallback(async () => {
    if (!activeProfileId || !selectedCharacter) {
      setError('Choose a user profile and character first.');
      return;
    }
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
  }, [activeProfileId, requestPromptPack, selectedCharacter]);

  const parseEditedResponse = useCallback(async () => {
    if (!rawResponse.trim()) {
      setError('Paste the model response first.');
      return;
    }
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

  const saveCharacterUpdate = useCallback(() => {
    if (!selectedCharacter || !proposedInstructions.trim()) return;
    const hasExisting = Boolean(String(selectedCharacter.model_instructions || '').trim());
    const prompt = hasExisting
      ? `Replace ${selectedCharacter.name}'s current Model Instructions with this reviewed version?`
      : `Save these reviewed Model Instructions to ${selectedCharacter.name}?`;
    if (!window.confirm(prompt)) return;
    saveCharacter({ ...selectedCharacter, model_instructions: proposedInstructions.trim() });
    setSuccess(`${selectedCharacter.name}'s Model Instructions were updated.`);
  }, [proposedInstructions, saveCharacter, selectedCharacter]);

  const applyProposedProfile = useCallback(async () => {
    const memories = parsedResult?.revised_user_profile_memories;
    if (!activeProfileId || !Array.isArray(memories) || memories.length === 0) return;
    if (!window.confirm(`Replace all saved profile memories with these ${memories.length} reviewed entries? This cannot be undone automatically.`)) return;
    setError('');
    setSuccess('');
    try {
      setStage('saving-profile');
      const response = await fetchWithTimeout(
        `${MEMORY_API_URL}/memory/curator/apply_profile`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: activeProfileId, memories }),
        },
        60000
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Status ${response.status}`);
      setSuccess(`${data.saved ?? memories.length} profile memories were saved.`);
    } catch (saveError) {
      setError(formatFetchError(saveError, { timeoutMs: 60000 }));
    } finally {
      setStage('');
    }
  }, [MEMORY_API_URL, activeProfileId, parsedResult]);

  const copyPrompt = useCallback(async () => {
    if (!promptPack?.combined) return;
    try {
      await navigator.clipboard.writeText(promptPack.combined);
      setSuccess('Prompt copied.');
    } catch {
      setError('Mirid could not copy the prompt to the clipboard.');
    }
  }, [promptPack]);

  const stageText = {
    gathering: 'Gathering the selected history…',
    reviewing: `Asking ${requestRoute.effectiveModel || 'the selected model'} to review it…`,
    reading: 'Reading the proposed changes…',
    'saving-profile': 'Saving reviewed profile memories…',
  }[stage];

  return (
    <div className="space-y-5 rounded-xl border bg-card/70 p-5">
      <div>
        <h2 className="text-lg font-semibold">Refresh how a character responds to you</h2>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Mirid reviews this character’s instructions against the history and memories you allow, then proposes a more consistent way for them to respond to you. You review the result before anything is saved.
        </p>
      </div>

      {!activeProfileId ? (
        <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-800 dark:text-amber-200">
          Choose a user profile first. Memories are stored separately for each profile.
        </p>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label>Character to refresh</Label>
          <Select value={characterId} onValueChange={setCharacterId}>
            <SelectTrigger><SelectValue placeholder="Choose a character" /></SelectTrigger>
            <SelectContent>
              {selectableCharacters.map((character) => (
                <SelectItem key={character.id} value={character.id}>{character.name || character.id}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="rounded-lg border bg-muted/15 p-3 text-sm">
          <p className="font-medium">Model used for the review</p>
          <p className="mt-1 text-muted-foreground">
            {canRunModel ? requestRoute.effectiveModel : 'No text model selected'}
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm">
          <Checkbox checked={includeTranscriptActive} onCheckedChange={(value) => setIncludeTranscriptActive(Boolean(value))} className="mt-0.5" />
          <span><span className="block font-medium">Use the current chat</span><span className="mt-1 block text-xs text-muted-foreground">Lets the review learn from your recent interaction with this character.</span></span>
        </label>
        <label className="flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm">
          <Checkbox checked={includeBackendMemories} onCheckedChange={(value) => setIncludeBackendMemories(Boolean(value))} className="mt-0.5" />
          <span><span className="block font-medium">Use memories about me</span><span className="mt-1 block text-xs text-muted-foreground">Includes durable facts attached to your current user profile.</span></span>
        </label>
        <label className="flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm">
          <Checkbox checked={includeCharacterMemories} onCheckedChange={(value) => setIncludeCharacterMemories(Boolean(value))} className="mt-0.5" />
          <span><span className="block font-medium">Use this character’s memories</span><span className="mt-1 block text-xs text-muted-foreground">Includes what this character has learned separately from its own chats with you.</span></span>
        </label>
        <label className="flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm">
          <Checkbox checked={alsoRewriteUserProfile} onCheckedChange={(value) => setAlsoRewriteUserProfile(Boolean(value))} className="mt-0.5" />
          <span><span className="block font-medium">Also propose cleaner profile memories</span><span className="mt-1 block text-xs text-muted-foreground">Optional. Saving that list remains a separate confirmed action.</span></span>
        </label>
      </div>

      <div className="space-y-2">
        <Label htmlFor="realignment-notes">What should improve? <span className="font-normal text-muted-foreground">Optional</span></Label>
        <Textarea
          id="realignment-notes"
          value={extraNotes}
          onChange={(event) => setExtraNotes(event.target.value)}
          placeholder="For example: shorter replies, stronger continuity, fewer repeated questions, clearer boundaries."
          className="min-h-[72px]"
        />
      </div>

      <details className="group rounded-lg border bg-muted/10">
        <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium hover:bg-muted/30">
          <ChevronRight className="h-4 w-4 transition-transform group-open:rotate-90" />
          More context options
        </summary>
        <div className="space-y-4 border-t px-3 py-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Include another chat</Label>
              <Select value={extraConvId || '__none__'} onValueChange={(value) => setExtraConvId(value === '__none__' ? '' : value)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">No additional chat</SelectItem>
                  {(conversations || []).map((conversation) => (
                    <SelectItem key={conversation.id} value={conversation.id}>
                      {(conversation.title || 'Chat').slice(0, 48)}{conversation.id === activeConversation ? ' · current' : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Transcript limit</Label>
              <Input type="number" min={5000} max={2000000} value={transcriptMaxChars} onChange={(event) => setTranscriptMaxChars(Number(event.target.value))} />
            </div>
          </div>

          <div className="flex flex-wrap gap-4 text-sm">
            <label className="flex cursor-pointer items-center gap-2"><Checkbox checked={includeRollingActive} onCheckedChange={(value) => setIncludeRollingActive(Boolean(value))} />Use current continuity summary</label>
            <label className="flex cursor-pointer items-center gap-2"><Checkbox checked={includeRollingExtra} onCheckedChange={(value) => setIncludeRollingExtra(Boolean(value))} disabled={!extraConvId} />Use additional chat summary</label>
          </div>

          {includeCharacterMemories ? (
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Character-memory selection</Label>
                <Select value={agenticMode} onValueChange={setAgenticMode}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ranked">Most important memories first</SelectItem>
                    <SelectItem value="rag">Find memories matching a topic</SelectItem>
                    <SelectItem value="full">Include as much as fits</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Character-memory limit</Label>
                <Input type="number" min={4000} max={500000} value={agenticMaxChars} onChange={(event) => setAgenticMaxChars(Number(event.target.value))} />
              </div>
              {agenticMode === 'rag' ? (
                <div className="space-y-2 md:col-span-2">
                  <Label>Topic to look for</Label>
                  <Input value={ragQuery} onChange={(event) => setRagQuery(event.target.value)} placeholder="For example: tone, boundaries or recurring preferences" />
                </div>
              ) : null}
            </div>
          ) : null}

          {alsoRewriteUserProfile ? (
            <div className="space-y-2 max-w-md">
              <Label>Profile-memory approach</Label>
              <Select value={profileRewriteMode} onValueChange={setProfileRewriteMode}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="merge">Merge duplicates and preserve distinct facts</SelectItem>
                  <SelectItem value="from_scratch">Build a fresh list from the selected context</SelectItem>
                </SelectContent>
              </Select>
            </div>
          ) : null}

          <details className="rounded-md border px-3 py-2 text-xs">
            <summary className="cursor-pointer font-medium">Current instructions sent as the baseline</summary>
            <Textarea value={currentInstructions} onChange={(event) => setCurrentInstructions(event.target.value)} className="mt-3 min-h-[140px] font-mono text-xs" spellCheck={false} />
          </details>
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" onClick={runAutomatedReview} disabled={Boolean(stage) || !apiReady || !activeProfileId || !selectedCharacter || !canRunModel}>
          {stage && stage !== 'saving-profile' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
          Review this character
        </Button>
        {stageText ? <span className="text-sm text-muted-foreground">{stageText}</span> : null}
      </div>

      {error ? <p className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</p> : null}
      {success ? <p className="flex items-center gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-800 dark:text-emerald-200"><CheckCircle2 className="h-4 w-4" />{success}</p> : null}

      {parsedResult ? (
        <div className="space-y-4 rounded-xl border border-primary/25 bg-primary/5 p-4">
          <div>
            <h3 className="font-semibold">Proposed character update</h3>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">Read and edit this before saving. It replaces only this character’s Model Instructions; the character card, chats and memories remain intact.</p>
          </div>

          {Array.isArray(parsedResult.delta_vs_current_instructions) && parsedResult.delta_vs_current_instructions.length ? (
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {parsedResult.delta_vs_current_instructions.slice(0, 8).map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}
            </ul>
          ) : null}

          <Textarea
            aria-label="Proposed model instructions"
            value={proposedInstructions}
            onChange={(event) => setProposedInstructions(event.target.value)}
            className="min-h-[180px]"
            placeholder="The model did not return an apply-ready Model Instructions block. You can inspect its full response under Run elsewhere below."
          />
          <Button type="button" onClick={saveCharacterUpdate} disabled={!proposedInstructions.trim()}>
            <CheckCircle2 className="mr-2 h-4 w-4" />Save to {selectedCharacter?.name || 'character'}
          </Button>

          {Array.isArray(parsedResult.revised_user_profile_memories) && parsedResult.revised_user_profile_memories.length ? (
            <div className="space-y-2 border-t pt-4">
              <p className="text-sm font-medium">Proposed profile-memory list: {parsedResult.revised_user_profile_memories.length} entries</p>
              <p className="text-xs text-muted-foreground">This replaces the current profile-memory list only after a separate confirmation.</p>
              <Button type="button" variant="outline" onClick={applyProposedProfile} disabled={stage === 'saving-profile'}>
                {stage === 'saving-profile' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <CheckCircle2 className="mr-2 h-4 w-4" />}
                Save reviewed profile memories
              </Button>
            </div>
          ) : null}

          {parsedResult.revised_character_instructions ? (
            <details className="rounded-md border bg-background/60 px-3 py-2 text-xs">
              <summary className="cursor-pointer font-medium">Full rewritten character prompt</summary>
              <Textarea readOnly value={String(parsedResult.revised_character_instructions)} className="mt-3 min-h-[160px] font-mono text-xs" spellCheck={false} />
            </details>
          ) : null}
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
            <Button type="button" size="sm" variant="outline" onClick={buildForAnotherModel} disabled={Boolean(stage) || !apiReady || !activeProfileId || !selectedCharacter}>Build prompt</Button>
            <Button type="button" size="sm" variant="ghost" onClick={copyPrompt} disabled={!promptPack?.combined}><Copy className="mr-2 h-4 w-4" />Copy prompt</Button>
          </div>
          {promptPack?.combined ? <Textarea readOnly value={promptPack.combined.slice(0, 24000)} className="min-h-[140px] font-mono text-xs" spellCheck={false} /> : null}
          <Textarea value={rawResponse} onChange={(event) => setRawResponse(event.target.value)} className="min-h-[120px] font-mono text-xs" placeholder="Paste a model response here" spellCheck={false} />
          <Button type="button" size="sm" variant="secondary" onClick={parseEditedResponse} disabled={Boolean(stage) || !rawResponse.trim()}>Read pasted response</Button>
        </div>
      </details>
    </div>
  );
}
