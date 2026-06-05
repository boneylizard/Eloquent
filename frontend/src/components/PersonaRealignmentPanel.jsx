import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import { fetchWithTimeout } from '../config/api';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Loader2, Copy, Sparkles, FileJson, CheckCircle2, ChevronRight } from 'lucide-react';

const ACK_STORAGE_KEY = 'eloquent:personaRealignmentAckV1';

function loadAck() {
  try {
    return localStorage.getItem(ACK_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function saveAck() {
  try {
    localStorage.setItem(ACK_STORAGE_KEY, '1');
  } catch {
    /* ignore */
  }
}

function speakerLabel(msg) {
  const r = msg?.role;
  if (r === 'user') return 'User';
  if (r === 'assistant' || r === 'bot') return 'Assistant';
  return r ? String(r) : 'Message';
}

function transcriptFromMessages(messages, maxChars) {
  if (!Array.isArray(messages) || !messages.length) return '';
  const parts = [];
  let total = 0;
  for (const m of messages) {
    const body = String(m?.content ?? '').trim();
    if (!body) continue;
    const line = `${speakerLabel(m)}: ${body}\n\n`;
    if (total + line.length > maxChars) break;
    parts.push(line);
    total += line.length;
  }
  return parts.join('').trim();
}

function toCharacterCard(character) {
  if (!character || typeof character !== 'object') return null;
  const pick = (k) => (character[k] != null ? String(character[k]).trim() : '');
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

/** One numbered block in the tutorial layout */
function TutorialStep({ step, title, subtitle, children }) {
  return (
    <section className="rounded-xl border border-border/80 bg-card/60 shadow-sm overflow-hidden">
      <header className="flex gap-3 items-start px-4 py-3 border-b border-border/60 bg-muted/25">
        <span className="flex h-9 min-w-[2.25rem] shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold tabular-nums">
          {step}
        </span>
        <div className="min-w-0 pt-0.5">
          <h3 className="text-sm font-semibold leading-snug">{title}</h3>
          {subtitle ? <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{subtitle}</p> : null}
        </div>
      </header>
      <div className="p-4 space-y-3">{children}</div>
    </section>
  );
}

/**
 * Settings → Memory Browser → Persona realign (beta).
 * Tutorial-style UI for persona realignment prompt pack + optional results handling.
 */
export default function PersonaRealignmentPanel() {
  const { activeProfileId } = useMemory();
  const {
    MEMORY_API_URL,
    portsReady,
    storageHydrated,
    characters = [],
    conversations = [],
    activeConversation,
    userProfile,
    buildSystemPrompt,
  } = useApp();

  const apiReady = portsReady && storageHydrated;
  const apiUrl = MEMORY_API_URL;

  const [acknowledged, setAcknowledged] = useState(loadAck);
  const [characterId, setCharacterId] = useState('');
  const [includeTranscriptActive, setIncludeTranscriptActive] = useState(true);
  const [transcriptMaxChars, setTranscriptMaxChars] = useState(120000);
  const [extraConvId, setExtraConvId] = useState('');
  const [includeRollingActive, setIncludeRollingActive] = useState(true);
  const [includeRollingExtra, setIncludeRollingExtra] = useState(false);
  const [agenticMode, setAgenticMode] = useState('ranked');
  const [agenticMaxChars, setAgenticMaxChars] = useState(48000);
  const [ragQuery, setRagQuery] = useState('');
  const [includeBackendMemories, setIncludeBackendMemories] = useState(true);
  const [extraNotes, setExtraNotes] = useState('');
  const [currentInstructions, setCurrentInstructions] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [alsoRewriteUserProfile, setAlsoRewriteUserProfile] = useState(false);
  const [profileRewriteMode, setProfileRewriteMode] = useState('merge');
  const [realignPaste, setRealignPaste] = useState('');
  const [realignParsed, setRealignParsed] = useState(null);
  const [realignParseError, setRealignParseError] = useState(null);
  const [applyProfileBusy, setApplyProfileBusy] = useState(false);

  const selectableCharacters = useMemo(
    () => (characters || []).filter((c) => c?.id && c.chat_role !== 'user'),
    [characters]
  );

  useEffect(() => {
    if (!characterId && selectableCharacters.length) {
      setCharacterId(selectableCharacters[0].id);
    }
  }, [characterId, selectableCharacters]);

  const selectedCharacter = useMemo(
    () => (characters || []).find((c) => c.id === characterId),
    [characters, characterId]
  );

  useEffect(() => {
    if (!selectedCharacter) return;
    try {
      setCurrentInstructions(buildSystemPrompt(selectedCharacter) || '');
    } catch {
      setCurrentInstructions(selectedCharacter.model_instructions || '');
    }
  }, [selectedCharacter, buildSystemPrompt]);

  const activeConv = useMemo(
    () => (conversations || []).find((c) => c.id === activeConversation),
    [conversations, activeConversation]
  );

  const extraConv = useMemo(
    () => (conversations || []).find((c) => c.id === extraConvId),
    [conversations, extraConvId]
  );

  const handleAcknowledge = () => {
    saveAck();
    setAcknowledged(true);
  };

  const buildPayload = useCallback(() => {
    const transcripts = [];
    if (includeTranscriptActive && activeConv?.messages?.length) {
      const t = transcriptFromMessages(activeConv.messages, Number(transcriptMaxChars) || 120000);
      if (t) transcripts.push(t);
    }
    if (extraConvId && extraConv?.messages?.length) {
      const t2 = transcriptFromMessages(extraConv.messages, Math.floor((Number(transcriptMaxChars) || 120000) / 2));
      if (t2) transcripts.push(t2);
    }

    const rollingPacks = [];
    if (includeRollingActive && activeConv?.rollingMemoryPack?.trim()) {
      rollingPacks.push(activeConv.rollingMemoryPack.trim());
    }
    if (includeRollingExtra && extraConv?.rollingMemoryPack?.trim()) {
      rollingPacks.push(extraConv.rollingMemoryPack.trim());
    }

    const card = toCharacterCard(selectedCharacter);
    const displayName =
      (userProfile?.name || userProfile?.username || userProfile?.displayName || '').trim() || undefined;

    return {
      user_id: activeProfileId,
      character_id: characterId,
      character_name: selectedCharacter?.name || undefined,
      user_display_name: displayName,
      character_card: card,
      current_character_instructions: currentInstructions,
      rolling_packs: rollingPacks.length ? rollingPacks : undefined,
      transcripts: transcripts.length ? transcripts : undefined,
      include_backend_memories: includeBackendMemories,
      agentic_mode: agenticMode,
      agentic_max_chars: Math.min(500000, Math.max(4000, Number(agenticMaxChars) || 48000)),
      agentic_rag_query: agenticMode === 'rag' && ragQuery.trim() ? ragQuery.trim() : undefined,
      extra_notes: extraNotes.trim() || undefined,
      also_rewrite_user_profile: alsoRewriteUserProfile,
      user_profile_rewrite_mode: profileRewriteMode === 'from_scratch' ? 'from_scratch' : 'merge',
    };
  }, [
    activeProfileId,
    characterId,
    selectedCharacter,
    userProfile,
    currentInstructions,
    includeTranscriptActive,
    transcriptMaxChars,
    activeConv,
    extraConv,
    extraConvId,
    includeRollingActive,
    includeRollingExtra,
    includeBackendMemories,
    agenticMode,
    agenticMaxChars,
    ragQuery,
    extraNotes,
    alsoRewriteUserProfile,
    profileRewriteMode,
  ]);

  const runBuildPack = async () => {
    if (!activeProfileId || !characterId) {
      setError('Select an active profile and character.');
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const body = buildPayload();
      const res = await fetchWithTimeout(
        `${apiUrl}/memory/persona_realignment/prompt_pack`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
        120000
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      }
      if (data.status !== 'success' || !data.combined) {
        throw new Error(data.detail || 'Unexpected response from prompt pack');
      }
      setResult(data);
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const copyCombined = async () => {
    if (!result?.combined) return;
    try {
      await navigator.clipboard.writeText(result.combined);
    } catch {
      /* ignore */
    }
  };

  const copyCharacterInstructions = async () => {
    const t = realignParsed?.revised_character_instructions;
    if (!t) return;
    try {
      await navigator.clipboard.writeText(String(t));
    } catch {
      /* ignore */
    }
  };

  const parseRealignResponse = async () => {
    setRealignParseError(null);
    setRealignParsed(null);
    if (!realignPaste.trim()) {
      setRealignParseError('Paste the model JSON first.');
      return;
    }
    try {
      const res = await fetchWithTimeout(
        `${apiUrl}/memory/persona_realignment/parse_response`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ raw_text: realignPaste }),
        },
        60000
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setRealignParsed(data);
    } catch (e) {
      setRealignParseError(e?.message || String(e));
    }
  };

  const applyProposedUserProfile = async () => {
    const memories = realignParsed?.revised_user_profile_memories;
    if (!activeProfileId || !Array.isArray(memories) || memories.length === 0) {
      alert('No revised_user_profile_memories to apply. Parse a response that includes them.');
      return;
    }
    if (
      !window.confirm(
        `Replace ALL backend profile memories for this user with ${memories.length} proposed rows? This cannot be undone automatically.`
      )
    ) {
      return;
    }
    setApplyProfileBusy(true);
    setError(null);
    try {
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
      alert(`Saved ${data.saved ?? memories.length} profile memories.`);
      setRealignPaste('');
      setRealignParsed(null);
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setApplyProfileBusy(false);
    }
  };

  if (!acknowledged) {
    return (
      <Alert className="border-amber-500/40 bg-amber-50/50 dark:bg-amber-950/20">
        <Sparkles className="h-4 w-4 shrink-0" />
        <AlertTitle className="text-base">Persona realignment — quick orientation</AlertTitle>
        <AlertDescription className="space-y-4 mt-3 text-sm leading-relaxed">
          <p>
            This helps you refresh how a <strong>character</strong> should behave <strong>for you</strong>, using your memories,
            chats, and notes — packed into <em>one big prompt</em> you send to a strong model.
          </p>
          <p className="text-muted-foreground">
            Nothing is saved automatically. You copy the model&apos;s answer and paste only what you agree with (character text into the editor,
            optional profile list via the button at the end).
          </p>
          <ol className="list-decimal pl-5 space-y-2 text-muted-foreground border border-border/50 rounded-lg py-3 px-2 bg-background/50">
            <li>Pick the character and tweak their current instructions if needed.</li>
            <li>Choose what data to attach (defaults are usually fine).</li>
            <li>Build prompt → copy → paste into your best model → ask for JSON only.</li>
            <li>Paste the reply back here to extract text and optionally apply profile memories.</li>
          </ol>
          <p className="text-xs text-muted-foreground">Uses extra tokens on purpose — run it rarely, when a big refresh is worth it.</p>
          <Button type="button" size="sm" variant="default" onClick={handleAcknowledge}>
            Continue to the steps
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="rounded-xl border border-border/70 bg-gradient-to-br from-muted/40 to-transparent px-4 py-3">
        <p className="text-sm font-medium flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary shrink-0" />
          How this tutorial is laid out
        </p>
        <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
          Follow the numbered steps in order. Advanced knobs stay tucked away so you are not stuck reading jargon on day one.
        </p>
      </div>

      {!apiReady && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground rounded-lg border border-dashed px-3 py-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Connecting…
        </div>
      )}

      {!activeProfileId && (
        <p className="text-sm text-amber-800 dark:text-amber-300 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2">
          Choose a <strong>user profile</strong> in the profile selector first — this ties memories to the right person.
        </p>
      )}

      <TutorialStep
        step={1}
        title="Pick the character you’re realigning"
        subtitle="This is whose instructions we’re improving — the snapshot below loads from your character card."
      >
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label>Character</Label>
            <Select value={characterId} onValueChange={setCharacterId}>
              <SelectTrigger>
                <SelectValue placeholder="Choose character" />
              </SelectTrigger>
              <SelectContent>
                {selectableCharacters.map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {c.name || c.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Optional: second chat</Label>
            <Select value={extraConvId || '__none__'} onValueChange={(v) => setExtraConvId(v === '__none__' ? '' : v)}>
              <SelectTrigger>
                <SelectValue placeholder="None" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">None — only use my active chat</SelectItem>
                {(conversations || []).map((c) => (
                  <SelectItem key={c.id} value={c.id}>
                    {(c.title || 'Chat').slice(0, 48)} {c.id === activeConversation ? '(active)' : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">Adds that chat&apos;s transcript and (if you enable it below) its rolling summary.</p>
          </div>
        </div>

        <div className="space-y-2">
          <Label>Starting instructions (you can edit before building)</Label>
          <Textarea
            value={currentInstructions}
            onChange={(e) => setCurrentInstructions(e.target.value)}
            className="min-h-[120px] text-sm font-mono"
            spellCheck={false}
          />
          <p className="text-xs text-muted-foreground">
            The model sees this as the baseline to improve — tweak here if something is already wrong.
          </p>
        </div>
      </TutorialStep>

      <TutorialStep
        step={2}
        title="Choose what context to pack in"
        subtitle="Defaults include profile memories, this character’s long-term memories about you, your active chat, and continuity notes. Uncheck anything you want to skip."
      >
        <div className="rounded-lg border border-border/50 bg-muted/15 p-3 space-y-3">
          <div className="flex flex-wrap gap-x-6 gap-y-2">
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <Checkbox checked={includeBackendMemories} onCheckedChange={setIncludeBackendMemories} />
              Saved profile memories (about you)
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <Checkbox checked={includeTranscriptActive} onCheckedChange={setIncludeTranscriptActive} />
              Active chat transcript
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <Checkbox checked={includeRollingActive} onCheckedChange={setIncludeRollingActive} />
              Active rolling memory summary
            </label>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <Checkbox checked={includeRollingExtra} onCheckedChange={setIncludeRollingExtra} disabled={!extraConvId} />
              Second chat rolling summary
            </label>
          </div>
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <Label className="text-xs">Max characters from transcripts</Label>
              <Input
                type="number"
                min={5000}
                max={2000000}
                value={transcriptMaxChars}
                onChange={(e) => setTranscriptMaxChars(Number(e.target.value))}
                className="h-9 w-36"
              />
            </div>
          </div>
        </div>

        <details className="group rounded-lg border border-border/60 bg-muted/10">
          <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted/30 rounded-lg">
            <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" />
            Advanced: how much “character memory” (agentic) to send
          </summary>
          <div className="px-3 pb-4 pt-1 space-y-3 border-t border-border/40">
            <p className="text-xs text-muted-foreground leading-relaxed">
              If you don&apos;t know what this means, leave <strong>ranked</strong> and the default size. Raise the limit only if this character&apos;s
              memory file is huge and you need more of it in one shot.
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-xs">Mode</Label>
                <Select value={agenticMode} onValueChange={setAgenticMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ranked">Ranked (recommended)</SelectItem>
                    <SelectItem value="rag">Search-like pick (needs embeddings on server)</SelectItem>
                    <SelectItem value="full">Pack as much as fits</SelectItem>
                    <SelectItem value="none">Don&apos;t send character memory file</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Character memory budget (characters)</Label>
                <Input
                  type="number"
                  min={4000}
                  max={500000}
                  value={agenticMaxChars}
                  onChange={(e) => setAgenticMaxChars(Number(e.target.value))}
                  className="h-9"
                />
              </div>
            </div>
            {agenticMode === 'rag' && (
              <div className="space-y-2">
                <Label className="text-xs">What to search for (optional)</Label>
                <Input
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  placeholder="e.g. tone, boundaries, things I’ve asked for before…"
                />
              </div>
            )}
          </div>
        </details>

        <details className="group rounded-lg border border-border/60 bg-muted/10">
          <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-foreground hover:bg-muted/30 rounded-lg">
            <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" />
            Optional: also propose a fresh list of profile memories (same model reply)
          </summary>
          <div className="px-3 pb-4 pt-1 space-y-3 border-t border-border/40">
            <p className="text-xs text-muted-foreground leading-relaxed">
              Same JSON answer can include a cleaned-up <strong>user profile memory list</strong>. You still decide later whether to save it (Step 5).
              Turn off &quot;Saved profile memories&quot; above if the bullet list is huge but you still want indexed rows only — saves tokens.
            </p>
            <label className="flex items-start gap-2 text-sm cursor-pointer">
              <Checkbox
                checked={alsoRewriteUserProfile}
                onCheckedChange={(v) => setAlsoRewriteUserProfile(!!v)}
                className="mt-0.5"
              />
              <span>Ask for a rewritten profile memory list in the same JSON</span>
            </label>
            {alsoRewriteUserProfile && (
              <div className="space-y-2 max-w-md">
                <Label className="text-xs">Style</Label>
                <Select value={profileRewriteMode} onValueChange={setProfileRewriteMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="merge">Merge & dedupe what I already have</SelectItem>
                    <SelectItem value="from_scratch">Rebuild from everything you’re sending</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </details>

        <div className="space-y-2">
          <Label>Anything else the model should know for this run?</Label>
          <Textarea
            value={extraNotes}
            onChange={(e) => setExtraNotes(e.target.value)}
            className="min-h-[64px] text-sm"
            placeholder="Optional — e.g. focus on shorter replies, or stress-test boundaries…"
          />
        </div>
      </TutorialStep>

      <TutorialStep
        step={3}
        title="Build the prompt and copy it"
        subtitle="Creates one long message ready for your smartest model. Use “Copy” — the preview below may be truncated."
      >
        <div className="flex flex-wrap gap-2 items-center">
          <Button type="button" onClick={runBuildPack} disabled={busy || !apiReady || !activeProfileId || !characterId}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Sparkles className="h-4 w-4 mr-2" />}
            Build prompt pack
          </Button>
          {result?.combined && (
            <Button type="button" variant="outline" size="sm" onClick={copyCombined}>
              <Copy className="h-4 w-4 mr-2" />
              Copy full prompt
            </Button>
          )}
        </div>

        {error && (
          <div className="text-sm text-red-700 dark:text-red-400 bg-red-500/10 rounded-md p-3 border border-red-500/20">
            {error}
          </div>
        )}

        {result?.stats && (
          <p className="text-xs text-muted-foreground">
            Rough size ~{result.stats.bundle_chars?.toLocaleString?.() ?? result.stats.bundle_chars} characters · character-memory rows included:{' '}
            {result.stats.agentic_insight_count} · profile memories rows: {result.stats.backend_memory_count}
            {result.stats.also_rewrite_user_profile ? (
              <> · also asking for profile rewrite ({result.stats.user_profile_rewrite_mode || 'merge'})</>
            ) : null}
          </p>
        )}

        {result?.combined && (
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">Preview (first chunk only)</Label>
            <Textarea readOnly value={result.combined.slice(0, 24000)} className="min-h-[160px] text-xs font-mono" spellCheck={false} />
            {result.combined.length > 24000 && (
              <p className="text-xs text-amber-700 dark:text-amber-400">Preview is cut off — always use Copy for the full prompt.</p>
            )}
          </div>
        )}
      </TutorialStep>

      <TutorialStep
        step={4}
        title="Run it in your model"
        subtitle="Paste the copied prompt into the model you trust for hard reasoning. Ask it to answer with JSON only (no chat around it)."
      >
        <ul className="text-xs text-muted-foreground space-y-2 list-disc pl-5 leading-relaxed">
          <li>Use your best / largest context model if you can — this is a one-off quality pass.</li>
          <li>If you route through this app&apos;s chat, using something like “testing / judge” mode avoids extra memory noise on that request.</li>
        </ul>
        <details className="text-xs border border-border/50 rounded-md px-3 py-2 bg-muted/20">
          <summary className="cursor-pointer font-medium text-foreground">Technical detail</summary>
          <p className="mt-2 text-muted-foreground">
            Backend route is <code className="text-[11px] px-1 rounded bg-muted">/memory/persona_realignment/prompt_pack</code>.
            For local /generate, <code className="text-[11px] px-1 rounded bg-muted">request_purpose: model_testing</code> skips extra retrieval on that turn.
          </p>
        </details>
      </TutorialStep>

      <TutorialStep
        step={5}
        title="Paste the reply back and use what you like"
        subtitle="Nothing saves until you say so. Character instructions are copy-only; replacing saved profile memories is a separate confirm button."
      >
        <Textarea
          value={realignPaste}
          onChange={(e) => setRealignPaste(e.target.value)}
          placeholder="Paste the model’s entire JSON reply here…"
          className="min-h-[100px] text-xs font-mono"
          spellCheck={false}
        />
        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" variant="secondary" onClick={parseRealignResponse} disabled={!realignPaste.trim()}>
            <FileJson className="h-4 w-4 mr-2" />
            Parse JSON
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={applyProposedUserProfile}
            disabled={applyProfileBusy || !realignParsed?.revised_user_profile_memories?.length}
          >
            {applyProfileBusy ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <CheckCircle2 className="h-4 w-4 mr-2" />
            )}
            Apply proposed profile memories (destructive)
          </Button>
        </div>
        {realignParseError && <p className="text-xs text-red-600">{realignParseError}</p>}
        {realignParsed?.status === 'success' && (
          <div className="space-y-3 text-sm border border-border/50 rounded-lg p-3 bg-muted/10">
            {realignParsed.user_profile_rewrite_summary && (
              <p className="text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Profile changes summary:</span> {realignParsed.user_profile_rewrite_summary}
              </p>
            )}
            {realignParsed.revised_character_instructions != null && realignParsed.revised_character_instructions !== '' && (
              <div className="space-y-1">
                <div className="flex flex-wrap gap-2 items-center justify-between">
                  <Label className="text-xs font-semibold">New character instructions — paste into Character editor</Label>
                  <Button type="button" size="sm" variant="ghost" className="h-7 text-xs" onClick={copyCharacterInstructions}>
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </Button>
                </div>
                <Textarea
                  readOnly
                  value={String(realignParsed.revised_character_instructions)}
                  className="min-h-[120px] text-xs font-mono"
                  spellCheck={false}
                />
              </div>
            )}
            {realignParsed.revised_model_instructions != null && String(realignParsed.revised_model_instructions).trim() !== '' && (
              <div className="space-y-1">
                <Label className="text-xs">Extra model-style instructions (if present)</Label>
                <Textarea
                  readOnly
                  value={String(realignParsed.revised_model_instructions)}
                  className="min-h-[64px] text-xs font-mono"
                  spellCheck={false}
                />
              </div>
            )}
            <p className="text-xs">
              Proposed profile memory rows: <strong>{realignParsed.revised_user_profile_memories?.length ?? 0}</strong>
              {!realignParsed.has_user_profile_memories ? ' (none in this JSON)' : ''}
            </p>
            {(realignParsed.revised_user_profile_memories?.length ?? 0) > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-muted-foreground">Peek at first rows</summary>
                <pre className="mt-2 max-h-36 overflow-auto rounded border bg-background/80 p-2 whitespace-pre-wrap">
                  {JSON.stringify(realignParsed.revised_user_profile_memories.slice(0, 8), null, 2)}
                </pre>
              </details>
            )}
          </div>
        )}
      </TutorialStep>
    </div>
  );
}
