import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import { fetchWithTimeout } from '../config/api';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
import { Checkbox } from './ui/checkbox';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Loader2, Copy, Sparkles, FileJson, CheckCircle2, ChevronDown, ChevronUp, Send, X, RotateCcw } from 'lucide-react';

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

/**
 * Chat-embedded persona realignment panel.
 * Same functionality as the Settings panel but designed for inline use in chat.
 * Props:
 *   onClose()        — close the panel
 *   onSendToChat(text) — send the built prompt as the next chat message
 *   onCharacterUpdated() — called after auto-character-creator finishes
 */
export default function ChatPersonaRealignmentPanel({ onClose, onSendToChat, onCharacterUpdated }) {
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
    isGenerating,
  } = useApp();

  const apiReady = portsReady && storageHydrated;
  const apiUrl = MEMORY_API_URL;

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
  const [showAdvanced, setShowAdvanced] = useState(false);

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

  const buildPayload = useCallback(() => {
    const transcripts = [];
    if (includeTranscriptActive && activeConv?.messages?.length) {
      const t = transcriptFromMessages(activeConv.messages, Number(transcriptMaxChars) || 120000);
      if (t) {
        const title = activeConv.title || 'Active Chat';
        transcripts.push(`[Transcript: ${title}]\n\n${t}`);
      }
    }
    if (extraConvId && extraConv?.messages?.length) {
      const t2 = transcriptFromMessages(extraConv.messages, Math.floor((Number(transcriptMaxChars) || 120000) / 2));
      if (t2) {
        const title = extraConv.title || 'Additional Chat';
        transcripts.push(`[Transcript: ${title}]\n\n${t2}`);
      }
    }

    const rollingPacks = [];
    if (includeRollingActive && activeConv?.rollingMemoryPack?.trim()) {
      const title = activeConv.title || 'Active Chat';
      rollingPacks.push(`[Rolling Pack: ${title}]\n\n${activeConv.rollingMemoryPack.trim()}`);
    }
    if (includeRollingExtra && extraConv?.rollingMemoryPack?.trim()) {
      const title = extraConv.title || 'Additional Chat';
      rollingPacks.push(`[Rolling Pack: ${title}]\n\n${extraConv.rollingMemoryPack.trim()}`);
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
    activeProfileId, characterId, selectedCharacter, userProfile, currentInstructions,
    includeTranscriptActive, transcriptMaxChars, activeConv, extraConv, extraConvId,
    includeRollingActive, includeRollingExtra, includeBackendMemories, agenticMode,
    agenticMaxChars, ragQuery, extraNotes, alsoRewriteUserProfile, profileRewriteMode,
  ]);

  const handleBuildAndSend = async () => {
    if (!activeProfileId || !characterId) {
      setError('Select a profile and character.');
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

  const handleSendToChat = () => {
    if (!result?.combined) return;
    onSendToChat?.(result.combined);
  };

  const handleCopy = async () => {
    if (!result?.combined) return;
    try {
      await navigator.clipboard.writeText(result.combined);
    } catch { /* ignore */ }
  };

  const handleBuildAgain = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="rounded-xl border border-border/70 bg-card shadow-md overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 bg-muted/30">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary shrink-0" />
          <span className="text-sm font-medium">Persona Realignment</span>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClose}
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="p-3 space-y-3 max-h-[60vh] overflow-y-auto">
        {/* Character selector */}
        <div className="space-y-1.5">
          <Label className="text-xs">Character</Label>
          <Select value={characterId} onValueChange={setCharacterId}>
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="Choose character" />
            </SelectTrigger>
            <SelectContent>
              {selectableCharacters.map((c) => (
                <SelectItem key={c.id} value={c.id} className="text-xs">
                  {c.name || c.id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Second chat selector */}
        <div className="space-y-1.5">
          <Label className="text-xs">Second chat (optional)</Label>
          <Select value={extraConvId || '__none__'} onValueChange={(v) => setExtraConvId(v === '__none__' ? '' : v)}>
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="None" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__none__" className="text-xs">None — active chat only</SelectItem>
              {(conversations || []).map((c) => (
                <SelectItem key={c.id} value={c.id} className="text-xs">
                  {(c.title || 'Chat').slice(0, 40)}{c.id === activeConversation ? ' (active)' : ''}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Current instructions */}
        <div className="space-y-1.5">
          <Label className="text-xs">Current instructions (editable)</Label>
          <Textarea
            value={currentInstructions}
            onChange={(e) => setCurrentInstructions(e.target.value)}
            className="min-h-[80px] text-xs font-mono resize-none"
            spellCheck={false}
          />
        </div>

        {/* Quick toggles */}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <Checkbox checked={includeBackendMemories} onCheckedChange={setIncludeBackendMemories} />
            Profile memories
          </label>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <Checkbox checked={includeTranscriptActive} onCheckedChange={setIncludeTranscriptActive} />
            Active transcript
          </label>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <Checkbox checked={includeRollingActive} onCheckedChange={setIncludeRollingActive} />
            Rolling summary
          </label>
          {extraConvId && (
            <label className="flex items-center gap-1.5 text-xs cursor-pointer">
              <Checkbox checked={includeRollingExtra} onCheckedChange={setIncludeRollingExtra} />
              2nd chat summary
            </label>
          )}
        </div>

        {/* Transcript max chars */}
        <div className="space-y-1">
          <Label className="text-xs">Max transcript chars</Label>
          <Input
            type="number"
            min={5000}
            max={2000000}
            value={transcriptMaxChars}
            onChange={(e) => setTranscriptMaxChars(Number(e.target.value))}
            className="h-8 w-32 text-xs"
          />
        </div>

        {/* Advanced toggle */}
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          Advanced options
        </button>

        {showAdvanced && (
          <div className="space-y-3 pl-1 border-l-2 border-border/40">
            <div className="grid gap-2 md:grid-cols-2">
              <div className="space-y-1">
                <Label className="text-xs">Agentic mode</Label>
                <Select value={agenticMode} onValueChange={setAgenticMode}>
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ranked" className="text-xs">Ranked</SelectItem>
                    <SelectItem value="rag" className="text-xs">Search-like</SelectItem>
                    <SelectItem value="full" className="text-xs">Pack as much</SelectItem>
                    <SelectItem value="none" className="text-xs">Skip</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Memory budget (chars)</Label>
                <Input
                  type="number"
                  min={4000}
                  max={500000}
                  value={agenticMaxChars}
                  onChange={(e) => setAgenticMaxChars(Number(e.target.value))}
                  className="h-8 text-xs"
                />
              </div>
            </div>
            {agenticMode === 'rag' && (
              <div className="space-y-1">
                <Label className="text-xs">Search query</Label>
                <Input
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  placeholder="e.g. tone, boundaries, things I've asked for"
                  className="h-8 text-xs"
                />
              </div>
            )}
            <label className="flex items-center gap-1.5 text-xs cursor-pointer">
              <Checkbox
                checked={alsoRewriteUserProfile}
                onCheckedChange={(v) => setAlsoRewriteUserProfile(!!v)}
              />
              Also propose profile memory rewrite
            </label>
            {alsoRewriteUserProfile && (
              <Select value={profileRewriteMode} onValueChange={setProfileRewriteMode}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="merge" className="text-xs">Merge &amp; dedupe</SelectItem>
                  <SelectItem value="from_scratch" className="text-xs">Rebuild from scratch</SelectItem>
                </SelectContent>
              </Select>
            )}
            <div className="space-y-1">
              <Label className="text-xs">Extra notes for the model</Label>
              <Textarea
                value={extraNotes}
                onChange={(e) => setExtraNotes(e.target.value)}
                className="min-h-[48px] text-xs resize-none"
                placeholder="Optional — e.g. focus on shorter replies"
              />
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-xs text-red-700 dark:text-red-400 bg-red-500/10 rounded-md p-2 border border-red-500/20">
            {error}
          </div>
        )}

        {/* Build / Send actions */}
        {!result ? (
          <div className="flex gap-2">
            <Button
              type="button"
              size="sm"
              onClick={handleBuildAndSend}
              disabled={busy || !apiReady || !activeProfileId || !characterId}
              className="gap-1.5"
            >
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Build prompt
            </Button>
          </div>
        ) : (
          <div className="space-y-2">
            {result?.stats && (
              <p className="text-[10px] text-muted-foreground">
                ~{result.stats.bundle_chars?.toLocaleString?.() ?? result.stats.bundle_chars} chars
                {' '}&middot; {result.stats.agentic_insight_count} agentic rows
                {' '}&middot; {result.stats.backend_memory_count} profile rows
              </p>
            )}

            <div className="flex gap-2 flex-wrap">
              <Button
                type="button"
                size="sm"
                onClick={handleSendToChat}
                disabled={isGenerating}
                className="gap-1.5"
              >
                <Send className="h-3.5 w-3.5" />
                Send to chat
              </Button>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={handleCopy}
                className="gap-1.5"
              >
                <Copy className="h-3.5 w-3.5" />
                Copy
              </Button>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={handleBuildAgain}
                className="gap-1.5"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Rebuild
              </Button>
            </div>

            {/* Preview */}
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                Preview prompt
              </summary>
              <Textarea
                readOnly
                value={result.combined?.slice(0, 12000) || ''}
                className="min-h-[100px] text-[10px] font-mono mt-1 resize-none"
                spellCheck={false}
              />
              {(result.combined?.length || 0) > 12000 && (
                <p className="text-[10px] text-amber-600 mt-1">
                  Truncated preview — use Copy for the full prompt.
                </p>
              )}
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
