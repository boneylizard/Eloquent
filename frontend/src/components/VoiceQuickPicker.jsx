import React, { useMemo, useState, useEffect, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Loader2 } from 'lucide-react';
import { getBackendUrl } from '../config/api';

const KOKORO_VOICE_FALLBACK = [
  { id: 'af_heart', name: 'Am. English Female (Heart)' },
  { id: 'af_alloy', name: 'Am. English Female (Alloy)' },
  { id: 'af_aoede', name: 'Am. English Female (Aoede)' },
  { id: 'af_bella', name: 'Am. English Female (Bella)' },
  { id: 'af_jessica', name: 'Am. English Female (Jessica)' },
  { id: 'af_kore', name: 'Am. English Female (Kore)' },
  { id: 'af_nicole', name: 'Am. English Female (Nicole)' },
  { id: 'af_nova', name: 'Am. English Female (Nova)' },
  { id: 'af_river', name: 'Am. English Female (River)' },
  { id: 'af_sarah', name: 'Am. English Female (Sarah)' },
  { id: 'af_sky', name: 'Am. English Female (Sky)' },
  { id: 'am_adam', name: 'Am. English Male (Adam)' },
  { id: 'am_echo', name: 'Am. English Male (Echo)' },
];

export function buildVoiceExperimentTargets(multiRoleMode, characters, activeCharacterIds, activeCharacter) {
  const roster = (characters || []).filter((c) => (c?.chat_role || 'npc') !== 'user');
  if (multiRoleMode && roster.length) {
    const ids =
      Array.isArray(activeCharacterIds) && activeCharacterIds.length
        ? activeCharacterIds
        : roster.map((c) => c.id);
    const allow = new Set(ids);
    const filtered = roster.filter((c) => allow.has(c.id));
    return filtered.length ? filtered : roster;
  }
  if (activeCharacter) return [activeCharacter];
  return roster.length ? [roster[0]] : [];
}

function PickerBody({
  isChatterboxEngine,
  isKokoroEngine,
  targets,
  selectedCharacterId,
  setSelectedCharacterId,
  selectedCharacter,
  voiceOptions,
  isFetchingVoices,
  onVoiceChange,
  hintClassName,
  darkSheet,
}) {
  if (!isChatterboxEngine && !isKokoroEngine) {
    return (
      <p className={hintClassName || 'text-sm text-muted-foreground'}>
        Quick voice switch works with Kokoro or Chatterbox. Change the TTS engine in Settings → Audio.
      </p>
    );
  }

  if (!targets.length) {
    return (
      <p className={hintClassName || 'text-sm text-muted-foreground'}>
        No characters loaded yet. Create or pick a character first.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {targets.length > 1 && (
        <div className="space-y-2">
          <Label className={hintClassName ? 'text-xs text-white/70' : 'text-xs'}>Character</Label>
          <Select value={selectedCharacterId} onValueChange={setSelectedCharacterId}>
            <SelectTrigger
              className={
                darkSheet ? 'border-white/20 bg-black/50 text-white' : ''
              }
            >
              <SelectValue placeholder="Pick character" />
            </SelectTrigger>
            <SelectContent
              className={
                darkSheet
                  ? 'z-[10060] max-h-[40vh] overflow-y-auto border border-white/10 bg-zinc-900 text-zinc-100'
                  : 'max-h-[50vh] overflow-y-auto'
              }
            >
              {targets.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {c.name || 'Unnamed'}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className={hintClassName || 'text-xs text-muted-foreground'}>
            In multi-role mode, each roster member keeps their own clone until you change it here.
          </p>
        </div>
      )}

      <div className="space-y-2">
        <Label className={hintClassName ? 'text-xs text-white/70' : 'text-xs'}>
          {isChatterboxEngine ? 'Voice clone' : 'Voice'}
        </Label>
        <Select
          value={selectedCharacter?.ttsVoice || 'default'}
          disabled={isFetchingVoices || !selectedCharacter}
          onValueChange={onVoiceChange}
        >
          <SelectTrigger
            className={
              darkSheet ? 'border-white/20 bg-black/50 text-white' : ''
            }
          >
            <SelectValue placeholder={isFetchingVoices ? 'Loading…' : 'Pick a voice'} />
          </SelectTrigger>
          <SelectContent
            className={
              darkSheet
                ? 'z-[10060] max-h-[min(50vh,320px)] overflow-y-auto border border-white/10 bg-zinc-900 text-zinc-100'
                : 'max-h-[min(50vh,320px)] overflow-y-auto'
            }
          >
            <SelectItem value="default">Default (global setting)</SelectItem>
            {voiceOptions.map((v) => (
              <SelectItem key={v.id} value={v.id}>
                {v.name || v.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {isFetchingVoices && (
          <div className="flex items-center gap-2 text-xs opacity-70">
            <Loader2 className="h-3 w-3 animate-spin" />
            Loading voice list…
          </div>
        )}
      </div>

      <p className={hintClassName || 'text-xs text-muted-foreground'}>
        Saved on this character — the next spoken reply uses it (same as roster / character editor).
      </p>
    </div>
  );
}

/**
 * Quick experiment UI for per-character TTS voice / Chatterbox clones.
 * `variant`: `dialog` (Chat, Focus) or `call-sheet` (Call overlay — dark bottom sheet).
 */
export default function VoiceQuickPicker({
  open,
  onOpenChange,
  variant = 'dialog',
  primaryApiUrl: primaryApiUrlProp,
}) {
  const {
    PRIMARY_API_URL,
    settings,
    characters,
    activeCharacter,
    activeCharacterIds,
    saveCharacter,
  } = useApp();

  const primaryApiUrl = primaryApiUrlProp || PRIMARY_API_URL || getBackendUrl();
  const ttsEngine = settings?.ttsEngine || 'kokoro';
  const isChatterboxEngine = ttsEngine === 'chatterbox' || ttsEngine === 'chatterbox_turbo' || ttsEngine === 'chatterbox_nano' || ttsEngine === 'voxcpm';
  const isKokoroEngine = ttsEngine === 'kokoro';
  const multiRoleMode = settings?.multiRoleMode === true;

  const targets = useMemo(
    () => buildVoiceExperimentTargets(multiRoleMode, characters, activeCharacterIds, activeCharacter),
    [multiRoleMode, characters, activeCharacterIds, activeCharacter]
  );

  const [selectedCharacterId, setSelectedCharacterId] = useState('');
  const [availableVoices, setAvailableVoices] = useState({ chatterbox_voices: [], kokoro_voices: [] });
  const [isFetchingVoices, setIsFetchingVoices] = useState(false);

  useEffect(() => {
    if (!open) return;
    const first = targets[0]?.id || '';
    setSelectedCharacterId((prev) => {
      if (prev && targets.some((t) => t.id === prev)) return prev;
      if (activeCharacter?.id && targets.some((t) => t.id === activeCharacter.id)) return activeCharacter.id;
      return first;
    });
  }, [open, targets, activeCharacter?.id]);

  const selectedCharacter = useMemo(
    () => targets.find((t) => t.id === selectedCharacterId) || targets[0] || null,
    [targets, selectedCharacterId]
  );

  const voiceOptions = useMemo(() => {
    if (isChatterboxEngine) return availableVoices?.chatterbox_voices || [];
    if (isKokoroEngine) {
      if (availableVoices?.kokoro_voices?.length) return availableVoices.kokoro_voices;
      return KOKORO_VOICE_FALLBACK;
    }
    return [];
  }, [availableVoices, isChatterboxEngine, isKokoroEngine]);

  const fetchVoices = useCallback(async () => {
    if (!isChatterboxEngine && !isKokoroEngine) return;
    setIsFetchingVoices(true);
    try {
      const base = primaryApiUrl || getBackendUrl();
      const res = await fetch(`${String(base).replace(/\/+$/, '')}/tts/voices`);
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      setAvailableVoices(data || { chatterbox_voices: [], kokoro_voices: [] });
    } catch (e) {
      console.warn('[VoiceQuickPicker] voices fetch failed', e);
      setAvailableVoices({ chatterbox_voices: [], kokoro_voices: [] });
    } finally {
      setIsFetchingVoices(false);
    }
  }, [primaryApiUrl, isChatterboxEngine, isKokoroEngine]);

  useEffect(() => {
    if (!open) return;
    if (!isChatterboxEngine && !isKokoroEngine) return;
    fetchVoices();
  }, [open, isChatterboxEngine, isKokoroEngine, fetchVoices]);

  const onVoiceChange = useCallback(
    async (value) => {
      if (!selectedCharacter) return;
      saveCharacter({ ...selectedCharacter, ttsVoice: value });
      if (isChatterboxEngine && value && value !== 'default') {
        try {
          await fetch(`${String(primaryApiUrl).replace(/\/+$/, '')}/tts/save-voice-preference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_id: value, engine: ttsEngine }),
          });
        } catch (err) {
          console.warn('[VoiceQuickPicker] save-voice-preference failed', err);
        }
      }
    },
    [selectedCharacter, saveCharacter, isChatterboxEngine, primaryApiUrl, ttsEngine]
  );

  const body = (
    <PickerBody
      isChatterboxEngine={isChatterboxEngine}
      isKokoroEngine={isKokoroEngine}
      targets={targets}
      selectedCharacterId={selectedCharacterId || targets[0]?.id || ''}
      setSelectedCharacterId={setSelectedCharacterId}
      selectedCharacter={selectedCharacter}
      voiceOptions={voiceOptions}
      isFetchingVoices={isFetchingVoices}
      onVoiceChange={onVoiceChange}
      hintClassName={variant === 'call-sheet' ? 'text-xs text-white/60' : undefined}
      darkSheet={variant === 'call-sheet'}
    />
  );

  if (variant === 'call-sheet') {
    if (!open) return null;
    return (
      <>
        <div
          className="fixed inset-0 z-[10000] bg-black/70 backdrop-blur-sm"
          aria-hidden
          onClick={() => onOpenChange(false)}
        />
        <div className="fixed bottom-0 left-0 right-0 z-[10001] max-h-[min(78vh,520px)] flex flex-col rounded-t-2xl border border-white/15 bg-zinc-950 shadow-2xl">
          <div className="mx-auto mt-3 h-1 w-10 shrink-0 rounded-full bg-white/25" />
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <h3 className="text-base font-semibold text-white tracking-tight">Voices</h3>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              className="rounded-full px-3 py-1 text-sm text-white/70 hover:bg-white/10 hover:text-white"
            >
              Done
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto px-4 pb-6 pt-2">{body}</div>
        </div>
      </>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Quick voice</DialogTitle>
          <DialogDescription>
            Swap Kokoro voices or Chatterbox clones per character without opening the full roster or Settings.
          </DialogDescription>
        </DialogHeader>
        {body}
        <DialogFooter>
          <Button type="button" variant="secondary" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
