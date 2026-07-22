import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowRight, CheckCircle2, Loader2 } from 'lucide-react';
import { fetchWithTimeout, formatFetchError } from '../config/api';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

export default function CharacterMemoryTransferPanel({
  apiUrl,
  apiReady,
  activeProfileId,
  characters = [],
  onApplied,
}) {
  const [profiles, setProfiles] = useState([]);
  const [sourceId, setSourceId] = useState('');
  const [targetId, setTargetId] = useState('');
  const [mode, setMode] = useState('merge');
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const characterNames = useMemo(
    () => new Map((characters || []).filter((character) => character?.id).map((character) => [character.id, character.name || character.id])),
    [characters]
  );

  const loadProfiles = useCallback(async () => {
    if (!apiReady || !apiUrl || !activeProfileId) {
      setProfiles([]);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await fetchWithTimeout(
        `${apiUrl.replace(/\/$/, '')}/memory/agentic/list?user_id=${encodeURIComponent(activeProfileId)}`,
        {},
        25000
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== 'success') {
        throw new Error(data.detail || data.error || `Status ${response.status}`);
      }
      const nextProfiles = Array.isArray(data.profiles) ? data.profiles.filter((profile) => profile?.character_id) : [];
      setProfiles(nextProfiles);
      setSourceId((current) => current || nextProfiles[0]?.character_id || '');
    } catch (fetchError) {
      setProfiles([]);
      setError(formatFetchError(fetchError, { timeoutMs: 25000 }));
    } finally {
      setLoading(false);
    }
  }, [activeProfileId, apiReady, apiUrl]);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  useEffect(() => {
    if (sourceId && targetId === sourceId) setTargetId('');
  }, [sourceId, targetId]);

  const targetCharacters = useMemo(
    () => (characters || []).filter((character) => character?.id && character.id !== sourceId && character.chat_role !== 'user'),
    [characters, sourceId]
  );

  const sourceProfile = profiles.find((profile) => profile.character_id === sourceId);
  const sourceName = characterNames.get(sourceId) || sourceId;
  const targetName = characterNames.get(targetId) || targetId;

  const applyTransfer = useCallback(async () => {
    if (!activeProfileId || !sourceId || !targetId) {
      setError('Choose both the source and destination characters.');
      return;
    }
    const action = mode === 'replace'
      ? `Replace everything ${targetName} remembers with a copy of ${sourceName}'s memories? This cannot be undone automatically.`
      : `Add ${sourceName}'s memories to ${targetName}? Existing duplicate lines will be skipped.`;
    if (!window.confirm(action)) return;

    setApplying(true);
    setError('');
    setSuccess('');
    try {
      const response = await fetchWithTimeout(
        `${apiUrl.replace(/\/$/, '')}/memory/agentic/copy_to_character`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: activeProfileId,
            source_character_id: sourceId,
            target_character_id: targetId,
            mode,
          }),
        },
        25000
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== 'success') {
        throw new Error(data.detail || data.error || `Status ${response.status}`);
      }
      setSuccess(
        mode === 'replace'
          ? `${targetName} now has ${data.written_count ?? 0} copied memories.`
          : `${data.added ?? 0} memories were added to ${targetName}; duplicates were left out.`
      );
      await loadProfiles();
      onApplied?.();
    } catch (applyError) {
      setError(formatFetchError(applyError, { timeoutMs: 25000 }));
    } finally {
      setApplying(false);
    }
  }, [activeProfileId, apiUrl, loadProfiles, mode, onApplied, sourceId, sourceName, targetId, targetName]);

  if (!activeProfileId) {
    return <p className="text-sm text-muted-foreground">Choose a user profile before moving character memories.</p>;
  }

  return (
    <div className="space-y-5 rounded-xl border bg-card/70 p-5">
      <div>
        <h2 className="text-lg font-semibold">Move memories between characters</h2>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          Each character keeps a separate record of what it has learned about you. Copy that history when you want a new character to inherit the same continuity.
        </p>
      </div>

      {loading ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" />Loading character memories…</p>
      ) : profiles.length === 0 ? (
        <p className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
          No character-specific memories exist for this user profile yet. They appear after Mirid has recorded long-term observations from a character chat.
        </p>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-[1fr_auto_1fr] md:items-end">
            <div className="space-y-2">
              <Label>Copy memories from</Label>
              <Select value={sourceId} onValueChange={setSourceId}>
                <SelectTrigger><SelectValue placeholder="Choose a character" /></SelectTrigger>
                <SelectContent>
                  {profiles.map((profile) => (
                    <SelectItem key={profile.character_id} value={profile.character_id}>
                      {characterNames.get(profile.character_id) || profile.character_id} · {profile.count ?? profile.insights?.length ?? 0} memories
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <ArrowRight className="mx-auto mb-2 hidden h-5 w-5 text-muted-foreground md:block" aria-hidden="true" />
            <div className="space-y-2">
              <Label>Into</Label>
              <Select value={targetId} onValueChange={setTargetId}>
                <SelectTrigger><SelectValue placeholder="Choose a different character" /></SelectTrigger>
                <SelectContent>
                  {targetCharacters.map((character) => (
                    <SelectItem key={character.id} value={character.id}>{character.name || character.id}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2 max-w-md">
            <Label>How to combine them</Label>
            <Select value={mode} onValueChange={(value) => setMode(value === 'replace' ? 'replace' : 'merge')}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="merge">Add memories and skip duplicates</SelectItem>
                <SelectItem value="replace">Replace the destination character's memories</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs leading-relaxed text-muted-foreground">
              The source character is never changed. Replacing removes the destination character's existing memory history.
            </p>
          </div>

          <Button type="button" onClick={applyTransfer} disabled={applying || !sourceId || !targetId}>
            {applying ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ArrowRight className="mr-2 h-4 w-4" />}
            {mode === 'replace' ? 'Replace memories' : 'Copy memories'}
          </Button>
        </>
      )}

      {error ? <p className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</p> : null}
      {success ? <p className="flex items-center gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-800 dark:text-emerald-200"><CheckCircle2 className="h-4 w-4" />{success}</p> : null}
    </div>
  );
}
