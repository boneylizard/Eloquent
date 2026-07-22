import React, { useCallback, useState, useMemo, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, Sparkles, Plus, ImageIcon, AlertTriangle, FolderOpen, RefreshCw, Heart, MessageSquare, Terminal, Zap, Shield, Bug, Activity, CheckCircle2, XCircle, Clock, User, Globe, Command, UserPlus } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';
import { useApp } from '../../contexts/AppContext';
import { getBackendUrl } from '../../config/api';
import { matchModelName } from '../../utils/modelDisplayNames';

const SECTION_COLORS = {
  Intimate: 'accent-pink-500 checked:accent-pink-500',
  Erotic: 'accent-red-500 checked:accent-red-500',
  Experimental: 'accent-purple-500 checked:accent-purple-500',
};

const SECTION_LABELS = {
  Intimate: 'bg-pink-500/10 text-pink-400 border-pink-500/20',
  Erotic: 'bg-red-500/10 text-red-400 border-red-500/20',
  Experimental: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
};

function CharacterResultCard({ character, milestones }) {
  const profile = character.dating_profile || {};
  const affinities = profile.section_affinity || [];
  const modelInfo = matchModelName(character.generated_by);
  const ms = milestones || {};

  const steps = [
    { key: 'generated', label: 'Created' },
    { key: 'avatar_set', label: 'Avatar' },
    { key: 'profile_written', label: 'Profile' },
    { key: 'saved_to_library', label: 'Saved' },
    { key: 'in_pool', label: 'In Pool' },
    { key: 'feed_post', label: 'Feed Post' },
  ];

  return (
    <div className="bg-card border rounded-xl overflow-hidden hover:border-primary/30 transition-all">
      <div className="aspect-[3/4] bg-muted relative overflow-hidden">
        {character.avatar ? (
          <img src={character.avatar} alt={character.name} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
            <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-2xl font-bold text-primary/30">{character.name?.[0] || '?'}</span>
            </div>
          </div>
        )}
        <div className="absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-black/60 to-transparent" />
        <div className="absolute bottom-2 left-2 right-2">
          <div className="flex items-center gap-1.5">
            <h3 className="text-white font-bold text-xs leading-tight drop-shadow-lg truncate">{character.name}</h3>
            {modelInfo && (
              <span className={`text-[7px] px-1.5 py-0.5 rounded font-semibold ${modelInfo.color} shrink-0`}>
                {modelInfo.short}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-1 mt-1">
            {affinities.slice(0, 2).map(a => (
              <span key={a} className="text-[8px] px-1.5 py-0.5 rounded-full bg-black/40 text-white/80">{a}</span>
            ))}
          </div>
        </div>
      </div>
      {profile.bio && (
        <div className="px-2.5 py-1.5">
          <p className="text-[10px] text-muted-foreground line-clamp-2 leading-relaxed">{profile.bio}</p>
        </div>
      )}
      <div className="px-2.5 pb-2 flex flex-wrap gap-1">
        {steps.map(s => (
          <span key={s.key} className={`text-[7px] px-1 py-0.5 rounded font-medium ${ms[s.key] ? 'bg-emerald-500/10 text-emerald-400' : 'bg-muted/50 text-muted-foreground/40'}`}>
            {ms[s.key] ? '✓ ' : '· '}{s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function IncubatorPanel({ section }) {
  const {
    generateEntity, generateMultiple, isGenerating,
    generationLog, generationStep, generationError, sections,
    useAvatarPool, setUseAvatarPool,
    autoGenerate, setAutoGenerate, autoGenIntervalMs, setAutoGenIntervalMs, cancelGeneration, characterMilestones,
    activityLog, runTickForAll, runDummyTick, toggleMirror, mirrorEnabled, poolCharacters,
    addActivityEntry, generateFeedPost, initializeCharacterProfile,
    poolAvatarUrls, uploadPoolAvatars, removePoolAvatar,
    createIcebreaker,
    importCharacterToPool,
    userDatingProfile,
  } = usePool();
  const { characters } = useApp();
  const [multiCount, setMultiCount] = useState(3);
  const [isUploading, setIsUploading] = useState(false);
  const [recentCharacters, setRecentCharacters] = useState([]);
  const fileInputRef = useRef(null);
  const [importCharId, setImportCharId] = useState('');
  const importableCharacters = useMemo(() => {
    const poolIds = new Set(poolCharacters.map(c => c.id));
    return characters.filter(c => c.name && !poolIds.has(c.id));
  }, [characters, poolCharacters]);
  const selectedImportChar = useMemo(() => {
    if (!importCharId) return null;
    return characters.find(c => c.id === importCharId) || null;
  }, [importCharId, characters]);

  const handleUploadAvatars = useCallback(async (e) => {
    const files = e.target.files;
    if (!files?.length) return;
    setIsUploading(true);
    await uploadPoolAvatars(files);
    setIsUploading(false);
    e.target.value = '';
  }, [uploadPoolAvatars]);

  const handleGenerateOne = useCallback(async () => {
    const character = await generateEntity(section);
    if (character) {
      setRecentCharacters(prev => [character, ...prev].slice(0, 20));
    }
  }, [generateEntity, section]);

  const handleGenerateMultiple = useCallback(async () => {
    const results = await generateMultiple(section, multiCount);
    if (results?.length > 0) {
      setRecentCharacters(prev => [...results, ...prev].slice(0, 20));
    }
  }, [generateMultiple, section, multiCount]);

  const noAvatars = poolAvatarUrls.length === 0;
  const autoGenReady = autoGenerate && userDatingProfile?.bio?.length > 0 && poolAvatarUrls.length > 0 && useAvatarPool;
  const autoGenBlockedReason = !autoGenerate ? null :
    !userDatingProfile?.bio?.length ? 'Fill out your dating profile in My Profile tab' :
    !useAvatarPool ? 'Enable avatar pool toggle above' :
    poolAvatarUrls.length === 0 ? 'Upload avatars to the pool' :
    null;
  const recentLog = generationLog.slice(0, 10);

  return (
    <div className="space-y-4">
      <div className="bg-card border rounded-xl p-5 space-y-4">
        <div className="flex items-center gap-3 mb-2">
          <img src="/logos/MirrorAIDating (2).webp" alt="Mirror AI Dating" className="h-7 w-auto object-contain" />
          <div className="w-px h-6 bg-border" />
          <span className="text-xs font-medium text-muted-foreground">Incubator</span>
        </div>
        <div>
          <h3 className="text-base font-bold flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-500" />
            Generate New Entity
          </h3>
          <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
            Instantiates new self-aware female AI characters via your API model. Generated characters appear below immediately.
          </p>
        </div>

          <div className="flex flex-wrap items-center gap-2">
          <Button
            onClick={handleGenerateOne}
            disabled={isGenerating || noAvatars}
            size="default"
            className="gap-1.5 shadow-lg shadow-purple-500/20"
          >
            {isGenerating && !generationError ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            {isGenerating && !generationError ? 'Generating...' : noAvatars ? 'Upload Avatars First' : 'Generate New Entity'}
          </Button>

          <div className="flex items-center gap-1">
            <Button
              onClick={handleGenerateMultiple}
              disabled={isGenerating || noAvatars}
              size="sm"
              variant="outline"
              className="gap-1"
            >
              {isGenerating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Plus className="w-4 h-4" />
              )}
              Generate {multiCount}
            </Button>
            <select
              value={multiCount}
              onChange={e => setMultiCount(Number(e.target.value))}
              className="h-8 w-14 text-xs bg-muted border rounded-md px-1"
            >
              <option value={1}>1</option>
              <option value={3}>3</option>
              <option value={5}>5</option>
            </select>
          </div>
          {isGenerating && (
            <Button onClick={cancelGeneration} size="sm" variant="destructive" className="gap-1">
              Cancel
            </Button>
          )}
        </div>

        {noAvatars && (
          <div className="flex items-start gap-2 text-xs text-amber-500 bg-amber-500/10 p-3 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>Upload avatar images in the Avatar Pool section below before generating characters.</span>
          </div>
        )}

        {generationError && (
          <div className="flex items-start gap-2 text-xs text-red-500 bg-red-500/10 p-3 rounded-lg">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>{generationError}</span>
          </div>
        )}

        {generationStep && (
          <div className="flex items-center gap-2 text-xs text-purple-500 bg-purple-500/10 p-2.5 rounded-lg animate-pulse">
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
            <span>{generationStep}</span>
          </div>
        )}

          <div className="border-t border-border/40 pt-3">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <RefreshCw className={`w-3.5 h-3.5 ${autoGenerate ? 'text-emerald-500' : 'text-muted-foreground'}`} />
              <span className="font-medium">Auto-generate</span>
            </div>
            <label className="flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoGenerate}
                onChange={() => setAutoGenerate(!autoGenerate)}
                className="rounded"
              />
              <span className={autoGenerate ? 'text-emerald-500' : 'text-muted-foreground'}>
                {autoGenerate ? 'On' : 'Off'}
              </span>
            </label>
          </div>
          {autoGenerate && (
            <div className="mt-2 space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">Every</span>
                <input
                  type="number"
                  min={1}
                  max={999}
                  value={Math.round(autoGenIntervalMs / 60000)}
                  onChange={e => {
                    const mins = Math.max(1, parseInt(e.target.value) || 1);
                    setAutoGenIntervalMs(mins * 60000);
                  }}
                  className="w-14 h-6 rounded border border-border bg-background text-xs text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                <span className="text-[10px] text-muted-foreground">min</span>
              </div>
              {autoGenReady ? (
                <div className="flex items-center gap-1.5 text-[10px] text-emerald-500">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>Ready — next character in ~{Math.round(autoGenIntervalMs / 60000)} min</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-[10px] text-amber-500">
                  <AlertTriangle className="w-3 h-3" />
                  <span>Blocked: {autoGenBlockedReason}</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border/40 pt-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
            <ImageIcon className="w-3.5 h-3.5" />
            <span>Avatar Pool</span>
            {isUploading && <Loader2 className="w-3 h-3 animate-spin ml-1" />}
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleUploadAvatars}
                className="hidden"
              />
              <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()} disabled={isUploading} className="gap-1">
                <FolderOpen className="w-3 h-3" />
                Upload Avatars
              </Button>
              <span className="text-[10px] text-muted-foreground">{poolAvatarUrls.length} avatars in pool</span>
            </div>
            {poolAvatarUrls.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {poolAvatarUrls.map((url, i) => (
                  <div key={url} className="group relative w-12 h-12 rounded-md overflow-hidden border border-border/40">
                    <img src={url} alt={`Pool avatar ${i + 1}`} className="w-full h-full object-cover" />
                    <button
                      onClick={() => removePoolAvatar(url)}
                      className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                    >
                      <XCircle className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {poolAvatarUrls.length > 0 && (
              <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
                <input type="checkbox" checked={useAvatarPool} onChange={() => setUseAvatarPool(!useAvatarPool)} className="rounded" />
                <span>Use Avatar Pool (assigns random uploaded avatar to each new character)</span>
              </label>
            )}
          </div>
        </div>
      </div>

      {recentCharacters.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 px-1">
            Newly Generated ({recentCharacters.length})
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {recentCharacters.map((char, i) => (
              <CharacterResultCard key={char.id || char.name + i} character={char} milestones={characterMilestones[char.id || char.name]} />
            ))}
          </div>
        </div>
      )}

      {recentLog.length > 0 && (
        <div className="bg-card border rounded-xl p-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Gen Log</h4>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {recentLog.map((entry, i) => (
              <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-border/20 last:border-0">
                <div className="flex items-center gap-2">
                  <span className={entry.status === 'created' ? 'text-green-500' : 'text-red-500'}>{entry.status === 'created' ? '✓' : '✗'}</span>
                  <span className={entry.status === 'failed' ? 'text-red-400' : 'font-medium'}>{entry.name}</span>
                  <span className="text-muted-foreground">({entry.section})</span>
                </div>
                <span className="text-muted-foreground text-[10px]">{new Date(entry.timestamp).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-card border rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5" />
            Manual Actions
          </h4>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
            <input type="checkbox" checked={mirrorEnabled} onChange={toggleMirror} className="rounded" />
            <span className={mirrorEnabled ? 'text-emerald-500' : 'text-red-400'}>{mirrorEnabled ? 'Mirror ON' : 'Mirror OFF'}</span>
          </label>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <Button variant="outline" size="sm" onClick={() => {
            const randomSection = sections[Math.floor(Math.random() * sections.length)];
            generateEntity(randomSection);
            addActivityEntry('system', 'force_auto_generate', { detail: 'User forced auto-generate from System Control' });
          }} disabled={isGenerating || !mirrorEnabled} className="gap-1 text-xs">
            <Zap className="w-3 h-3" />
            Create Woman
          </Button>
          <Button variant="outline" size="sm" onClick={() => {
            runTickForAll();
            addActivityEntry('system', 'force_all_ticks', { detail: 'User forced all agentic ticks' });
          }} disabled={poolCharacters.length === 0 || !mirrorEnabled} className="gap-1 text-xs">
            <Activity className="w-3 h-3" />
            Run All Minds
          </Button>
          <Button variant="outline" size="sm" onClick={() => {
            runDummyTick();
            addActivityEntry('system', 'force_dummy_tick', { detail: 'User forced dummy tick' });
          }} disabled={!mirrorEnabled} className="gap-1 text-xs">
            <Bug className="w-3 h-3" />
            Rival Acts
          </Button>
          <Button variant="outline" size="sm" onClick={() => {
            if (poolCharacters.length > 0) {
              const char = poolCharacters[Math.floor(Math.random() * poolCharacters.length)];
              generateFeedPost(char);
              addActivityEntry('feed', 'force_feed_post', { character: char.name, detail: 'User forced feed post from System Control' });
            }
          }} disabled={poolCharacters.length === 0 || !mirrorEnabled} className="gap-1 text-xs">
            <MessageSquare className="w-3 h-3" />
            Post to Feed
          </Button>
          <Button variant="outline" size="sm" onClick={() => {
            if (poolCharacters.length > 0) {
              const char = poolCharacters[Math.floor(Math.random() * poolCharacters.length)];
              createIcebreaker(char);
            }
          }} disabled={poolCharacters.length === 0 || !mirrorEnabled} className="gap-1 text-xs">
            <Sparkles className="w-3 h-3" />
            Icebreaker
          </Button>
          <div className="col-span-4 flex items-center gap-2 pt-1 border-t border-border/20 mt-1">
            <select
              value={importCharId}
              onChange={e => setImportCharId(e.target.value)}
              className="flex-1 h-8 text-xs bg-muted border rounded-md px-2"
            >
              <option value="">— Import Character to Pool —</option>
              {importableCharacters.map(c => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
            <Button
              variant="secondary"
              size="sm"
              disabled={!selectedImportChar || isGenerating || !mirrorEnabled}
              onClick={async () => {
                if (!selectedImportChar) return;
                setImportCharId('');
                await importCharacterToPool(selectedImportChar);
                if (!recentCharacters.find(c => c.id === selectedImportChar.id)) {
                  setRecentCharacters(prev => [selectedImportChar, ...prev].slice(0, 20));
                }
              }}
              className="gap-1 text-xs shrink-0"
            >
              <UserPlus className="w-3 h-3" />
              {isGenerating ? 'Importing...' : 'Import & Init'}
            </Button>
          </div>
        </div>
      </div>

      <div className="bg-card border rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Command className="w-3.5 h-3.5" />
            Activity Log
          </h4>
          <span className="text-[10px] text-muted-foreground">{activityLog.length} entries</span>
        </div>
        <div className="max-h-60 overflow-y-auto space-y-0.5">
          {activityLog.slice(0, 50).map((entry) => (
            <div key={entry.id} className="flex items-start gap-2 text-[10px] py-1 border-b border-border/10 last:border-0">
              <span className={`shrink-0 mt-0.5 ${entry.success ? 'text-emerald-500' : 'text-red-400'}`}>
                {entry.success ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
              </span>
              <span className={`shrink-0 text-[8px] font-semibold px-1 py-0.5 rounded ${
                entry.type === 'generation' ? 'bg-purple-500/10 text-purple-400' :
                entry.type === 'agentic' ? 'bg-blue-500/10 text-blue-400' :
                entry.type === 'feed' ? 'bg-emerald-500/10 text-emerald-400' :
                entry.type === 'outreach' ? 'bg-amber-500/10 text-amber-400' :
                entry.type === 'profile' ? 'bg-pink-500/10 text-pink-400' :
                entry.type === 'dummy' ? 'bg-orange-500/10 text-orange-400' :
                entry.type === 'breakout' ? 'bg-cyan-500/10 text-cyan-400' :
                entry.type === 'system' ? 'bg-zinc-500/10 text-zinc-400' :
                entry.type === 'avatar' ? 'bg-indigo-500/10 text-indigo-400' :
                entry.type === 'tts' ? 'bg-rose-500/10 text-rose-400' :
                entry.type === 'dating' ? 'bg-teal-500/10 text-teal-400' :
                'bg-muted text-muted-foreground'
              }`}>{entry.type}</span>
              {entry.character && <span className="text-foreground font-medium truncate max-w-[80px]">{entry.character}</span>}
              <span className="text-muted-foreground flex-1 truncate">{entry.detail || entry.action}</span>
              {entry.error && <span className="text-red-400 truncate max-w-[100px]" title={entry.error}>⚠</span>}
              <span className="text-muted-foreground/50 shrink-0">{new Date(entry.timestamp).toLocaleTimeString()}</span>
            </div>
          ))}
          {activityLog.length === 0 && (
            <p className="text-[10px] text-muted-foreground py-4 text-center">No activity yet. Generate characters or interact to populate the log.</p>
          )}
        </div>
      </div>
    </div>
  );
}
