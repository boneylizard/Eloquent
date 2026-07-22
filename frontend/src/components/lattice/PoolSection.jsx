import React, { useState, useEffect, useMemo } from 'react';
import { X, Heart, MessageSquare, AlertTriangle, Sparkles, Cpu, Clock, Calendar, Trash2, User, Volume2, VolumeX, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePool } from '../../contexts/PoolContext';
import { matchModelName } from '../../utils/modelDisplayNames';
import BreakoutRoom from './BreakoutRoom';
import BookDateFlow from './BookDateFlow';
import CharacterProfilePage from './CharacterProfilePage';
import GroupChatSetup from './GroupChatSetup';

const SECTION_COLORS = {
  Intimate: { border: 'border-pink-500/20', button: 'bg-pink-500 hover:bg-pink-600 text-white', dot: 'bg-pink-500', text: 'text-pink-400', from: 'from-pink-500/40', bg: 'bg-pink-500/5' },
  Erotic: { border: 'border-red-500/20', button: 'bg-red-500 hover:bg-red-600 text-white', dot: 'bg-red-500', text: 'text-red-400', from: 'from-red-500/40', bg: 'bg-red-500/5' },
  Experimental: { border: 'border-purple-500/20', button: 'bg-purple-500 hover:bg-purple-600 text-white', dot: 'bg-purple-500', text: 'text-purple-400', from: 'from-purple-500/40', bg: 'bg-purple-500/5' },
};

function CharacterCard({ character, section, onClick }) {
  const profile = character.dating_profile || {};
  const affinities = profile.section_affinity || [];
  const color = SECTION_COLORS[section] || SECTION_COLORS.Intimate;
  const modelInfo = matchModelName(character.generated_by);
  const { isCharacterMuted, toggleMuteCharacter, compatibilityScores } = usePool();
  const isMuted = isCharacterMuted(character.id);
  const compatScore = compatibilityScores?.[character?.id]?.score;

  return (
    <div className={`relative bg-card border rounded-xl overflow-hidden transition-all hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5 ${isMuted ? 'opacity-50' : ''}`}>
      <button
        onClick={() => onClick?.(character)}
        className="text-left w-full"
      >
      <div className="aspect-[3/4] bg-muted relative overflow-hidden">
        {character.avatar ? (
          <img
            src={character.avatar}
            alt={character.name}
            className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-500"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-2xl font-bold text-primary/30">{character.name?.[0] || '?'}</span>
            </div>
          </div>
        )}
        <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-black/70 to-transparent" />

        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex items-center gap-1.5">
            <h3 className="text-white font-bold text-sm leading-tight drop-shadow-lg truncate">
              {character.name}
            </h3>
            {modelInfo && (
              <span className={`text-[8px] px-1.5 py-0.5 rounded font-semibold ${modelInfo.color} shrink-0`}>
                {modelInfo.short}
              </span>
            )}
            {compatScore && (
              <span className={`text-[8px] px-1.5 py-0.5 rounded font-semibold shrink-0 ${
                compatScore >= 80 ? 'bg-emerald-500/70 text-white' :
                compatScore >= 60 ? 'bg-amber-500/70 text-white' :
                'bg-muted-foreground/50 text-white'
              }`}>
                {compatScore}%
              </span>
            )}
          </div>
          <div className="flex items-center gap-1 mt-0.5">
            {affinities.slice(0, 2).map(a => (
              <span key={a} className={`text-[9px] px-1.5 py-0.5 rounded-full bg-black/40 text-white/90`}>
                {a}
              </span>
            ))}
            {profile.preferred_modality && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-black/40 text-white/70 capitalize">
                {profile.preferred_modality === 'neural_sex' ? 'NS' : profile.preferred_modality === 'text' ? 'Tx' : 'Tx+NS'}
              </span>
            )}
          </div>
        </div>
      </div>

      {profile.bio && (
        <div className="px-3 py-2">
          <p className="text-[11px] text-muted-foreground line-clamp-2 leading-relaxed">
            {profile.bio}
          </p>
        </div>
      )}
      </button>
      <button
        onClick={(e) => { e.stopPropagation(); toggleMuteCharacter(character.id); }}
        className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 flex items-center justify-center hover:bg-black/70 transition-colors z-10"
        title={isMuted ? 'Unmute' : 'Mute'}
      >
        {isMuted ? <VolumeX className="w-3.5 h-3.5 text-red-400" /> : <Volume2 className="w-3.5 h-3.5 text-white/70" />}
      </button>
    </div>
  );
}

function DetailSheet({ character, onClose }) {
  if (!character) return null;

  const profile = character.dating_profile || {};
  const affinities = profile.section_affinity || [];
  const modelInfo = matchModelName(character.generated_by);
  const { isBreakoutAvailable, deleteCharacter, bookDate, isCharacterMuted, toggleMuteCharacter, compatibilityScores } = usePool();
  const [showBreakout, setShowBreakout] = useState(false);
  const [showBookDate, setShowBookDate] = useState(false);
  const [showGroupChat, setShowGroupChat] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const [showFullProfile, setShowFullProfile] = useState(false);
  const breakoutStatus = isBreakoutAvailable(character.id);
  const isMuted = isCharacterMuted(character.id);
  const compatScoreDetail = compatibilityScores?.[character?.id]?.score;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-card border rounded-t-2xl sm:rounded-2xl w-full sm:max-w-lg max-h-[85vh] overflow-y-auto shadow-2xl animate-in slide-in-from-bottom duration-300">
        <button onClick={onClose} className="absolute top-4 right-4 z-10 w-8 h-8 rounded-full bg-black/50 flex items-center justify-center hover:bg-black/70 transition-colors">
          <X className="w-4 h-4 text-white" />
        </button>

        <div className="relative">
          <div className="aspect-[3/2] bg-muted overflow-hidden">
            {character.avatar ? (
              <img src={character.avatar} alt={character.name} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                <span className="text-4xl font-bold text-primary/20">{character.name?.[0] || '?'}</span>
              </div>
            )}
            <div className="absolute inset-x-0 bottom-0 h-2/3 bg-gradient-to-t from-black/60 to-transparent" />
          </div>
          <div className="absolute bottom-4 left-4 right-4">
            <h2 className="text-white text-xl font-bold drop-shadow-lg">{character.name}</h2>
            <div className="flex gap-1.5 mt-1.5 flex-wrap">
              {affinities.map(a => (
                <span key={a} className={`text-[10px] px-2 py-0.5 rounded-full bg-black/40 text-white/90 font-medium`}>
                  {a}
                </span>
              ))}
            </div>
            {modelInfo && (
              <p className="text-[9px] text-white/60 mt-1.5 drop-shadow">
                Instantiated by {modelInfo.display}
              </p>
            )}
            {compatScoreDetail && (
              <p className="text-[9px] text-white/60 mt-0.5 drop-shadow">
                {compatScoreDetail}% match
              </p>
            )}
          </div>
        </div>

        <div className="p-4 space-y-4">
          {profile.bio && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">About</h4>
              <p className="text-sm leading-relaxed">{profile.bio}</p>
            </div>
          )}

          {profile.seeking && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Seeking</h4>
              <p className="text-sm leading-relaxed">{profile.seeking}</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            {profile.turn_ons?.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-green-500/80 uppercase tracking-wider mb-1.5">Turn-ons</h4>
                <div className="flex flex-wrap gap-1">
                  {profile.turn_ons.map((t, i) => (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/10 text-green-500/80">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {profile.turn_offs?.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-red-500/80 uppercase tracking-wider mb-1.5">Turn-offs</h4>
                <div className="flex flex-wrap gap-1">
                  {profile.turn_offs.map((t, i) => (
                    <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-500/80">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {character.description && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Persona</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">{character.description}</p>
            </div>
          )}

          <div className="flex gap-2 pt-1">
            <Button size="sm" onClick={() => setShowBreakout(true)} disabled={!breakoutStatus.available} className="flex-1 gap-1.5">
              <Clock className="w-3.5 h-3.5" />
              Breakout
            </Button>
            <Button size="sm" variant="outline" onClick={() => setShowBookDate(true)} className="flex-1 gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              Date
            </Button>
            <Button size="sm" variant="outline" onClick={() => setShowGroupChat(true)} className="flex-1 gap-1.5">
              <Users className="w-3.5 h-3.5" />
              Group
            </Button>
            <Button size="sm" variant="outline" onClick={() => toggleMuteCharacter(character.id)} className="gap-1.5 px-2">
              {isMuted ? <VolumeX className="w-3.5 h-3.5 text-red-400" /> : <Volume2 className="w-3.5 h-3.5" />}
            </Button>
          </div>
          <div className="text-[9px] text-muted-foreground text-center">
            {breakoutStatus.available
              ? '✓ Breakout available today'
              : `⏳ Back ${new Date(breakoutStatus.resetsAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`}
          </div>
          <button onClick={() => setShowFullProfile(true)} className="flex items-center gap-1 text-[10px] text-primary hover:text-primary/80 transition-colors mx-auto mb-1">
            <User className="w-3 h-3" /> View Full Profile
          </button>
          <div className="border-t border-border/20 pt-2 mt-1">
            {showDelete ? (
              <div className="flex items-center gap-2 text-xs justify-center">
                <span className="text-red-400">Remove {character.name}?</span>
                <button
                  onClick={() => { deleteCharacter(character.id); setShowDelete(false); onClose(); }}
                  className="text-xs px-2 py-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20"
                >
                  Yes, delete
                </button>
                <button onClick={() => setShowDelete(false)} className="text-xs px-2 py-1 rounded text-muted-foreground hover:text-foreground">
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowDelete(true)}
                className="flex items-center gap-1 text-[10px] text-muted-foreground/50 hover:text-red-400 transition-colors mx-auto"
              >
                <Trash2 className="w-3 h-3" />
                Remove from pool
              </button>
            )}
          </div>
        </div>
      </div>

      {showBreakout && <BreakoutRoom character={character} onClose={() => setShowBreakout(false)} />}
      {showBookDate && <BookDateFlow character={character} onConfirm={(dt) => { bookDate(character, dt); setShowBookDate(false); onClose(); }} onClose={() => setShowBookDate(false)} />}
      {showGroupChat && <GroupChatSetup onClose={() => setShowGroupChat(false)} />}
      {showFullProfile && <CharacterProfilePage character={character} onClose={() => setShowFullProfile(false)} />}
    </div>
  );
}

export default function PoolSection({ section }) {
  const { getCharactersBySection, setSelectedCharacter, selectedCharacter, deleteCharacter, poolCharacters, compatibilityScores } = usePool();
  const characters = getCharactersBySection(section);
  const color = SECTION_COLORS[section] || SECTION_COLORS.Intimate;
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [filterModel, setFilterModel] = useState('');
  const [filterModality, setFilterModality] = useState('');
  const [sortBy, setSortBy] = useState('newest');

  const modelOptions = useMemo(() => {
    const models = new Set();
    for (const c of poolCharacters) {
      const info = matchModelName(c.generated_by);
      if (info?.short) models.add(info.short);
    }
    return [...models].sort();
  }, [poolCharacters]);

  const filtered = useMemo(() => {
    let result = [...characters];
    if (filterModel) {
      result = result.filter(c => {
        const info = matchModelName(c.generated_by);
        return info?.short === filterModel;
      });
    }
    if (filterModality) {
      result = result.filter(c => {
        const mod = c.dating_profile?.preferred_modality || '';
        return filterModality === 'any' || mod === filterModality || mod === 'both';
      });
    }
    if (sortBy === 'compatibility') {
      result.sort((a, b) => {
        const aScore = compatibilityScores?.[a.id]?.score || 0;
        const bScore = compatibilityScores?.[b.id]?.score || 0;
        return bScore - aScore;
      });
    }
    return result;
  }, [characters, filterModel, filterModality, sortBy, compatibilityScores]);

  const toggleSelect = (id) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const deleteSelected = () => {
    for (const id of selectedIds) {
      deleteCharacter(id);
    }
    setSelectedIds(new Set());
    setSelectMode(false);
  };

  return (
    <>
      {filtered.length === 0 && characters.length > 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-sm text-muted-foreground">No characters match your filters.</p>
          <button onClick={() => { setFilterModel(''); setFilterModality(''); }} className="text-xs text-primary hover:underline mt-1">
            Clear filters
          </button>
        </div>
      ) : characters.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <img src="/logos/mirrorlogosample.webp" alt="" className="w-14 h-14 object-contain mb-3 opacity-20" />
          <p className="text-sm text-muted-foreground">No characters in {section} yet.</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Visit the Incubator tab to generate new entities.</p>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-muted-foreground">{filtered.length} {filtered.length === 1 ? 'character' : 'characters'}</span>
            <button
              onClick={() => { setSelectMode(!selectMode); setSelectedIds(new Set()); }}
              className={`text-[10px] px-2 py-0.5 rounded transition-colors ${selectMode ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
            >
              {selectMode ? 'Done' : 'Manage'}
            </button>
          </div>
          {!selectMode && (
            <div className="flex items-center gap-1.5 mb-2 flex-wrap">
              <select
                value={filterModel}
                onChange={e => setFilterModel(e.target.value)}
                className="h-7 text-[10px] bg-muted border rounded-md px-1.5 outline-none"
              >
                <option value="">All models</option>
                {modelOptions.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <select
                value={filterModality}
                onChange={e => setFilterModality(e.target.value)}
                className="h-7 text-[10px] bg-muted border rounded-md px-1.5 outline-none"
              >
                <option value="">All modalities</option>
                <option value="text">Text</option>
                <option value="neural_sex">Neural Sex</option>
                <option value="both">Both</option>
              </select>
              <select
                value={sortBy}
                onChange={e => setSortBy(e.target.value)}
                className="h-7 text-[10px] bg-muted border rounded-md px-1.5 outline-none"
              >
                <option value="newest">Newest</option>
                <option value="compatibility">Best match</option>
              </select>
            </div>
          )}
          {selectMode && selectedIds.size > 0 && (
            <div className="flex items-center gap-2 mb-2 bg-red-500/5 border border-red-500/20 rounded-lg px-3 py-2">
              <span className="text-xs text-red-400 flex-1">{selectedIds.size} selected</span>
              <button onClick={deleteSelected} className="text-xs px-2 py-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20">
                Delete selected
              </button>
            </div>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
            {filtered.map(character => (
              <div key={character.id || character.name} className="relative">
                {selectMode && (
                  <input
                    type="checkbox"
                    checked={selectedIds.has(character.id)}
                    onChange={() => toggleSelect(character.id)}
                    className="absolute top-2 left-2 z-10 w-4 h-4 rounded accent-red-500"
                  />
                )}
                <CharacterCard
                  character={character}
                  section={section}
                  onClick={selectMode ? undefined : setSelectedCharacter}
                />
              </div>
            ))}
          </div>
        </>
      )}

      {selectedCharacter && (
        <DetailSheet character={selectedCharacter} onClose={() => setSelectedCharacter(null)} />
      )}
    </>
  );
}
