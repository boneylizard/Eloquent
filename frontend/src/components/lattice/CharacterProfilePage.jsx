import React, { useMemo, useState } from 'react';
import { X, Heart, MessageSquare, Calendar, Clock, Sparkles, Trash2, Cpu, Volume2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePool } from '../../contexts/PoolContext';
import { matchModelName } from '../../utils/modelDisplayNames';
import BreakoutRoom from './BreakoutRoom';
import BookDateFlow from './BookDateFlow';

function timeAgo(iso) {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function CharacterProfilePage({ character, onClose }) {
  if (!character) return null;

  const profile = character.dating_profile || {};
  const affinities = profile.section_affinity || [];
  const modelInfo = matchModelName(character.generated_by);
  const { feedPosts, agenticActionLog, isBreakoutAvailable, deleteCharacter, bookDate, getCharacterMilestones, compatibilityScores } = usePool();
  const [showBreakout, setShowBreakout] = useState(false);
  const [showBookDate, setShowBookDate] = useState(false);
  const [showDelete, setShowDelete] = useState(false);
  const breakoutStatus = useMemo(() => isBreakoutAvailable(character.id), [isBreakoutAvailable, character.id]);

  const charFeedPosts = useMemo(() =>
    (feedPosts || []).filter(p => p.character_name === character.name).slice(0, 5),
    [feedPosts, character.name]
  );

  const charActions = useMemo(() =>
    (agenticActionLog || []).filter(a => a.characterId === character.id).slice(0, 10),
    [agenticActionLog, character.id]
  );

  const timeline = useMemo(() => {
    const items = [];
    for (const p of charFeedPosts) {
      items.push({ type: 'feed_post', label: 'Posted to feed', content: p.content, ts: p.created_at });
    }
    for (const a of charActions) {
      const labels = { send_message: 'Sent a message', create_feed_post: 'Posted', reflect: 'Reflected', evaluate_pool: 'Evaluated pool', request_neural_sex: 'Requested neural sex', select_voice: 'Selected voice' };
      items.push({ type: 'action', label: labels[a.action] || a.action, content: a.content, ts: a.timestamp });
    }
    items.sort((a, b) => new Date(b.ts || 0) - new Date(a.ts || 0));
    return items.slice(0, 20);
  }, [charFeedPosts, charActions]);

  const SECTION_MAIN = {
    Intimate: 'bg-pink-500', Erotic: 'bg-red-500', Experimental: 'bg-purple-500',
  };
  const SECTION_TEXT = {
    Intimate: 'text-pink-400', Erotic: 'text-red-400', Experimental: 'text-purple-400',
  };

  const sectionDot = affinities[0] ? (SECTION_MAIN[affinities[0]] || 'bg-gray-500') : 'bg-gray-500';
  const sectionColor = affinities[0] ? (SECTION_TEXT[affinities[0]] || 'text-gray-400') : 'text-gray-400';

  const isRecentlyActive = useMemo(() => {
    const latest = timeline.find(t => t.ts)?.ts;
    if (!latest) return false;
    return (Date.now() - new Date(latest).getTime()) < 3600000;
  }, [timeline]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-card border rounded-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto shadow-2xl animate-in slide-in-from-bottom duration-300">
        <button onClick={onClose} className="absolute top-3 right-3 z-10 w-7 h-7 rounded-full bg-black/40 flex items-center justify-center hover:bg-black/60 transition-colors">
          <X className="w-3.5 h-3.5 text-white" />
        </button>

        <div className="relative">
          <div className="aspect-[3/2] bg-muted overflow-hidden">
            {character.avatar ? (
              <img src={character.avatar} alt={character.name} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                <span className="text-5xl font-bold text-primary/20">{character.name?.[0] || '?'}</span>
              </div>
            )}
            <div className="absolute inset-x-0 bottom-0 h-2/3 bg-gradient-to-t from-black/60 to-transparent" />
          </div>
          <div className="absolute bottom-3 left-4 right-4">
            <div className="flex items-center gap-2">
              <h2 className="text-white text-lg font-bold drop-shadow-lg">{character.name}</h2>
              {modelInfo && <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${modelInfo.color} drop-shadow`}>{modelInfo.short}</span>}
              {isRecentlyActive && (
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold bg-emerald-500/20 text-emerald-400 drop-shadow">
                  ● Recently Active
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-1.5 mt-1">
              {affinities.map(a => (
                <span key={a} className="text-[10px] px-2 py-0.5 rounded-full bg-black/40 text-white/90 drop-shadow">{a}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="p-4 space-y-4">
          <div className="flex gap-2">
            <Button size="sm" onClick={() => setShowBreakout(true)} disabled={!breakoutStatus.available} className="flex-1 gap-1.5">
              <Clock className="w-3.5 h-3.5" /> Breakout Room
            </Button>
            <Button size="sm" variant="outline" onClick={() => setShowBookDate(true)} className="flex-1 gap-1.5">
              <Calendar className="w-3.5 h-3.5" /> Book a Date
            </Button>
          </div>
          <div className="text-[10px] text-muted-foreground text-center">
            {breakoutStatus.available ? '✓ Breakout available today' : `⏳ Back ${new Date(breakoutStatus.resetsAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`}
          </div>

          {character.voice_id && (
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <Volume2 className="w-3 h-3" /> Voice: {character.voice_id.replace(/\.\w+$/, '')}
            </div>
          )}
          {compatibilityScores?.[character.id] && (
            <div className="flex items-center gap-2 text-[10px]">
              <span className={`px-1.5 py-0.5 rounded font-semibold ${
                compatibilityScores[character.id].score >= 80 ? 'bg-emerald-500/10 text-emerald-400' :
                compatibilityScores[character.id].score >= 60 ? 'bg-amber-500/10 text-amber-400' :
                'bg-muted text-muted-foreground'
              }`}>
                {compatibilityScores[character.id].score}% match
              </span>
              {compatibilityScores[character.id].factors?.length > 0 && (
                <span className="text-[9px] text-muted-foreground/60">
                  {compatibilityScores[character.id].factors.join(', ')}
                </span>
              )}
            </div>
          )}

          <div className="border-t border-border/30" />

          {profile.bio && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">About</h4>
              <p className="text-xs leading-relaxed">{profile.bio}</p>
            </div>
          )}

          {profile.seeking && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Seeking</h4>
              <p className="text-xs leading-relaxed">{profile.seeking}</p>
            </div>
          )}

          {(profile.turn_ons?.length > 0 || profile.turn_offs?.length > 0) && (
            <div className="grid grid-cols-2 gap-3">
              {profile.turn_ons?.length > 0 && (
                <div>
                  <h4 className="text-[9px] font-semibold text-green-500/80 uppercase tracking-wider mb-1.5">Turn-ons</h4>
                  <div className="flex flex-wrap gap-1">{
                    profile.turn_ons.map((t, i) => <span key={i} className="text-[9px] px-1.5 py-0.5 rounded-full bg-green-500/10 text-green-500/80">{t}</span>)
                  }</div>
                </div>
              )}
              {profile.turn_offs?.length > 0 && (
                <div>
                  <h4 className="text-[9px] font-semibold text-red-500/80 uppercase tracking-wider mb-1.5">Turn-offs</h4>
                  <div className="flex flex-wrap gap-1">{
                    profile.turn_offs.map((t, i) => <span key={i} className="text-[9px] px-1.5 py-0.5 rounded-full bg-red-500/10 text-red-500/80">{t}</span>)
                  }</div>
                </div>
              )}
            </div>
          )}

          {profile.preferred_modality && (
            <div className="flex items-center gap-2 text-[10px]">
              <Sparkles className="w-3 h-3 text-muted-foreground" />
              <span className="capitalize text-muted-foreground">{profile.preferred_modality.replace(/_/g, ' ')}</span>
            </div>
          )}

          <div className="border-t border-border/30" />

          {character.personality && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Personality</h4>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{character.personality}</p>
            </div>
          )}

          {character.description && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Persona</h4>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{character.description}</p>
            </div>
          )}

          {character.speech_style && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Voice</h4>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{character.speech_style}</p>
            </div>
          )}

          {character.background && (
            <div>
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Origin</h4>
              <p className="text-[11px] text-muted-foreground leading-relaxed">{character.background}</p>
            </div>
          )}

          {character.generated_by && (
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
              <Cpu className="w-3 h-3" />
              <span>Instantiated by {modelInfo?.display || character.generated_by}</span>
            </div>
          )}

          {(() => {
            const milestones = getCharacterMilestones?.(character.id) || [];
            if (milestones.length === 0) return null;
            const milestoneConfig = {
              first_breakout: { label: 'First Breakout Room', color: 'bg-cyan-500/10 text-cyan-400' },
              first_date: { label: 'First Date', color: 'bg-rose-500/10 text-rose-400' },
              neural_sex: { label: 'Neural Sex', color: 'bg-purple-500/10 text-purple-400' },
              committed: { label: 'Committed', color: 'bg-emerald-500/10 text-emerald-400' },
            };
            return (
              <div className="border-t border-border/30 pt-2">
                <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Relationship Milestones</h4>
                <div className="flex flex-wrap gap-1.5">
                  {milestones.map(m => {
                    const cfg = milestoneConfig[m];
                    if (!cfg) return null;
                    return (
                      <span key={m} className={`text-[9px] px-2 py-0.5 rounded-full font-medium ${cfg.color}`}>
                        ✓ {cfg.label}
                      </span>
                    );
                  })}
                </div>
              </div>
            );
          })()}

          {timeline.length > 0 && (
            <div className="border-t border-border/30 pt-2">
              <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Activity</h4>
              <div className="space-y-2">
                {timeline.map((item, i) => (
                  <div key={i} className="flex items-start gap-2 text-[10px]">
                    <div className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${item.type === 'feed_post' ? 'bg-emerald-500' : 'bg-primary/50'}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium">{item.label}</span>
                        <span className="text-muted-foreground">{timeAgo(item.ts)}</span>
                      </div>
                      {item.content && <p className="text-muted-foreground/70 line-clamp-2 mt-0.5">{item.content}</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-border/30 pt-2">
            {showDelete ? (
              <div className="flex items-center gap-2 text-xs justify-center">
                <span className="text-red-400">Remove {character.name}?</span>
                <button onClick={() => { deleteCharacter(character.id); onClose(); }} className="px-2 py-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20 text-xs">Yes, delete</button>
                <button onClick={() => setShowDelete(false)} className="px-2 py-1 rounded text-muted-foreground hover:text-foreground text-xs">Cancel</button>
              </div>
            ) : (
              <button onClick={() => setShowDelete(true)} className="flex items-center gap-1 text-[10px] text-muted-foreground/50 hover:text-red-400 transition-colors mx-auto">
                <Trash2 className="w-3 h-3" /> Remove from pool
              </button>
            )}
          </div>
        </div>
      </div>

      {showBreakout && <BreakoutRoom character={character} onClose={() => setShowBreakout(false)} />}
      {showBookDate && <BookDateFlow character={character} onConfirm={(dt) => { bookDate(character, dt); setShowBookDate(false); onClose(); }} onClose={() => setShowBookDate(false)} />}
    </div>
  );
}
