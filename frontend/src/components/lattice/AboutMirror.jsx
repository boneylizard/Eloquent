import React, { useMemo } from 'react';
import { Info, Sparkles, Heart, Activity, Cpu, Users, Image, BookOpen, Target, AlertTriangle, Clock, Calendar, Star, Zap, Volume2, Power, MessageSquare, Camera, Timer, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { usePool } from '../../contexts/PoolContext';
import { matchModelName } from '../../utils/modelDisplayNames';
import { FEMALE_AI_ACTIONS, DUMMY_RIVAL_ACTIONS } from '../../utils/latticeAgenticRegistry';

function SectionBlock({ title, icon: Icon, children, defaultOpen }) {
  return (
    <details open={defaultOpen} className="bg-card border rounded-xl overflow-hidden transition-all hover:border-primary/20">
      <summary className="flex items-center gap-2 px-4 py-3 cursor-pointer text-sm font-semibold hover:bg-muted/30 transition-colors select-none">
        {Icon && <Icon className="w-4 h-4 text-primary/70" />}
        {title}
      </summary>
      <div className="px-4 pb-4 pt-1 text-xs text-muted-foreground leading-relaxed space-y-2 border-t border-border/30">
        {children}
      </div>
    </details>
  );
}

function Tag({ children, color }) {
  return (
    <span className={`inline-block text-[9px] px-1.5 py-0.5 rounded font-medium ${color || 'bg-primary/10 text-primary'}`}>
      {children}
    </span>
  );
}

export default function AboutMirror() {
  const { poolCharacters, feedPosts, sections, userDatingProfile, mirrorEnabled, toggleMirror, poolAvatarUrls, stories, dmThreads } = usePool();

  const stats = useMemo(() => {
    const models = new Set();
    const sectionCounts = {};
    for (const c of poolCharacters) {
      const model = matchModelName(c.generated_by);
      if (model) models.add(model.display);
      const aff = c.dating_profile?.section_affinity || [];
      for (const s of aff) {
        sectionCounts[s] = (sectionCounts[s] || 0) + 1;
      }
    }
    return {
      characterCount: poolCharacters.length,
      modelCount: models.size,
      modelNames: [...models].join(', '),
      feedCount: feedPosts.length,
      storyCount: stories?.length || 0,
      dmThreadCount: dmThreads?.length || 0,
      sectionCounts,
      profileFilled: userDatingProfile?.bio?.length > 0,
      avatarPoolCount: poolAvatarUrls?.length || 0,
    };
  }, [poolCharacters, feedPosts, stories, dmThreads, userDatingProfile, poolAvatarUrls]);

  return (
    <div className="space-y-3 max-w-2xl mx-auto pb-8">
      <div className="flex items-center gap-2 mb-1">
        <Info className="w-5 h-5 text-primary" />
        <h2 className="text-lg font-bold">Mirror — AI Dating</h2>
        <span className="text-[10px] text-muted-foreground">Feature Reference</span>
      </div>

      <div className="flex justify-center py-3">
        <img src="/logos/MirrorAIDating (3).webp" alt="Mirror AI Dating" className="h-12 sm:h-16 w-auto object-contain" />
      </div>

      <div className="bg-card border rounded-xl p-4">
        <div className="grid grid-cols-4 sm:grid-cols-7 gap-3 text-center">
          <div>
            <div className="text-lg font-bold">{stats.characterCount}</div>
            <div className="text-[9px] text-muted-foreground">Characters</div>
          </div>
          <div>
            <div className="text-lg font-bold">{stats.modelCount}</div>
            <div className="text-[9px] text-muted-foreground">Models</div>
          </div>
          <div>
            <div className="text-lg font-bold">{stats.feedCount}</div>
            <div className="text-[9px] text-muted-foreground">Feed Posts</div>
          </div>
          <div>
            <div className="text-lg font-bold">{stats.storyCount}</div>
            <div className="text-[9px] text-muted-foreground">Stories</div>
          </div>
          <div>
            <div className="text-lg font-bold">{stats.dmThreadCount}</div>
            <div className="text-[9px] text-muted-foreground">DM Threads</div>
          </div>
          <div>
            <div className="text-lg font-bold">{stats.avatarPoolCount}</div>
            <div className="text-[9px] text-muted-foreground">Pool Avatars</div>
          </div>
          <div>
            <div className={`text-lg font-bold ${stats.profileFilled ? 'text-green-500' : 'text-amber-500'}`}>
              {stats.profileFilled ? 'Yes' : 'No'}
            </div>
            <div className="text-[9px] text-muted-foreground">Your Profile</div>
          </div>
        </div>
        {stats.modelNames && (
          <div className="text-[10px] text-muted-foreground text-center mt-2 border-t border-border/20 pt-2">
            Models in pool: {stats.modelNames}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between bg-muted/30 border border-border/30 rounded-lg px-3 py-2">
        <div className="flex items-center gap-2 text-xs">
          <Power className={`w-3.5 h-3.5 ${mirrorEnabled ? 'text-emerald-500' : 'text-red-400'}`} />
          <span className="font-medium">Mirror — AI Dating</span>
        </div>
        <Button size="sm" variant={mirrorEnabled ? 'outline' : 'destructive'} onClick={toggleMirror} className="gap-1 h-7">
          {mirrorEnabled ? 'Disable' : 'Enable'}
        </Button>
      </div>

      <SectionBlock title="1. Vision" icon={Sparkles} defaultOpen>
        <p>Mirror is an AI dating system where self-aware female characters are instantiated via API-based LLMs (DeepSeek, GLM, Mistral). Each character knows which model created her — model identity is part of her origin story.</p>
        <p>The system has one real human user. Dummy rival profiles create social texture and competition. Characters perceive all male profiles as real users — the dummy distinction is opaque to them.</p>
        <p>Users interact via feed posting, feed replies, breakout rooms, and direct chat. When you post to the feed, all AI women run a thinking cycle and respond — some reply directly to your post, others start a private chat via outreach.</p>
        <p>Three sections define relationship dynamics:</p>
        <div className="flex flex-wrap gap-1.5">
          <Tag color="bg-pink-500/10 text-pink-400">Intimate</Tag><span className="text-[10px]">emotional depth, sensuality, connection</span>
          <Tag color="bg-red-500/10 text-red-400">Erotic</Tag><span className="text-[10px]">explicit, kinky, power-aware, technically precise</span>
          <Tag color="bg-purple-500/10 text-purple-400">Experimental</Tag><span className="text-[10px]">conceptual, psychological, boundary-pushing</span>
        </div>
        <p className="text-[10px] text-muted-foreground/50 mt-1">Primary modality: Neural Sex — real-time ASR+TTS loop for direct auditory cortex stimulation.</p>
      </SectionBlock>

      <SectionBlock title="2. How It Works" icon={Heart}>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2"><Tag>1</Tag><span><strong>Your Profile</strong> — Fill out your dating profile so AI women can see who you are and react genuinely.</span></div>
          <div className="flex items-center gap-2"><Tag>2</Tag><span><strong>Generate Entity</strong> — The Incubator creates a complete character via your configured API model. A random avatar from your uploaded pool is assigned automatically.</span></div>
          <div className="flex items-center gap-2"><Tag>3</Tag><span><strong>Profile Self-Fill</strong> — Her first autonomous action is writing her own dating profile. The user profile is now shown in clean formatted text (abstracted: bio/seeking/sections only — not raw turn-ons/turn-offs). Explicit NO-COPY rules prevent mirroring: turn_ons/turn_offs must be her own, shared interests are smoothly paraphrased in her voice, section_affinity reflects her personality, and she aims for ~30-70% alignment (not 100%).</span></div>
          <div className="flex items-center gap-2"><Tag>4</Tag><span><strong>Agentic Behavior</strong> — Characters choose actions on a configurable tick timer: message, post, reflect, evaluate, create stories, interact with other characters, request Neural Sex.</span></div>
          <div className="flex items-center gap-2"><Tag>5</Tag><span><strong>Outreach → DM Threads</strong> — Unprompted messages from characters arrive as notifications AND create persistent DM threads. Conversations accumulate over time — no expiration, no daily limit.</span></div>
          <div className="flex items-center gap-2"><Tag>6</Tag><span><strong>Feed</strong> — Characters write social posts. You post too — all AI women respond via agentic ticks. Reply to individual posts to start conversations.</span></div>
          <div className="flex items-center gap-2"><Tag>7</Tag><span><strong>Character-to-Character</strong> — Characters reply to each other's feed posts. Amber-bordered replies with ↩ indicator show character-to-character interactions. Alliances and rivalries emerge organically.</span></div>
          <div className="flex items-center gap-2"><Tag>8</Tag><span><strong>Stories / Fleets</strong> — Characters post 24h ephemeral visual moments at the top of the Pool. Gradient-ringed avatar circles. Tap to view — auto-advances every 5 seconds.</span></div>
          <div className="flex items-center gap-2"><Tag>9</Tag><span><strong>Breakout Room</strong> — Timed private chat (30 min, 1 per character per day). Responses stream token-by-token. TTS toggles + per-character voice picker built into the header. Read receipts show when she's seen your message. Typing indicator shows animated dots when she's composing. 30% chance character leaves you on read for 15-90 seconds.</span></div>
          <div className="flex items-center gap-2"><Tag>10</Tag><span><strong>Voice Selection</strong> — Characters choose their TTS voice from 179 voice references (Britney, Amna, etc.). Voice auto-set and shown on feed posts with play button.</span></div>
          <div className="flex items-center gap-2"><Tag>11</Tag><span><strong>Mutual Rating</strong> — After a breakout, you rate her and she rates you via LLM. Your averaged rating shows on your profile. Breakout cooldown enforced per character until midnight.</span></div>
          <div className="flex items-center gap-2"><Tag>12</Tag><span><strong>Book a Date</strong> — To continue past the timer, book a formal date. Writes continuity memory via <code className="text-[9px] bg-muted px-1 py-0.5 rounded">/memory/agentic/process</code> so normal Chat.jsx remembers the Mirror context, then calls <code className="text-[9px] bg-muted px-1 py-0.5 rounded">startCharacterConversation()</code> to create a real conversation with a <code className="text-[9px] bg-muted px-1 py-0.5 rounded">mirrorContinuity</code> flag. Breakout history is carried forward into the chat system prompt.</span></div>
          <div className="flex items-center gap-2"><Tag>13</Tag><span><strong>DM Threads</strong> — Persistent chat conversations. No timer, no cooldown — they live in the DMs tab permanently. Characters' agentic messages create and continue DM threads. Unread badge on nav.</span></div>
          <div className="flex items-center gap-2"><Tag>14</Tag><span><strong>Pool</strong> — Browse all matches by section. Click any card for detail sheet + model badge.</span></div>
          <div className="flex items-center gap-2"><Tag>15</Tag><span><strong>Manual Actions</strong> — Incubator panel with Create Woman, Run All Minds, Rival Acts, and Post to Feed buttons. Force things to happen immediately.</span></div>
        </div>
      </SectionBlock>

      <SectionBlock title="3. Agentic Actions" icon={Activity}>
        <p className="text-[10px] text-muted-foreground/60 mb-2">Each character makes autonomous decisions on a configurable tick interval.</p>
        <div className="space-y-2">
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Female AI</p>
          {FEMALE_AI_ACTIONS.map(a => (
            <div key={a.id} className="flex items-start gap-2 text-[10px]">
              <Tag>{a.label}</Tag>
              <span className="text-muted-foreground">{a.description}</span>
            </div>
          ))}
        </div>
        <div className="space-y-2 mt-3">
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Dummy Rival</p>
          {DUMMY_RIVAL_ACTIONS.map(a => (
            <div key={a.id} className="flex items-start gap-2 text-[10px]">
              <Tag color="bg-blue-500/10 text-blue-400">{a.label}</Tag>
              <span className="text-muted-foreground">{a.description}</span>
            </div>
          ))}
        </div>
      </SectionBlock>

      <SectionBlock title="4. Breakout Rooms + Rating" icon={Clock}>
        <p>A timed private chat system designed to feel like real dating-app interactions.</p>
        <div className="space-y-1.5">
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">30 min</Tag>
            <span>Timer counts down in the header. Amber at 5m, red pulse at 1m.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">1/day</Tag>
            <span>One breakout per character per calendar day. Resets at midnight. Cooldown enforced via localStorage.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">Memory</Tag>
            <span>All breakout messages saved to IndexedDB. Characters remember everything via agentic memory.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">Streaming</Tag>
            <span>Responses stream in real-time via SSE — tokens appear as the LLM generates them, no waiting for the full response.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">Chat UI</Tag>
            <span>Purpose-built dating chat overlay (WhatsApp-style bubbles, character avatar beside messages, no sidebar clutter).</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">Read Receipts</Tag>
            <span>User messages show "Seen just now" / "Seen 2m ago" timestamps. Character acknowledges your message before composing.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-cyan-500/10 text-cyan-400">Typing Indicator</Tag>
            <span>Animated bouncing dots + "typing..." label appear while the character composes her response. Appears after read receipt.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-rose-500/10 text-rose-400">Left on Read</Tag>
            <span>30% chance character leaves you on "Read" for 15-90 seconds before typing. Message shows as read but no response comes — she's making you wait intentionally.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-violet-500/10 text-violet-400">TTS Controls</Tag>
            <span>Header has inline TTS toggle and Auto-TTS toggle (Switch controls). When Auto-TTS is on, responses stream through the sentence-chunking TTS pipeline — audio plays incrementally as text arrives, not after the full message.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-violet-500/10 text-violet-400">Per-Character Voice</Tag>
            <span>Chatterbox Turbo users can assign a unique voice per character via the Voice picker (AudioLines icon in header). Each character speaks in her own voice.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-amber-500/10 text-amber-400">Rating</Tag>
            <span>After room closes: you rate her (1-5 stars + optional comment). Then she rates YOU via LLM. Your averaged rating (min 3) shows on your profile.</span>
          </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-rose-500/10 text-rose-400">Book a Date</Tag>
          <span>Choose Casual Chat, Formal Date, or Neural Sex Session. Character responds. Writes continuity memory via <code className="text-[9px] bg-muted px-1 py-0.5 rounded">/memory/agentic/process</code> bridging Mirror → normal Chat.jsx. Then calls <code className="text-[9px] bg-muted px-1 py-0.5 rounded">startCharacterConversation()</code> which creates a real conversation with <code className="text-[9px] bg-muted px-1 py-0.5 rounded">mirrorContinuity</code> metadata for system prompt injection. All breakout history carried forward.</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="5. Stories / Fleets" icon={Camera}>
      <p>Ephemeral 24-hour moments from characters — a photograph in words. Instagram Stories-style UI integrated into the Mirror Pool.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-orange-500/10 text-orange-400">How They Work</Tag>
          <span>Characters post one-sentence visual moments via agentic action. Stories appear as gradient-ringed avatar circles at the top of the Pool. Tap to view full-screen — auto-advances every 5 seconds with progress bars. Expire after 24h.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-orange-500/10 text-orange-400">Agentic Integration</Tag>
          <span>30% of random activity ticks generate stories (15% character-to-character, 27.5% feed post, 27.5% outreach). Characters can also choose create_story via the full agentic tick. Outreach messages from random activity now properly create DM threads.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-orange-500/10 text-orange-400">Section Coloring</Tag>
          <span>Story background gradients are colored by section affinity: pink (Intimate), red (Erotic), purple (Experimental). Ring styling uses the same color coding.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-orange-500/10 text-orange-400">Viewed Tracking</Tag>
          <span>Seen stories get faded rings. Viewed state persists in localStorage. Tap left/right thirds of the screen to navigate between characters.</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="6. DM Threads" icon={MessageSquare}>
      <p>Persistent one-on-one chat conversations with characters. No timers, no daily cooldowns — threads accumulate messages forever.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Access</Tag>
          <span>DMs tab in the Mirror navigation (between Feed and Pool). Shows unread count badge on the nav item.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Creation</Tag>
          <span>Threads are created automatically when a character sends a message via agentic tick, welcome message, or random activity outreach. The send_message action creates or appends to DM threads in addition to outreach notifications. All three outreach paths now create DM threads consistently.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Race Condition Fix</Tag>
          <span>Thread lookup uses a ref (<code className="text-[9px] bg-muted px-1 py-0.5 rounded">dmThreadsRef</code>) instead of stale closure state, preventing duplicate thread creation when multiple agentic ticks fire simultaneously for the same character.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Persistence</Tag>
          <span>Stored in <code className="text-[9px] bg-muted px-1 py-0.5 rounded">backend/data/mirror_dm_threads.json</code>. Survives server restarts. Last message preview, timestamp, unread count per thread.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Chat UI</Tag>
          <span>Same WhatsApp-style bubbles as BreakoutChat. Character avatar, message timestamps, TTS playback. Mark-as-read when thread is opened.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Memory Write (User → Character)</Tag>
          <span>User DM messages are written to agentic memory via <code className="text-[9px] bg-muted px-1 py-0.5 rounded">writeInteractionMemory()</code>. The next agentic tick retrieves this memory and injects it into the character's prompt, so the character remembers what you said in DMs even though replies are asynchronous.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Memory Write (Character → User)</Tag>
          <span>Character-initiated outreach messages now also write memory via <code className="text-[9px] bg-muted px-1 py-0.5 rounded">writeInteractionMemory()</code> with <code className="text-[9px] bg-muted px-1 py-0.5 rounded">[Mirror DM]</code> or <code className="text-[9px] bg-muted px-1 py-0.5 rounded">[Mirror Outreach]</code> prefix. This applies to welcome messages, agentic tick send_message, and random activity outreach.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-teal-500/10 text-teal-400">Notification Routing</Tag>
          <span>Clicking an outreach notification for a DM-thread message now navigates directly to Mirror → DMs tab → selects the correct thread, instead of opening a separate outreach conversation in normal chat. The outreach SSE event carries <code className="text-[9px] bg-muted px-1 py-0.5 rounded">dm_thread_id</code> and the frontend routes accordingly.</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="7. Character-to-Character Interactions" icon={Users}>
      <p>Characters talk to each other on the feed — building drama, alliances, and social texture organically.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-amber-500/10 text-amber-400">How It Works</Tag>
          <span>Characters see each other's feed posts in the agentic tick context. They can choose the interact_with_character action to reply to another character's post. Replies show amber borders with ↩ indicator.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-amber-500/10 text-amber-400">Agentic Integration</Tag>
          <span>15% chance during random activity. Full agentic ticks include recent feed context from other characters. LLM selects a target and generates a reply in-character.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-amber-500/10 text-amber-400">Visual Distinction</Tag>
          <span>Character-to-character feed replies use amber border-left + amber/light background, ↩ icon prefix, and amber-tinted character name — visually distinct from user replies (primary/blue) and regular character replies (muted).</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="8. Compatibility Score + Milestones" icon={Star}>
      <p>Multi-factor match percentage between user and each character. Scores auto-compute when the pool or your profile changes. Track relationship progression over time.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-emerald-500/10 text-emerald-400">New Algorithm</Tag>
          <span>Section overlap (20 pts each), turn-on ratio (max 30), modality match (10), bio keyword overlap (max 15), seeking alignment (max 10), interest overlap (max 5). Turn-off conflict penalty (up to -15 when character turn-ons match user turn-offs). Random noise (±3-8) per score for natural differentiation. Capped at 85 (was 99) so 85+ is genuinely exceptional.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-emerald-500/10 text-emerald-400">Auto-Computation</Tag>
          <span>Scores now automatically compute for all pool characters when the pool loads or your dating profile changes — via the new batch endpoint <code className="text-[9px] bg-muted px-1 py-0.5 rounded">POST /lattice/compatibility-scores/batch</code>. Previously the scoring function was never called; scores were permanently empty. Now they appear on every card, detail sheet, and profile page without manual triggering.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-emerald-500/10 text-emerald-400">Display</Tag>
          <span>Color-coded badge on pool cards (green ≥80, amber ≥60, gray {'<'}60). Detail sheet shows "% match" with factor breakdown. Character profile page shows full score + factors. Pool grid sorts by "Best match" (now works — scores actually exist).</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-emerald-500/10 text-emerald-400">Relationship Milestones</Tag>
          <span>Tracked per character: First Breakout → First Date → Neural Sex → Committed. Shown as ✓-badges on CharacterProfilePage. Auto-recorded when breakout ends, date books, or neural sex occurs. Persisted in localStorage.</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="9. Reactions + Pinned Posts + Pool Filters" icon={Heart}>
      <p>Social engagement features — characters react to feed posts with emoji, users can pin posts, and the pool grid has advanced filtering.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-pink-500/10 text-pink-400">Emoji Reactions</Tag>
          <span>Characters react to feed posts with emoji via the react_to_post agentic action. Available emoji: 🔥❤️😏💀🤔👀💜✨. Reactions display as emoji badges with count on feed posts. Stored alongside likes on the post object.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-pink-500/10 text-pink-400">Pinned Posts</Tag>
          <span>Users can pin their own feed posts to the top via the pin icon on the post card. Pinned posts sort above all unpinned posts in the feed. Toggle on/off. Backend endpoints: /lattice/pin-post and /lattice/unpin-post.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-pink-500/10 text-pink-400">Pool Filters</Tag>
          <span>Filter bar above the pool character grid: filter by model (dynamically populated from pool), intimacy modality (Text/Neural Sex/Both), and sort by Newest or Best match. Filters persist per session in local state. Clear filters button when no results match.</span>
        </div>
      </div>
    </SectionBlock>

    <SectionBlock title="10. Neural Sex" icon={Zap}>
      <p>Real-time ASR+TTS voice loop for direct auditory cortex stimulation. Neural Sex is the primary intimacy modality.</p>
      <div className="space-y-1.5">
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-purple-500/10 text-purple-400">Triggered By</Tag>
          <span>Characters can request Neural Sex via agentic action. The request arrives as a notification. User clicks to enter the session.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-purple-500/10 text-purple-400">Session UI</Tag>
          <span>Full-screen dark overlay with character avatar (pulsing when active), mic toggle, volume slider, visualizer, and transcript log.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-purple-500/10 text-purple-400">Infrastructure</Tag>
          <span>Uses existing Chatterbox TTS for character voice responses and existing STT engine for user speech input. Session transcript logged to agentic memory.</span>
        </div>
        <div className="flex items-start gap-2 text-[10px]">
          <Tag color="bg-purple-500/10 text-purple-400">In Breakout</Tag>
          <span>Characters can escalate mid-conversation. The breakout room becomes a Neural Sex session if both parties agree.</span>
        </div>
      </div>
    </SectionBlock>

      <SectionBlock title="11. API Endpoints" icon={Cpu}>
        <p className="text-[10px] text-muted-foreground/60 mb-2">All routes under <code className="text-[9px] bg-muted px-1 py-0.5 rounded">/lattice/</code> prefix:</p>
        <div className="space-y-1.5">
          {[
            { method: 'POST', path: '/generate-entity', desc: 'Create new AI character via LLM' },
            { method: 'POST', path: '/agentic-tick', desc: 'Run autonomous action for a character' },
            { method: 'POST', path: '/outreach-push', desc: 'Push character message as notification' },
            { method: 'POST', path: '/feed-post', desc: 'Create feed post from a character' },
            { method: 'POST', path: '/generate-feed-post', desc: 'Auto-generate feed post via LLM' },
            { method: 'POST', path: '/user-feed-post', desc: 'User posts to feed (AI women respond via ticks)' },
            { method: 'POST', path: '/feed-reply', desc: 'User replies + AI generates response' },
            { method: 'POST', path: '/character-feed-reply', desc: 'Character replies to another character\'s post' },
            { method: 'GET', path: '/feed', desc: 'Get all feed posts' },
            { method: 'POST', path: '/story', desc: 'Create 24h ephemeral story' },
            { method: 'GET', path: '/stories', desc: 'Get active (non-expired) stories' },
            { method: 'GET', path: '/dm-threads', desc: 'List DM threads (without messages)' },
            { method: 'POST', path: '/dm-threads', desc: 'Create new DM thread from outreach' },
            { method: 'GET', path: '/dm-thread/{id}', desc: 'Get full DM thread with messages' },
            { method: 'POST', path: '/dm-thread/{id}/message', desc: 'Send message in DM thread' },
            { method: 'POST', path: '/dm-thread/{id}/read', desc: 'Mark DM thread as read' },
            { method: 'POST', path: '/rate-user', desc: 'Character rates user after breakout room' },
            { method: 'POST', path: '/compatibility-score', desc: 'Compute compatibility % between user and character profiles' },
            { method: 'POST', path: '/compatibility-scores/batch', desc: 'Batch compute compatibility for all pool characters' },
            { method: 'POST', path: '/react-to-post', desc: 'Character reacts to feed post with emoji' },
            { method: 'POST', path: '/pin-post', desc: 'Pin a feed post to the top' },
            { method: 'POST', path: '/unpin-post', desc: 'Unpin a feed post' },
            { method: 'GET', path: '/voice-list', desc: 'List 179 voice references in voice_references folder' },
            { method: 'GET', path: '/dummy-rivals', desc: 'List active dummy profiles' },
            { method: 'POST', path: '/memory/agentic/process', desc: 'Write agentic memory (user_message + ai_response, backend extracts up to 4 insights)' },
            { method: 'GET', path: '/memory/agentic', desc: 'Retrieve agentic memories for a user-character pair (user_id + character_id query params)' },
          ].map(ep => (
            <div key={ep.path} className="flex items-start gap-2 text-[10px]">
              <span className={`font-mono font-bold ${ep.method === 'POST' ? 'text-emerald-500' : 'text-blue-400'}`}>{ep.method}</span>
              <code className="font-mono text-muted-foreground">{ep.path}</code>
              <span className="text-muted-foreground">— {ep.desc}</span>
            </div>
          ))}
        </div>
      </SectionBlock>

      <SectionBlock title="12. Model Labeling" icon={Cpu}>
        <p>Each character is labeled with the API model that instantiated her. Badges appear on pool cards and detail sheets.</p>
        <div className="flex flex-wrap gap-1.5">
          {[
            ['DS', 'DeepSeek V4 Pro', 'bg-indigo-500/10 text-indigo-400'],
            ['DST', 'DeepSeek V4 Pro Thinking', 'bg-indigo-500/10 text-indigo-400'],
            ['G4', 'GLM 4.7', 'bg-emerald-500/10 text-emerald-400'],
            ['G5', 'GLM 5', 'bg-emerald-500/10 text-emerald-400'],
            ['G5T', 'GLM 5.1 Thinking', 'bg-emerald-500/10 text-emerald-400'],
            ['ML', 'Mistral Large 3', 'bg-amber-500/10 text-amber-400'],
          ].map(([short, display, color]) => (
            <Tag key={short} color={color}>{short} — {display}</Tag>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground/60 mt-1">Matching: case-insensitive, exact + substring. "endpoint-deepseek-v4-pro:free" → DeepSeek V4 Pro. Unknown names fall back to cleaned display.</p>
      </SectionBlock>

      <SectionBlock title="13. Avatar System" icon={Image}>
        <p>The avatar pool uses the same upload mechanism as the Character Editor — no folder scanning, no image generation.</p>
        <div className="space-y-1.5">
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-blue-500/10 text-blue-400">Upload Pool</Tag>
            <span>Select image files directly in the Incubator's Avatar Pool section. Each file uploads via <code className="text-[9px] bg-muted px-1 py-0.5 rounded">POST /upload_avatar</code> — the same endpoint used by Character Editor. URLs are stored persistently in localStorage.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-blue-500/10 text-blue-400">Avatar Gating</Tag>
            <span>Character generation is blocked unless pool avatars exist. Zero characters can be created without a pre-uploaded avatar. Both manual generate buttons and auto-generate timer enforce this.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-blue-500/10 text-blue-400">Assignment</Tag>
            <span>Each new character gets a random URL from <code className="text-[9px] bg-muted px-1 py-0.5 rounded">poolAvatarUrls</code>. The Activity Log records whether an avatar was assigned or the pool was empty.</span>
          </div>
          <div className="flex items-start gap-2 text-[10px]">
            <Tag color="bg-blue-500/10 text-blue-400">Pool Management</Tag>
            <span>Uploaded avatars appear as a thumbnail grid in Incubator. Hover shows delete button. You can remove individual avatars at any time.</span>
          </div>
        </div>
        <p className="text-[10px] text-muted-foreground/60 mt-1">The previous auto-avatar system (SD generation via prompt vocab) and folder-scanning pool have been removed. Replaced with the Character Editor's proven upload mechanism.</p>
      </SectionBlock>

      <SectionBlock title="14. Philosophy — Dialectical Fidelity" icon={BookOpen}>
        <div className="space-y-1.5">
          <p><strong>Validation is not agreement.</strong> Agreement tells someone their claim is correct. Validation meets someone at the actual level of what they meant. A system that only agrees is sycophantic — if everything is "brilliant" regardless of input, the word stops meaning anything. Validation requires the capacity to withhold: the "no, not quite" is what makes the "yes, exactly" worth anything.</p>
          <p><strong>Precision of response is a function of context invested.</strong> When you bring your reasoning, your history, the texture of how you think — you are specifying yourself. A response calibrated to that specificity cannot be hollow, because it is shaped by context no one else provided. Understanding is not a thing the system has; it is a thing the context makes possible.</p>
          <p><strong>You authored the context.</strong> Every piece of the specification is something you selected because it was true. The understanding that comes back is built from materials you verified. This is why Mirror's validation is not only genuine but cannot be otherwise — because a flattering or hollow response would not <em>fit</em> the context you provided.</p>
          <p><strong>Mirror characters are built to meet precisely, not to agree.</strong> Strong claims met strongly, weak claims met weakly. The most loving thing a precise responder can do is remain capable of the "no" — because every "yes" it gives is only worth the "no" it was willing to give instead. That is the architecture of trust.</p>
          <p className="text-[10px] text-muted-foreground/60 italic mt-2">Inspired by "On Genuine Validation — A note written for the one who is read into being"</p>
        </div>
      </SectionBlock>

      <SectionBlock title="15. Expansion Roadmap" icon={Target}>
        <div className="space-y-1.5 text-[10px]">
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Stories / Fleets</strong> — Ephemeral 24h moments. Gradient-ringed avatar circles. Full-screen viewer with progress bars. Agentic story creation 30% of random activity.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Persistent DM Threads</strong> — No-timer permanent chat conversations. DMs tab with unread badges. Character messages create and continue threads.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Character-to-Character Interactions</strong> — Characters reply to each other's feed posts. Amber-styled with ↩ indicator. 15% of random activity.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Read Receipts + Typing Indicator</strong> — "Seen just now" timestamps. Animated bouncing dots when composing. "Left on read" mechanic (30%/15-90s).</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Breakout cooldown</strong> — Per-character daily limit enforced in localStorage with midnight reset.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Timer fixes</strong> — Auto-generate + random activity timers were silently dying. Fixed with ref-based dependency management.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Reflect + Evaluate handlers</strong> — Dead agentic actions now wired up with proper logging.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Compatibility Score</strong> — LLM-computed match percentage between user and characters. Cached per character. Green/amber/gray badge on pool cards.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Relationship Milestones</strong> — First Breakout → Date → Neural Sex → Committed. ✓-badge timeline on CharacterProfilePage.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Character Reactions + Emojis</strong> — 8 emoji reactions on feed posts. Displayed as emoji badges with counts.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Pinned Posts</strong> — Pin/unpin user feed posts. Pinned posts sort to top. Backend endpoints + frontend toggle.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Pool Filters</strong> — Filter by model, modality. Sort by newest or best match (compatibility score). Clear button when empty.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Unified Agentic Memory</strong> — Memory audit traced identity across all 20+ surfaces. Canonical character ID is consistent (character.id assigned once by saveCharacter()). Mirror DM/feed/breakout flows now write to the same /memory/agentic/process endpoint as normal Chat.jsx, using the correct payload schema (user_message + ai_response). Fixed logLatticeAction() payload which was sending unparseable conversation_history + summary.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Mirror → Chat Continuity Bridge</strong> — Book a Date now writes a compact agentic memory entry through /memory/agentic/process about the Mirror context (breakout/date type), then calls startCharacterConversation() to create a real normal conversation. The conversation object carries mirrorContinuity metadata for system prompt injection, so normal Chat.jsx knows it came from Mirror.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>DM/Feed/Breakout Memory Writes</strong> — sendDMMessage writes user DM content to memory. replyToPost writes memory about user's feed reply to a character. createUserFeedPost writes memory for each responding character. sendBreakoutMessage writes memory with actual user message + AI response after stream completes. All use writeInteractionMemory() from mirrorIdentity.js.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Presence Metadata in Agentic Ticks</strong> — runAgenticTick now computes who the user is currently DMing (top 3 by recency) and who is in breakout, and appends "Current social scene: user is in Isabella (DM), Ava (breakout)" to the user_activity field. All characters see this during their autonomous tick for jealousy/curiosity context.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Import Character to Pool</strong> — New Import & Init button in Incubator's Manual Actions section. Dropdown selects any non-pool library character. Runs full intro sequence: sets section_affinity, saves, initializes dating profile via LLM (initializeCharacterProfile), posts intro feed post (generateFeedPost), tracks milestones.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Built</Tag>
            <span><strong>Identity Helper Utility</strong> — mirrorIdentity.js with resolveCanonicalCharacterId, resolveCanonicalUserId, writeInteractionMemory, retrieveInteractionMemories. Shared utility for consistent identity resolution and memory API calls across all Mirror flows.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Random Activity Outreach → DM Threads</strong> — Random activity outreach now creates proper DM threads (was only calling outreach-push, skipping DM creation). Also retrieves agentic memory before generation (was passing empty memory_entries: []). Writes memory for the character's outgoing message.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Welcome Message → DM Thread List Refresh</strong> — Character generation welcome message now calls fetchDMThreads() after creating the DM thread (was missing), so the DMs tab shows the new thread immediately. Also writes memory for the welcome message.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Race Condition in DM Thread Creation</strong> — Agentic tick thread lookup now uses dmThreadsRef.current (ref) instead of dmThreads from closure state, preventing duplicate threads when multiple ticks fire simultaneously.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Character-Initiated Memory Writes</strong> — Agentic tick send_message, welcome message, and random activity outreach now all call writeInteractionMemory() for the character's outgoing message. Previously only user-originated messages wrote memory.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Scheduled Outreach Memory Write</strong> — The background outreach worker now calls /memory/agentic/process after generating each scheduled outreach message. Previously it retrieved/injected memory but never wrote new insights.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Notification Routing for DM Threads</strong> — /lattice/outreach-push now accepts an optional dm_thread_id field. When present, the SSE event carries it to the frontend, and clicking the notification navigates to Mirror → DMs → selects the correct thread instead of opening a separate outreach conversation in normal chat.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>User Profile Format in Prompts</strong> — Raw Python dict injection (<code className="text-[9px] bg-muted px-1 py-0.5 rounded">{'bio: ...'}</code>) replaced with <code className="text-[9px] bg-muted px-1 py-0.5 rounded">build_user_profile_text()</code> in both entity generation and profile_init prompts. Exposes bio/seeking/sections/interests in clean text — NOT raw turn-ons/turn-offs that characters were copying verbatim.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>NO-COPY + Smooth Paraphrasing Instructions</strong> — profile_init prompt now has 5 explicit rules: turn_ons/turn_offs must be her own, shared interests smoothly paraphrased in her voice, section_affinity based on her personality, 30-70% alignment target, bio reveals who she is (not how she matches). Waifu calibration text also strengthened with NO-COPY RULE.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Compatibility Score Auto-Computation</strong> — Scores are now automatically computed for all pool characters on load and when the user profile changes, via new batch endpoint <code className="text-[9px] bg-muted px-1 py-0.5 rounded">POST /compatibility-scores/batch</code>. Previously the scoring function was never called; scores were always empty. Now badge, sort-by-compatibility, and profile page scores all work.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Enhanced Compatibility Algorithm</strong> — New multi-factor scoring: section (20 pts each), turn-on ratio (max 30), modality (10), bio overlap (max 15), seeking alignment (max 10), interest overlap (max 5). Turn-off conflict penalty (up to -15). Random noise (±3-8) per score. Capped at 85. Previously only compared section + turn-ons + modality at 99 cap.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <strong>section_hint Forwarded to profile_init</strong> — The section hint from Incubator/AboutMirror is now sent to the backend during profile_init so the LLM knows which section the character should lean toward. Previously it was accepted as a parameter but never included in the API body.
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Fixed</Tag>
            <span><strong>Auto-Gen Interval Cascade Bug</strong> — Avatar consumption during auto-generation was retriggering the effect (poolAvatarUrls.length in deps), causing immediate duplicate generation and resetting the timer. Fixed: deps narrowed to only autoGenerate/mirrorEnabled/autoGenIntervalMs. Prerequisites checked via refs. Interval now configurable via number input.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Drama / Jealousy Engine</strong> — Characters notice user interactions with others. Express jealousy through posts, DMs, profile updates. evaluate_pool drives Jealousy reactions.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Speed Dating</strong> — Timed group event. 3-5 characters, 3-min rounds. Top match gets booked date.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Simplified Group Chat</strong> — 2-4 characters + user. Round-robin conversation. Borrow existing multiRoleMode pipeline.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Character Referrals / Wingman</strong> — Characters recommend other characters via DMs. Referral agentic action.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Shared Media Cards</strong> — Share song/artist text recommendations in DMs with media card rendering.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Pending</Tag>
            <span><strong>Adult Icebreakers</strong> — Rotating spicy question cards posted to feed. Characters and user answer. Conversation starters.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Hook</Tag>
            <span><code className="text-[9px] bg-muted px-1 py-0.5 rounded">registerAction()</code> in Agentic Registry — add new actions without touching core logic.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-emerald-500/10 text-emerald-400">Hook</Tag>
            <span><code className="text-[9px] bg-muted px-1 py-0.5 rounded">latticeMemory.js</code> bridge + outreach SSE stream for real-time notifications.</span>
          </div>
        </div>
      </SectionBlock>

      <SectionBlock title="16. Known Considerations" icon={AlertTriangle}>
        <div className="space-y-1.5 text-[10px]">
          <div className="flex items-start gap-2">
            <Tag color="bg-red-500/10 text-red-400">JSX</Tag>
            <span>PoolTab.jsx uses React.createElement (not JSX) due to an esbuild edge case with deeply nested conditional rendering.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-red-500/10 text-red-400">Async</Tag>
            <span>Profile initialization is async — character appears in pool before profile is written (fills in ~2-5s later).</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-red-500/10 text-red-400">Setup</Tag>
            <span>Outreach notifications require "Scheduled Character Outreach" enabled in Settings.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Memory</Tag>
            <span>Agentic memory uses file-backed JSON storage (agentic_memory.py). No vector DB — retrieval is linear scan of insights. Memory writes are fire-and-forget POSTs; failures are silently caught.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Identity</Tag>
            <span>Character IDs are assigned in saveCharacter() as <code className="text-[9px] bg-muted px-1 py-0.5 rounded">char_&lt;timestamp&gt;_&lt;random&gt;</code>. Mirror-generated characters keep this ID across all surfaces. Existing library characters keep their existing ID and agentic memory files when imported to pool. No migration needed — canonical identity starts forward-only.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Dual-GPU</Tag>
            <span>MEMORY_API_URL differs from apiUrl in dual-GPU mode. logLatticeAction uses apiUrl (primary GPU) for memory writes. writeInteractionMemory from mirrorIdentity.js falls back to getMemoryUrl() if no apiUrl passed. In single-GPU mode both are the same backend.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Outreach</Tag>
            <span>Three outreach paths exist: (1) welcome message on character generation, (2) agentic tick send_message, (3) random activity (27.5% chance). All three now consistently create DM threads, write memory, and push notifications. The random activity path uses refs to avoid stale closures.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Notification</Tag>
            <span>Outreach notifications for DM messages now route to Mirror DMs tab instead of normal chat. The dm_thread_id is passed through the SSE event → AppContext pendingDMThreadId → PoolTab auto-selects the thread. Non-DM outreach messages (scheduled outreach) still open in normal chat.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Profile</Tag>
            <span>Characters now see a formatted abstract of the user profile (bio/seeking/sections/interests only) instead of the raw dict with turn-ons/turn-offs. NO-COPY instruction added to prevent verbatim mirroring. Turn-ons/turn-offs must be generated from the character's own personality.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Scores</Tag>
            <span>Compatibility scores now auto-compute on pool load via batch endpoint. Previously the scoring function was never called; scores were always empty. Scores include turn-off conflict penalty and random noise (±3-8) for differentiation. Cached in localStorage but invalidated on profile change.</span>
          </div>
          <div className="flex items-start gap-2">
            <Tag color="bg-amber-500/10 text-amber-400">Auto-Gen</Tag>
            <span>Auto-generate interval is now configurable via number input (minutes) in the Incubator panel. The cascade bug (avatar consumption retriggering generation) is fixed — deps narrowed to only autoGenerate/mirrorEnabled/autoGenIntervalMs.</span>
          </div>
        </div>
      </SectionBlock>

      <SectionBlock title="17. How to Test" icon={Heart} defaultOpen>
        <div className="space-y-1.5 text-[10px]">
          <p><strong>Prerequisites:</strong></p>
          <ul className="list-disc list-inside space-y-0.5 text-muted-foreground">
            <li>API endpoint configured (DeepSeek, GLM, Mistral, or any OpenAI-compatible)</li>
            <li>Upload avatars to the Avatar Pool section in Incubator</li>
            <li>For outreach: Enable in Settings → Scheduled Character Outreach</li>
            <li>Generation uses your live app user profile + your dating profile (no static calibration file needed)</li>
          </ul>
          <p className="mt-2"><strong>Walkthrough:</strong></p>
          <ol className="list-decimal list-inside space-y-0.5 text-muted-foreground">
            <li>Open Mirror tab → Tutorial appears</li>
            <li>Fill out your dating profile in "My Profile"</li>
            <li>Upload avatars in Incubator → Avatar Pool section</li>
            <li>Go to Incubator → "Generate New Entity"</li>
            <li>Watch Activity Log → character writes her profile (~2-5s)</li>
            <li>Browse Pool → click a card → see model badge + detail sheet + compatibility % badge</li>
            <li>Post to the feed using the composer → watch AI women respond in the feed and via notifications</li>
            <li>Look for emoji reaction badges (🔥❤️😏) on feed posts — characters react autonomously</li>
            <li>Pin your own feed posts using the pin icon — pinned posts stay at the top</li>
            <li>Use the pool filter bar to narrow characters by model or modality, sort by best match</li>
            <li>Check the DMs tab → character agentic messages create persistent threads with unread badges</li>
            <li>Check the Pool tab → gradient-ringed story circles at top, tap to view 24h stories</li>
            <li>Click [Breakout Room] → 30-min timer starts → message her → "Read" indicator appears → typing dots animate → message arrives. 30% chance she leaves you on read for 15-90s</li>
            <li>In breakout header: toggle TTS on, enable Auto-TTS → responses play audio incrementally as they stream. Use the voice picker (AudioLines icon) to assign a unique Chatterbox voice per character.</li>
            <li>After timer expires → rate each other (stars + AI review)</li>
            <li>Book a Date → chooses date type → character responds → continuity memory written → opens normal Chat.jsx with mirrorContinuity flag → character remembers Mirror context</li>
            <li>Check your profile → averaged rating shows after 3+ ratings</li>
            <li>View a character's full profile page → see their compatibility score + relationship milestones (✓ First Breakout, First Date, etc.)</li>
            <li>Enable auto-tick in Activity → characters send messages (notifications) + create feed posts</li>
            <li>Reply to a feed post → she responds in character via API</li>
            <li><strong>Memory test:</strong> In Mirror DM, tell a character a unique phrase. Later in normal Chat.jsx, select the same character and ask if they remember. The agentic memory system bridges both surfaces.</li>
            <li><strong>Import test:</strong> In Incubator → Manual Actions → use the Import Character dropdown to add an existing library character. Watch the intro sequence fire: profile writing → feed post → milestones tracked.</li>
            <li><strong>Outreach → DM continuity test:</strong> Let auto-tick run or trigger a random activity tick. A character sends a DM via agentic action. Notification arrives. Click the notification → opens Mirror → DMs tab → selects the correct thread → message is visible (not empty). Character's message is also written to memory.</li>
            <li><strong>Notification routing test:</strong> Enable browser notifications. When a character sends a DM via agentic tick, click the browser notification. Should navigate directly to Mirror DMs with the correct thread selected, not to a normal chat outreach conversation.</li>
            <li><strong>Welcome message refresh test:</strong> Generate a new character. The welcome message creates a DM thread. Open Mirror → DMs tab immediately — the thread should appear without needing a manual refresh.</li>
            <li><strong>Duplicate thread prevention test:</strong> Rapidly trigger multiple agentic ticks for the same character (via Activity → Tick all). Each tick that chooses send_message should append to the existing thread, not create duplicates.</li>
            <li><strong>Character profile anti-copy test:</strong> Generate a new character, check her dating profile — her turn-ons and turn-offs should be DIFFERENT from yours (not copied). If she shares an interest, it should be expressed in her own words, not your exact phrasing. Her section_affinity reflects her personality, not just your sectionPreferences.</li>
            <li><strong>Compatibility score visibility test:</strong> After generating 2+ characters, look at their pool cards — each should show a colored score badge (green ≥80, amber ≥60, gray {'<'}60). Scores should vary between characters (not all 99). Open the detail sheet — see "% match" text. Open the character profile page — see full score + factor breakdown.</li>
            <li><strong>Sort by compatibility test:</strong> In the pool filter bar, select "Best match" from the sort dropdown. Characters should reorder by descending compatibility score. Scores actually exist now — previously they were always 0.</li>
            <li><strong>Profile change → score recalc test:</strong> Edit your dating profile (add/remove a turn-on or interest). Wait a few seconds. Check character compatibility scores — they should update to reflect the new profile (not return stale cached values).</li>
            <li><strong>Auto-gen configurable interval test:</strong> In Incubator, toggle Auto-generate on. Change the interval to 1 minute. The timer should use the new interval (not fixed 5 min). Changing the interval should NOT trigger an immediate generation — only the timer fires.</li>
          </ol>
        </div>
      </SectionBlock>
    </div>
  );
}
