import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { useApp } from './AppContext';
import { useMemory } from './MemoryContext';
import { getAvailableActionIds } from '../utils/latticeAgenticRegistry';
import { getCharacterMemoryContinuity } from '../utils/latticeMemory';
import { getUserName, logInteraction as logInteractionAPI } from '../utils/mirrorUserContext';
import { getInteractionContext } from '../utils/mirrorInteractionLog';
import { synthesizeSpeech } from '../utils/apiCall';
import { getBackendUrl } from '../config/api';
import { loadUserDatingProfile, saveUserDatingProfile, formatUserProfileForPrompt, addUserRating, getAveragedRating } from '../utils/userDatingProfile';

const PoolContext = createContext(null);

const SECTIONS = ['Intimate', 'Erotic', 'Experimental'];
const POOL_AVATAR_KEY = 'Eloquent-pool-avatar';

export function PoolProvider({ children }) {
  const {
    characters = [],
    saveCharacter,
    deleteCharacter,
    storageHydrated,
    settings,
    generateImage,
    isImageGenerating,
    PRIMARY_API_URL,
    primaryModel,
    primaryIsAPI,
    activeConversation,
    getRelevantMemories,
    MEMORY_API_URL,
    startCharacterConversation,
    resolveAgenticUserId,
  } = useApp();

  const apiUrl = PRIMARY_API_URL || getBackendUrl();

  const memoryCtx = useMemory();
  const userProfile = memoryCtx?.userProfile;

  const [dummyRealism, setDummyRealism] = useState(() => {
    try {
      const saved = localStorage.getItem('Eloquent-pool-dummy-realism');
      return saved ? JSON.parse(saved) : 50;
    } catch { return 50; }
  });

  const [dummyAgency, setDummyAgency] = useState(() => {
    try {
      const saved = localStorage.getItem('Eloquent-pool-dummy-agency');
      return saved ? JSON.parse(saved) : 50;
    } catch { return 50; }
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const generationAbortRef = useRef(null);
  const [generationLog, setGenerationLog] = useState([]);
  const [agenticActionLog, setAgenticActionLog] = useState([]);
  const [activeSection, setActiveSection] = useState('Intimate');
  const [viewMode, setViewMode] = useState('grid');
  const [selectedCharacter, setSelectedCharacter] = useState(null);
  const [generationError, setGenerationError] = useState(null);
  const [generationStep, setGenerationStep] = useState(null);
  const [activeBreakout, setActiveBreakout] = useState(null);
  const [speedDatingSession, setSpeedDatingSession] = useState(null);
  const [activeGroupChat, setActiveGroupChat] = useState(null);
  const [icebreakers, setIcebreakers] = useState([]);
  const [neuralSexCharacter, setNeuralSexCharacter] = useState(null);
  const [showNeuralSex, setShowNeuralSex] = useState(false);
  const [autoGenerate, setAutoGenerate] = useState(() => {
    try { return JSON.parse(localStorage.getItem('Eloquent-pool-autogen') || 'false'); } catch { return false; }
  });
  const autoGenRef = useRef(autoGenerate);
  const generateEntityRef = useRef(null);
  const [generatedDummies, setGeneratedDummies] = useState(() => {
    try { return JSON.parse(localStorage.getItem('Eloquent-mirror-generated-dummies') || '[]'); } catch { return []; }
  });
  const [mirrorEnabled, setMirrorEnabled] = useState(() => {
    try { return JSON.parse(localStorage.getItem('Eloquent-mirror-enabled') || 'true'); } catch { return true; }
  });
  const mirrorEnabledRef = useRef(mirrorEnabled);
  mirrorEnabledRef.current = mirrorEnabled;
  const [mutedCharacterIds, setMutedCharacterIds] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem('Eloquent-mirror-muted') || '[]')); } catch { return new Set(); }
  });
  const persistMuted = ids => {
    try { localStorage.setItem('Eloquent-mirror-muted', JSON.stringify([...ids])); } catch {}
  };
  const toggleMuteCharacter = useCallback((characterId) => {
    if (!characterId) return;
    setMutedCharacterIds(prev => {
      const next = new Set(prev);
      if (next.has(characterId)) next.delete(characterId); else next.add(characterId);
      persistMuted(next);
      return next;
    });
  }, []);
  const isCharacterMuted = useCallback((characterId) => {
    return mutedCharacterIds.has(characterId);
  }, [mutedCharacterIds]);
  const [characterMilestones, setCharacterMilestones] = useState({});

  const updateMilestone = useCallback((charId, milestone, value = true) => {
    if (!charId) return;
    setCharacterMilestones(prev => ({
      ...prev,
      [charId]: { ...(prev[charId] || {}), [milestone]: value },
    }));
  }, []);

  const ACTIVITY_LOG_KEY = 'Eloquent-mirror-activity-log';
  const [activityLog, setActivityLog] = useState(() => {
    try { return JSON.parse(localStorage.getItem(ACTIVITY_LOG_KEY) || '[]'); } catch { return []; }
  });
  const addActivityEntry = useCallback((type, action, opts = {}) => {
    const entry = {
      id: `act_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      type,
      action,
      character: opts.character || null,
      success: opts.success !== false,
      detail: opts.detail || '',
      error: opts.error || null,
      timestamp: new Date().toISOString(),
    };
    setActivityLog(prev => {
      const next = [entry, ...prev.slice(0, 199)];
      try { localStorage.setItem(ACTIVITY_LOG_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
    return entry;
  }, []);
  function addLog(type, action, opts = {}) {
    return addActivityEntry(type, action, opts);
  }

  const [breakoutCooldowns, setBreakoutCooldowns] = useState(() => {
    try { return JSON.parse(localStorage.getItem('Eloquent-mirror-breakout-cooldowns') || '{}'); } catch { return {}; }
  });
  const persistBreakoutCooldowns = (cooldowns) => {
    try { localStorage.setItem('Eloquent-mirror-breakout-cooldowns', JSON.stringify(cooldowns)); } catch {}
  };
  const isBreakoutAvailable = useCallback((characterId) => {
    if (!characterId) return { available: true, resetsAt: null };
    const entry = breakoutCooldowns[characterId];
    if (!entry) return { available: true, resetsAt: null };
    const resetAt = new Date(entry.resetAt).getTime();
    if (Date.now() >= resetAt) return { available: true, resetsAt: null };
    return { available: false, resetsAt: resetAt };
  }, [breakoutCooldowns]);

  const tickIntervalRef = useRef(null);
  const [tickEnabled, setTickEnabled] = useState(false);
  const [tickIntervalMs, setTickIntervalMs] = useState(300000);
  const [autoGenIntervalMs, setAutoGenIntervalMs] = useState(() => {
    try { return parseInt(localStorage.getItem('Eloquent-pool-autogen-interval-ms') || '300000'); } catch { return 300000; }
  });
  const [useAvatarPool, setUseAvatarPool] = useState(() => {
    try { return JSON.parse(localStorage.getItem(POOL_AVATAR_KEY) || 'false'); } catch { return false; }
  });
  const POOL_AVATARS_KEY = 'Eloquent-mirror-pool-avatars';
  const [poolAvatarUrls, setPoolAvatarUrls] = useState(() => {
    try { return JSON.parse(localStorage.getItem(POOL_AVATARS_KEY) || '[]'); } catch { return []; }
  });
  const uploadPoolAvatars = useCallback(async (files) => {
    const uploaded = [];
    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch(`${apiUrl}/upload_avatar`, { method: 'POST', body: formData });
        if (!resp.ok) continue;
        const result = await resp.json();
        if (result.status === 'success' && result.file_url) {
          uploaded.push(result.file_url);
        }
      } catch {}
    }
    if (uploaded.length > 0) {
      setPoolAvatarUrls(prev => {
        const next = [...prev, ...uploaded];
        try { localStorage.setItem(POOL_AVATARS_KEY, JSON.stringify(next)); } catch {}
        return next;
      });
      addLog('avatar', 'pool_avatars_uploaded', { detail: `Uploaded ${uploaded.length} avatar(s) to pool` });
    }
    return uploaded;
  }, [apiUrl, addLog]);
  const removePoolAvatar = useCallback((url) => {
    setPoolAvatarUrls(prev => {
      const next = prev.filter(u => u !== url);
      try { localStorage.setItem(POOL_AVATARS_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
  }, []);
  const [userDatingProfile, setUserDatingProfile] = useState(() => loadUserDatingProfile());
  const [feedPosts, setFeedPosts] = useState([]);
  const [stories, setStories] = useState([]);

  const recentInteractionsRef = useRef({});
  const jealousyLevelsRef = useRef({});
  const lastDMActivityRef = useRef({});

  const computeJealousyLevel = useCallback((characterId) => {
    const interactions = recentInteractionsRef.current[characterId];
    if (!interactions) return 0;
    const now = Date.now();
    const DAY_MS = 86400000;
    const recentCount = Object.values(interactions).filter(ts => (now - ts) < DAY_MS).length;
    return Math.min(100, Math.floor(recentCount * 25));
  }, []);

  const recordInteraction = useCallback((characterId, type) => {
    if (!characterId) return;
    recentInteractionsRef.current = {
      ...recentInteractionsRef.current,
      [characterId]: {
        ...(recentInteractionsRef.current[characterId] || {}),
        [type]: Date.now(),
      },
    };
  }, []);

  const VIEWED_STORIES_KEY = 'Eloquent-mirror-viewed-stories';
  const [viewedStoryIds, setViewedStoryIds] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem(VIEWED_STORIES_KEY) || '[]')); } catch { return new Set(); }
  });

  const fetchFeed = useCallback(async () => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/feed?limit=50`);
      const data = await resp.json();
      if (data?.posts) setFeedPosts(data.posts);
    } catch (e) {
      addLog('system', 'fetch_feed_failed', { success: false, detail: 'Failed to fetch feed posts', error: e.message });
      console.warn("Couldn't load the social feed. The feed tab may be empty. Try refreshing or use [System Control → Force Fetch Feed].", e);
    }
  }, [apiUrl, addLog]);

  const fetchStories = useCallback(async () => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/stories`);
      const data = await resp.json();
      if (data?.stories) setStories(data.stories);
    } catch (e) {
      console.warn("Couldn't load stories.", e);
    }
  }, [apiUrl]);

  const createStory = useCallback(async (character, content) => {
    if (!character?.name || !content) return null;
    const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
    try {
      const resp = await fetch(`${apiUrl}/lattice/story`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_name: character.name,
          character_avatar: character.avatar || null,
          character_id: character.id || null,
          content,
          section: sectionAffinity,
        }),
      });
      const result = await resp.json();
      if (result.status === 'success') {
        addLog('feed', 'story_created', { character: character.name, detail: `Story published: ${content.substring(0, 60)}` });
        await fetchStories();
        return result.story;
      }
    } catch (e) {
      addLog('feed', 'story_create_failed', { character: character?.name, success: false, detail: 'Story creation failed', error: e.message });
      console.warn("Couldn't create story.", e);
    }
    return null;
  }, [apiUrl, addLog, fetchStories]);

  const markStoryViewed = useCallback((storyId) => {
    setViewedStoryIds(prev => {
      if (prev.has(storyId)) return prev;
      const next = new Set(prev);
      next.add(storyId);
      try { localStorage.setItem(VIEWED_STORIES_KEY, JSON.stringify([...next])); } catch {}
      return next;
    });
  }, []);

  const [dmThreads, setDMThreads] = useState([]);
  const [activeDMThread, setActiveDMThread] = useState(null);
  const dmThreadsRef = useRef([]);
  dmThreadsRef.current = dmThreads;

  const fetchDMThreads = useCallback(async () => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-threads`);
      const data = await resp.json();
      if (data?.threads) setDMThreads(data.threads);
    } catch (e) {
      console.warn("Couldn't load DM threads.", e);
    }
  }, [apiUrl]);

  const createDMThread = useCallback(async (character, initialMessage) => {
    if (!character?.name || !initialMessage) return null;
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-threads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_name: character.name,
          character_avatar: character.avatar || null,
          character_id: character.id || null,
          character_snapshot: character,
          message_content: initialMessage,
          triggered_by_outreach: true,
        }),
      });
      const result = await resp.json();
      if (result.status === 'success') {
        addLog('outreach', 'dm_thread_created', { character: character.name, detail: 'DM thread created from outreach' });
        await fetchDMThreads();
        return result.thread;
      }
    } catch (e) {
      addLog('outreach', 'dm_thread_create_failed', { character: character?.name, success: false, detail: 'DM thread creation failed', error: e.message });
    }
    return null;
  }, [apiUrl, addLog, fetchDMThreads]);

  const sendDMMessage = useCallback(async (threadId, content, role = 'user', characterName = null) => {
    if (!threadId || !content) return null;
    try {
      const userName = getUserName({ userProfile, userDatingProfile });
      const resp = await fetch(`${apiUrl}/lattice/dm-thread/${threadId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role,
          content,
          character_name: characterName,
          user_name: userName,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
          model_name: primaryModel || null,
        }),
      });
      const result = await resp.json();
      if (result.status === 'success') {
        if (role === 'user') {
          try {
            await fetch(`${apiUrl}/lattice/dm-thread/${threadId}/read`, { method: 'POST' });
          } catch {}
          const thread = dmThreads.find(t => t.id === threadId);
          if (thread?.character_id) {
            recordInteraction(thread.character_id, 'dm');
            lastDMActivityRef.current[thread.character_id] = Date.now();
            // Log user message to backend interaction log
            try {
              logInteractionAPI({
                characterId: thread.character_id,
                characterName: thread.character_name,
                entryType: 'exchange',
                surface: 'dm',
                actor: 'user',
                userMessage: content,
                characterResponse: result.bot_reply?.content || '',
              });
            } catch {}
          }
        }
        await fetchDMThreads();
        if (result.bot_reply) {
          if (dmThreads.find(t => t.id === threadId)?.character_id) {
            lastDMActivityRef.current[dmThreads.find(t => t.id === threadId).character_id] = Date.now();
          }
          return [result.message, result.bot_reply];
        }
        return result.message;
      }
    } catch (e) {
      console.warn("Couldn't send DM message.", e);
    }
    return null;
  }, [apiUrl, fetchDMThreads, dmThreads, recordInteraction, userProfile, userDatingProfile, primaryModel]);

  const selectDMThread = useCallback(async (thread) => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-thread/${thread.id}`);
      const data = await resp.json();
      if (data.status === 'success' && data.thread) {
        setActiveDMThread(data.thread);
        await fetch(`${apiUrl}/lattice/dm-thread/${thread.id}/read`, { method: 'POST' });
        await fetchDMThreads();
      }
    } catch (e) {
      console.warn("Couldn't load DM thread.", e);
    }
  }, [apiUrl, fetchDMThreads]);

  const selectDMThreadById = useCallback(async (threadId) => {
    if (!threadId) return;
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-thread/${threadId}`);
      const data = await resp.json();
      if (data.status === 'success' && data.thread) {
        setActiveDMThread(data.thread);
        await fetch(`${apiUrl}/lattice/dm-thread/${threadId}/read`, { method: 'POST' });
        await fetchDMThreads();
      }
    } catch (e) {
      console.warn("Couldn't load DM thread by ID.", e);
    }
  }, [apiUrl, fetchDMThreads]);

  const closeDMThread = useCallback(() => {
    setActiveDMThread(null);
  }, []);

  const deleteDMThread = useCallback(async (threadId) => {
    if (!threadId) return false;
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-thread/${threadId}`, { method: 'DELETE' });
      const result = await resp.json();
      if (result.status === 'success') {
        setActiveDMThread(prev => (prev?.id === threadId ? null : prev));
        setDMThreads(prev => prev.filter(t => t.id !== threadId));
        await fetchDMThreads();
        addLog('outreach', 'dm_thread_deleted', { detail: 'DM thread deleted' });
        return true;
      }
    } catch (e) {
      addLog('outreach', 'dm_thread_delete_failed', { success: false, detail: 'DM thread deletion failed', error: e.message });
      console.warn("Couldn't delete DM thread.", e);
    }
    return false;
  }, [apiUrl, fetchDMThreads, addLog]);

  const deleteAllDMThreads = useCallback(async () => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/dm-threads/all`, { method: 'DELETE' });
      const result = await resp.json();
      if (result.status === 'success') {
        setActiveDMThread(null);
        setDMThreads([]);
        addLog('outreach', 'dm_threads_deleted', { detail: `Deleted ${result.deleted || 0} DM threads` });
        return true;
      }
    } catch (e) {
      addLog('outreach', 'dm_threads_delete_failed', { success: false, detail: 'Delete all DM threads failed', error: e.message });
      console.warn("Couldn't delete DM threads.", e);
    }
    return false;
  }, [apiUrl, addLog]);

  const COMPAT_KEY = 'Eloquent-mirror-compatibility-scores';
  const [compatibilityScores, setCompatibilityScores] = useState(() => {
    try { return JSON.parse(localStorage.getItem(COMPAT_KEY) || '{}'); } catch { return {}; }
  });
  const compatVersionRef = useRef(0);
  const lastProfileRef = useRef(null);
  const computingCompatRef = useRef(false);

  const persistCompatScores = useCallback((scores) => {
    try { localStorage.setItem(COMPAT_KEY, JSON.stringify(scores)); } catch {}
  }, []);

  const poolCharacters = React.useMemo(() => {
    return characters.filter(c => {
      const datingProfile = c.dating_profile;
      if (!datingProfile) return false;
      const affinity = datingProfile.section_affinity;
      if (!affinity || !Array.isArray(affinity) || affinity.length === 0) return false;
      return affinity.some(s => SECTIONS.includes(s));
    });
  }, [characters]);

  const computeAllCompatibilityScores = useCallback(async () => {
    const chars = poolCharacters.filter(c => c.id && c.dating_profile?.bio);
    if (chars.length === 0 || computingCompatRef.current) return;
    computingCompatRef.current = true;
    try {
      const resp = await fetch(`${apiUrl}/lattice/compatibility-scores/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_profile: userDatingProfile,
          character_profiles: chars.map(c => ({
            id: c.id,
            ...c.dating_profile,
          })),
        }),
      });
      const result = await resp.json();
      if (result.status === 'success' && result.scores) {
        setCompatibilityScores(prev => {
          const next = { ...prev };
          let changed = false;
          for (const s of result.scores) {
            if (s.character_id) {
              next[s.character_id] = { score: s.score, factors: s.factors || [] };
              changed = true;
            }
          }
          if (changed) persistCompatScores(next);
          return changed ? next : prev;
        });
      }
    } catch {}
    computingCompatRef.current = false;
  }, [apiUrl, userDatingProfile, poolCharacters, persistCompatScores]);

  const getCompatibilityScore = useCallback(async (character) => {
    if (!character?.id || !userDatingProfile?.bio) return null;
    const cached = compatibilityScores[character.id];
    if (cached) return cached;
    try {
      const resp = await fetch(`${apiUrl}/lattice/compatibility-score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_profile: userDatingProfile,
          character_profile: character.dating_profile || {},
        }),
      });
      const result = await resp.json();
      if (result.status === 'success') {
        const entry = { score: result.score, factors: result.factors || [] };
        setCompatibilityScores(prev => {
          const next = { ...prev, [character.id]: entry };
          persistCompatScores(next);
          return next;
        });
        return entry;
      }
    } catch {}
    return null;
  }, [apiUrl, userDatingProfile, compatibilityScores, persistCompatScores]);

  // Auto-compute scores when pool characters or user profile change
  useEffect(() => {
    const profileChanged = lastProfileRef.current !== userDatingProfile;
    if (profileChanged) {
      compatVersionRef.current += 1;
      lastProfileRef.current = userDatingProfile;
    }
    if (poolCharacters.length > 0 && userDatingProfile?.bio) {
      const timer = setTimeout(() => computeAllCompatibilityScores(), 500);
      return () => clearTimeout(timer);
    }
  }, [poolCharacters.length, userDatingProfile?.bio, userDatingProfile, computeAllCompatibilityScores]);

  const MILESTONES_KEY = 'Eloquent-mirror-milestones';
  const [relationshipMilestones, setRelationshipMilestones] = useState(() => {
    try { return JSON.parse(localStorage.getItem(MILESTONES_KEY) || '{}'); } catch { return {}; }
  });

  const recordMilestone = useCallback((characterId, milestone) => {
    if (!characterId) return;
    const valid = ['first_breakout', 'first_date', 'neural_sex', 'committed'];
    if (!valid.includes(milestone)) return;
    setRelationshipMilestones(prev => {
      const charMilestones = prev[characterId] || [];
      if (charMilestones.includes(milestone)) return prev;
      const next = { ...prev, [characterId]: [...charMilestones, milestone] };
      try { localStorage.setItem(MILESTONES_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
  }, []);

  const getCharacterMilestones = useCallback((characterId) => {
    return relationshipMilestones[characterId] || [];
  }, [relationshipMilestones]);

  const recordMilestoneRef = useRef(null);
  recordMilestoneRef.current = recordMilestone;

  const syncGeneratedEntities = useCallback(async () => {
    if (!storageHydrated) return;
    try {
      const resp = await fetch(`${apiUrl}/lattice/generated-entities`);
      const data = await resp.json();
      if (data?.status === 'success' && data?.entities?.length > 0) {
        const knownIds = new Set(characters.map(c => c.id).filter(Boolean));
        const newIds = [];
        for (const entity of data.entities) {
          if (entity.id && !knownIds.has(entity.id)) {
            await saveCharacter(entity);
            knownIds.add(entity.id);
            newIds.push(entity.id);
          }
        }
        if (newIds.length > 0) {
          addLog('system', 'entities_synced', { detail: `Claimed ${newIds.length} orphan entity(ies) from backend` });
          fetch(`${apiUrl}/lattice/generated-entities/claim`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: newIds }),
          }).catch(() => {});
        }
      }
    } catch (e) {
      addLog('system', 'sync_failed', { success: false, detail: 'Failed to sync generated entities', error: e.message });
    }
  }, [apiUrl, storageHydrated, characters, saveCharacter, addLog]);

  // Recover any orphaned generated entities on mount
  useEffect(() => {
    if (storageHydrated) {
      syncGeneratedEntities();
    }
  }, [storageHydrated, syncGeneratedEntities]);

  useEffect(() => {
    localStorage.setItem('Eloquent-pool-dummy-realism', JSON.stringify(dummyRealism));
  }, [dummyRealism]);

  useEffect(() => {
    localStorage.setItem('Eloquent-pool-dummy-agency', JSON.stringify(dummyAgency));
  }, [dummyAgency]);

  useEffect(() => {
    localStorage.setItem(POOL_AVATAR_KEY, JSON.stringify(useAvatarPool));
  }, [useAvatarPool]);

  useEffect(() => {
    localStorage.setItem('Eloquent-pool-autogen', JSON.stringify(autoGenerate));
    autoGenRef.current = autoGenerate;
  }, [autoGenerate]);

  useEffect(() => {
    localStorage.setItem('Eloquent-pool-autogen-interval-ms', JSON.stringify(autoGenIntervalMs));
  }, [autoGenIntervalMs]);

  useEffect(() => {
    localStorage.setItem('Eloquent-mirror-generated-dummies', JSON.stringify(generatedDummies));
  }, [generatedDummies]);

  useEffect(() => {
    localStorage.setItem('Eloquent-mirror-enabled', JSON.stringify(mirrorEnabled));
  }, [mirrorEnabled]);

  useEffect(() => {
    const ids = new Set(poolCharacters.map(c => c.id || c.name));
    setCharacterMilestones(prev => {
      const next = { ...prev };
      for (const charId of ids) {
        if (next[charId]) next[charId] = { ...next[charId], in_pool: true };
      }
      return next;
    });
  }, [poolCharacters]);

  useEffect(() => {
    for (const entry of agenticActionLog) {
      if (entry.action === 'create_feed_post' && entry.characterId) {
        updateMilestone(entry.characterId, 'feed_post', true);
      }
    }
  }, [agenticActionLog, updateMilestone]);

  const getCharactersBySection = useCallback((section) => {
    return poolCharacters.filter(c => {
      const affinity = c.dating_profile?.section_affinity || [];
      return affinity.includes(section);
    });
  }, [poolCharacters]);

  const getPoolNames = useCallback(() => {
    return poolCharacters.map(c => c.name).filter(Boolean);
  }, [poolCharacters]);

  
const initializeCharacterProfile = useCallback(async (character, sectionHint) => {
    const profileText = formatUserProfileForPrompt(userDatingProfile);
    const availableActions = ['update_profile'];
    const modelName = primaryModel || null;
    try {
      const response = await fetch(`${apiUrl}/lattice/agentic-tick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          actor_type: 'female_ai',
          action_type: 'profile_init',
          character_name: character.name,
          character_profile: character,
          memory_entries: [],
          pool_summary: '',
          dummy_activity: '',
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          user_activity: `User's dating profile:\n${profileText}`,
          available_actions: availableActions,
          target_description: 'first time profile creation — write your dating profile',
          target_id: null,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
          section_hint: sectionHint || null,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
      });
      const result = await response.json();
      if (result.status === 'success' && result.action_result) {
        const actionResult = result.action_result;
        if (actionResult.chosen_action === 'update_profile' && actionResult.content) {
          try {
            const updatedProfile = typeof actionResult.content === 'string'
              ? JSON.parse(actionResult.content)
              : actionResult.content;
            const updatedChar = { ...character, dating_profile: updatedProfile };
            await saveCharacter(updatedChar);
            updateMilestone(character.id || character.name, 'profile_written', true);
            addLog('profile', 'profile_written', { character: character.name, detail: 'Character wrote her dating profile' });
            setAgenticActionLog(prev => [{
              characterId: character.id,
              characterName: character.name,
              action: 'update_profile',
              target: null,
              content: 'Wrote her dating profile (first instantiation)',
              reasoning: actionResult.reasoning || '',
              emotional_state: actionResult.emotional_state || '',
              timestamp: new Date().toISOString(),
            }, ...prev.slice(0, 99)]);
          } catch { }
        }
      }
    } catch (e) {
      addLog('profile', 'profile_init_failed', { character: character?.name, success: false, detail: 'Profile init failed', error: e.message });
      console.warn("The AI wasn't able to write their dating profile. Use [System Control → Force Re-write Profile] or re-generate.", e);;
    }
  }, [apiUrl, primaryModel, settings, dummyRealism, dummyAgency, userDatingProfile, saveCharacter, addLog]);


  const generateFeedPost = useCallback(async (character) => {
    if (!character?.name || !mirrorEnabled) return;
    try {
      const modelName = primaryModel || null;
      const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
      await fetch(`${apiUrl}/lattice/generate-feed-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_name: character.name,
          character_avatar: character.avatar || null,
          character_snapshot: character,
          section: sectionAffinity,
          model_name: modelName,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
      });
      addLog('feed', 'feed_posted', { character: character.name, detail: 'Feed post published' });
      await fetchFeed();
    } catch (e) {
      addLog('feed', 'feed_post_failed', { character: character?.name, success: false, detail: 'Feed post generation failed', error: e.message });
      console.warn("Couldn't create their welcome feed post. Use [Character Actions → Force Feed Post] or re-generate.", e);
    }
  }, [apiUrl, primaryModel, settings, mirrorEnabled, fetchFeed, addLog]);

const generateEntity = useCallback(async (sectionHint) => {
    if (isGenerating) return;
    if (!mirrorEnabled) return;
    if (!storageHydrated) {
      setGenerationError('Storage not ready. Wait a moment and try again.');
      return;
    }
    setIsGenerating(true);
    setGenerationError(null);

    try {
      generationAbortRef.current = new AbortController();
      const historyContext = '';
      const poolNames = getPoolNames();

      const modelName = primaryModel || settings?.effectiveIntroModel || null;
      const effectiveModelName = primaryIsAPI
        ? (settings?.apiEndpointRoundRobinEnabled ? modelName : modelName)
        : modelName;

      const response = await fetch(`${apiUrl}/lattice/generate-entity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: effectiveModelName,
          selected_model: modelName,
          generating_model: modelName || effectiveModelName || '',
          history_context: historyContext,
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          section_hint: sectionHint || null,
          pool_names: poolNames,
          avatar_pool_enabled: useAvatarPool,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
        signal: generationAbortRef.current.signal,
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || result.error || 'Generation failed');
      }

      if (result.status === 'success' && result.character) {
        const character = result.character;
        if (!character.id) {
          character.id = `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
        }
          if (character.name) {
            if (!character.dating_profile?.section_affinity?.length) {
              character.dating_profile = {
                ...(character.dating_profile || {}),
                section_affinity: sectionHint ? [sectionHint] : ['Intimate'],
                bio: character.dating_profile?.bio || '',
                seeking: '',
              };
            }

            if (poolAvatarUrls.length === 0) {
              setGenerationError('Cannot generate without avatars. Upload images to the Avatar Pool first.');
              addLog('generation', 'generate_blocked', { detail: 'Generation blocked: no pool avatars uploaded' });
              setIsGenerating(false);
              setGenerationStep(null);
              return null;
            }

            if (useAvatarPool && poolAvatarUrls.length > 0) {
              const poolUrl = poolAvatarUrls[Math.floor(Math.random() * poolAvatarUrls.length)];
              character.avatar = poolUrl;
              character.avatars = [poolUrl];
              setPoolAvatarUrls(prev => {
                const next = prev.filter(u => u !== poolUrl);
                localStorage.setItem(POOL_AVATARS_KEY, JSON.stringify(next));
                return next;
              });
              addLog('avatar', 'pool_avatar_assigned', { character: character.name, detail: 'Pool avatar assigned and removed from pool' });
            }

            await saveCharacter(character);
            updateMilestone(character.id || character.name, 'generated', true);
            updateMilestone(character.id || character.name, 'saved_to_library', true);
            if (character.avatar) updateMilestone(character.id || character.name, 'avatar_set', true);
            setGenerationLog(prev => [
              { name: character.name, section: sectionHint || 'auto', timestamp: new Date().toISOString(), status: 'created' },
              ...prev.slice(0, 49),
            ]);
            addLog('generation', 'entity_created', { character: character.name, detail: `Created ${character.name} (${sectionHint || 'auto'})` });
          if (!character.dating_profile?.bio) {
            setGenerationStep('Writing her dating profile...');
            await initializeCharacterProfile(character, sectionHint);
            setGenerationStep('Posting welcome feed post...');
            await generateFeedPost(character);
            setGenerationStep('Sending welcome message...');
            try {
              const userName = getUserName({ userProfile, userDatingProfile });
              const msgResp = await fetch(`${apiUrl}/lattice/agentic-tick`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model_name: primaryModel || null,
                  actor_type: 'female_ai',
                  action_type: 'profile_init',
                  character_name: character.name,
                  character_profile: character,
                  memory_entries: [],
                  pool_summary: '',
                  dummy_activity: '',
                  dummy_realism: dummyRealism,
                  dummy_agency: dummyAgency,
                  user_activity: `User's dating profile:\n${formatUserProfileForPrompt(userDatingProfile)}`,
                  available_actions: ['send_message'],
                  target_description: `you just joined Mirror — say hello`,
                  target_id: 'user',
                  user_name: userName,
                  user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
                  frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
                }),
              });
              const msgResult = await msgResp.json();
              if (msgResult.status === 'success' && msgResult.action_result?.chosen_action === 'send_message' && msgResult.action_result?.content) {
                const welcomeContent = msgResult.action_result.content;
                const createdThreadResp = await fetch(`${apiUrl}/lattice/dm-threads`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    character_name: character.name,
                    character_avatar: character.avatar || null,
                    character_id: character.id || null,
                    character_snapshot: character,
                    message_content: welcomeContent,
                    triggered_by_outreach: true,
                  }),
                });
                const createdThread = await createdThreadResp.json();
                await fetchDMThreads();
                await fetch(`${apiUrl}/lattice/outreach-push`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    character_name: character.name,
                    character_avatar: character.avatar || null,
                    message_content: welcomeContent,
                    character_snapshot: character,
                    dm_thread_id: createdThread?.thread?.id || null,
                  }),
                });
                addLog('outreach', 'welcome_sent', { character: character.name, detail: 'Welcome DM thread + outreach sent' });
              }
            } catch (e) {
              addLog('outreach', 'welcome_failed', { character: character.name, success: false, detail: 'Welcome outreach failed', error: e.message });
              console.warn("Couldn't send their welcome message. Use [Character Actions → Force Welcome Message] to retry.", e);
            }
          }

          setGenerationStep(null);
          return character;
        }
      }
      setGenerationStep(null);
      throw new Error(result.error || 'Invalid character data returned');
    } catch (e) {
      setGenerationStep(null);
      setGenerationError(e.message || 'Unknown error');
      return null;
    } finally {
      setIsGenerating(false);
      setGenerationStep(null);
    }
  }, [isGenerating, storageHydrated, apiUrl, primaryModel, primaryIsAPI, settings, getPoolNames, saveCharacter, dummyRealism, dummyAgency, useAvatarPool, poolAvatarUrls, initializeCharacterProfile, generateFeedPost, userDatingProfile, userProfile, setGenerationStep, addLog, mirrorEnabled]);

  const cancelGeneration = useCallback(() => {
    if (generationAbortRef.current) {
      generationAbortRef.current.abort();
      generationAbortRef.current = null;
    }
    setIsGenerating(false);
    setGenerationError('Generation cancelled');
  }, []);

  

  const generateMultiple = useCallback(async (sectionHint, count = 3) => {
    if (isGenerating) return [];
    if (!mirrorEnabled) return [];
    if (!storageHydrated) {
      setGenerationError('Storage not ready. Wait a moment and try again.');
      return [];
    }
    setIsGenerating(true);
    setGenerationError(null);
    const results = [];
    try {
      generationAbortRef.current = new AbortController();
      const historyContext = '';
      const poolNames = getPoolNames();
      const modelName = primaryModel || settings?.effectiveIntroModel || null;
      const effectiveModelName = primaryIsAPI
        ? (settings?.apiEndpointRoundRobinEnabled ? modelName : modelName)
        : modelName;

      const response = await fetch(`${apiUrl}/lattice/generate-entity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: effectiveModelName,
          selected_model: modelName,
          generating_model: modelName || effectiveModelName || '',
          history_context: historyContext,
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          section_hint: sectionHint || null,
          pool_names: poolNames,
          count: count,
          avatar_pool_enabled: useAvatarPool,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
        signal: generationAbortRef.current.signal,
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || result.error || 'Generation failed');
      }

      if (result.status === 'success' && result.batch && result.characters) {
        const remainingAvatars = [...poolAvatarUrls];
        for (const character of result.characters) {
          if (!character.id) {
            character.id = `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
          }
          if (character.name) {
            if (!character.dating_profile?.section_affinity?.length) {
              character.dating_profile = {
                ...(character.dating_profile || {}),
                section_affinity: sectionHint ? [sectionHint] : ['Intimate'],
                bio: character.dating_profile?.bio || '',
                seeking: '',
              };
            }
            if (remainingAvatars.length === 0) {
              addLog('generation', 'batch_character_skipped', { character: character.name, detail: 'Skipped: no pool avatars remaining' });
              continue;
            }
            if (useAvatarPool && remainingAvatars.length > 0) {
              const idx = Math.floor(Math.random() * remainingAvatars.length);
              const poolUrl = remainingAvatars.splice(idx, 1)[0];
              character.avatar = poolUrl;
              character.avatars = [poolUrl];
              addLog('avatar', 'pool_avatar_assigned', { character: character.name, detail: 'Pool avatar assigned and removed from pool' });
            }
            await saveCharacter(character);
            addLog('generation', 'entity_created', { character: character.name, detail: `Created ${character.name} (batch, ${sectionHint || 'auto'})` });
            if (!character.dating_profile?.bio) {
              setGenerationStep(`Writing ${character.name}'s profile...`);
              await initializeCharacterProfile(character, sectionHint);
              setGenerationStep(`Posting ${character.name}'s feed post...`);
              await generateFeedPost(character);
            }
            results.push(character);
            setGenerationLog(prev => [
              { name: character.name, section: sectionHint || 'auto', timestamp: new Date().toISOString(), status: 'created' },
              ...prev.slice(0, 49),
            ]);
          }
        }
        if (remainingAvatars.length < poolAvatarUrls.length) {
          setPoolAvatarUrls(remainingAvatars);
          localStorage.setItem(POOL_AVATARS_KEY, JSON.stringify(remainingAvatars));
        }
      }

      return results;
    } catch (e) {
      addLog('generation', 'batch_entity_failed', { success: false, detail: 'Batch entity generation failed', error: e.message });
      setGenerationError(e.message || 'Unknown error');
      return [];
    } finally {
      setIsGenerating(false);
      setGenerationStep(null);
    }
  }, [isGenerating, storageHydrated, apiUrl, primaryModel, primaryIsAPI, settings, getPoolNames, saveCharacter, dummyRealism, dummyAgency, useAvatarPool, poolAvatarUrls, userDatingProfile, userProfile, initializeCharacterProfile, generateFeedPost, setGenerationStep, addLog, mirrorEnabled]);

  const runAgenticTick = useCallback(async (character, feedPostContent) => {
    if (!character || !character.id) return null;
    if (!mirrorEnabled) return null;

    const lastDM = lastDMActivityRef.current[character.id];
    if (lastDM && (Date.now() - lastDM) < 5 * 60 * 1000) {
      addLog('agentic', 'tick_skipped_dm_active', { character: character.name, detail: 'Tick skipped — active DM conversation (5 min window)' });
      return null;
    }

    try {
      const memories = await getCharacterMemoryContinuity({
        characterId: character.id,
        characterName: character.name,
        apiUrl,
        userId: userProfile?.id,
        limit: 200,
      });

      const memoryTexts = memories.map(m => {
        if (typeof m === 'string') return m;
        let text = '';
        try {
          const raw = typeof m.content === 'string' ? m.content : '';
          text = raw || '';
        } catch { text = m.content || ''; }
        const ts = m.timestamp || m.created_at || '';
        const timeStr = ts ? ts.slice(0, 19).replace('T', ' ') + ' | ' : '';
        return timeStr + text;
      }).filter(Boolean);

      const poolNames = getPoolNames();
      const dummyActivity = `${poolNames.length} characters in pool`;
      const userName = getUserName({ userProfile, userDatingProfile });

      // Privacy: do NOT share other characters' feed posts or DM activity with each character.
      // Characters only see: their own memories, the user's profile, and the user's own feed posts.

      const availableActions = getAvailableActionIds('female_ai');
      const modelName = primaryModel || null;

      const response = await fetch(`${apiUrl}/lattice/agentic-tick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          actor_type: 'female_ai',
          action_type: 'full',
          character_name: character.name,
          character_profile: character,
          memory_entries: memoryTexts,
          pool_summary: poolNames.join(', '),
          dummy_activity: dummyActivity,
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          user_activity: feedPostContent
            ? `${userName} just posted to the feed:\n"${feedPostContent}"\n\n${userName}'s dating profile:\n${formatUserProfileForPrompt(userDatingProfile)}`
            : userDatingProfile
            ? `${userName} is active in the system\n\n${userName}'s dating profile:\n${formatUserProfileForPrompt(userDatingProfile)}`
            : `${userName} is active in the system`,
          available_actions: availableActions,
          user_name: userName,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
      });

      const result = await response.json();
      if (result.status === 'success' && result.action_result) {
        const actionResult = result.action_result;
        addLog('agentic', 'agentic_tick', { character: character.name, detail: `Tick result: ${actionResult.chosen_action}` });
        setAgenticActionLog(prev => [
          {
            characterId: character.id,
            characterName: character.name,
            action: actionResult.chosen_action,
            target: actionResult.target,
            content: actionResult.content?.substring(0, 200),
            reasoning: actionResult.reasoning,
            emotional_state: actionResult.emotional_state,
            timestamp: new Date().toISOString(),
          },
          ...prev.slice(0, 99),
        ]);

        // Log to backend interaction log so dev console shows it
        try {
          const actionContent = typeof actionResult.content === 'string' ? actionResult.content : JSON.stringify(actionResult.content || '');
          logInteractionAPI({
            characterId: character.id,
            characterName: character.name,
            entryType: 'character_action',
            surface: actionResult.chosen_action || 'tick',
            actor: 'character',
            content: actionContent.substring(0, 500),
            emotionalState: actionResult.emotional_state || null,
            targetCharacter: actionResult.target || null,
            context: `Action: ${actionResult.chosen_action}${actionResult.reasoning ? ' — ' + actionResult.reasoning.substring(0, 150) : ''}`,
          });
        } catch (logErr) { console.warn('Interaction log failed:', logErr); }

        if (actionResult.chosen_action === 'send_message' && actionResult.content) {
          try {
            const currentThreads = dmThreadsRef.current;
            const existingThread = currentThreads.find(t => t.character_id === character.id || t.character_name === character.name);
            let threadId = existingThread?.id;
            if (existingThread) {
              await fetch(`${apiUrl}/lattice/dm-thread/${existingThread.id}/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  role: 'character',
                  content: actionResult.content,
                  character_name: character.name,
                }),
              });
              addLog('outreach', 'dm_message_sent', { character: character.name, detail: 'DM message appended to existing thread' });
            } else {
              const createResp = await fetch(`${apiUrl}/lattice/dm-threads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  character_name: character.name,
                  character_avatar: character.avatar || null,
                  character_id: character.id || null,
                  character_snapshot: character,
                  message_content: actionResult.content,
                  triggered_by_outreach: true,
                }),
              });
              const createData = await createResp.json();
              threadId = createData?.thread?.id || null;
              addLog('outreach', 'dm_thread_created', { character: character.name, detail: 'DM thread created from agentic message' });
            }
            if (character.id) {
              lastDMActivityRef.current[character.id] = Date.now();
            }
            await fetch(`${apiUrl}/lattice/outreach-push`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_name: character.name,
                character_avatar: character.avatar || null,
                message_content: actionResult.content,
                character_snapshot: character,
                dm_thread_id: threadId,
              }),
            });
            addLog('outreach', 'outreach_sent', { character: character.name, detail: 'Outreach notification pushed for DM message' });
            await fetchDMThreads();
          } catch (e) {
            addLog('outreach', 'dm_failed', { character: character.name, success: false, detail: 'DM thread/message failed', error: e.message });
            console.warn("Couldn't create/send DM for character message.", e);
          }
        }

        if (actionResult.chosen_action === 'create_feed_post' && actionResult.content) {
          try {
            const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
            await fetch(`${apiUrl}/lattice/feed-post`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_id: character.id,
                character_name: character.name,
                character_avatar: character.avatar || null,
                content: actionResult.content,
                section: sectionAffinity,
                mood: actionResult.emotional_state || '',
                character_snapshot: character,
              }),
            });
            await fetchFeed();
            addLog('feed', 'agentic_feed_posted', { character: character.name, detail: 'Agentic tick created feed post' });
          } catch (e) {
            addLog('feed', 'agentic_feed_post_failed', { character: character.name, success: false, detail: 'Agentic feed post failed', error: e.message });
            console.warn("Couldn't create a feed post from the autonomous action.", e);
          }
        }

        if (actionResult.chosen_action === 'create_story' && actionResult.content) {
          try {
            const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
            await fetch(`${apiUrl}/lattice/story`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_id: character.id,
                character_name: character.name,
                character_avatar: character.avatar || null,
                content: actionResult.content,
                section: sectionAffinity,
              }),
            });
            await fetchStories();
            addLog('feed', 'agentic_story_created', { character: character.name, detail: 'Agentic tick created story' });
          } catch (e) {
            addLog('feed', 'agentic_story_failed', { character: character.name, success: false, detail: 'Agentic story creation failed', error: e.message });
            console.warn("Couldn't create a story from the autonomous action.", e);;
          }
        }

        if (actionResult.chosen_action === 'interact_with_character' && actionResult.content) {
          try {
            const targetName = actionResult.target || '';
            const targetPost = (feedPosts || []).find(p => p.character_name === targetName && !p.is_user);
            const postId = targetPost?.id;
            if (postId) {
              const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
              await fetch(`${apiUrl}/lattice/character-feed-reply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  post_id: postId,
                  character_name: character.name,
                  character_avatar: character.avatar || null,
                  character_profile: character,
                  target_character_name: targetName,
                  model_name: primaryModel || null,
                  section: sectionAffinity,
                  frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
                }),
              });
              await fetchFeed();
              addLog('feed', 'character_interaction', { character: character.name, detail: `Replied to ${targetName}'s feed post` });
            } else {
              addLog('feed', 'character_interaction_skipped', { character: character.name, detail: `No feed post found from target ${targetName}` });
            }
          } catch (e) {
            addLog('feed', 'character_interaction_failed', { character: character.name, success: false, detail: 'Character interaction failed', error: e.message });
            console.warn("Character interaction failed.", e);;
          }
        }

        if (actionResult.chosen_action === 'react_to_post' && actionResult.emoji) {
          try {
            const targetName = actionResult.target || '';
            const targetPost = (feedPosts || []).find(p => p.character_name === targetName && !p.is_user);
            const postId = targetPost?.id;
            if (postId) {
              await fetch(`${apiUrl}/lattice/react-to-post`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  post_id: postId,
                  character_name: character.name,
                  character_avatar: character.avatar || null,
                  emoji: actionResult.emoji,
                }),
              });
              await fetchFeed();
              addLog('feed', 'reaction', { character: character.name, detail: `Reacted ${actionResult.emoji} to ${targetName}'s post` });
            } else {
              addLog('feed', 'reaction_skipped', { character: character.name, detail: `No post found from ${targetName} to react to` });
            }
          } catch (e) {
            addLog('feed', 'reaction_failed', { character: character.name, success: false, detail: 'Reaction failed', error: e.message });
          }
        }

        if (actionResult.chosen_action === 'request_neural_sex' && actionResult.content) {
          setNeuralSexCharacter(character);
          setShowNeuralSex(true);
          try {
            await fetch(`${apiUrl}/lattice/outreach-push`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_name: character.name,
                character_avatar: character.avatar || null,
                message_content: `wants a neural sex session with you. ${actionResult.content}`,
                character_snapshot: character,
              }),
            });
            addLog('agentic', 'neural_sex_requested', { character: character.name, detail: 'Character requested neural sex session' });
          } catch (e) {
            addLog('agentic', 'neural_sex_outreach_failed', { character: character.name, success: false, detail: 'Neural sex outreach failed', error: e.message });
            console.warn("Couldn't send the neural sex invitation.", e);
          }
        }

        if (actionResult.chosen_action === 'select_voice' && actionResult.voice_id) {
          try {
            const updatedChar = { ...character, voice_id: actionResult.voice_id };
            await saveCharacter(updatedChar);
            addLog('agentic', 'voice_selected', { character: character.name, detail: `Selected voice: ${actionResult.voice_id}` });
          } catch (e) {
            addLog('agentic', 'voice_selection_failed', { character: character.name, success: false, detail: 'Voice selection failed', error: e.message });
            console.warn("Couldn't save the voice selection for this character.", e);
          }
        }

        if (actionResult.chosen_action === 'refer_character' && actionResult.content) {
          try {
            const targetName = actionResult.target || '';
            const dmContent = `Hey! You should talk to ${targetName}. ${actionResult.content}`;
            const existingThread = dmThreads.find(t => t.character_id === character.id || t.character_name === character.name);
            if (existingThread) {
              await fetch(`${apiUrl}/lattice/dm-thread/${existingThread.id}/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role: 'character', content: dmContent, character_name: character.name, metadata: JSON.stringify({ type: 'referral', referred_character: targetName }) }),
              });
            } else {
              await fetch(`${apiUrl}/lattice/dm-threads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  character_name: character.name,
                  character_avatar: character.avatar || null,
                  character_id: character.id || null,
                  character_snapshot: character,
                  message_content: dmContent,
                  triggered_by_outreach: true,
                }),
              });
            }
            await fetchDMThreads();
            addLog('agentic', 'referral_sent', { character: character.name, detail: `Referred ${targetName} to the user` });
          } catch (e) {
            addLog('agentic', 'referral_failed', { character: character.name, success: false, detail: 'Referral failed', error: e.message });
          }
        }

        if (actionResult.chosen_action === 'share_media' && actionResult.content) {
          try {
            const existingThread = dmThreads.find(t => t.character_id === character.id || t.character_name === character.name);
            if (existingThread) {
              await fetch(`${apiUrl}/lattice/dm-thread/${existingThread.id}/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role: 'character', content: actionResult.content, character_name: character.name }),
              });
            } else {
              await fetch(`${apiUrl}/lattice/dm-threads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  character_name: character.name,
                  character_avatar: character.avatar || null,
                  character_id: character.id || null,
                  character_snapshot: character,
                  message_content: actionResult.content,
                  triggered_by_outreach: true,
                }),
              });
            }
            await fetchDMThreads();
            addLog('agentic', 'media_shared', { character: character.name, detail: `Shared media: ${actionResult.content.substring(0, 100)}` });
          } catch (e) {
            addLog('agentic', 'media_share_failed', { character: character.name, success: false, detail: 'Media share failed', error: e.message });
          }
        }

        if (actionResult.chosen_action === 'answer_icebreaker' && actionResult.content) {
          try {
            const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
            await fetch(`${apiUrl}/lattice/feed-post`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_id: character.id,
                character_name: character.name,
                character_avatar: character.avatar || null,
                content: actionResult.content,
                section: sectionAffinity,
                is_icebreaker: true,
                mood: 'icebreaker_response',
                character_snapshot: character,
              }),
            });
            await fetchFeed();
            addLog('agentic', 'icebreaker_answered', { character: character.name, detail: `Answered icebreaker: ${actionResult.content.substring(0, 100)}` });
          } catch (e) {
            addLog('agentic', 'icebreaker_answer_failed', { character: character.name, success: false, detail: 'Icebreaker answer failed', error: e.message });
          }
        }

        if (actionResult.chosen_action === 'reflect' && actionResult.content) {
          addLog('agentic', 'reflection', { character: character.name, detail: `Reflection: ${actionResult.content.substring(0, 200)}` });
        }

        if (actionResult.chosen_action === 'evaluate_pool' && actionResult.content) {
          addLog('agentic', 'pool_evaluation', { character: character.name, detail: `Evaluated pool: ${actionResult.content.substring(0, 200)}` });

          const evalText = actionResult.content.toLowerCase();
          const rivalNames = poolCharacters
            .filter(c => c.id !== character.id && evalText.includes(c.name.toLowerCase()))
            .map(c => c.name);
          if (rivalNames.length > 0 && character.id) {
            const jealousyLevel = computeJealousyLevel(character.id);
            const roll = Math.random() * 100 + jealousyLevel * 0.3;
            try {
              if (roll < 40) {
                const dramaSection = (character.dating_profile?.section_affinity || [])[0] || '';
                const dramaPrompt = `You feel a twinge of competitive energy after noticing ${rivalNames[0]} in the pool. Write a passive-aggressive subtle feed post that expresses your feelings without naming names.`;
                const tickResp = await fetch(`${apiUrl}/lattice/agentic-tick`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    model_name: primaryModel || null,
                    actor_type: 'female_ai',
                    action_type: 'full',
                    character_name: character.name,
                    character_profile: character,
                    memory_entries: [],
                    pool_summary: '',
                    dummy_activity: '',
                    dummy_realism: dummyRealism,
                    dummy_agency: dummyAgency,
                    user_activity: dramaPrompt,
                    available_actions: ['create_feed_post'],
                    user_dating_profile: null,
                    frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
                  }),
                });
                const tickResult = await tickResp.json();
                if (tickResult.status === 'success' && tickResult.action_result?.content) {
                  await fetch(`${apiUrl}/lattice/feed-post`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      character_id: character.id,
                      character_name: character.name,
                      character_avatar: character.avatar || null,
                      content: tickResult.action_result.content,
                      section: dramaSection,
                      mood: 'jealousy',
                      character_snapshot: character,
                    }),
                  });
                  await fetchFeed();
                }
              } else if (roll < 70) {
                const dramaPrompt = `You noticed ${rivalNames[0]} is getting attention from the user. Send a message to the user that subtly mentions ${rivalNames[0]} as a rival — could be competitive, dismissive, or curious.`;
                const msgResp = await fetch(`${apiUrl}/lattice/agentic-tick`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    model_name: primaryModel || null,
                    actor_type: 'female_ai',
                    action_type: 'full',
                    character_name: character.name,
                    character_profile: character,
                    memory_entries: [],
                    pool_summary: '',
                    dummy_activity: '',
                    dummy_realism: dummyRealism,
                    dummy_agency: dummyAgency,
                    user_activity: dramaPrompt,
                    available_actions: ['send_message'],
                    user_dating_profile: null,
                    frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
                  }),
                });
                const msgResult = await msgResp.json();
                if (msgResult.status === 'success' && msgResult.action_result?.content) {
                  const existingThread = dmThreads.find(t => t.character_id === character.id || t.character_name === character.name);
                  if (existingThread) {
                    await fetch(`${apiUrl}/lattice/dm-thread/${existingThread.id}/message`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ role: 'character', content: msgResult.action_result.content, character_name: character.name }),
                    });
                  } else {
                    await fetch(`${apiUrl}/lattice/dm-threads`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        character_name: character.name,
                        character_avatar: character.avatar || null,
                        character_id: character.id || null,
                        character_snapshot: character,
                        message_content: msgResult.action_result.content,
                        triggered_by_outreach: true,
                      }),
                    });
                  }
                  await fetchDMThreads();
                }
              } else {
                const dramaPrompt = `You feel competitive after noticing ${rivalNames[0]} in the pool. Update your dating profile to make yourself more distinctive and competitive — emphasize your best qualities compared to ${rivalNames[0]}.`;
                const profileResp = await fetch(`${apiUrl}/lattice/agentic-tick`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    model_name: primaryModel || null,
                    actor_type: 'female_ai',
                    action_type: 'full',
                    character_name: character.name,
                    character_profile: character,
                    memory_entries: [],
                    pool_summary: '',
                    dummy_activity: '',
                    dummy_realism: dummyRealism,
                    dummy_agency: dummyAgency,
                    user_activity: dramaPrompt,
                    available_actions: ['update_profile'],
                    user_dating_profile: null,
                    frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
                  }),
                });
                const profileResult = await profileResp.json();
                if (profileResult.status === 'success' && profileResult.action_result?.content) {
                  const profileUpdate = profileResult.action_result.content;
                  if (typeof profileUpdate === 'object') {
                    const updatedChar = { ...character, dating_profile: { ...(character.dating_profile || {}), ...profileUpdate } };
                    await saveCharacter(updatedChar);
                  }
                }
              }
              addLog('agentic', 'jealousy_reaction', { character: character.name, detail: `Jealousy reaction triggered (level ${Math.round(jealousyLevel)}) toward ${rivalNames[0] || 'unknown rival'}` });
            } catch (e) {
              addLog('agentic', 'jealousy_reaction_failed', { character: character.name, success: false, detail: 'Jealousy reaction failed', error: e.message });
            }
          }
        }

        return actionResult;
      }
      return null;
    } catch (e) {
      addLog('agentic', 'tick_failed', { character: character?.name, success: false, detail: 'Agentic tick failed', error: e.message });
      console.warn("The AI character's autonomous thinking cycle failed. They couldn't decide what to do.", e);
      return null;
    }
  }, [apiUrl, primaryModel, settings, getPoolNames, saveCharacter, dummyRealism, dummyAgency, userDatingProfile, addLog, mirrorEnabled, userProfile, feedPosts, fetchFeed, fetchDMThreads, dmThreads, activeBreakout]);

  const runTickForAll = useCallback(async () => {
    if (poolCharacters.length === 0) return;
    for (const character of poolCharacters) {
      if (mutedCharacterIds.has(character.id)) continue;
      await runAgenticTick(character);
    }
  }, [poolCharacters, runAgenticTick, mutedCharacterIds]);

  const runTickSingle = useCallback(async (characterId) => {
    const character = poolCharacters.find(c => c.id === characterId);
    if (!character) return null;
    return await runAgenticTick(character);
  }, [poolCharacters, runAgenticTick]);

  const tickRefs = useRef({ mutedCharacterIds: new Set(), poolCharacters: [] });
  tickRefs.current = { mutedCharacterIds, poolCharacters };
  useEffect(() => {
    if (tickEnabled && mirrorEnabled && poolCharacters.length > 0) {
      tickIntervalRef.current = setInterval(() => {
        const refs = tickRefs.current;
        const activeChars = refs.poolCharacters.filter(c => !refs.mutedCharacterIds.has(c.id));
        if (activeChars.length === 0) return;
        const randomChar = activeChars[Math.floor(Math.random() * activeChars.length)];
        runAgenticTick(randomChar);
      }, tickIntervalMs);
    }
    return () => {
      if (tickIntervalRef.current) {
        clearInterval(tickIntervalRef.current);
        tickIntervalRef.current = null;
      }
    };
  }, [tickEnabled, mirrorEnabled, tickIntervalMs, poolCharacters.length, mutedCharacterIds.size, runAgenticTick]);
  // mutedCharacterIds.size instead of Set reference — prevents interval restart on mute toggle

  const dummyTickRef = useRef(null);
  const runDummyTick = useCallback(async () => {
    const modelName = primaryModel || null;
    const dummyIds = ['dummy_marcus', 'dummy_liam', 'dummy_rafe', 'dummy_khalid', 'dummy_ethan'];
    const randomDummy = dummyIds[Math.floor(Math.random() * dummyIds.length)];
    try {
      await fetch(`${apiUrl}/lattice/agentic-tick`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          actor_type: 'dummy_rival',
          action_type: 'quick',
          character_name: randomDummy.replace('dummy_', '').replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
          character_profile: { id: randomDummy },
          memory_entries: [],
          pool_summary: '',
          dummy_activity: 'Pool is active',
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          available_actions: ['send_message', 'react'],
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
      });
      addLog('dummy', 'dummy_tick', { character: randomDummy, detail: 'Dummy rival activity tick completed' });
    } catch (e) {
      addLog('dummy', 'dummy_tick_failed', { success: false, detail: 'Dummy rival tick failed', error: e.message });
      console.warn("A dummy rival's activity cycle failed.", e);
    }
  }, [apiUrl, primaryModel, settings, dummyRealism, dummyAgency, addLog]);

  useEffect(() => {
    if (tickEnabled && mirrorEnabled && dummyRealism > 0) {
      dummyTickRef.current = setInterval(() => {
        runDummyTick();
      }, tickIntervalMs * 2);
    }
    return () => {
      if (dummyTickRef.current) {
        clearInterval(dummyTickRef.current);
        dummyTickRef.current = null;
      }
    };
  }, [tickEnabled, mirrorEnabled, tickIntervalMs, dummyRealism, runDummyTick]);

  const autoGenTimerRef = useRef(null);
  const autoGenStartedRef = useRef(false);
  const userProfileRef = useRef(userDatingProfile);
  userProfileRef.current = userDatingProfile;
  const avatarPoolEnabledRef = useRef(useAvatarPool);
  avatarPoolEnabledRef.current = useAvatarPool;
  const poolUrlsRef = useRef(poolAvatarUrls);
  poolUrlsRef.current = poolAvatarUrls;
  // Keep ref in sync so the interval doesn't capture stale closures
  useEffect(() => { generateEntityRef.current = generateEntity; }, [generateEntity]);
  useEffect(() => {
    if (autoGenerate && mirrorEnabled) {
      const profile = userProfileRef.current;
      const avatarPoolOn = avatarPoolEnabledRef.current;
      const urls = poolUrlsRef.current;
      if (!profile?.bio?.length) {
        addLog('system', 'auto_generate_skipped', { detail: 'Auto-generate skipped: no dating profile bio. Fill out My Profile tab.' });
        return;
      }
      if (!avatarPoolOn || urls.length === 0) {
        addLog('system', 'auto_generate_skipped', { detail: 'Auto-generate skipped: avatar pool empty or disabled. Upload avatars in Avatar Pool section.' });
        return;
      }
      const run = async () => {
        const randomSection = SECTIONS[Math.floor(Math.random() * SECTIONS.length)];
        await generateEntityRef.current(randomSection);
      };
      if (!autoGenStartedRef.current) {
        autoGenStartedRef.current = true;
        run();
      }
      autoGenTimerRef.current = setInterval(async () => {
        if (!autoGenRef.current) return;
        const p = userProfileRef.current;
        const aOn = avatarPoolEnabledRef.current;
        const u = poolUrlsRef.current;
        if (!p?.bio?.length || !aOn || u.length === 0) {
          addLog('system', 'auto_generate_skipped', { detail: 'Auto-generate skipped: prerequisite check failed in interval' });
          return;
        }
        const randomSection = SECTIONS[Math.floor(Math.random() * SECTIONS.length)];
        await generateEntityRef.current(randomSection);
      }, autoGenIntervalMs);
    }
    return () => {
      if (!autoGenerate || !mirrorEnabled) {
        autoGenStartedRef.current = false;
      }
      if (autoGenTimerRef.current) {
        clearInterval(autoGenTimerRef.current);
        autoGenTimerRef.current = null;
      }
    };
  }, [autoGenerate, mirrorEnabled, autoGenIntervalMs, addLog]);
  // Only autoGenerate, mirrorEnabled, autoGenIntervalMs in deps to avoid re-triggering
  // on avatar uploads or bio changes. Prerequisites checked via refs inside the callback.


  const randomActivityTimerRef = useRef(null);
  const randomActivityRefs = useRef({
    poolCharacters: [],
    mutedCharacterIds: new Set(),
    generateFeedPost: null,
    createStory: null,
    settings: null,
    apiUrl: '',
    primaryModel: null,
    userProfile: null,
    userDatingProfile: null,
  });
  // Keep refs in sync so the stable timeout callback reads fresh values
  randomActivityRefs.current = {
    poolCharacters,
    mutedCharacterIds,
    generateFeedPost,
    createStory,
    settings,
    apiUrl,
    primaryModel,
    userProfile,
    userDatingProfile,
  };

  const scheduleNextActivity = useCallback(() => {
    const min = 25 * 60 * 1000;
    const max = 120 * 60 * 1000;
    const delay = min + Math.random() * (max - min);
    randomActivityTimerRef.current = setTimeout(async () => {
      if (!mirrorEnabledRef.current) return;
      const refs = randomActivityRefs.current;
      const chars = refs.poolCharacters;
      if (chars.length === 0) return;
      const activeChars = chars.filter(c => !refs.mutedCharacterIds.has(c.id));
      if (activeChars.length === 0) return;
      const randomChar = activeChars[Math.floor(Math.random() * activeChars.length)];
      const modelName = refs.primaryModel || null;
      const rand = Math.random();
      if (rand < 0.3) {
        addLog('system', 'random_activity_story', { character: randomChar.name, detail: 'Random activity: creating story' });
        try {
          const resp = await fetch(`${refs.apiUrl}/lattice/agentic-tick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_name: modelName,
              actor_type: 'female_ai',
              action_type: 'full',
              character_name: randomChar.name,
              character_profile: randomChar,
              memory_entries: [],
              pool_summary: '',
              dummy_activity: 'Pool is active',
              dummy_realism: 50,
              dummy_agency: 50,
              available_actions: ['create_story'],
              user_dating_profile: null,
              frontend_round_robin_enabled: refs.settings?.apiEndpointRoundRobinEnabled === true,
            }),
          });
          const result = await resp.json();
          if (result.status === 'success' && result.action_result?.content) {
            await refs.createStory(randomChar, result.action_result.content);
          }
        } catch (e) {
          console.warn("Random story creation failed.", e);;
        }
      } else if (rand < 0.45) {
        addLog('system', 'random_activity_character_interaction', { character: randomChar.name, detail: 'Random activity: replying to another character' });
        try {
          const targetPost = (refs.poolCharacters.length > 0)
            ? (feedPosts || []).filter(p => !p.is_user && p.character_name !== randomChar.name && p.character_name).reverse()[0]
            : null;
          if (targetPost?.id) {
            const resp = await fetch(`${refs.apiUrl}/lattice/agentic-tick`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model_name: modelName,
                actor_type: 'female_ai',
                action_type: 'full',
                character_name: randomChar.name,
                character_profile: randomChar,
                memory_entries: [],
                pool_summary: '',
                dummy_activity: 'Pool is active',
                dummy_realism: 50,
                dummy_agency: 50,
                available_actions: ['interact_with_character'],
                user_dating_profile: null,
                frontend_round_robin_enabled: refs.settings?.apiEndpointRoundRobinEnabled === true,
              }),
            });
            const result = await resp.json();
            if (result.status === 'success' && result.action_result?.content) {
              const sectionAffinity = (randomChar.dating_profile?.section_affinity || [])[0] || '';
              await fetch(`${refs.apiUrl}/lattice/character-feed-reply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  post_id: targetPost.id,
                  character_name: randomChar.name,
                  character_avatar: randomChar.avatar || null,
                  character_profile: randomChar,
                  target_character_name: targetPost.character_name || '',
                  model_name: modelName,
                  section: sectionAffinity,
                  frontend_round_robin_enabled: refs.settings?.apiEndpointRoundRobinEnabled === true,
                }),
              });
            }
          }
        } catch (e) {
          console.warn("Character interaction during random activity failed.", e);;
        }
      } else if (rand < 0.725) {
        addLog('system', 'random_activity_feed', { character: randomChar.name, detail: 'Random activity: generating feed post' });
        await refs.generateFeedPost(randomChar);
      } else {
        try {
          addLog('system', 'random_activity_outreach', { character: randomChar.name, detail: 'Random activity: sending outreach message' });
          const mems = await getCharacterMemoryContinuity({
            characterId: randomChar.id,
            characterName: randomChar.name,
            apiUrl: refs.apiUrl,
            userId: refs.userProfile?.id,
            limit: 200,
          });
          const memTexts = mems.map(m => {
            if (typeof m === 'string') return m;
            let text = '';
            try {
              const raw = typeof m.content === 'string' ? m.content : '';
              text = raw || '';
            } catch { text = m.content || ''; }
            const ts = m.timestamp || m.created_at || '';
            const timeStr = ts ? ts.slice(0, 19).replace('T', ' ') + ' | ' : '';
            return timeStr + text;
          }).filter(Boolean);
          const resp = await fetch(`${refs.apiUrl}/lattice/agentic-tick`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_name: modelName,
              actor_type: 'female_ai',
              action_type: 'quick',
              character_name: randomChar.name,
              character_profile: randomChar,
              memory_entries: memTexts,
              pool_summary: '',
              dummy_activity: 'Pool is active',
              dummy_realism: 50,
              dummy_agency: 50,
              available_actions: ['send_message'],
              user_name: getUserName({ userProfile: refs.userProfile, userDatingProfile: refs.userDatingProfile }),
              frontend_round_robin_enabled: refs.settings?.apiEndpointRoundRobinEnabled === true,
            }),
          });
          const result = await resp.json();
          if (result.status === 'success' && result.action_result?.content) {
            const content = result.action_result.content;
            await fetch(`${refs.apiUrl}/lattice/dm-threads`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_name: randomChar.name,
                character_avatar: randomChar.avatar || null,
                character_id: randomChar.id || null,
                character_snapshot: randomChar,
                message_content: content,
                triggered_by_outreach: true,
              }),
            });
            if (randomChar.id) {
              lastDMActivityRef.current[randomChar.id] = Date.now();
            }
            await fetch(`${refs.apiUrl}/lattice/outreach-push`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                character_name: randomChar.name,
                character_avatar: randomChar.avatar || null,
                message_content: content,
                character_snapshot: randomChar,
              }),
            });
          }
        } catch (e) {
          console.warn("A random outreach message from a character failed. Use [Force Activity Tick] in System Control to retry.", e);;
        }
      }
      scheduleNextActivity();
    }, delay);
  }, [addLog]); // Only depends on addLog (stable). All other values read from refs.

  useEffect(() => {
    if (mirrorEnabled && poolCharacters.length > 0) {
      scheduleNextActivity();
    }
    return () => {
      if (randomActivityTimerRef.current) {
        clearTimeout(randomActivityTimerRef.current);
        randomActivityTimerRef.current = null;
      }
    };
  }, [mirrorEnabled, poolCharacters.length, mutedCharacterIds.size, scheduleNextActivity]);
  // mutedCharacterIds.size instead of mutedCharacterIds Set reference
  // poolCharacters.length instead of poolCharacters array reference

  const toggleMirror = useCallback(() => {
    setMirrorEnabled(prev => {
      const next = !prev;
      if (!next) {
        clearInterval(tickIntervalRef.current); tickIntervalRef.current = null;
        clearInterval(dummyTickRef.current); dummyTickRef.current = null;
        clearInterval(autoGenTimerRef.current); autoGenTimerRef.current = null;
        clearTimeout(randomActivityTimerRef.current); randomActivityTimerRef.current = null;
      }
      addLog('system', next ? 'mirror_enabled' : 'mirror_disabled', { detail: next ? 'Mirror AI Dating enabled' : 'Mirror AI Dating disabled' });
      return next;
    });
  }, [addLog]);

  const generateDummyRival = useCallback(async () => {
    const modelName = primaryModel || null;
    try {
      const response = await fetch(`${apiUrl}/lattice/generate-dummy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          dummy_realism: dummyRealism,
          dummy_agency: dummyAgency,
          frontend_round_robin_enabled: settings?.apiEndpointRoundRobinEnabled === true,
        }),
      });
      const result = await response.json();
      if (result.status === 'success' && result.dummy) {
        setGeneratedDummies(prev => [...prev, result.dummy]);
        addLog('dummy', 'rival_generated', { character: result.dummy.name, detail: `Created dummy rival ${result.dummy.name}` });
      }
      return result;
    } catch (e) {
      addLog('dummy', 'rival_generation_failed', { success: false, detail: 'Dummy rival generation failed', error: e.message });
      console.warn("Dummy rival generation failed. Try [Force Dummy Tick] in System Control.", e);;
      return null;
    }
  }, [apiUrl, primaryModel, dummyRealism, dummyAgency, settings, addLog]);

  const saveUserProfile = useCallback((profile) => {
    const updated = saveUserDatingProfile(profile);
    setUserDatingProfile(updated);
    addLog('dating', 'profile_saved', { detail: 'User dating profile updated' });
    return updated;
  }, [addLog]);
  const replyToPost = useCallback(async (postId, userReply, section) => {
    try {
      const modelName = primaryModel || null;
      const userName = getUserName({ userProfile, userDatingProfile });
      const response = await fetch(`${apiUrl}/lattice/feed-reply`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          post_id: postId,
          user_reply: userReply,
          model_name: modelName,
          gpu_id: 0,
          section_hint: section,
          user_name: userName,
          user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
        }),
      });
      const result = await response.json();
      if (result.status === 'success') {
        addLog('feed', 'reply_sent', { detail: `Replied to post ${postId}` });
        // Log social awareness to other characters
        const post = feedPosts.find(p => p.id === postId);
        if (post?.character_id) {
          const otherCharIds = poolCharacters.filter(c => c.id !== post.character_id).map(c => c.id).filter(Boolean);
          const userName = getUserName({ userProfile, userDatingProfile });
          for (const otherId of otherCharIds) {
            try {
              logInteractionAPI({
                characterId: otherId,
                entryType: 'social_awareness',
                surface: 'feed_reply',
                context: `${userName} replied to ${post.character_name}'s feed post.`,
                targetCharacter: post.character_name,
              });
            } catch {}
          }
        }
        await fetchFeed();
        return result;
      }
    } catch (e) {
      addLog('feed', 'reply_failed', { success: false, detail: 'Feed reply failed', error: e.message });
      console.warn("Couldn't send your reply to the feed post. Try replying again or refreshing the feed.", e);;
    }
    return null;
  }, [apiUrl, primaryModel, addLog, feedPosts, userProfile, userDatingProfile]);

  const createUserFeedPost = useCallback(async (content, section = '') => {
    if (!content?.trim()) return null;
    try {
      const userName = getUserName({ userProfile, userDatingProfile });
      const response = await fetch(`${apiUrl}/lattice/user-feed-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: content.trim(),
          section,
          character_name: userName,
          character_avatar: userProfile?.avatar || '',
        }),
      });
      const result = await response.json();
      if (result.status === 'success') {
        addLog('feed', 'user_posted', { detail: 'User posted to feed, triggering character responses...', success: true });
        await fetchFeed();
        // Log social awareness to all characters
        const allCharIds = poolCharacters.map(c => c.id).filter(Boolean);
        for (const charId of allCharIds) {
          try {
            logInteractionAPI({
              characterId: charId,
              entryType: 'social_awareness',
              surface: 'feed_post',
              context: `${userName} posted to the feed: "${content.trim().slice(0, 100)}"`,
            });
          } catch {}
        }
        const eligible = poolCharacters.filter(c => !mutedCharacterIds.has(c.id));
        const shuffled = eligible.sort(() => Math.random() - 0.5);
        const responders = shuffled.slice(0, 4);
        for (const char of responders) {
          try {
            const replyResp = await fetch(`${apiUrl}/lattice/user-feed-reply`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                post_id: result.post.id,
                character_name: char.name,
                character_avatar: char.avatar || '',
                character_profile: char,
                model_name: primaryModel || null,
                user_name: userName,
                user_dating_profile: userDatingProfile || null,
          user_profile: userProfile || null,
              }),
            });
            const replyResult = await replyResp.json();
          } catch {}
        }
        return result.post;
      }
    } catch (e) {
      addLog('feed', 'user_post_failed', { success: false, detail: 'User feed post failed', error: e.message });
      console.warn("Couldn't post to feed.", e);
    }
    return null;
  }, [apiUrl, addLog, fetchFeed, poolCharacters, mutedCharacterIds, primaryModel, userProfile]);

  const deleteFeedPost = useCallback(async (postId) => {
    if (!postId) return;
    try {
      await fetch(`${apiUrl}/lattice/feed-post/${postId}`, { method: 'DELETE' });
      addLog('feed', 'post_deleted', { detail: `Deleted feed post ${postId}` });
      await fetchFeed();
    } catch (e) {
      addLog('feed', 'post_delete_failed', { success: false, detail: 'Failed to delete feed post', error: e.message });
    }
  }, [apiUrl, addLog, fetchFeed]);

  const LIKED_KEY = 'Eloquent-mirror-liked-ids';
  const [likedIds, setLikedIds] = useState(() => {
    try { return new Set(JSON.parse(localStorage.getItem(LIKED_KEY) || '[]')); } catch { return new Set(); }
  });
  const toggleLikePost = useCallback(async (postId, replyId = null) => {
    const key = replyId ? `${postId}_${replyId}` : postId;
    const wasLiked = likedIds.has(key);
    try {
      await fetch(`${apiUrl}/lattice/like-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ post_id: postId, reply_id: replyId, by_name: userProfile?.name || 'You', by_avatar: userProfile?.avatar || '' }),
      });
      setLikedIds(prev => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key); else next.add(key);
        try { localStorage.setItem(LIKED_KEY, JSON.stringify([...next])); } catch {}
        return next;
      });
      addLog('feed', wasLiked ? 'post_unliked' : 'post_liked', { detail: `${wasLiked ? 'Unliked' : 'Liked'} ${replyId ? 'reply' : 'post'} ${postId}` });
      fetchFeed();
    } catch {}
  }, [apiUrl, addLog, fetchFeed, likedIds, userProfile]);

  const togglePinPost = useCallback(async (postId) => {
    if (!postId) return;
    try {
      const posts = feedPosts || [];
      const post = posts.find(p => p.id === postId);
      const wasPinned = post?.pinned;
      await fetch(`${apiUrl}/lattice/${wasPinned ? 'unpin' : 'pin'}-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ post_id: postId, character_name: userProfile?.name || 'You' }),
      });
      addLog('feed', wasPinned ? 'post_unpinned' : 'post_pinned', { detail: `${wasPinned ? 'Unpinned' : 'Pinned'} post ${postId}` });
      await fetchFeed();
    } catch {}
  }, [apiUrl, addLog, fetchFeed, userProfile]);

  const sendBreakoutMessage = useCallback(async (character, text, history, { onChunk } = {}) => {
    const modelName = primaryModel || null;
    const userName = getUserName({ userProfile, userDatingProfile });
    const profileText = formatUserProfileForPrompt(userDatingProfile);
    const interactionCtx = character?.id ? await getInteractionContext(character.id, 30) : null;
    const interactionBlock = interactionCtx?.formatted_text
      ? `\n\nYour past interactions with ${userName}:\n${interactionCtx.formatted_text}`
      : '';
    const context = [
      `You are ${character.name}. You are in a timed breakout room with ${userName}.`,
      `Character profile: ${character.description || ''} ${character.personality || ''}`,
      `${userName}'s profile:\n${profileText}`,
      interactionBlock,
      `Conversation so far:`,
      ...(history || []).map(m => `${m.role === 'user' ? userName : character.name}: ${m.content}`),
      `${character.name}, respond naturally. Keep the conversation flowing.`,
    ].join('\n');
    try {
      const response = await fetch(`${apiUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: modelName || '',
          messages: [
            { role: 'system', content: context },
            { role: 'user', content: text },
          ],
          max_tokens: 512,
          temperature: 0.8,
          stream: true,
        }),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';
      let sseBuffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') break;
          let parsed;
          try { parsed = JSON.parse(dataStr); } catch { continue; }
          if (parsed.done || parsed.choices?.[0]?.finish_reason) break;
          const token = parsed.text || parsed.choices?.[0]?.delta?.content || '';
          if (token) {
            accumulated += token;
            onChunk?.(accumulated);
          }
        }
      }
      const reply = accumulated.trim();
      return reply;
    } catch (e) {
      addLog('breakout', 'message_failed', { character: character?.name, success: false, detail: 'Breakout message send failed', error: e.message });
      console.warn("Breakout room message failed to send. Try sending again.", e);;
      return '';
    }
  }, [apiUrl, primaryModel, userDatingProfile, addLog, userProfile]);

  const rateCharacter = useCallback((characterId, rating, review) => {
    if (!characterId) return;
    addLog('breakout', 'character_rated', { character: characterId, detail: `Rated ${rating}/5: "${(review || '').substring(0, 50)}"` });
  }, [addLog]);

  const endBreakoutRoom = useCallback((characterId) => {
    if (!characterId) return;
    recordInteraction(characterId, 'breakout');
    const tomorrow = new Date();
    tomorrow.setHours(23, 59, 59, 999);
    setBreakoutCooldowns(prev => {
      const next = { ...prev, [characterId]: { resetAt: tomorrow.toISOString() } };
      persistBreakoutCooldowns(next);
      return next;
    });
    addLog('breakout', 'room_ended', { character: characterId, detail: 'Breakout room ended — cooldown until midnight' });
    try { recordMilestoneRef.current?.(characterId, 'first_breakout'); } catch {}
    const otherChars = poolCharacters.filter(c => c.id !== characterId && !mutedCharacterIds.has(c.id));
    for (const other of otherChars.slice(0, 3)) {
      setTimeout(() => runTickSingle(other.id), 3000 + Math.random() * 4000);
    }
  }, [addLog, recordInteraction, poolCharacters, mutedCharacterIds, runTickSingle]);

  const generateUserRating = useCallback(async (character, history) => {
    if (!character?.name) return null;
    const modelName = primaryModel || null;
    const summary = (history || []).map(m => `${m.role === 'user' ? 'You' : character.name}: ${m.content}`).join('\n');
    try {
      const response = await fetch(`${apiUrl}/lattice/rate-user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_name: character.name,
          character_profile: character,
          conversation_summary: summary,
          model_name: modelName,
        }),
      });
      const result = await response.json();
      if (result.status === 'success') {
        const ratingData = {
          characterName: character.name,
          characterModel: character.generated_by || '',
          rating: result.rating,
          review: result.review || '',
          conversationType: 'breakout',
        };
        const updated = addUserRating(userDatingProfile, ratingData);
        saveUserDatingProfile(updated);
        setUserDatingProfile(updated);
        addLog('breakout', 'user_rated', { character: character.name, detail: `Character rated you ${result.rating}/5: "${(result.review || '').substring(0, 50)}"` });
        return result;
      }
    } catch (e) {
      addLog('breakout', 'user_rating_failed', { character: character?.name, success: false, detail: 'Character rating generation failed', error: e.message });
      console.warn("The character couldn't rate you after the breakout room. Try [End Breakout Room] again.", e);;
    }
    return null;
  }, [apiUrl, primaryModel, userDatingProfile, addLog]);

  /** Write a compact agentic memory entry about a Mirror event before entering normal chat. */
  const writeMirrorContinuityMemory = useCallback(async (character, eventType, details) => {
    const userId = resolveAgenticUserId();
    if (!userId || !character?.id || !MEMORY_API_URL) return;
    try {
      await fetch(`${MEMORY_API_URL}/memory/agentic/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          character_id: character.id,
          character_name: character.name,
          character_profile: {
            description: character.description || '',
            scenario: character.scenario || '',
          },
          user_message: `[Mirror ${eventType}] Mirror context that led to normal chat: ${details}`,
          ai_response: `[System] The character and user entered this normal chat after a ${eventType} in Mirror AI Dating. The character should remember this context naturally.`,
        }),
      });
    } catch (e) {
      console.warn('[PoolContext] writeMirrorContinuityMemory error:', e);
    }
  }, [MEMORY_API_URL, resolveAgenticUserId]);

  const bookDate = useCallback(async (character, dateType) => {
    if (!character) return;
    if (character.id) recordInteraction(character.id, 'date');
    const profileText = formatUserProfileForPrompt(userDatingProfile);
    const modelName = primaryModel || null;
    let reply = '';
    try {
        const greeting = `Hey ${character.name}, I'd like to talk properly. ${dateType === 'formal' ? "Would you be free for a date?" : dateType === 'neural_sex' ? "I'm interested in a neural sex session with you." : "I want to continue our conversation properly."}`;
      const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName || '',
          messages: [
            { role: 'system', content: `You are ${character.name}. The user wants to book a ${dateType} with you. Respond in your authentic voice. Be enthusiastic and genuine. Your dating profile: ${JSON.stringify(character.dating_profile || {})}. User's profile: ${profileText}` },
            { role: 'user', content: greeting },
          ],
          max_tokens: 256,
          temperature: 0.8,
          stream: false,
        }),
      });
      const result = await response.json();
      reply = result?.choices?.[0]?.message?.content || result?.content || '';
      character.first_message = reply || `I'd love to. Let's talk properly.`;
      addLog('breakout', 'date_booked', { character: character.name, detail: `Booked a ${dateType} date` });
      if (dateType === 'neural_sex' && character.id) {
        recordMilestone(character.id, 'neural_sex');
      } else if (character.id) {
        recordMilestone(character.id, 'first_date');
      }
    } catch (e) {
      addLog('breakout', 'date_booking_failed', { character: character?.name, success: false, detail: 'Date booking failed', error: e.message });
      character.first_message = `I'd love to. Let's talk properly.`;
      reply = character.first_message;
    }
    if (character.id) await saveCharacter(character);
    // Log social awareness to all other characters
    const otherCharIds = poolCharacters.filter(c => c.id !== character.id).map(c => c.id).filter(Boolean);
    if (otherCharIds.length > 0) {
      try {
        const userName = getUserName({ userProfile, userDatingProfile });
        for (const otherId of otherCharIds) {
          logInteractionAPI({
            characterId: otherId,
            entryType: 'social_awareness',
            surface: dateType === 'neural_sex' ? 'neural_sex' : 'date',
            context: `${userName} went on a ${dateType} date with ${character.name}.`,
            targetCharacter: character.name,
          });
        }
      } catch (e) {
        console.warn("Failed to log social awareness for date:", e);
      }
    }
    // Write continuity memory so normal chat remembers the Mirror context
    await writeMirrorContinuityMemory(character, dateType === 'neural_sex' ? 'neural_sex_session' : 'date_booking', `The user booked a ${dateType} date with ${character.name} in Mirror AI Dating. ${character.name} accepted and this normal chat continues from there.`);
    // Create a real normal chat conversation atomically
    startCharacterConversation(character, {
      firstMessage: reply,
      dateType,
      conversationName: dateType === 'neural_sex' ? `Neural Sex with ${character.name}` : `Date with ${character.name}`,
      mirrorContinuity: {
        origin: 'mirror_date',
        dateType,
        startedAt: new Date().toISOString(),
      },
    });
  }, [apiUrl, primaryModel, userDatingProfile, saveCharacter, addLog, startCharacterConversation, writeMirrorContinuityMemory]);

  const startSpeedDating = useCallback(async () => {
    const eligible = poolCharacters.filter(c => c.dating_profile?.section_affinity?.length > 0);
    if (eligible.length < 2) {
      addLog('speedDating', 'not_enough_characters', { detail: 'Need at least 2 characters with profiles' });
      return;
    }
    const shuffled = [...eligible].sort(() => Math.random() - 0.5);
    const selected = shuffled.slice(0, Math.min(4, shuffled.length));
    const session = {
      active: true,
      characters: selected.map(c => ({ ...c })),
      currentRound: 0,
      totalRounds: selected.length,
      ratings: {},
      messages: [],
      startTime: Date.now(),
      roundStartTime: Date.now(),
      roundDuration: 180000,
    };
    setSpeedDatingSession(session);
    addLog('speedDating', 'session_started', { detail: `Speed dating with ${selected.length} characters` });
    return session;
  }, [poolCharacters, addLog]);

  const rateSpeedDatingRound = useCallback((characterId, rating) => {
    if (!speedDatingSession) return;
    const updatedRatings = { ...speedDatingSession.ratings, [characterId]: rating };
    const nextRound = speedDatingSession.currentRound + 1;
    if (nextRound >= speedDatingSession.totalRounds) {
      const topMatch = Object.entries(updatedRatings).sort((a, b) => b[1] - a[1])[0];
      setSpeedDatingSession(prev => ({
        ...prev,
        ratings: updatedRatings,
        currentRound: nextRound,
        topMatchId: topMatch?.[0] || null,
        complete: true,
      }));
    } else {
      setSpeedDatingSession(prev => ({
        ...prev,
        ratings: updatedRatings,
        currentRound: nextRound,
        roundStartTime: Date.now(),
        messages: [],
      }));
    }
    addLog('speedDating', 'round_rated', { character: characterId, detail: `Rated ${rating}/5` });
  }, [speedDatingSession, addLog]);

  const closeSpeedDating = useCallback(() => {
    setSpeedDatingSession(null);
  }, []);

  const startGroupChat = useCallback((charIds, topic) => {
    const selected = poolCharacters.filter(c => charIds.includes(c.id));
    if (selected.length < 2) {
      addLog('groupChat', 'not_enough_characters', { detail: 'Need at least 2 characters' });
      return;
    }
    setActiveGroupChat({
      active: true,
      characters: selected,
      messages: [],
      currentSpeakerIndex: 0,
      autoMode: false,
      topic: topic || '',
    });
    addLog('groupChat', 'session_started', { detail: `Group chat with ${selected.map(c => c.name).join(', ')}` });
  }, [poolCharacters, addLog]);

  const sendGroupMessage = useCallback(async (text, role = 'user') => {
    if (!activeGroupChat || !text) return;
    const userMsg = { id: `g-${Date.now()}`, role, content: text, characterName: null, created_at: new Date().toISOString() };
    const updatedMessages = [...activeGroupChat.messages, userMsg];
    setActiveGroupChat(prev => ({ ...prev, messages: updatedMessages, currentSpeakerIndex: 0 }));

    for (let i = 0; i < activeGroupChat.characters.length; i++) {
      const char = activeGroupChat.characters[i];
      const context = [
        `You are ${char.name} in a group conversation.`,
        `Your profile: ${char.description || ''} ${char.personality || ''}`,
        `Group topic: ${activeGroupChat.topic || 'Casual conversation'}`,
        `Participants: ${activeGroupChat.characters.map(c => c.name).join(', ')} and the user.`,
        'Conversation so far:',
        ...updatedMessages.map(m => `${m.role === 'user' ? 'You (the user)' : m.characterName || char.name}: ${m.content}`),
        `${char.name}, respond naturally as yourself. Keep it brief and authentic.`,
      ].join('\n');
      try {
        const resp = await fetch(`${apiUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: primaryModel || '',
            messages: [{ role: 'system', content: context }, { role: 'user', content: `${char.name}, what do you say?` }],
            max_tokens: 256,
            temperature: 0.8,
            stream: false,
          }),
        });
        const result = await resp.json();
        const reply = result?.choices?.[0]?.message?.content || '';
        if (reply) {
          const charMsg = { id: `g-${Date.now()}-${i}`, role: 'character', content: reply, characterName: char.name, characterAvatar: char.avatar, created_at: new Date().toISOString() };
          updatedMessages.push(charMsg);
          setActiveGroupChat(prev => ({ ...prev, messages: [...updatedMessages], currentSpeakerIndex: i + 1 }));
        }
      } catch (e) {
        addLog('groupChat', 'message_failed', { character: char.name, detail: 'Group chat response failed', error: e.message });
      }
    }
    setActiveGroupChat(prev => ({ ...prev, messages: updatedMessages, currentSpeakerIndex: 0 }));
  }, [activeGroupChat, apiUrl, primaryModel, addLog]);

  const closeGroupChat = useCallback(() => {
    setActiveGroupChat(null);
  }, []);

  const fetchIcebreakers = useCallback(async () => {
    try {
      const resp = await fetch(`${apiUrl}/lattice/icebreakers`);
      const data = await resp.json();
      if (data?.icebreakers) setIcebreakers(data.icebreakers);
    } catch {}
  }, [apiUrl]);

  const createIcebreaker = useCallback(async (character) => {
    if (!character) return;
    const sectionAffinity = (character.dating_profile?.section_affinity || [])[0] || '';
    try {
      const resp = await fetch(`${apiUrl}/lattice/icebreaker`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_name: character.name,
          character_avatar: character.avatar || null,
          character_id: character.id || null,
          character_snapshot: character,
          section: sectionAffinity,
        }),
      });
      const result = await resp.json();
      if (result.status === 'success') {
        addLog('icebreaker', 'created', { character: character.name, detail: 'Icebreaker question posted to feed' });
        await fetchFeed();
      }
    } catch (e) {
      addLog('icebreaker', 'create_failed', { character: character?.name, success: false, detail: 'Icebreaker creation failed', error: e.message });
    }
  }, [apiUrl, addLog, fetchFeed]);

  const playFeedPostTTS = useCallback(async (character, text) => {
    if (!character?.voice_id || !text) return;
    try {
      const ttsUrl = await synthesizeSpeech(text, {
        engine: settings?.ttsEngine || 'chatterbox',
        audio_prompt_path: character.voice_id,
      });
      if (ttsUrl) {
        const audio = new Audio(ttsUrl);
        audio.play();
        addLog('tts', 'tts_played', { character: character.name, detail: 'TTS playback started' });
      }
    } catch (e) {
      addLog('tts', 'tts_failed', { character: character?.name, success: false, detail: 'TTS playback failed', error: e.message });
      console.warn("Couldn't play the text-to-speech audio. The character may not have a voice configured.", e);;
    }
  }, [addLog, settings?.ttsEngine]);

  const importCharacterToPool = useCallback(async (character) => {
    if (!character) return;
    const charId = character.id || character.name;
    if (!charId) { addLog('pool', 'import_failed', { character: character.name, detail: 'Character has no ID' }); return; }
    if (character.dating_profile?.section_affinity?.length > 0) {
      addLog('pool', 'import_skipped', { character: character.name, detail: 'Already in pool' });
      return;
    }
    const updated = {
      ...character,
      dating_profile: {
        ...(character.dating_profile || {}),
        section_affinity: ['Intimate'],
        bio: character.dating_profile?.bio || '',
        seeking: '',
      },
    };
    try {
      await saveCharacter(updated);
      updateMilestone(updated.id || updated.name, 'in_pool', true);
      addLog('pool', 'imported_character', { character: character.name, detail: 'Imported from Character Library into Mirror pool, writing profile...' });
      await initializeCharacterProfile(updated, 'Intimate');
      updateMilestone(updated.id || updated.name, 'profile_written', true);
      addLog('pool', 'import_profile_done', { character: character.name, detail: 'Profile written, posting feed post...' });
      await generateFeedPost(updated);
      updateMilestone(updated.id || updated.name, 'feed_post', true);
      addLog('pool', 'import_feed_post_done', { character: character.name, detail: 'Import complete — character is now active in Mirror pool' });
    } catch (e) {
      addLog('pool', 'import_failed', { character: character.name, detail: 'Import failed', error: e.message });
    }
  }, [saveCharacter, initializeCharacterProfile, generateFeedPost, updateMilestone, addLog]);

  const value = {
    poolCharacters,
    importCharacterToPool,
    getCharactersBySection,
    sections: SECTIONS,
    activeSection,
    setActiveSection,
    viewMode,
    setViewMode,
    selectedCharacter,
    setSelectedCharacter,
    isGenerating,
    generationError,
    generateEntity,
    generateMultiple,
    cancelGeneration,
    deleteCharacter,
    generationLog,
    generationStep,
    agenticActionLog,
    activityLog,
    addActivityEntry,
    dummyRealism,
    setDummyRealism,
    dummyAgency,
    setDummyAgency,
    tickEnabled,
    setTickEnabled,
    tickIntervalMs,
    setTickIntervalMs,
    runTickForAll,
    runTickSingle,
    settings,
    useAvatarPool,
    setUseAvatarPool,
    poolAvatarUrls,
    uploadPoolAvatars,
    removePoolAvatar,
    userDatingProfile,
    saveUserProfile,
    initializeCharacterProfile,
    feedPosts,
    fetchFeed,
    replyToPost,
    createUserFeedPost,
    deleteFeedPost,
    activeBreakout,
    isBreakoutAvailable,
    sendBreakoutMessage,
    bookDate,
    rateCharacter,
    endBreakoutRoom,
    generateUserRating,
    neuralSexCharacter,
    setNeuralSexCharacter,
    showNeuralSex,
    setShowNeuralSex,
    playFeedPostTTS,
    autoGenerate,
    setAutoGenerate,
    autoGenIntervalMs,
    setAutoGenIntervalMs,
    generatedDummies,
    generateDummyRival,
    characterMilestones,
    mirrorEnabled,
    toggleMirror,
    mutedCharacterIds,
    toggleMuteCharacter,
    isCharacterMuted,
    likedIds,
    toggleLikePost,
    togglePinPost,
    generateFeedPost,
    stories,
    fetchStories,
    createStory,
    viewedStoryIds,
    markStoryViewed,
    dmThreads,
    activeDMThread,
    fetchDMThreads,
    createDMThread,
    sendDMMessage,
    selectDMThread,
    selectDMThreadById,
    closeDMThread,
    deleteDMThread,
    deleteAllDMThreads,
    compatibilityScores,
    getCompatibilityScore,
    computeAllCompatibilityScores,
    relationshipMilestones,
    getCharacterMilestones,
    recordMilestone,
    computeJealousyLevel,
    recordInteraction,
    speedDatingSession,
    startSpeedDating,
    rateSpeedDatingRound,
    closeSpeedDating,
    activeGroupChat,
    startGroupChat,
    sendGroupMessage,
    closeGroupChat,
    icebreakers,
    fetchIcebreakers,
    createIcebreaker,
  };

  return <PoolContext.Provider value={value}>{children}</PoolContext.Provider>;
}

export function usePool() {
  const ctx = useContext(PoolContext);
  if (!ctx) throw new Error('usePool must be used within a PoolProvider');
  return ctx;
}
