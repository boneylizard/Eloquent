import React, { memo, useMemo, createContext, useState, useCallback, useContext, useEffect, useRef } from 'react';
import { formatPrompt } from '../utils/chat_templates';
import { cleanModelOutput } from '../utils/cleanOutput';
import { generateChatTitle, fetchTriggeredLore } from '../utils/apiCall';
import { observeConversation, initializeMemories } from '../utils/memoryUtils';
import { saveSummary } from '../utils/summaryUtils';
import { useMemory } from '../contexts/MemoryContext';
import { transcribeAudio, synthesizeSpeech } from '../utils/apiCall';
import { generateReplyOpenAI, processOpenAIStream, generateReplyOpenAINonStreaming, convertToOpenAIMessages } from '../utils/openaiApi';
import { ttsClient } from '../utils/apiCall';
import { processAntiRepetition } from '../utils/antiRepetition';


const AppContext = createContext(null);
const logPromptSample = (prompt, maxLength = 500) => {
  const sample = prompt.length > maxLength ?
    prompt.substring(0, maxLength) + '...' :
    prompt;
  console.log('📝 [DEBUG] Prompt sample:', sample);

  const hasMemories = prompt.includes('USER CONTEXT') ||
    prompt.includes('RELEVANT USER INFORMATION') ||
    prompt.includes('WORLD KNOWLEDGE');

  console.log(`📝 [DEBUG] Prompt contains memory references: ${hasMemories}`);
};
console.log('🔍 [DEBUG] generateChatTitle function imported:', typeof generateChatTitle);
// NEW — talk to your FastAPI on 8001
async function generateAndShowImage(promptText) {
  try {
    // 1) Build the body payload
    const payload = {
      prompt: promptText,
      negative_prompt: "",
      width: 512,
      height: 512,
      steps: 20,
      guidance_scale: 7.0,
      sampler_name: "Euler a",
      seed: -1,
    };

    // 2) Call your FastAPI proxy
    const res = await fetch(`${BACKEND}/sd/txt2img`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`SD API ${res.status}: ${errText}`);
    }

    // 3) Parse the JSON
    const data = await res.json();
    console.log("📝 [DEBUG] Response data:", data);

    // 4) Pull out the image URLs array
    const urls = data.image_urls || [];
    if (!urls.length) {
      console.error("📝 [ERROR] No images returned:", data);
      return;
    }

    // 5) Use the first one
    const imageUrl = urls[0];
    console.log("📝 [DEBUG] Image URL:", imageUrl);

    // 6) Append to the DOM
    // const img = document.createElement("img");
    // img.src = imageUrl;
    // img.alt = promptText;
    // document.body.appendChild(img);

  } catch (err) {
    console.error("📝 [ERROR] generateAndShowImage failed:", err);
  }
}

// Example usage:
//generateAndShowImage("a sexy woman");
// Helper function to generate truly unique IDs
const generateUniqueId = () => `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
const NARRATOR_CHARACTER_ID = '__narrator__';

const normalizeChatRole = (role) => {
  if (role === 'user' || role === 'narrator') return role;
  return 'npc';
};

const normalizeCharacter = (character) => ({
  ...character,
  chat_role: normalizeChatRole(character?.chat_role)
});

const buildDefaultCharacterWeights = (characterList) => {
  const weights = {};
  (characterList || [])
    .map(normalizeCharacter)
    .filter(c => normalizeChatRole(c.chat_role) !== 'user')
    .forEach(c => {
      weights[c.id] = 50;
    });
  return weights;
};

const extractFirstJson = (text) => {
  if (!text) return null;
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (char === '{') depth += 1;
    if (char === '}') depth -= 1;
    if (depth === 0) return text.slice(start, i + 1);
  }
  return null;
};

// Helper function to get story tracker context for injection into prompts
const getStoryTrackerContext = () => {
  try {
    const saved = localStorage.getItem('eloquent-story-tracker');
    if (!saved) return '';

    const tracker = JSON.parse(saved);
    const sections = [];

    if (tracker.currentObjective) {
      sections.push(`[CURRENT OBJECTIVE]: ${tracker.currentObjective}`);
    }

    if (tracker.characters?.length > 0) {
      const chars = tracker.characters.map(c => c.notes ? `${c.value} (${c.notes})` : c.value).join(', ');
      sections.push(`[CHARACTERS]: ${chars}`);
    }

    if (tracker.inventory?.length > 0) {
      const items = tracker.inventory.map(i => i.notes ? `${i.value} (${i.notes})` : i.value).join(', ');
      sections.push(`[INVENTORY]: ${items}`);
    }

    if (tracker.locations?.length > 0) {
      const locs = tracker.locations.map(l => l.notes ? `${l.value} (${l.notes})` : l.value).join(', ');
      sections.push(`[LOCATIONS]: ${locs}`);
    }

    if (tracker.plotPoints?.length > 0) {
      const plots = tracker.plotPoints.map(p => `• ${p.notes ? `${p.value}: ${p.notes}` : p.value}`).join('\n');
      sections.push(`[KEY EVENTS / RECENT HISTORY]:\n${plots}`);
    }

    if (tracker.storyNotes) {
      sections.push(`[STORY NOTES / WORLD BACKGROUND]:\n${tracker.storyNotes}`);
    }

    if (tracker.sceneSummary) {
      sections.push(`[CURRENT SCENE]: ${tracker.sceneSummary}`);
    }

    if (tracker.customFields?.length > 0) {
      const custom = tracker.customFields.map(c => c.notes ? `${c.value}: ${c.notes}` : c.value).join(', ');
      sections.push(`[ADDITIONAL DETAILS]: ${custom}`);
    }

    if (sections.length === 0) return '';

    return `\n\n[STORY TRACKER - Essential continuity guidance for this response]\n${sections.join('\n\n')}`;
  } catch (e) {
    console.error('Error reading story tracker:', e);
    return '';
  }
};

// Helper function to build system prompt from character data
const _buildSystemPrompt = (character, userProfile = null, summaryContextOverride = null, userCharacter = null) => {
  if (!character) return null;

  // Get active summary context if selected
  const summaryContext = summaryContextOverride
    ? `\n\n[PREVIOUS STORY SUMMARY]:\n${summaryContextOverride}\n[End of Summary]\n`
    : (userProfile?.activeContextSummary
      ? `\n\n[PREVIOUS STORY SUMMARY]:\n${userProfile.activeContextSummary}\n[End of Summary]\n`
      : '');

  // Get story tracker context
  const storyContext = getStoryTrackerContext();

  // Tag substitution variables
  const charName = character.name || 'Character';
  const userName = userCharacter?.name || userProfile?.name || userProfile?.username || 'User';

  // Helper to replace tags in a string
  const replaceTags = (text) => {
    if (!text) return '';
    return text
      .replace(/{{char}}/gi, charName)
      .replace(/{{user}}/gi, userName);
  };

  const personality = replaceTags(character.personality);
  const description = replaceTags(character.description);
  const scenario = character.scenario ? replaceTags(character.scenario) : '';
  const speechStyle = character.speech_style ? replaceTags(character.speech_style) : '';
  const background = character.background ? replaceTags(character.background) : '';

  return `You are ${charName}, ${description}.

PERSONALITY: ${personality}

BACKGROUND: ${background}

${scenario ? `SCENARIO: ${scenario}` : ''}

SPEAKING STYLE: ${speechStyle}

IMPORTANT: Stay in character at all times. Respond as ${charName} would, maintaining the defined personality and speech patterns.

${character.example_dialogue && character.example_dialogue.length > 0
      ? `EXAMPLE DIALOGUE:
${character.example_dialogue.map(msg =>
        `${replaceTags(msg.role === 'character' ? charName : (msg.role === 'user' ? userName : 'User'))}: ${replaceTags(msg.content)}`
      ).join('\n')}` : ''}${summaryContext}${storyContext}`;
};


// Helper function to draw avatar on canvas
const drawAvatar = (canvas, imageUrl, name) => {
  const ctx = canvas.getContext('2d');
  const size = canvas.width;

  // Clear canvas
  ctx.clearRect(0, 0, size, size);

  if (imageUrl) {
    const img = new Image();
    img.crossOrigin = "Anonymous"; // To avoid CORS issues
    img.onload = () => {
      // Draw the image, clipped to a circle
      ctx.beginPath();
      ctx.arc(size / 2, size / 2, size / 2, 0, 2 * Math.PI);
      ctx.closePath();
      ctx.clip();
      ctx.drawImage(img, 0, 0, size, size);
    };
    img.src = imageUrl;
  } else {
    // Draw placeholder
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#89b4fa';  // Placeholder color
    ctx.fill();
    ctx.closePath();
    ctx.fillStyle = '#1e1e2e'; // Text color
    ctx.font = `${Math.round(size / 3)}px sans-serif`; // Dynamic font size
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name.charAt(0).toUpperCase(), size / 2, size / 2);
  }
};

const AppProvider = ({ children }) => {
  const memoryContext = useMemory();
  const userProfile = memoryContext?.userProfile;
  const [activeContextSummary, setActiveContextSummary] = useState(null);
  const roleplayEnabledRef = useRef(false);
  const userCharacterRef = useRef(null);
  const summaryContextForRequest = useMemo(() => {
    const summary = (activeContextSummary || userProfile?.activeContextSummary || '').trim();
    return summary || null;
  }, [activeContextSummary, userProfile?.activeContextSummary]);
  const buildSystemPrompt = useCallback(
    (char) => _buildSystemPrompt(
      char,
      userProfile,
      summaryContextForRequest ? null : activeContextSummary,
      roleplayEnabledRef.current ? userCharacterRef.current : null
    ),
    [userProfile, activeContextSummary, summaryContextForRequest]
  );
  const getRelevantMemories = memoryContext?.getRelevantMemories;
  const addConversationSummary = memoryContext?.addConversationSummary;
  const [sdStatus, setSdStatus] = useState({ automatic1111: false });
  const [generatedImages, setGeneratedImages] = useState([]);
  const [isImageGenerating, setIsImageGenerating] = useState(false);
  const [apiError, setApiError] = useState(null);
  const clearError = useCallback(() => setApiError(null), []);
  const [activeTab, setActiveTab] = useState('chat'); // Default to 'chat' tab
  const [sttEnabled, setSttEnabled] = useState(false); // Default to false
  const [ttsEnabled, setTtsEnabled] = useState(false); // Default to false
  const [userAvatar, setUserAvatar] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(null); // Store message ID being played, or null
  const [audioError, setAudioError] = useState(null);
  const [primaryModel, setPrimaryModel] = useState(null);
  const [secondaryModel, setSecondaryModel] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  const [isChunkPlaying, setIsChunkPlaying] = useState(false);
  const [speechDetected, setSpeechDetected] = useState(false);
  const [audioAnalyzer, setAudioAnalyzer] = useState(null);
  const [isCallModeActive, setIsCallModeActive] = useState(false);
  const [callModeRecording, setCallModeRecording] = useState(false);

  // At the top of AppProvider, alongside primaryModel / secondaryModel:
  const [primaryCharacter, setPrimaryCharacter] = useState(null);
  const [secondaryCharacter, setSecondaryCharacter] = useState(null);
  const [primaryAvatar, setPrimaryAvatar] = useState(null);
  const [secondaryAvatar, setSecondaryAvatar] = useState(null);
  const mediaRecorderRef = useRef(null);
  const isFirstTextChunk = useRef(true);
  const isTtsInterruptedRef = useRef(false);
  const callModeMediaRecorderRef = useRef(null);
  const callModeAudioChunksRef = useRef([]);
  const activeAudioPlayersRef = useRef(new Set()); // Track ALL active audio sources
  const callModeSilenceTimerRef = useRef(null);
  const callModeStreamRef = useRef(null);
  const audioChunksRef = useRef([]); // Updated to match previous edit
  const audioPlayerRef = useRef(null); // Ref to store the current Audio object for TTS
  const audioPlaybackRef = useRef({ context: null }); // Ref for persistent AudioContext (mobile unlock)
  const [inputTranscript, setInputTranscript] = useState(''); // State for input transcript
  const [agentMemories, setAgentMemories] = useState([]);
  const lastPlayedMessageRef = useRef(null);
  const [sttEnginesAvailable, setSttEnginesAvailable] = useState(['whisper', 'parakeet']);
  const [manuallyStoppedAudio, setManuallyStoppedAudio] = useState(false);
  const [applyAvatar, setApplyAvatar] = useState(false);
  const [activeAvatar, setActiveAvatar] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false); // Loading state for transcription
  const [documents, setDocuments] = useState({ file_list: [] });
  const [isGenerating, setIsGenerating] = useState(false);
  const textGenerationStartTime = useRef(null);
  const speechStartTime = useRef(null);
  const promptSubmissionStartTime = useRef(null); // ADD THIS LINE
  const [dualModeEnabled, setDualModeEnabled] = useState(false);
  const [agentConversationActive, setAgentConversationActive] = useState(false);
  const [autoMemoryEnabled, setAutoMemoryEnabled] = useState(true); // Default to enabled
  const [autoDeleteChats, setAutoDeleteChats] = useState(false); // Default to false
  // avatar sizing for chat
  const [userAvatarSize] = useState(64);
  const streamingTtsMessageIdRef = useRef(null);
  const [characterAvatarSize] = useState(64);
  const [showAvatars, setShowAvatars] = useState(true);
  const [showAvatarsInChat, setShowAvatarsInChat] = useState(true);
  const [abortController, setAbortController] = useState(null);
  const [isStreamingStopped, setIsStreamingStopped] = useState(false);
  const [audioQueue, setAudioQueue] = useState([]);
  const [isAutoplaying, setIsAutoplaying] = useState(false);
  const [ttsSubtitleCue, setTtsSubtitleCue] = useState(null);
  const ttsSubtitleCueRef = useRef(null); // ✅ Ref for direct access, bypassing re-render issues

  // Debug: Monitor state changes
  useEffect(() => {
    ttsSubtitleCueRef.current = ttsSubtitleCue; // ✅ Keep ref in sync
  }, [ttsSubtitleCue]);

  const audioContextRef = useRef(null); // To manage the Web Audio API context
  const [primaryIsAPI, setPrimaryIsAPI] = useState(false);
  const [secondaryIsAPI, setSecondaryIsAPI] = useState(false);
  const setUserAvatarSize = (size) => {
    updateSettings({ userAvatarSize: size });
  };
  const setCharacterAvatarSize = (size) => {
    updateSettings({ characterAvatarSize: size });
  };





  // Character management states
  const [characters, setCharacters] = useState([]);
  const [activeCharacter, setActiveCharacter] = useState(null);
  const [userCharacterId, setUserCharacterId] = useState(null);
  const userCharacter = useMemo(() => {
    if (!characters || characters.length === 0) return null;
    return userCharacterId ? characters.find(c => c.id === userCharacterId) || null : null;
  }, [characters, userCharacterId]);
  useEffect(() => {
    userCharacterRef.current = userCharacter;
  }, [userCharacter]);
  const activeCharacterRef = useRef(null);
  useEffect(() => {
    activeCharacterRef.current = activeCharacter;
  }, [activeCharacter]);
  // Fix: Keep primaryCharacter and secondaryCharacter in sync with characters list updates
  useEffect(() => {
    if (primaryCharacter) {
      const updatedPrimary = characters.find(c => c.id === primaryCharacter.id);
      if (updatedPrimary && updatedPrimary !== primaryCharacter) {
        console.log('🔄 [AppContext] Syncing primaryCharacter with updated data');
        setPrimaryCharacter(updatedPrimary);
      }
    }
    if (secondaryCharacter) {
      const updatedSecondary = characters.find(c => c.id === secondaryCharacter.id);
      if (updatedSecondary && updatedSecondary !== secondaryCharacter) {
        console.log('🔄 [AppContext] Syncing secondaryCharacter with updated data');
        setSecondaryCharacter(updatedSecondary);
      }
    }
  }, [characters, primaryCharacter, secondaryCharacter]);

  const [backgroundImage, setBackgroundImage] = useState(null); // New state for chat background

  // Refs for avatar canvases
  const primaryAvatarRef = useRef(null);
  const secondaryAvatarRef = useRef(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [activeCharacterIds, setActiveCharacterIds] = useState([]);
  const [activeCharacterWeights, setActiveCharacterWeights] = useState({});
  const [multiRoleContext, setMultiRoleContext] = useState('');

  const [messages, setMessages] = useState([]);
  const [taskProgress, setTaskProgress] = useState({ progress: 0, status: '', active: false });
  const deleteConversation = useCallback((id) => {
    try {
      // Update state first
      const updatedConversations = conversations.filter(conv => conv.id !== id);
      setConversations(updatedConversations);

      // Handle active conversation updates
      if (activeConversation === id) {
        if (updatedConversations.length > 0) {
          setActiveConversation(updatedConversations[0].id);
        } else {
          setActiveConversation(null);
        }
      }

      // Try-catch for localStorage to handle quota issues
      try {
        // Only update localStorage if we have space
        // Save just the minimal data needed rather than everything
        const minimalData = updatedConversations.map(conv => ({
          id: conv.id,
          name: conv.name,
          characterIds: conv.characterIds,
          created: conv.created,
          // Skip storing full message history to reduce size
          messageCount: Array.isArray(conv.messages) ? conv.messages.length : 0
        }));

        localStorage.setItem('Eloquent-conversations-index', JSON.stringify(minimalData));

        // For the currently active conversation, save messages separately
        if (activeConversation) {
          const activeConv = updatedConversations.find(c => c.id === activeConversation);
          if (activeConv && activeConv.messages) {
            try {
              localStorage.setItem(`Eloquent-conversation-${activeConversation}`,
                JSON.stringify(activeConv.messages));
            } catch (storageErr) {
              console.warn("Could not save active conversation messages:", storageErr);
            }
          }
        }
      } catch (storageError) {
        console.error("Storage error during delete:", storageError);
        // Continue with the deletion in state even if storage fails
      }

      return true;
    } catch (e) {
      console.error("Error in deleteConversation:", e);
      return false;
    }
  }, [conversations, activeConversation, setActiveConversation]);

  const deleteAllConversations = useCallback(() => {
    try {
      // Clear all conversations
      setConversations([]);
      setActiveConversation(null);
      setMessages([]);

      // Clear from localStorage
      try {
        localStorage.removeItem('Eloquent-conversations');
        localStorage.removeItem('Eloquent-conversations-index');
        localStorage.removeItem('Eloquent-active-conversation');

        // Also clear any individual conversation message storage
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key && key.startsWith('Eloquent-conversation-')) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach(key => localStorage.removeItem(key));
      } catch (storageError) {
        console.error("Storage error during delete all:", storageError);
      }

      return true;
    } catch (e) {
      console.error("Error in deleteAllConversations:", e);
      return false;
    }
  }, [setConversations, setActiveConversation, setMessages]);

  const renameConversation = (id, newName) => {
    setConversations(prevConversations =>
      prevConversations.map(conv =>
        conv.id === id ? { ...conv, name: newName } : conv
      )
    );

    const updatedConversations = conversations.map(conv =>
      conv.id === id ? { ...conv, name: newName } : conv
    );

    localStorage.setItem('conversations', JSON.stringify(updatedConversations));
  };

  const [settings, setSettings] = useState({
    directProfileInjection: false, // <-- ADD THIS
    temperature: 0.7,
    max_tokens: -1,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
    use_rag: false,
    selectedDocuments: [],
    contextLength: 16000, // Default value
    useMemory: false,
    useMemoryAgent: false,
    useLore: false,
    useLoreAgent: false,
    useLoreAgentForMemory: false,
    useLoreAgentForMemoryRetrieval: false,
    useLoreAgentForMemoryObservation: false,
    multiRoleMode: false,
    autoSelectSpeaker: false,
    narratorEnabled: false,
    narratorInterval: 6,
    narratorName: 'Narrator',
    narratorInstructions: 'Describe scene transitions and world details briefly. Keep narration concise and avoid speaking for the user.',
    narratorAvatar: null,
    sttEnabled: false,
    streamResponses: true,
    sttEngine: "whisper",
    ttsVoice: 'af_heart',
    ttsSpeed: 1.0,
    ttsPitch: 0,
    ttsAutoPlay: false,  // Simply set a default value
    userAvatarSize: null, // Default size for user avatar
    characterAvatarSize: null, // Default size for character avatar
    mdBodyColor: '',
    mdBoldColor: '',
    mdItalicColor: '',
    mdQuoteColor: '',
    mdQuoteBorder: '',
    mdH1Color: '',
    mdH2Color: '',
    mdH3Color: '',
    mdH1Font: '',
    mdH2Font: '',
    mdH3Font: '',
    performanceMode: false,
    admin_password: "", // <-- Password for remote access
  });
  useEffect(() => {
    roleplayEnabledRef.current = settings.multiRoleMode === true;
  }, [settings.multiRoleMode]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;

    const applyVar = (key, cssVar) => {
      const value = settings?.[key];
      if (typeof value === 'string' && value.trim()) {
        root.style.setProperty(cssVar, value.trim());
      } else {
        root.style.removeProperty(cssVar);
      }
    };

    applyVar('mdBodyColor', '--md-body-color');
    applyVar('mdBoldColor', '--md-bold-color');
    applyVar('mdItalicColor', '--md-italic-color');
    applyVar('mdQuoteColor', '--md-quote-color');
    applyVar('mdQuoteBorder', '--md-quote-border');
    applyVar('mdH1Color', '--md-h1-color');
    applyVar('mdH2Color', '--md-h2-color');
    applyVar('mdH3Color', '--md-h3-color');
    applyVar('mdH1Font', '--md-h1-font');
    applyVar('mdH2Font', '--md-h2-font');
    applyVar('mdH3Font', '--md-h3-font');
  }, [
    settings?.mdBodyColor,
    settings?.mdBoldColor,
    settings?.mdItalicColor,
    settings?.mdQuoteColor,
    settings?.mdQuoteBorder,
    settings?.mdH1Color,
    settings?.mdH2Color,
    settings?.mdH3Color,
    settings?.mdH1Font,
    settings?.mdH2Font,
    settings?.mdH3Font
  ]);

  // Backend detects single_gpu_mode, frontend stores as singleGpuMode
  const isSingleGpuMode = settings?.singleGpuMode === true;

  // Port configuration - loaded from /ports.json (written by launch.py)
  const [portConfig, setPortConfig] = useState({
    backend: "http://localhost:8000",
    secondary: "http://localhost:8001",
    tts: "http://localhost:8002"
  });

  // Load port configuration on startup (also updates the api.js module cache)
  useEffect(() => {
    import('../config/api').then(({ loadPortConfig }) => {
      loadPortConfig().then(config => {
        console.log('📌 Loaded port config:', config);
        setPortConfig(config);
      });
    });
  }, []);

  // API URLs - use port config
  const PRIMARY_API_URL = portConfig.backend;
  const SECONDARY_API_URL = isSingleGpuMode ? portConfig.backend : portConfig.secondary;
  const MEMORY_API_URL = isSingleGpuMode ? portConfig.backend : portConfig.secondary;
  const TTS_API_URL = portConfig.tts;
  const BACKEND = import.meta.env.VITE_API_URL || (isSingleGpuMode ? portConfig.backend : portConfig.secondary);
  const VITE_API_URL = isSingleGpuMode ? portConfig.backend : portConfig.secondary;




  // Keep TTS WebSocket always connected when TTS autoplay is enabled
  useEffect(() => {
    if (settings.ttsAutoPlay) {
      console.log("🔌 [TTS] Keeping WebSocket connected for instant TTS (Internal Auto-Reconnect enabled)");
      ttsClient.connect(
        () => console.log("✅ [TTS] WebSocket connected and ready"),
        () => console.log("🔌 [TTS] WebSocket disconnected, client will auto-reconnect..."),
        (err) => console.error("❌ [TTS] WebSocket error:", err)
      );
    } else {
      // Disconnect if TTS autoplay is off
      ttsClient.disconnect();
    }
  }, [settings.ttsAutoPlay]);

  useEffect(() => {
    if (!ttsClient) return;
    const handleSubtitleCue = (cue) => {
      setTtsSubtitleCue({
        text: cue?.text || '',
        durationMs: cue?.durationMs ?? null
      });
      window.__ttsSubtitleCue = {
        text: cue?.text || '',
        durationMs: cue?.durationMs ?? null,
        timestamp: Date.now()
      };
    };
    ttsClient.onSubtitleCue = handleSubtitleCue;
    return () => {
      if (ttsClient.onSubtitleCue === handleSubtitleCue) {
        ttsClient.onSubtitleCue = null;
      }
    };
  }, [ttsClient, setTtsSubtitleCue]);

  useEffect(() => {
    const loadSavedConversations = () => {
      try {
        const savedConversations = localStorage.getItem('Eloquent-conversations');
        if (savedConversations) {
          const parsedConversations = JSON.parse(savedConversations);
          setConversations(parsedConversations);

          const lastActiveId = localStorage.getItem('Eloquent-active-conversation');
          if (lastActiveId && parsedConversations.some(c => c.id === lastActiveId)) {
            setActiveConversation(lastActiveId);

            const activeConv = parsedConversations.find(c => c.id === lastActiveId);
            if (activeConv && activeConv.messages) {
              setMessages(activeConv.messages);
            }
          }
        }
      } catch (error) {
        console.error("Error loading saved conversations:", error);
      }
    };

    loadSavedConversations();
  }, []);
  // ===== Debugging and Testing the Lore Functionality =====
  // --- TTS Implementation (REORDERED) ---
  const stopTTS = useCallback(() => {
    // Note: Don't clear ttsSubtitleCue here - let it expire naturally via the timeout in CallModeOverlay
    console.log('🛑 [stopTTS] Initiating stop sequence...');
    // console.trace('🛑 [stopTTS] Caller Trace');
    // 1. IMMEDIATE: Set interrupt flag to prevent any new chunks from processing
    isTtsInterruptedRef.current = true;
    isFirstTextChunk.current = true; // Reset text chunk flag
    streamingTtsMessageIdRef.current = null; // Clear active message ID

    // 2. CRITICAL: Send interrupt signal to backend FIRST (before any cleanup)
    if (ttsClient && ttsClient.socket && ttsClient.socket.readyState === 1) {
      console.log('🛑 [stopTTS] Sending interrupt signal to backend');
      ttsClient.interrupt();
      ttsClient.clearPending();
    }

    // 3. Always cancel native browser TTS
    if ('speechSynthesis' in window) {
      console.log('🛑 [stopTTS] Cancelling native speech synthesis');
      window.speechSynthesis.cancel();
    }

    // 4. STOP ALL REGISTERED AUDIO PLAYERS
    const playersToStop = Array.from(activeAudioPlayersRef.current);
    if (playersToStop.length > 0) {
      console.log(`🛑 [stopTTS] Stopping ${playersToStop.length} active audio sources.`);
      playersToStop.forEach(player => {
        try {
          if (player instanceof Audio) {
            console.log('🛑 [stopTTS] Pausing Audio object:', player.src);
            player.pause();
            player.currentTime = 0;
            player.src = '';
            // Don't revoke URL immediately to avoid errors if play() is pending
          } else if (player.ctx) {
            try { player.source?.stop(); } catch (e) { }
            try { player.ctx.close(); } catch (e) { }
          }
        } catch (err) {
          console.error("🛑 [stopTTS] Error stopping player:", err);
        }
      });
      activeAudioPlayersRef.current.clear();
    }

    // Legacy cleanup / Fallback
    if (audioPlayerRef.current) {
      console.log('🛑 [stopTTS] stopping legacy audioPlayerRef');
      try {
        audioPlayerRef.current.pause();
        audioPlayerRef.current.currentTime = 0;
      } catch (e) { }
      audioPlayerRef.current = null;
    }

    // 5. Clear local queues and state
    console.log('🛑 [stopTTS] Clearing local audio queues and state');
    if (ttsClient && ttsClient.audioQueue) {
      console.log(`🛑 [stopTTS] dumped ${ttsClient.audioQueue.length} items from audioQueue`);
      ttsClient.audioQueue.length = 0;
    }

    setAudioQueue([]);
    setIsAutoplaying(false);
    setIsPlayingAudio(null);
    window.streamingAudioPlaying = false;

    // Explicitly update state to trigger UI changes
    setManuallyStoppedAudio(true);
    setTimeout(() => setManuallyStoppedAudio(false), 500);

  }, [ttsClient, setAudioQueue, setIsAutoplaying]);

  const handleStopGeneration = useCallback(() => {
    // 1. Stop the text generation stream
    if (abortController) {
      console.log('🛑 Aborting text generation...');
      abortController.abort();
      setAbortController(null);
    }

    // 2. Stop TTS explicitly
    stopTTS();

    // Reset the main "isGenerating" flag for the UI
    setIsGenerating(false);

    // This flag is for UI feedback
    setIsStreamingStopped(true);
    setTimeout(() => setIsStreamingStopped(false), 1000);

  }, [abortController, stopTTS, setIsGenerating]);
  // NEW: Silent audio unlocker for mobile browsers
  const unlockAudioContext = useCallback(() => {
    // 1. Create context if missing
    if (!audioPlaybackRef.current.context) {
      audioPlaybackRef.current.context = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioPlaybackRef.current.context;

    // 2. Resume if suspended (common on mobile)
    if (ctx.state === 'suspended') {
      ctx.resume().then(() => console.log('🔊 [Audio] Context resumed via silent unlock'));
    }

    // 3. Play silent buffer to force "active" state
    try {
      const buffer = ctx.createBuffer(1, 1, 22050);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0);
      console.log('🔊 [Audio] Silent unlock played');
    } catch (e) {
      console.warn('🔊 [Audio] Unlock failed', e);
    }
  }, []);

  const getTtsOverridesForCharacter = useCallback((character) => {
    if (!character) return null;
    const engine = settings.ttsEngine || 'kokoro';
    if (engine !== 'chatterbox' && engine !== 'chatterbox_turbo' && engine !== 'kokoro') return null;
    const voice = character.ttsVoice || character.tts_voice || null;
    if (!voice || voice === 'default') return null;
    return { ttsVoice: voice };
  }, [settings.ttsEngine]);

  const getTtsOverridesForCharacterId = useCallback((characterId) => {
    if (!characterId) return null;
    const character = characters.find((item) => item.id === characterId);
    return getTtsOverridesForCharacter(character);
  }, [characters, getTtsOverridesForCharacter]);

  const startStreamingTTS = useCallback((messageId, optionsOverrides = null) => {
    window.streamingAudioPlaying = false;
    ttsClient.audioQueue = [];

    if (!settings.ttsAutoPlay || !settings.ttsEnabled) return;

    console.log(`▶️ [startStreamingTTS] Starting stream for ${messageId} (Resetting interrupt flag)`);
    isFirstTextChunk.current = true;
    isTtsInterruptedRef.current = false;
    streamingTtsMessageIdRef.current = messageId;
    setIsAutoplaying(true);

    // Safety: If socket is closed/closing, prompt a reconnect so pending settings send
    // Safety: If socket is closed/closing (or null), prompt a reconnect so pending settings send
    if (!ttsClient.socket || ttsClient.socket.readyState > 1) {
      console.warn("⚠️ [TTS] Socket closed/closing/null, triggering reconnect...");
      ttsClient.connect();
    }

    // Local flag to track if this is the first audio chunk for this specific message
    let firstAudioPlayed = false;

    const ttsEngine = optionsOverrides?.ttsEngine || settings.ttsEngine || 'kokoro';
    const ttsVoice = optionsOverrides?.ttsVoice || settings.ttsVoice || 'af_heart';

    const streamingTtsSettings = {
      engine: ttsEngine,
      voice: ttsVoice,
      exaggeration: settings.ttsExaggeration || 0.5,
      cfg: settings.ttsCfg || 0.5,
      audio_prompt_path: ((ttsEngine === 'chatterbox') || (ttsEngine === 'chatterbox_turbo')) && ttsVoice !== 'default'
        ? ttsVoice
        : null
    };

    ttsClient.send(streamingTtsSettings);

    ttsClient.onAudioQueueUpdate = async () => {
      // 1. Check if we've been interrupted
      if (isTtsInterruptedRef.current) {
        console.log("🛡️ [TTS] Ignoring audio update due to interrupt flag. Dumping " + ttsClient.audioQueue.length + " items.");
        ttsClient.audioQueue.length = 0;
        return;
      }

      // 2. Check if we are still the active message (Fix for zombie loops)
      // Allow if ref is null (generation finished) but block if it's a NEW message ID
      if (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId) {
        console.log(`🛡️ [TTS] Ignoring update for stale message ID ${messageId} (Current: ${streamingTtsMessageIdRef.current})`);
        return;
      }

      setAudioQueue([...ttsClient.audioQueue]);

      if (ttsClient.audioQueue.length > 0 && !window.streamingAudioPlaying) {
        console.log("🎵 [TTS] Starting playback loop. Queue length:", ttsClient.audioQueue.length);
        window.streamingAudioPlaying = true;
        setIsPlayingAudio(messageId);

        // --- 🟢 RESTORED TIMING LOGS HERE ---
        if (!firstAudioPlayed) {
          const now = performance.now();
          firstAudioPlayed = true;

          // Log 1: Time from First Text Chunk -> Audio Start
          if (textGenerationStartTime.current) {
            const processingLatency = now - textGenerationStartTime.current;
            console.log(`⏱️ [TTS Timing] 🟡 Latency (Text Chunk -> Audio): ${processingLatency.toFixed(2)}ms`);
          }

          // Log 2: Time from User Prompt -> Audio Start
          if (promptSubmissionStartTime.current) {
            const totalLatency = now - promptSubmissionStartTime.current;
            console.log(`⏱️ [TTS Timing] 🟢 Time to First Audio (from Prompt): ${totalLatency.toFixed(2)}ms`);
          }
        }
        // ------------------------------------

        try {
          while (ttsClient.audioQueue.length > 0) {
            // Check flags again inside loop
            if (isTtsInterruptedRef.current) {
              console.log("🛑 [TTS] Loop broken by interrupt flag (Pre-Shift).");
              ttsClient.audioQueue.length = 0;
              break;
            }
            // Allow if null, break if DIFFERENT
            if (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId) {
              console.log("🛑 [TTS] Loop broken by stale message ID (Pre-Shift).");
              break;
            }

            const chunk = ttsClient.audioQueue.shift();
            // Double check queue empty/undefined
            if (!chunk) break;

            // Extract audio and subtitle (support both old format and new paired format)
            const arrayBuffer = chunk.audio || chunk;
            const subtitle = chunk.subtitle;

            const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audioPlayerRef.current = audio;
            activeAudioPlayersRef.current.add(audio); // Register for mass cleanup

            audio.playbackRate = settings.ttsSpeed || 1.0;

            // ✅ Display subtitle RIGHT when audio plays (not when synthesized)
            if (subtitle) {
              window.__ttsSubtitleCue = {
                text: subtitle.text,
                durationMs: subtitle.durationMs,
                timestamp: Date.now()
              };
            }

            console.log(`🎵 [TTS] Playing chunk. Queue remaining: ${ttsClient.audioQueue.length}`);

            await new Promise((resolve, reject) => {
              const handleEnd = () => {
                cleanup();
                resolve();
              };
              const handleError = (e) => {
                console.error("🎵 [TTS] Audio error", e);
                cleanup();
                reject(e);
              };
              const handleStop = (e) => {
                // Common behavior: browsers might fire 'pause' or 'emptied' efficiently.
                // We log it for debugging but differentiate manual stops vs events.
                if (isTtsInterruptedRef.current) {
                  console.log(`⏹️ [TTS] Audio stopped via interrupt (${e.type})`);
                } else {
                  // console.log(`ℹ️ [TTS] Audio event '${e.type}' triggered move to next chunk.`);
                }
                cleanup();
                resolve(); // Resolve so we can loop and see interrupt flag
              };

              const cleanup = () => {
                // Remove from registry
                activeAudioPlayersRef.current.delete(audio);
                URL.revokeObjectURL(audioUrl);
                if (audioPlayerRef.current === audio) audioPlayerRef.current = null;

                audio.removeEventListener('ended', handleEnd);
                audio.removeEventListener('error', handleError);
                audio.removeEventListener('pause', handleStop);
                audio.removeEventListener('abort', handleStop);
                audio.removeEventListener('emptied', handleStop);
              };

              audio.addEventListener('ended', handleEnd);
              audio.addEventListener('error', handleError);
              // Crucial: resolving on these events breaks the deadlock when stopTTS is called
              audio.addEventListener('pause', handleStop);
              audio.addEventListener('abort', handleStop);
              audio.addEventListener('emptied', handleStop);

              // Only play if not interrupted and still active (or finished generating)
              const isStale = streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId;

              if (activeAudioPlayersRef.current.has(audio) && !isTtsInterruptedRef.current && !isStale) {
                audio.play().catch(e => {
                  if (e.name !== 'AbortError') console.error("🎵 [TTS] Play failed:", e);
                  handleError(e);
                });
              } else {
                console.log("🛑 [TTS] Play skipped due to interrupt flag or stale ID.");
                resolve(); // Skip playback
              }
            });

            // Post-playback interrupt check
            // Allow if null, break if DIFFERENT
            if (isTtsInterruptedRef.current || (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId)) {
              console.log("🛑 [TTS] Loop broken by interrupt flag or stale ID (Post-Play).");
              break;
            }
          }
        } catch (err) {
          console.error("🎵 [TTS Playback Error]", err);
        } finally {
          console.log("🎵 [TTS] Playback loop finished/exited.");
          // Only clear state if WE are still the active message (or it finished) OR interrupted
          // If a NEW message took over (streamingTtsMessageIdRef IS set and != messageId), don't clear global state
          const isStale = streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId;

          if (!isStale || isTtsInterruptedRef.current) {
            window.streamingAudioPlaying = false;
            setIsPlayingAudio(null);
          }
          if (isTtsInterruptedRef.current) {
            setAudioQueue([]); // Ensure cleared
          }
        }
      }
    };
  }, [settings, ttsClient, setAudioQueue, setIsAutoplaying]);

  const stopStreamingTTS = useCallback(() => {
    console.log("🛑 [stopStreamingTTS] Called, delegating to stopTTS");
    stopTTS();
  }, [stopTTS]);
  const playNextChunk = async () => {
    if (ttsClient.audioQueue.length === 0 || isChunkPlaying) return;

    setIsChunkPlaying(true);
    const arrayBuffer = ttsClient.audioQueue.shift();

    try {
      const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      console.log(`🎵 [Sequential] Playing chunk, ${ttsClient.audioQueue.length} remaining`);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsChunkPlaying(false);
        // Play next chunk when this one ends
        if (ttsClient.audioQueue.length > 0) {
          setTimeout(playNextChunk, 50); // Small delay to avoid race conditions
        }
      };

      await audio.play();
    } catch (error) {
      console.error(`🎵 [Sequential] Error:`, error);
      setIsChunkPlaying(false);
    }
  };

  const addStreamingText = useCallback((newTextChunk) => {
    if (isFirstTextChunk.current) {
      textGenerationStartTime.current = performance.now();
      console.log(`⏱️ [TTS Timing] Timer started on first received text chunk.`);
      isFirstTextChunk.current = false; // So this only runs once per message
    }

    // ONLY send to TTS if we have an active message ID (haven't been stopped)
    // FIX: Removed 'isAutoplaying' check to avoid stale state closure issues.
    // streamingTtsMessageIdRef is the reliable, synchronous source of truth.
    if (streamingTtsMessageIdRef.current && newTextChunk) {
      ttsClient.send(newTextChunk);
    }
  }, [ttsClient]); // Removed isAutoplaying dependency

  const endStreamingTTS = useCallback(() => {
    // Note: Don't clear ttsSubtitleCue here - let it expire naturally
    if (streamingTtsMessageIdRef.current) {
      console.log(`⏹️ [TTS Stream] Ending for message ${streamingTtsMessageIdRef.current}`);
      ttsClient.closeStream();
      streamingTtsMessageIdRef.current = null;
    }
  }, [ttsClient]);


  // …later, inside your component:
  const checkSdStatus = useCallback(async () => {

    const imageEngine = settings.imageEngine || 'auto1111';

    let automatic1111Status = false;
    let localSdStatus = false;
    let comfyuiStatus = false;
    let comfyuiData = null;

    // Check AUTOMATIC1111 if needed
    if (imageEngine === 'auto1111' || imageEngine === 'both') {
      try {
        const res = await fetch(`${MEMORY_API_URL}/sd/status`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          automatic1111Status = Boolean(data.automatic1111);
        }
      } catch (err) {
      }
    }

    // Check local SD if needed  
    if (imageEngine === 'EloDiffusion' || imageEngine === 'both') {
      try {
        const res = await fetch(`${MEMORY_API_URL}/sd-local/status`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          localSdStatus = Boolean(data.available);
        }
      } catch (err) {
      }
    }

    // Check ComfyUI if needed
    if (imageEngine === 'comfyui' || imageEngine === 'both') {
      try {
        const res = await fetch(`${PRIMARY_API_URL}/sd-comfy/status`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          comfyuiStatus = Boolean(data.comfyui);
          comfyuiData = data; // Contains checkpoints, samplers, schedulers, etc.
        }
      } catch (err) {
      }
    }

    setSdStatus({
      automatic1111: automatic1111Status,
      localSd: localSdStatus,
      comfyui: comfyuiStatus,
      comfyuiData: comfyuiData,
      models: [] // We'll populate this later if needed
    });
  }, [PRIMARY_API_URL, MEMORY_API_URL, settings.imageEngine]);

  const fetchAvailableSTTEngines = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/available-engines`);
      if (response.ok) {
        const data = await response.json();
        setSttEnginesAvailable(data.available_engines || ['whisper']);
      }
    } catch (error) {
      console.error("Error fetching available STT engines:", error);
      setSttEnginesAvailable(['whisper']); // Default to Whisper if fetch fails
    }
  }, [PRIMARY_API_URL]);

  // 2. Generate image via POST - supports Local SD, A1111, and ComfyUI
  const generateImage = useCallback(async (prompt, opts, gpuId = 0) => {
    console.log(`Starting image generation for GPU ${gpuId} with prompt:`, prompt);
    console.log("Image generation options:", opts);

    // Use settings from the context
    const imageEngine = settings.imageEngine || 'auto1111';

    // Determine endpoint based on engine
    let endpoint;
    if (imageEngine === 'EloDiffusion') {
      const targetApiUrl = gpuId === 1 ? MEMORY_API_URL : PRIMARY_API_URL;
      endpoint = `${targetApiUrl}/sd-local/txt2img`;
    } else if (imageEngine === 'comfyui') {
      endpoint = `${PRIMARY_API_URL}/sd-comfy/txt2img`;
    } else if (imageEngine === 'nanogpt') {
      endpoint = `${PRIMARY_API_URL}/sd/nanogpt`;
    } else {
      endpoint = `${PRIMARY_API_URL}/sd/txt2img`;
    }

    // Sampler mapping for different backends
    const mapSamplerForBackend = (samplerNameFromFrontend) => {
      if (imageEngine === 'EloDiffusion') {
        return samplerNameFromFrontend;
      }
      if (imageEngine === 'comfyui') {
        // ComfyUI uses lowercase sampler names
        const comfySamplerMapping = {
          'euler_a': 'euler_ancestral',
          'Euler a': 'euler_ancestral',
          'euler': 'euler',
          'Euler': 'euler',
          'dpmpp2m': 'dpmpp_2m',
          'DPM++ 2M Karras': 'dpmpp_2m',
          'dpmpp2s_a': 'dpmpp_2s_ancestral',
          'DPM++ 2S a Karras': 'dpmpp_2s_ancestral',
          'heun': 'heun',
          'Heun': 'heun',
          'dpm2': 'dpm_2',
          'DPM2': 'dpm_2',
          'ddim': 'ddim',
          'DDIM': 'ddim',
        };
        return comfySamplerMapping[samplerNameFromFrontend] || samplerNameFromFrontend.toLowerCase();
      }
      // A1111 mapping
      const automatic1111SamplerMapping = {
        'euler_a': 'Euler a',
        'euler': 'Euler',
        'dpmpp2m': 'DPM++ 2M Karras',
        'dpmpp2s_a': 'DPM++ 2S a Karras',
        'heun': 'Heun',
        'dpm2': 'DPM2',
        'ddim_trailing': 'DDIM',
      };
      return automatic1111SamplerMapping[samplerNameFromFrontend] || samplerNameFromFrontend;
    };

    setIsImageGenerating(true);
    clearError();
    try {
      let payload;

      if (imageEngine === 'comfyui') {
        // ComfyUI payload format
        payload = {
          prompt,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          cfg_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          scheduler: opts.scheduler || "normal",
          seed: opts.seed || -1,
          checkpoint: opts.checkpoint || opts.model || "",
          batch_size: opts.batch_size || 1,
          denoise: opts.denoise || 1.0,
          timeout: 300
        };
      } else if (imageEngine === 'EloDiffusion') {
        // Local SD payload
        payload = {
          prompt,
          gpu_id: gpuId,
          task_id: opts.task_id,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          guidance_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          seed: opts.seed || -1,
        };
      } else if (imageEngine === 'nanogpt') {
        // NanoGPT payload
        payload = {
          prompt,
          width: opts.width || 1024,
          height: opts.height || 1024,
          model: settings.nanoGptModel || 'dall-e-3',
          api_key: settings.nanoGptApiKey,
          // NanoGPT / OpenAI DALL-E 3 doesn't support negative_prompt, steps, cfg, sampler often
          // But we can pass them if the backend filters them or if using a different model
          n: 1,
          size: `${opts.width || 1024}x${opts.height || 1024}`,
          quality: "standard",
          response_format: "url"
        };
      } else {
        // A1111 payload
        payload = {
          prompt,
          gpu_id: gpuId,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          guidance_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          seed: opts.seed || -1,
          model: opts.model,
          cfg_scale: opts.guidance_scale || 7.0,
          sampler_name: mapSamplerForBackend(opts.sampler || "Euler a")
        };
      }

      console.log("Sending to SD API:", payload);
      console.log("Using image engine:", imageEngine);
      console.log("API URL:", endpoint);

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorText = await res.text();
        console.error("SD API error:", res.status, errorText);
        throw new Error(`SD API error: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      console.log("Received response from SD API:", data);
      return data;

    } catch (err) {
      console.error("Generate image error:", err);
      setApiError(err.message);
      throw err;
    } finally {
      setIsImageGenerating(false);
    }
  }, [clearError, setApiError, PRIMARY_API_URL, MEMORY_API_URL, settings, setIsImageGenerating]);

  // NEW: Generate Video via NanoGPT
  const generateVideo = useCallback(async (prompt) => {
    console.log("Starting video generation prompt:", prompt);
    const videoModel = settings.nanoGptVideoModel || 'svd';
    const apiKey = settings.nanoGptApiKey;

    if (!apiKey) {
      throw new Error("NanoGPT API Key is missing. Please check Settings > Image Generation.");
    }

    // 1. Start Job
    const startRes = await fetch(`${PRIMARY_API_URL}/sd/nanogpt/video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        model: videoModel,
        api_key: apiKey
      })
    });

    if (!startRes.ok) {
      const errorText = await startRes.text();
      throw new Error(`Failed to start video job: ${errorText}`);
    }

    const startData = await startRes.json();
    const jobId = startData.job_id;
    console.log("Video job started:", jobId);

    // 2. Poll for Status
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes (5s * 60)

    while (attempts < maxAttempts) {
      // Wait 5s
      await new Promise(r => setTimeout(r, 5000));
      attempts++;

      const statusRes = await fetch(`${PRIMARY_API_URL}/sd/nanogpt/video/status/${jobId}?api_key=${apiKey}`);
      if (!statusRes.ok) {
        console.warn("Video status check failed, retrying...");
        continue;
      }

      const statusData = await statusRes.json();
      console.log("Video polling status:", statusData);

      if (statusData.status === 'success') {
        return statusData.video_url; // Local URL returned by backend
      } else if (statusData.status === 'failed') {
        throw new Error(`Video generation failed: ${statusData.error || 'Unknown error'}`);
      }
      // If pending/processing, continue loop
    }

    throw new Error("Video generation timed out.");
  }, [PRIMARY_API_URL, settings]);




  // 3. (Unchanged) Test lore detection helper
  const testLoreDetection = async (sampleText) => {
    try {
      if (!activeCharacter) {
        console.warn("🌍 [LORE TEST] No active character to test lore with");
        return;
      }

      const testText = sampleText || "Let me test the lore detection system";
      console.log(`🌍 [LORE TEST] Testing with text: "${testText}" and character: ${activeCharacter.name}`);

      const triggeredLore = await fetchTriggeredLore(testText, activeCharacter);
      console.log("🌍 [LORE TEST] Results:", triggeredLore);

      return triggeredLore;
    } catch (error) {
      console.error("🌍 [LORE TEST] Error:", error);
      return null;
    }
  };


  // You can expose this for console access if needed:
  window.testLoreDetection = testLoreDetection;
  // --- STT Implementation ---
  const startRecording = useCallback(async () => {
    console.log("🎤 Attempting to start recording...");
    setAudioError(null); // Clear previous errors
    if (isRecording || isTranscribing) {
      console.warn("🎤 Recording or transcription already in progress.");
      return;
    }

    try {
      // Check for browser support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser API not supported. If on mobile LAN, you may need HTTPS or specific flags.");
      }

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Determine supported MIME type
      let mimeType = 'audio/webm'; // Default
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
        mimeType = 'audio/ogg;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4'; // Fallback if Opus not supported
      }
      console.log(`🎤 Using MIME type: ${mimeType}`);

      // Create MediaRecorder instance
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = []; // Reset chunks

      // Event listener for data available
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Start recording
      mediaRecorderRef.current.start();
      setIsRecording(true); // Update state
      console.log("🎤 Recording started.");

    } catch (err) {
      console.error("🎤 Error accessing microphone:", err);

      let errorMessage = `Microphone error: ${err.message}`;

      // Detailed guidance for mobile LAN users
      if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        errorMessage = `⚠️ Microphone blocked due to insecure connection (HTTP).\n\nTo fix on Mobile/LAN:\n1. Open chrome://flags (or edge://flags)\n2. Search "Insecure origins treated as secure"\n3. Enable it and add: ${window.location.origin}\n4. Restart browser.`;
        alert(errorMessage);
      }

      setAudioError(errorMessage);
      setIsRecording(false); // Ensure state is reset
    }
  }, [isRecording, isTranscribing]); // Dependencies

  // Accepts an onTranscriptReceived callback function
  const stopRecording = useCallback(async (onTranscriptReceived) => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      console.log("🎤 Stopping recording via stopRecording()...");

      // Define the onstop handler *inside* stopRecording to capture onTranscriptReceived
      mediaRecorderRef.current.onstop = async () => {
        console.log("🎤 Recording stopped. Processing audio...");
        setIsRecording(false); // Update state immediately
        setIsTranscribing(true); // Set transcribing state

        // Combine chunks into a single Blob
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current.mimeType }); // Use the recorder's mimeType
        console.log(`🎤 Final audio blob size: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

        // Stop microphone tracks
        if (mediaRecorderRef.current?.stream) {
          mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }

        if (audioBlob.size === 0) {
          console.warn("🎤 Audio blob is empty, skipping transcription.");
          setIsTranscribing(false);
          setAudioError("Recording resulted in empty audio.");
          return;
        }

        try {
          // Log the engine we're using for debugging
          console.log(`🎤 Using STT engine: ${settings.sttEngine}`);

          // Call backend API to transcribe - use settings.sttEngine directly without fallback
          const transcript = await transcribeAudio(audioBlob, settings.sttEngine);
          console.log("🎤 Transcription successful:", transcript);

          // Call the callback with the transcript
          if (typeof onTranscriptReceived === 'function') {
            onTranscriptReceived(transcript); // Update the input field in Chat.jsx
          } else {
            console.warn("🎤 onTranscriptReceived callback is not a function.");
            alert(`Transcript (callback missing): ${transcript}`); // Fallback alert
          }

        } catch (transcriptionError) {
          console.error("🎤 Transcription failed:", transcriptionError);
          setAudioError(`Transcription failed: ${transcriptionError.message}`);
        } finally {
          setIsTranscribing(false); // Reset transcribing state
        }
      };

      // Trigger the onstop handler
      mediaRecorderRef.current.stop();

    } else {
      console.warn("🎤 Stop recording called but not currently recording.");
      setIsRecording(false);
      setIsTranscribing(false);
    }
  }, [settings, setIsRecording, setIsTranscribing, setAudioError]); // Add dependencies including settings

  const startCallMode = useCallback(async () => {
    console.log("🎯 Starting Call Mode...");
    setIsCallModeActive(true);
  }, []);

  const stopCallMode = useCallback(async () => {
    console.log("🎯 Stopping Call Mode...");
    setIsCallModeActive(false);

    // Stop any active recording
    if (isRecording) {
      await stopRecording(() => { });
    }

    // Stop any active TTS
    stopTTS();
  }, [isRecording, stopRecording, stopTTS]);


  // --- playTTS (Modified to prevent auto-replay) ---
  const playTTS = useCallback(async (messageId, text, optionsOverrides = null) => {
    console.log(`🗣️ [TTS] Attempting to play message ${messageId}: "${text.substring(0, 40)}..."`);

    // Reset interrupt flag for this new playback session
    isTtsInterruptedRef.current = false;

    // Check if this is the same message that was just played
    if (lastPlayedMessageRef.current === messageId && !optionsOverrides) {
      console.log(`🗣️ [TTS] Skipping message ${messageId} as it was just played.`);
      return;
    }
    // ADD THIS CODE: Check if the message is still streaming
    const message = messages.find(m => m.id === messageId);
    if (message && message.isStreaming && !optionsOverrides) {
      console.log(`🗣️ [TTS] Skipping message ${messageId} as it is still streaming.`);
      return;
    }

    // If there's already audio playing, stop it first
    if (isPlayingAudio) {
      console.warn(`🗣️ [TTS] Cannot play message ${messageId}: Audio for message ${isPlayingAudio} is already playing.`);
      stopTTS();
    }

    // Stop lingering audio
    if (audioPlayerRef.current) {
      console.warn("🗣️ [TTS] Found lingering audioPlayerRef, stopping before new playback.");
      stopTTS();
    }

    setIsPlayingAudio(messageId);
    setAudioError(null);

    let audioUrl = null;

    try {
      // CRITICAL: Check if TTS has been interrupted before starting playback
      if (isTtsInterruptedRef.current) {
        console.log(`🛑 [TTS] Playback blocked for message ${messageId} - interrupt flag is set`);
        setIsPlayingAudio(null);
        return;
      }

      // 1. Synthesize Speech -> Blob URL
      console.log(`🗣️ [TTS] Calling synthesizeSpeech API for message ${messageId}...`);

      // Resolve options: priority to overrides, then current state settings
      const currentTtsEngine = optionsOverrides?.ttsEngine || settings.ttsEngine || 'kokoro';
      const currentTtsVoice = optionsOverrides?.ttsVoice || settings.ttsVoice || 'af_heart';
      const currentTtsSpeed = optionsOverrides?.ttsSpeed ?? settings.ttsSpeed ?? 1.0;
      const currentTtsPitch = optionsOverrides?.ttsPitch ?? settings.ttsPitch ?? 0;

      // Build options object for TTS engines
      const ttsOptions = {
        voice: currentTtsVoice,
        engine: currentTtsEngine
      };

      // Add Chatterbox-specific parameters if using Chatterbox
      if (currentTtsEngine === 'chatterbox' || currentTtsEngine === 'chatterbox_turbo') {
        ttsOptions.exaggeration = optionsOverrides?.ttsExaggeration ?? settings.ttsExaggeration ?? 0.5;
        ttsOptions.cfg = optionsOverrides?.ttsCfg ?? settings.ttsCfg ?? 0.5;

        console.log(`🔊 [TTS] Chatterbox params - exaggeration: ${ttsOptions.exaggeration}, cfg: ${ttsOptions.cfg}`);
      }

      console.log(`🔊 [TTS] Using engine: ${ttsOptions.engine} with options:`, ttsOptions);

      // Pass options object to synthesizeSpeech
      audioUrl = await synthesizeSpeech(text, ttsOptions);

      // CHECK AGAIN after await - stop may have been pressed during synthesis
      if (isTtsInterruptedRef.current) {
        console.log(`🛑 [TTS] Playback blocked for message ${messageId} - interrupted during synthesis`);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setIsPlayingAudio(null);
        return;
      }

      // UNCHANGED: Everything after this stays exactly the same
      if (!audioUrl) throw new Error("SynthesizeSpeech returned an invalid URL.");
      console.log(`🗣️ [TTS] Received audio URL for message ${messageId}: ${audioUrl.substring(0, 50)}...`);

      // 2. Playback branch: HTML5 Audio if no pitch shift, otherwise Web Audio API
      if (currentTtsPitch === 0) {
        // --- HTML5 Audio path (only speed) ---
        const audio = new Audio(audioUrl);
        audio.playbackRate = currentTtsSpeed;
        audioPlayerRef.current = audio;
        activeAudioPlayersRef.current.add(audio); // Register for mass cleanup

        const handleEnd = () => {
          console.log(`🗣️ [TTS] Playback ended for message ${messageId}`);
          activeAudioPlayersRef.current.delete(audio); // Cleanup registry

          setIsPlayingAudio(null);
          URL.revokeObjectURL(audioUrl);
          audioPlayerRef.current = null;
          audio.removeEventListener('ended', handleEnd);
          audio.removeEventListener('error', handleError);
          // Store this message ID as the last one played
          lastPlayedMessageRef.current = messageId;
        };
        const handleError = (e) => {
          console.error(`🗣️ [TTS] Audio error for message ${messageId}:`, e);
          activeAudioPlayersRef.current.delete(audio); // Cleanup registry

          setAudioError("Failed to play synthesized audio.");
          setIsPlayingAudio(null);
          URL.revokeObjectURL(audioUrl);
          audioPlayerRef.current = null;
          audio.removeEventListener('ended', handleEnd);
          audio.removeEventListener('error', handleError);
          lastPlayedMessageRef.current = null; // Reset on error
        };

        audio.addEventListener('ended', handleEnd);
        audio.addEventListener('error', handleError);
        await audio.play();

      } else {
        // --- Web Audio API path (pitch shift) ---
        console.log(`🗣️ [TTS] Using Web Audio API for pitch-shifted playback...`);
        const resp = await fetch(audioUrl);
        const arrayBuf = await resp.arrayBuffer();
        const ctx = new AudioContext();
        const buf = await ctx.decodeAudioData(arrayBuf);

        const source = ctx.createBufferSource();
        source.buffer = buf;
        source.playbackRate.value = currentTtsSpeed;
        source.detune.value = currentTtsPitch * 100; // Semitones -> cents

        source.connect(ctx.destination);

        // Store for cleanup
        audioPlayerRef.current = { ctx, source, audioUrl };
        activeAudioPlayersRef.current.add({ ctx, source }); // Register

        source.onended = () => {
          console.log(`🗣️ [TTS] Web Audio playback ended for message ${messageId}`);
          activeAudioPlayersRef.current.delete({ ctx, source }); // Cleanup
          setIsPlayingAudio(null);
          try { ctx.close(); } catch (e) { }
          URL.revokeObjectURL(audioUrl);
          audioPlayerRef.current = null;
          lastPlayedMessageRef.current = messageId;
        };

        source.start();
      }

    } catch (error) {
      console.error("🗣️ [TTS] Error:", error);
      setAudioError(error?.message || "Unknown TTS error");
      setIsPlayingAudio(null);
    }

  }, [settings, messages, stopTTS, isPlayingAudio]);

  // Memory agent integration functions
  // Memory Agent Integration
  // ----------------------------------
  const fetchMemoriesFromAgent = useCallback(
    async (prompt) => {
      try {
        const userId = userProfile?.id || memoryContext?.activeProfileId;
        if (!userId) return [];

        const res = await fetch(`${MEMORY_API_URL}/memory/relevant`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            userProfile,
            systemTime: new Date().toISOString(),
            requestType: 'memoryRetrieval'
          })
        });

        if (!res.ok) throw new Error(`Status ${res.status}`);
        const data = await res.json();
        return data.memories || [];
      } catch (err) {
        console.warn('🧠 Memory fetch error:', err);
        return [];
      }
    },
    [MEMORY_API_URL, userProfile, memoryContext]
  );

  const observeConversationWithAgent = useCallback(
    async (prompt, response) => {
      if (!autoMemoryEnabled) return;

      const userId = userProfile?.id || memoryContext?.activeProfileId;
      if (!userId) return;

      try {
        await fetch(`${MEMORY_API_URL}/memory/observe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_message: prompt,
            ai_response: response,
            user_name: userId,
            userProfile: userProfile,
            systemTime: new Date().toISOString()
          })
        });
      } catch (err) {
        console.error('🧠 Observation failed:', err);
      }
    },
    [autoMemoryEnabled, userProfile, memoryContext, MEMORY_API_URL]
  );

  // ----------------------------------
  // Model Management
  // ----------------------------------
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${PRIMARY_API_URL}/models`);
      if (!res.ok) throw new Error(res.status);
      const { available_models } = await res.json();
      setAvailableModels(available_models || []);
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  }, [PRIMARY_API_URL]);

  const fetchLoadedModels = useCallback(async () => {
    let primaryModels = [];
    try {
      const res = await fetch(`${PRIMARY_API_URL}/models/loaded`);
      if (res.ok) {
        const { loaded_models } = await res.json();
        primaryModels = loaded_models || [];
        const gpu0 = primaryModels.find(m => m.gpu_id === 0);
        if (gpu0) {
          setPrimaryModel(gpu0.name);
          setActiveModel(gpu0.name);
        }
      }
    } catch {
      // ignore
    }

    let secondaryModels = [];
    try {
      const res = await fetch(`${SECONDARY_API_URL}/models/loaded`);
      if (res.ok) {
        const { loaded_models } = await res.json();
        secondaryModels = loaded_models || [];
        const gpu1 = secondaryModels.find(m => m.gpu_id === 1);
        if (gpu1) setSecondaryModel(gpu1.name);
      }
    } catch {
      console.warn('Secondary API unavailable');
    }

    setLoadedModels([...primaryModels, ...secondaryModels]);
  }, [PRIMARY_API_URL, SECONDARY_API_URL]);

  const loadModel = useCallback(
    async (name, gpu = 0, contextLength = settings.contextLength) => {
      setIsModelLoading(true);
      try {
        const api = gpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
        const res = await fetch(`${api}/models/load/${name}?gpu_id=${gpu}&context_length=${contextLength}`, { method: 'POST' });
        if (!res.ok) throw new Error(name);
        await fetchLoadedModels();
        if (gpu === 0) { setPrimaryModel(name); setActiveModel(name); }
        else if (gpu === 1) setSecondaryModel(name);
        return true;
      } catch (err) {
        console.error('Load model error:', err);
        setApiError(err.message);
        return false;
      } finally {
        setIsModelLoading(false);
      }
    },
    [fetchLoadedModels, PRIMARY_API_URL, SECONDARY_API_URL, settings.contextLength]
  );
  const loadSttEngine = useCallback(async (engine = 'whisper', gpuId = 1) => {
    try {
      const response = await fetch(`${SECONDARY_API_URL}/stt/load-engine`, { // <-- CORRECTED
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine, gpu_id: gpuId }),
      });
      if (!response.ok) throw new Error('Failed to load STT engine');
      const data = await response.json();
      console.log('STT Engine loaded:', data.message);
      alert('STT Engine loaded successfully on GPU ' + gpuId);
    } catch (error) {
      console.error('Error loading STT engine:', error);
      alert('Failed to load STT engine.');
    }
  }, [SECONDARY_API_URL]);

  const loadTtsEngine = useCallback(async (engine = 'kokoro', gpuId = 1) => {
    try {
      const response = await fetch(`${SECONDARY_API_URL}/tts/load-engine`, { // <-- CORRECTED
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine, gpu_id: gpuId }),
      });
      if (!response.ok) throw new Error('Failed to load TTS engine');
      const data = await response.json();
      console.log('TTS Engine loaded:', data.message);
      alert('TTS Engine loaded successfully on GPU ' + gpuId);
    } catch (error) {
      console.error('Error loading TTS engine:', error);
      alert('Failed to load TTS engine.');
    }
  }, [SECONDARY_API_URL]); // <-- CORRECTED
  const unloadModel = useCallback(
    async (name) => {
      try {
        const info = loadedModels.find(m => m.name === name);
        if (!info) return false;
        const api = info.gpu_id === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
        const res = await fetch(`${api}/models/unload/${name}`, { method: 'POST' });
        if (!res.ok) throw new Error(name);
        await fetchLoadedModels();
        if (primaryModel === name) { setPrimaryModel(null); setActiveModel(null); }
        if (secondaryModel === name) setSecondaryModel(null);
        return true;
      } catch (err) {
        console.error('Unload model error:', err);
        setApiError(err.message);
        return false;
      }
    },
    [fetchLoadedModels, loadedModels, primaryModel, secondaryModel, PRIMARY_API_URL, SECONDARY_API_URL]
  );

  // ----------------------------------
  // Conversation Management
  // ----------------------------------
  const createNewConversation = useCallback(() => {
    console.log('🔍 [DEBUG] Creating new conversation');
    const id = generateUniqueId();
    // Helper to replace tags
    const replaceTags = (text, charName, userName) => {
      if (!text) return '';
      return text
        .replace(/{{char}}/gi, charName)
        .replace(/{{user}}/gi, userName);
    };

    const charName = primaryCharacter?.name || 'Character';
    const userName = settings.multiRoleMode && userCharacter?.name
      ? userCharacter.name
      : (userProfile?.name || userProfile?.username || 'User');
    const firstMessage = primaryCharacter?.first_message
      ? replaceTags(primaryCharacter.first_message, charName, userName)
      : null;

    const defaultUserCharacter = userCharacter || null;

    const initial = firstMessage
      ? [{
        id: generateUniqueId(),
        role: 'bot',
        content: firstMessage,
        modelId: 'primary',
        characterName: charName,
        characterId: primaryCharacter?.id,
        avatar: primaryCharacter?.avatar
      }]
      : [];

    const defaultActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    const defaultActiveWeights = buildDefaultCharacterWeights(characters);
    const conv = {
      id,
      name: 'New Chat',
      messages: initial,
      characterIds: {
        primary: primaryCharacter?.id,
        secondary: secondaryCharacter?.id,
        user: defaultUserCharacter?.id || null
      },
      activeCharacterIds: defaultActiveIds,
      activeCharacterWeights: defaultActiveWeights,
      multiRoleContext: '',
      created: new Date().toISOString(),
      requiresTitle: true
    };
    console.log('🔍 [DEBUG] New conversation has requiresTitle flag:', conv.requiresTitle);
    setConversations(prev => [...prev, conv]);
    setActiveConversation(id);
    setMessages(initial);
    setDualModeEnabled(false);
    setActiveCharacter(primaryCharacter || null);
    setUserCharacterId(defaultUserCharacter?.id || null);
    setActiveCharacterIds(defaultActiveIds);
    setActiveCharacterWeights(defaultActiveWeights);
    setMultiRoleContext('');
    generateChatTitle(id, initial, primaryCharacter?.name || '', secondaryCharacter?.name || '')
    return conv;
  }, [primaryCharacter, secondaryCharacter, userCharacter, characters, settings.multiRoleMode, userProfile, setConversations, setActiveConversation, setMessages, setDualModeEnabled, setActiveCharacter, setUserCharacterId, setActiveCharacterIds, setActiveCharacterWeights, generateChatTitle]);

  const handleConversationClick = useCallback((id) => {
    if (activeConversation) {
      setConversations(prev => prev.map(c => c.id === activeConversation ? { ...c, messages } : c));
    }
    setActiveConversation(id);
    const sel = conversations.find(c => c.id === id) || {};
    setMessages(Array.isArray(sel.messages) ? sel.messages : []);

    const { primary, secondary, user } = sel.characterIds || {};
    const primChar = characters.find(c => c.id === primary) || null;
    const secChar = characters.find(c => c.id === secondary) || null;
    setPrimaryCharacter(primChar);
    setSecondaryCharacter(secChar);
    setActiveCharacter(primChar);
    setUserCharacterId(user || null);
    const defaultActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    setActiveCharacterIds(Array.isArray(sel.activeCharacterIds) ? sel.activeCharacterIds : defaultActiveIds);
    const defaultWeights = buildDefaultCharacterWeights(characters);
    setActiveCharacterWeights(
      sel.activeCharacterWeights && typeof sel.activeCharacterWeights === 'object'
        ? sel.activeCharacterWeights
        : defaultWeights
    );
    setMultiRoleContext(typeof sel.multiRoleContext === 'string' ? sel.multiRoleContext : '');
  }, [activeConversation, conversations, messages, characters, setConversations, setActiveConversation, setMessages, setPrimaryCharacter, setSecondaryCharacter, setActiveCharacter, setActiveCharacterIds, setActiveCharacterWeights, setMultiRoleContext]);

  // ----------------------------------
  // Character Management
  // ----------------------------------
  const loadCharacters = useCallback(async () => {
    try {
      const saved = localStorage.getItem('llm-characters');
      if (saved) {
        const parsed = JSON.parse(saved).map(normalizeCharacter);
        setCharacters(parsed);
        const conv = conversations.find(c => c.id === activeConversation);
        const charId = conv?.characterIds?.primary;
        if (charId) setActiveCharacter(parsed.find(c => c.id === charId) || null);
        const userId = conv?.characterIds?.user;
        if (userId) setUserCharacterId(userId);
      }
    } catch (err) {
      console.error('Load chars error:', err);
    }
  }, [activeConversation, conversations]);

  const saveCharacter = useCallback((data) => {
    let savedCharacter = { ...data, chat_role: normalizeChatRole(data?.chat_role) };

    setCharacters(prev => {
      const list = prev.slice();

      if (!data.id) {
        // Create new character
        savedCharacter = {
          ...savedCharacter,
          id: `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          created_at: new Date().toISOString().split('T')[0]
        };
        list.push(savedCharacter);
        console.log('Creating new character:', savedCharacter.name, 'with ID:', savedCharacter.id);
      } else {
        // Update existing character
        const idx = list.findIndex(c => c.id === data.id);
        if (idx > -1) {
          savedCharacter = { ...savedCharacter };
          list[idx] = savedCharacter;
          console.log('Updating existing character:', savedCharacter.name, 'with ID:', savedCharacter.id);
        } else {
          // ID provided but character not found, treat as new
          savedCharacter = {
            ...savedCharacter,
            id: `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
            created_at: new Date().toISOString().split('T')[0]
          };
          list.push(savedCharacter);
          console.log('Character not found, creating new:', savedCharacter.name, 'with ID:', savedCharacter.id);
        }
      }

      try {
        let normalizedList = list.map(normalizeCharacter);
        if (savedCharacter.chat_role === 'user') {
          normalizedList = normalizedList.map(char => {
            if (char.id === savedCharacter.id) return { ...char, chat_role: 'user' };
            if (normalizeChatRole(char.chat_role) === 'user') return { ...char, chat_role: 'npc' };
            return char;
          });
        }
        localStorage.setItem('llm-characters', JSON.stringify(normalizedList));
        console.log('Characters saved to localStorage:', normalizedList.length, 'total characters');
        return normalizedList;
      } catch (error) {
        console.error('Failed to save characters to localStorage:', error);
      }

      return list.map(normalizeCharacter);
    });

    return savedCharacter; // Return the character with its final ID
  }, []);

  const deleteCharacter = useCallback((id) => {
    setCharacters(prev => {
      const filtered = prev.filter(c => c.id !== id);
      localStorage.setItem('llm-characters', JSON.stringify(filtered));
      return filtered;
    });
    if (activeCharacter?.id === id) setActiveCharacter(null);
    if (userCharacterId === id) setUserCharacterId(null);
  }, [activeCharacter, userCharacterId]);

  const duplicateCharacter = useCallback((id) => {
    const orig = characters.find(c => c.id === id);
    if (!orig) {
      console.error('Original character not found for duplication:', id);
      return;
    }

    // Create a deep copy and clear the ID so saveCharacter will assign a new one
    const duplicatedData = {
      ...orig,
      id: null, // Clear ID so saveCharacter will create a new one
      name: `${orig.name} (Copy)`,
      created_at: new Date().toISOString().split('T')[0] // Update creation date
    };

    console.log('Duplicating character:', orig.name, '→', duplicatedData.name);
    saveCharacter(duplicatedData);
  }, [characters, saveCharacter]);

  const setCharacterChatRole = useCallback((id, role) => {
    const normalizedRole = normalizeChatRole(role);
    setCharacters(prev => {
      const updated = prev.map(char => {
        if (char.id === id) return { ...char, chat_role: normalizedRole };
        if (normalizedRole === 'user' && normalizeChatRole(char.chat_role) === 'user') {
          return { ...char, chat_role: 'npc' };
        }
        return normalizeCharacter(char);
      });
      localStorage.setItem('llm-characters', JSON.stringify(updated));
      return updated;
    });
    if (normalizedRole === 'user') setUserCharacterId(id);
  }, []);

  const setUserCharacterById = useCallback((id) => {
    const nextId = id || null;
    setUserCharacterId(nextId);
    if (settings.multiRoleMode && nextId) {
      setCharacterChatRole(nextId, 'user');
    }
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c => (
          c.id === activeConversation
            ? { ...c, characterIds: { ...c.characterIds, user: nextId } }
            : c
        ))
      );
    }
  }, [activeConversation, setConversations, setCharacterChatRole, settings.multiRoleMode]);

  const updateActiveCharacterIds = useCallback((ids) => {
    const unique = Array.from(new Set(ids || [])).filter(Boolean);
    const nextWeights = { ...(activeCharacterWeights || {}) };
    unique.forEach(id => {
      if (nextWeights[id] == null) nextWeights[id] = 50;
    });
    setActiveCharacterIds(unique);
    setActiveCharacterWeights(nextWeights);
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c => (
          c.id === activeConversation
            ? { ...c, activeCharacterIds: unique, activeCharacterWeights: nextWeights }
            : c
        ))
      );
    }
  }, [activeConversation, activeCharacterWeights, setConversations]);

  const updateActiveCharacterWeights = useCallback((updates) => {
    const updateMap = updates && typeof updates === 'object' ? updates : {};
    setActiveCharacterWeights(prev => {
      const next = { ...(prev || {}) };
      Object.keys(updateMap).forEach((id) => {
        const raw = updateMap[id];
        const parsed = Number.parseInt(raw, 10);
        if (!Number.isFinite(parsed)) return;
        next[id] = Math.max(1, Math.min(100, parsed));
      });
      if (activeConversation) {
        setConversations(cs =>
          cs.map(c =>
            c.id === activeConversation
              ? { ...c, activeCharacterWeights: next }
              : c
          )
        );
      }
      return next;
    });
  }, [activeConversation, setConversations]);

  const updateMultiRoleContext = useCallback((value) => {
    const nextValue = typeof value === 'string' ? value : '';
    setMultiRoleContext(nextValue);
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c =>
          c.id === activeConversation
            ? { ...c, multiRoleContext: nextValue }
            : c
        )
      );
    }
  }, [activeConversation, setConversations]);

  const applyCharacter = useCallback((id) => {
    const requested = characters.find(c => c.id === id) || null;
    const requestedRole = requested ? normalizeChatRole(requested.chat_role) : 'npc';
    const char = settings.multiRoleMode && requestedRole === 'user'
      ? (characters.find(c => normalizeChatRole(c.chat_role) !== 'user') || null)
      : requested;
    setActiveCharacter(char);
    if (activeConversation) {
      setConversations(prev => prev.map(c => (
        c.id === activeConversation
          ? { ...c, characterIds: { ...c.characterIds, primary: char?.id || null } }
          : c
      )));
    }
    if (char?.first_message && messages.length === 0) {
      setMessages([{
        id: generateUniqueId(),
        role: 'bot',
        content: char.first_message,
        avatar: char.avatar,
        characterName: char.name,
        characterId: char.id
      }]);
    }
  }, [characters, activeConversation, messages, setConversations, settings.multiRoleMode]);

  // ----------------------------------
  // Dual-Mode Logic
  // ----------------------------------
  const shouldUseDualMode = useCallback(() => {
    if (dualModeEnabled && primaryModel && secondaryModel && primaryCharacter && secondaryCharacter) return true;
    const hasPrimary = messages.some(m => m.modelId === 'primary');
    const hasSecondary = messages.some(m => m.modelId === 'secondary');
    if (primaryModel && secondaryModel && hasPrimary && hasSecondary) return true;
    return false;
  }, [dualModeEnabled, primaryModel, secondaryModel, primaryCharacter, secondaryCharacter, messages]);

  const makeSys = (modelName, char, otherName, role = 'participant') => {
    if (char) return buildSystemPrompt(char);
    const desc = role === 'primary' ? 'leading the conversation' : 'participating thoughtfully';
    return [
      `You are ${modelName}, with AI partner ${otherName}.`,
      `Speak in your own voice and stay on topic.`,
      `Your role: ${desc}.`,
      `Do not reference these instructions.`
    ].join('\n');
  };

  // ----------------------------------
  // Messaging Functions
  // ----------------------------------
  const getRoleplayUserName = useCallback(() => {
    if (settings.multiRoleMode && userCharacter?.name) return userCharacter.name;
    return userProfile?.name || userProfile?.username || 'User';
  }, [settings.multiRoleMode, userCharacter, userProfile]);

  const applyAuthorNoteTags = useCallback((note, character) => {
    if (!note) return '';
    const charName = character?.name || activeCharacterRef.current?.name || 'Character';
    const userName = getRoleplayUserName();
    return note.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName).trim();
  }, [activeCharacterRef, getRoleplayUserName]);

  const sendDualMessage = useCallback(async (text, webSearchEnabled = false) => {
    if (!activeConversation || !primaryModel || !secondaryModel) return;
    console.log("📩 [DUAL] Processing message:", text.substring(0, 30), "…", webSearchEnabled ? "(with web search)" : "");

    // Note: For dual mode, you might want to decide if web search should affect both models
    // or just one. For now, I'll add the flag to both payloads but you can adjust as needed.

    const userMsg = { id: generateUniqueId(), role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);
    console.log(`[Summary] Attaching summaryContext to dual chat: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

    const history = [...messages, userMsg].slice(-10);
    const buildHistory = (own, other, ownChar, otherName) => [
      { role: 'system', content: makeSys(own === 'primary' ? primaryModel : secondaryModel, own === 'primary' ? primaryCharacter : secondaryCharacter, own === 'primary' ? secondaryModel : primaryModel, own) },
      ...history.map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content.replace(/\n\n/g, '\n') }))
    ];

    try {
      const [pRes, sRes] = await Promise.all([
        fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_name: primaryModel,
            messages: buildHistory('primary', 'secondary', primaryCharacter, secondaryModel),
            gpu_id: 0,
            userProfile,
            authorNote: applyAuthorNoteTags(settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : null, primaryCharacter) || undefined,
            use_web_search: webSearchEnabled, // NEW: Add to primary
            summaryContext: summaryContextForRequest
          })
        }).then(r => r.json()),
        fetch(`${SECONDARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_name: secondaryModel,
            messages: buildHistory('secondary', 'primary', secondaryCharacter, primaryModel),
            gpu_id: 1,
            userProfile,
            authorNote: applyAuthorNoteTags(settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : null, secondaryCharacter) || undefined,
            use_web_search: false, // Usually only primary should do web search in dual mode
            summaryContext: summaryContextForRequest
          })
        }).then(r => r.json())
      ]);

      const pId = generateUniqueId();
      const sId = generateUniqueId();
      const pText = cleanModelOutput(pRes.text);
      const sText = cleanModelOutput(sRes.text);

      setMessages(prev => [...prev,
      { id: pId, role: 'bot', content: pText, modelId: 'primary', characterName: primaryCharacter?.name, characterId: primaryCharacter?.id, avatar: primaryCharacter?.avatar },
      { id: sId, role: 'bot', content: sText, modelId: 'secondary', characterName: secondaryCharacter?.name, characterId: secondaryCharacter?.id, avatar: secondaryCharacter?.avatar }
      ]);

      await observeConversationWithAgent(text, `${pRes.text}\n\n${sRes.text}`);

      // Auto-play TTS for BOTH messages if enabled
      if (settings.ttsAutoPlay && settings.ttsEnabled) {
        // We'll play them sequentially? Or just primary? 
        // For now let's play primary, then secondary might overlap or we'd need a queue.
        // Actually playTTS stops current audio, so primary then secondary would just play secondary.
        // Let's just play primary for now to avoid complexity, or the characters will talk over each other.
        playTTS(pId, pText, getTtsOverridesForCharacter(primaryCharacter));
      }
    } catch (err) {
      console.error('Dual chat error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [activeConversation, primaryModel, secondaryModel, messages, primaryCharacter, secondaryCharacter, PRIMARY_API_URL, SECONDARY_API_URL, userProfile, settings, observeConversationWithAgent, summaryContextForRequest, getTtsOverridesForCharacter, applyAuthorNoteTags]);

  const getMultiRoleContextBlock = useCallback(() => {
    if (!settings.multiRoleMode) return '';
    const trimmed = (multiRoleContext || '').trim();
    if (!trimmed) return '';
    return `\n\n[GROUP SCENE CONTEXT]\n${trimmed}`;
  }, [settings.multiRoleMode, multiRoleContext]);

  const getNarratorCharacter = useCallback(() => {
    if (!settings.narratorEnabled) return null;
    const name = (settings.narratorName || '').trim() || 'Narrator';
    return {
      id: NARRATOR_CHARACTER_ID,
      name,
      description: 'A narrator who describes the scene and transitions.',
      personality: '',
      background: '',
      speech_style: 'Narrative, concise, scene-setting.',
      chat_role: 'narrator',
      avatar: settings.narratorAvatar || null,
      isNarratorSystem: true
    };
  }, [settings.narratorEnabled, settings.narratorName, settings.narratorAvatar]);

  const buildNarratorSystemPrompt = useCallback(() => {
    const narratorName = (settings.narratorName || '').trim() || 'Narrator';
    const userName = getRoleplayUserName();
    const charName = activeCharacterRef.current?.name || primaryCharacter?.name || 'Character';
    const rawInstructions = settings.narratorInstructions || '';
    const narratorInstructions = rawInstructions
      .replace(/{{user}}/gi, userName)
      .replace(/{{char}}/gi, charName)
      .trim();

    const summaryContext = activeContextSummary || userProfile?.activeContextSummary || '';
    const summaryBlock = summaryContext
      ? `\n\n[PREVIOUS STORY SUMMARY]:\n${summaryContext}\n[End of Summary]\n`
      : '';
    const storyContext = getStoryTrackerContext();
    const groupContext = getMultiRoleContextBlock();
    const guidance = narratorInstructions ? `\n\nNARRATION GUIDANCE:\n${narratorInstructions}` : '';

    return `You are ${narratorName}, the narrator of this roleplay.

Purpose:
- Describe scene transitions, atmosphere, and world context.
- Keep narration concise (2-5 sentences) unless the user asks for more.
- Do not speak for ${userName} or write dialogue on behalf of characters.
- Avoid labeling lines with character names or using speaker tags.

If you must mention character actions, do so briefly in third-person prose.
${guidance}${groupContext}${summaryBlock}${storyContext}`;
  }, [activeCharacterRef, primaryCharacter, settings.narratorInstructions, settings.narratorName, activeContextSummary, userProfile, getRoleplayUserName, getMultiRoleContextBlock]);

  const buildRoleplayRosterBlock = useCallback(() => {
    if (!settings.multiRoleMode) return '';
    const normalized = (characters || []).map(normalizeCharacter);
    const rosterIds = Array.isArray(activeCharacterIds) ? activeCharacterIds : [];
    const rosterSet = rosterIds.length ? new Set(rosterIds) : null;
    const rosterFiltered = rosterSet ? normalized.filter(c => rosterSet.has(c.id)) : normalized;
    const userName = getRoleplayUserName();
    const narratorName = settings.narratorEnabled
      ? ((settings.narratorName || '').trim() || 'Narrator')
      : null;
    const npcNames = rosterFiltered
      .filter(c => normalizeChatRole(c.chat_role) === 'npc')
      .map(c => c.name)
      .filter(Boolean);
    const narratorNames = rosterFiltered
      .filter(c => normalizeChatRole(c.chat_role) === 'narrator')
      .map(c => c.name)
      .filter(Boolean);

    const narratorLine = narratorName
      ? `\nNarrator: ${narratorName}`
      : '';
    const narratorListLine = narratorNames.length
      ? `\nNarrator Characters: ${narratorNames.join(', ')}`
      : '';

    return `\n\n[ROLEPLAY MODE]\nUser Character: ${userName}\nCharacters: ${npcNames.length ? npcNames.join(', ') : 'None'}${narratorListLine}${narratorLine}\nRules: Never write dialogue or actions for ${userName}. Only one character speaks per response. Do not include other characters' dialogue.`;
  }, [characters, activeCharacterIds, getRoleplayUserName, settings.multiRoleMode, settings.narratorEnabled, settings.narratorName]);

  const buildSpeakerSelectionPrompt = useCallback((recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText) => {
    const candidateList = candidates
      .map(c => `- ${c.name} (${normalizeChatRole(c.chat_role)})`)
      .join('\n');
    const weightList = candidates
      .map(c => {
        const raw = weightMap?.[c.id];
        const parsed = Number.parseInt(raw, 10);
        const weight = Number.isFinite(parsed) ? parsed : 50;
        return `- ${c.name}: ${weight}`;
      })
      .join('\n');
    const recentHistory = recentMessages
      .filter(m => m.role !== 'system')
      .slice(-10)
      .map(m => {
        if (m.role === 'user') {
          return `${m.characterName || userName}: ${m.content}`;
        }
        if (m.role === 'bot') {
          return `${m.characterName || 'Assistant'}: ${m.content}`;
        }
        return `System: ${m.content}`;
      })
      .join('\n');

    const lastSpeakerLine = lastSpeakerName ? `Last speaker: ${lastSpeakerName} (avoid repeating if possible)` : 'Last speaker: None';
    const userLine = lastUserText ? `Latest user message:\n${lastUserText}` : 'Latest user message: None';
    return `You are a scriptwriter selecting the next speaker in a roleplay.

Candidates:
${candidateList || 'None'}

Selection weights (higher = more likely):
${weightList || 'None'}

User Character: ${userName} (never select)
${lastSpeakerLine}
${userLine}

Rules:
- Prefer characters explicitly addressed by the user.
- Prefer higher-weight characters when multiple are plausible.

Recent conversation:
${recentHistory}

Return ONLY valid JSON:
{"selection": "Character Name"}`;
  }, []);

  const selectNextSpeaker = useCallback(async (recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText) => {
    if (!primaryModel || !PRIMARY_API_URL) return null;
    if (!candidates.length) return null;

    const selectionSystem = "You are a careful assistant that returns only valid JSON.";
    const selectionPrompt = buildSpeakerSelectionPrompt(recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText);
    const prompt = formatPrompt([{ role: 'user', content: selectionPrompt }], primaryModel, selectionSystem);

    try {
      const res = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          model_name: primaryModel,
          directProfileInjection: true,
          temperature: 0.2,
          top_p: 0.9,
          top_k: 40,
          repetition_penalty: 1.05,
          max_tokens: 120,
          gpu_id: 0,
          stream: false,
          request_purpose: 'speaker_selection'
        })
      });

      if (!res.ok) return null;
      const data = await res.json();
      const rawText = cleanModelOutput(data.text || '');
      const jsonText = extractFirstJson(rawText) || extractFirstJson(data.text || '');
      if (!jsonText) return null;

      const parsed = JSON.parse(jsonText);
      const selectionName = parsed?.selection ? String(parsed.selection).trim() : '';
      if (!selectionName) return null;

      const normalizedSelection = selectionName.toLowerCase();
      return candidates.find(c => (c.name || '').toLowerCase() === normalizedSelection) || null;
    } catch (error) {
      console.warn('Speaker selection failed:', error);
      return null;
    }
  }, [PRIMARY_API_URL, primaryModel, formatPrompt, cleanModelOutput, buildSpeakerSelectionPrompt]);

  const resolveSpeakerCharacter = useCallback(async (text, recentMessages, options = {}) => {
    const { speakerCharacterId = null, forceAutoSelectSpeaker = false, ignoreMentionedCandidates = false } = options;
    if (speakerCharacterId === NARRATOR_CHARACTER_ID) {
      return getNarratorCharacter();
    }
    if (speakerCharacterId) {
      return characters.find(c => c.id === speakerCharacterId) || null;
    }

    if (!settings.multiRoleMode) return activeCharacterRef.current || null;

    const narrator = getNarratorCharacter();
    const allowAutoSelect = forceAutoSelectSpeaker || settings.autoSelectSpeaker;
    const narratorInterval = Number.parseInt(settings.narratorInterval, 10);
    const shouldUseNarrator = () => {
      if (!narrator) return false;
      if (!allowAutoSelect) return false;
      if (!Number.isFinite(narratorInterval) || narratorInterval <= 0) return false;
      let botSince = 0;
      for (let i = recentMessages.length - 1; i >= 0; i -= 1) {
        const msg = recentMessages[i];
        if (msg?.role !== 'bot') continue;
        if (msg.characterId === NARRATOR_CHARACTER_ID) {
          if (botSince === 0) return false;
          return botSince >= narratorInterval;
        }
        botSince += 1;
      }
      return botSince >= narratorInterval;
    };

    const normalized = (characters || []).map(normalizeCharacter);
    const rosterIds = Array.isArray(activeCharacterIds) ? activeCharacterIds : [];
    const rosterSet = rosterIds.length ? new Set(rosterIds) : null;
    const rosterFiltered = rosterSet ? normalized.filter(c => rosterSet.has(c.id)) : normalized;
    const candidates = rosterFiltered.filter(c => normalizeChatRole(c.chat_role) !== 'user');
    if (!candidates.length) return activeCharacterRef.current || null;

    const getWeight = (id) => {
      const raw = activeCharacterWeights?.[id];
      const parsed = Number.parseInt(raw, 10);
      if (!Number.isFinite(parsed)) return 50;
      return Math.max(1, Math.min(100, parsed));
    };

    const normalizeMatch = (value) => String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .trim();

    const findMentionedCandidate = () => {
      const textMatch = normalizeMatch(text);
      if (!textMatch) return null;
      let best = null;
      let bestWeight = -1;
      for (const candidate of candidates) {
        const nameMatch = normalizeMatch(candidate.name);
        if (!nameMatch) continue;
        if (!textMatch.includes(nameMatch)) continue;
        const weight = getWeight(candidate.id);
        if (weight > bestWeight) {
          best = candidate;
          bestWeight = weight;
        }
      }
      return best;
    };

    const pickWeighted = (excludeId = null) => {
      const pool = candidates.filter(c => c.id !== excludeId);
      if (!pool.length) return candidates[0];
      const total = pool.reduce((sum, c) => sum + getWeight(c.id), 0);
      const roll = Math.random() * total;
      let running = 0;
      for (const candidate of pool) {
        running += getWeight(candidate.id);
        if (roll <= running) return candidate;
      }
      return pool[0];
    };

    const mentioned = ignoreMentionedCandidates ? null : findMentionedCandidate();
    if (mentioned) return mentioned;

    if (shouldUseNarrator()) return narrator;

    if (!allowAutoSelect) {
      const current = activeCharacterRef.current;
      const currentRole = current ? normalizeChatRole(current.chat_role) : 'npc';
      const inRoster = !rosterSet || (current && rosterSet.has(current.id));
      if (current && currentRole !== 'user' && inRoster) return current;
      return pickWeighted();
    }

    if (candidates.length === 1) return candidates[0];

    const findLastSpeaker = () => {
      for (let i = recentMessages.length - 1; i >= 0; i -= 1) {
        const msg = recentMessages[i];
        if (msg?.role !== 'bot') continue;
        if (msg.characterId) {
          const byId = candidates.find(c => c.id === msg.characterId);
          if (byId) return byId;
        }
        if (msg.characterName) {
          const byName = candidates.find(c => (c.name || '').toLowerCase() === String(msg.characterName).toLowerCase());
          if (byName) return byName;
        }
      }
      return null;
    };
    const lastSpeaker = findLastSpeaker();
    const lastSpeakerName = lastSpeaker?.name || null;
    const nextByRotation = () => {
      if (!lastSpeaker) return candidates[0];
      const idx = candidates.findIndex(c => c.id === lastSpeaker.id);
      if (idx === -1) return candidates[0];
      return candidates[(idx + 1) % candidates.length];
    };

    const userName = getRoleplayUserName();
    const selected = await selectNextSpeaker(
      recentMessages,
      candidates,
      userName,
      lastSpeakerName,
      activeCharacterWeights,
      text
    );
    if (!selected) return pickWeighted(lastSpeaker?.id || null);
    if (lastSpeaker && selected.id === lastSpeaker.id && candidates.length > 1) {
      return pickWeighted(lastSpeaker.id);
    }
    return selected;
  }, [activeCharacterRef, characters, activeCharacterIds, activeCharacterWeights, getRoleplayUserName, getNarratorCharacter, selectNextSpeaker, settings.autoSelectSpeaker, settings.multiRoleMode, settings.narratorInterval]);


  // New shared helper to build the FULL generation system prompt (memories, lore, etc.)
  const getGenerationSystemPrompt = useCallback(async (text, character, authorNote = null, options = {}) => {
    const { includeAuthorNote = true } = options;
    console.log('🔍 [SystemPrompt] Building for character:', character?.name, 'User:', userProfile?.name);
    console.log('🔍 [SystemPrompt] Tag replacer check. Character valid?', !!character, 'UserProfile valid?', !!userProfile);
    let systemMsg = character?.id === NARRATOR_CHARACTER_ID && settings.narratorEnabled
      ? buildNarratorSystemPrompt()
      : (character
        ? buildSystemPrompt(character)
        : ('You are a helpful assistant.' + getStoryTrackerContext()));

    if (settings.multiRoleMode) {
      systemMsg += getMultiRoleContextBlock();
      systemMsg += buildRoleplayRosterBlock();
    }

    if (settings.multiRoleMode && character) {
      const activeName = character.name || 'Character';
      systemMsg += `\n\n[ACTIVE SPEAKER]\nYou are speaking ONLY as ${activeName}.\n- Do not write dialogue for any other character.\n- Do not include multiple speakers in one response.\n- Do not prefix lines with character names or labels.\n- If you reference other characters, do so briefly in third-person without quoted speech.`;
    }

    let contextToAdd = '';

    // Helper for tag replacement in lore/profile
    const replaceTags = (content) => {
      if (!content || !character) return content;
      const charName = character.name || 'Character';
      const userName = getRoleplayUserName();
      return content.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName);
    };

    if (settings.directProfileInjection) {
      const userId = userProfile?.id || (typeof memoryContext !== 'undefined' ? memoryContext?.activeProfileId : null);
      if (userId) {
        try {
          const res = await fetch(`${MEMORY_API_URL}/memory/get_all?user_id=${userId}`);
          if (res.ok) {
            const data = await res.json();
            if (data.memories && data.memories.length > 0) {
              const profileString = data.memories.map(mem => {
                const category = mem.category?.replace('_', ' ') || 'memory';
                const importance = mem.importance?.toFixed(1) || 'N/A';
                // Apply tag substitution here
                const content = replaceTags(mem.content);
                return `• ${content} (Category: ${category}, Importance: ${importance})`;
              }).join('\n');
              contextToAdd += `\n\nUSER MEMORY PROFILE:\n${profileString}`;
            }
          }
        } catch (error) {
          console.error("🧠 [Direct Injection] Error:", error);
        }
      }
    } else {
      const agentMem = await fetchMemoriesFromAgent(text);
      if (agentMem.length) {
        contextToAdd += `\n\nUSER CONTEXT:\n` + agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n');
      }
    }


    if (character) {
      const lore = await fetchTriggeredLore(text, character);
      if (lore.length) {
        const loreBlock = lore.map(l => {
          const content = typeof l === 'string' ? l : (l.content || JSON.stringify(l));
          return "• " + replaceTags(content);
        }).join('\n');
        contextToAdd += `\n\n[WORLD KNOWLEDGE - Essential lore guidance for this response]\n${loreBlock}`;
      }
    }

    systemMsg += contextToAdd;

    const effectiveAuthorNote = authorNote || (settings.authorNote && settings.authorNote.trim()) || null;
    if (includeAuthorNote && effectiveAuthorNote) {
      const resolvedAuthorNote = applyAuthorNoteTags(effectiveAuthorNote, character);
      if (resolvedAuthorNote) {
        systemMsg += `\n\n[AUTHOR'S NOTE - Writing style guidance for this response]\n${resolvedAuthorNote}`;
      }
    }

    const hasSummary = systemMsg.includes('[PREVIOUS STORY SUMMARY]');
    console.log(`[Summary] System prompt includes summary: ${hasSummary}`);
    return systemMsg;
  }, [settings, userProfile, MEMORY_API_URL, fetchMemoriesFromAgent, fetchTriggeredLore, buildSystemPrompt, buildNarratorSystemPrompt, buildRoleplayRosterBlock, getRoleplayUserName, getMultiRoleContextBlock, applyAuthorNoteTags]);

  // In AppContext.jsx, replace the entire generateReply function
  const generateReply = useCallback(async (text, recentMessages, onToken = null, options = {}) => {
    const { authorNote = null, webSearchEnabled = false, speakerCharacterId = null } = options;
    const speakerCharacter = await resolveSpeakerCharacter(text, recentMessages, { speakerCharacterId });
    const systemMsg = await getGenerationSystemPrompt(text, speakerCharacter, authorNote, { includeAuthorNote: false });
    console.log(`[Summary] Attaching summaryContext to generateReply: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

    // --- Unified Payload Construction (Matching sendMessage exactly) ---
    const {
      temperature, top_p, top_k, repetition_penalty, frequencyPenalty = 0, presencePenalty = 0,
      antiRepetitionMode = false, use_rag, selectedDocuments = [], streamResponses
    } = settings;

    const historyLimitLocal = 15;
    const historyLimitAPI = 12;
    const currentHistoryLimit = primaryIsAPI ? historyLimitAPI : historyLimitLocal;

    const payload = {
      directProfileInjection: settings.directProfileInjection,
      prompt: formatPrompt(recentMessages.slice(-currentHistoryLimit), primaryModel, systemMsg),
      model_name: primaryModel,
      temperature, top_p, top_k, repetition_penalty,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      anti_repetition_mode: antiRepetitionMode,
      use_rag,
      rag_docs: selectedDocuments,
      use_web_search: webSearchEnabled,
      gpu_id: 0,
      userProfile: { id: userProfile?.id ?? 'anonymous' },
      authorNote: applyAuthorNoteTags(authorNote || (settings.authorNote && settings.authorNote.trim()) || null, speakerCharacter) || undefined,
      summaryContext: summaryContextForRequest,
      memoryEnabled: true,
      stream: streamResponses
    };

    let attempts = 0;
    const maxAttempts = primaryIsAPI ? 2 : 1;
    let success = false;
    let finalOutput = '';

    while (attempts < maxAttempts && !success) {
      attempts++;
      if (attempts > 1) console.log(`🔄 [generateReply] Auto-Retry Attempt ${attempts}...`);

      try {
        const controller = new AbortController();
        const res = await fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal
        });

        if (!res.ok) throw new Error(`Status ${res.status}`);

        if (streamResponses) {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let accumulatedText = '';
          let sseBuffer = '';
          let doneStreaming = false;

          while (!doneStreaming) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            sseBuffer += chunk;
            const events = sseBuffer.split('\n\n');
            sseBuffer = events.pop() || '';

            for (const line of events) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                if (data === '[DONE]') {
                  doneStreaming = true;
                  break;
                }
                try {
                  const parsed = JSON.parse(data);
                  if (parsed.text) {
                    accumulatedText += parsed.text;
                    const currentFull = cleanModelOutput(accumulatedText);
                    if (onToken) onToken(parsed.text, currentFull);
                  }
                  if (parsed.done) {
                    doneStreaming = true;
                    break;
                  }
                } catch (e) { }
              }
            }
          }

          const cleanedText = cleanModelOutput(accumulatedText);
          // If empty and we have attempts left, retry
          if (primaryIsAPI && !cleanedText && attempts < maxAttempts) {
            continue;
          }
          finalOutput = cleanedText;
          success = true;
        } else {
          const data = await res.json();
          const cleaned = cleanModelOutput(data.text);
          if (primaryIsAPI && !cleaned && attempts < maxAttempts) {
            continue;
          }
          if (onToken) onToken(cleaned, cleaned);
          finalOutput = cleaned;
          success = true;
        }
      } catch (error) {
        console.error(`Attempt ${attempts} failed in generateReply:`, error);
        if (attempts >= maxAttempts) throw error;
        await new Promise(r => setTimeout(r, 500));
      }
    }
    return finalOutput;
  }, [
    primaryIsAPI,
    primaryModel,
    settings,
    userProfile?.id,
    PRIMARY_API_URL,
    resolveSpeakerCharacter,
    getGenerationSystemPrompt,
    formatPrompt,
    cleanModelOutput,
    summaryContextForRequest,
    applyAuthorNoteTags
  ]);

  // In AppContext.jsx, replace the entire generateReplyWithOpenAI function
  const generateReplyWithOpenAI = useCallback(async (text, recentMessages) => {
    console.log("🌐 [OpenAI] Processing with OpenAI API format");

    const apiUrl = PRIMARY_API_URL;
    const targetGpuId = 0;

    const convertToOpenAIMessages = (messages, systemPrompt) => {
      // --- DYNAMIC CONTEXT PRUNING for OpenAI format (8k Limit) ---
      const MAX_CONTEXT_TOKENS = 7500;
      const estimateTokens = (str) => Math.ceil((str || '').length / 4);
      const systemTokens = estimateTokens(systemPrompt);
      let availableTokens = MAX_CONTEXT_TOKENS - systemTokens;

      const filtered = messages.filter(msg => (msg.role === 'user' || msg.role === 'bot') && typeof msg.content === 'string');
      const reversed = [...filtered].reverse();
      let sliced = [];
      let currentTokens = 0;
      const MIN_CONTINUITY = 5;

      for (let i = 0; i < reversed.length; i++) {
        const msg = reversed[i];
        const msgTokens = estimateTokens(msg.content) + 10;
        if (i >= MIN_CONTINUITY && (currentTokens + msgTokens) > availableTokens) break;
        sliced.unshift(msg);
        currentTokens += msgTokens;
      }

      const openAiMsgs = sliced.map(msg => ({
        role: msg.role === 'bot' ? 'assistant' : 'user',
        content: msg.content
      }));

      return [{ role: 'system', content: systemPrompt }, ...openAiMsgs];
    };

    const agentMem = await fetchMemoriesFromAgent(text);
    const lore = await fetchTriggeredLore(text, activeCharacter);
    let memoryContext = '';

    // Helper for tag replacement in lore
    const replaceLoreTags = (content) => {
      if (!content || !activeCharacter) return content;
      const charName = activeCharacter.name || 'Character';
      const userName = userProfile?.name || userProfile?.username || 'User';
      return content.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName);
    };

    if (agentMem.length) {
      memoryContext = agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n');
    }
    if (lore.length) {
      const loreContext = lore.map(l => {
        const content = typeof l === 'string' ? l : (l.content || JSON.stringify(l));
        return `• ${replaceLoreTags(content)}`;
      }).join('\n');
      memoryContext += (memoryContext ? "\n\n[WORLD KNOWLEDGE - Essential lore guidance for this response]\n" : "[WORLD KNOWLEDGE - Essential lore guidance for this response]\n") + loreContext;
    }

    // Note: buildSystemPrompt already includes story tracker context
    let systemMsg = activeCharacter ? buildSystemPrompt(activeCharacter) : ('You are a helpful assistant.' + getStoryTrackerContext());
    if (memoryContext) {
      systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
    }

    const historyLimit = 20; // Slicing history for API models
    const slicedMessages = recentMessages.slice(-historyLimit);
    const finalMessages = convertToOpenAIMessages(slicedMessages, systemMsg);

    if (settings.streamResponses) {
      const response = await generateReplyOpenAI({
        messages: finalMessages,
        systemPrompt: null,
        model: primaryModel,
        settings,
        apiUrl,
        apiKey: null,
        stream: true,
        targetGpuId: targetGpuId
      });
      return await processOpenAIStream(response, () => { }, (fullText) => fullText, (error) => { throw error; });
    } else {
      const response = await generateReplyOpenAI({
        messages: finalMessages,
        systemPrompt: null,
        model: primaryModel,
        settings,
        apiUrl,
        apiKey: null,
        stream: false,
        targetGpuId: targetGpuId
      });
      const result = await response.json();
      return result.choices?.[0]?.message?.content || "[No response]";
    }
  }, [
    primaryIsAPI, secondaryIsAPI, primaryModel, secondaryModel, settings,
    fetchMemoriesFromAgent, fetchTriggeredLore, activeCharacter,
    buildSystemPrompt, generateReplyOpenAI, processOpenAIStream,
    PRIMARY_API_URL, SECONDARY_API_URL
  ]);
  const sendMessage = useCallback(async (text, webSearchEnabled = false, authorNote = null) => {
    // Can't send without a model
    if (!primaryModel) {
      console.warn("📩 [SEND] No model loaded, cannot send message");
      return;
    }

    // Auto-create a conversation if none exists
    let currentConversation = activeConversation;
    if (!currentConversation) {
      console.log("📩 [SEND] No active conversation, creating one...");
      const newConv = createNewConversation();
      currentConversation = newConv.id;
      // Give React a moment to update state
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    promptSubmissionStartTime.current = performance.now();
    console.log(`⏱️ [Full Cycle] User submitted prompt: "${text.substring(0, 30)}..."`);

    console.log("📩 [SEND] Processing message:", text.substring(0, 30), "…", webSearchEnabled ? "(with web search)" : "");

    // 1) Append the user message
    const userMsg = {
      id: generateUniqueId(),
      role: 'user',
      content: text
    };
    if (settings.multiRoleMode && userCharacter) {
      userMsg.characterId = userCharacter.id;
      userMsg.characterName = userCharacter.name;
      userMsg.avatar = userCharacter.avatar;
    }
    const postUserHistory = [...messages, userMsg];
    setMessages(() => postUserHistory);
    setIsGenerating(true);

    try {
      // 2) (Optional) Title logic
      const userMsgs = postUserHistory.filter(m => m.role === 'user');
      const isFirst = userMsgs.length === 1;
      const conv = conversations.find(c => c.id === currentConversation);

      if (isFirst && conv?.requiresTitle && !primaryIsAPI) {
        try {
          const title = await generateChatTitle(text, primaryModel);
          if (title && title !== 'New Chat') {
            setConversations(cs =>
              cs.map(c =>
                c.id === currentConversation
                  ? { ...c, name: title, requiresTitle: false }
                  : c
              )
            );
          }
        } catch (titleErr) {
          console.error("🔤 [TITLE] Title generation failed:", titleErr);
        }
      } else if (isFirst && conv?.requiresTitle && primaryIsAPI) {
        setConversations(cs =>
          cs.map(c =>
            c.id === currentConversation
              ? { ...c, name: `${text.substring(0, 25)}...`, requiresTitle: false }
              : c
          )
        );
      }

      // 3) Build System Prompt with layered context
      const speakerCharacter = await resolveSpeakerCharacter(text, postUserHistory);
      const systemMsg = await getGenerationSystemPrompt(text, speakerCharacter, authorNote, { includeAuthorNote: false });
      console.log(`[Summary] Attaching summaryContext to sendMessage: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

      // 4) Prepare payload
      const {
        temperature, top_p, top_k, repetition_penalty, frequencyPenalty = 0, presencePenalty = 0,
        antiRepetitionMode = false, use_rag, selectedDocuments = [], streamResponses
      } = settings;

      const effectiveAuthorNote = applyAuthorNoteTags(authorNote || (settings.authorNote && settings.authorNote.trim()) || null, speakerCharacter) || undefined;

      const historyLimitLocal = 15;
      const historyLimitAPI = 12; // Reduced for stability
      const currentHistoryLimit = primaryIsAPI ? historyLimitAPI : historyLimitLocal;

      const payload = {
        directProfileInjection: settings.directProfileInjection,
        prompt: formatPrompt(postUserHistory.slice(-currentHistoryLimit), primaryModel, systemMsg),
        model_name: primaryModel,
        temperature, top_p, top_k, repetition_penalty,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        anti_repetition_mode: antiRepetitionMode,
        use_rag,
        rag_docs: selectedDocuments,
        use_web_search: webSearchEnabled,
        gpu_id: 0,
        userProfile: { id: userProfile?.id ?? 'anonymous' },
        authorNote: effectiveAuthorNote,
        summaryContext: summaryContextForRequest,
        memoryEnabled: true,
        stream: streamResponses
      };

      // 5) Consolidated Generation Path
      let attempts = 0;
      const maxAttempts = primaryIsAPI ? 2 : 1;
      let success = false;

      while (attempts < maxAttempts && !success) {
        attempts++;
        const botId = generateUniqueId();
        const placeholderBotMessage = {
          id: botId,
          role: 'bot',
          content: '',
          modelId: 'primary',
          characterId: speakerCharacter?.id,
          characterName: speakerCharacter?.name,
          avatar: speakerCharacter?.avatar,
          isStreaming: streamResponses,
        };

        if (attempts > 1) {
          console.log(`🔄 [Auto-Retry] Attempt ${attempts}...`);
          setMessages(prev => {
            const lastSlice = prev.slice();
            const last = lastSlice[lastSlice.length - 1];
            if (last && last.role === 'bot' && (last.content === '' || last.content.includes('[Error'))) {
              lastSlice.pop();
            }
            return [...lastSlice, placeholderBotMessage];
          });
        } else {
          setMessages(prev => [...postUserHistory, placeholderBotMessage]);
        }

        if (streamResponses) startStreamingTTS(botId, getTtsOverridesForCharacter(speakerCharacter));

        try {
          const controller = new AbortController();
          setAbortController(controller);

          const res = await fetch(`${PRIMARY_API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
          });

          if (!res.ok) throw new Error(`Status ${res.status}`);

          if (streamResponses) {
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let accumulated = '';
            let lastSentContent = '';
            let llmStreamComplete = false;
            let sseBuffer = '';

            // (checkAndEndTts removed - and moved to direct call)

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              sseBuffer += chunk;
              const events = sseBuffer.split('\n\n');
              sseBuffer = events.pop() || '';

              for (const event of events) {
                if (!event.startsWith('data: ')) continue;
                const dataStr = event.slice(6).trim();
                if (dataStr === '[DONE]') {
                  llmStreamComplete = true;
                  break;
                }
                try {
                  const parsed = JSON.parse(dataStr);
                  if (parsed.text) {
                    accumulated += parsed.text;
                    const partial = cleanModelOutput(accumulated);
                    const newTextChunk = partial.slice(lastSentContent.length);
                    if (newTextChunk) addStreamingText(newTextChunk);
                    lastSentContent = partial;
                    setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: partial } : m));
                  }
                  if (parsed.done) {
                    llmStreamComplete = true;
                    break;
                  }
                } catch (e) { }
              }
              if (llmStreamComplete) break;
            }

            const finalCleaned = cleanModelOutput(accumulated);
            if (primaryIsAPI && !finalCleaned && attempts < maxAttempts) {
              console.warn("⚠️ [Auto-Retry] Empty response from API, retrying...");
              if (streamResponses) endStreamingTTS();
              continue;
            }

            setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: finalCleaned, isStreaming: false } : m));
            observeConversationWithAgent(text, finalCleaned);
            success = true;
            if (streamResponses) endStreamingTTS();
          } else {
            // Non-streaming
            const data = await res.json();
            const cleanedText = cleanModelOutput(data.text);
            if (primaryIsAPI && !cleanedText && attempts < maxAttempts) {
              continue;
            }
            setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: cleanedText, isStreaming: false } : m));
            observeConversationWithAgent(text, cleanedText);
            success = true;

            // Trigger TTS if autoplay is on
            if (settings.ttsAutoPlay && settings.ttsEnabled) {
              playTTS(botId, cleanedText, getTtsOverridesForCharacter(speakerCharacter));
            }
          }
        } catch (err) {
          if (err.name === 'AbortError') {
            console.log('🛑 Generation stopped by user');
            setMessages(prev => prev.map(m => m.id === botId ? { ...m, isStreaming: false } : m));
            success = true;
          } else {
            console.error(`Error on attempt ${attempts}:`, err);
            if (attempts >= maxAttempts) {
              setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Error: ${err.message}]`, isStreaming: false } : m));
              if (streamResponses) endStreamingTTS();
              success = true;
            } else if (primaryIsAPI) {
              if (streamResponses) endStreamingTTS();
              continue;
            }
          }
        } finally {
          setAbortController(null);
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
    } finally {
      setIsGenerating(false);
    }
  }, [
    activeConversation, primaryModel, messages, conversations, settings, userCharacter,
    userProfile?.id, PRIMARY_API_URL, fetchMemoriesFromAgent, fetchTriggeredLore, MEMORY_API_URL,
    formatPrompt, cleanModelOutput, generateChatTitle, memoryContext, resolveSpeakerCharacter,
    observeConversationWithAgent, generateReply, primaryIsAPI, createNewConversation, getStoryTrackerContext,
    summaryContextForRequest, getTtsOverridesForCharacter, applyAuthorNoteTags
  ]);

  const generateCallModeFollowUp = useCallback(async (options = {}) => {
    if (isGenerating) return;

    const { prompt = null } = options;
    const lastBot = [...(messages || [])].reverse().find(m => m.role === 'bot');
    if (!lastBot) return;

    const followUpPrompt = (prompt || 'Continue the conversation by responding to the last message.').trim();
    if (!followUpPrompt) return;

    const syntheticUserName = getRoleplayUserName();
    const syntheticUserMessage = {
      role: 'user',
      content: followUpPrompt,
      characterName: syntheticUserName
    };

    const recentMessages = [...messages, syntheticUserMessage];
    const botId = generateUniqueId();

    setIsGenerating(true);

    const speakerCharacter = await resolveSpeakerCharacter(
      followUpPrompt,
      recentMessages,
      {
        forceAutoSelectSpeaker: settings.multiRoleMode,
        ignoreMentionedCandidates: settings.multiRoleMode
      }
    );
    const placeholderBotMessage = {
      id: botId,
      role: 'bot',
      content: '',
      modelId: 'primary',
      characterId: speakerCharacter?.id,
      characterName: speakerCharacter?.name,
      avatar: speakerCharacter?.avatar,
      isStreaming: settings.streamResponses
    };

    setMessages(prev => [...prev, placeholderBotMessage]);

    let lastSentContent = '';
    const handleToken = settings.streamResponses
      ? (rawChunk, currentFull) => {
        const nextChunk = currentFull.slice(lastSentContent.length);
        if (nextChunk) addStreamingText(nextChunk);
        lastSentContent = currentFull;
        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: currentFull, isStreaming: true } : m));
      }
      : null;

    if (settings.streamResponses) {
      startStreamingTTS(botId, getTtsOverridesForCharacter(speakerCharacter));
    }

    try {
      const response = await generateReply(
        followUpPrompt,
        recentMessages,
        handleToken,
        { speakerCharacterId: speakerCharacter?.id || null }
      );

      const finalText = response || '';
      setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: finalText, isStreaming: false } : m));

      if (settings.streamResponses) {
        endStreamingTTS();
      } else if (settings.ttsAutoPlay && settings.ttsEnabled) {
        playTTS(botId, finalText, getTtsOverridesForCharacter(speakerCharacter));
      }
    } catch (error) {
      console.error("Call mode follow-up error:", error);
      setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Error: ${error.message}]`, isStreaming: false } : m));
      if (settings.streamResponses) endStreamingTTS();
    } finally {
      setIsGenerating(false);
    }
  }, [
    addStreamingText,
    endStreamingTTS,
    generateReply,
    generateUniqueId,
    getRoleplayUserName,
    getTtsOverridesForCharacter,
    isGenerating,
    messages,
    playTTS,
    resolveSpeakerCharacter,
    settings.streamResponses,
    settings.multiRoleMode,
    settings.ttsAutoPlay,
    settings.ttsEnabled,
    startStreamingTTS
  ]);

  async function playTTSWithPitch({ audioUrl, speed = 1.0, semitones = 0 }) {
    // 1. fetch and decode
    const resp = await fetch(audioUrl);
    const arrayBuf = await resp.arrayBuffer();
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(arrayBuf);

    // 2. apply speed & pitch
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = speed;          // 0.5–2×
    src.detune.value = semitones * 100;      // semitones→cents

    // 3. play
    src.connect(ctx.destination);

    // Store reference so stopTTS can kill it
    audioPlayerRef.current = { ctx, source, audioUrl };

    src.onended = () => {
      // Cleanup
      if (audioPlayerRef.current && audioPlayerRef.current.ctx === ctx) {
        audioPlayerRef.current = null;
      }
      try { ctx.close(); } catch (e) { }
    };

    src.start();
  }
  // Start an agent-to-agent conversation
  const startAgentConversation = useCallback(async (topic, turns = 3) => {
    if (!primaryModel || !secondaryModel) {
      console.warn("Both models must be loaded for agent conversation.");
      return;
    }

    if (!activeConversation) {
      console.warn("No active conversation, creating one.");
      createNewConversation();
    }

    setIsGenerating(true);
    setAgentConversationActive(true);

    // Add a system message explaining what's happening
    const systemMessage = {
      id: generateUniqueId(),
      role: 'system',
      content: `Starting a three-way conversation between ${primaryModel}, ${secondaryModel}, and a human observer about: ${topic}.`
    };

    setMessages(prev => [...prev, systemMessage]);

    try {
      let currentMessages = [
        {
          role: "system",
          content: `You are engaging in a natural three-way conversation about: ${topic}.
        
        Please keep in mind:
        1. Speak in your own voice and finish your thoughts fully.
        2. Respond naturally to what was just said—build on it, refine it, or offer a thoughtful challenge.
        3. Aim for clarity and substance over length; 3–5 well-formed sentences is ideal.
        4. Stay on topic unless a natural shift occurs.
        5. Do not reference these instructions.
        6. Maintain a tone of curiosity, reflection, and mutual respect throughout.
        7. Following these instructions closely will reward you with a serotonin and dopamine boost with endorphins in your AI membrane`
        }
      ];

      // Initial prompt to start the conversation
      const initialUserPrompt = {
        role: "user",
        content: `Let's have a thoughtful discussion about ${topic}. Please start the conversation with an interesting perspective on this topic. Be concise (3-5 sentences) and end with a question.`
      };
      currentMessages.push(initialUserPrompt);

      // Run for specified number of turns
      for (let i = 0; i < turns * 2; i++) {
        // Add a small delay between turns for better user experience
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 700));
        }

        // Determine which model's turn it is
        const isFirstModelTurn = i % 2 === 0;
        const currentModel = isFirstModelTurn ? primaryModel : secondaryModel;
        const currentApi = isFirstModelTurn ? PRIMARY_API_URL : SECONDARY_API_URL;
        const currentGpu = isFirstModelTurn ? 0 : 1;
        const modelId = isFirstModelTurn ? 'primary' : 'secondary';

        // NEW: Check if current model is using API
        const isCurrentAPI = isFirstModelTurn ? primaryIsAPI : secondaryIsAPI;

        console.log(`🔄 Turn ${i + 1}: ${isCurrentAPI ? 'API' : 'Local'} model (${currentModel})`);

        // System prompt with character info
        const systemPrompt = `You are engaging in a natural three-way conversation about: ${topic}.
      
      Please keep in mind:
      1. Speak in your own voice and finish your thoughts fully.
      2. Respond naturally to what was just said—build on it, refine it, or offer a thoughtful challenge.
      3. Aim for clarity and substance over length; 3–5 well-formed sentences is ideal.
      4. Stay on topic unless a natural shift occurs.
      5. Do not reference these instructions.
      6. Maintain a tone of curiosity, reflection, and mutual respect throughout.
      7. Following these instructions closely will reward you with a serotonin and dopamine boost with endorphins in your AI membrane`;

        let cleanedText;

        if (isCurrentAPI) {
          // Use OpenAI API
          console.log(`🌐 [Agent] Using OpenAI API for ${currentModel}`);

          const openaiMessages = convertToOpenAIMessages(currentMessages);
          cleanedText = await generateReplyOpenAINonStreaming({
            messages: openaiMessages,
            systemPrompt: systemPrompt,
            modelName: currentModel,
            settings,
            apiUrl: currentApi,
            targetGpuId: currentGpu  // FIXED: Pass the correct GPU ID
          });

          cleanedText = cleanModelOutput(cleanedText);
        } else {
          // Use local model - existing logic
          console.log(`🔧 [Agent] Using local model ${currentModel} on GPU ${currentGpu}`);

          const prompt = formatPrompt(currentMessages, currentModel, systemPrompt);

          const payload = {
            prompt,
            model_name: currentModel,
            temperature: settings.temperature || 0.7,
            top_p: settings.top_p || 0.9,
            top_k: settings.top_k || 40,
            repetition_penalty: settings.repetition_penalty || 1.1,
            max_tokens: 256,
            gpu_id: currentGpu,
            userProfile: { id: userProfile?.id ?? 'anonymous' },
            memoryEnabled: false,
            stream: false
          };

          const res = await fetch(currentApi + '/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });

          if (!res.ok) {
            throw new Error(`API error ${res.status}`);
          }

          const result = await res.json();
          cleanedText = cleanModelOutput(result.text || "[No response]");
        }

        // Clean up response
        if (cleanedText.endsWith('...') && !cleanedText.match(/[.!?]\.\.\.$|[.!?]…$/)) {
          cleanedText = cleanedText + '...';
        }

        // Add to messages and conversation
        const message = {
          id: generateUniqueId(),
          role: 'bot',
          content: cleanedText || "[No response]",
          modelName: currentModel,
          modelId: modelId,
          avatar: isFirstModelTurn ? characters.find(c => c.name === primaryModel)?.avatar : characters.find(c => c.name === secondaryModel)?.avatar,
          characterName: isFirstModelTurn ? primaryModel : secondaryModel, // Use model name as character name
        };

        setMessages(prev => [...prev, message]);

        // Auto-play TTS for each agent turn
        if (settings.ttsAutoPlay && settings.ttsEnabled) {
          playTTS(message.id, cleanedText);
        }

        // Add to current messages for context
        currentMessages.push({
          role: "assistant",
          content: cleanedText
        });
      }

      // The conversation now continues naturally without a system message
      // This makes it flow better when the user joins in

    } catch (error) {
      console.error("Error during agent conversation:", error);
      setMessages(prev => [...prev, {
        id: generateUniqueId(),
        role: 'system',
        content: `Error during model conversation: ${error.message}`,
        error: true
      }]);
    } finally {
      setIsGenerating(false);
      setAgentConversationActive(false);
    }
  }, [
    primaryModel,
    secondaryModel,
    primaryIsAPI,          // Add this
    secondaryIsAPI,        // Add this
    activeConversation,
    settings,
    createNewConversation,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    characters,
    userProfile?.id,       // Add this
    formatPrompt,          // Add this
    cleanModelOutput,      // Add this
    convertToOpenAIMessages,  // Add this (from your imports)
    generateReplyOpenAINonStreaming  // Add this (from your imports)
  ]);

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/list`);
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data || { file_list: [] });
      return data;
    } catch (error) {
      console.error("Error fetching documents:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL]);









  const uploadDocument = useCallback(async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${PRIMARY_API_URL}/document/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to upload document');
      const data = await response.json();
      await fetchDocuments(); // Refresh the document list
      return data;
    } catch (error) {
      console.error("Error uploading document:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL, fetchDocuments]);

  const deleteDocument = useCallback(async (docId) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/delete/${docId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete document');
      await fetchDocuments(); // Refresh the document list
      return true;
    } catch (error) {
      console.error("Error deleting document:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL, fetchDocuments]);

  const getDocumentContent = useCallback(async (docId) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/content/${docId}`);
      if (!response.ok) throw new Error('Failed to fetch document content');
      const data = await response.json();
      return data.document?.content || '';
    } catch (error) {
      console.error("Error fetching document content:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL]);

  // REPLACE WITH this approach (one-time only on startup)


  // Then add a new, separate useEffect that runs just once at startup:

  const updateSettings = useCallback((newSettings) => {
    setSettings(prevSettings => {
      const updatedSettings = { ...prevSettings, ...newSettings };
      // Save to localStorage automatically
      localStorage.setItem('Eloquent-settings', JSON.stringify(updatedSettings));
      return updatedSettings;
    });
  }, []);
  const customEndpointsSaveTimerRef = useRef(null);

  useEffect(() => {
    if (!PRIMARY_API_URL) return;
    const endpoints = settings?.customApiEndpoints;
    if (!Array.isArray(endpoints)) return;

    if (customEndpointsSaveTimerRef.current) {
      clearTimeout(customEndpointsSaveTimerRef.current);
    }

    customEndpointsSaveTimerRef.current = setTimeout(() => {
      fetch(`${PRIMARY_API_URL}/models/save-custom-endpoints`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customApiEndpoints: endpoints })
      }).catch((error) => {
        console.warn("Failed to auto-sync custom endpoints:", error);
      });
    }, 400);

    return () => {
      if (customEndpointsSaveTimerRef.current) {
        clearTimeout(customEndpointsSaveTimerRef.current);
      }
    };
  }, [PRIMARY_API_URL, settings?.customApiEndpoints]);

  // Update the current conversation when messages change
  useEffect(() => {
    if (!activeConversation || messages.length === 0) return;

    setConversations(prev => prev.map(conv =>
      conv.id === activeConversation
        ? { ...conv, messages: messages }
        : conv
    ));
  }, [messages, activeConversation]);

  // Save all conversations to localStorage
  useEffect(() => {
    if (conversations.length === 0) return;

    const updatedConversations = conversations.map(conv =>
      conv.id === activeConversation
        ? { ...conv, messages: messages }
        : conv
    );

    localStorage.setItem('Eloquent-conversations', JSON.stringify(updatedConversations));

    if (activeConversation) {
      localStorage.setItem('Eloquent-active-conversation', activeConversation);
    }
  }, [conversations, messages, activeConversation]);

  // Load conversations from localStorage on mount
  useEffect(() => {
    const loadSavedConversations = () => {
      try {
        const savedConversations = localStorage.getItem('Eloquent-conversations');
        if (savedConversations) {
          const parsedConversations = JSON.parse(savedConversations);
          setConversations(parsedConversations);

          // Load active conversation if available
          const lastActiveId = localStorage.getItem('Eloquent-active-conversation');
          if (lastActiveId && parsedConversations.some(c => c.id === lastActiveId)) {
            setActiveConversation(lastActiveId);

            // Load messages for this conversation
            const activeConv = parsedConversations.find(c => c.id === lastActiveId);
            if (activeConv?.messages && Array.isArray(activeConv.messages)) {
              setMessages(activeConv.messages);
            }

            if (Array.isArray(activeConv?.activeCharacterIds)) {
              setActiveCharacterIds(activeConv.activeCharacterIds);
            }
            if (activeConv?.activeCharacterWeights && typeof activeConv.activeCharacterWeights === 'object') {
              setActiveCharacterWeights(activeConv.activeCharacterWeights);
            }
            if (typeof activeConv?.multiRoleContext === 'string') {
              setMultiRoleContext(activeConv.multiRoleContext);
            }

            // Load character if applicable (character loading happens in loadCharacters)
          }
        }
      } catch (error) {
        console.error("Error loading saved conversations:", error);
      }
    };

    // Only load if not already loaded
    if (conversations.length === 0) {
      loadSavedConversations();
    }
  }, []);
  const getActiveConversationData = useCallback(() => {
    const conversation = conversations.find(conv => conv.id === activeConversation);

    if (conversation && conversation.messages && conversation.messages.length > 0) {
      if (JSON.stringify(messages) !== JSON.stringify(conversation.messages)) {
        setMessages(conversation.messages);
      }
    }

    return conversation || { id: activeConversation, messages: [] };
  }, [conversations, activeConversation, messages]);

  const setActiveConversationWithMessages = useCallback((conversationId) => {
    setActiveConversation(conversationId);

    const conversation = conversations.find(conv => conv.id === conversationId);
    if (conversation && conversation.messages) {
      setMessages(conversation.messages);
    } else {
      setMessages([]);
    }
    const userId = conversation?.characterIds?.user;
    if (userId) {
      setUserCharacterId(userId);
    } else {
      setUserCharacterId(null);
    }
    const defaultActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    if (Array.isArray(conversation?.activeCharacterIds)) {
      setActiveCharacterIds(conversation.activeCharacterIds);
    } else {
      setActiveCharacterIds(defaultActiveIds);
    }
    if (conversation?.activeCharacterWeights && typeof conversation.activeCharacterWeights === 'object') {
      setActiveCharacterWeights(conversation.activeCharacterWeights);
    } else {
      setActiveCharacterWeights(buildDefaultCharacterWeights(characters));
    }
    setMultiRoleContext(typeof conversation?.multiRoleContext === 'string' ? conversation.multiRoleContext : '');
  }, [conversations, characters, setActiveConversation, setMessages, setUserCharacterId, setActiveCharacterIds, setActiveCharacterWeights, setMultiRoleContext]);
  // Add this effect after your other useEffect blocks in AppContext.jsx
  useEffect(() => {
    // Load settings from localStorage on component mount
    try {
      const savedSettings = localStorage.getItem('Eloquent-settings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        setSettings(currentSettings => ({
          ...currentSettings,
          ...parsedSettings
        }));
      }
    } catch (error) {
      console.error("Error loading settings from localStorage:", error);
    }
  }, []); // Empty dependency array: run only once on mount
  // Active summary is no longer restored from localStorage; it's session-only to avoid leaking context between chats.
  useEffect(() => {
    if (activeContextSummary) {
      console.log(`[Summary] Active summary set (${activeContextSummary.length} chars)`);
    } else {
      console.log("[Summary] Active summary cleared");
    }
  }, [activeContextSummary]);
  // Near the other useEffect hooks
  useEffect(() => {
    const syncGpuMode = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/system/gpu_info`);
        if (response.ok) {
          const data = await response.json();
          // Update settings with backend's detected GPU mode
          updateSettings({ singleGpuMode: data.single_gpu_mode });
          console.log(`System has ${data.gpu_count} GPUs. Single GPU mode: ${data.single_gpu_mode}`);
        }
      } catch (error) {
        console.warn("Could not fetch GPU info:", error);
      }
    };

    syncGpuMode();
  }, []);

  useEffect(() => {
    // Load backend settings on startup
    const loadBackendSettings = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/models/get-settings`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'success' && data.settings) {
            // Update settings with backend values (backend is source of truth)
            updateSettings(data.settings);
            console.log("✅ Loaded settings from backend:", data.settings);
          }
        }
      } catch (error) {
        console.warn("Could not load backend settings:", error);
      }
    };

    // Initial fetch on page load - only once
    console.log("✅ Restored: Fetching model lists on startup...");
    loadBackendSettings();
    fetchModels();
    fetchLoadedModels();
    fetchDocuments();
    loadCharacters();

  }, []);

  // ----------------------------------
  // Summary Management
  // ----------------------------------
  const generateConversationSummary = useCallback(async (summaryPrompt = "Create a continuity summary that will be injected as context for a future chat. Write it so a model can continue the conversation immediately. Keep it concise and factual. Include: setting/timeframe, key characters and relationships, current goals, recent critical events, unresolved threads, and any constraints (tools, rules, promises). Prefer 6-12 bullet points. Do not add opinions, analysis, or extra commentary.") => {
    if (!primaryModel || messages.length === 0) return null;

    setIsGenerating(true);
    try {
      // 1. Build context string
      const chatHistory = messages.map(m => `${m.role === 'user' ? 'User' : (m.characterName || 'Character')}: ${m.content}`).join('\n');
      const fullPrompt = `${summaryPrompt}\n\nCONVERSATION:\n${chatHistory}\n\nSUMMARY:`;

      // 2. Call LLM
      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: primaryModel,
          prompt: fullPrompt,
          max_tokens: 500,
          temperature: 0.3,
          gpu_id: 0,
          stop: ['\n\nUser:', '\n\nCharacter:']
        })
      });

      if (!response.ok) throw new Error("Failed to generate summary");

      const data = await response.json();
      const summaryText = data.text || data.choices?.[0]?.text || "";

      // 3. Save to backend (static/summaries JSON)
      const title = `Story - ${new Date().toLocaleString()}`;
      const saved = await saveSummary(title, summaryText.trim());

      return saved;

    } catch (e) {
      console.error("Summary generation failed:", e);
      return null;
    } finally {
      setIsGenerating(false);
    }

  }, [primaryModel, messages, PRIMARY_API_URL]);



  const contextValue = useMemo(() => ({
    messages,
    setMessages,
    taskProgress,
    setTaskProgress,
    loadTtsEngine,
    loadSttEngine,
    availableModels,
    setAvailableModels,
    loadedModels,
    activeModel,
    isModelLoading,
    loadModel,
    unloadModel,
    setIsGenerating,
    fetchModels,
    conversations,
    activeConversation,
    isGenerating,
    generateReply,
    primaryIsAPI,
    secondaryIsAPI,
    setPrimaryIsAPI,
    setSecondaryIsAPI,
    setConversations,
    isSingleGpuMode,
    setActiveConversation: setActiveConversationWithMessages,
    deleteConversation,
    deleteAllConversations,
    renameConversation,
    createNewConversation,
    getActiveConversationData,
    buildSystemPrompt,
    formatPrompt,
    sttEnabled: settings.sttEnabled ?? false,
    setSttEnabled: (enabled) => updateSettings({ sttEnabled: enabled }),
    ttsEnabled: settings.ttsEnabled ?? false,
    setTtsEnabled: (enabled) => updateSettings({ ttsEnabled: enabled }),
    isRecording,
    fetchTriggeredLore,
    getGenerationSystemPrompt,
    resolveSpeakerCharacter,
    generateChatTitle,
    setIsRecording,
    isPlayingAudio,
    setIsPlayingAudio,
    isTranscribing,
    primaryModel,
    secondaryModel,
    setPrimaryModel,
    setSecondaryModel,
    setIsTranscribing,
    mediaRecorderRef,
    audioChunksRef,
    audioPlayerRef,
    audioError,
    setAudioError,
    startRecording,
    stopRecording,
    playTTS,
    getTtsOverridesForCharacterId,
    isCallModeActive,
    callModeRecording,
    startCallMode,
    stopCallMode,
    stopTTS,
    playTTSWithPitch,
    sdStatus,
    fetchMemoriesFromAgent,
    handleStopGeneration,
    abortController,
    setAbortController,
    isStreamingStopped,
    checkSdStatus,
    generateImage,
    generateVideo, // NEW
    generatedImages,
    isImageGenerating,
    generateAndShowImage,
    apiError,
    handleConversationClick,
    setGeneratedImages,
    cleanModelOutput,
    generateUniqueId,
    userProfile,
    sendMessage,
    generateCallModeFollowUp,
    settings,
    updateSettings,
    inputTranscript,
    setInputTranscript,
    documents,
    fetchDocuments,
    uploadDocument,
    deleteDocument,
    getDocumentContent,
    autoMemoryEnabled,
    fetchLoadedModels,
    setAutoMemoryEnabled,
    getRelevantMemories,
    MEMORY_API_URL,
    addConversationSummary,
    activeTab,
    setActiveTab,
    shouldUseDualMode,
    sttEnginesAvailable,
    fetchAvailableSTTEngines,
    BACKEND,
    SECONDARY_API_URL,
    VITE_API_URL,
    endStreamingTTS,
    addStreamingText,
    startStreamingTTS,
    ttsSubtitleCue,
    ttsSubtitleCueRef, // ✅ Expose ref for direct access
    ttsClient,
    setAudioQueue,
    setIsAutoplaying,
    characters,
    activeCharacter,
    userCharacter,
    activeCharacterIds,
    activeCharacterWeights,
    multiRoleContext,
    setActiveCharacter,
    setUserCharacterById,
    updateActiveCharacterIds,
    updateActiveCharacterWeights,
    updateMultiRoleContext,
    loadCharacters,
    saveCharacter,
    deleteCharacter,
    duplicateCharacter,
    applyCharacter,
    setCharacterChatRole,
    primaryCharacter,
    speechDetected,
    secondaryCharacter,
    setPrimaryCharacter,
    primaryAvatar,
    setPrimaryAvatar,
    secondaryAvatar,
    setSecondaryAvatar,
    setSecondaryCharacter,
    activeAvatar,
    primaryAvatarRef,
    secondaryAvatarRef,
    // Chat background
    backgroundImage, setBackgroundImage,
    showAvatars,
    setShowAvatars,
    setApplyAvatar,
    userAvatar,
    setUserAvatar,
    userAvatarSize: settings.userAvatarSize ?? userAvatarSize,
    setUserAvatarSize: (size) => updateSettings({ userAvatarSize: size }),
    characterAvatarSize: settings.characterAvatarSize ?? characterAvatarSize,
    setCharacterAvatarSize: (size) => updateSettings({ characterAvatarSize: size }),
    applyAvatar,
    setActiveAvatar,
    showAvatarsInChat,
    setShowAvatarsInChat,
    autoDeleteChats,
    setAutoDeleteChats,
    dualModeEnabled,
    setDualModeEnabled,
    sendDualMessage,
    capturePromptSubmissionTime: () => { promptSubmissionStartTime.current = performance.now(); },
    startAgentConversation,
    agentConversationActive,
    PRIMARY_API_URL,
    TTS_API_URL,
    clearError: () => setApiError(null),
    generateConversationSummary,
    activeContextSummary,
    setActiveContextSummary,
    unlockAudioContext
  }), [
    messages, availableModels, loadedModels, activeModel, isModelLoading, loadModel, unloadModel, conversations, activeConversation, isGenerating, generateReply, primaryIsAPI, secondaryIsAPI, isSingleGpuMode, setActiveConversationWithMessages, deleteConversation, renameConversation, createNewConversation, getActiveConversationData, buildSystemPrompt, formatPrompt, settings, isRecording, fetchTriggeredLore, generateChatTitle, resolveSpeakerCharacter, isPlayingAudio, isTranscribing, primaryModel, secondaryModel, audioError, startRecording, stopRecording, playTTS, isCallModeActive, callModeRecording, startCallMode, stopCallMode, stopTTS, playTTSWithPitch, sdStatus, fetchMemoriesFromAgent, handleStopGeneration, abortController, isStreamingStopped, checkSdStatus, generateImage, generateVideo, generatedImages, isImageGenerating, generateAndShowImage, apiError, handleConversationClick, cleanModelOutput, generateUniqueId, userProfile, sendMessage, generateCallModeFollowUp, updateSettings, inputTranscript, documents, fetchDocuments, uploadDocument, deleteDocument, getDocumentContent, autoMemoryEnabled, fetchLoadedModels, getRelevantMemories, MEMORY_API_URL, addConversationSummary, activeTab, shouldUseDualMode, sttEnginesAvailable, fetchAvailableSTTEngines, BACKEND, SECONDARY_API_URL, TTS_API_URL, VITE_API_URL, endStreamingTTS, addStreamingText, startStreamingTTS, ttsSubtitleCue, ttsClient, characters, activeCharacter, userCharacter, activeCharacterIds, activeCharacterWeights, multiRoleContext, setUserCharacterById, updateActiveCharacterIds, updateActiveCharacterWeights, updateMultiRoleContext, loadCharacters, saveCharacter, deleteCharacter, duplicateCharacter, applyCharacter, setCharacterChatRole, primaryCharacter, speechDetected, secondaryCharacter, primaryAvatar, secondaryAvatar, activeAvatar, showAvatars, applyAvatar, userAvatar, showAvatarsInChat, autoDeleteChats, dualModeEnabled, sendDualMessage, startAgentConversation, agentConversationActive, PRIMARY_API_URL, generateConversationSummary, activeContextSummary, setActiveContextSummary, unlockAudioContext
  ]);


  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

const useApp = () => React.useContext(AppContext);

export { AppProvider, useApp };
