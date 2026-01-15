import React, { memo, useMemo, createContext, useState, useCallback, useContext, useEffect, useRef } from 'react';
import { formatPrompt } from '../utils/chat_templates';
import { cleanModelOutput } from '../utils/cleanOutput';
import { generateChatTitle, fetchTriggeredLore } from '../utils/apiCall';
import { observeConversation, initializeMemories } from '../utils/memoryUtils';
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

    if (tracker.customFields?.length > 0) {
      const custom = tracker.customFields.map(c => c.notes ? `${c.value}: ${c.notes}` : c.value).join(', ');
      sections.push(`[ADDITIONAL DETAILS]: ${custom}`);
    }

    if (sections.length === 0) return '';

    return `\n\n[STORY STATE / TRACKER - Use this for continuity and world knowledge]\n${sections.join('\n\n')}`;
  } catch (e) {
    console.error('Error reading story tracker:', e);
    return '';
  }
};

// Helper function to build system prompt from character data
const _buildSystemPrompt = (character, userProfile = null) => {
  if (!character) return null;

  // Get story tracker context
  const storyContext = getStoryTrackerContext();

  // Tag substitution variables
  const charName = character.name || 'Character';
  const userName = userProfile?.name || userProfile?.username || 'User';

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
      ).join('\n')}` : ''}${storyContext}`;
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
  const buildSystemPrompt = useCallback((char) => _buildSystemPrompt(char, userProfile), [userProfile]);
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
  const callModeSilenceTimerRef = useRef(null);
  const callModeStreamRef = useRef(null);
  const audioChunksRef = useRef([]); // Updated to match previous edit
  const audioPlayerRef = useRef(null); // Ref to store the current Audio object for TTS
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
  const activeCharacterRef = useRef(null);
  useEffect(() => {
    activeCharacterRef.current = activeCharacter;
  }, [activeCharacter]);
  const [backgroundImage, setBackgroundImage] = useState(null); // New state for chat background

  // Refs for avatar canvases
  const primaryAvatarRef = useRef(null);
  const secondaryAvatarRef = useRef(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
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
    sttEnabled: false,
    streamResponses: true,
    sttEngine: "whisper",
    ttsVoice: 'af_heart',
    ttsSpeed: 1.0,
    ttsPitch: 0,
    ttsAutoPlay: false,  // Simply set a default value
    userAvatarSize: null, // Default size for user avatar
    characterAvatarSize: null, // Default size for character avatar
  });

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
    // We still use the audioPlayerRef for one-off TTS playback, so we keep this logic.
    // The new useEffect handles the streaming queue playback.
    if (audioPlayerRef.current) {
      const ref = audioPlayerRef.current;
      if (ref instanceof Audio || typeof ref.pause === 'function') {
        ref.pause();
        ref.currentTime = 0;
        if (ref.src && ref.src.startsWith('blob:')) {
          URL.revokeObjectURL(ref.src);
        }
      } else if (ref.source && ref.ctx) {
        try {
          ref.source.stop();
          ref.ctx.close();
        } catch (e) { /* Ignore errors */ }
        if (ref.audioUrl && ref.audioUrl.startsWith('blob:')) {
          URL.revokeObjectURL(ref.audioUrl);
        }
      }
      audioPlayerRef.current = null;
    }

    // If we manually stop, we MUST also kill the streaming queue and state.
    console.log('🛑 Manually stopping TTS. Signaling backend interrupt.');
    isTtsInterruptedRef.current = true;
    ttsClient.interrupt();
    ttsClient.clearPending();
    if (ttsClient.socket) ttsClient.socket.close(); // Soft close: triggers auto-reconnect if enabled

    setAudioQueue([]);      // Clear any buffered audio
    setIsAutoplaying(false); // Stop the autoplay loop
    setIsPlayingAudio(null);
    window.streamingAudioPlaying = false;
    ttsClient.audioQueue.length = 0; // Clear the queue
    streamingTtsMessageIdRef.current = null; // KILL the current stream identification
  }, [ttsClient, setAudioQueue, setIsAutoplaying]);

  const handleStopGeneration = useCallback(() => {
    // 1. Stop the text generation stream
    if (abortController) {
      console.log('🛑 Aborting text generation...');
      abortController.abort();
      setAbortController(null);
    }

    // 2. Stop TTS explicitly (will also soft-close socket and trigger auto-reconnect)
    stopTTS();

    // Reset the main "isGenerating" flag for the UI
    setIsGenerating(false);

    // This flag is for UI feedback, you can keep it
    setIsStreamingStopped(true);
    setTimeout(() => setIsStreamingStopped(false), 1000);

  }, [abortController, stopTTS, setIsGenerating]);
  const startStreamingTTS = useCallback((messageId) => {
    window.streamingAudioPlaying = false;
    ttsClient.audioQueue = [];

    if (!settings.ttsAutoPlay || !settings.ttsEnabled) return;

    isFirstTextChunk.current = true;
    isTtsInterruptedRef.current = false;
    streamingTtsMessageIdRef.current = messageId;
    setIsAutoplaying(true);

    const streamingTtsSettings = {
      engine: settings.ttsEngine || 'kokoro',
      voice: settings.ttsVoice || 'af_heart',
      exaggeration: settings.ttsExaggeration || 0.5,
      cfg: settings.ttsCfg || 0.5,
      audio_prompt_path: (settings.ttsEngine === 'chatterbox' && settings.ttsVoice !== 'default')
        ? settings.ttsVoice
        : null
    };

    ttsClient.send(streamingTtsSettings);

    ttsClient.onAudioQueueUpdate = async () => {
      setAudioQueue([...ttsClient.audioQueue]);

      if (ttsClient.audioQueue.length > 0 && !window.streamingAudioPlaying) {
        window.streamingAudioPlaying = true;
        setIsPlayingAudio(messageId);

        try {
          while (ttsClient.audioQueue.length > 0) {
            // Check for manual interruption before playing each chunk
            if (isTtsInterruptedRef.current) break;

            const arrayBuffer = ttsClient.audioQueue.shift();
            const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audioPlayerRef.current = audio;
            audio.playbackRate = settings.ttsSpeed || 1.0;

            await new Promise((resolve, reject) => {
              audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                if (audioPlayerRef.current === audio) audioPlayerRef.current = null;
                resolve();
              };
              audio.onerror = (e) => {
                URL.revokeObjectURL(audioUrl);
                if (audioPlayerRef.current === audio) audioPlayerRef.current = null;
                reject(e);
              };
              audio.play().catch(reject);
            });
          }
        } catch (err) {
          console.error("🎵 [TTS Playback Error]", err);
        } finally {
          window.streamingAudioPlaying = false;
          setIsPlayingAudio(null);
        }
      }
    };
  }, [settings, ttsClient, setAudioQueue, setIsAutoplaying]);
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
    // and if autoplay is still enabled for this session
    if (streamingTtsMessageIdRef.current && newTextChunk && isAutoplaying) {
      ttsClient.send(newTextChunk);
    }
  }, [ttsClient, isAutoplaying]);

  const endStreamingTTS = useCallback(() => {
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
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          guidance_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          seed: opts.seed || -1,
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
      setAudioError(`Microphone access denied or error: ${err.message}`);
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
  }, [isRecording, stopRecording]);


  // --- playTTS (Modified to prevent auto-replay) ---
  const playTTS = useCallback(async (messageId, text, optionsOverrides = null) => {
    console.log(`🗣️ [TTS] Attempting to play message ${messageId}: "${text.substring(0, 40)}..."`);

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
      if (currentTtsEngine === 'chatterbox') {
        ttsOptions.exaggeration = optionsOverrides?.ttsExaggeration ?? settings.ttsExaggeration ?? 0.5;
        ttsOptions.cfg = optionsOverrides?.ttsCfg ?? settings.ttsCfg ?? 0.5;

        console.log(`🔊 [TTS] Chatterbox params - exaggeration: ${ttsOptions.exaggeration}, cfg: ${ttsOptions.cfg}`);
      }

      console.log(`🔊 [TTS] Using engine: ${ttsOptions.engine} with options:`, ttsOptions);

      // Pass options object to synthesizeSpeech
      audioUrl = await synthesizeSpeech(text, ttsOptions);

      // UNCHANGED: Everything after this stays exactly the same
      if (!audioUrl) throw new Error("SynthesizeSpeech returned an invalid URL.");
      console.log(`🗣️ [TTS] Received audio URL for message ${messageId}: ${audioUrl.substring(0, 50)}...`);

      // 2. Playback branch: HTML5 Audio if no pitch shift, otherwise Web Audio API
      if (currentTtsPitch === 0) {
        // --- HTML5 Audio path (only speed) ---
        const audio = new Audio(audioUrl);
        audio.playbackRate = currentTtsSpeed;
        audioPlayerRef.current = audio;

        const handleEnd = () => {
          console.log(`🗣️ [TTS] Playback ended for message ${messageId}`);
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
        console.log(`🗣️ [TTS] Started HTML5 Audio playback for message ${messageId}`);
      } else {
        // --- Web Audio API path (speed + pitch) ---
        const resp = await fetch(audioUrl);
        const arrayBuf = await resp.arrayBuffer();
        const ctx = new AudioContext();
        const audioBuffer = await ctx.decodeAudioData(arrayBuf);

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.playbackRate.value = currentTtsSpeed;
        source.detune.value = currentTtsPitch * 100; // semitones → cents
        source.connect(ctx.destination);

        // store so stopTTS can reach it if needed
        audioPlayerRef.current = { ctx, source, audioUrl };

        source.onended = () => {
          console.log(`🗣️ [TTS] Web Audio playback ended for message ${messageId}`);
          setIsPlayingAudio(null);
          URL.revokeObjectURL(audioUrl);
          audioPlayerRef.current = null;
          ctx.close();
          // Store this message ID as the last one played
          lastPlayedMessageRef.current = messageId;
        };

        source.start();
        console.log(`🗣️ [TTS] Started Web Audio playback for message ${messageId}`);
      }
    } catch (error) {
      console.error(`🗣️ [TTS] Error in playTTS for message ${messageId}:`, error);
      setAudioError(error.message || "Failed to synthesize or play audio");
      setIsPlayingAudio(null);
      if (audioUrl) {
        try { URL.revokeObjectURL(audioUrl); }
        catch (e) { console.error("🗣️ [TTS] Error revoking URL:", e); }
      }
      audioPlayerRef.current = null;
      lastPlayedMessageRef.current = null; // Reset on error
    }
  }, [isPlayingAudio, stopTTS, settings.ttsSpeed, settings.ttsPitch, settings.ttsVoice, messages]);








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
    const userName = userProfile?.name || userProfile?.username || 'User';
    const firstMessage = primaryCharacter?.first_message
      ? replaceTags(primaryCharacter.first_message, charName, userName)
      : null;

    const initial = firstMessage
      ? [{ id: generateUniqueId(), role: 'bot', content: firstMessage, modelId: 'primary', characterName: charName, avatar: primaryCharacter?.avatar }]
      : [];

    const conv = { id, name: 'New Chat', messages: initial, characterIds: { primary: primaryCharacter?.id, secondary: secondaryCharacter?.id }, created: new Date().toISOString(), requiresTitle: true };
    console.log('🔍 [DEBUG] New conversation has requiresTitle flag:', conv.requiresTitle);
    setConversations(prev => [...prev, conv]);
    setActiveConversation(id);
    setMessages(initial);
    setDualModeEnabled(false);
    setActiveCharacter(primaryCharacter || null);
    generateChatTitle(id, initial, primaryCharacter?.name || '', secondaryCharacter?.name || '')
    return conv;
  }, [primaryCharacter, secondaryCharacter, setConversations, setActiveConversation, setMessages, setDualModeEnabled, setActiveCharacter, generateChatTitle]);

  const handleConversationClick = useCallback((id) => {
    if (activeConversation) {
      setConversations(prev => prev.map(c => c.id === activeConversation ? { ...c, messages } : c));
    }
    setActiveConversation(id);
    const sel = conversations.find(c => c.id === id) || {};
    setMessages(Array.isArray(sel.messages) ? sel.messages : []);

    const { primary, secondary } = sel.characterIds || {};
    const primChar = characters.find(c => c.id === primary) || null;
    const secChar = characters.find(c => c.id === secondary) || null;
    setPrimaryCharacter(primChar);
    setSecondaryCharacter(secChar);
    setActiveCharacter(primChar);
  }, [activeConversation, conversations, messages, characters, setConversations, setActiveConversation, setMessages, setPrimaryCharacter, setSecondaryCharacter, setActiveCharacter]);

  // ----------------------------------
  // Character Management
  // ----------------------------------
  const loadCharacters = useCallback(async () => {
    try {
      const saved = localStorage.getItem('llm-characters');
      if (saved) {
        const parsed = JSON.parse(saved);
        setCharacters(parsed);
        const conv = conversations.find(c => c.id === activeConversation);
        const charId = conv?.characterIds?.primary;
        if (charId) setActiveCharacter(parsed.find(c => c.id === charId) || null);
      }
    } catch (err) {
      console.error('Load chars error:', err);
    }
  }, [activeConversation, conversations]);

  const saveCharacter = useCallback((data) => {
    let savedCharacter = data;

    setCharacters(prev => {
      const list = prev.slice();

      if (!data.id) {
        // Create new character
        savedCharacter = {
          ...data,
          id: `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          created_at: new Date().toISOString().split('T')[0]
        };
        list.push(savedCharacter);
        console.log('Creating new character:', savedCharacter.name, 'with ID:', savedCharacter.id);
      } else {
        // Update existing character
        const idx = list.findIndex(c => c.id === data.id);
        if (idx > -1) {
          savedCharacter = { ...data };
          list[idx] = savedCharacter;
          console.log('Updating existing character:', savedCharacter.name, 'with ID:', savedCharacter.id);
        } else {
          // ID provided but character not found, treat as new
          savedCharacter = {
            ...data,
            id: `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
            created_at: new Date().toISOString().split('T')[0]
          };
          list.push(savedCharacter);
          console.log('Character not found, creating new:', savedCharacter.name, 'with ID:', savedCharacter.id);
        }
      }

      try {
        localStorage.setItem('llm-characters', JSON.stringify(list));
        console.log('Characters saved to localStorage:', list.length, 'total characters');
      } catch (error) {
        console.error('Failed to save characters to localStorage:', error);
      }

      return list;
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
  }, [activeCharacter]);

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

  const applyCharacter = useCallback((id) => {
    const char = characters.find(c => c.id === id) || null;
    setActiveCharacter(char);
    if (activeConversation) {
      setConversations(prev => prev.map(c => (c.id === activeConversation ? { ...c, characterIds: { ...c.characterIds, primary: id } } : c)));
    }
    if (char?.first_message && messages.length === 0) {
      setMessages([{ id: generateUniqueId(), role: 'bot', content: char.first_message, avatar: char.avatar, characterName: char.name }]);
    }
  }, [characters, activeConversation, messages, setConversations]);

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
  const sendDualMessage = useCallback(async (text, webSearchEnabled = false) => {
    if (!activeConversation || !primaryModel || !secondaryModel) return;
    console.log("📩 [DUAL] Processing message:", text.substring(0, 30), "…", webSearchEnabled ? "(with web search)" : "");

    // Note: For dual mode, you might want to decide if web search should affect both models
    // or just one. For now, I'll add the flag to both payloads but you can adjust as needed.

    const userMsg = { id: generateUniqueId(), role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);

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
            authorNote: settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : undefined,
            use_web_search: webSearchEnabled // NEW: Add to primary
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
            authorNote: settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : undefined,
            use_web_search: false // Usually only primary should do web search in dual mode
          })
        }).then(r => r.json())
      ]);

      const pId = generateUniqueId();
      const sId = generateUniqueId();
      const pText = cleanModelOutput(pRes.text);
      const sText = cleanModelOutput(sRes.text);

      setMessages(prev => [...prev,
      { id: pId, role: 'bot', content: pText, modelId: 'primary', characterName: primaryCharacter?.name, avatar: primaryCharacter?.avatar },
      { id: sId, role: 'bot', content: sText, modelId: 'secondary', characterName: secondaryCharacter?.name, avatar: secondaryCharacter?.avatar }
      ]);

      await observeConversationWithAgent(text, `${pRes.text}\n\n${sRes.text}`);

      // Auto-play TTS for BOTH messages if enabled
      if (settings.ttsAutoPlay && settings.ttsEnabled) {
        // We'll play them sequentially? Or just primary? 
        // For now let's play primary, then secondary might overlap or we'd need a queue.
        // Actually playTTS stops current audio, so primary then secondary would just play secondary.
        // Let's just play primary for now to avoid complexity, or the characters will talk over each other.
        playTTS(pId, pText);
      }
    } catch (err) {
      console.error('Dual chat error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [activeConversation, primaryModel, secondaryModel, messages, primaryCharacter, secondaryCharacter, PRIMARY_API_URL, SECONDARY_API_URL, userProfile, settings, observeConversationWithAgent]);
  // New shared helper to build the FULL generation system prompt (memories, lore, etc.)
  const getGenerationSystemPrompt = useCallback(async (text, character, authorNote = null) => {
    console.log('🔍 [SystemPrompt] Building for character:', character?.name, 'User:', userProfile?.name);
    console.log('🔍 [SystemPrompt] Tag replacer check. Character valid?', !!character, 'UserProfile valid?', !!userProfile);
    let systemMsg = character
      ? buildSystemPrompt(character)
      : ('You are a helpful assistant.' + getStoryTrackerContext());

    let contextToAdd = '';

    // Helper for tag replacement in lore/profile
    const replaceTags = (content) => {
      if (!content || !character) return content;
      const charName = character.name || 'Character';
      const userName = userProfile?.name || userProfile?.username || 'User';
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
        contextToAdd += `\n\nWORLD KNOWLEDGE:\n${loreBlock}`;
      }
    }

    systemMsg += contextToAdd;

    const effectiveAuthorNote = authorNote || (settings.authorNote && settings.authorNote.trim()) || null;
    if (effectiveAuthorNote) {
      systemMsg += `\n\n[AUTHOR'S NOTE - Writing style guidance for this response]\n${effectiveAuthorNote}`;
    }

    return systemMsg;
  }, [settings, userProfile, MEMORY_API_URL, fetchMemoriesFromAgent, fetchTriggeredLore]);

  // In AppContext.jsx, replace the entire generateReply function
  const generateReply = useCallback(async (text, recentMessages, onToken = null, options = {}) => {
    const { authorNote = null, webSearchEnabled = false } = options;
    const systemMsg = await getGenerationSystemPrompt(text, activeCharacterRef.current, authorNote);

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
      authorNote: authorNote || (settings.authorNote && settings.authorNote.trim()) || undefined,
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
    activeCharacter,
    userProfile?.id,
    PRIMARY_API_URL,
    getGenerationSystemPrompt,
    formatPrompt,
    cleanModelOutput
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
      memoryContext += (memoryContext ? "\n\nWORLD KNOWLEDGE:\n" : "WORLD KNOWLEDGE:\n") + loreContext;
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
      const systemMsg = await getGenerationSystemPrompt(text, activeCharacterRef.current, authorNote);

      // 4) Prepare payload
      const {
        temperature, top_p, top_k, repetition_penalty, frequencyPenalty = 0, presencePenalty = 0,
        antiRepetitionMode = false, use_rag, selectedDocuments = [], streamResponses
      } = settings;

      const effectiveAuthorNote = authorNote || (settings.authorNote && settings.authorNote.trim()) || undefined;

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
          id: botId, role: 'bot', content: '', modelId: 'primary',
          characterName: activeCharacter?.name, avatar: activeCharacter?.avatar, isStreaming: streamResponses,
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

        if (streamResponses) startStreamingTTS(botId);

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
              playTTS(botId, cleanedText);
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
    activeConversation, primaryModel, messages, conversations, settings, activeCharacter,
    userProfile?.id, PRIMARY_API_URL, fetchMemoriesFromAgent, fetchTriggeredLore, MEMORY_API_URL,
    buildSystemPrompt, formatPrompt, cleanModelOutput, generateChatTitle, memoryContext,
    observeConversationWithAgent, generateReply, primaryIsAPI, createNewConversation, getStoryTrackerContext
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
  }, [conversations, setActiveConversation, setMessages]);
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
    ttsClient,
    setAudioQueue,
    setIsAutoplaying,
    characters,
    activeCharacter,
    setActiveCharacter,
    loadCharacters,
    saveCharacter,
    deleteCharacter,
    duplicateCharacter,
    applyCharacter,
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
    startAgentConversation,
    agentConversationActive,
    PRIMARY_API_URL,
    TTS_API_URL,
    clearError: () => setApiError(null)
  }), [
    messages, availableModels, loadedModels, activeModel, isModelLoading, loadModel, unloadModel, conversations, activeConversation, isGenerating, generateReply, primaryIsAPI, secondaryIsAPI, isSingleGpuMode, setActiveConversationWithMessages, deleteConversation, renameConversation, createNewConversation, getActiveConversationData, buildSystemPrompt, formatPrompt, settings, isRecording, fetchTriggeredLore, generateChatTitle, isPlayingAudio, isTranscribing, primaryModel, secondaryModel, audioError, startRecording, stopRecording, playTTS, isCallModeActive, callModeRecording, startCallMode, stopCallMode, stopTTS, playTTSWithPitch, sdStatus, fetchMemoriesFromAgent, handleStopGeneration, abortController, isStreamingStopped, checkSdStatus, generateImage, generatedImages, isImageGenerating, generateAndShowImage, apiError, handleConversationClick, cleanModelOutput, generateUniqueId, userProfile, sendMessage, updateSettings, inputTranscript, documents, fetchDocuments, uploadDocument, deleteDocument, getDocumentContent, autoMemoryEnabled, fetchLoadedModels, getRelevantMemories, MEMORY_API_URL, addConversationSummary, activeTab, shouldUseDualMode, sttEnginesAvailable, fetchAvailableSTTEngines, BACKEND, SECONDARY_API_URL, TTS_API_URL, VITE_API_URL, endStreamingTTS, addStreamingText, startStreamingTTS, ttsClient, characters, activeCharacter, loadCharacters, saveCharacter, deleteCharacter, duplicateCharacter, applyCharacter, primaryCharacter, speechDetected, secondaryCharacter, primaryAvatar, secondaryAvatar, activeAvatar, showAvatars, applyAvatar, userAvatar, showAvatarsInChat, autoDeleteChats, dualModeEnabled, sendDualMessage, startAgentConversation, agentConversationActive, PRIMARY_API_URL
  ]);

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

const useApp = () => React.useContext(AppContext); // Remove 'export' here
export { AppProvider, useApp }; // <-- This should be at the end
