import React, { memo, useMemo, createContext, useState, useCallback, useContext, useEffect, useRef } from 'react';
import { formatPrompt } from '../utils/chat_templates';
import { cleanModelOutput } from '../utils/cleanOutput';
import { generateChatTitle, fetchTriggeredLore } from '../utils/apiCall';
import { observeConversation, initializeMemories } from '../utils/memoryUtils';
import { useMemory } from '../contexts/MemoryContext';
import { transcribeAudio, synthesizeSpeech } from '../utils/apiCall';
import { generateReplyOpenAI, processOpenAIStream, generateReplyOpenAINonStreaming, convertToOpenAIMessages } from '../utils/openaiApi';
import { ttsClient } from '../utils/apiCall';


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
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = promptText;
    document.body.appendChild(img);

  } catch (err) {
    console.error("📝 [ERROR] generateAndShowImage failed:", err);
  }
}

// Example usage:
//generateAndShowImage("a sexy woman");
// Helper function to generate truly unique IDs
const generateUniqueId = () => `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

// Helper function to build system prompt from character data
const buildSystemPrompt = (character) => {
  if (!character) return null;

  return `You are ${character.name}, ${character.description}.

PERSONALITY: ${character.personality}

BACKGROUND: ${character.background || ''}

${character.scenario ? `SCENARIO: ${character.scenario}` : ''}

SPEAKING STYLE: ${character.speech_style || ''}

IMPORTANT: Stay in character at all times. Respond as ${character.name} would, maintaining the defined personality and speech patterns.

${character.example_dialogue && character.example_dialogue.length > 0
    ? `EXAMPLE DIALOGUE:
${character.example_dialogue.map(msg =>
      `${msg.role === 'character' ? character.name : 'User'}: ${msg.content}`
    ).join('\n')}` : ''}`;
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
  const [primaryCharacter,   setPrimaryCharacter]   = useState(null);
  const [secondaryCharacter, setSecondaryCharacter] = useState(null);
  const [primaryAvatar,     setPrimaryAvatar]     = useState(null);
  const [secondaryAvatar,   setSecondaryAvatar]   = useState(null);
  const mediaRecorderRef = useRef(null);
  const isFirstTextChunk = useRef(false);
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

  // Refs for avatar canvases
  const primaryAvatarRef = useRef(null);
  const secondaryAvatarRef = useRef(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [messages, setMessages] = useState([]);
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
  // API URLs for primary and secondary models
  const PRIMARY_API_URL = "http://localhost:8000";
// If single GPU mode, everything goes to port 8000. Otherwise, secondary/memory go to 8001
  const SECONDARY_API_URL = isSingleGpuMode ? "http://localhost:8000" : "http://localhost:8001";
  const MEMORY_API_URL = isSingleGpuMode ? "http://localhost:8000" : "http://localhost:8001";
  const BACKEND = import.meta.env.VITE_API_URL || (isSingleGpuMode ? "http://127.0.0.1:8000" : "http://127.0.0.1:8001");
  const VITE_API_URL = isSingleGpuMode ? "http://127.0.0.1:8000" : "http://127.0.0.1:8001";


  // One-time memory initialization on app load
  useEffect(() => {
    if (memoryContext) {
      initializeMemories(memoryContext)
        .then(success => {
          console.log("🧠 Memory system initialized");    // always log
          if (success) {
            console.log("🧠 Memory system is ready to use.");
          }
        })
      .catch(err => {
        console.error("🧠 Memory system failed to initialize:", err);
        });
    }
  }, [memoryContext]);



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
const handleStopGeneration = useCallback(() => {
  // 1. Stop the text generation stream
  if (abortController) {
    console.log('🛑 Aborting text generation...');
    abortController.abort();
    setAbortController(null);
  }
  
  // 3. Clear any audio that's already buffered on the frontend
  console.log('🛑 Clearing frontend audio queue...');
  setAudioQueue([]);
  
  // 4. Reset the autoplaying state to prevent the queue from starting again
  console.log('🛑 Resetting autoplay state...');
  setIsAutoplaying(false);

  // Reset the main "isGenerating" flag for the UI
  setIsGenerating(false);
  
  // This flag is for UI feedback, you can keep it
  setIsStreamingStopped(true);
  setTimeout(() => setIsStreamingStopped(false), 1000);

}, [abortController, ttsClient, setAudioQueue, setIsAutoplaying, setIsGenerating]);
const startStreamingTTS = useCallback((messageId) => {
    window.streamingAudioPlaying = false; 
    ttsClient.audioQueue = [];   
  if (!settings.ttsAutoPlay || !settings.ttsEnabled) return;

  // TIMING: Record when we start the TTS process
  isFirstTextChunk.current = true;
  console.log(`⏱️ [TTS Timing] Initializing TTS for message ${messageId}`);
  streamingTtsMessageIdRef.current = messageId;
  setIsAutoplaying(true);
  
  // RESET the callback for each new message
  ttsClient.onAudioQueueUpdate = async () => {
    console.log(`🎵 [Audio Queue] Queue updated, length: ${ttsClient.audioQueue.length}`);
    setAudioQueue([...ttsClient.audioQueue]);
    
    // Simple check: if queue has chunks and we're not already playing, start playing
    if (ttsClient.audioQueue.length > 0 && !window.streamingAudioPlaying) {
      window.streamingAudioPlaying = true;
      
      // TIMING: Record when speech actually starts and calculate latency
speechStartTime.current = performance.now();
const ttsLatency = speechStartTime.current - textGenerationStartTime.current;
const fullCycleLatency = speechStartTime.current - promptSubmissionStartTime.current;

console.log(`🚀 [TTS Timing] Speech started! TTS latency: ${ttsLatency.toFixed(0)}ms`);
console.log(`🚀 [Full Cycle] Complete latency (prompt → speech): ${fullCycleLatency.toFixed(0)}ms`);
      
      setIsPlayingAudio(messageId);

      while (ttsClient.audioQueue.length > 0) {
        const arrayBuffer = ttsClient.audioQueue.shift();
        console.log(`🎵 [Playing] Chunk, ${ttsClient.audioQueue.length} remaining`);
        
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        
        audio.playbackRate = settings.ttsSpeed || 1.0;
        
        await new Promise(resolve => {
          audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            resolve();
          };
          audio.play();
        });
      }
      
      window.streamingAudioPlaying = false;
      setIsPlayingAudio(null);
      console.log(`🎵 [Done] All chunks played for message ${messageId}`);
      endStreamingTTS();
    }
  };
  
  // Connect if needed
  if (!ttsClient.socket || ttsClient.socket.readyState !== 1) {
    ttsClient.connect(
      () => console.log("✅ [TTS Stream] WebSocket connected."),
      () => console.log("🛑 [TTS Stream] WebSocket disconnected."),
      (error) => console.error("❌ [TTS Stream] WebSocket error:", error)
    );
  }
}, [settings.ttsAutoPlay, settings.ttsEnabled, settings.ttsSpeed, ttsClient, setAudioQueue, setIsAutoplaying]);
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
    if (streamingTtsMessageIdRef.current && newTextChunk) {
        ttsClient.send(newTextChunk);
    }
}, [ttsClient]);

const endStreamingTTS = useCallback(() => {
    if (streamingTtsMessageIdRef.current) {
        console.log(`⏹️ [TTS Stream] Ending for message ${streamingTtsMessageIdRef.current}`);
        
        // This is correct. It tells the backend the stream of text is over.
        ttsClient.closeStream(); 
        
        // REMOVE OR COMMENT OUT THIS LINE:
        // ttsClient.disconnect(); 
        
        streamingTtsMessageIdRef.current = null;
    }
}, [ttsClient]);


// …later, inside your component:
const checkSdStatus = useCallback(async () => {

  const imageEngine = settings.imageEngine || 'auto1111';
  
  let automatic1111Status = false;
  let localSdStatus = false;
  
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
  
  setSdStatus({
    automatic1111: automatic1111Status,
    localSd: localSdStatus,
    models: [] // We'll populate this later if needed
  });
}, [MEMORY_API_URL, settings.imageEngine]);

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

// 2. Generate image via POST /sd/txt2img
// Replace the entire function with this updated version:
const generateImage = useCallback(async (prompt, opts) => {
  console.log("Starting image generation with prompt:", prompt);
  console.log("Image generation options:", opts);

  // Define imageEngine at the top so it's available for all logic
  const imageEngine = settings.imageEngine || 'auto1111';

  // Nest the mapSampler function here so it has access to imageEngine
  const mapSampler = (sampler) => {
    if (imageEngine === 'EloDiffusion') {
      const samplerMap = {
        'Euler a': 'euler_a',
        'Euler': 'euler',
        'Heun': 'heun',
        'DPM2': 'dpm2',
        'DPM++ 2S a': 'dpmpp2s_a',
        'DPM++ 2M': 'dpmpp2m',
        'DDIM': 'ddim_trailing'
      };
      return samplerMap[sampler] || 'euler_a';
    }
    return sampler; // Keep original for A1111
  };

  setIsImageGenerating(true);
  clearError();
  try {
    const payload = {
      prompt,
      negative_prompt: opts.negative_prompt || "",
      width: opts.width || 512,
      height: opts.height || 512,
      steps: opts.steps || 20,
      guidance_scale: opts.guidance_scale || 7.0,
      sampler: mapSampler(opts.sampler || "Euler a"), // This now works correctly
      seed: opts.seed || -1,
      ...(imageEngine !== 'EloDiffusion' && {
        model: opts.model,
        cfg_scale: opts.guidance_scale || 7.0,
        sampler_name: opts.sampler || "Euler a"
      })
    };

    const endpoint = imageEngine === 'EloDiffusion'
      ? `${MEMORY_API_URL}/sd-local/txt2img`
      : `${MEMORY_API_URL}/sd/txt2img`;

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
  // Add settings.imageEngine to the dependency array
}, [clearError, setApiError, MEMORY_API_URL, settings.imageEngine]);



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
    await stopRecording(() => {});
  }
}, [isRecording, stopRecording]);
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
  
  // --- New, Crucial Logic ---
  // If we manually stop, we MUST also kill the streaming queue and state.
  console.log('🛑 Manually stopping TTS. Clearing queue and resetting autoplay.');
  ttsClient.disconnect(); // Disconnect the streaming source
  setAudioQueue([]);      // Clear any buffered audio
  setIsAutoplaying(false); // Stop the autoplay loop
  setIsPlayingAudio(null);
  setIsAutoplaying(false);
  window.streamingAudioPlaying = false;
  ttsClient.audioQueue.length = 0; // Clear the queue
  // Clear the ID of the message being played

}, [ttsClient, setAudioQueue, setIsAutoplaying]); // Added dependencies


// --- playTTS (Modified to prevent auto-replay) ---
const playTTS = useCallback(async (messageId, text) => {
  console.log(`🗣️ [TTS] Attempting to play message ${messageId}: "${text.substring(0, 40)}..."`);
  
  // Check if this is the same message that was just played
  if (lastPlayedMessageRef.current === messageId) {
    console.log(`🗣️ [TTS] Skipping message ${messageId} as it was just played.`);
    return;
  }
  // ADD THIS CODE: Check if the message is still streaming
  const message = messages.find(m => m.id === messageId);
  if (message && message.isStreaming) {
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
  
  // NEW: Build options object for both engines
const ttsOptions = {
  voice: settings.ttsVoice || (settings.ttsEngine === 'kyutai' ? 'default' : 'af_heart'),
  engine: settings.ttsEngine || 'kokoro'
};

  // NEW: Add Chatterbox-specific parameters if using Chatterbox
  if (settings.ttsEngine === 'chatterbox') {
    ttsOptions.exaggeration = settings.ttsExaggeration || 0.5;
    ttsOptions.cfg = settings.ttsCfg || 0.5;
    
    // If using a custom voice, add the file path
    if (settings.ttsVoice && settings.ttsVoice !== 'default' && settings.ttsVoice.startsWith('voice_ref_')) {
      ttsOptions.audio_prompt_path = `/static/voice_references/${settings.ttsVoice}`;
    }
  }

  // UPDATED: Pass options object instead of just voice
  audioUrl = await synthesizeSpeech(text, ttsOptions);
  
  // UNCHANGED: Everything after this stays exactly the same
  if (!audioUrl) throw new Error("SynthesizeSpeech returned an invalid URL.");
  console.log(`🗣️ [TTS] Received audio URL for message ${messageId}: ${audioUrl.substring(0, 50)}...`);
    // 2. Playback branch: HTML5 Audio if no pitch shift, otherwise Web Audio API
    if (settings.ttsPitch === 0) {
      // --- HTML5 Audio path (only speed) ---
      const audio = new Audio(audioUrl);
      audio.playbackRate = settings.ttsSpeed;
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
      source.playbackRate.value = settings.ttsSpeed;
      source.detune.value = settings.ttsPitch * 100; // semitones → cents
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
  const initial = primaryCharacter?.first_message
    ? [{ id: generateUniqueId(), role: 'bot', content: primaryCharacter.first_message, modelId: 'primary', characterName: primaryCharacter.name, avatar: primaryCharacter.avatar }]
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
  setCharacters(prev => {
    const list = prev.slice();
    if (!data.id) {
      data.id = `char_${Date.now()}`;
      data.created_at = new Date().toISOString().split('T')[0];
      list.push(data);
    } else {
      const idx = list.findIndex(c => c.id === data.id);
      if (idx > -1) list[idx] = data;
    }
    localStorage.setItem('llm-characters', JSON.stringify(list));
    return list;
  });
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
  if (!orig) return;
  saveCharacter({ ...orig, id: `char_${Date.now()}`, name: `${orig.name} (Copy)` });
}, [characters, saveCharacter]);

const applyCharacter = useCallback((id) => {
  const char = characters.find(c => c.id === id) || null;
  setActiveCharacter(char);
  if (activeConversation) {
    setConversations(prev => prev.map(c => (c.id === activeConversation ? { ...c, characterIds: { ...c.characterIds, primary: id }} : c)));
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
        headers: { 'Content-Type':'application/json' }, 
        body: JSON.stringify({ 
          model_name: primaryModel, 
          messages: buildHistory('primary','secondary', primaryCharacter, secondaryModel), 
          gpu_id: 0, 
          userProfile,
          use_web_search: webSearchEnabled // NEW: Add to primary
        }) 
      }).then(r => r.json()),
      fetch(`${SECONDARY_API_URL}/generate`, { 
        method: 'POST', 
        headers: { 'Content-Type':'application/json' }, 
        body: JSON.stringify({ 
          model_name: secondaryModel, 
          messages: buildHistory('secondary','primary', secondaryCharacter, primaryModel), 
          gpu_id: 1, 
          userProfile,
          use_web_search: false // Usually only primary should do web search in dual mode
        }) 
      }).then(r => r.json())
    ]);

    setMessages(prev => [...prev,
      { id: generateUniqueId(), role: 'bot', content: cleanModelOutput(pRes.text), modelId: 'primary', characterName: primaryCharacter?.name, avatar: primaryCharacter?.avatar },
      { id: generateUniqueId(), role: 'bot', content: cleanModelOutput(sRes.text), modelId: 'secondary', characterName: secondaryCharacter?.name, avatar: secondaryCharacter?.avatar }
    ]);

    await observeConversationWithAgent(text, `${pRes.text}\n\n${sRes.text}`);
  } catch (err) {
    console.error('Dual chat error:', err);
  } finally {
    setIsGenerating(false);
  }
}, [activeConversation, primaryModel, secondaryModel, messages, primaryCharacter, secondaryCharacter, PRIMARY_API_URL, SECONDARY_API_URL, userProfile, observeConversationWithAgent]);
// In AppContext.jsx, replace the entire generateReply function
// In AppContext.jsx, replace the entire generateReply function
// In AppContext.jsx, replace the entire generateReply function
const generateReply = useCallback(async (text, recentMessages) => {
    if (primaryIsAPI && primaryModel) {
        // --- API Model Path ---
        const agentMem = await fetchMemoriesFromAgent(text);
        const lore = await fetchTriggeredLore(text, activeCharacter);
        let memoryContext = '';
        if (agentMem.length) {
            memoryContext = agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n');
        }
        if (lore.length) {
            const loreBlock = lore.map(l => "• " + (typeof l === 'string' ? l : l.content || JSON.stringify(l))).join('\n');
            memoryContext += (memoryContext ? "\n\nWORLD KNOWLEDGE:\n" : "WORLD KNOWLEDGE:\n") + loreBlock;
        }
        let systemMsg = activeCharacter ? buildSystemPrompt(activeCharacter) : 'You are a helpful assistant...';
        if (memoryContext) {
            systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
        }

        const sanitizedMessages = recentMessages
            .filter(msg => (msg.role === 'user' || msg.role === 'bot') && typeof msg.content === 'string' && msg.content.trim() !== '')
            .map(msg => ({
                role: msg.role === 'bot' ? 'assistant' : 'user',
                content: msg.content
            }));
        
        const finalMessages = [{ role: 'system', content: systemMsg }, ...sanitizedMessages];

        try {
            const response = await generateReplyOpenAI({
                messages: finalMessages,
                model: primaryModel,
                settings: settings, // Pass the whole settings object
                apiUrl: PRIMARY_API_URL,
                stream: settings.streamResponses,
                targetGpuId: 0
            });

            if (settings.streamResponses) {
                return await processOpenAIStream(response, () => {}, (fullText) => fullText, (error) => { throw error; });
            } else {
                const result = await response.json();
                return result.choices?.[0]?.message?.content || "[No response]";
            }
        } catch (error) {
            console.error("API call failed in generateReply:", error);
            throw error;
        }

    } else {
        // --- Local Model Path (No changes needed here) ---
        // Your existing, working logic for local models goes here.
        // I have included it fully for completeness.
        const agentMem = await fetchMemoriesFromAgent(text);
        const lore = await fetchTriggeredLore(text, activeCharacter);
        let memoryContext = '';
        if (agentMem.length) {
            memoryContext = agentMem.map((m,i) => `[${i+1}] ${m.content}`).join('\n');
        }
        if (lore.length) {
            const loreContext = lore.map(l => "• " + (typeof l === 'string' ? l : l.content || JSON.stringify(l))).join('\n');
            memoryContext += (memoryContext ? "\n\nWORLD KNOWLEDGE:\n" : "WORLD KNOWLEDGE:\n") + loreContext;
        }
        let systemMsg = activeCharacter ? buildSystemPrompt(activeCharacter) : 'You are a helpful assistant...';
        if (memoryContext) {
            systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
        }
        const payload = {
            prompt: formatPrompt(recentMessages, primaryModel, systemMsg),
            model_name: primaryModel,
            temperature: settings.temperature,
            top_p: settings.top_p,
            top_k: settings.top_k,
            repetition_penalty: settings.repetition_penalty,
            max_tokens: settings.max_tokens,
            use_rag: settings.use_rag,
            rag_docs: settings.selectedDocuments || [],
            gpu_id: 0,
            userProfile: { id: userProfile?.id ?? 'anonymous' },
            memoryEnabled: true
        };
        if (settings.streamResponses) {
            const streamRes = await fetch(`${PRIMARY_API_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({...payload, stream: true})
            });
            if (!streamRes.ok) throw new Error(`API returned status ${streamRes.status}`);
            const reader = streamRes.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = '';
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, {stream: true});
                for (const line of chunk.split('\n\n')) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') return cleanModelOutput(accumulatedText);
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.text) accumulatedText += parsed.text;
                        } catch (e) {}
                    }
                }
            }
            return cleanModelOutput(accumulatedText);
        } else {
            const res = await fetch(`${PRIMARY_API_URL}/generate`, {
                method: 'POST',
                headers: { 'Content-Type':'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error(`API returned status ${res.status}`);
            const { text: botText } = await res.json();
            return cleanModelOutput(botText);
        }
    }
}, [
  primaryIsAPI,
  primaryModel,
  settings,
  fetchMemoriesFromAgent,
  fetchTriggeredLore,
  activeCharacter,
  userProfile?.id,
  PRIMARY_API_URL,
  buildSystemPrompt,
  formatPrompt,
  cleanModelOutput
]);

// In AppContext.jsx, replace the entire generateReplyWithOpenAI function
const generateReplyWithOpenAI = useCallback(async (text, recentMessages) => {
    console.log("🌐 [OpenAI] Processing with OpenAI API format");

    const apiUrl = PRIMARY_API_URL;
    const targetGpuId = 0;

    const convertToOpenAIMessages = (messages, systemPrompt) => {
        const openAiMsgs = messages
            .filter(msg => (msg.role === 'user' || msg.role === 'bot') && typeof msg.content === 'string')
            .map(msg => ({
                role: msg.role === 'bot' ? 'assistant' : 'user',
                content: msg.content
            }));
        return [{ role: 'system', content: systemPrompt }, ...openAiMsgs];
    };

    const agentMem = await fetchMemoriesFromAgent(text);
    const lore = await fetchTriggeredLore(text, activeCharacter);
    let memoryContext = '';

    if (agentMem.length) {
        memoryContext = agentMem.map((m,i) => `[${i+1}] ${m.content}`).join('\n');
    }
    if (lore.length) {
        const loreContext = lore.map(l => (typeof l === 'string' ? `• ${l}` : `• ${l.content || JSON.stringify(l)}`)).join('\n');
        memoryContext += (memoryContext ? "\n\nWORLD KNOWLEDGE:\n" : "WORLD KNOWLEDGE:\n") + loreContext;
    }

    let systemMsg = activeCharacter ? buildSystemPrompt(activeCharacter) : 'You are a helpful assistant.';
    if (memoryContext) {
        systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
    }

    const finalMessages = convertToOpenAIMessages(recentMessages, systemMsg);

    if (settings.streamResponses) {
        const response = await generateReplyOpenAI({
            messages: finalMessages,
            systemPrompt: null,
            model: primaryModel, // <-- FIX: Changed from modelName to model
            settings,
            apiUrl,
            apiKey: null,
            stream: true,
            targetGpuId: targetGpuId
        });
        return await processOpenAIStream(response, () => {}, (fullText) => fullText, (error) => { throw error; });
    } else {
        const response = await generateReplyOpenAI({
            messages: finalMessages,
            systemPrompt: null,
            model: primaryModel, // <-- FIX: Changed from modelName to model
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
const sendMessage = useCallback(async (text, webSearchEnabled = false) => {
  if (!activeConversation || !primaryModel) return;
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
    const isFirst  = userMsgs.length === 1;
    const conv     = conversations.find(c => c.id === activeConversation);

    if (isFirst && conv?.requiresTitle && !primaryIsAPI) {
      try {
        const title = await generateChatTitle(text, primaryModel);
        if (title && title !== 'New Chat') {
          setConversations(cs =>
            cs.map(c =>
              c.id === activeConversation
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
              c.id === activeConversation
                ? { ...c, name: `${text.substring(0, 25)}...`, requiresTitle: false }
                : c
            )
          );
    }

    // 3) Build memory + lore context
    const agentMem = await fetchMemoriesFromAgent(text);
    const lore     = await fetchTriggeredLore(text, activeCharacter);
    let memoryContext = '';

    if (agentMem.length) {
      memoryContext = agentMem.map((m,i) => `[${i+1}] ${m.content}`).join('\n');
    }
    if (lore.length) {
      const loreBlock = lore
        .map(l => "• " + (typeof l === 'string' ? l : l.content || JSON.stringify(l)))
        .join('\n');
      memoryContext += (memoryContext ? "\n\nWORLD KNOWLEDGE:\n" : "WORLD KNOWLEDGE:\n") + loreBlock;
    }

    let systemMsg = activeCharacter
      ? buildSystemPrompt(activeCharacter)
      : 'You are a helpful assistant...';
    if (memoryContext) {
      systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
    }

    // 4) Prepare payload
    const {
      temperature,
      top_p,
      top_k,
      repetition_penalty,
      use_rag,
      selectedDocuments = [],
      streamResponses
    } = settings;

    const payload = {
      prompt: formatPrompt(postUserHistory.slice(-5), primaryModel, systemMsg),
      model_name: primaryModel,
      temperature,
      top_p,
      top_k,
      repetition_penalty,
      use_rag,
      rag_docs: selectedDocuments,
      use_web_search: webSearchEnabled,
      gpu_id: 0,
      userProfile: { id: userProfile?.id ?? 'anonymous' },
      memoryEnabled: true
    };
    
    // 5) Streaming vs. Non-Streaming Logic
    if (streamResponses) {
        const botId = generateUniqueId();
        const placeholderBotMessage = {
            id: botId,
            role: 'bot',
            content: '',
            modelId: 'primary',
            characterName: activeCharacter?.name,
            avatar: activeCharacter?.avatar,
            isStreaming: true,
        };
        setMessages(prev => [...postUserHistory, placeholderBotMessage]);

        // Start the TTS WebSocket stream as soon as we create the placeholder
        console.log('🔍 [DEBUG] About to call startStreamingTTS with botId:', botId);
        console.log('🔍 [DEBUG] settings.ttsAutoPlay:', settings.ttsAutoPlay);
        console.log('🔍 [DEBUG] ttsEnabled:', ttsEnabled);
        console.log('🔍 [DEBUG] settings.ttsEnabled:', settings.ttsEnabled);
        startStreamingTTS(botId);

        if (primaryIsAPI && primaryModel) {
            // --- API MODEL PATH ---
            try {
                const response = await generateReplyOpenAI({
                    messages: postUserHistory,
                    systemPrompt: systemMsg,
                    modelName: primaryModel,
                    settings,
                    apiUrl: PRIMARY_API_URL,
                    stream: true,
                    targetGpuId: 0
                });

                let accumulated = '';
                let lastSentContent = '';
                await processOpenAIStream(
                    response,
                    (token) => { // onToken
                        accumulated += token;
                        const partial = cleanModelOutput(accumulated);
                        
                        const newTextChunk = partial.slice(lastSentContent.length);
                        addStreamingText(newTextChunk);
                        lastSentContent = partial;

                        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: partial } : m));
                    },
                    (finalText) => { // onComplete
                        const cleanedText = cleanModelOutput(finalText);
                        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: cleanedText, isStreaming: false } : m));
                        observeConversationWithAgent(text, cleanedText);
                        endStreamingTTS();
                    },
                    (error) => { // onError
                        console.error("OpenAI API Stream Error:", error);
                        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[API Error: ${error.message}]`, isStreaming: false } : m));
                        endStreamingTTS();
                    }
                );
            } catch (err) {
                console.error("Failed to initiate OpenAI API stream:", err);
                setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Error: Could not connect to API endpoint]`, isStreaming: false } : m));
                endStreamingTTS();
            } finally {
                setAbortController(null);
            }

        } else {
            // --- LOCAL MODEL PATH ---
            const controller = new AbortController();
            setAbortController(controller);
            
            try {
                const streamRes = await fetch(`${PRIMARY_API_URL}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...payload, stream: true }),
                    signal: controller.signal
                });

                if (!streamRes.ok) throw new Error(`Stream error ${streamRes.status}`);

                const reader = streamRes.body.getReader();
                const decoder = new TextDecoder();
                let accumulated = '';
                let lastSentContent = '';

                while (true) {
                    const { done: readerDone, value } = await reader.read();
                    if (readerDone) {
                        const finalText = cleanModelOutput(accumulated);
                        setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: finalText, isStreaming: false } : m));
                        await observeConversationWithAgent(text, finalText);
                        endStreamingTTS();
                        break;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    for (const line of chunk.split('\n\n')) {
                        if (!line.startsWith('data: ')) continue;
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                          const finalText = cleanModelOutput(accumulated);
                          setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: finalText, isStreaming: false } : m));
                          await observeConversationWithAgent(text, finalText);

                          // FIXED: Wait for final audio before ending TTS
                          console.log('🎵 [TTS] LLM stream ended, waiting 1s for final audio...');
                          setTimeout(() => {
                            console.log('🎵 [TTS] Timeout reached, ending TTS stream');
                            endStreamingTTS();
                          }, 1000);

                          return;
                        }
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.text) {
                                accumulated += parsed.text;
                                const partial = cleanModelOutput(accumulated);

                                const newTextChunk = partial.slice(lastSentContent.length);
                                addStreamingText(newTextChunk);
                                lastSentContent = partial;

                                setMessages(prev =>
                                    prev.map(m =>
                                        m.id === botId
                                            ? { ...m, content: partial }
                                            : m
                                    )
                                );
                            }
                        } catch (e) {
                           // Ignore parsing errors for non-JSON chunks
                        }
                    }
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    console.log('🛑 Generation stopped by user');
                    setMessages(prev => prev.map(m => m.id === botId ? { ...m, isStreaming: false } : m));
                } else {
                    console.error("Local stream error:", error);
                    setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Stream Error: ${error.message}]`, isStreaming: false } : m));
                }
                endStreamingTTS();
            } finally {
                setAbortController(null);
            }
        }
    } else {
      // --- Non-streaming logic (remains unchanged) ---
      const botContent = await generateReply(text, postUserHistory.slice(-5));
      const botMsg = {
        id: generateUniqueId(),
        role: 'bot',
        content: botContent,
        modelId: 'primary',
        characterName: activeCharacter?.name,
        avatar: activeCharacter?.avatar
      };
      setMessages(prev => [...postUserHistory, botMsg]);
      await observeConversationWithAgent(text, botContent);
    }

  } catch (err) {
    console.error("Chat error:", err);
  } finally {
    setIsGenerating(false);
  }
}, [
  activeConversation,
  primaryModel,
  messages,
  conversations,
  settings,
  activeCharacter,
  userProfile?.id,
  PRIMARY_API_URL,
  fetchMemoriesFromAgent,
  fetchTriggeredLore,
  buildSystemPrompt,
  formatPrompt,
  cleanModelOutput,
  generateChatTitle,
  observeConversationWithAgent,
  generateReply,
  primaryIsAPI,
  generateReplyOpenAI,
  processOpenAIStream
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
useEffect(() => {
  // Initial fetch on page load - only once
  fetchModels();
  fetchLoadedModels();
  fetchDocuments();
  loadCharacters();
  fetchAvailableSTTEngines();
  
  // No interval, no dependencies - runs exactly once on component mount
}, []);

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
  
const contextValue = useMemo(() => ({
    messages,
    setMessages,
    availableModels,
    setAvailableModels,
    loadedModels,
    activeModel,
    isModelLoading,
    loadModel,
    unloadModel,
    setIsGenerating,
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
    clearError: () => setApiError(null)
}), [
    messages, availableModels, loadedModels, activeModel, isModelLoading, loadModel, unloadModel, conversations, activeConversation, isGenerating, generateReply, primaryIsAPI, secondaryIsAPI, isSingleGpuMode, setActiveConversationWithMessages, deleteConversation, renameConversation, createNewConversation, getActiveConversationData, buildSystemPrompt, formatPrompt, settings, isRecording, fetchTriggeredLore, generateChatTitle, isPlayingAudio, isTranscribing, primaryModel, secondaryModel, audioError, startRecording, stopRecording, playTTS, isCallModeActive, callModeRecording, startCallMode, stopCallMode, stopTTS, playTTSWithPitch, sdStatus, fetchMemoriesFromAgent, handleStopGeneration, abortController, isStreamingStopped, checkSdStatus, generateImage, generatedImages, isImageGenerating, generateAndShowImage, apiError, handleConversationClick, cleanModelOutput, generateUniqueId, userProfile, sendMessage, updateSettings, inputTranscript, documents, fetchDocuments, uploadDocument, deleteDocument, getDocumentContent, autoMemoryEnabled, fetchLoadedModels, getRelevantMemories, MEMORY_API_URL, addConversationSummary, activeTab, shouldUseDualMode, sttEnginesAvailable, fetchAvailableSTTEngines, BACKEND, SECONDARY_API_URL, VITE_API_URL, endStreamingTTS, addStreamingText, startStreamingTTS, ttsClient, characters, activeCharacter, loadCharacters, saveCharacter, deleteCharacter, duplicateCharacter, applyCharacter, primaryCharacter, speechDetected, secondaryCharacter, primaryAvatar, secondaryAvatar, activeAvatar, showAvatars, applyAvatar, userAvatar, showAvatarsInChat, autoDeleteChats, dualModeEnabled, sendDualMessage, startAgentConversation, agentConversationActive, PRIMARY_API_URL
]);

return (
  <AppContext.Provider value={contextValue}>
    {children}
  </AppContext.Provider>
);
};

const useApp = () => React.useContext(AppContext); // Remove 'export' here
export { AppProvider, useApp }; // <-- This should be at the end
