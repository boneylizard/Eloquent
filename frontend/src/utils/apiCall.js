
// apiCall.js
import { getTemplateForModel } from './chat_templates';
import { retrieveRelevantMemories, formatMemoriesForPrompt } from './memoryUtils';
import { getBackendUrl, getSecondaryUrl } from '../config/api';

// Function to retrieve the currently active user profile object from localStorage
function getUserProfile() {
  try {
    const memoryStateStr = localStorage.getItem('user-profiles'); // Key used in MemoryContext
    if (!memoryStateStr) {
       console.warn('getUserProfile: No user-profiles found in localStorage.');
       return null;
    }

    const memoryState = JSON.parse(memoryStateStr);

    // Ensure the structure is as expected
    if (!memoryState || !Array.isArray(memoryState.profiles) || !memoryState.activeProfileId) {
        console.warn('getUserProfile: Invalid structure in localStorage user-profiles item.');
        return null;
    }

    // Find the active profile object using the activeProfileId
    const activeProfile = memoryState.profiles.find(p => p.id === memoryState.activeProfileId);

    if (!activeProfile) {
         console.warn(`getUserProfile: Active profile with ID ${memoryState.activeProfileId} not found in profiles array.`);
         // Fallback to the first profile if active one isn't found? Optional.
         // return memoryState.profiles[0] || null;
         return null;
    }

    // Return the complete active profile object
    return activeProfile;

  } catch (error) {
    console.error('🧠 [ERROR] Failed to get or parse user profile from localStorage:', error);
    return null;
  }
}

// Function to format the prompt with templates
function formatPrompt(messages, modelName, memoryContext = null) {
  const template = getTemplateForModel(modelName);
  let prompt = '';

  // Add system message
  const systemMessage = messages.find(m => m.role === 'system') || { content: template.default_system };
  
  // If we have memory context, add it to the system message
  let systemContent = systemMessage.content;
  if (memoryContext && memoryContext.trim()) {
    // Add memory context before the original system message
    systemContent = `${memoryContext}\n\n${systemContent}`;
  }
  
  prompt += template.system_start + systemContent + template.system_end;

  // Add user/assistant conversation messages
  const conversationMessages = messages.filter(m => m.role !== 'system');
  for (const message of conversationMessages) {
    if (message.role === 'user') {
      prompt += template.user_start + message.content + template.user_end;
    } else if (message.role === 'assistant') {
      prompt += template.assistant_start + message.content + template.assistant_end;
    }
  } 

  // Add assistant prefix for the next response
  prompt += template.assistant_start;
  return prompt;
}

// Function to get relevant memories for the prompt
function getMemoriesForPrompt(userMessage) {
  // Get relevant memories from localStorage
  const relevantMemories = retrieveRelevantMemories(userMessage);
  
  // Format memories for prompt inclusion
  const formattedMemories = formatMemoriesForPrompt(relevantMemories);
  
  console.log(`🧠 [INFO] Retrieved ${relevantMemories.length} relevant memories from userProfile`);
  
  return {
    memories: relevantMemories,
    formatted_memories: formattedMemories,
    memory_count: relevantMemories.length
  };
}

// Helper function to try fetching from backend, fall back to local retrieval
async function getMemoriesFromBackendOrLocal(userMessage) {
  try {
    // Get the userProfile for sending to backend
    // This function should return the active profile object
    const userProfile = getUserProfile();
    
    // Check if single GPU mode is enabled in settings
    const settings = JSON.parse(localStorage.getItem('LiangLocal-settings') || '{}');
    const singleGpuMode = settings.singleGpuMode === true;
    const targetGpu = singleGpuMode ? 0 : 1;


    
    // Try the backend approach first
    console.log('🧠 [INFO] Attempting to retrieve memories from backend (GPU 1)');
    const response = await fetch(`${getBackendUrl()}/relevant`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        prompt: userMessage,
        userProfile: userProfile, // Send the active profile object to the backend
        systemTime: new Date().toISOString(),
        gpu_id: targetGpu, // ✅ add this line to specify GPU ID
      })
    });
    
    if (!response.ok) {
      console.warn(`🧠 [WARN] Backend memory retrieval failed: ${response.status}, falling back to local retrieval`);
      return getMemoriesForPrompt(userMessage);
    }
    
    const result = await response.json();
    
    // Check if backend returned actual memories
    if (result.status === "success" && result.memory_count > 0) {
      console.log(`🧠 [INFO] Retrieved ${result.memory_count} memories from backend (GPU 1)`);
      return {
        memories: result.memories || [],
        formatted_memories: result.formatted_memories || "",
        memory_count: result.memory_count || 0
      };
    }
    
    // If backend didn't find memories or returned special instruction, use local retrieval
    console.log('🧠 [INFO] Backend returned no memories, using local retrieval');
    return getMemoriesForPrompt(userMessage);
    
  } catch (error) {
    // If any error occurs, fall back to local retrieval
    console.warn('🧠 [WARN] Error with backend memory retrieval, falling back to local:', error);
    return getMemoriesForPrompt(userMessage);
  }
}

// Main API call function with memory integration
export const apiCall = async (messages, modelName, options = {}) => {
  // Extract the latest user message for memory retrieval
  const userMessages = messages.filter(m => m.role === 'user');
  const latestUserMessage = userMessages.length > 0 ? userMessages[userMessages.length - 1].content : '';
  
  // Get memories relevant to the latest user message
  const memoryResult = await getMemoriesFromBackendOrLocal(latestUserMessage);
  
  // Format the prompt with memory context
  return callModelAPI(messages, modelName, options, memoryResult.formatted_memories);
};

export function callModelAPI(messages, modelName, options = {}, memoryContext = "") {
  const prompt   = formatPrompt(messages, modelName, memoryContext);
  const maxTokens = options.max_tokens ?? getContextLength();

  return fetch(`${getBackendUrl()}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      max_tokens:    maxTokens,
      temperature:   options.temperature    ?? 0.7,
      model_name:    modelName,
      stop:          options.stop_tokens   ?? [],
      stream:        false,
      context_length: getContextLength(),
      gpu_id:        0,
      userProfile:   getUserProfile(),
    }),
  })
  .then(async res => {
    const raw = await res.text();
    if (!res.ok) {
      console.error("callModelAPI: non-OK status", res.status, raw);
      throw new Error(`API ${res.status}: ${raw}`);
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch (e) {
      console.error("callModelAPI: invalid JSON", raw);
      throw e;
    }
    if (!data || typeof data !== "object") {
      console.error("callModelAPI: bad payload", data);
      throw new Error("Invalid JSON payload");
    }
    if (typeof data.text === "string") return data.text;
    console.error("callModelAPI: missing `text` field", data);
    throw new Error("Missing `text` in response");
  })
  .catch(err => {
    console.error("callModelAPI: uncaught error", err);
    throw err;
  });
}
  

// For streaming responses (if your backend supports it)
export function streamModelAPI(messages, modelName, onChunk, onDone, onError, options = {}) {
  // Extract the latest user message for memory retrieval
  const userMessages = messages.filter(m => m.role === 'user');
  const latestUserMessage = userMessages.length > 0 ? userMessages[userMessages.length - 1].content : '';
  
  // Get memories synchronously (might need to refactor to async for production)
  const userProfile = getUserProfile(); // Call function here
  const memoryResult = getMemoriesForPrompt(latestUserMessage);
  
  // Format the prompt with memory context
  const prompt = formatPrompt(messages, modelName, memoryResult.formatted_memories);
  
  const defaultOptions = {
    max_tokens: -1,
    temperature: 0.7,
    stop_tokens: ["<|im_end|>", "</s>"],
    stream: true // ✅ enable streaming by default
  };  
  
  const requestOptions = { ...defaultOptions, ...options };
  
  fetch(`${getBackendUrl()}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      prompt: prompt,
      max_tokens: requestOptions.max_tokens,
      temperature: requestOptions.temperature,
      model_name: modelName,
      stop: requestOptions.stop_tokens,
      stream: true,
      memory_included: memoryResult.memory_count > 0,
      gpu_id: 0, // ✅ add this line to specify GPU ID
      userProfile: userProfile(), // Use the active profile object
    }),
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    function readChunk() {
      reader.read().then(({ done, value }) => {
        if (done) {
          onDone();
          return;
        }
        
        try {
          const chunk = decoder.decode(value, { stream: true });
          // Parse SSE format or whatever format your backend uses
          // This assumes line-by-line JSON objects
          const lines = chunk.split('\n').filter(line => line.trim() !== '');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6);
              if (jsonStr === '[DONE]') {
                onDone();
                return;
              }
              
              try {
                const data = JSON.parse(jsonStr);
                onChunk(data.text || data.token || data.chunk || '');
              } catch (e) {
                console.warn('Could not parse chunk:', jsonStr);
              }
            }
          }
        } catch (error) {
          console.error('Error processing stream:', error);
          onError(error);
          return;
        }
        
        readChunk();
      }).catch(error => {
        console.error('Error reading stream:', error);
        onError(error);
      });
    }
    
    readChunk();
  })
  .catch(error => {
    console.error('Error initiating stream:', error);
    onError(error);
  });
}

// Updated streamModelAPI with async memory retrieval (preferred approach)
export async function streamModelAPIWithMemory(messages, modelName, onChunk, onDone, onError, options = {}) {
  try {
    // Extract the latest user message for memory retrieval
    const userMessages = messages.filter(m => m.role === 'user');
    const latestUserMessage = userMessages.length > 0 ? userMessages[userMessages.length - 1].content : '';
    
    // Get memories asynchronously with proper error handling
    const memoryResult = await getMemoriesFromBackendOrLocal(latestUserMessage);
    
    // Format the prompt with memory context
    const prompt = formatPrompt(messages, modelName, memoryResult.formatted_memories);
    // userProfile = getUserProfile(); // Call function here
    const userProfile = getUserProfile(); // Call function here
    
    const defaultOptions = {
      max_tokens: -1,
      temperature: 0.7,
      stop_tokens: ["<|im_end|>", "</s>"],
      stream: true
    };  
    
    const requestOptions = { ...defaultOptions, ...options };
    
    // Continue with streaming API call
    streamAPICall(prompt, modelName, requestOptions, onChunk, onDone, onError, memoryResult.memory_count > 0);
  } catch (error) {
    console.error('Error preparing streaming request:', error);
    onError(error);
  }
}

// Helper function to perform the actual streaming API call
function streamAPICall(prompt, modelName, options, onChunk, onDone, onError, hasMemories) {
  fetch(`${getBackendUrl()}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      prompt: prompt,
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      model_name: modelName,
      stop: options.stop_tokens,
      stream: true,
      memory_included: hasMemories, // Metadata for tracking 
      gpu_id: 0, // ✅ add this line to specify GPU ID
      userProfile: userProfile, // Use the active profile object
    }),
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    function readChunk() {
      reader.read().then(({ done, value }) => {
        if (done) {
          onDone();
          return;
        }
        
        try {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n').filter(line => line.trim() !== '');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6);
              if (jsonStr === '[DONE]') {
                onDone();
                return;
              }
              
              try {
                const data = JSON.parse(jsonStr);
                onChunk(data.text || data.token || data.chunk || '');
              } catch (e) {
                console.warn('Could not parse chunk:', jsonStr);
              }
            }
          }
        } catch (error) {
          console.error('Error processing stream:', error);
          onError(error);
          return;
        }
        
        readChunk();
      }).catch(error => {
        console.error('Error reading stream:', error);
        onError(error);
      });
    }
    
    readChunk();
  })
  .catch(error => {
    console.error('Error initiating stream:', error);
    onError(error);
  });
}

// --- Context Length Helpers ---
export function saveContextLength(length) {
  localStorage.setItem('preferredContextLength', length.toString());
}

export function getContextLength() {
  const saved = localStorage.getItem('preferredContextLength');
  return saved ? parseInt(saved, 10) : 4096; // Default fallback
}


// Replace the existing fetchTriggeredLore function in apiCall.js with this one:

export async function fetchTriggeredLore(message, activeCharacter) {
  // Enhanced logging that shows ALL possible lore structures
  console.log('🌍 [LORE] Character data inspection:', {
    hasCharacter: Boolean(activeCharacter),
    name: activeCharacter?.name,
    // Log ALL properties that might contain lore
    allProps: activeCharacter ? Object.keys(activeCharacter) : [],
    loreProps: activeCharacter ? Object.keys(activeCharacter).filter(key => 
      key.toLowerCase().includes('lore')
    ) : [],
    // Debug actual lore structures
    loreEntries: activeCharacter?.loreEntries,
    loreKeywords: activeCharacter?.loreKeywords,
    lore: activeCharacter?.lore
  });
  
  if (!message) {
    console.warn('🌍 [LORE] Missing message parameter');
    return [];
  }
  
  if (!activeCharacter) {
    console.warn('🌍 [LORE] Missing activeCharacter parameter');
    return [];
  }
  
  // First try local keyword detection based on character structure
  try {
    const normalizedMessage = message.toLowerCase();
    const triggeredLore = [];
    
    // APPROACH 1: Try standard loreEntries array (Python backend expects this)
    if (Array.isArray(activeCharacter.loreEntries) && activeCharacter.loreEntries.length > 0) {
      console.log(`🌍 [LORE] Found ${activeCharacter.loreEntries.length} loreEntries to check`);
      
      for (const entry of activeCharacter.loreEntries) {
        // Skip invalid entries
        if (!entry || typeof entry !== 'object' || !entry.content) continue;
        
        // Get keywords array
        const keywords = Array.isArray(entry.keywords) ? entry.keywords : [];
        
        // Check if any keyword matches
        for (const keyword of keywords) {
          if (!keyword || typeof keyword !== 'string') continue;
          
          if (normalizedMessage.includes(keyword.toLowerCase())) {
            console.log(`🌍 [LORE] Matched keyword "${keyword}" from loreEntries`);
            triggeredLore.push({
              keyword: keyword,
              content: entry.content,
              importance: entry.importance || 0.8,
              source: 'loreEntries'
            });
            break; // Only match once per entry
          }
        }
      }
    }
    // APPROACH 2: Try loreKeywords object (new format)
    else if (activeCharacter.loreKeywords && typeof activeCharacter.loreKeywords === 'object') {
      console.log(`🌍 [LORE] Found loreKeywords object with ${Object.keys(activeCharacter.loreKeywords).length} entries`);
      
      for (const [keyword, content] of Object.entries(activeCharacter.loreKeywords)) {
        if (!keyword || typeof keyword !== 'string' || !content) continue;
        
        if (normalizedMessage.includes(keyword.toLowerCase())) {
          console.log(`🌍 [LORE] Matched keyword "${keyword}" from loreKeywords`);
          triggeredLore.push({
            keyword: keyword,
            content: content,
            importance: 0.8, // Default importance
            source: 'loreKeywords'
          });
        }
      }
    }
    // APPROACH 3: Look for other structures that might contain lore
    else {
      console.log(`🌍 [LORE] No standard lore structures found, looking for alternatives`);
      
      // Check if any property might contain lore keywords
      const loreProps = Object.keys(activeCharacter).filter(key => 
        key.toLowerCase().includes('lore') || 
        key.toLowerCase().includes('knowledge')
      );
      
      for (const prop of loreProps) {
        const value = activeCharacter[prop];
        
        // Handle objects that might be lore mappings
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          console.log(`🌍 [LORE] Checking object property ${prop} for lore`);
          
          for (const [key, content] of Object.entries(value)) {
            if (!key || typeof key !== 'string' || !content) continue;
            
            if (normalizedMessage.includes(key.toLowerCase())) {
              console.log(`🌍 [LORE] Matched keyword "${key}" from ${prop}`);
              triggeredLore.push({
                keyword: key,
                content: typeof content === 'string' ? content : JSON.stringify(content),
                importance: 0.8,
                source: prop
              });
            }
          }
        }
        // Handle arrays that might contain lore entries
        else if (Array.isArray(value)) {
          console.log(`🌍 [LORE] Checking array property ${prop} for lore`);
          
          for (const item of value) {
            if (!item || typeof item !== 'object') continue;
            
            // Check if item has content and keywords fields
            if (item.content && Array.isArray(item.keywords)) {
              for (const keyword of item.keywords) {
                if (!keyword || typeof keyword !== 'string') continue;
                
                if (normalizedMessage.includes(keyword.toLowerCase())) {
                  console.log(`🌍 [LORE] Matched keyword "${keyword}" from ${prop}`);
                  triggeredLore.push({
                    keyword: keyword,
                    content: item.content,
                    importance: item.importance || 0.8,
                    source: prop
                  });
                  break;
                }
              }
            }
          }
        }
      }
    }
    
    // Return local matches if we found any
    if (triggeredLore.length > 0) {
      console.log(`🌍 [LORE] Local detection found ${triggeredLore.length} matches`);
      return triggeredLore;
    }
  } catch (error) {
    console.error("🌍 [LORE] Error in local lore detection:", error);
  }
  
  // If local detection didn't find anything, try the backend API
  try {
    console.log(`🌍 [LORE] Calling backend /memory/detect_keywords API...`);
    
    // Create a compatible character object for the backend
    // Check if we have appropriate lore structure to send
    let loreEntries = null;
    
    if (Array.isArray(activeCharacter.loreEntries)) {
      loreEntries = activeCharacter.loreEntries;
    } else if (activeCharacter.loreKeywords && typeof activeCharacter.loreKeywords === 'object') {
      // Convert loreKeywords to loreEntries format for backend
      loreEntries = Object.entries(activeCharacter.loreKeywords).map(([keyword, content]) => ({
        content: content,
        keywords: [keyword]
      }));
    }
    
    // Only proceed if we have lore entries to send
    if (!loreEntries || loreEntries.length === 0) {
      console.log(`🌍 [LORE] No lore entries to send to backend`);
      return [];
    }
    
    const characterForBackend = {
      id: activeCharacter.id,
      name: activeCharacter.name,
      loreEntries: loreEntries
    };
    
    console.log(`🌍 [LORE] Sending ${loreEntries.length} lore entries to backend`);

    const response = await fetch(`${getBackendUrl()}/memory/detect_keywords`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        activeCharacter: characterForBackend
      })
    });

    if (!response.ok) {
      console.warn(`🌍 [LORE] Backend API failed: ${response.status}`);
      return [];
    }
    
    const result = await response.json();
    
    if (result.status === "success" && Array.isArray(result.lore_triggered)) {
      const loreCount = result.lore_triggered.length;
      console.log(`🌍 [LORE] Backend found ${loreCount} triggered entries`);
      return result.lore_triggered;
    } else {
      console.warn(`🌍 [LORE] Unexpected backend response:`, result);
      return [];
    }
  } catch (error) {
    console.error("🌍 [LORE] Backend detection failed:", error);
    return []; 
  }
}

export const synthesizeSpeech = async (text, options = {}) => {
  try {
    // Handle both old format (string voice) and new format (options object)
    let voice, engine, audio_prompt_path, exaggeration, cfg;
    
    if (typeof options === 'string') {
      // Old format: synthesizeSpeech(text, voice)
      voice = options;
      engine = 'kokoro';
      exaggeration = 0.5;
      cfg = 0.5;
    } else {
      // New format: synthesizeSpeech(text, { voice, engine, ... })
      voice = options.voice || 'af_heart';
      engine = options.engine || 'kokoro';
      audio_prompt_path = options.audio_prompt_path;
      exaggeration = options.exaggeration || 0.5;
      cfg = options.cfg || 0.5;
    }

    console.log(`Calling TTS API with engine "${engine}" and voice "${voice}":`, text.substring(0, 50));

    const payload = {
      text,
      voice,
      engine,
      exaggeration,
      cfg
    };

    // Add voice cloning path if provided
    if (audio_prompt_path) {
      payload.audio_prompt_path = audio_prompt_path;
      console.log(`Using voice cloning with reference: ${audio_prompt_path}`);
    }
    // Dia voice cloning
    if (engine === 'dia' && options.dia_audio_prompt_path) {
      payload.dia_audio_prompt_path = options.dia_audio_prompt_path;
    }
    
    const response = await fetch(`${getBackendUrl()}/tts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      console.error("TTS API error:", error);
      throw new Error(`TTS failed: ${response.status} - ${error.detail}`);
    }

    const audioBlob = await response.blob();

    if (audioBlob.size === 0 || !audioBlob.type.startsWith('audio/')) {
      throw new Error("Received invalid or empty audio data from backend.");
    }

    return URL.createObjectURL(audioBlob); // Temporary blob URL for playback
  } catch (error) {
    console.error("🔥 Error calling synthesizeSpeech:", error);
    throw error;
  }
};

// Add these new helper functions to apiCall.js:

export const uploadVoiceReference = async (audioFile) => {
  try {
    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await fetch(`${getBackendUrl()}/tts/upload-voice`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Upload failed: ${response.status} - ${error.detail}`);
    }

    const result = await response.json();
    console.log('Voice reference uploaded:', result);
    return result;
  } catch (error) {
    console.error('🔥 Error uploading voice reference:', error);
    throw error;
  }
};

export const getAvailableVoices = async () => {
  try {
    const response = await fetch(`${getBackendUrl()}/tts/voices`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch voices: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Available voices:', data);
    return data;
  } catch (error) {
    console.error('🔥 Error fetching available voices:', error);
    throw error;
  }
};

// Function to transcribe audio using the STT API
// This function assumes the backend is running and accessible at the specified URL
export const transcribeAudio = async (audioBlob, engine = "whisper") => {
  console.log(`📝 Sending audio blob for transcription with engine: ${engine}`);

  try {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm'); // or .ogg or .wav, based on your MIME type

    // Add engine as a query parameter
    const response = await fetch(`${getBackendUrl()}/transcribe?engine=${engine}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("STT API failed:", errorText);
      throw new Error(`Transcription failed: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    console.log("📝 Transcript received:", result.transcript);
    return result.transcript;

  } catch (err) {
    console.error("🔥 Error calling transcribeAudio:", err);
    throw err;
  }
};


// --- Chat Title Generation ---
export const generateChatTitle = async (message, modelName) => {
  console.log("🔤 [TITLE] Generating title for message:", message.substring(0, 30) + "...");
  
  if (!message || typeof message !== 'string' || message.trim() === '') {
    console.warn("🔤 [TITLE] Empty message provided, returning default title");
    return "New Chat";
  }
  
  try {
    // Handle different modelName formats more carefully
    let actualModelName;
    if (typeof modelName === 'object' && modelName !== null) {
      // Try to extract model name from object with multiple fallbacks
      actualModelName = modelName.name || modelName.model_name || modelName.model;
      console.log("🔤 [TITLE] Extracted model name from object:", actualModelName);
    } else if (typeof modelName === 'string') {
      actualModelName = modelName;
      console.log("🔤 [TITLE] Using string model name:", actualModelName);
    } else {
      console.warn("🔤 [TITLE] Invalid model name, returning default title");
      return "New Chat"; // Just return default without API call if model is invalid
    }
    
    // Additional validation to prevent "Unknown" model error
    if (!actualModelName || actualModelName === "Unknown") {
      console.warn("🔤 [TITLE] Missing or invalid model name:", actualModelName);
      return "New Chat";
    }
    
    console.log(`🔤 [TITLE] Using model for title generation: ${actualModelName}`);
    
    // Simple and direct API call
    const apiUrl = `${getBackendUrl()}/generate`;
    console.log("🔤 [TITLE] API URL:", apiUrl);
    const payload = {
      prompt: `Generate a short title (3-5 words) for a chat that starts with this message. Reply with ONLY the title, no quotes or explanations:\n\n${message}`,
      model_name: actualModelName,
      temperature: 0.7,
      max_tokens: 20,
      stream: false,
      gpu_id: 0,
      request_purpose: "title_generation" // <<< ADD THIS LINE
    };
    
    console.log("🔤 [TITLE] Sending API request with payload:", {
      modelName: payload.model_name,
      promptPreview: payload.prompt.substring(0, 60) + "..."
    });
    
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    console.log("🔤 [TITLE] API response status:", response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error("🔤 [TITLE] API error response:", errorText);
      return "New Chat"; // Return default title on API error
    }
    
    const data = await response.json();
    console.log("🔤 [TITLE] Raw API response:", data);
    
    if (!data || !data.text) {
      console.error("🔤 [TITLE] Missing text in API response");
      return "New Chat";
    }
    
    let title = data.text.trim();
    console.log("🔤 [TITLE] Raw title:", title);
    
    // More thorough cleaning of common artifacts
    title = title
      .replace(/^["'`]|["'`]$/g, '') // Remove quotes at start/end
      .replace(/^title:?\s*|^chat title:?\s*/i, '') // Remove "Title:" prefix
      .replace(/^the\s+/i, '') // Remove leading "The"
      .replace(/[.!?]$/, '') // Remove ending punctuation
      .trim();
    
    console.log("🔤 [TITLE] Cleaned title:", title);
    
    // Only check if completely empty, not minimum length
    if (!title) {
      console.warn("🔤 [TITLE] Empty title, using default");
      return "New Chat";
    }
    
    // Title length caps
    if (title.length > 40) {
      title = title.substring(0, 37) + "...";
      console.log("🔤 [TITLE] Truncated long title:", title);
    }
    
    return title;
  } catch (error) {
    console.error("🔤 [TITLE] Title generation error:", error);
    return "New Chat"; // Fallback title on any error
  }
};
class TTSWebSocketClient {
  constructor() {
    this.socket = null;
    this.audioQueue = [];
    this.isPlaying = false;
    this.onAudioQueueUpdate = null;
    this.settingsSent = false;
    this.pendingSettings = null;
    this.isConnecting = false;
  }

  connect(onOpen, onClose, onError) {
    // Prevent multiple connections
    if (this.socket && this.socket.readyState < 2) {
      console.warn("TTS WebSocket is already connected or connecting.");
      return;
    }

    // Connect to the TTS service on port 8002
    this.socket = new WebSocket('ws://localhost:8002/tts-stream');

    // Track connection state
    this.isConnecting = true;

    this.socket.onopen = () => {
      console.log("✅ [WebSocket] Connection established.");
      this.isConnecting = false;
      
      // Send any pending settings if we have them
      if (this.pendingSettings) {
        console.log("🔄 [WebSocket] Sending pending settings:", this.pendingSettings);
        this.socket.send(JSON.stringify(this.pendingSettings));
        this.settingsSent = true;
        this.pendingSettings = null;
      }
      
      if (onOpen) onOpen();
    };

    this.socket.onmessage = async (event) => {
      // The backend will send audio as binary data (a Blob)
      if (event.data instanceof Blob) {
        const arrayBuffer = await event.data.arrayBuffer();
        this.audioQueue.push(arrayBuffer);
        // Notify the AppContext that new audio is available
        if (this.onAudioQueueUpdate) this.onAudioQueueUpdate();
      }
    };

    this.socket.onclose = (event) => {
      console.log("🛑 [WebSocket] Connection closed.", event.reason);
      this.socket = null;
      this.settingsSent = false;
      this.isConnecting = false;
      if (onClose) onClose();
    };

    this.socket.onerror = (error) => {
      console.error("❌ [WebSocket] Error:", error);
      this.isConnecting = false;
      if (onError) onError(error);
    };
  }

// Send a chunk of text to the backend for synthesis
send(text) {
  
  const isSettings = typeof text === 'object' && (text.engine || text.voice);
  
  // For settings, just store them and send if connected
  if (isSettings) {
    console.log("🔧 [WebSocket] Received settings for new message:", text);
    this.pendingSettings = text;
    this.settingsSent = false;
    
    if (this.socket && this.socket.readyState === 1) {
      console.log("📤 [WebSocket] Sending settings immediately (already connected)");
      this.socket.send(JSON.stringify(text));
      this.settingsSent = true;
      this.pendingSettings = null;
    }
    return;
  }
  
  // For text - since we keep connection open, this should always work
  if (!this.socket || this.socket.readyState !== 1) {
    console.error("❌ [WebSocket] Not connected! This shouldn't happen with always-open connection");
    return;
  }
  
  if (!this.settingsSent && this.pendingSettings) {
    // Send pending settings first if we have them
    this.socket.send(JSON.stringify(this.pendingSettings));
    this.settingsSent = true;
    this.pendingSettings = null;
  }
  
  // Send the text
  this.socket.send(JSON.stringify({ text }));
}

  // Signal end of current message stream
  closeStream() {
    if (this.socket && this.socket.readyState === 1) {
      console.log("🏁 [WebSocket] Sending end signal for current message");
      this.socket.send("--END--");
      this.settingsSent = false;  // Reset for next message
      // Keep connection open for next message!
    }
  }

  // Disconnect the WebSocket entirely
  disconnect() {
    if (this.socket) {
      console.log("👋 [WebSocket] Disconnecting WebSocket");
      this.socket.close();
      this.settingsSent = false;
    }
  }

  // Retrieve the next audio chunk from the queue
  getNextAudio() {
    return this.audioQueue.shift();
  }
}

export const uploadDiaVoiceReference = async (audioFile) => {
  try {
    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await fetch(`${getBackendUrl()}/tts/upload-dia-voice`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading Dia voice:', error);
    throw error;
  }
};

export const getDiaVoices = async () => {
  try {
    const response = await fetch(`${getBackendUrl()}/tts/dia-voices`);
    if (!response.ok) {
      throw new Error(`Failed to fetch Dia voices: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching Dia voices:', error);
    throw error;
  }
};
// Export a single instance of the client for the whole app to use
export const ttsClient = new TTSWebSocketClient();