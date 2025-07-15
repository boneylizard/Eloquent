// src/utils/memoryUtils.js - FINAL CORRECTED VERSION #2 (April 9, 2025)
// Fixes duplicate sync AND ensures all needed functions are exported.

// Memory API endpoint base URL
function getBackendUrl() {
  try {
    const settings = JSON.parse(localStorage.getItem('LiangLocal-settings') || '{}');
    const isSingleGpuMode = settings.singleGpuMode === true;
    return isSingleGpuMode ? 'http://localhost:8000' : 'http://localhost:8001';
  } catch (error) {
    console.warn('Could not read GPU mode from settings, defaulting to dual GPU mode');
    return 'http://localhost:8001';
  }
}
const MEMORY_API_URL = getBackendUrl(); // Ensure this port is correct

// --- HELPER FUNCTIONS ---

/**
 * Helper to generate unique IDs if needed
 * @returns {string} - Generated unique ID
 */
const generateId = () => `mem_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

/**
 * Helper to normalize content for reliable duplicate detection
 * (Lowercase, trim, remove common punctuation, reduce whitespace)
 * @param {string} content - Memory content to normalize
 * @returns {string} - Normalized content
 */
const normalizeContent = (content) => {
  if (!content || typeof content !== 'string') return '';
  return content
    .toLowerCase()
    .replace(/[.,!?;:]/g, '') // Remove common punctuation
    .replace(/\s+/g, ' ')    // Replace multiple spaces with single space
    .trim();
};

/**
 * Helper function to extract n-grams from text (Used by local retrieval)
 * @param {string} text - Text to extract n-grams from
 * @param {number} n - Size of n-grams
 * @returns {Array} - Array of n-grams
 */
function extractNGrams(text, n) {
  if (!text || typeof text !== 'string') return [];
  const nGrams = [];
  const cleanText = text.replace(/\s+/g, ' ').trim();
  for (let i = 0; i <= cleanText.length - n; i++) {
    nGrams.push(cleanText.substring(i, i + n));
  }
  return nGrams;
}


// --- CORE SYNCHRONIZATION FUNCTION ---

/**
 * Fetches all memories from backend (/get_all), compares with local storage (via context),
 * filters out duplicates (both existing in frontend AND within the new batch),
 * and adds only unique new memories to local storage via memoryContext.
 * Intended to be called AFTER /generate response is processed by frontend.
 *
 * @param {object} memoryContext - The memory context from useMemory() providing { memories, addMemory }
 * @returns {Promise<boolean>} - Whether the sync attempt completed successfully
 */
export const syncBackendMemoriesToLocalStorage = async (memoryContext, activeProfileId) => {
  // Check if context and required function exist
  if (!memoryContext || typeof memoryContext.addMemory !== 'function' || !Array.isArray(memoryContext.memories)) {
    console.error("🧠 [ERROR] Sync: Memory context, addMemory function, or memories array is missing/invalid.");
    return false;
  }

  try {
    // 1. Fetch ALL memories from the backend's persistent store
    console.log("🧠 [INFO] Sync: Fetching all memories from backend (/memory/get_all)...");
    // Add check for activeProfileId
    if (!activeProfileId) {
      console.error("🧠 [ERROR] Sync: Cannot sync backend memories without an activeProfileId.");
      return false;
    }
    console.log(`🧠 [INFO] Sync: Fetching all memories from backend (/memory/get_all) for user: ${activeProfileId}...`); // Log user ID
    // Use user_id as the query parameter name
    const response = await fetch(`${MEMORY_API_URL}/memory/get_all?user_id=${activeProfileId}`);
    if (!response) {
      console.error("🧠 [ERROR] Sync: No response from backend. Check server status.");
      return false;
    }

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`🧠 [ERROR] Sync: Failed to fetch backend memories: ${response.status} - ${errorText}`);
      return false;
    }

    const data = await response.json();
    const backendMemories = Array.isArray(data.memories) ? data.memories : [];
    console.log(`🧠 [INFO] Sync: Retrieved ${backendMemories.length} memories from backend.`);

    if (backendMemories.length === 0) {
      console.log("🧠 [INFO] Sync: No memories on backend to sync.");
      return true; // Nothing to do
    }

    // 2. Get existing memory contents from context for duplicate detection
    const existingMemories = memoryContext.memories;
    const existingNormalizedContents = new Set(
      existingMemories.map(m => normalizeContent(m.content))
    );
    console.log(`🧠 [INFO] Sync: Comparing against ${existingNormalizedContents.size} unique normalized contents in frontend state.`);

    // 3. Filter out memories whose normalized content already exists in the frontend state
    const memoriesNotInFrontend = backendMemories.filter(memory => {
        if (!memory || typeof memory.content !== 'string') return false;
        return !existingNormalizedContents.has(normalizeContent(memory.content));
    });
    console.log(`🧠 [INFO] Sync: Found ${memoriesNotInFrontend.length} memories not already present in frontend.`);

    // 4. Deduplicate the *new* batch itself based on normalized content
    const uniqueNewMemoriesMap = new Map();
    memoriesNotInFrontend.forEach(memory => {
        const normalizedContentKey = normalizeContent(memory.content);
        if (!uniqueNewMemoriesMap.has(normalizedContentKey)) {
            uniqueNewMemoriesMap.set(normalizedContentKey, { id: memory.id || generateId(), ...memory });
        } else {
             console.log(`🧠 [INFO] Sync: Skipping duplicate memory within the new backend batch: ${memory.content.substring(0, 50)}...`);
        }
    });
    const uniqueNewMemories = Array.from(uniqueNewMemoriesMap.values());
    console.log(`🧠 [INFO] Sync: After internal batch deduplication, ${uniqueNewMemories.length} unique new memories remain.`);

    // 5. Add only the unique new memories to local storage via the context
    if (uniqueNewMemories.length > 0) {
      console.log(`🧠 [INFO] Sync: Adding ${uniqueNewMemories.length} unique new memories to frontend state via context...`);
      uniqueNewMemories.forEach(memoryToAdd => {
          console.debug("🧠 [DEBUG] Sync: Adding memory:", memoryToAdd);
          memoryContext.addMemory(memoryToAdd);
      });
       console.log(`🧠 [INFO] Sync: Finished adding ${uniqueNewMemories.length} new memories to frontend state.`);
    } else {
       console.log("🧠 [INFO] Sync: No unique new memories found to add to frontend state.");
    }

    return true;

  } catch (error) {
    console.error("🧠 [ERROR] Sync: Memory synchronization failed:", error);
    return false;
  }
};

// === INITIALIZATION ===
// Module-scope flag so we only log once per page load
let _hasLoggedInitialSync = false;
// localStorage key to persist across reloads
const _initialSyncKey = 'memoryInitSyncComplete_v1';

/**
 * Performs the initial memory synchronization if needed.
 * @param {{ alreadySynced: boolean, sync: ()=>Promise<void> }} memoryContext
 * @returns {Promise<boolean>} true if sync ran, false if it was skipped
 */
export async function initializeMemories(memoryContext) {
  const alreadyContext = Boolean(memoryContext?.alreadySynced);
  const alreadyStorage = localStorage.getItem(_initialSyncKey) === 'true';

  // If either in-memory flag or localStorage says we’ve already synced, skip
  if (alreadyContext || alreadyStorage) {
    if (!_hasLoggedInitialSync) {
      console.info("🧠 [INFO] Initial memory sync already complete - skipping");
      _hasLoggedInitialSync = true;
    }
    return false;
  }

  // Otherwise perform the sync
  try {
    await memoryContext.sync();

    // Mark as done in localStorage so future reloads skip
    localStorage.setItem(_initialSyncKey, 'true');
    return true;
  } catch (err) {
    console.error("🧠 [ERROR] Initial memory sync failed:", err);
    return false;
  }
}

// --- CONTEXT RETRIEVAL & FORMATTING ---

/**
 * Retrieves relevant memories for context, using the backend /relevant endpoint.
 * Sends the current userProfile (from localStorage via context) to the backend.
 *
 * @param {string} prompt - User message
 * @param {object} memoryContext - The memory context from useMemory() providing userProfile
 * @returns {Promise<object>} - Object like { memories: [], formatted_memories: "", memory_count: 0 }
 */
export const getRelevantMemoriesFromBackend = async (prompt, memoryContext) => {
   // ... (Implementation from previous version - seems okay) ...
   try {
    const userProfile = memoryContext?.userProfile;
    if (!userProfile) {
        console.warn("🧠 [WARN] No active userProfile in context for getRelevantMemoriesFromBackend");
    }

    console.log(`🧠 [INFO] Calling backend /relevant for prompt: ${prompt.substring(0, 50)}...`);
    const response = await fetch(`${MEMORY_API_URL}/relevant`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        userProfile: userProfile || {}, // Send profile or empty object
        systemTime: new Date().toISOString()
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`🧠 [ERROR] /relevant API error: ${response.status} - ${errorText}`);
      return { memories: [], formatted_memories: "", memory_count: 0 };
    }

    const result = await response.json();

    if (result.status === "success") {
      console.log(`🧠 [INFO] /relevant retrieved ${result.memory_count || 0} relevant memories from ${result.retrieval_source || 'backend'}`);
      return {
        memories: result.memories || [],
        formatted_memories: result.formatted_memories || "",
        memory_count: result.memory_count || 0
      };
    } else {
      console.error(`🧠 [ERROR] /relevant retrieval failed: ${result.error || 'Unknown error'}`);
      return { memories: [], formatted_memories: "", memory_count: 0 };
    }

  } catch (error) {
    console.error("🧠 [ERROR] /relevant call failed:", error);
    return { memories: [], formatted_memories: "", memory_count: 0 };
  }
};

/**
 * Formats memories for inclusion in the AI prompt
 *
 * @param {Array} memories - Array of memory objects (assumed pre-sorted by relevance)
 * @param {number} maxMemories - Maximum number of memories to include
 * @param {number} maxChars - Maximum total characters in formatted output
 * @returns {string} - Formatted memory string for prompt
 */
export const formatMemoriesForPrompt = (memories, maxMemories = 5, maxChars = 1000) => {
   // ... (Implementation from previous version - seems okay) ...
   if (!Array.isArray(memories) || memories.length === 0) {
    return "";
  }

  const topMemories = memories.slice(0, maxMemories);
  let formatted = "RELEVANT USER INFORMATION:\n";
  let totalChars = formatted.length;
  let memoriesIncludedCount = 0;

  for (const memory of topMemories) {
    if (!memory || typeof memory.content !== 'string') continue;

    const content = memory.content.replace(/[\r\n]+/g, ' ');
    const category = (memory.category || 'general').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const importance = parseFloat(memory.importance || 0.5);
    const stars = "★".repeat(Math.max(1, Math.min(5, Math.round(importance * 5))));
    const memoryLine = `• [${category}] ${content} (${stars})\n`;

    if (totalChars + memoryLine.length > maxChars) {
        const availableChars = maxChars - totalChars - (category.length + 10); // Approx space
        if (availableChars > 20) {
             formatted += `• [${category}] ${content.substring(0, availableChars)}... (${stars})\n`;
             memoriesIncludedCount++;
        }
        break;
    }

    formatted += memoryLine;
    totalChars += memoryLine.length;
    memoriesIncludedCount++;

    if (memoriesIncludedCount >= maxMemories) break;
  }

  return formatted.trim();
};


// --- FUNCTIONS NEEDED BY OTHER MODULES (Ensure they are exported) ---

/**
 * Retrieves relevant memories directly from localStorage. Exported as it's used elsewhere (e.g., apiCall.js).
 *
 * @param {string} query - Query to match memories against
 * @param {number} limit - Maximum number of memories to return
 * @returns {Array} - Array of relevant memories
 */
export const retrieveRelevantMemories = (query, limit = 5) => { // Restored EXPORT
  try {
    // ... (Implementation from previous correct version) ...
     if (!query) return [];
     console.log(`🧠 [INFO] Retrieving memories locally for: ${query.substring(0, 50)}...`);
     const userProfilesStr = localStorage.getItem('user-profiles');
     if (!userProfilesStr) return [];

     let userProfile;
     try {
       const userProfiles = JSON.parse(userProfilesStr);
       if (!userProfiles || !userProfiles.profiles || !userProfiles.activeProfileId) return [];
       userProfile = userProfiles.profiles.find(p => p.id === userProfiles.activeProfileId);
       if (!userProfile) userProfile = userProfiles.profiles[0];
     } catch (e) { return []; }

     if (!userProfile || !userProfile.memories || !Array.isArray(userProfile.memories)) return [];

     const memories = userProfile.memories;
     if (memories.length === 0) return [];

     const queryWords = query.toLowerCase().split(/\W+/).filter(w => w.length > 2);

     const scoredMemories = memories
       .filter(m => m && m.content)
       .map(memory => {
         const content = memory.content.toLowerCase();
         const memoryWords = content.split(/\W+/).filter(w => w.length > 2);
         const matchingWords = queryWords.filter(word => memoryWords.includes(word));
         const wordScore = queryWords.length > 0 ? matchingWords.length / queryWords.length : 0;
         const relevanceScore = wordScore;
         return { ...memory, relevanceScore };
       })
       .filter(m => m.relevanceScore > 0.1)
       .sort((a, b) => b.relevanceScore - a.relevanceScore)
       .slice(0, limit);

     console.log(`🧠 [INFO] Found ${scoredMemories.length} relevant memories locally`);
     return scoredMemories;
  } catch (error) {
    console.error("🧠 [ERROR] Error retrieving memories locally:", error);
    return [];
  }
};

/**
 * This function is called by AppContext.jsx after /generate completes.
 * Its purpose now is solely to trigger the background sync.
 * Exported because AppContext.jsx imports it.
 *
 * @param {string} prompt - The user's message (may not be needed anymore)
 * @param {string} response - The AI's response (may not be needed anymore)
 * @param {string} currentUserName - The current user's name (may not be needed anymore)
 * @param {string} activeConversation - The active conversation ID (may not be needed anymore)
 * @param {object} memoryContext - The memory context from useMemory()
 * @returns {Promise<object>} - Result of the sync attempt
 */
export const observeConversation = async (prompt, response, currentUserName, activeConversation, memoryContext) => { // Restored EXPORT
  try {
    console.log("🧠 [INFO] observeConversation triggered: Initiating background sync check.");
    // Get activeProfileId from context safely
    const activeProfileId = memoryContext?.activeProfileId; // Indentation fixed
    // Pass activeProfileId to the sync function
    const success = await syncBackendMemoriesToLocalStorage(memoryContext, activeProfileId);
    if (success) {
        console.log("🧠 [INFO] observeConversation: Background sync completed successfully.");
    } else {
        console.error("🧠 [ERROR] observeConversation: Background sync failed.");
    }
    // Return status object for consistency with other functions

    return {
        status: success ? "sync_attempted_success" : "sync_attempted_failed",
        reason: "Automatic observation disabled, ran sync instead."
    };
  } catch (error) {
    console.error("🧠 [ERROR] Error in observeConversation wrapper (triggering sync):", error);
    return { status: "error", error: error.message };
  }
};


// --- Potentially Unused / Manual Trigger Only ---

/*
// Keep definition if manual purging is needed via UI, but it's not called automatically.
export const purgeBackendMemories = async () => {
   // ... (Implementation from previous version) ...
   try {
    console.warn("🧠 [WARN] Attempting to purge ALL backend memories...");
    const response = await fetch(`${MEMORY_API_URL}/memory/purge`, {
      method: 'POST'
    });
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`🧠 [ERROR] Failed to purge backend memories: ${response.status} - ${errorText}`);
      return false;
    }
    const result = await response.json();
    if (result.status === 'success') {
      console.log("🧠 [INFO] Successfully purged backend memories.");
      return true;
    } else {
      console.error("🧠 [ERROR] Purge command failed on backend:", result);
      return false;
    }
  } catch (error) {
    console.error("🧠 [ERROR] Error purging backend memories:", error);
    return false;
  }
};
*/