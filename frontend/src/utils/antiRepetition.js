/**
 * Anti-Repetition Utilities
 * Detects and removes repeated phrases across LLM responses
 */

/**
 * Find repeated phrases between a new response and previous responses
 * @param {string} newResponse - The latest response to check
 * @param {string[]} previousResponses - Array of previous assistant responses
 * @param {number} minPhraseWords - Minimum words to consider a phrase (default: 6)
 * @returns {string[]} Array of repeated phrases found
 */
export function findRepeatedPhrases(newResponse, previousResponses, minPhraseWords = 6) {
  if (!newResponse || !previousResponses || previousResponses.length === 0) {
    return [];
  }

  const repeatedPhrases = new Set();
  const newWords = newResponse.toLowerCase().split(/\s+/);
  
  // Only check last 5 responses to avoid over-filtering
  const recentResponses = previousResponses.slice(-5);
  
  for (const prevResponse of recentResponses) {
    if (!prevResponse) continue;
    
    const prevText = prevResponse.toLowerCase();
    
    // Sliding window to find matching phrases
    for (let i = 0; i <= newWords.length - minPhraseWords; i++) {
      // Try different phrase lengths from minPhraseWords to minPhraseWords + 10
      for (let len = minPhraseWords; len <= Math.min(minPhraseWords + 10, newWords.length - i); len++) {
        const phrase = newWords.slice(i, i + len).join(' ');
        
        // Skip very common phrases that are okay to repeat
        if (isCommonPhrase(phrase)) continue;
        
        // Check if this phrase exists in previous response
        if (prevText.includes(phrase)) {
          repeatedPhrases.add(phrase);
          break; // Don't add longer versions of the same phrase
        }
      }
    }
  }
  
  return Array.from(repeatedPhrases);
}

/**
 * Common phrases that are okay to repeat (greetings, connectors, etc.)
 */
const COMMON_PHRASES = new Set([
  'i think',
  'i believe',
  'you know',
  'of course',
  'in fact',
  'for example',
  'such as',
  'as well',
  'at least',
  'at first',
  'so far',
  'right now',
  'thank you',
  'you\'re welcome',
  'i understand',
  'let me',
  'i can',
  'i will',
  'would you',
  'could you',
]);

function isCommonPhrase(phrase) {
  // Check if it's a common acceptable phrase
  for (const common of COMMON_PHRASES) {
    if (phrase.includes(common)) return true;
  }
  
  // Skip very short or generic phrases
  if (phrase.split(' ').length < 5) return true;
  
  return false;
}

/**
 * Remove repeated phrases from a response
 * @param {string} response - The response to clean
 * @param {string[]} repeatedPhrases - Phrases to remove
 * @returns {string} Cleaned response
 */
export function removeRepeatedPhrases(response, repeatedPhrases) {
  if (!repeatedPhrases || repeatedPhrases.length === 0) {
    return response;
  }
  
  let cleaned = response;
  
  // Sort by length (longest first) to avoid partial replacements
  const sortedPhrases = [...repeatedPhrases].sort((a, b) => b.length - a.length);
  
  for (const phrase of sortedPhrases) {
    // Case-insensitive replacement
    const regex = new RegExp(escapeRegex(phrase), 'gi');
    cleaned = cleaned.replace(regex, '');
  }
  
  // Clean up any resulting double spaces, double newlines, etc.
  cleaned = cleaned
    .replace(/\s{3,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^\s+/gm, '') // Remove leading whitespace from lines
    .replace(/\s+$/gm, '') // Remove trailing whitespace from lines
    .trim();
  
  return cleaned;
}

/**
 * Escape special regex characters in a string
 */
function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Detect if a response has significant boilerplate at start/end
 * @param {string} response - The response to check
 * @param {string[]} previousResponses - Previous responses to compare against
 * @returns {{ hasBoilerplate: boolean, boilerplateStart: string|null, boilerplateEnd: string|null }}
 */
export function detectBoilerplate(response, previousResponses) {
  if (!previousResponses || previousResponses.length < 2) {
    return { hasBoilerplate: false, boilerplateStart: null, boilerplateEnd: null };
  }
  
  const sentences = response.split(/(?<=[.!?])\s+/);
  if (sentences.length < 3) {
    return { hasBoilerplate: false, boilerplateStart: null, boilerplateEnd: null };
  }
  
  const firstSentence = sentences[0].toLowerCase().trim();
  const lastSentence = sentences[sentences.length - 1].toLowerCase().trim();
  
  let boilerplateStart = null;
  let boilerplateEnd = null;
  
  // Check if first/last sentences appear in multiple previous responses
  let startCount = 0;
  let endCount = 0;
  
  for (const prev of previousResponses.slice(-5)) {
    if (!prev) continue;
    const prevLower = prev.toLowerCase();
    
    if (prevLower.includes(firstSentence) && firstSentence.length > 20) {
      startCount++;
    }
    if (prevLower.includes(lastSentence) && lastSentence.length > 20) {
      endCount++;
    }
  }
  
  if (startCount >= 2) {
    boilerplateStart = sentences[0];
  }
  if (endCount >= 2) {
    boilerplateEnd = sentences[sentences.length - 1];
  }
  
  return {
    hasBoilerplate: boilerplateStart !== null || boilerplateEnd !== null,
    boilerplateStart,
    boilerplateEnd
  };
}

/**
 * Process a response to remove detected repetition
 * @param {string} response - The new response
 * @param {string[]} previousResponses - Previous assistant responses
 * @param {object} options - Processing options
 * @returns {{ cleanedResponse: string, removedPhrases: string[], hadBoilerplate: boolean }}
 */
export function processAntiRepetition(response, previousResponses, options = {}) {
  const {
    removeBoilerplate = true,
    removePhrases = true,
    minPhraseWords = 6
  } = options;
  
  let cleanedResponse = response;
  let removedPhrases = [];
  let hadBoilerplate = false;
  
  // Step 1: Detect and optionally remove boilerplate sentences
  if (removeBoilerplate) {
    const boilerplate = detectBoilerplate(response, previousResponses);
    hadBoilerplate = boilerplate.hasBoilerplate;
    
    if (boilerplate.boilerplateStart) {
      cleanedResponse = cleanedResponse.replace(boilerplate.boilerplateStart, '').trim();
      removedPhrases.push(`[Start] ${boilerplate.boilerplateStart.substring(0, 50)}...`);
    }
    if (boilerplate.boilerplateEnd) {
      cleanedResponse = cleanedResponse.replace(boilerplate.boilerplateEnd, '').trim();
      removedPhrases.push(`[End] ${boilerplate.boilerplateEnd.substring(0, 50)}...`);
    }
  }
  
  // Step 2: Find and remove repeated phrases
  if (removePhrases) {
    const repeated = findRepeatedPhrases(cleanedResponse, previousResponses, minPhraseWords);
    if (repeated.length > 0) {
      cleanedResponse = removeRepeatedPhrases(cleanedResponse, repeated);
      removedPhrases.push(...repeated.map(p => p.substring(0, 50) + (p.length > 50 ? '...' : '')));
    }
  }
  
  return {
    cleanedResponse,
    removedPhrases,
    hadBoilerplate
  };
}

export default {
  findRepeatedPhrases,
  removeRepeatedPhrases,
  detectBoilerplate,
  processAntiRepetition
};

