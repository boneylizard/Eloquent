// src/utils/cleanOutput.js

/**
 * Enhanced model output cleaning function that handles various patterns
 * commonly found in LLM outputs, including dual chat mode special patterns.
 * Significantly enhanced to prevent instruction leakage and meta-commentary.
 * 
 * @param {string} text - The raw model output text to clean
 * @param {boolean} isDualChat - Whether this is from a dual chat conversation
 * @return {string} - The cleaned text
 */
export function cleanModelOutput(text, isDualChat = false) {
  if (!text) return "";
  
  // Special handling for dual chat mode
  if (isDualChat) {
    return cleanDualChatOutput(text);
  }
  
  // --- Standard cleaning for normal mode ---
  
  // 1. Handle common meta-commentary and instruction leakage at the beginning
  const beginningPatterns = [
    // Meta-commentary about completing responses
    /^(?:In order|To indicate|To signal|To let you know|To show|To mark|To ensure) .*?(?:completion|finished|complete|end|done)[^.]*\./i,
    /^(?:I'll|I will) (?:now|just|simply) (?:respond|answer|provide)[^.]*\./i,
    /^(?:I hope|I trust|I believe) (?:this|that|my|the) (?:response|answer|information|explanation) (?:is|was|has been)[^.]*\./i,
    
    // Phrases that acknowledge instructions
    /^As (?:instructed|requested|per your request|mentioned in the instructions)[^.]*\./i,
    /^Following (?:your|the) instructions[^.]*\./i,
    /^According to (?:your|the) instructions[^.]*\./i,
    
    // Stock response beginnings
    /^(?:I'd be happy to|I'll help you with that|Sure,? (?:I can|here's|let me)|Certainly)[^.]*\./i,
    /^(?:Here is|Here's|Below is|The following is)[^.]*\./i,
    /^(?:Let me|I will|I'll) (?:provide|give|share|offer)[^.]*\./i,
    
    // Assistant prefixes
    /^(###\s*)?((assistant|ai|bot|claude|llm|chatbot|gpt|model):\s*)/i,
    /^\*\*\s*assistant\s*\*\*\s*:/i
  ];
  
  // Apply each beginning pattern replacement
  let cleaned = text;
  for (const pattern of beginningPatterns) {
    const match = cleaned.match(pattern);
    if (match) {
      const restIndex = cleaned.indexOf('.', match[0].length - 1);
      if (restIndex > 0) {
        cleaned = cleaned.substring(restIndex + 1).trim();
      }
    }
  }
  
  // 2. Handle mid-text assistant interruptions and meta-commentary
  // These regex patterns target common formats of assistant interruptions
  const midTextPatterns = [
    /\n\s*\*\*\s*assistant\s*\*\*\s*:/gi,  // Bold with newline: \n**Assistant:**
    /\n\s*\*\s*assistant\s*\*\s*:/gi,      // Italic with newline: \n*Assistant:*
    /\n\s*(###)?\s*(assistant|ai|bot|claude|llm|chatbot|gpt|model):\s*/gi, // Plain or with ###
    /\n\s*\[\s*(assistant|ai|bot|claude|llm|chatbot|gpt|model)\s*\]:\s*/gi, // [Assistant]:
    // Mid-text meta-commentary
    /\n\s*(?:to signal|to indicate|to mark|to show|to let you know|to ensure) (?:completion|that I'm done|I've finished|the end)[^.]*\./gi
  ];
  
  // Apply each pattern replacement
  midTextPatterns.forEach(pattern => {
    cleaned = cleaned.replace(pattern, "\n");
  });
  
  // 3. Truncate at any user/human markers (indicating a new turn)
  const userMarkers = [
    /###\s*user:/i,
    /\n+user:/i,
    /\n+human:/i,
    /<user>/i,
    /\n+---\s*\n+/,  // Section dividers often used to separate turns
    /\n*\[user\]:/i,
    /\*\*user\*\*:/i  // Bold markdown user marker
  ];
  
  for (const marker of userMarkers) {
    const match = cleaned.match(marker);
    if (match && match.index !== undefined) {
      cleaned = cleaned.substring(0, match.index).trim();
    }
  }
  
  // 4. Remove trailing continuation markers and meta-commentary endings
  const trailingPatterns = [
    // Standard continuation markers
    /###\.\.\.$/m,                       // "###..."
    /\.\.\.$/m,                          // "..."
    /###\s*$/m,                          // "###" with optional whitespace
    /continue\s*(>>>|→|—|-->|→→→)$/i,    // "continue >>>" or similar
    /\[continue\]$/i,                    // "[continue]"
    
    // Meta-commentary endings
    /\s*(?:Is there anything else|Do you need|Let me know|Hope this helps|If you have any).*?$/i,
    /\s*Please feel free to.*?$/i,
    /\s*Don't hesitate to.*?$/i,
    /\s*I'm here to.*?$/i,
    /\s*(?:to indicate|to signal|to mark|to show) (?:completion|the end|that I'm done|I've finished).*?$/i
  ];
  
  trailingPatterns.forEach(pattern => {
    cleaned = cleaned.replace(pattern, "");
  });
  
  // 5. Remove XML-like tags that some models generate
  cleaned = cleaned.replace(/<\/?assistant>|<\/?ai>|<\/?response>|<\/?answer>|<\/?completion>/g, "");
  
  // 6. Normalize whitespace - collapse multiple newlines to max 2
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n");
  
  // 7. Trim leading/trailing whitespace
  cleaned = cleaned.trim();
  
  return cleaned;
}

/**
 * Special cleaning function for dual chat mode outputs.
 * Handles the unique patterns and issues in model-to-model conversation.
 * Enhanced with additional patterns to prevent instruction leakage.
 */
function cleanDualChatOutput(text) {
  if (!text) return "";
  
  // 1. Remove control phrases that keep appearing
  let cleaned = text;
  
  // Remove meta-commentary and instruction leakage more aggressively
  const metaCommentaryPatterns = [
    // Beginning patterns
    /^\s*to\s+(signal|indicate|finish|return|mark|show|let).*?(response|answer|end|complete|stop|control|finished)[^.]*\./i,
    /^\s*I am responding to your request[^.]*\./i,
    /^\s*In response to your query[^.]*\./i,
    /^\s*(?:I'll|I will) (?:now|just|simply) (?:respond|answer|provide)[^.]*\./i,
    /^\s*(?:I hope|I trust|I believe) (?:this|that|my|the) (?:response|answer|information|explanation) (?:is|was|has been)[^.]*\./i,
    /^\s*(?:I'd be happy to|I'll help you with that|Sure,? (?:I can|here's|let me)|Certainly)[^.]*\./i,
    /^\s*(?:Here is|Here's|Below is|The following is)[^.]*\./i,
    
    // Mid-text patterns
    /\s+to\s+(signal|indicate|finish|return|mark|show|let).*?(response|answer|end|complete|stop|control|finished)[^.]*\./gi,
    /\s+This (?:answer|response) (?:is|was) provided to (?:help|address|clarify)[^.]*\./gi,
    /\s+I have (?:completed|finished|prepared) my (?:response|answer|analysis)[^.]*\./gi
  ];
  
  // Apply all patterns
  for (const pattern of metaCommentaryPatterns) {
    const match = cleaned.match(pattern);
    if (match) {
      // If it's at the beginning, remove the whole sentence
      if (pattern.toString().includes('^')) {
        const restIndex = cleaned.indexOf('.', match[0].length - 1);
        if (restIndex > 0) {
          cleaned = cleaned.substring(restIndex + 1).trim();
        }
      } else {
        // For mid-text patterns, just remove the matches
        cleaned = cleaned.replace(pattern, " ");
      }
    }
  }
  
  // 2. Remove thinking sections
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/g, "");
  
  // 3. Remove The Assistant markers with enhanced patterns
  const assistantMarkerPatterns = [
    /\s*(\|?—\s*The Assistant\s*\|?|\(The Assistant\)|\(?\|?—.*?Assistant.*?\|?\)?)\s*/g,
    /\s*(\|?—\s*The AI\s*\|?|\(The AI\)|\(?\|?—.*?AI.*?\|?\)?)\s*/g,
    /\s*(\|?—\s*The Model\s*\|?|\(The Model\)|\(?\|?—.*?Model.*?\|?\)?)\s*/g
  ];
  
  assistantMarkerPatterns.forEach(pattern => {
    cleaned = cleaned.replace(pattern, "\n\n");
  });
  
  // 4. Remove generic endings that plague dual chat
  const genericEndings = [
    /Feel free to share any.*?I may have missed\.$/i,
    /Let me know if you have any.*?$/i,
    /What are your thoughts on this\?$/i,
    /Would you like me to elaborate.*?$/i,
    /I'd be happy to discuss.*?$/i,
    /I hope that helps.*?$/i,
    /I hope this.*?useful.*?$/i,
    /I'm here if you need.*?$/i,
    /Is there anything else.*?$/i
  ];
  
  genericEndings.forEach(pattern => {
    cleaned = cleaned.replace(pattern, "");
  });
  
  // 5. Ensure proper formatting
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n"); // Normalize excessive newlines
  cleaned = cleaned.trim();
  
  return cleaned;
}

/**
 * Extended API that selects the appropriate cleaning method
 * and adds improved handling for dual chat mode.
 */
export function processDualChatOutput(text, modelName) {
  if (!text) return "";
  
  // First apply the dual chat cleaning
  let processed = cleanDualChatOutput(text);
  
  // Ensure there's a sensible ending if needed
  if (!processed.match(/[.!?]\s*$/)) {
    const lastChar = processed.slice(-1);
    // Only add period if it doesn't end with punctuation
    if (lastChar !== "." && lastChar !== "!" && lastChar !== "?") {
      processed += ".";
    }
  }
  
  return processed;
}

/**
 * Function to strip meta-commentary ONLY, preserving other formatting.
 * Useful when you want to keep most of the structure but just remove meta-commentary.
 */
export function stripMetaCommentary(text) {
  if (!text) return "";
  
  const metaPatterns = [
    // Beginning meta-commentary
    /^(?:In order|To indicate|To signal|To let you know|To show|To mark|To ensure) .*?(?:completion|finished|complete|end|done)[^.]*\./i,
    /^(?:I'll|I will) (?:now|just|simply) (?:respond|answer|provide)[^.]*\./i,
    /^As (?:instructed|requested|per your request|mentioned in the instructions)[^.]*\./i,
    
    // Mid-text meta-commentary
    /\s+(?:to signal|to indicate|to mark|to show|to let you know|to ensure) (?:completion|that I'm done|I've finished|the end)[^.]*\./gi,
    
    // Ending meta-commentary
    /\s*(?:In conclusion|To sum up|To summarize|In summary),.*?(?:to signal|to indicate|to mark|to show) (?:completion|the end|that I'm done|I've finished).*?$/i,
    /\s*This (?:answer|response) has been provided to (?:help|address|clarify).*?$/i,
    /\s*I have (?:completed|finished|prepared) my (?:response|answer|analysis).*?$/i
  ];
  
  let cleaned = text;
  
  // Apply all patterns
  for (const pattern of metaPatterns) {
    if (pattern.toString().includes('^')) {
      // Beginning pattern
      const match = cleaned.match(pattern);
      if (match) {
        const restIndex = cleaned.indexOf('.', match[0].length - 1);
        if (restIndex > 0) {
          cleaned = cleaned.substring(restIndex + 1).trim();
        }
      }
    } else if (pattern.toString().includes('$')) {
      // Ending pattern
      cleaned = cleaned.replace(pattern, "");
    } else {
      // Mid-text pattern
      cleaned = cleaned.replace(pattern, " ");
    }
  }
  
  return cleaned.trim();
}