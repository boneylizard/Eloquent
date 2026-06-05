// src/utils/cleanOutput.js

import { stripThinkTags } from './thinkStreamParser';

export function cleanModelOutput(text, isDualChat = false) {
  if (!text) return "";

  // Special handling for dual chat mode
  if (isDualChat) {
    return cleanDualChatOutput(text);
  }

  // For normal chat, return the raw text with minimal changes so we don't
  // accidentally strip headings, sections, or other useful content.
  return String(text);
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

  // 2. Remove thinking sections (same tag variants as chat stream parser)
  cleaned = stripThinkTags(cleaned);

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