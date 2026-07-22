/**
 * Normalize SSE JSON chunks from /generate into visible + reasoning text parts.
 */

const REASONING_FIELD_NAMES = [
  'reasoning', 'reasoning_content', 'thinking', 'reasoning_text',
  'reason', 'think', 'internal_monologue', 'chain_of_thought',
  'thought', 'thought_process',
];

function normalizeSseError(err) {
  if (err == null) return '';
  if (typeof err === 'string') return err.trim();
  if (typeof err === 'object' && err.message) return String(err.message).trim();
  return String(err).trim();
}

/** Scan a dict for reasoning/thinking content under known field names only. */
function extractReasoningFromDict(d) {
  if (!d || typeof d !== 'object') return '';

  // Providers sometimes stamp serving metadata (deployment ids, fingerprints)
  // onto stream chunks, so unknown fields are never treated as reasoning.
  for (const field of REASONING_FIELD_NAMES) {
    const val = d[field];
    if (typeof val === 'string' && val) return val;
    if (Array.isArray(val) && val.length) {
      const parts = [];
      for (const item of val) {
        if (item && typeof item === 'object') {
          const piece = item.text || item.content || '';
          if (piece) parts.push(String(piece));
        } else if (typeof item === 'string' && item) {
          parts.push(item);
        }
      }
      if (parts.length) return parts.join('');
    }
  }

  return '';
}

export function extractSseStreamParts(parsed) {
  if (!parsed || typeof parsed !== 'object') {
    return { text: '', reasoning: '', error: '', raw: parsed?.raw };
  }
  if (parsed.type === 'error') {
    const msg = normalizeSseError(parsed.error ?? parsed.message ?? parsed.detail);
    if (msg) return { text: '', reasoning: '', error: msg, raw: parsed.raw };
  }
  if (parsed.error != null) {
    const msg = normalizeSseError(parsed.error);
    if (msg) return { text: '', reasoning: '', error: msg, raw: parsed.raw };
  }

  // ── Top-level fields (NanoGPT, some proxies emit flat JSON) ──
  const topText = parsed.text != null ? String(parsed.text) : '';
  const topReasoning = extractReasoningFromDict(parsed);
  if (topText !== '' || topReasoning !== '') {
    return { text: topText, reasoning: topReasoning, error: '', raw: parsed.raw };
  }

  // ── choices-based format (OpenAI, OpenRouter, NanoGPT, etc.) ──
  const choice = parsed.choices?.[0];
  const choiceReasoning = choice && typeof choice === 'object'
    ? extractReasoningFromDict(choice)
    : '';
  const delta = choice?.delta;
  if (delta && typeof delta === 'object') {
    // NanoGPT reasoning fields: delta.reasoning or delta.reasoning_content
    // IMPORTANT: Check these BEFORE content extraction to ensure reasoning chunks
    // with empty content are properly captured
    let reasoning = extractReasoningFromDict(delta) || choiceReasoning;
    
    // Also check reasoning_details array (OpenRouter)
    if (!reasoning && Array.isArray(delta.reasoning_details)) {
      reasoning = delta.reasoning_details
        .map((d) => (d && (d.text || d.content)) || '')
        .join('');
    }
    
    // Content can be empty string when reasoning is present - don't skip!
    // Use explicit null/undefined check instead of falsy check
    const content = delta.content !== null && delta.content !== undefined 
      ? String(delta.content) 
      : '';
    
    // Return if we have either content or reasoning
    if (content !== '' || reasoning !== '') {
      return { text: content, reasoning: reasoning, error: '', raw: parsed.raw };
    }
  }
  if (choiceReasoning) {
    return { text: '', reasoning: choiceReasoning, error: '', raw: parsed.raw };
  }

  // ── message format (non-delta / chat-completion shape) ──
  const message = choice?.message;
  if (message && typeof message === 'object') {
    const content = message.content;
    const reasoning = extractReasoningFromDict(message);
    if (typeof content === 'string' && content) {
      return { text: content, reasoning: reasoning || '', error: '', raw: parsed.raw };
    }
    if (Array.isArray(content)) {
      const textParts = [];
      for (const item of content) {
        if (item && typeof item === 'object' && item.type === 'text' && item.text) {
          textParts.push(item.text);
        }
      }
      if (textParts.length) {
        return { text: textParts.join(''), reasoning: reasoning || '', error: '', raw: parsed.raw };
      }
    }
    // content may be null/undefined but reasoning present
    if (reasoning) {
      return { text: '', reasoning, error: '', raw: parsed.raw };
    }
  }

  return { text: '', reasoning: '', error: '', raw: parsed.raw };
}
