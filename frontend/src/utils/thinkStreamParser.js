/**

 * Streaming-safe parser for NanoGPT-style reasoning tags.

 *

 * Goal: split a token stream into:

 * - visible text (rendered in the assistant bubble)

 * - reasoning text (shown in collapsible "Reasoning" block)

 *

 * Handles tags split across chunks by keeping a small carry buffer.

 */

export function createThinkStreamParser() {

  return {

    inThink: false,

    carry: '',

    visible: '',

    reasoning: '',

  };

}



/** Tag pairs seen from NanoGPT / GLM / DeepSeek-style streams */

const THINK_TAG_VARIANTS = [

  { open: '<think>', close: '</think>' },

  { open: '<thinking>', close: '</thinking>' },

];



function findNextTag(input, fromIndex, inThink) {

  let bestIdx = -1;

  let best = null;

  for (const tags of THINK_TAG_VARIANTS) {

    if (!tags.open || !tags.close) continue;

    const needle = inThink ? tags.close : tags.open;

    const idx = input.indexOf(needle, fromIndex);

    if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {

      bestIdx = idx;

      best = { ...tags, needle, isClose: inThink };

    }

  }

  return bestIdx === -1 ? null : { index: bestIdx, ...best };

}



function tagPrefixMatches(possible) {

  if (!possible) return false;

  return THINK_TAG_VARIANTS.some(

    (t) => t.open.startsWith(possible) || t.close.startsWith(possible),

  );

}



/** True when a string contains a complete think open/close tag. */

export function contentHasThinkTags(text) {

  const s = String(text || '');

  if (!s) return false;

  return THINK_TAG_VARIANTS.some((t) => s.includes(t.open) || s.includes(t.close));

}



/**

 * @param {ReturnType<typeof createThinkStreamParser>} state

 * @param {string} chunk

 * @returns {{ visibleDelta: string, reasoningDelta: string }}

 */

export function consumeThinkStreamChunk(state, chunk) {

  const input = String(state.carry || '') + String(chunk || '');

  state.carry = '';



  let i = 0;

  let visibleDelta = '';

  let reasoningDelta = '';



  while (i < input.length) {

    const tag = findNextTag(input, i, state.inThink);



    if (!state.inThink) {

      if (!tag) {

        visibleDelta += input.slice(i);

        i = input.length;

        break;

      }

      visibleDelta += input.slice(i, tag.index);

      state.inThink = true;

      i = tag.index + tag.open.length;

      continue;

    }



    if (!tag) {

      reasoningDelta += input.slice(i);

      i = input.length;

      break;

    }

    reasoningDelta += input.slice(i, tag.index);

    state.inThink = false;

    i = tag.index + tag.close.length;

  }



  const maxCarry = 24;

  if (!state.inThink) {

    const suffix = visibleDelta.slice(-maxCarry);

    const lastLt = suffix.lastIndexOf('<');

    if (lastLt !== -1) {

      const possible = suffix.slice(lastLt);

      if (tagPrefixMatches(possible)) {

        state.carry = possible;

        visibleDelta = visibleDelta.slice(0, visibleDelta.length - possible.length);

      }

    }

  } else {

    const suffix = reasoningDelta.slice(-maxCarry);

    const lastLt = suffix.lastIndexOf('<');

    if (lastLt !== -1) {

      const possible = suffix.slice(lastLt);

      if (tagPrefixMatches(possible)) {

        state.carry = possible;

        reasoningDelta = reasoningDelta.slice(0, reasoningDelta.length - possible.length);

      }

    }

  }



  state.visible += visibleDelta;

  state.reasoning += reasoningDelta;



  return { visibleDelta, reasoningDelta };

}



export function finalizeThinkStream(state) {

  if (state.carry) {

    if (state.inThink) state.reasoning += state.carry;

    else state.visible += state.carry;

    state.carry = '';

  }



  return {

    visible: state.visible,

    reasoning: state.reasoning,

    inThink: state.inThink,

  };

}



/**

 * Parse a complete assistant string into visible + reasoning segments.

 * Uses the same tag rules as the streaming parser (tags may appear anywhere).

 */

export function parseThinkContent(text) {

  const input = text == null ? '' : String(text);

  if (!input) {

    return { visible: '', reasoning: '' };

  }

  const state = createThinkStreamParser();

  consumeThinkStreamChunk(state, input);

  const fin = finalizeThinkStream(state);

  return {

    visible: fin.visible.trimStart(),

    reasoning: fin.reasoning,

  };

}



/** Visible text only — for TTS, snippets, and legacy call sites. */

export function stripThinkTags(text) {

  return parseThinkContent(text).visible;

}



/**

 * Resolve bubble + reasoning display for a stored/streamed bot message.

 * Prefer stream-parsed reasoningText when present; otherwise extract from content.

 */

export function resolveMessageThinkDisplay(rawContent, existingReasoningText) {

  const parsed = parseThinkContent(rawContent);

  const hasExisting =

    typeof existingReasoningText === 'string' && existingReasoningText.trim().length > 0;

  return {

    visibleContent: parsed.visible,

    reasoningText: hasExisting ? existingReasoningText : parsed.reasoning,

  };

}



/**

 * Before render / IDB load: move inline think tags from content → reasoningText.

 * @param {object} message

 * @returns {object}

 */

/** Merge streaming / finalize reasoning metadata onto a bot message. */
export function applyReasoningMetaToBotMessage(message, meta = {}, { capReasoning = false } = {}) {
  if (!message) return message;
  const reasoningEnabled =
    meta.reasoningEnabled === true ||
    message.reasoningEnabled === true ||
    capReasoning === true;
  const patch = {
    ...message,
    reasoningEnabled,
    reasoningStreaming: meta.reasoningStreaming === true,
  };
  if (meta.reasoningStartedAtMs != null) {
    patch.reasoningStartedAtMs = meta.reasoningStartedAtMs;
  } else if (!reasoningEnabled) {
    patch.reasoningStartedAtMs = null;
  }
  if (meta.reasoningSeconds != null) {
    patch.reasoningSeconds = meta.reasoningSeconds;
  }
  if (typeof meta.reasoningText === 'string') {
    patch.reasoningText = meta.reasoningText;
  }
  if (meta.reasoningCapabilitySource) {
    patch.reasoningCapabilitySource = meta.reasoningCapabilitySource;
  }
  return patch;
}

/** Fields to persist on a bot message after the reasoning stream controller finalizes. */
export function buildBotReasoningFinalizePatch(finStream) {
  const reasoningEnabled = finStream?.reasoningEnabled === true;
  const reasoningText = reasoningEnabled ? String(finStream.reasoning || '').trim() : '';
  const reasoningSeconds =
    finStream?.reasoningSeconds != null
      ? finStream.reasoningSeconds
      : reasoningEnabled && finStream?.reasoningStartedAtMs != null
        ? Math.max(0, Math.round((Date.now() - finStream.reasoningStartedAtMs) / 1000))
        : null;
  const patch = {
    reasoningEnabled,
    reasoningStreaming: false,
    reasoningStartedAtMs: reasoningEnabled ? (finStream.reasoningStartedAtMs ?? null) : null,
    reasoningSeconds: reasoningEnabled ? reasoningSeconds : null,
    reasoningText,
  };
  if (finStream?.reasoningCapabilitySource) {
    patch.reasoningCapabilitySource = finStream.reasoningCapabilitySource;
  }
  return patch;
}

export function hydrateBotMessageThinkFields(message) {

  if (!message || message.role !== 'bot') return message;



  const existingReasoning =

    typeof message.reasoningText === 'string' && message.reasoningText.trim().length > 0;



  const rawContent = message.content == null ? '' : String(message.content);

  if (!rawContent) {
    if (!existingReasoning) return message;
    const next = { ...message };
    let changed = false;
    if (next.reasoningEnabled !== true) {
      next.reasoningEnabled = true;
      changed = true;
    }
    return changed ? next : message;
  }



  if (existingReasoning) {

    const parsed = parseThinkContent(rawContent);

    if (parsed.visible === rawContent) {
      if (message.reasoningEnabled === true && message.reasoningCapabilitySource) {
        return message;
      }
      const next = { ...message, reasoningEnabled: true };
      if (!next.reasoningCapabilitySource && contentHasThinkTags(rawContent)) {
        next.reasoningCapabilitySource = 'inline';
      }
      return next;
    }

    return {

      ...message,

      content: parsed.visible,

      reasoningEnabled: message.reasoningEnabled === true || Boolean(parsed.reasoning.trim()),

      reasoningCapabilitySource:
        message.reasoningCapabilitySource ||
        (contentHasThinkTags(rawContent) ? 'inline' : undefined),

    };

  }



  if (!contentHasThinkTags(rawContent)) return message;



  const parsed = parseThinkContent(rawContent);

  if (!parsed.reasoning.trim()) return message;



  return {

    ...message,

    content: parsed.visible,

    reasoningText: parsed.reasoning,

    reasoningEnabled: true,

    reasoningCapabilitySource: 'inline',

  };

}



/** Hydrate all bot messages in a conversation shard. */

export function hydrateMessagesThinkFields(messages) {

  if (!Array.isArray(messages)) return [];

  return messages.map(hydrateBotMessageThinkFields);

}



/**

 * Streaming state machine for CAP reasoning + inline &lt;redacted_thinking&gt; tags in content.

 * Dedicated delta.reasoning is appended separately; content deltas run through the tag parser.

 */

export function createReasoningStreamController({

  capReasoning = false,

  modelImpliesReasoning = false,

  debugThinking = false,

} = {}) {

  const thinkState = createThinkStreamParser();

  let reasoningEnabled = Boolean(capReasoning || modelImpliesReasoning);

  let inlineThinkDetected = false;

  let providerReasoningDetected = false;

  let loggedInlineThink = false;

  const resolveCapabilitySource = () => {

    if (inlineThinkDetected) return 'inline';

    if (providerReasoningDetected) return 'provider';

    return null;

  };

  let thinkBlockStartedAtMs = null;

  let reasoningSeconds = null;

  let reasoningText = '';

  let visibleText = '';



  const processChunk = ({ deltaText = '', deltaReasoning = '' } = {}) => {

    let reasoningStreaming = false;

    const dedicatedReasoning = String(deltaReasoning || '');



    if (dedicatedReasoning) {

      thinkBlockStartedAtMs = thinkBlockStartedAtMs ?? Date.now();

      reasoningText += dedicatedReasoning;

      reasoningEnabled = true;

      reasoningStreaming = true;

      if (!providerReasoningDetected) {

        providerReasoningDetected = true;

        if (debugThinking) {

          console.debug('[think-stream] provider reasoning delta', {

            len: dedicatedReasoning.length,

            preview: dedicatedReasoning.slice(0, 120),

          });

        }

      }

    }

    const chunkText = String(deltaText || '');

    if (!chunkText) {

      return {

        visibleDelta: '',

        reasoningDelta: dedicatedReasoning,

        reasoningEnabled,

        reasoningStreaming,

        inlineThinkDetected,

        reasoningCapabilitySource: resolveCapabilitySource(),

        reasoningSeconds,

        reasoningStartedAtMs: thinkBlockStartedAtMs,

        reasoningText,

        visibleText,

        inThink: thinkState.inThink,

      };

    }



    const wasInThink = thinkState.inThink;

    const { visibleDelta, reasoningDelta } = consumeThinkStreamChunk(thinkState, chunkText);



    if (reasoningDelta) {

      reasoningText += reasoningDelta;

      reasoningEnabled = true;

      inlineThinkDetected = true;

      if (debugThinking && !loggedInlineThink) {

        loggedInlineThink = true;

        console.debug('[think-stream] inlineThinkDetected', {

          reasoningDeltaLen: reasoningDelta.length,

          preview: reasoningDelta.slice(0, 120),

        });

      }

    }

    if (visibleDelta) {

      visibleText += visibleDelta;

    }



    if (!wasInThink && thinkState.inThink) {

      thinkBlockStartedAtMs = thinkBlockStartedAtMs ?? Date.now();

    }

    if (wasInThink && !thinkState.inThink && thinkBlockStartedAtMs != null) {

      reasoningSeconds = Math.max(

        0,

        Math.round((Date.now() - thinkBlockStartedAtMs) / 1000),

      );

      thinkBlockStartedAtMs = null;

    }



    if (thinkState.inThink) {

      reasoningStreaming = true;

    } else if (!dedicatedReasoning) {

      reasoningStreaming = false;

    }

    if (!reasoningStreaming && thinkBlockStartedAtMs != null) {

      reasoningSeconds = Math.max(

        0,

        Math.round((Date.now() - thinkBlockStartedAtMs) / 1000),

      );

      thinkBlockStartedAtMs = null;

    }



    return {

      visibleDelta,

      reasoningDelta: [reasoningDelta, dedicatedReasoning].filter(Boolean).join(''),

      reasoningEnabled,

      reasoningStreaming,

      inlineThinkDetected,

      reasoningCapabilitySource: resolveCapabilitySource(),

      reasoningSeconds,

      reasoningStartedAtMs: thinkBlockStartedAtMs,

      reasoningText,

      visibleText,

      inThink: thinkState.inThink,

    };

  };



  const finalize = () => {

    const fin = finalizeThinkStream(thinkState);

    if (fin.visible) {

      visibleText = fin.visible;

    }

    if (fin.reasoning) {

      reasoningText = [reasoningText, fin.reasoning].filter(Boolean).join('');

      reasoningEnabled = true;

      inlineThinkDetected = inlineThinkDetected || Boolean(String(fin.reasoning).trim());

    }

    if (thinkState.inThink && thinkBlockStartedAtMs != null) {

      reasoningSeconds = Math.max(

        0,

        Math.round((Date.now() - thinkBlockStartedAtMs) / 1000),

      );

    }

    return {

      visible: visibleText,

      reasoning: reasoningText,

      reasoningEnabled,

      inlineThinkDetected,

      reasoningCapabilitySource: resolveCapabilitySource(),

      reasoningSeconds,

      reasoningStartedAtMs: thinkBlockStartedAtMs,

      inThink: thinkState.inThink,

    };

  };



  return {

    processChunk,

    finalize,

    getThinkState: () => thinkState,

  };

}


