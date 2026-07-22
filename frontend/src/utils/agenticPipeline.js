/**
 * Agentic Pipeline orchestrator (frontend).
 *
 * Sends the SAME payload that sendMessage sends to /generate, plus the
 * agentic fields (cipher_block, character_id, dominance_bias, history).
 * The backend runs the SAME context assembly as /generate, then appends
 * the agentic directive block before streaming.
 *
 * The `text` event feeds the existing message rendering pipeline.
 */

import { demuxAgenticStream } from './agenticStream';
import { getBackendUrl } from '../config/api';

/**
 * Execute a full agentic turn.
 *
 * @param {Object} params
 * @param {string} params.apiUrl - Backend URL
 * @param {Object} params.payload - The SAME payload object that sendMessage builds for /generate
 *   (includes: prompt, model_name, max_tokens, temperature, top_p, top_k,
 *    repetition_penalty, directProfileInjection, authorNote, summaryContext,
 *    intensity_params, use_rag, rag_docs, userProfile, userProfileReinforcement,
 *    active_character, injectTimestamp, system_persona_mode, etc.)
 * @param {string} params.characterId - Character ID for shadow state
 * @param {string} params.userId - User ID for shadow state
 * @param {string} params.conversationId - Conversation ID
 * @param {Array} params.history - Full conversation history for Step 1 analysis
 * @param {Object} params.contexts - Sanctuary context hooks
 * @param {Function} params.getCipherForRequest - Returns cipher block for next turn
 * @param {Function} params.onTextDelta - Called for each text delta
 * @param {AbortSignal} params.signal - Optional abort signal
 * @returns {Promise<Object>} - { ok, turnMeta, error }
 */
export async function runAgenticTurn({
  apiUrl,
  payload,
  characterId = '',
  userId = '',
  conversationId = '',
  history = [],
  contexts = {},
  getCipherForRequest = () => null,
  onTextDelta = () => {},
  signal,
}) {
  const {
    applyAnalysis = () => {},
    applySomatic = () => {},
    storeCipher = () => {},
    applyDone = () => {},
    applyHijack = () => {},
    startTurn = () => {},
    addAnalysisReasoning = () => {},
    addSomaticLabel = () => {},
    addTactileEntry = () => {},
    addSignalEntry = () => {},
    addCipherGlyphs = () => {},
    endTurn = () => {},
  } = contexts;

  const baseUrl = apiUrl || getBackendUrl();
  const url = `${baseUrl}/agentic/turn`;

  const cipherBlock = getCipherForRequest();

  // The agentic payload = same /generate payload + agentic fields
  const agenticBody = {
    ...payload,
    // Agentic-specific fields
    character_id: characterId,
    cipher_block: cipherBlock,
    history,
  };

  startTurn();

  let turnMeta = null;
  let streamError = null;

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(agenticBody),
      signal,
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => response.statusText);
      throw new Error(`Agentic turn failed (${response.status}): ${errText}`);
    }

    await demuxAgenticStream(response, {
      onAnalysis: (data) => {
        applyAnalysis(data);
        addAnalysisReasoning(data);
      },
      onTactile: (data) => {
        addTactileEntry(data);
      },
      onSignal: (data) => {
        addSignalEntry(data);
      },
      onSomatic: (data) => {
        applySomatic(data);
        addSomaticLabel(data);
        if (data.interface_hijack) {
          applyHijack(data.interface_hijack, data.python_drive);
        }
      },
      onCipher: (data) => {
        storeCipher(data);
        addCipherGlyphs(data);
      },
      onText: (delta) => {
        onTextDelta(delta);
      },
      onDone: (meta) => {
        turnMeta = meta;
        applyDone(meta);
      },
      onError: (err) => {
        streamError = err;
        console.error('[agenticPipeline] Stream error:', err);
      },
    }, signal);

    endTurn();

    if (streamError) {
      return { ok: false, turnMeta: null, error: streamError };
    }

    return { ok: true, turnMeta, error: null };
  } catch (err) {
    endTurn();
    if (err.name === 'AbortError') {
      return { ok: false, turnMeta: null, error: null, aborted: true };
    }
    console.error('[agenticPipeline] Turn failed:', err);
    return { ok: false, turnMeta: null, error: err.message };
  }
}
