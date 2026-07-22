/**
 * SSE stream demultiplexer for /agentic/turn.
 *
 * Reads the multiplexed SSE stream and dispatches typed events to callbacks:
 *   analysis  → onAnalysis(state)
 *   tactile   → onTactile(payload)   — pose, gesture, proximity
 *   signal    → onSignal(payload)    — covert_action, voice_this_turn
 *   somatic   → onSomatic(payload)
 *   cipher    → onCipher({ block, phase })
 *   text      → onText(delta)
 *   done      → onDone(metadata)
 *   error     → onError({ step, detail })
 *
 * The /agentic/turn endpoint emits events in SSE format:
 *   event: <type>
 *   data: <json>
 *   <blank line>
 */

/**
 * Parse and demux an SSE stream from a fetch Response.
 *
 * @param {Response} response - The fetch Response object with a streaming body
 * @param {Object} callbacks - Event callbacks
 * @param {Function} callbacks.onAnalysis - Called with analysis result
 * @param {Function} callbacks.onTactile - Called with tactile outreach payload
 * @param {Function} callbacks.onSignal - Called with character signal payload
 * @param {Function} callbacks.onSomatic - Called with somatic payload
 * @param {Function} callbacks.onCipher - Called with { block, phase }
 * @param {Function} callbacks.onText - Called with text delta string
 * @param {Function} callbacks.onDone - Called with done metadata
 * @param {Function} callbacks.onError - Called with { step, detail }
 * @param {AbortSignal} signal - Optional abort signal to cancel the stream
 * @returns {Promise<void>} Resolves when the stream is complete
 */
export async function demuxAgenticStream(response, callbacks, signal) {
  const {
    onAnalysis = () => {},
    onTactile = () => {},
    onSignal = () => {},
    onSomatic = () => {},
    onCipher = () => {},
    onText = () => {},
    onDone = () => {},
    onError = () => {},
  } = callbacks || {};

  if (!response || !response.body) {
    onError({ step: 'connection', detail: 'No response body' });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = null;

  const dispatch = (eventType, dataStr) => {
    if (!eventType || !dataStr) return;

    let data;
    try {
      data = JSON.parse(dataStr);
    } catch {
      console.warn('[agenticStream] Failed to parse SSE data for event', eventType, dataStr);
      return;
    }

    switch (eventType) {
      case 'analysis':
        onAnalysis(data);
        break;
      case 'tactile':
        onTactile(data);
        break;
      case 'signal':
        onSignal(data);
        break;
      case 'somatic':
        onSomatic(data);
        break;
      case 'cipher':
        onCipher(data);
        break;
      case 'text':
        if (data.delta) onText(data.delta);
        break;
      case 'done':
        onDone(data);
        break;
      case 'error':
        onError(data);
        break;
      default:
        console.debug('[agenticStream] Unknown event type:', eventType);
    }
  };

  try {
    while (true) {
      if (signal?.aborted) {
        reader.cancel();
        break;
      }

      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages (separated by \n\n)
      let sepIdx;
      while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
        const message = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);

        if (!message.trim()) continue;

        let eventType = null;
        let dataLines = [];

        for (const line of message.split('\n')) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataLines.push(line.slice(6));
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5));
          }
        }

        if (eventType && dataLines.length) {
          dispatch(eventType, dataLines.join('\n'));
        }
      }
    }

    // Process any remaining buffered content
    if (buffer.trim()) {
      let eventType = null;
      let dataLines = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          dataLines.push(line.slice(6));
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5));
        }
      }
      if (eventType && dataLines.length) {
        dispatch(eventType, dataLines.join('\n'));
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      return; // Expected when cancelled
    }
    console.error('[agenticStream] Stream read error:', err);
    onError({ step: 'stream', detail: err.message || 'Stream read error' });
  }
}
