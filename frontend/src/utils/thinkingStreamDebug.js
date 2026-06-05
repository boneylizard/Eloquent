/**
 * Gated console logging for extended-thinking / :thinking model streams.
 * Enable via model id containing ":thinking", Settings → showReasoningDiagnostics,
 * or localStorage eloquent-thinking-stream-debug=1
 */

export function isThinkingStreamDebugEnabled({ modelName, settings } = {}) {
  if (settings?.showReasoningDiagnostics === true) return true;
  if (/:thinking/i.test(String(modelName || ''))) return true;
  try {
    if (localStorage.getItem('eloquent-thinking-stream-debug') === '1') return true;
  } catch {
    /* ignore */
  }
  return false;
}

/** @returns {(deltaText: string, deltaReasoning: string, extra?: object) => void} */
export function createThinkingStreamChunkLogger({ modelName, settings, label = 'sse' } = {}) {
  let count = 0;
  return (deltaText, deltaReasoning, extra = {}) => {
    if (!isThinkingStreamDebugEnabled({ modelName, settings })) return;
    if (count >= 3) return;
    count += 1;
    const text = String(deltaText || '');
    const reasoning = String(deltaReasoning || '');
    console.debug(`[think-stream:${label}] chunk #${count}`, {
      deltaTextLen: text.length,
      deltaReasoningLen: reasoning.length,
      deltaTextPreview: text.slice(0, 120),
      deltaReasoningPreview: reasoning.slice(0, 120),
      hasThinkOpenTag: /<think>|<thinking>/i.test(text + reasoning),
      ...extra,
    });
  };
}
