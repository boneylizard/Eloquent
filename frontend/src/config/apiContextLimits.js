/** Default total context budget (tokens) for API packing: system + rolling memory + history. */
export const API_CONTEXT_WINDOW_TOKENS_DEFAULT = 64000;

/** Floor used by the settings slider and `selectApiHistoryWithinContext`. */
export const API_CONTEXT_WINDOW_MIN = 8192;

/**
 * Ceiling for the API context window slider and runtime packing (~1M-token-class models).
 * Keep in sync with the Generation settings slider `max`.
 */
export const API_CONTEXT_WINDOW_MAX = 1048576;

/** Step for the API context window slider (balances precision vs. drag range). */
export const API_CONTEXT_WINDOW_SLIDER_STEP = 8192;

export function clampApiContextWindowTokens(raw) {
  let v = Number(raw);
  if (!Number.isFinite(v) || v <= 0) v = API_CONTEXT_WINDOW_TOKENS_DEFAULT;
  return Math.min(API_CONTEXT_WINDOW_MAX, Math.max(API_CONTEXT_WINDOW_MIN, v));
}

export function formatApiContextWindowShort(tokens) {
  const n = Math.max(0, Number(tokens) || 0);
  if (n >= 1_000_000) {
    const m = n / 1e6;
    const s = m >= 10 ? String(Math.round(m)) : m.toFixed(2).replace(/\.?0+$/, '');
    return `${s}M tokens`;
  }
  return `${Math.round(n / 1024)}k tokens`;
}
