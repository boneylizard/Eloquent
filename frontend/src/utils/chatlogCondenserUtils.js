/** Shared helpers for Chatlog condenser panels. */

export function normalizeEndpointModelId(id) {
  if (!id || typeof id !== 'string') return id;
  if (id.startsWith('endpoint-endpoint-')) {
    return id.replace(/^endpoint-endpoint-/, 'endpoint-');
  }
  return id;
}

export function formatApiError(data, statusText) {
  const d = data?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((x) => {
        if (typeof x !== 'object' || x == null) return String(x);
        const loc = Array.isArray(x.loc) ? x.loc.filter((p) => p !== 'body').join('.') : '';
        const prefix = loc ? `${loc}: ` : '';
        return prefix + (x.msg || JSON.stringify(x));
      })
      .join('; ');
  }
  if (d && typeof d === 'object') return JSON.stringify(d);
  return statusText || 'Request failed';
}

export const CHUNK_TOKENS_DEFAULT = 16000;
export const CHARS_PER_TOKEN = 4;

export function normalizeCondenseParams({ targetRatio, chunkTokens, overlapTurns }) {
  let ratio = Number(targetRatio);
  if (!Number.isFinite(ratio)) ratio = 0.4;
  else if (ratio > 1) ratio = ratio / 100;
  if (ratio <= 0) ratio = 0.4;

  let chunk = Math.round(Number(chunkTokens));
  if (!Number.isFinite(chunk) || chunk < 1) chunk = CHUNK_TOKENS_DEFAULT;

  let overlap = Math.round(Number(overlapTurns));
  if (!Number.isFinite(overlap) || overlap < 0) overlap = 5;

  return { target_ratio: ratio, chunk_target_tokens: chunk, overlap_turns: overlap };
}

export function speakerLabel(msg) {
  const r = msg?.role;
  if (r === 'user') return 'User';
  if (r === 'assistant' || r === 'bot') return 'Assistant';
  return r ? String(r) : 'Message';
}

/** Prefer longer draft text (server vs local) so Stop does not wipe in-flight output. */
export function pickLongerDraft(...parts) {
  let best = '';
  for (const p of parts) {
    const s = String(p ?? '').trim();
    if (s.length > best.length) best = s;
  }
  return best;
}

/** Keep optimistic UI messages when a refresh races an in-flight send. */
export function mergeSessionMessages(localMessages, serverMessages) {
  const local = Array.isArray(localMessages) ? localMessages : [];
  const server = Array.isArray(serverMessages) ? serverMessages : [];
  if (server.length >= local.length) return server;
  return local;
}

export function transcriptFromMessages(messages, maxChars = 500000) {
  if (!Array.isArray(messages) || !messages.length) return '';
  const parts = [];
  let total = 0;
  for (const m of messages) {
    const body = String(m?.content ?? '').trim();
    if (!body) continue;
    const line = `**${speakerLabel(m)}:** ${body}\n\n`;
    if (total + line.length > maxChars) break;
    parts.push(line);
    total += line.length;
  }
  return parts.join('').trim();
}

function handleCondenserSseData(data, { onToken, onDone, onError }) {
  if (data.type === 'token' && data.text) {
    onToken?.(data.text);
    return null;
  }
  if (data.type === 'done') {
    onDone?.(data);
    return data;
  }
  if (data.type === 'error') {
    const err = new Error(data.detail || 'Stream error');
    onError?.(err);
    throw err;
  }
  return null;
}

function parseCondenserSseLine(line, handlers) {
  if (!line.startsWith('data: ')) return null;
  try {
    const data = JSON.parse(line.slice(6));
    return handleCondenserSseData(data, handlers);
  } catch (parseErr) {
    if (parseErr?.name === 'AbortError') throw parseErr;
    if (parseErr?.message && !String(parseErr.message).includes('JSON')) {
      throw parseErr;
    }
    return null;
  }
}

/**
 * Read condenser session SSE (token | done | error).
 */
/** Orchestrator SSE: step_start | token | step_done | failover | completed | error | status */
export async function readOrchestratorStream(response, handlers, signal) {
  const { onEvent, onError } = handlers || {};
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = formatApiError(data, detail);
    } catch {
      try {
        detail = (await response.text()) || detail;
      } catch {
        /* ignore */
      }
    }
    throw new Error(detail);
  }
  if (!response.body) throw new Error('No response body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const dispatch = (data) => {
    onEvent?.(data);
    if (data.type === 'error') {
      const err = new Error(data.detail || 'Stream error');
      onError?.(err);
      throw err;
    }
    return data;
  };

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        throw new DOMException('Aborted', 'AbortError');
      }
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          dispatch(JSON.parse(line.slice(6)));
        } catch (parseErr) {
          if (parseErr?.name === 'AbortError') throw parseErr;
          if (parseErr?.message && !String(parseErr.message).includes('JSON')) {
            throw parseErr;
          }
        }
      }
    }
    const tail = buffer.trim();
    if (tail) {
      for (const line of tail.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          dispatch(JSON.parse(line.slice(6)));
        } catch {
          /* ignore tail parse */
        }
      }
    }
    return null;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

export async function readCondenserSessionStream(response, { onToken, onDone, onError, signal }) {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = formatApiError(data, detail);
    } catch {
      try {
        detail = (await response.text()) || detail;
      } catch {
        /* ignore */
      }
    }
    throw new Error(detail);
  }
  if (!response.body) throw new Error('No response body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const handlers = { onToken, onDone, onError };

  const processLines = (lines) => {
    for (const line of lines) {
      const donePayload = parseCondenserSseLine(line, handlers);
      if (donePayload) return donePayload;
    }
    return null;
  };

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        throw new DOMException('Aborted', 'AbortError');
      }
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      const finished = processLines(lines);
      if (finished) return finished;
    }

    buffer += decoder.decode();
    const tail = buffer.trim();
    if (tail) {
      const finished = processLines(tail.split('\n'));
      if (finished) return finished;
    }
    return null;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}
