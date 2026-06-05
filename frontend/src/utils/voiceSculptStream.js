/**
 * Client for POST /voice-sculpt/sculpt-stream (SSE progress + done/error).
 */

import { safeErrorMessage } from '../config/api';

function sculptStreamErrorMessage(data) {
  if (!data || typeof data !== 'object') return 'Pipeline failed';
  const parts = [data.detail, data.install_hint].filter(
    (p) => typeof p === 'string' && p.trim(),
  );
  return parts[0] || 'Pipeline failed';
}

export async function runVoiceSculptStream(apiUrl, body, { onProgress, signal } = {}) {
  const res = await fetch(`${apiUrl}/voice-sculpt/sculpt-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text();
    let detail = text || `Request failed (${res.status})`;
    try {
      const parsed = JSON.parse(text);
      if (parsed.detail) detail = typeof parsed.detail === 'string' ? parsed.detail : JSON.stringify(parsed.detail);
    } catch {
      /* use raw text */
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      let data;
      try {
        data = JSON.parse(line.slice(6));
      } catch {
        continue;
      }
      if (data.type === 'progress') {
        onProgress?.(data);
      } else if (data.type === 'done') {
        return data;
      } else if (data.type === 'error') {
        throw new Error(sculptStreamErrorMessage(data));
      }
    }
  }

  throw new Error('Stream ended without a result');
}

export function sculptBodyFromQueueJob(job) {
  const source = (job.source || '').trim();
  const sourceLines = source.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  return {
    source,
    source_type: 'local_path',
    output_name: (job.outputName || '').trim() || undefined,
    accent_model: job.accentModel || 'default',
    skip_rvc: job.skipRvc !== false,
    skip_uvr: job.skipUvr !== false,
    combine_mode: 'morph',
    morph_balance: (job.morphBalance ?? 50) / 100,
    pitch: job.pitch ?? 0,
    index_rate: job.indexRate,
    protect: job.protect ?? 0.33,
    voice_prompt: (job.voicePrompt || '').trim() || undefined,
    _sourceCount: sourceLines.length,
  };
}
