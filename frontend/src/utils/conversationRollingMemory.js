/**
 * Rolling conversation memory for subscription/API chats: merge newly archived
 * turns into a structured pack so we do not resend full verbatim history every turn.
 */

import { formatPrompt } from './chat_templates';
import { createRouteTraceId, logRouteTrace, resolveUnifiedRequestRoute } from './requestRouting';

const MAX_TRANSCRIPT_CHARS = 120000;

const COMPACTION_SYSTEM = `You maintain a continuity memory pack for an ongoing roleplay or chat.

You will receive:
1) EXISTING_MEMORY_PACK — structured notes from earlier compaction passes (may be empty).
2) NEW_TURNS_TO_MERGE — verbatim speaker-labeled lines that are leaving the "recent window".

Your job is to produce ONE JSON object that merges both into a single updated pack.

Merge priorities (minimize continuity loss):
1) Carry forward nearly everything important from EXISTING_MEMORY_PACK unless NEW_TURNS clearly contradicts it — then update factually and reflect the change in continuity_notes if needed.
2) Add only NEW information from NEW_TURNS; dedupe overlapping facts instead of dropping older established facts.
3) Prefer keeping slightly redundant bullets over omitting a fact that might matter later.

Rules:
- Be concrete: proper nouns, numbers, decisions, promises, conflicts, locations, time cues.
- Include emotional tone of the user when discernible (factual, not poetic).
- Do NOT invent events not supported by the inputs.
- Do NOT output markdown fences or commentary — ONLY raw JSON.

Required JSON shape (fill every key; use empty string or empty array when unknown):
{
  "scene_anchor": "where/when/mode of interaction in plain language",
  "characters": [{"name": "...", "note": "role, stance, goals"}],
  "committed_facts": ["...", "..."],
  "relationship_dynamics": ["...", "..."],
  "open_threads": ["unresolved questions or plot hooks"],
  "themes": ["...", "..."],
  "user_affect": "short factual read on user's mood/frustration/energy if visible",
  "promises_or_constraints": ["things agreed or rules stated"],
  "continuity_notes": "anything else critical so the next replies stay coherent"
}`;

function extractFirstJson(text) {
  if (!text) return null;
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (char === '{') depth += 1;
    if (char === '}') depth -= 1;
    if (depth === 0) return text.slice(start, i + 1);
  }
  return null;
}

export function formatRollingTranscript(messages, getSpeakerLabel) {
  return (messages || [])
    .map((m) => {
      const who = getSpeakerLabel(m);
      return `${who}: ${m.content}`;
    })
    .join('\n\n');
}

export function parsedJsonToRollingPack(obj) {
  if (!obj || typeof obj !== 'object') return '';

  const lines = [];
  const push = (label, val) => {
    if (val == null || val === '') return;
    lines.push(`${label}: ${typeof val === 'string' ? val : JSON.stringify(val)}`);
  };

  push('Scene', obj.scene_anchor);
  if (Array.isArray(obj.characters) && obj.characters.length) {
    lines.push(
      'Characters:',
      ...obj.characters.map((c) =>
        typeof c === 'object' && c
          ? `  - ${c.name || '?'}: ${c.note || ''}`.trim()
          : `  - ${String(c)}`
      )
    );
  }
  const bullets = (title, arr) => {
    if (!Array.isArray(arr) || !arr.length) return;
    lines.push(`${title}:`);
    arr.forEach((x) => lines.push(`  • ${String(x)}`));
  };
  bullets('Facts', obj.committed_facts);
  bullets('Relationships', obj.relationship_dynamics);
  bullets('Open threads', obj.open_threads);
  bullets('Themes', obj.themes);
  push('User affect', obj.user_affect);
  bullets('Promises / constraints', obj.promises_or_constraints);
  push('Notes', obj.continuity_notes);

  return lines.join('\n').trim();
}

export function parseCompactionResponse(raw, cleanModelOutput) {
  const cleaned = typeof cleanModelOutput === 'function' ? cleanModelOutput(raw || '') : String(raw || '');
  const jsonStr = extractFirstJson(cleaned);
  if (!jsonStr) return cleaned.trim();
  try {
    const parsed = JSON.parse(jsonStr);
    const formatted = parsedJsonToRollingPack(parsed);
    return formatted || cleaned.trim();
  } catch {
    return cleaned.trim();
  }
}

/**
 * Calls /generate once to fold archived turns into rollingPack text.
 */
export async function mergeRollingMemoryPack({
  apiBaseUrl,
  modelName,
  primaryIsAPI = false,
  settings = {},
  existingPack,
  messagesToFold,
  formatPrompt: formatPromptFn,
  cleanModelOutput,
  getSpeakerLabel,
}) {
  if (!apiBaseUrl || !modelName || !messagesToFold?.length) {
    return (existingPack || '').trim();
  }

  let transcript = formatRollingTranscript(messagesToFold, getSpeakerLabel);
  if (transcript.length > MAX_TRANSCRIPT_CHARS) {
    transcript =
      '[Older lines truncated for this compaction pass]\n' +
      transcript.slice(transcript.length - MAX_TRANSCRIPT_CHARS);
  }

  const userBlob = [
    'EXISTING_MEMORY_PACK:\n',
    existingPack && existingPack.trim() ? existingPack.trim() : '(none)',
    '\n\nNEW_TURNS_TO_MERGE:\n',
    transcript,
    '\n\nRespond with ONLY the JSON object described in your system instructions.',
  ].join('');

  const prompt = formatPromptFn([{ role: 'user', content: userBlob }], modelName, COMPACTION_SYSTEM);
  const route = resolveUnifiedRequestRoute({
    primaryModel: modelName,
    primaryIsAPI,
    settings,
    requestPurpose: 'rolling_memory_compaction',
  });
  const traceId = createRouteTraceId();
  logRouteTrace({
    action: 'rolling_memory_compaction',
    route,
    requestPurpose: 'rolling_memory_compaction',
    traceId,
  });

  const res = await fetch(`${apiBaseUrl}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Router-Trace-Id': traceId },
    body: JSON.stringify({
      prompt,
      model_name: route.effectiveModel || modelName,
      max_tokens: 3072,
      temperature: 0.22,
      top_p: 0.9,
      top_k: 40,
      repetition_penalty: 1.05,
      frequency_penalty: 0,
      presence_penalty: 0,
      anti_repetition_mode: false,
      use_rag: false,
      rag_docs: [],
      use_web_search: false,
      gpu_id: 0,
      userProfile: { id: 'rolling-memory-compaction' },
      request_purpose: 'rolling_memory_compaction',
      selected_model: route.selectedModel || undefined,
      round_robin_enabled: route.autoEnabled,
      memoryEnabled: false,
      stream: false,
    }),
  });

  if (!res.ok) {
    throw new Error(`Rolling memory compaction HTTP ${res.status}`);
  }

  const data = await res.json();
  const rawText = data?.text ?? '';
  return parseCompactionResponse(rawText, cleanModelOutput);
}
