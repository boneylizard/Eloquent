/**
 * New-chat character introduction — mirrors call-mode "about this character"
 * context assembly (character, user profile, system prompt, agentic memory via full_generation).
 */

import {
  CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES,
  buildCallModeAboutPrompt,
  parseCallModeAboutResponse,
  normalizeCallModeAboutData,
  splitUnstructuredAboutText,
} from './callModeCharacterAbout.js';
import { buildFlowGenerateRequestBody, readFlowGenerateError, resolveFlowGenerateConfig } from './flowGenerateApi';

export const CHARACTER_INTRO_REQUEST_PURPOSE = 'character_intro';

/** Maps intro JSON keys → UI section labels */
export const CHARACTER_INTRO_UI_LABELS = {
  headline: 'Meeting',
  who_they_are: 'Who they are',
  how_they_engage: 'How they engage with you',
  tone: 'Tone & presence',
  voice_line: 'In their voice',
  themes: 'Themes',
};

const INTRO_OUTPUT_SPEC = `You are writing a welcoming character introduction for someone starting a NEW chat (no prior messages in this thread).

Return ONLY valid JSON (no markdown fences, no commentary, no extra keys):
{
  "headline": "",
  "who_they_are": "",
  "how_they_engage": "",
  "tone": "",
  "voice_line": "",
  "themes": []
}

FIELD RUBRIC:
• headline (≤14 words): A warm, inviting snapshot — like a chapter title for this meeting.
• who_they_are: Stable core — temperament, values, what defines them. Ground in the character card. 2–3 sentences. No plot recap.
• how_they_engage: How they typically relate to the user — warmth, distance, playfulness, tension. Use agentic memory / user context when present. 2–3 sentences.
• tone: How they sound and feel in conversation — pace, register, emotional color. 1–2 sentences.
• voice_line: ONE first-person line (≤30 words) in character voice. Not meta. Not a summary.
• themes: 2–4 short tags (1–3 words each) for motifs that may shape this chat.

GLOBAL:
- Ground claims in the data below; do not invent major facts.
- This is a fresh chat — do not reference nonexistent prior turns in this thread.
- Each fact belongs in exactly ONE field.
- Write for someone deciding whether to begin — vivid, concise, no filler.`;

export const DEFAULT_CHARACTER_INTRO_PROMPT = `${INTRO_OUTPUT_SPEC}

{{CHARACTER_BLOCK}}
{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

export const DEFAULT_CHARACTER_INTRO_INSTRUCTIONS = `${INTRO_OUTPUT_SPEC}

Stay grounded in the character identity established above.

{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

const INTRO_TEXT_FIELDS = ['headline', 'who_they_are', 'how_they_engage', 'tone', 'voice_line'];

export const CHARACTER_INTRO_MAX_ATTEMPTS = 3;

export const CHARACTER_INTRO_JSON_SCHEMA_HINT = `{
  "headline": "",
  "who_they_are": "",
  "how_they_engage": "",
  "tone": "",
  "voice_line": "",
  "themes": []
}`;

const INTRO_SECTION_FIELDS = ['who_they_are', 'how_they_engage', 'tone', 'voice_line'];

const INTRO_SALVAGE_PATTERNS = [
  ['headline', /"headline"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['who_they_are', /"who_they_are"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['how_they_engage', /"how_they_engage"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['tone', /"tone"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['voice_line', /"voice_line"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['who_they_are', /"whoTheyAre"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['how_they_engage', /"howTheyEngage"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['voice_line', /"voiceLine"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['who_they_are', /"essence"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['how_they_engage', /"relationship"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['tone', /"presence"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ['voice_line', /"voice_note"\s*:\s*"((?:\\.|[^"\\])*)"/i],
];

function stripJsonFences(text) {
  const cleaned = (text || '').trim();
  const fence = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/i);
  return fence ? fence[1].trim() : cleaned;
}

function decodeJsonStringFragment(value) {
  try {
    return JSON.parse(`"${value}"`);
  } catch {
    return value.replace(/\\"/g, '"').replace(/\\n/g, '\n');
  }
}

/** Extract outermost {...} respecting JSON string boundaries. */
export function extractFirstJsonObject(text) {
  if (!text) return null;
  const start = text.indexOf('{');
  if (start === -1) return null;
  let inString = false;
  let escape = false;
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === '\\') escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

function jsonClosingSuffix(blob) {
  let inString = false;
  let escape = false;
  const stack = [];
  for (const ch of blob || '') {
    if (inString) {
      if (escape) escape = false;
      else if (ch === '\\') escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') stack.push('}');
    else if (ch === '[') stack.push(']');
    else if ((ch === '}' || ch === ']') && stack.length && stack[stack.length - 1] === ch) {
      stack.pop();
    }
  }
  let suffix = '';
  if (inString) suffix += '"';
  suffix += [...stack].reverse().join('');
  return suffix;
}

/** Best-effort repair for model output cut off mid-JSON. */
export function repairTruncatedJsonBlob(blob) {
  const trimmed = (blob || '').trim();
  if (!trimmed) return trimmed;
  const candidates = [trimmed + jsonClosingSuffix(trimmed)];
  const seen = new Set();
  for (const cand of candidates) {
    if (!cand || seen.has(cand)) continue;
    seen.add(cand);
    try {
      JSON.parse(cand);
      return cand;
    } catch {
      /* try next */
    }
  }
  return trimmed + jsonClosingSuffix(trimmed);
}

export function countIntroSections(data) {
  if (!data) return 0;
  return INTRO_SECTION_FIELDS.filter((f) => String(data[f] || '').trim()).length;
}

/** Headline plus at least one narrative section — show intro with amber partial note. */
export function introJsonIsPartialUsable(data) {
  if (!data || typeof data !== 'object') return false;
  const headline = String(data.headline || '').trim();
  if (!headline) return false;
  return countIntroSections(data) >= 1;
}

/** Enough sections to treat as a complete intro (no partial banner). */
export function introJsonIsFullyUsable(data) {
  if (!introJsonIsPartialUsable(data)) return false;
  return countIntroSections(data) >= 2;
}

export function salvageIntroFields(blob) {
  const salvaged = {};
  for (const [field, pattern] of INTRO_SALVAGE_PATTERNS) {
    if (salvaged[field]) continue;
    const match = (blob || '').match(pattern);
    if (match?.[1]) salvaged[field] = decodeJsonStringFragment(match[1]).trim();
  }
  const themesMatch = (blob || '').match(/"themes"\s*:\s*\[([\s\S]*?)\]/i);
  if (themesMatch) {
    try {
      const themes = JSON.parse(`[${themesMatch[1]}]`);
      if (Array.isArray(themes)) salvaged.themes = themes.filter(Boolean);
    } catch {
      /* noop */
    }
  }
  if (!Object.keys(salvaged).length) return null;
  const normalized = normalizeCharacterIntroData(salvaged);
  normalized._salvaged = true;
  return normalized;
}

export function buildCharacterIntroRepairPrompt(brokenJson) {
  const excerpt = (brokenJson || '').trim().slice(0, 12000);
  return `You repair broken character introduction JSON. Output ONLY one valid JSON object — no markdown fences, no commentary.

Preserve every complete field from the broken input; fill only what is missing. Escape quotes inside strings.

Required schema (exact keys):
${CHARACTER_INTRO_JSON_SCHEMA_HINT}

BROKEN_JSON:
${excerpt}

Respond with ONLY the repaired JSON object.`;
}

function classifyIntroParseResult(data, { salvaged = false } = {}) {
  const normalized = normalizeCharacterIntroData(data);
  if (!introJsonIsPartialUsable(normalized)) {
    return null;
  }
  const partial = salvaged || !introJsonIsFullyUsable(normalized);
  return {
    structured: true,
    data: normalized,
    partial,
    salvaged: Boolean(salvaged),
  };
}

/** Map call-mode about fields into intro shape when reusing legacy parsers */
export function mapAboutDataToIntro(aboutData) {
  if (!aboutData) return null;
  return normalizeCharacterIntroData({
    headline: aboutData.headline,
    who_they_are: aboutData.essence || aboutData.who_they_are,
    how_they_engage: aboutData.relationship || aboutData.how_they_engage,
    tone: [aboutData.presence, aboutData.tone].filter(Boolean).join(' ').trim(),
    voice_line: aboutData.voice_note || aboutData.voice_line,
    themes: aboutData.themes,
  });
}

export function normalizeCharacterIntroData(raw) {
  if (!raw || typeof raw !== 'object') return raw;
  const data = {
    headline: raw.headline || '',
    who_they_are: raw.who_they_are || raw.whoTheyAre || raw.essence || '',
    how_they_engage: raw.how_they_engage || raw.howTheyEngage || raw.relationship || '',
    tone: raw.tone || raw.presence || '',
    voice_line: raw.voice_line || raw.voiceLine || raw.voice_note || '',
    themes: Array.isArray(raw.themes) ? raw.themes.filter(Boolean) : [],
  };
  for (const field of INTRO_TEXT_FIELDS) {
    data[field] = String(data[field] || '').trim();
  }
  data.themes = data.themes
    .map((t) => String(t).trim().replace(/^["']|["']$/g, '').slice(0, 40))
    .slice(0, 4);
  return data;
}

/** True when intro has enough content to show (including salvaged partial). */
export function isCharacterIntroReady(result) {
  if (!result?.structured || !result?.data) return false;
  return introJsonIsPartialUsable(result.data);
}

export function isCharacterIntroPartial(result) {
  return Boolean(result?.partial || result?.salvaged || result?.data?._salvaged);
}

export const DEFAULT_CHAT_TITLE = 'New Chat';
const INTRO_CHAT_TITLE_MAX_LEN = 40;

function truncateChatTitle(text, maxLen = INTRO_CHAT_TITLE_MAX_LEN) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return '';
  if (trimmed.length <= maxLen) return trimmed;
  return `${trimmed.slice(0, maxLen - 3)}...`;
}

/** Whether an intro-derived title may replace the conversation sidebar name. */
export function conversationAcceptsIntroTitle(conv) {
  if (!conv) return false;
  if (conv.titleSource === 'manual') return false;
  if (conv.requiresTitle === true) return true;
  if ((conv.name || '').trim() === DEFAULT_CHAT_TITLE) return true;
  if (conv.titleSource === 'intro') return true;
  return false;
}

/** Sidebar title from parsed intro JSON: headline → who_they_are → "{character} intro". */
export function deriveIntroChatTitle(introResult, { characterName = 'Character' } = {}) {
  if (!isCharacterIntroReady(introResult)) return null;
  const headline = truncateChatTitle(introResult.data?.headline);
  if (headline) return headline;
  const who = truncateChatTitle(introResult.data?.who_they_are);
  if (who) return who;
  const name = String(characterName || 'Character').trim() || 'Character';
  return truncateChatTitle(`${name} intro`);
}

/** Render intro JSON as a single polished assistant markdown message. */
export function formatCharacterIntroAsMarkdown(data) {
  const normalized = normalizeCharacterIntroData(data);
  if (!introJsonIsPartialUsable(normalized)) return '';

  const lines = [];
  if (normalized.headline) {
    lines.push(`## ${normalized.headline}`);
    lines.push('');
  }
  const sections = [
    ['who_they_are', CHARACTER_INTRO_UI_LABELS.who_they_are],
    ['how_they_engage', CHARACTER_INTRO_UI_LABELS.how_they_engage],
    ['tone', CHARACTER_INTRO_UI_LABELS.tone],
  ];
  for (const [key, label] of sections) {
    if (normalized[key]) {
      lines.push(`### ${label}`);
      lines.push(normalized[key]);
      lines.push('');
    }
  }
  if (normalized.voice_line) {
    lines.push(`### ${CHARACTER_INTRO_UI_LABELS.voice_line}`);
    lines.push(`> "${normalized.voice_line}"`);
    lines.push('');
  }
  if (normalized.themes?.length) {
    lines.push(`*${CHARACTER_INTRO_UI_LABELS.themes}:* ${normalized.themes.join(' · ')}`);
  }
  return lines.join('\n').trim();
}

/** Bot message(s) to seed the thread when the intro UX completes. */
export function buildCharacterIntroSeedMessages(introResult, { character, avatar, generateId } = {}) {
  if (!isCharacterIntroReady(introResult) || typeof generateId !== 'function') return [];
  const content = formatCharacterIntroAsMarkdown(introResult.data);
  if (!content) return [];
  const charName = character?.name || 'Character';
  return [{
    id: generateId(),
    role: 'bot',
    content,
    modelId: 'primary',
    characterName: charName,
    characterId: character?.id,
    avatar: avatar ?? undefined,
    isCharacterIntro: true,
  }];
}

/** UI state for CharacterIntroExperience */
export function getCharacterIntroStatus({ loading, error, result }) {
  if (isCharacterIntroReady(result)) return 'ready';
  if (loading) return 'loading';
  if (error) return 'error';
  return 'loading';
}

export function parseCharacterIntroResponse(rawText) {
  const trimmedRaw = rawText?.trim() || '';
  const cleaned = stripJsonFences(trimmedRaw);
  const blob = extractFirstJsonObject(cleaned) || cleaned;
  const candidates = [blob, repairTruncatedJsonBlob(blob)].filter(Boolean);

  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === 'object') {
        const mapped = mapAboutDataToIntro(parsed) || parsed;
        const classified = classifyIntroParseResult(mapped);
        if (classified) {
          return { ...classified, rawText: trimmedRaw };
        }
      }
    } catch {
      /* try repair/salvage */
    }
  }

  const salvaged = salvageIntroFields(blob || cleaned);
  if (salvaged) {
    const classified = classifyIntroParseResult(salvaged, { salvaged: true });
    if (classified) {
      return { ...classified, rawText: trimmedRaw };
    }
  }

  const parsed = parseCallModeAboutResponse(trimmedRaw);
  if (parsed.structured && parsed.data) {
    const mapped = mapAboutDataToIntro(parsed.data);
    const classified = classifyIntroParseResult(mapped || parsed.data);
    if (classified) {
      return { ...classified, rawText: parsed.rawText || trimmedRaw };
    }
  }

  const split = splitUnstructuredAboutText(trimmedRaw);
  if (split) {
    const mapped = mapAboutDataToIntro(split) || normalizeCharacterIntroData(split);
    const classified = classifyIntroParseResult(mapped);
    if (classified) {
      return { ...classified, rawText: trimmedRaw };
    }
  }

  return { structured: false, data: null, rawText: trimmedRaw };
}

export function buildCharacterIntroPrompt({
  character,
  userProfile,
  messages,
  customPrompt,
  historyLimit = 8,
  characterSystemPrompt = null,
  systemPromptMode = CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.flat,
}) {
  const template =
    customPrompt?.trim()
    || (systemPromptMode !== CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.flat && characterSystemPrompt?.trim()
      ? DEFAULT_CHARACTER_INTRO_INSTRUCTIONS
      : DEFAULT_CHARACTER_INTRO_PROMPT);

  return buildCallModeAboutPrompt({
    character,
    userProfile,
    messages: messages?.length ? messages : [],
    customPrompt: template,
    historyLimit,
    characterSystemPrompt,
    systemPromptMode,
  });
}

async function readGenerateStream(response, onPartial) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let sseBuffer = '';
  let promptTokens = null;
  let completionTokens = null;
  let sawDone = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    sseBuffer += decoder.decode(value, { stream: true });
    const events = sseBuffer.split('\n\n');
    sseBuffer = events.pop() || '';

    for (const line of events) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') {
        sawDone = true;
        continue;
      }
      try {
        const parsed = JSON.parse(data);
        if (parsed?.error) {
          const err = typeof parsed.error === 'string'
            ? parsed.error
            : (parsed.error?.message || JSON.stringify(parsed.error));
          throw new Error(err || 'Character intro stream failed');
        }
        const usage = parsed?.usage;
        if (usage && typeof usage === 'object') {
          if (typeof usage.prompt_tokens === 'number') promptTokens = usage.prompt_tokens;
          if (typeof usage.completion_tokens === 'number') completionTokens = usage.completion_tokens;
        }
        const chunk = parsed.text ?? parsed.token ?? '';
        if (chunk) {
          fullText += chunk;
          onPartial?.(fullText);
        }
      } catch {
        /* ignore partial parse */
      }
    }
  }
  console.info(
    `[character_intro_stream_diag] prompt_tokens=${promptTokens ?? 'unknown'} completion_tokens=${completionTokens ?? 'unknown'} chars=${fullText.length} saw_done=${sawDone}`,
  );
  if (!fullText.trim()) {
    throw new Error('Character intro returned no text. Check routing/provider logs for this intro request.');
  }
  return fullText;
}

export async function fetchCharacterIntro({
  apiUrl,
  modelName,
  character,
  userProfile,
  messages = [],
  settings = {},
  onPartial,
  signal,
  characterSystemPrompt = null,
  systemPromptMode,
  resolveCharacterSystemPrompt,
}) {
  let resolvedSystemPrompt = characterSystemPrompt;
  if (typeof resolveCharacterSystemPrompt === 'function') {
    resolvedSystemPrompt = await resolveCharacterSystemPrompt();
  }

  const mode =
    systemPromptMode
    || settings.characterIntroSystemPromptMode
    || settings.callModeAboutCharacterSystemPromptMode
    || CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation;

  const prompt = buildCharacterIntroPrompt({
    character,
    userProfile,
    messages,
    customPrompt: settings.characterIntroPrompt,
    historyLimit: settings.characterIntroHistoryLimit ?? 8,
    characterSystemPrompt: resolvedSystemPrompt,
    systemPromptMode: mode,
  });

  const flowConfig = resolveFlowGenerateConfig({
    flowKind: 'characterIntro',
    settings,
    apiUrl,
    fallbackModelName: modelName,
  });
  const endpoint = flowConfig.url;

  const maxAttempts = settings.characterIntroMaxAttempts ?? CHARACTER_INTRO_MAX_ATTEMPTS;
  let lastRaw = '';
  let lastResult = parseCharacterIntroResponse('');
  let bestPartialResult = null;

  const basePrompt = prompt;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const attemptPrompt =
      attempt === 0 ? basePrompt : buildCharacterIntroRepairPrompt(lastRaw);

    const attemptResponse = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify(
        buildFlowGenerateRequestBody({
          flowKind: 'characterIntro',
          settings,
          apiUrl,
          fallbackModelName: modelName,
          basePayload: {
            prompt: attemptPrompt,
            max_tokens: settings.characterIntroMaxTokens ?? 900,
            temperature: settings.characterIntroTemperature ?? 0.55,
            stop: ['```'],
            stream: true,
            gpu_id: 0,
            memoryEnabled: false,
            request_purpose: settings.characterIntroRequestPurpose || CHARACTER_INTRO_REQUEST_PURPOSE,
            active_character: character || null,
            userProfile: userProfile ? { id: userProfile.id, name: userProfile.name } : null,
          },
        })
      ),
    });

    if (!attemptResponse.ok) {
      throw new Error(await readFlowGenerateError(attemptResponse));
    }

    lastRaw = await readGenerateStream(attemptResponse, onPartial);
    lastResult = parseCharacterIntroResponse(lastRaw);
    lastResult.attempts = attempt + 1;

    if (isCharacterIntroReady(lastResult)) {
      return lastResult;
    }
    if (lastResult?.structured && introJsonIsPartialUsable(lastResult.data)) {
      bestPartialResult = lastResult;
    }
  }

  if (bestPartialResult) {
    return bestPartialResult;
  }

  return {
    ...lastResult,
    structured: false,
    data: lastResult?.data ?? null,
    rawText: lastRaw,
    exhausted: true,
    rawExcerpt: (lastRaw || '').trim().slice(0, 500),
    attempts: maxAttempts,
  };
}
