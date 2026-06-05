/** Experimental call-mode "about this character" — customizable prompt + structured JSON response. */

import { buildFlowGenerateRequestBody, readFlowGenerateError, resolveFlowGenerateConfig } from './flowGenerateApi';

export const CALL_MODE_ABOUT_REQUEST_PURPOSE = 'call_mode_character_about';

/** How the About request assembles character context before intelligence-sheet instructions. */
export const CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES = {
  flat: 'flat',
  character_card: 'character_card',
  full_generation: 'full_generation',
};

/** Maps JSON keys → on-screen card labels (keep in sync with CallModeAboutPanel). */
export const CALL_MODE_ABOUT_UI_LABELS = {
  headline: 'Banner headline',
  essence: 'Essence',
  presence: 'On this call',
  relationship: 'With you',
  current_state: 'Right now',
  themes: 'Theme tags',
  story_so_far: 'Story so far',
  watch_for: 'Watch for',
  voice_note: 'Voice',
};

const CALL_MODE_ABOUT_OUTPUT_SPEC = `You are filling a fixed UI layout — NOT writing one essay. Each JSON key becomes its own labeled card on screen. The user reads cards independently.

Return ONLY valid JSON (no markdown fences, no commentary, no extra keys) with exactly these keys:
{
  "headline": "",
  "essence": "",
  "presence": "",
  "relationship": "",
  "current_state": "",
  "themes": [],
  "story_so_far": "",
  "watch_for": "",
  "voice_note": ""
}

FIELD RUBRIC — obey strictly; do not repeat facts or sentences across fields:

• headline (banner, ≤12 words): Punchy snapshot of this moment. Not a summary paragraph.

• essence (card "Essence"): Stable character core only — temperament, values, verbal habits, what makes them them. Ground in the character card. NO plot events, NO current scene, NO relationship commentary. 1–2 sentences.

• presence (card "On this call"): How they come across on this live call — energy, pace, warmth, tension, subtext in their recent replies. Sensory/impressionistic. NOT backstory. 2–3 sentences.

• relationship (card "With you"): The dynamic between this character and the user — trust, distance, power, affection, unresolved friction. Cite patterns from chat. NOT a plot recap. 1–2 sentences.

• current_state (card "Right now"): This instant in the scene — location, immediate goal, emotional temperature, what they're doing. Present tense. 1–2 sentences.

• themes (tags): Exactly 2–4 short tags (1–3 words each). Motifs active in this conversation (e.g. "fragile trust", "unspoken guilt"). NOT sentences.

• story_so_far (card "Story so far"): Neutral chronological recap of chat events only — who did what, in order. No analysis, no advice, no character psychology. 2–4 sentences max.

• watch_for (card "Watch for"): ONE concrete under-read signal — a contradiction, tell, bait, or thread the user might miss. Actionable. Exactly 1 sentence.

• voice_note (card "Voice"): ONE first-person line as the character (≤25 words). Their dialect and attitude. NOT meta ("As an AI…"). NOT a summary.

GLOBAL RULES:
- Each fact belongs in exactly ONE field. Zero duplication across fields.
- If a sentence could fit two fields, choose the more specific field and delete it from the other.
- Ground every claim in the data below; do not invent major facts.
- Write for someone glancing mid-call — tight, specific, no filler.
- Before outputting, verify each field answers ONLY its card's question.`;

export const DEFAULT_CALL_MODE_ABOUT_INSTRUCTIONS = `${CALL_MODE_ABOUT_OUTPUT_SPEC}

Stay grounded in the character identity established above.

{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

export const DEFAULT_CALL_MODE_ABOUT_PROMPT = `${CALL_MODE_ABOUT_OUTPUT_SPEC}

{{CHARACTER_BLOCK}}
{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

const STORY_TRACKER_KEY = 'eloquent-story-tracker';

function formatCharacterBlock(character) {
  if (!character) return 'CHARACTER:\n(No character selected)\n';
  const lines = [`CHARACTER: ${character.name || 'Unknown'}`];
  const fields = [
    ['Persona / description', character.description],
    ['Personality', character.personality],
    ['Background', character.background],
    ['Scenario', character.scenario],
    ['Speech style', character.speech_style],
    ['Model instructions', character.model_instructions],
    ['First message', character.first_message],
    ['Ethics justification', character.ethics_justification],
  ];
  for (const [label, value] of fields) {
    if (value && String(value).trim()) lines.push(`${label}: ${String(value).trim()}`);
  }
  if (Array.isArray(character.example_dialogue) && character.example_dialogue.length) {
    const examples = character.example_dialogue
      .filter((e) => e?.content?.trim())
      .slice(0, 4)
      .map((e) => `  ${e.role}: ${e.content.trim()}`)
      .join('\n');
    if (examples) lines.push(`Example dialogue:\n${examples}`);
  }
  if (Array.isArray(character.loreEntries) && character.loreEntries.length) {
    const lore = character.loreEntries
      .slice(0, 6)
      .map((e) => `  - ${e.content || e}`)
      .join('\n');
    lines.push(`Lore:\n${lore}`);
  }
  return `${lines.join('\n')}\n`;
}

function formatUserBlock(userProfile) {
  if (!userProfile) return 'USER:\n(Anonymous player)\n';
  const lines = [`USER: ${userProfile.name || userProfile.username || 'Player'}`];
  if (userProfile.personal_info?.bio) lines.push(`Bio: ${userProfile.personal_info.bio}`);
  if (userProfile.personal_info?.location) lines.push(`Location: ${userProfile.personal_info.location}`);
  const p = userProfile.personality;
  if (p) {
    if (typeof p === 'string') lines.push(`Personality: ${p}`);
    else {
      if (p.traits?.length) lines.push(`Traits: ${Array.isArray(p.traits) ? p.traits.join(', ') : p.traits}`);
      if (p.interests?.length) lines.push(`Interests: ${Array.isArray(p.interests) ? p.interests.join(', ') : p.interests}`);
      if (p.values?.length) lines.push(`Values: ${Array.isArray(p.values) ? p.values.join(', ') : p.values}`);
    }
  }
  if (userProfile.background) lines.push(`Background: ${userProfile.background}`);
  return `${lines.join('\n')}\n`;
}

function getStoryTrackerBlock() {
  try {
    const raw = localStorage.getItem(STORY_TRACKER_KEY);
    if (!raw) return '';
    const data = JSON.parse(raw);
    const parts = [];
    if (data.currentObjective) parts.push(`Objective: ${data.currentObjective}`);
    if (data.storyNotes) parts.push(`Notes: ${data.storyNotes}`);
    if (data.plotPoints?.length) {
      parts.push(`Recent events: ${data.plotPoints.slice(-4).map((p) => p.value).join('; ')}`);
    }
    if (!parts.length) return '';
    return `STORY TRACKER:\n${parts.join('\n')}\n`;
  } catch {
    return '';
  }
}

function formatChatHistory(messages, limit = 40) {
  if (!Array.isArray(messages) || !messages.length) {
    return 'CHAT HISTORY:\n(No messages yet)\n';
  }
  const recent = messages
    .filter((m) => m?.content && m.type !== 'image' && m.type !== 'video')
    .slice(-limit);
  const lines = recent.map((m) => {
    const speaker =
      m.role === 'user'
        ? m.characterName || 'User'
        : m.characterName || 'Character';
    const text = String(m.content).replace(/\s+/g, ' ').trim();
    const clipped = text.length > 600 ? `${text.slice(0, 600)}…` : text;
    return `${speaker}: ${clipped}`;
  });
  return `CHAT HISTORY (most recent ${recent.length} turns):\n${lines.join('\n')}\n`;
}

function applyContextPlaceholders(template, { character, userProfile, messages, historyLimit, characterSystemPrompt }) {
  const storyBlock = getStoryTrackerBlock();
  return template
    .replace(/\{\{CHARACTER_SYSTEM_PROMPT\}\}/g, characterSystemPrompt?.trim() || '')
    .replace('{{CHARACTER_BLOCK}}', formatCharacterBlock(character))
    .replace('{{USER_BLOCK}}', formatUserBlock(userProfile))
    .replace('{{STORY_BLOCK}}', storyBlock || 'STORY TRACKER:\n(none)\n')
    .replace('{{CHAT_HISTORY}}', formatChatHistory(messages, historyLimit));
}

export function buildCallModeAboutPrompt({
  character,
  userProfile,
  messages,
  customPrompt,
  historyLimit = 40,
  characterSystemPrompt = null,
  systemPromptMode = CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.flat,
}) {
  const usesSystemPrompt =
    (systemPromptMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.character_card
      || systemPromptMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation)
    && Boolean(characterSystemPrompt?.trim());

  const custom = customPrompt?.trim() || '';
  const instructionTemplate = custom
    ? custom
    : (usesSystemPrompt ? DEFAULT_CALL_MODE_ABOUT_INSTRUCTIONS : DEFAULT_CALL_MODE_ABOUT_PROMPT);

  if (usesSystemPrompt) {
    const body = applyContextPlaceholders(instructionTemplate, {
      character,
      userProfile,
      messages,
      historyLimit,
      characterSystemPrompt,
    });

    if (body.includes(characterSystemPrompt.trim())) {
      return body;
    }

    if (body.includes('{{CHARACTER_SYSTEM_PROMPT}}')) {
      return body;
    }

    return `${characterSystemPrompt.trim()}

═══════════════════════════════════════
CALL MODE — CHARACTER INTELLIGENCE SHEET
═══════════════════════════════════════

${body}`;
  }

  return applyContextPlaceholders(instructionTemplate, {
    character,
    userProfile,
    messages,
    historyLimit,
    characterSystemPrompt,
  });
}

const ABOUT_TEXT_FIELDS = [
  'headline',
  'essence',
  'presence',
  'relationship',
  'current_state',
  'story_so_far',
  'watch_for',
  'voice_note',
];

const FIELD_LABEL_STRIP = {
  headline: /^(headline|banner)\s*:\s*/i,
  essence: /^(essence|core)\s*:\s*/i,
  presence: /^(presence|on this call)\s*:\s*/i,
  relationship: /^(relationship|with you)\s*:\s*/i,
  current_state: /^(current_state|current state|right now)\s*:\s*/i,
  story_so_far: /^(story_so_far|story so far)\s*:\s*/i,
  watch_for: /^(watch_for|watch for)\s*:\s*/i,
  voice_note: /^(voice_note|voice)\s*:\s*/i,
};

function splitSentences(text) {
  return String(text)
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 8);
}

function sentenceKey(s) {
  return s.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();
}

export function normalizeCallModeAboutData(raw) {
  if (!raw || typeof raw !== 'object') return raw;

  const data = {
    headline: raw.headline || '',
    essence: raw.essence || raw.summary || '',
    presence: raw.presence || '',
    relationship: raw.relationship || '',
    current_state: raw.current_state || raw.currentState || '',
    themes: Array.isArray(raw.themes) ? raw.themes.filter(Boolean) : [],
    story_so_far: raw.story_so_far || raw.storySoFar || '',
    watch_for: raw.watch_for || raw.watchFor || '',
    voice_note: raw.voice_note || raw.voiceNote || '',
  };

  for (const field of ABOUT_TEXT_FIELDS) {
    let v = String(data[field] || '').trim();
    if (FIELD_LABEL_STRIP[field]) v = v.replace(FIELD_LABEL_STRIP[field], '').trim();
    data[field] = v;
  }

  for (const field of ABOUT_TEXT_FIELDS) {
    if (!data[field]) continue;
    const seen = new Set();
    const unique = [];
    for (const sent of splitSentences(data[field])) {
      const key = sentenceKey(sent);
      if (!key || seen.has(key)) continue;
      seen.add(key);
      unique.push(sent);
    }
    data[field] = unique.join(' ').trim();
  }

  data.themes = data.themes
    .map((t) => String(t).trim())
    .filter(Boolean)
    .map((t) => t.replace(/^["']|["']$/g, '').slice(0, 40))
    .slice(0, 4);

  return data;
}

function extractJsonObject(text) {
  if (!text) return null;
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
  }
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) {
    try {
      return JSON.parse(fenceMatch[1].trim());
    } catch {
    }
  }
  const start = trimmed.indexOf('{');
  const end = trimmed.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(trimmed.slice(start, end + 1));
    } catch {
    }
  }
  return null;
}

/** Plain-text section headers (Essence:, ## With you, etc.) when JSON parse fails. */
const PLAIN_SECTION_HEADERS = [
  { field: 'headline', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:headline|banner(?:\s+headline)?)\s*[:—\-]\s*/gi },
  { field: 'essence', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:essence|core)\s*[:—\-]\s*/gi },
  { field: 'presence', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:presence|on this call)\s*[:—\-]\s*/gi },
  { field: 'relationship', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:relationship|with you)\s*[:—\-]\s*/gi },
  { field: 'current_state', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:current[_\s-]?state|right now)\s*[:—\-]\s*/gi },
  { field: 'story_so_far', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:story[_\s-]?so[_\s-]?far|story recap)\s*[:—\-]\s*/gi },
  { field: 'watch_for', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:watch[_\s-]?for)\s*[:—\-]\s*/gi },
  { field: 'voice_note', pattern: /(?:^|\n)\s*(?:#{1,3}\s*)?(?:voice[_\s-]?note|voice)\s*[:—\-]\s*/gi },
];

function extractLabeledSectionsFromPlainText(text) {
  if (!text?.trim()) return null;

  const markers = [];
  for (const { field, pattern } of PLAIN_SECTION_HEADERS) {
    const re = new RegExp(pattern.source, pattern.flags);
    let match;
    while ((match = re.exec(text)) !== null) {
      markers.push({
        field,
        contentStart: match.index + match[0].length,
        headerStart: match.index,
      });
    }
  }

  if (markers.length === 0) return null;

  markers.sort((a, b) => a.headerStart - b.headerStart);
  const data = {};
  for (let i = 0; i < markers.length; i += 1) {
    const { field, contentStart } = markers[i];
    const end = i + 1 < markers.length ? markers[i + 1].headerStart : text.length;
    const chunk = text.slice(contentStart, end).trim();
    if (chunk) data[field] = data[field] ? `${data[field]} ${chunk}` : chunk;
  }
  return Object.keys(data).length > 0 ? data : null;
}

export function splitUnstructuredAboutText(text) {
  if (!text?.trim()) return null;

  const parsed = extractJsonObject(text);
  if (parsed && typeof parsed === 'object') {
    return normalizeCallModeAboutData(parsed);
  }

  const data = {};
  const fieldPatterns = [
    ['headline', /"headline"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['presence', /"presence"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['essence', /"essence"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['relationship', /"relationship"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['current_state', /"current_state"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['story_so_far', /"story_so_far"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['watch_for', /"watch_for"\s*:\s*"((?:\\.|[^"\\])*)"/i],
    ['voice_note', /"voice_note"\s*:\s*"((?:\\.|[^"\\])*)"/i],
  ];
  for (const [field, re] of fieldPatterns) {
    const m = text.match(re);
    if (m?.[1]) {
      try {
        data[field] = JSON.parse(`"${m[1]}"`);
      } catch {
        data[field] = m[1].replace(/\\"/g, '"');
      }
    }
  }

  const themesMatch = text.match(/"themes"\s*:\s*\[([\s\S]*?)\]/i);
  if (themesMatch) {
    try {
      const themes = JSON.parse(`[${themesMatch[1]}]`);
      if (Array.isArray(themes)) data.themes = themes.filter(Boolean);
    } catch {
      /* noop */
    }
  }

  if (Object.keys(data).length > 0) return normalizeCallModeAboutData(data);

  const labeled = extractLabeledSectionsFromPlainText(text);
  if (labeled && Object.keys(labeled).length > 0) {
    return normalizeCallModeAboutData(labeled);
  }

  const cleaned = text.replace(/```(?:json)?/gi, '').replace(/```/g, '').trim();
  if (!cleaned) return null;
  // Unlabeled blob: one neutral recap card (do not scatter sentences across mislabeled cards).
  return normalizeCallModeAboutData({ story_so_far: cleaned });
}

export function parseCallModeAboutResponse(rawText) {
  const parsed = extractJsonObject(rawText);
  if (parsed && typeof parsed === 'object') {
    return {
      structured: true,
      data: normalizeCallModeAboutData(parsed),
      rawText: rawText?.trim() || '',
    };
  }
  return {
    structured: false,
    data: null,
    rawText: rawText?.trim() || '',
  };
}

async function readGenerateStream(response, onPartial) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let sseBuffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    sseBuffer += decoder.decode(value, { stream: true });
    const events = sseBuffer.split('\n\n');
    sseBuffer = events.pop() || '';

    for (const line of events) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      try {
        const parsed = JSON.parse(data);
        const chunk = parsed.text ?? parsed.token ?? '';
        if (chunk) {
          fullText += chunk;
          onPartial?.(fullText);
        }
      } catch {
      }
    }
  }
  return fullText;
}

export async function fetchCallModeCharacterAbout({
  apiUrl,
  modelName,
  character,
  userProfile,
  messages,
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
    || settings.callModeAboutCharacterSystemPromptMode
    || CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.flat;

  const prompt = buildCallModeAboutPrompt({
    character,
    userProfile,
    messages,
    customPrompt: settings.callModeAboutCharacterPrompt,
    historyLimit: settings.callModeAboutCharacterHistoryLimit ?? 40,
    characterSystemPrompt: resolvedSystemPrompt,
    systemPromptMode: mode,
  });

  const flowConfig = resolveFlowGenerateConfig({
    flowKind: 'callModeAbout',
    settings,
    apiUrl,
    fallbackModelName: modelName,
  });
  const endpoint = flowConfig.url;

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify(
      buildFlowGenerateRequestBody({
        flowKind: 'callModeAbout',
        settings,
        apiUrl,
        fallbackModelName: modelName,
        basePayload: {
          prompt,
          max_tokens: settings.callModeAboutCharacterMaxTokens ?? 1200,
          temperature: settings.callModeAboutCharacterTemperature ?? 0.45,
          stop: ['```'],
          stream: true,
          gpu_id: 0,
          memoryEnabled: false,
          request_purpose: settings.callModeAboutCharacterRequestPurpose || CALL_MODE_ABOUT_REQUEST_PURPOSE,
          active_character: character || null,
          userProfile: userProfile ? { id: userProfile.id, name: userProfile.name } : null,
        },
      })
    ),
  });

  if (!response.ok) {
    throw new Error(await readFlowGenerateError(response));
  }

  const rawText = await readGenerateStream(response, onPartial);
  return parseCallModeAboutResponse(rawText);
}
