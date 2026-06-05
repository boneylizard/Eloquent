/**
 * Character-as-System mode: a Settings character card fills the base system layer (model
 * instructions, profile, agentic memory). The user's selected chat character still runs as
 * the roleplay "Character Persona" layer on top (one character through another system).
 */

import {
  CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES,
  buildCallModeAboutPrompt,
} from './callModeCharacterAbout.js';
import {
  CHARACTER_INTRO_MAX_ATTEMPTS,
  parseCharacterIntroResponse,
  isCharacterIntroReady,
  introJsonIsPartialUsable,
  buildCharacterIntroRepairPrompt,
} from './characterIntro.js';
import { buildFlowGenerateRequestBody, readFlowGenerateError, resolveFlowGenerateConfig } from './flowGenerateApi';

export const SYSTEM_INTRO_REQUEST_PURPOSE = 'system_intro';

/** Separator between system-persona layer and active chat character roleplay layer. */
export const CHARACTER_PERSONA_LAYER_MARKER = 'Character Persona:';

export function composeLayeredSystemPrompt(systemLayer, characterLayer) {
  const system = (systemLayer || '').trim();
  const character = (characterLayer || '').trim();
  if (!system) return character;
  if (!character) return system;
  return `${system}\n\n${CHARACTER_PERSONA_LAYER_MARKER}\n${character}`;
}

export function parseLayeredSystemPrompt(combined) {
  const text = (combined || '').trim();
  if (!text) return { systemLayer: '', characterLayer: '' };
  const idx = text.indexOf(CHARACTER_PERSONA_LAYER_MARKER);
  if (idx === -1) return { systemLayer: text, characterLayer: '' };
  return {
    systemLayer: text.slice(0, idx).trim(),
    characterLayer: text.slice(idx + CHARACTER_PERSONA_LAYER_MARKER.length).trim(),
  };
}

/** Human-readable split for debug / preview UIs (matches backend layer labels). */
export function formatLayeredPromptPreview(combined) {
  const { systemLayer, characterLayer } = parseLayeredSystemPrompt(combined);
  const parts = [];
  if (systemLayer) parts.push(`System Prompt:\n${systemLayer}`);
  if (characterLayer) parts.push(`Character Persona:\n${characterLayer}`);
  if (!parts.length && combined?.trim()) return `System Prompt:\n${combined.trim()}`;
  return parts.join('\n\n');
}

export const SYSTEM_INTRO_UI_LABELS = {
  headline: 'Overview',
  who_they_are: 'What this system is',
  how_they_engage: 'How it works with you',
  tone: 'Tone & style',
  voice_line: 'Sample line',
  themes: 'Themes',
};

const SYSTEM_INTRO_OUTPUT_SPEC = `You are writing a welcoming "about this system" sheet for someone starting a NEW chat (no prior messages). The system persona is defined by the system instructions below — not a roleplay character meeting the user.

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
• headline (≤14 words): A clear snapshot of what this assistant/system is for right now.
• who_they_are: Stable core of the system persona — purpose, expertise, boundaries. Ground in system instructions. 2–3 sentences. No fictional meeting framing.
• how_they_engage: How this system typically supports the user — collaboration style, constraints, memory use. Use user profile / agentic context when present. 2–3 sentences.
• tone: How responses should feel — pace, register, formality. 1–2 sentences.
• voice_line: ONE representative line (≤30 words) illustrating the system's voice. Not meta. Not "As an AI…".
• themes: 2–4 short tags (1–3 words each) for active motifs.

GLOBAL:
- Describe the SYSTEM persona, not a fictional character visiting the user.
- Ground claims in the data below; do not invent major facts.
- Fresh chat — do not reference nonexistent prior turns in this thread.
- Each fact belongs in exactly ONE field.`;

export const DEFAULT_SYSTEM_INTRO_PROMPT = `${SYSTEM_INTRO_OUTPUT_SPEC}

{{CHARACTER_BLOCK}}
{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

export const DEFAULT_SYSTEM_INTRO_INSTRUCTIONS = `${SYSTEM_INTRO_OUTPUT_SPEC}

Stay grounded in the system identity established above.

{{USER_BLOCK}}
{{STORY_BLOCK}}
{{CHAT_HISTORY}}`;

/**
 * Build system prompt from character card fields without roleplay wrapper.
 */
export function buildCharacterAsSystemPrompt(
  character,
  userProfile = null,
  summaryContextOverride = null,
  userCharacter = null,
  storyContext = ''
) {
  if (!character) return null;

  const summaryContext = summaryContextOverride
    ? `\n\n[PREVIOUS STORY SUMMARY]:\n${summaryContextOverride}\n[End of Summary]\n`
    : (userProfile?.activeContextSummary
      ? `\n\n[PREVIOUS STORY SUMMARY]:\n${userProfile.activeContextSummary}\n[End of Summary]\n`
      : '');

  const personaName = character.name || 'System';
  const userName = userCharacter?.name || userProfile?.name || userProfile?.username || 'User';

  const replaceTags = (text) => {
    if (!text) return '';
    return text
      .replace(/{{char}}/gi, personaName)
      .replace(/{{user}}/gi, userName);
  };

  const blocks = [];
  const ethics = character.ethics_justification ? replaceTags(character.ethics_justification).trim() : '';
  if (ethics) {
    blocks.push(
      `[SCOPE & SAFETY]\n${ethics}\n(Apply to this assistant's behavior; do not narrate this block.)`
    );
  }

  const modelInstructions = character.model_instructions ? replaceTags(character.model_instructions).trim() : '';
  if (modelInstructions) {
    blocks.push(modelInstructions);
  } else {
    const description = replaceTags(character.description);
    const personality = replaceTags(character.personality);
    if (description || personality) {
      blocks.push(
        [description, personality ? `Personality: ${personality}` : ''].filter(Boolean).join('\n\n')
      );
    }
  }

  const optional = [
    ['Background', character.background],
    ['Scenario', character.scenario],
    ['Speaking style', character.speech_style],
  ];
  for (const [label, value] of optional) {
    const t = value ? replaceTags(value).trim() : '';
    if (t) blocks.push(`${label}: ${t}`);
  }

  if (Array.isArray(character.example_dialogue) && character.example_dialogue.length) {
    const examples = character.example_dialogue
      .filter((m) => m?.content?.trim())
      .map((m) => {
        const role = m.role === 'user' ? userName : personaName;
        return `${role}: ${replaceTags(m.content)}`;
      })
      .join('\n');
    if (examples) blocks.push(`Example exchanges (style reference only):\n${examples}`);
  }

  let out = blocks.join('\n\n').trim();
  if (!out) {
    out = `You are a helpful assistant (${personaName}).`;
  }
  return `${out}${summaryContext}${storyContext || ''}`.trim();
}

export function buildSystemAboutPrompt({
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
      ? DEFAULT_SYSTEM_INTRO_INSTRUCTIONS
      : DEFAULT_SYSTEM_INTRO_PROMPT);

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
        /* ignore */
      }
    }
  }
  return fullText;
}

export async function fetchSystemIntro({
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
    || settings.systemIntroSystemPromptMode
    || settings.characterIntroSystemPromptMode
    || settings.callModeAboutCharacterSystemPromptMode
    || CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation;

  const prompt = buildSystemAboutPrompt({
    character,
    userProfile,
    messages,
    customPrompt: settings.systemIntroPrompt || settings.characterIntroPrompt,
    historyLimit: settings.systemIntroHistoryLimit ?? settings.characterIntroHistoryLimit ?? 8,
    characterSystemPrompt: resolvedSystemPrompt,
    systemPromptMode: mode,
  });

  const flowConfig = resolveFlowGenerateConfig({
    flowKind: 'systemIntro',
    settings,
    apiUrl,
    fallbackModelName: modelName,
  });
  const endpoint = flowConfig.url;

  const maxAttempts = settings.systemIntroMaxAttempts ?? settings.characterIntroMaxAttempts ?? CHARACTER_INTRO_MAX_ATTEMPTS;
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
          flowKind: 'systemIntro',
          settings,
          apiUrl,
          fallbackModelName: modelName,
          basePayload: {
            prompt: attemptPrompt,
            max_tokens: settings.systemIntroMaxTokens ?? settings.characterIntroMaxTokens ?? 900,
            temperature: settings.systemIntroTemperature ?? settings.characterIntroTemperature ?? 0.55,
            stop: ['```'],
            stream: true,
            gpu_id: 0,
            memoryEnabled: false,
            request_purpose: settings.systemIntroRequestPurpose || SYSTEM_INTRO_REQUEST_PURPOSE,
            system_persona_mode: true,
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

  if (bestPartialResult) return bestPartialResult;
  return {
    ...lastResult,
    structured: false,
    rawText: lastRaw,
    rawExcerpt: (lastRaw || '').slice(0, 500),
  };
}

export function resolveSystemPersonaCharacter(characters, settings, conversation = null) {
  const id =
    conversation?.systemPersonaCharacterId
    || settings?.systemPersonaCharacterId
    || null;
  if (!id) return null;
  return (characters || []).find((c) => c.id === id) || null;
}

export function isSystemPersonaModeActive(settings, conversation = null) {
  if (conversation != null) {
    return conversation.systemPersona === true;
  }
  return settings?.useCharacterAsSystemPrompt === true && !!settings?.systemPersonaCharacterId;
}
