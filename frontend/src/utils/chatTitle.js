import { getBackendUrl } from '../config/api.js';

export const DEFAULT_CHAT_TITLE = 'New Chat';
export const CHAT_TITLE_MAX_LENGTH = 56;
const CHAT_TITLE_MAX_WORDS = 8;
const CHAT_TITLE_TIMEOUT_MS = 25_000;

function compactWhitespace(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function trimToBoundary(value, maxLength = CHAT_TITLE_MAX_LENGTH) {
  const text = compactWhitespace(value);
  if (text.length <= maxLength) return text;
  const clipped = text.slice(0, maxLength + 1);
  const boundary = clipped.lastIndexOf(' ');
  return (boundary >= Math.floor(maxLength * 0.6) ? clipped.slice(0, boundary) : clipped.slice(0, maxLength))
    .replace(/[\s,;:–—-]+$/g, '')
    .trim();
}

function limitWords(value, maxWords = CHAT_TITLE_MAX_WORDS) {
  const words = compactWhitespace(value).split(' ').filter(Boolean);
  return words.length > maxWords ? words.slice(0, maxWords).join(' ') : words.join(' ');
}

function stripMessageFormatting(value) {
  return String(value || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/gi, ' ')
    .replace(/^\s*(?:user|human|me)\s*:\s*/i, '')
    .replace(/^\s*[-*#>]+\s*/gm, '')
    .replace(/[*_~`]/g, ' ');
}

export function createFallbackChatTitle(message) {
  const cleaned = stripMessageFormatting(message);
  const firstLine = cleaned.split(/\r?\n/).map(compactWhitespace).find(Boolean) || '';
  const firstThought = firstLine.split(/(?<=[.!?])\s+/)[0] || firstLine;
  const title = trimToBoundary(limitWords(firstThought))
    .replace(/^["'“”‘’]+|["'“”‘’.,!?;:]+$/g, '')
    .trim();
  return title || DEFAULT_CHAT_TITLE;
}

function parsePossibleJsonTitle(value) {
  const text = String(value || '').trim();
  if (!text.startsWith('{')) return text;
  try {
    const parsed = JSON.parse(text);
    return parsed?.title || parsed?.name || text;
  } catch {
    return text;
  }
}

export function cleanGeneratedChatTitle(value, fallback = DEFAULT_CHAT_TITLE) {
  let title = String(value || '')
    .replace(/<think>[\s\S]*?<\/think>/gi, ' ')
    .replace(/```(?:json|text)?/gi, '')
    .replace(/```/g, '')
    .trim();
  title = parsePossibleJsonTitle(title);
  title = String(title || '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean) || '';
  title = title
    .replace(/^\s*[-*#>]+\s*/, '')
    .replace(/^(?:here(?:'s| is)\s+)?(?:(?:a|the)\s+)?(?:chat|conversation)?\s*title\s*[:–—-]\s*/i, '')
    .replace(/^[*_~"'“”‘’`]+|[*_~"'“”‘’`]+$/g, '')
    .replace(/[.!?;:]+$/g, '')
    .trim();
  title = trimToBoundary(limitWords(title));
  if (!title || /^(?:new chat|untitled|chat|conversation)$/i.test(title)) return fallback;
  return title;
}

export function buildChatTitlePrompt(messages = []) {
  const userMessage = messages.find((message) => message?.role === 'user')?.content || '';
  const assistantMessage = messages.find((message) =>
    message?.role === 'assistant' || message?.role === 'bot'
  )?.content || '';
  const parts = [
    'Write a specific, natural title for this conversation.',
    'Use 3 to 7 words. Capture the subject, scene, or intent rather than merely repeating the opening words.',
    'Do not use quotation marks, labels, markdown, or the words “chat” and “conversation”.',
    'Return only the title.',
    '',
    `First user message:\n${String(userMessage).slice(0, 1600)}`,
  ];
  if (assistantMessage) {
    parts.push('', `First assistant reply:\n${String(assistantMessage).slice(0, 1600)}`);
  }
  return parts.join('\n');
}

function resolveModelName(modelName) {
  if (typeof modelName === 'string') return modelName.trim();
  if (!modelName || typeof modelName !== 'object') return '';
  return String(modelName.id || modelName.name || modelName.model_name || modelName.model || '').trim();
}

function extractResponseText(data) {
  return data?.text
    || data?.response
    || data?.content
    || data?.choices?.[0]?.message?.content
    || data?.choices?.[0]?.text
    || '';
}

export async function generateChatTitle({
  messages = [],
  modelName,
  apiBaseUrl = null,
  selectedModel = null,
  roundRobinEnabled = false,
  fetchImpl = globalThis.fetch,
  timeoutMs = CHAT_TITLE_TIMEOUT_MS,
} = {}) {
  const firstUserMessage = messages.find((message) => message?.role === 'user')?.content || '';
  const fallback = createFallbackChatTitle(firstUserMessage);
  const resolvedModel = resolveModelName(modelName);
  if (!resolvedModel || typeof fetchImpl !== 'function') {
    return { title: fallback, source: 'fallback' };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const baseUrl = String(apiBaseUrl || getBackendUrl()).replace(/\/$/, '');
    const response = await fetchImpl(`${baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: buildChatTitlePrompt(messages),
        model_name: resolvedModel,
        selected_model: selectedModel || undefined,
        round_robin_enabled: Boolean(roundRobinEnabled),
        request_purpose: 'title_generation',
        temperature: 0.25,
        top_p: 0.9,
        max_tokens: 64,
        stream: false,
        gpu_id: 0,
        use_rag: false,
        use_web_search: false,
        memoryEnabled: false,
        injectTimestamp: false,
      }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Title request failed with HTTP ${response.status}`);
    const data = await response.json();
    const title = cleanGeneratedChatTitle(extractResponseText(data), fallback);
    return { title, source: title === fallback ? 'fallback' : 'generated' };
  } catch {
    return { title: fallback, source: 'fallback' };
  } finally {
    clearTimeout(timeout);
  }
}
