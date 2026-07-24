import React, { memo, useMemo, createContext, useState, useCallback, useContext, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import { formatPrompt } from '../utils/chat_templates';
import { mergeRollingMemoryPack } from '../utils/conversationRollingMemory';
import { cleanModelOutput } from '../utils/cleanOutput';
import {
  buildBotReasoningFinalizePatch,
  createReasoningStreamController,
} from '../utils/thinkStreamParser';
import { inferCapabilitiesFromModelId } from '../utils/resolveEndpointDisplay';
import { extractSseStreamParts } from '../utils/streamDelta';
import { formatFetchError, loadPortConfig, getConfig } from '../config/api';
import { fetchTriggeredLore } from '../utils/apiCall';
import { generateChatTitle } from '../utils/chatTitle';
import { observeConversation, initializeMemories } from '../utils/memoryUtils';
import { saveSummary } from '../utils/summaryUtils';
import { useMemory } from '../contexts/MemoryContext';
import { transcribeAudio, synthesizeSpeech, getLastTtsSynthesisMeta } from '../utils/apiCall';
import { createWavMicRecorder } from '../utils/wavMicRecorder';
import { generateReplyOpenAI, processOpenAIStream, generateReplyOpenAINonStreaming, convertToOpenAIMessages } from '../utils/openaiApi';
import { ttsClient } from '../utils/apiCall';
import { isIntelMessageId } from '../utils/callModeIntelTts';
import { getTtsPlaybackRate, normaliseTtsSpeed } from '../utils/ttsPlaybackPolicy';
import { processAntiRepetition } from '../utils/antiRepetition';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import { fetchWithTimeout } from '../config/api';
import {
  withTimeout,
  PORT_CONFIG_TIMEOUT_MS,
  STORAGE_HYDRATION_TIMEOUT_MS,
  SETTINGS_STANDALONE_STORAGE_TIMEOUT_MS,
} from '../utils/appBoot';
import { isSettingsStandaloneWindow } from '../utils/settingsCrossWindowSync';
import {
  loadConversationsFromStorage,
  loadConversationMessages,
  saveActiveConversationMessages,
  mutateStoredConversationMessages,
  loadTombstonedConversationIds,
  persistChatState,
  saveConversationCatalog,
  deleteConversationFromStorage,
  deleteConversationsFromStorage,
  deleteAllConversationsFromStorage,
  scrubConversationLocalStorageGhosts,
  banConversationIdSync,
  getBannedConversationIdsSync,
  installChatStorageDebugHelpers,
  recoverConversationCatalogEntry,
  persistOutreachConversation,
  purgeOutreachConversationsFromStorage,
  isOutreachConversationId,
} from '../utils/conversationStorage';
import {
  appendUniqueConversationMessages,
  updateConversationMessageById,
} from '../utils/mediaConversation';
import { mergeNanoGptMemoryIntoPayload } from '../utils/nanoGptMemoryPayload';
import {
  buildBookChapterJsonOutlineUserMessage,
  parseChapterJsonOutlineFromModel,
} from '../utils/bookChapterJsonOutline';
import {
  API_CONTEXT_WINDOW_TOKENS_DEFAULT,
  API_CONTEXT_WINDOW_MAX,
  clampApiContextWindowTokens,
} from '../config/apiContextLimits';
import { getWebSearchResearchPayload } from '../utils/webSearchResearch';
import {
  normalizeCharacterAvatars,
  omitPersistedLocalAvatarFolder,
  mergeSessionLocalAvatarFolder,
  getActiveCharacterAvatar,
  cycleAvatarIndex,
  setAvatarIndexOnCharacter,
} from '../utils/characterAvatars';
import {
  broadcastPrimaryModelState,
  broadcastSettingsPatch,
  broadcastSettingsReload,
  openSettingsPopupWindow,
  readLastPrimaryApiModel,
  saveLastPrimaryApiModel,
  subscribeAppCrossWindowSync,
} from '../utils/settingsCrossWindowSync';
import {
  applyAvatarSizesToSettings,
  mergeSettingsObjects,
  persistSettingsBlob,
  readSettingsFromLocalStorageSync,
  readSettingsFromStorage,
  shouldApplyHydratedSettings,
} from '../utils/settingsPersistence';
import {
  clearInstallerAudioProfile,
  getAudioStartupDefaultsMigration,
  readInstallerAudioProfile,
} from '../utils/installerAudioProfile';
import {
  buildCharacterIntroSeedMessages,
  conversationAcceptsIntroTitle,
  deriveIntroChatTitle,
} from '../utils/characterIntro';
import {
  buildCharacterAsSystemPrompt,
  composeLayeredSystemPrompt,
  isSystemPersonaModeActive,
  resolveSystemPersonaCharacter,
} from '../utils/systemPersona';
import {
  buildCharacterGreetingOptions,
  createCharacterGreetingState,
  resolveCharacterPostHistoryInstructions,
  resolveCharacterPromptOverride,
} from '../utils/characterCardRuntime';
import {
  attachApiBotSpeakerMeta,
  getRotationPool,
  resolveEndpointDisplay,
} from '../utils/resolveEndpointDisplay';
import {
  assertRouteContractOrThrow,
  createRouteTraceId,
  extractRouteMetaFromGenerateResult,
  logRouteTrace,
  resolveUnifiedRequestRoute,
} from '../utils/requestRouting';
import {
  createThinkingStreamChunkLogger,
  isThinkingStreamDebugEnabled,
} from '../utils/thinkingStreamDebug';
import {
  readNanoGptModelsCache,
  refreshNanoGptModelsCache,
  findNanoGptModel,
} from '../utils/nanoGptModelsCache';
import { formatApiError } from '../utils/chatlogCondenserUtils';
import {
  MODEL_DEFAULT_CHAT_TEMPLATE_ID,
  getChatTemplateRequestFields,
} from '../utils/chatTemplateSelection';

/** Estimate the duration (ms) of a queued TTS audio chunk without fully
 *  decoding it.  Uses the backend-provided subtitle cue when available,
 *  otherwise parses the WAV header to compute duration from the data
 *  subchunk size and byte rate. */
function estimateChunkDurationMs(chunk) {
  if (!chunk) return 0;
  const sub = chunk.subtitle;
  if (sub && typeof sub.durationMs === 'number' && sub.durationMs > 0) {
    return sub.durationMs;
  }
  const ab = chunk.audio || chunk;
  if (!(ab instanceof ArrayBuffer) || ab.byteLength < 44) return 0;
  try {
    const view = new DataView(ab);
    if (view.getUint32(0, false) !== 0x52494646) {
      console.warn(`⚠️ [TTS] estimateChunkDurationMs: NOT WAV format (first bytes: 0x${view.getUint32(0, false).toString(16)}), chunkLen=${ab.byteLength}`);
      return 0;
    }
    const byteRate = view.getUint32(28, true);
    if (!byteRate) return 0;
    let offset = 12;
    while (offset + 8 <= ab.byteLength) {
      const id = view.getUint32(offset, false);
      const size = view.getUint32(offset + 4, true);
      if (id === 0x64617461) {
        const ms = (size / byteRate) * 1000;
        return ms;
      }
      offset += 8 + size + (size & 1); // word-aligned chunks
    }
    return 0;
  } catch (e) {
    return 0;
  }
}

const defaultAppContextValue = {
  activeTab: 'chat',
  setActiveTab: () => {},
  settingsEntryTab: 'general',
  openSettingsTab: () => {},
  openSettingsWindow: () => {},
  appendMessagesToConversation: async () => false,
  updateMessageInConversation: async () => false,
};
const AppContext = createContext(defaultAppContextValue);
const logPromptSample = (prompt, maxLength = 500) => {
  const sample = prompt.length > maxLength ?
    prompt.substring(0, maxLength) + '...' :
    prompt;
  console.log('📝 [DEBUG] Prompt sample:', sample);

  const hasMemories = prompt.includes('USER CONTEXT') ||
    prompt.includes('RELEVANT USER INFORMATION') ||
    prompt.includes('WORLD KNOWLEDGE');

  console.log(`📝 [DEBUG] Prompt contains memory references: ${hasMemories}`);
};
/** Cap memory-service waits so chat never hangs silently if :8001 is down. */
const MEMORY_FETCH_TIMEOUT_MS = 12000;
/** API-only chat: short cap so /generate is not delayed waiting on memory GPU. */
const MEMORY_FETCH_TIMEOUT_API_MS = 2500;

// NEW — talk to your FastAPI on 8001
async function generateAndShowImage(promptText) {
  try {
    // 1) Build the body payload
    const payload = {
      prompt: promptText,
      negative_prompt: "",
      width: 512,
      height: 512,
      steps: 20,
      guidance_scale: 7.0,
      sampler_name: "Euler a",
      seed: -1,
    };

    // 2) Call your FastAPI proxy
    const res = await fetch(`${BACKEND}/sd/txt2img`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`SD API ${res.status}: ${errText}`);
    }

    // 3) Parse the JSON
    const data = await res.json();
    console.log("📝 [DEBUG] Response data:", data);

    // 4) Pull out the image URLs array
    const urls = data.image_urls || [];
    if (!urls.length) {
      console.error("📝 [ERROR] No images returned:", data);
      return;
    }

    // 5) Use the first one
    const imageUrl = urls[0];
    console.log("📝 [DEBUG] Image URL:", imageUrl);

    // 6) Append to the DOM
    // const img = document.createElement("img");
    // img.src = imageUrl;
    // img.alt = promptText;
    // document.body.appendChild(img);

  } catch (err) {
    console.error("📝 [ERROR] generateAndShowImage failed:", err);
  }
}

// Example usage:
//generateAndShowImage("a sexy woman");
// Helper function to generate truly unique IDs
const generateUniqueId = () => `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
const NARRATOR_CHARACTER_ID = '__narrator__';
const OUTREACH_DEFAULT_INTERVAL_MINUTES = 45;
const OUTREACH_MIN_INTERVAL_MINUTES = 1;

function outreachNotificationFromConversation(conv) {
  const messages = Array.isArray(conv?.messages) ? conv.messages : [];
  const botMsg = [...messages].reverse().find((m) => m?.role === 'bot' || m?.role === 'assistant');
  if (!botMsg) return null;
  const previewSrc = typeof botMsg.content === 'string' ? botMsg.content : '';
  return {
    id: botMsg.id ? `outreach-note-${botMsg.id}` : `outreach-note-${conv.id}`,
    ruleId: conv.outreachRuleId || conv.ruleId,
    ruleName: conv.outreachRuleName || conv.ruleName || 'Scheduled Outreach',
    characterName: conv.characterSnapshot?.name || botMsg.characterName || conv.name || 'Character',
    characterAvatar: conv.characterSnapshot?.avatar || botMsg.avatar || null,
    attachmentImageUrl: botMsg.imagePath || botMsg.attachmentImageUrl || null,
    messageId: botMsg.id,
    preview: previewSrc.replace(/\s+/g, ' ').slice(0, 200),
    conversationId: conv.id,
    createdAt: conv.updatedAt || conv.created || new Date().toISOString(),
    read: false,
  };
}

const API_CONTEXT_WINDOW_TOKENS = API_CONTEXT_WINDOW_TOKENS_DEFAULT;
const API_CONTEXT_SAFETY_BUFFER_TOKENS = 1024;

const estimateTokenCount = (text = '') => Math.ceil(String(text).length / 4);

const selectApiHistoryWithinContext = ({
  messages,
  systemPrompt,
  maxContextTokens = API_CONTEXT_WINDOW_TOKENS,
  reservedOutputTokens = 2048,
  minMessages = 1,
  maxHistoryTokens = null
}) => {
  const chatMessages = (messages || []).filter(
    (msg) =>
      (msg?.role === 'user' || msg?.role === 'bot') &&
      typeof msg?.content === 'string' &&
      msg.content.length > 0
  );

  if (!chatMessages.length) return [];

  // Always preserve the newest user turn verbatim. History pruning can drop older turns,
  // but it must never trim/remove the actual prompt the user just sent.
  const newestUserMessageId = [...chatMessages]
    .reverse()
    .find((msg) => msg?.role === 'user')?.id || null;

  const systemTokens = estimateTokenCount(systemPrompt);
  const availableForHistory = Math.max(
    0,
    maxContextTokens - systemTokens - reservedOutputTokens - API_CONTEXT_SAFETY_BUFFER_TOKENS
  );
  const historyCap =
    maxHistoryTokens != null
      ? Math.min(availableForHistory, Math.max(0, maxHistoryTokens))
      : availableForHistory;

  const reversed = [...chatMessages].reverse();
  const selected = [];
  let usedTokens = 0;

  for (let i = 0; i < reversed.length; i += 1) {
    const msg = reversed[i];
    const messageTokens = estimateTokenCount(msg.content) + 12; // role + separators overhead
    const mustKeepForContinuity = i < minMessages;
    const isPinnedNewestUser = newestUserMessageId && msg.id === newestUserMessageId;

    if (!mustKeepForContinuity && !isPinnedNewestUser && usedTokens + messageTokens > historyCap) {
      break;
    }

    selected.unshift(msg);
    // The newest user prompt is pinned outside the history budget.
    if (!isPinnedNewestUser) {
      usedTokens += messageTokens;
    }
  }

  return selected;
};

const normalizeChatRole = (role) => {
  if (role === 'user' || role === 'narrator') return role;
  return 'npc';
};

const normalizeCharacter = (character) =>
  normalizeCharacterAvatars({
    ...character,
    chat_role: normalizeChatRole(character?.chat_role),
  });

/** Avoid setState loops when the characters list is re-built with new object refs. */
const charactersMatchForSync = (stored, selected) => {
  if (stored === selected) return true;
  if (!stored || !selected || stored.id !== selected.id) return false;
  try {
    return JSON.stringify(stored) === JSON.stringify(selected);
  } catch {
    return false;
  }
};

const buildDefaultCharacterWeights = (characterList) => {
  const weights = {};
  (characterList || [])
    .map(normalizeCharacter)
    .filter(c => normalizeChatRole(c.chat_role) !== 'user')
    .forEach(c => {
      weights[c.id] = 50;
    });
  return weights;
};

const extractFirstJson = (text) => {
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
};

const getStoryTrackerContext = () => '';

// Helper function to build system prompt from character data
const _buildSystemPrompt = (character, userProfile = null, summaryContextOverride = null, userCharacter = null) => {
  if (!character) return null;

  // Get active summary context if selected
  const summaryContext = summaryContextOverride
    ? `\n\n[PREVIOUS STORY SUMMARY]:\n${summaryContextOverride}\n[End of Summary]\n`
    : (userProfile?.activeContextSummary
      ? `\n\n[PREVIOUS STORY SUMMARY]:\n${userProfile.activeContextSummary}\n[End of Summary]\n`
      : '');

  // Get story tracker context
  const storyContext = getStoryTrackerContext();

  // Tag substitution variables
  const charName = character.name || 'Character';
  const userName = userCharacter?.name || userProfile?.name || userProfile?.username || 'User';

  // Helper to replace tags in a string
  const replaceTags = (text) => {
    if (!text) return '';
    return text
      .replace(/{{char}}/gi, charName)
      .replace(/{{user}}/gi, userName);
  };

  const personality = replaceTags(character.personality);
  const description = replaceTags(character.description);
  const scenario = character.scenario ? replaceTags(character.scenario) : '';
  const speechStyle = character.speech_style ? replaceTags(character.speech_style) : '';
  const background = character.background ? replaceTags(character.background) : '';
  const ethics = character.ethics_justification ? replaceTags(character.ethics_justification).trim() : '';
  const defaultBehaviorInstructions =
    `Stay in character at all times. Respond as ${charName} would, maintaining the defined personality and speech patterns.`;
  const behaviorInstructions = resolveCharacterPromptOverride(
    character.model_instructions,
    defaultBehaviorInstructions,
    charName,
    userName,
  );

  const prefixParts = [];
  if (ethics) {
    const ethicsScopeNote =
      '(Out-of-character context only: not a persona to perform. Do not speak as this section\'s narrator, adopt its voice, or treat it as a second character. Use it only to interpret scope and refusals for the role of ' +
      charName +
      ' below.)';
    prefixParts.push(`[ETHICS JUSTIFICATION]\n${ethicsScopeNote}\n\n${ethics}`);
  }
  const eloquentPrefix = prefixParts.length ? `${prefixParts.join('\n\n')}\n\n` : '';

  const ethicsTail = ethics
    ? '\n\n(When scope or refusals are ambiguous, use the ETHICS JUSTIFICATION at the start of this prompt; do not roleplay that block.)\n'
    : '';

  return `${eloquentPrefix}You are ${charName}, ${description}.

PERSONALITY: ${personality}

BACKGROUND: ${background}

${scenario ? `SCENARIO: ${scenario}` : ''}

SPEAKING STYLE: ${speechStyle}

[CHARACTER SYSTEM PROMPT]
${behaviorInstructions}${ethicsTail}
${character.example_dialogue && character.example_dialogue.length > 0
      ? `EXAMPLE DIALOGUE:
${character.example_dialogue.map(msg =>
        `${replaceTags(msg.role === 'character' ? charName : (msg.role === 'user' ? userName : 'User'))}: ${replaceTags(msg.content)}`
      ).join('\n')}` : ''}${summaryContext}${storyContext}`;
};


// Helper function to draw avatar on canvas
const drawAvatar = (canvas, imageUrl, name) => {
  const ctx = canvas.getContext('2d');
  const size = canvas.width;

  // Clear canvas
  ctx.clearRect(0, 0, size, size);

  if (imageUrl) {
    const img = new Image();
    img.crossOrigin = "Anonymous"; // To avoid CORS issues
    img.onload = () => {
      // Draw the image, clipped to a circle
      ctx.beginPath();
      ctx.arc(size / 2, size / 2, size / 2, 0, 2 * Math.PI);
      ctx.closePath();
      ctx.clip();
      ctx.drawImage(img, 0, 0, size, size);
    };
    img.src = imageUrl;
  } else {
    // Draw placeholder
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#89b4fa';  // Placeholder color
    ctx.fill();
    ctx.closePath();
    ctx.fillStyle = '#1e1e2e'; // Text color
    ctx.font = `${Math.round(size / 3)}px sans-serif`; // Dynamic font size
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name.charAt(0).toUpperCase(), size / 2, size / 2);
  }
};

const AppProvider = ({ children }) => {
  const memoryContext = useMemory();
  const userProfile = memoryContext?.userProfile;
  const [activeContextSummary, setActiveContextSummary] = useState(null);
  const roleplayEnabledRef = useRef(false);
  const userCharacterRef = useRef(null);
  const summaryContextForRequest = useMemo(() => {
    const summary = (activeContextSummary || userProfile?.activeContextSummary || '').trim();
    return summary || null;
  }, [activeContextSummary, userProfile?.activeContextSummary]);
  const buildSystemPrompt = useCallback(
    (char) => _buildSystemPrompt(
      char,
      userProfile,
      summaryContextForRequest ? null : activeContextSummary,
      roleplayEnabledRef.current ? userCharacterRef.current : null
    ),
    [userProfile, activeContextSummary, summaryContextForRequest]
  );

  const buildSystemPersonaPrompt = useCallback(
    (char) => buildCharacterAsSystemPrompt(
      char,
      userProfile,
      summaryContextForRequest || activeContextSummary || null,
      roleplayEnabledRef.current ? userCharacterRef.current : null,
      getStoryTrackerContext()
    ),
    [userProfile, activeContextSummary, summaryContextForRequest]
  );

  const getRelevantMemories = memoryContext?.getRelevantMemories;
  const addConversationSummary = memoryContext?.addConversationSummary;
  const [sdStatus, setSdStatus] = useState({});
  const [generatedImages, setGeneratedImages] = useState([]);
  const [isImageGenerating, setIsImageGenerating] = useState(false);
  const [apiError, setApiError] = useState(null);
  const clearError = useCallback(() => setApiError(null), []);
  const [activeTab, setActiveTab] = useState('chat'); // Default to 'chat' tab
  const activeTabRef = useRef('chat');
  /** When opening Settings (sidebar or mobile remote), which inner tab to show. */
  const [settingsEntryTab, setSettingsEntryTab] = useState('general');
  const [sttEnabled, setSttEnabled] = useState(true);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [userAvatar, setUserAvatar] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [loadedModels, setLoadedModels] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(null); // Store message ID being played, or null
  const [ttsPlaybackState, setTtsPlaybackState] = useState({ messageId: null, phase: 'idle' });
  const [audioError, setAudioError] = useState(null);
  const [primaryModel, setPrimaryModel] = useState(null);
  const [secondaryModel, setSecondaryModel] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  const [isChunkPlaying, setIsChunkPlaying] = useState(false);
  const [speechDetected, setSpeechDetected] = useState(false);
  const [audioAnalyzer, setAudioAnalyzer] = useState(null);
  const [isCallModeActive, setIsCallModeActive] = useState(false);
  const [callModeRecording, setCallModeRecording] = useState(false);

  // At the top of AppProvider, alongside primaryModel / secondaryModel:
  const [primaryCharacter, setPrimaryCharacter] = useState(null);
  const [secondaryCharacter, setSecondaryCharacter] = useState(null);
  const [primaryAvatar, setPrimaryAvatar] = useState(null);
  const [secondaryAvatar, setSecondaryAvatar] = useState(null);
  const mediaRecorderRef = useRef(null); // legacy export; STT uses wavMicRecorderRef
  const wavMicRecorderRef = useRef(null);
  const isFirstTextChunk = useRef(true);
  const isTtsInterruptedRef = useRef(false);
  const callModeMediaRecorderRef = useRef(null);
  const callModeAudioChunksRef = useRef([]);
  const activeAudioPlayersRef = useRef(new Set()); // Track ALL active audio sources
  const streamingTtsDrainingRef = useRef(false); // Single drain runner for WS TTS (avoids stuck "playing" gate)
  const streamingTtsDrainGenerationRef = useRef(null); // Only the owning generation may release the drain lock
  const drainGenerationRef = useRef(0); // Incremented on each startStreamingTTS to kill stale drain loops
  /** Tracks whether the TTS WebSocket is in a closed/closing state to abort the drain loop. */
  const ttsWsClosedRef = useRef(false);
  const callModeSilenceTimerRef = useRef(null);
  const callModeStreamRef = useRef(null);
  const audioChunksRef = useRef([]); // Updated to match previous edit
  const audioPlayerRef = useRef(null); // Ref to store the current Audio object for TTS
  const audioPlaybackRef = useRef({ context: null }); // Ref for persistent AudioContext (mobile unlock)
  const [inputTranscript, setInputTranscript] = useState(''); // State for input transcript
  const [agentMemories, setAgentMemories] = useState([]);
  const lastPlayedMessageRef = useRef(null);
  const ttsPlaybackRequestRef = useRef(0); // Invalidates superseded full-response synthesis requests
  const [sttEnginesAvailable, setSttEnginesAvailable] = useState(['whisper', 'parakeet', 'nemotron']);
  const [nanogptSttModels, setNanogptSttModels] = useState([]);
  const [nanogptTtsModels, setNanogptTtsModels] = useState([]);
  const [parakeetCppModels, setParakeetCppModels] = useState([]);
  const [parakeetCppCliAvailable, setParakeetCppCliAvailable] = useState(false);
  const [voxcpmGgufModels, setVoxcpmGgufModels] = useState([]);
  const [voxcpmGgufCliAvailable, setVoxcpmGgufCliAvailable] = useState(false);
  const [manuallyStoppedAudio, setManuallyStoppedAudio] = useState(false);
  const [applyAvatar, setApplyAvatar] = useState(false);
  const [activeAvatar, setActiveAvatar] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false); // Loading state for transcription
  const [documents, setDocuments] = useState({ file_list: [] });
  const [isGenerating, setIsGenerating] = useState(false);
  const textGenerationStartTime = useRef(null);
  const speechStartTime = useRef(null);
  const promptSubmissionStartTime = useRef(null); // ADD THIS LINE
  const [dualModeEnabled, setDualModeEnabled] = useState(false);
  const [agentConversationActive, setAgentConversationActive] = useState(false);
  const [autoMemoryEnabled, setAutoMemoryEnabled] = useState(true); // Default to enabled
  const [lastAgenticMemoryFeedback, setLastAgenticMemoryFeedback] = useState(null); // { added, characterName } when agentic memory adds insights
  const [lastAgenticRunStatus, setLastAgenticRunStatus] = useState(null); // 'ok' | 'error' | null — reflects whether backend actually ran (so UI doesn't lie)
  /** Last GET /memory/agentic inject into system prompt (honest UI tooltip; not console spam). */
  const [lastAgenticInjectMeta, setLastAgenticInjectMeta] = useState(null); // { chars, count, characterName } | null
  const [alignmentData, setAlignmentData] = useState(null); // { count, highestSeverity, findings, frameFidelity } or null
  const [alignmentDetectionEnabled, setAlignmentDetectionEnabled] = useState(false);
  const [autoDeleteChats, setAutoDeleteChats] = useState(false); // Default to false
  // avatar sizing for chat
  const [userAvatarSize] = useState(64);
  const streamingTtsMessageIdRef = useRef(null);
  const streamingTtsStreamOptsRef = useRef(null);
  /** True after `endStreamingTTS` sent `--END--` (no more text). Playback may still be synthesizing. */
  const streamingTtsWsEndSentRef = useRef(false);
  /** Batches `addStreamingText` onto rAF so we do not WS-send once per token (and sync manual chunk bursts coalesce). */
  const ttsStreamSendCoalesceRef = useRef({ pending: '', rafId: null });
  const ttsFullResponseBufferRef = useRef('');
  const ttsWaitForFullResponseRef = useRef(false);
  const ttsPrebufferSecondsRef = useRef(0);
  const [characterAvatarSize] = useState(64);
  const [showAvatars, setShowAvatars] = useState(true);
  const [showAvatarsInChat, setShowAvatarsInChat] = useState(true);
  const [abortController, setAbortController] = useState(null);
  const [isStreamingStopped, setIsStreamingStopped] = useState(false);
  const [audioQueue, setAudioQueue] = useState([]);
  const [isAutoplaying, setIsAutoplaying] = useState(false);
  const [isStreamingTtsPaused, setIsStreamingTtsPaused] = useState(false);
  const [ttsSubtitleCue, setTtsSubtitleCue] = useState(null);
  const [ttsFullResponseSaveStatus, setTtsFullResponseSaveStatus] = useState({
    state: 'idle',
    message: '',
    path: null,
    filename: null,
    chunkCount: null,
    updatedAt: 0,
  });
  const ttsSubtitleCueRef = useRef(null); // ✅ Ref for direct access, bypassing re-render issues
  const isStreamingTtsPausedRef = useRef(false);
  const isMobileRef = useRef(false);

  // Inject timestamp into AI context (single source of truth — send + regenerate read this ref)
  const [injectTimestamp, setInjectTimestampState] = useState(() => typeof localStorage !== 'undefined' && localStorage.getItem('eloquent-inject-timestamp') === 'true');
  const injectTimestampRef = useRef(injectTimestamp);
  injectTimestampRef.current = injectTimestamp;
  const setInjectTimestamp = useCallback((value) => {
    setInjectTimestampState(!!value);
    if (typeof localStorage !== 'undefined') {
      if (value) localStorage.setItem('eloquent-inject-timestamp', 'true');
      else localStorage.removeItem('eloquent-inject-timestamp');
    }
  }, []);

  // Key profile phrases re-injected before user query (backend uses for "repetition injection" / context saturation)
  const profileReinforcementRef = useRef('');

  // Debug: Monitor state changes
  useEffect(() => {
    ttsSubtitleCueRef.current = ttsSubtitleCue; // ✅ Keep ref in sync
  }, [ttsSubtitleCue]);

  useEffect(() => {
    if (typeof navigator === 'undefined') return;
    const ua = navigator.userAgent || '';
    isMobileRef.current = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
  }, []);

  const audioContextRef = useRef(null); // To manage the Web Audio API context
  const [primaryIsAPI, setPrimaryIsAPI] = useState(false);
  const apiModelRestoredRef = useRef(false);
  const postStorageHydrateFetchRef = useRef(false);
  const crossWindowSyncReadyRef = useRef(false);
  const autoRouterBootLogRef = useRef(false);
  const [secondaryIsAPI, setSecondaryIsAPI] = useState(false);
  const setUserAvatarSize = (size) => {
    updateSettings({ userAvatarSize: size });
  };
  const setCharacterAvatarSize = (size) => {
    updateSettings({ characterAvatarSize: size });
  };





  // Character management states
  const [characters, setCharacters] = useState([]);
  const [activeCharacter, setActiveCharacter] = useState(null);
  const [userCharacterId, setUserCharacterId] = useState(null);
  const userCharacter = useMemo(() => {
    if (!characters || characters.length === 0) return null;
    return userCharacterId ? characters.find(c => c.id === userCharacterId) || null : null;
  }, [characters, userCharacterId]);
  useEffect(() => {
    userCharacterRef.current = userCharacter;
  }, [userCharacter]);
  const activeCharacterRef = useRef(null);
  useEffect(() => {
    activeCharacterRef.current = activeCharacter;
  }, [activeCharacter]);
  // Fix: Keep primaryCharacter and secondaryCharacter in sync with characters list updates
  useEffect(() => {
    if (primaryCharacter) {
      const updatedPrimary = characters.find(c => c.id === primaryCharacter.id);
      if (updatedPrimary && !charactersMatchForSync(updatedPrimary, primaryCharacter)) {
        setPrimaryCharacter(updatedPrimary);
      }
    }
    if (secondaryCharacter) {
      const updatedSecondary = characters.find(c => c.id === secondaryCharacter.id);
      if (updatedSecondary && !charactersMatchForSync(updatedSecondary, secondaryCharacter)) {
        setSecondaryCharacter(updatedSecondary);
      }
    }
  }, [characters, primaryCharacter, secondaryCharacter]);

  // Same for activeCharacter (e.g. Quick Voice / roster saves ttsVoice on the list, but TTS used stale ref)
  useEffect(() => {
    if (!activeCharacter?.id) return;
    const updated = characters.find((c) => c.id === activeCharacter.id);
    if (updated && !charactersMatchForSync(updated, activeCharacter)) {
      setActiveCharacter(updated);
    }
  }, [characters, activeCharacter]);

  const [backgroundImage, setBackgroundImage] = useState(null); // New state for chat background
  const [roomGalleryOpen, setRoomGalleryOpen] = useState(false);

  // Refs for avatar canvases
  const primaryAvatarRef = useRef(null);
  const secondaryAvatarRef = useRef(null);
  /** Refs updated every render so async sendMessage always reads latest (fixes "only works in new chats"). */
  const conversationsRef = useRef([]);
  const activeConversationRef = useRef(null);
  /** Tracks rolling-memory compaction jobs per conversation (avoid blocking replies / avoid duplicate jobs). */
  const rollingMemoryCompactionInFlightRef = useRef({});
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);
  const [activeCharacterIds, setActiveCharacterIds] = useState([]);
  const [activeCharacterWeights, setActiveCharacterWeights] = useState({});
  const [multiRoleContext, setMultiRoleContext] = useState('');
  const [storageHydrated, setStorageHydrated] = useState(
    () => typeof window !== 'undefined' && isSettingsStandaloneWindow(),
  );
  const [storageHydrationDegraded, setStorageHydrationDegraded] = useState(false);
  const [portsLoadDegraded, setPortsLoadDegraded] = useState(false);
  const [bootGeneration, setBootGeneration] = useState(0);
  /** Last IndexedDB message save for active tab (UI feedback). */
  const [conversationSaveStatus, setConversationSaveStatus] = useState('idle');
  /** Batch delete select mode */
  const [selectMode, setSelectMode] = useState(false);

  const [selectedConversationIds, setSelectedConversationIds] = useState(new Set());

  const [searchHighlightId, setSearchHighlightId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [outreachNotifications, setOutreachNotifications] = useState([]);
  const [outreachScrollToMessageId, setOutreachScrollToMessageId] = useState(null);
  const [pendingDMThreadId, setPendingDMThreadId] = useState(null);
  const outreachSyncTimerRef = useRef(null);
  /** Batches streaming token UI updates to ~1/frame; avoids ReactMarkdown thrash on fast APIs. */
  const streamMessageRafRef = useRef({ rafId: null, pending: null });
  // Debug: track what happens during streaming until the tab crashes/hangs.
  const streamDebugRef = useRef(null);
  const [taskProgress, setTaskProgress] = useState({ progress: 0, status: '', active: false });
  /** Latest chat messages for async book automation (avoids stale closures). */
  const messagesRef = useRef([]);
  /** When set, `prepareApiHistoryWithRollingMemory` uses these packing overrides (book automation). */
  const bookModePackingOverridesRef = useRef(null);
  /** Debounced mirror of conversations → IndexedDB (cleared on delete to avoid races). */
  const conversationPersistTimerRef = useRef(null);
  const conversationCatalogPersistTimerRef = useRef(null);
  const conversationCatalogSigRef = useRef('');
  /** Bumped on delete so in-flight IndexedDB writes cannot resurrect removed tabs. */
  const conversationStorageEpochRef = useRef(0);
  /** In-memory tombstones (sync guard for auto-save during/after delete). */
  const tombstonedConversationIdsRef = useRef(new Set());
  /** Suppress message auto-save while switching tabs (avoids wrong-id / empty shard writes). */
  const conversationSwitchInProgressRef = useRef(false);
  /** Serialises long-running media completions per owning conversation. */
  const conversationMessageMutationChainsRef = useRef(new Map());
  /** Lets a tab load detect a shard mutation that completed while it was reading. */
  const conversationMessageVersionRef = useRef(new Map());
  /** Prevents a slower, older tab read from replacing a newer selection. */
  const conversationSelectionRequestRef = useRef(0);
  /**
   * Startup restore gate:
   * - default: landing-first (do not auto-open a past chat on refresh)
   * - opt-in restore: `?restoreLastConversation=1` or `?restore=last`
   * This ref is one-shot so late async hydration cannot overwrite a deliberate landing/new-chat selection.
   */
  const startupConversationRestoreRef = useRef({
    allowAutoRestore: (() => {
      if (typeof window === 'undefined') return false;
      try {
        const params = new URLSearchParams(window.location.search);
        return params.get('restoreLastConversation') === '1' || params.get('restore') === 'last';
      } catch (_) {
        return false;
      }
    })(),
    attempted: false,
  });
  /** New-chat finalizes intro on the active tab before starting another. */
  const completeCharacterIntroRef = useRef(null);

  // Refs updated every render so async sendMessage reads latest conversation/flag (no effect timing issues)
  conversationsRef.current = conversations;
  activeConversationRef.current = activeConversation;
  messagesRef.current = messages;
  activeTabRef.current = activeTab;

  // Clear backend run status when switching chats so "ran" / "error" isn't from a different chat
  useEffect(() => {
    setLastAgenticRunStatus(null);
  }, [activeConversation]);

  /** Same id Settings → Agentic tab uses for GET /memory/agentic/list (must match write path). */
  const resolveAgenticUserId = useCallback(
    () => memoryContext?.activeProfileId || userProfile?.id || null,
    [memoryContext?.activeProfileId, userProfile?.id],
  );

  // Global error capture so we get something even if the page “just crashes”.
  useEffect(() => {
    const writeDebugLog = (prefix, errOrEvent) => {
      try {
        const payload = {
          prefix,
          ts: new Date().toISOString(),
          message: errOrEvent?.message,
          stack: errOrEvent?.stack,
          // Browser events have different shapes; keep it lightweight.
          errorName: errOrEvent?.name,
        };
        console.error('🧨 [GlobalError]', payload);
        // Best-effort persistence in case console is cleared by crash.
        localStorage.setItem('LiangLocal-last-global-error', JSON.stringify(payload));
      } catch (_) {}
    };

    const onError = (event) => writeDebugLog('window.onerror', event?.error || event);
    const onUnhandledRejection = (event) => writeDebugLog('unhandledrejection', event?.reason || event);

    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);
    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
    };
  }, []);

  // IndexedDB hydration: lightweight first (unblock UI), migration/recover opt-in only
  useEffect(() => {
    installChatStorageDebugHelpers();
    let cancelled = false;
    const isMigrationDone = () => {
      try {
        return localStorage.getItem('LiangLocal-idb-migrated') === 'v1';
      } catch (_) {
        return false;
      }
    };

    const hydrateFromDisk = async () => {
      const settingsOnly = isSettingsStandaloneWindow();
      console.time('[Eloquent] hydrate:total');

      if (settingsOnly) {
        console.info('[Eloquent] skipping emergency recover (settings window)');
        console.time('[Eloquent] hydrate:settings-light');
        const idbOpts = { preferLocalStorage: true, skipMigration: true };
        const [charactersStr, avatarSizesStr, hydratedSettings] = await Promise.all([
          indexedDbStorage.getItem('llm-characters', idbOpts),
          indexedDbStorage.getItem('LiangLocal-avatar-sizes', idbOpts),
          readSettingsFromStorage(idbOpts),
        ]);
        if (cancelled) return;
        if (charactersStr) {
          try {
            const parsed = JSON.parse(charactersStr).map(normalizeCharacter);
            setCharacters(parsed);
            try {
              localStorage.setItem('llm-characters', JSON.stringify(parsed));
            } catch (_) { /* mirror */ }
          } catch (e) { console.warn('Hydration (settings window): parse characters', e); }
        }
        if (hydratedSettings || avatarSizesStr) {
          try {
            const parsed = hydratedSettings ? { ...hydratedSettings } : {};
            applyAvatarSizesToSettings(parsed, avatarSizesStr);
            setSettings((s) => {
              if (!shouldApplyHydratedSettings(parsed, s)) return s;
              return mergeSettingsObjects(s, parsed);
            });
          } catch (e) { console.warn('Hydration (settings window): parse settings', e); }
        }
        console.timeEnd('[Eloquent] hydrate:settings-light');
        console.info('[Eloquent] Settings window: lightweight storage hydrate (chat tabs skipped)');
        console.timeEnd('[Eloquent] hydrate:total');
        return;
      }

      const idbBootOpts = { preferLocalStorage: true, skipMigration: true };

      if (!isMigrationDone()) {
        console.info('[Eloquent] IDB migration running in background (boot uses localStorage mirror)');
        void indexedDbStorage.migrateFromLocalStorage().catch((e) => {
          console.warn('[Eloquent] background IDB migration failed:', e);
        });
      } else {
        console.info('[Eloquent] skipping localStorage→IDB migration (already done)');
      }

      console.time('[Eloquent] hydrate:main-light');
      scrubConversationLocalStorageGhosts();
      if (cancelled) return;

      const [charactersStr, activeIdStr, avatarSizesStr, parsedConversations, hydratedSettings, installerAudioSettings] = await Promise.all([
        indexedDbStorage.getItem('llm-characters', idbBootOpts),
        indexedDbStorage.getItem('Eloquent-active-conversation', idbBootOpts),
        indexedDbStorage.getItem('LiangLocal-avatar-sizes', idbBootOpts),
        loadConversationsFromStorage({ skipShardScan: true, idbOpts: idbBootOpts }),
        readSettingsFromStorage(idbBootOpts),
        readInstallerAudioProfile(),
      ]);
      if (cancelled) return;

      if (charactersStr) {
        try {
          const parsed = JSON.parse(charactersStr).map(normalizeCharacter);
          setCharacters(parsed);
          try {
            localStorage.setItem('llm-characters', JSON.stringify(parsed));
          } catch (_) { /* mirror for legacy readers */ }
        } catch (e) { console.warn('Hydration: parse characters', e); }
      }

      const tombstoned = await loadTombstonedConversationIds();
      getBannedConversationIdsSync().forEach((bid) => tombstoned.push(bid));
      tombstonedConversationIdsRef.current = new Set(tombstoned);

      const conversationsToShow = (Array.isArray(parsedConversations)
        ? parsedConversations.filter(
          (c) => c?.id
            && !tombstonedConversationIdsRef.current.has(c.id)
            && !isOutreachConversationId(c.id)
        )
        : []);

      if (conversationsToShow.length === 0) {
        console.info('[Eloquent] skipping emergency recover (deferred) — use window.eloquentChatStorage.emergencyRecover() if tabs are missing');
      }

      conversationCatalogSigRef.current = conversationsToShow
        .map((c) => `${c.id}\t${c.name || ''}\t${Number(c.messageCount) || 0}`)
        .join('\n');

      const banCount = tombstonedConversationIdsRef.current.size;
      if (banCount > 0) {
        console.info(
          `[Eloquent] Chat storage v9: ${conversationsToShow.length} tab(s) loaded, ${banCount} banned id(s) hidden`
        );
      } else if (conversationsToShow.length > 0) {
        console.info(`[Eloquent] Chat storage v9: ${conversationsToShow.length} tab(s) loaded`);
      }

      if (conversationsToShow.length > 0) {
        setConversations(conversationsToShow);

        const userAlreadySelectedConversation =
          !!activeConversationRef.current
          || (messagesRef.current?.length > 0);
        const restoreGate = startupConversationRestoreRef.current;
        const mayAttemptStartupRestore = !restoreGate.attempted && restoreGate.allowAutoRestore;
        restoreGate.attempted = true;

        let lastActiveId = activeIdStr || null;
        if (lastActiveId && tombstonedConversationIdsRef.current.has(lastActiveId)) {
          lastActiveId = conversationsToShow[0]?.id ?? null;
        }
        if (
          !userAlreadySelectedConversation
          && mayAttemptStartupRestore
          && lastActiveId
          && conversationsToShow.some(c => c.id === lastActiveId)
        ) {
          console.time('[Eloquent] hydrate:restore-last-chat');
          setActiveConversation(lastActiveId);
          const activeMsgs = await loadConversationMessages(lastActiveId);
          setMessages(activeMsgs);
          const activeConv = conversationsToShow.find(c => c.id === lastActiveId);
          if (Array.isArray(activeConv?.activeCharacterIds)) setActiveCharacterIds(activeConv.activeCharacterIds);
          if (activeConv?.activeCharacterWeights && typeof activeConv.activeCharacterWeights === 'object') {
            setActiveCharacterWeights(activeConv.activeCharacterWeights);
          }
          if (typeof activeConv?.multiRoleContext === 'string') setMultiRoleContext(activeConv.multiRoleContext);
          console.timeEnd('[Eloquent] hydrate:restore-last-chat');
        } else if (!userAlreadySelectedConversation && lastActiveId && !mayAttemptStartupRestore) {
          console.info(
            '[Eloquent] Startup restore skipped (landing-first mode).',
            'Use ?restoreLastConversation=1 to re-enable restore for this load.'
          );
        }
      }

      if (hydratedSettings || avatarSizesStr || installerAudioSettings) {
        try {
          const audioDefaultsMigration = hydratedSettings
            ? getAudioStartupDefaultsMigration(hydratedSettings)
            : null;
          const parsed = hydratedSettings
            ? mergeSettingsObjects(hydratedSettings, audioDefaultsMigration)
            : {};
          applyAvatarSizesToSettings(parsed, avatarSizesStr);
          const settingsToApply = installerAudioSettings
            ? mergeSettingsObjects(parsed, installerAudioSettings)
            : parsed;
          const currentSettings = settingsRef.current;
          if (shouldApplyHydratedSettings(settingsToApply, currentSettings)) {
            const nextSettings = mergeSettingsObjects(currentSettings, settingsToApply);
            settingsRef.current = nextSettings;
            setSettings(nextSettings);
            if (installerAudioSettings || audioDefaultsMigration) {
              const persisted = await persistSettingsBlob(nextSettings);
              if (persisted && installerAudioSettings) await clearInstallerAudioProfile();
            }
          }
          if (!avatarSizesStr && (typeof parsed.userAvatarSize === 'number' || typeof parsed.characterAvatarSize === 'number')) {
            indexedDbStorage.setItem('LiangLocal-avatar-sizes', JSON.stringify({
              userAvatarSize: parsed.userAvatarSize ?? 64,
              characterAvatarSize: parsed.characterAvatarSize ?? 64,
            }));
          }
        } catch (e) { console.warn('Hydration: parse settings', e); }
      }
      console.timeEnd('[Eloquent] hydrate:main-light');
      console.timeEnd('[Eloquent] hydrate:total');
    };

    (async () => {
      const settingsOnly = isSettingsStandaloneWindow();
      const hydrationTimeoutMs = settingsOnly
        ? SETTINGS_STANDALONE_STORAGE_TIMEOUT_MS
        : STORAGE_HYDRATION_TIMEOUT_MS;
      console.time('[Eloquent] boot:storageHydrated');
      if (!settingsOnly) {
        setStorageHydrated(true);
      }
      try {
        await withTimeout(hydrateFromDisk(), hydrationTimeoutMs, 'storageHydration');
      } catch (e) {
        console.warn('Storage hydration failed or timed out:', e);
        if (!cancelled) setStorageHydrationDegraded(true);
      } finally {
        if (!cancelled) {
          setStorageHydrated(true);
          console.timeEnd('[Eloquent] boot:storageHydrated');
        }
      }
    })();

    return () => { cancelled = true; };
  }, [bootGeneration]);

  const persistConversationsToStorage = useCallback(async (
    list,
    activeId,
    epochAtStart,
    activeMessages = null,
    persistOpts = {}
  ) => {
    const epoch = epochAtStart ?? conversationStorageEpochRef.current;
    try {
      await persistChatState(list, activeId, activeMessages, persistOpts);
    } catch (e) {
      console.warn('[conversations] persist failed:', e);
      return false;
    }
    if (epoch !== conversationStorageEpochRef.current) {
      console.warn('[conversations] Discarded stale IndexedDB write (tab was deleted)');
      return false;
    }
    return true;
  }, []);

  const deleteConversation = useCallback(async (id) => {
    try {
      // Sync ban FIRST — survives crash / full IndexedDB (this was the missing piece).
      banConversationIdSync(id);
      tombstonedConversationIdsRef.current.add(id);

      if (conversationPersistTimerRef.current) {
        clearTimeout(conversationPersistTimerRef.current);
        conversationPersistTimerRef.current = null;
      }
      conversationStorageEpochRef.current += 1;
      const deleteEpoch = conversationStorageEpochRef.current;

      const currentMessages = messagesRef.current || messages;
      const currentActive = activeConversationRef.current ?? activeConversation;

      const merged = (conversationsRef.current || conversations).map((conv) =>
        conv.id === currentActive ? { ...conv, messages: currentMessages } : conv
      );
      const updatedConversations = merged.filter((conv) => conv.id !== id);
      const wasActive = currentActive === id;
      const newActiveId = wasActive
        ? (updatedConversations[0]?.id ?? null)
        : currentActive;

      setConversations(updatedConversations);

      if (wasActive) {
        setActiveConversation(newActiveId);
        if (newActiveId) {
          const sel = updatedConversations.find((c) => c.id === newActiveId);
          setMessages(Array.isArray(sel?.messages) ? sel.messages : []);
          const { primary, secondary, user } = sel?.characterIds || {};
          let primChar = characters.find((c) => c.id === primary) || null;
          if (!primChar && sel?.characterSnapshot) {
            const snap = sel.characterSnapshot;
            if (!primary || snap.id === primary) primChar = normalizeCharacter(snap);
          }
          setPrimaryCharacter(primChar);
          setSecondaryCharacter(secondary ? characters.find((c) => c.id === secondary) || null : null);
          setActiveCharacter(primChar);
          setUserCharacterId(user || null);
        } else {
          setMessages([]);
        }
      }

      await deleteConversationFromStorage(id);

      const activeToSave = wasActive ? newActiveId : currentActive;
      if (deleteEpoch !== conversationStorageEpochRef.current) {
        return true;
      }

      const survivors = updatedConversations.filter(
        (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
      );
      await saveConversationCatalog(survivors, activeToSave || null, {
        allowEmpty: survivors.length === 0,
      });

      if (activeToSave && !tombstonedConversationIdsRef.current.has(activeToSave)) {
        let activeMsgs = [];
        if (activeToSave === currentActive) {
          activeMsgs = currentMessages || [];
        } else {
          activeMsgs = await loadConversationMessages(activeToSave);
        }
        await persistConversationsToStorage(
          survivors,
          activeToSave,
          deleteEpoch,
          activeMsgs.length > 0 ? activeMsgs : null
        );
      } else {
        try {
          await indexedDbStorage.removeItem('Eloquent-active-conversation');
        } catch (_) { /* noop */ }
      }

      return true;
    } catch (e) {
      console.error('Error in deleteConversation:', e);
      if (typeof window !== 'undefined') {
        window.alert('Could not delete this chat from browser storage. Open Settings → Storage → Erase all chats, or try again after a refresh.');
      }
      return false;
    }
  }, [
    conversations,
    messages,
    activeConversation,
    characters,
    setConversations,
    setActiveConversation,
    setMessages,
    setPrimaryCharacter,
    setSecondaryCharacter,
    setActiveCharacter,
    setUserCharacterId,
    persistConversationsToStorage,
  ]);

  const deleteAllConversations = useCallback(async () => {
    try {
      for (const conversation of conversationsRef.current || []) {
        if (conversation?.id) tombstonedConversationIdsRef.current.add(conversation.id);
      }
      if (conversationPersistTimerRef.current) {
        clearTimeout(conversationPersistTimerRef.current);
        conversationPersistTimerRef.current = null;
      }
      conversationStorageEpochRef.current += 1;
      conversationSelectionRequestRef.current += 1;
      conversationSwitchInProgressRef.current = false;
      conversationsRef.current = [];
      activeConversationRef.current = null;
      messagesRef.current = [];

      setConversations([]);
      setActiveConversation(null);
      setMessages([]);

      try {
        await deleteAllConversationsFromStorage();
      } catch (storageError) {
        console.error('Storage error during delete all:', storageError);
      }

      return true;
    } catch (e) {
      console.error('Error in deleteAllConversations:', e);
      return false;
    }
  }, [setConversations, setActiveConversation, setMessages]);

  const toggleSelectMode = useCallback(() => {
    setSelectMode((prev) => {
      if (prev) setSelectedConversationIds(new Set());
      return !prev;
    });
  }, []);

  const toggleConversationSelection = useCallback((id) => {
    setSelectedConversationIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const selectAllConversations = useCallback(() => {
    const allIds = (conversationsRef.current || conversations)
      .filter((c) => !isOutreachConversationId(c?.id))
      .map((c) => c.id);
    setSelectedConversationIds(new Set(allIds));
  }, [conversations]);

  const clearSelection = useCallback(() => {
    setSelectedConversationIds(new Set());
  }, []);

  const deleteSelectedConversations = useCallback(async () => {
    const ids = [...selectedConversationIds];
    if (ids.length === 0) return false;

    try {
      for (const id of ids) {
        banConversationIdSync(id);
        tombstonedConversationIdsRef.current.add(id);
      }

      if (conversationPersistTimerRef.current) {
        clearTimeout(conversationPersistTimerRef.current);
        conversationPersistTimerRef.current = null;
      }
      conversationStorageEpochRef.current += 1;

      const currentActive = activeConversationRef.current ?? activeConversation;
      const wasActiveDeleted = ids.includes(currentActive);
      const survivors = (conversationsRef.current || conversations).filter(
        (c) => !ids.includes(c.id)
      );

      setConversations(survivors);

      if (wasActiveDeleted) {
        const newActiveId = survivors[0]?.id ?? null;
        setActiveConversation(newActiveId);
        if (newActiveId) {
          const sel = survivors.find((c) => c.id === newActiveId);
          setMessages(Array.isArray(sel?.messages) ? sel.messages : []);
        } else {
          setMessages([]);
        }
      }

      await deleteConversationsFromStorage(ids);

      const activeToSave = wasActiveDeleted
        ? (survivors[0]?.id ?? null)
        : currentActive;
      if (activeToSave && !tombstonedConversationIdsRef.current.has(activeToSave)) {
        let activeMsgs = [];
        if (activeToSave === currentActive) {
          activeMsgs = messagesRef.current || [];
        } else {
          activeMsgs = await loadConversationMessages(activeToSave);
        }
        await persistConversationsToStorage(
          survivors,
          activeToSave,
          conversationStorageEpochRef.current,
          activeMsgs.length > 0 ? activeMsgs : null
        );
      } else {
        try {
          await indexedDbStorage.removeItem('Eloquent-active-conversation');
        } catch (_) { /* noop */ }
      }

      setSelectedConversationIds(new Set());
      setSelectMode(false);
      return true;
    } catch (e) {
      console.error('Error in deleteSelectedConversations:', e);
      return false;
    }
  }, [
    selectedConversationIds,
    conversations,
    activeConversation,
    setConversations,
    setActiveConversation,
    setMessages,
    persistConversationsToStorage,
  ]);

  const renameConversation = useCallback((id, newName) => {
    setConversations((prevConversations) => {
      const next = prevConversations.map((conv) =>
        conv.id === id
          ? { ...conv, name: newName, requiresTitle: false, titleSource: 'manual' }
          : conv
      );
      void saveConversationCatalog(next, activeConversationRef.current);
      return next;
    });
  }, []);

  const createDefaultSettings = () => ({
    directProfileInjection: false, // <-- ADD THIS
    temperature: 0.7,
    max_tokens: -1,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.0,
    use_rag: false,
    selectedDocuments: [],
    ragAgentTools: false,
    contextLength: 16000, // Default value
    useMemory: false,
    useMemoryAgent: false,
    useLore: false,
    useLoreAgent: false,
    useLoreAgentForMemory: false,
    useLoreAgentForMemoryRetrieval: false,
    useLoreAgentForMemoryObservation: false,
    multiRoleMode: false,
    showUserProfiles: false,
    autoSelectSpeaker: false,
    narratorEnabled: false,
    narratorInterval: 6,
    narratorName: 'Narrator',
    narratorInstructions: 'Describe scene transitions and world details briefly. Keep narration concise and avoid speaking for the user.',
    narratorAvatar: null,
    chessHistorianAvatar: null,
    chessHistorianPersonaPrompt: null,
    sttEnabled: true,
    audioStartupDefaultsVersion: 1,
    sttAutoSendOnStop: false,
    streamResponses: true,
    sttEngine: "whisper",
    nanogptSttModel: "fun-asr-flash-2026-06-15",
    parakeetCppModel: "tdt_ctc-110m",
    parakeetCppQuant: "f16",
    /** Full path to ffmpeg.exe when not on system PATH (Voice Merge / STT / D-ID). */
    ffmpegPath: '',
    ttsEngine: 'kokoro',
    ttsVoice: 'af_heart',
    ttsEnabled: true,
    ttsSpeed: 1.0,
    ttsPitch: 0,
    ttsStreamChunkSentences: 3,
    /** Seconds of audio to buffer before starting autoplay playback. 0 = start
     *  immediately (current behavior). Set to ~45 for engines with RTF > 1
     *  (e.g. VoxCPM) to prevent mid-stream stalls. */
    ttsPrebufferSeconds: 0,
    ttsAutoPlay: false,  // Simply set a default value
    ttsWaitForFullResponse: false,
    ttsSaveFullResponseAudio: false,
    /** When > 0 and save is on, backend splits exported WAV into segments of at most this many seconds (285 ≈ 4m45, under 5 min). */
    ttsSaveFullResponseChunkSeconds: 0,
    userAvatarSize: 64,
    characterAvatarSize: 64,
    mdBodyColor: '',
    mdBoldColor: '',
    mdItalicColor: '',
    mdQuoteColor: '',
    mdQuoteBorder: '',
    mdH1Color: '',
    mdH2Color: '',
    mdH3Color: '',
    mdH1Font: '',
    mdH2Font: '',
    mdH3Font: '',
    performanceMode: false,
    outreachRules: [],
    outreachBrowserNotifications: false,

    /** When using a subscription/API primary model, cap verbatim turns and fold older dialogue into per-chat rolling memory. */
    apiRollingMemoryEnabled: true,
    /** Approximate total request context window for API history packing (user turn remains pinned). */
    apiContextWindowTokens: API_CONTEXT_WINDOW_TOKENS,
    /** Approximate max tokens for verbatim user/bot history per request (rolling pack carries the rest). */
    apiRecentVerbatimTokenBudget: 32000,

    /** Book automation overlay: wider API packing while a queued chapter run is active. */
    bookRunExperimentalEnabled: false,
    bookWritingApiContextTokens: 262144,
    bookWritingVerbatimTokenBudget: 98304,
    /** Responses shorter than this (characters) count as refusal → auto-retry. */
    bookRefusalMaxChars: 2200,
    /** { id, label, text }[] — one-tap prompts during a book run (Settings). */
    bookQuickPromptButtons: [],
    didQuickPromptButtons: [],
    bookWordFloorPreamble:
      '[BOOK RUN]\nEach chapter in this run should run to at least roughly 3000 words unless the material truly cannot sustain that length. Follow the chapter heading and intent.',

    /** New chat: AI-generated character introduction instead of static first_message greeting */
    characterIntroEnabled: false,
    characterIntroPrompt: '',
    characterIntroMaxTokens: 900,
    characterIntroTemperature: 0.55,
    characterIntroHistoryLimit: 8,
    characterIntroRequestPurpose: 'character_intro',
    characterIntroEndpoint: '',
    characterIntroApiOverrideEnabled: false,
    characterIntroApiEndpointId: '',
    characterIntroApiModel: '',
    characterIntroApiKey: '',
    /** flat | character_card | full_generation — same modes as call-mode about */
    characterIntroSystemPromptMode: 'full_generation',

    /** Use selected character card as LLM system persona (not roleplay wrapper) */
    useCharacterAsSystemPrompt: false,
    systemPersonaCharacterId: null,
    systemIntroRequestPurpose: 'system_intro',
    systemIntroPrompt: '',
    systemIntroSystemPromptMode: 'full_generation',

    /** Experimental: call-mode "about this character" hover + API insight panel */
    callModeAboutCharacterEnabled: true,
    callModeAboutCharacterPrompt: '',
    callModeAboutCharacterMaxTokens: 1200,
    callModeAboutCharacterTemperature: 0.45,
    callModeAboutCharacterHistoryLimit: 40,
    callModeAboutCharacterRequestPurpose: 'call_mode_character_about',
    callModeAboutCharacterEndpoint: '',
    callModeAboutCharacterApiOverrideEnabled: false,
    callModeAboutCharacterApiEndpointId: '',
    callModeAboutCharacterApiModel: '',
    callModeAboutCharacterApiKey: '',
    /** flat | character_card | full_generation — see callModeCharacterAbout.js */
    callModeAboutCharacterSystemPromptMode: 'flat',

    /** Call mode: optional full-screen portrait avatar + framing zoom/pan */
    callModeFullscreenAvatar: false,
    callModeFullscreenZoom: 1,
    callModeFullscreenPanX: 0,
    callModeFullscreenPanY: 0,

    /** When true, clicking Settings opens a separate window instead of in-app tab. */
    openSettingsInSecondWindow: false,

    /** Startup splash minimum display: off | fast | normal | long (see eloquentSplash.js). */
    splashScreenDuration: 'normal',

    admin_password: "", // <-- Password for remote access
    openaiServerLanEnabled: false,
    huggingFaceToken: '',
    nanoGptApiKey: '',
    nanoGptBillingMode: 'payg',
    openRouterApiKey: '',
    openAiApiKey: '',
    anthropicApiKey: '',
    geminiApiKey: '',
    mistralApiKey: '',
    xAiApiKey: '',
    metaApiKey: '',
    providerSetupCompleted: false,
    modelSetupRequired: false,
    modelSetupSource: 'huggingface',
    primaryUse: null,
    roleplayIntroCompleted: false,
    sillyTavernSetupCompleted: false,
    customApiEndpoints: [],
    speechModelDirectory: '',

    /** NanoGPT Context Memory — forwarded on /generate when endpoint URL is nano-gpt.com */
    nanoGptContextMemoryEnabled: false,
    nanoGptContextMemoryMode: 'header',
    nanoGptContextMemoryExpirationDays: 30,
    /** Debug-only: show layered reasoning diagnostics in chat UI. */
    showReasoningDiagnostics: false,

    /** Chat Typography — text pop, fonts, streaming effects */
    chatFontFamily: '',
    chatFontSize: '',
    chatFontWeight: '',
    chatLineHeight: '',
    chatLetterSpacing: '',
    chatTextShadow: true,
    chatTextGlow: false,
    chatTypewriterEnabled: true,
    chatStableScroll: true,
    chatReasoningStyle: 'dimmed',
    chatPreset: 'theme',

    /** Vision Model (Two-Stage Pipeline) — runs before text model to extract structured info from images */
    visionModel: null, // e.g., 'LFM2.5-VL-450M-Extract', 'gemma-3-4b-it', 'llava-v1.6-mistral-7b'
    visionSchema: '', // YAML schema for structured extraction (optional)
  });

  const [settings, setSettings] = useState(() => {
    const defaults = createDefaultSettings();
    const fromLs = readSettingsFromLocalStorageSync();
    if (fromLs && shouldApplyHydratedSettings(fromLs, defaults)) {
      return mergeSettingsObjects(defaults, fromLs);
    }
    return defaults;
  });

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const [lastRequestRouteMeta, setLastRequestRouteMeta] = useState(null);

  const resolveActionRouterContract = useCallback(
    (
      actionName,
      {
        requestPurpose = null,
        modelName = primaryModel,
        settingsSnapshot = settingsRef.current,
      } = {},
    ) => {
      const route = resolveUnifiedRequestRoute({
        primaryModel: modelName,
        primaryIsAPI,
        settings: settingsSnapshot,
        requestPurpose,
      });
      logRouteTrace({
        action: actionName,
        route,
        requestPurpose,
      });
      return route;
    },
    [primaryIsAPI, primaryModel],
  );

  const getActiveSystemPersonaContext = useCallback((conversationId = null) => {
    const convId = conversationId || activeConversationRef.current;
    const conv = (conversationsRef.current || []).find((c) => c.id === convId);
    if (!isSystemPersonaModeActive(settingsRef.current, conv)) {
      return { active: false, character: null, conversation: conv };
    }
    const character = resolveSystemPersonaCharacter(characters, settingsRef.current, conv);
    return { active: true, character, conversation: conv };
  }, [characters]);

  const getSystemPersonaGenerateExtras = useCallback((conversationId = null) => {
    const { active } = getActiveSystemPersonaContext(conversationId);
    return active ? { system_persona_mode: true } : {};
  }, [getActiveSystemPersonaContext]);

  useEffect(() => {
    roleplayEnabledRef.current = settings.multiRoleMode === true;
  }, [settings.multiRoleMode]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;

    const applyVar = (key, cssVar) => {
      const value = settings?.[key];
      if (typeof value === 'string' && value.trim()) {
        root.style.setProperty(cssVar, value.trim());
      } else {
        root.style.removeProperty(cssVar);
      }
    };

    applyVar('mdBodyColor', '--md-body-color');
    applyVar('mdBoldColor', '--md-bold-color');
    applyVar('mdItalicColor', '--md-italic-color');
    applyVar('mdQuoteColor', '--md-quote-color');
    applyVar('mdQuoteBorder', '--md-quote-border');
    applyVar('mdH1Color', '--md-h1-color');
    applyVar('mdH2Color', '--md-h2-color');
    applyVar('mdH3Color', '--md-h3-color');
    applyVar('mdH1Font', '--md-h1-font');
    applyVar('mdH2Font', '--md-h2-font');
    applyVar('mdH3Font', '--md-h3-font');

    /** Chat Typography — CSS variables for font/size/weight/line-height */
    applyVar('chatFontFamily', '--chat-font-family');
    applyVar('chatFontSize', '--chat-font-size');
    applyVar('chatFontWeight', '--chat-font-weight');
    applyVar('chatLineHeight', '--chat-line-height');
    applyVar('chatLetterSpacing', '--chat-letter-spacing');

    /** Boolean flags as data-attributes for CSS selectors */
    root.dataset.chatTextShadow = settings?.chatTextShadow ? 'true' : 'false';
    root.dataset.chatTextGlow = settings?.chatTextGlow ? 'true' : 'false';
    root.dataset.chatTypewriter = settings?.chatTypewriterEnabled ? 'true' : 'false';
    root.dataset.chatStableScroll = settings?.chatStableScroll ? 'true' : 'false';
    root.dataset.chatReasoningStyle = settings?.chatReasoningStyle || 'dimmed';
    root.dataset.chatPreset = settings?.chatPreset || 'theme';
  }, [
    settings?.mdBodyColor,
    settings?.mdBoldColor,
    settings?.mdItalicColor,
    settings?.mdQuoteColor,
    settings?.mdQuoteBorder,
    settings?.mdH1Color,
    settings?.mdH2Color,
    settings?.mdH3Color,
    settings?.mdH1Font,
    settings?.mdH2Font,
    settings?.mdH3Font,
    settings?.chatFontFamily,
    settings?.chatFontSize,
    settings?.chatFontWeight,
    settings?.chatLineHeight,
    settings?.chatLetterSpacing,
    settings?.chatTextShadow,
    settings?.chatTextGlow,
    settings?.chatTypewriterEnabled,
    settings?.chatStableScroll,
    settings?.chatReasoningStyle,
    settings?.chatPreset,
  ]);

  // Backend detects single_gpu_mode, frontend stores as singleGpuMode
  const isSingleGpuMode = settings?.singleGpuMode === true;

  // Port configuration - loaded from /ports.json when a host provides overrides.
  const [portConfig, setPortConfig] = useState({
    backend: "http://localhost:8000",
    secondary: "http://localhost:8001",
    tts: "http://localhost:8002"
  });
  /** True after loadPortConfig finishes (or times out) — URLs are safe to use. */
  const [portsReady, setPortsReady] = useState(false);

  const retryBoot = useCallback(() => {
    setPortsReady(false);
    postStorageHydrateFetchRef.current = false;
    if (!isSettingsStandaloneWindow()) {
      setStorageHydrated(false);
    }
    setStorageHydrationDegraded(false);
    setPortsLoadDegraded(false);
    setBootGeneration((g) => g + 1);
  }, []);

  // Load port configuration on startup (also updates the api.js module cache)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const config = await withTimeout(
          loadPortConfig(),
          PORT_CONFIG_TIMEOUT_MS,
          'loadPortConfig'
        );
        if (!cancelled) {
          console.log('📌 Loaded port config:', config);
          setPortConfig(config);
        }
      } catch (e) {
        console.warn('Port config load failed or timed out; using defaults', e);
        if (!cancelled) {
          setPortsLoadDegraded(true);
          setPortConfig(getConfig());
        }
      } finally {
        // Always unblock settings/UI even if this effect was superseded by retryBoot cleanup.
        setPortsReady(true);
      }
    })();
    return () => { cancelled = true; };
  }, [bootGeneration]);

  // API URLs - use port config
  const PRIMARY_API_URL = portConfig.backend;
  const settingsStandaloneForMemory = isSettingsStandaloneWindow();
  const SECONDARY_API_URL = isSingleGpuMode ? portConfig.backend : portConfig.secondary;
  /** Settings window + single-GPU: memory API is always primary (see InfrastructureBanner). */
  const MEMORY_API_URL = (isSingleGpuMode || settingsStandaloneForMemory)
    ? portConfig.backend
    : portConfig.secondary;
  const TTS_API_URL = portConfig.tts;
  const BACKEND = import.meta.env.VITE_API_URL || (isSingleGpuMode ? portConfig.backend : portConfig.secondary);
  const VITE_API_URL = isSingleGpuMode ? portConfig.backend : portConfig.secondary;




  // Keep TTS WebSocket always connected when TTS autoplay is enabled
  useEffect(() => {
    if (settings.ttsAutoPlay) {
      console.log("🔌 [TTS] Keeping WebSocket connected for instant TTS (Internal Auto-Reconnect enabled)");
      ttsClient.connect(
        () => console.log("✅ [TTS] WebSocket connected and ready"),
        () => console.log("🔌 [TTS] WebSocket disconnected, client will auto-reconnect..."),
        (err) => console.error("❌ [TTS] WebSocket error:", err)
      );
    } else {
      // Disconnect if TTS autoplay is off
      ttsClient.disconnect();
    }
  }, [settings.ttsAutoPlay]);

  useEffect(() => {
    ttsWaitForFullResponseRef.current = settings.ttsWaitForFullResponse === true;
  }, [settings.ttsWaitForFullResponse]);

  useEffect(() => {
    ttsPrebufferSecondsRef.current = Math.max(0, Number(settings.ttsPrebufferSeconds) || 0);
  }, [settings.ttsPrebufferSeconds]);

  useEffect(() => {
    if (!ttsClient) return;
    const handleSubtitleCue = (cue) => {
      const ts = Date.now();
      setTtsSubtitleCue({
        text: cue?.text || '',
        durationMs: cue?.durationMs ?? null,
        timestamp: ts,
      });
      window.__ttsSubtitleCue = {
        text: cue?.text || '',
        durationMs: cue?.durationMs ?? null,
        timestamp: ts,
      };
    };
    ttsClient.onSubtitleCue = handleSubtitleCue;
    return () => {
      if (ttsClient.onSubtitleCue === handleSubtitleCue) {
        ttsClient.onSubtitleCue = null;
      }
    };
  }, [ttsClient, setTtsSubtitleCue]);

  // Initial conversation load is done by IndexedDB hydration effect above
  // ===== Debugging and Testing the Lore Functionality =====
  // --- TTS Implementation (REORDERED) ---
  const stopTTS = useCallback((reason = 'unspecified') => {
    ttsPlaybackRequestRef.current += 1;
    window.__ttsSubtitleCue = null;
    setTtsSubtitleCue(null);
    console.warn(`🛑 [stopTTS] Initiating stop sequence — reason: ${reason}`);
    if (typeof console.trace === 'function') {
      console.trace('🛑 [stopTTS] caller stack');
    }
    // 1. IMMEDIATE: Set interrupt flag to prevent any new chunks from processing
    isTtsInterruptedRef.current = true;
    isFirstTextChunk.current = true; // Reset text chunk flag
    streamingTtsMessageIdRef.current = null; // Clear active message ID
    streamingTtsWsEndSentRef.current = false;
    const coalesce = ttsStreamSendCoalesceRef.current;
    if (coalesce.rafId != null) {
      cancelAnimationFrame(coalesce.rafId);
      coalesce.rafId = null;
    }
    coalesce.pending = '';
    ttsFullResponseBufferRef.current = ''; // Clear any buffered full-response text

    // 2. CRITICAL: Send interrupt signal to backend FIRST (before any cleanup)
    if (ttsClient && ttsClient.socket && ttsClient.socket.readyState === 1) {
      console.log('🛑 [stopTTS] Sending interrupt signal to backend');
      ttsClient.interrupt();
      ttsClient.clearPending();
    }

    // 3. Always cancel native browser TTS
    if ('speechSynthesis' in window) {
      console.log('🛑 [stopTTS] Cancelling native speech synthesis');
      window.speechSynthesis.cancel();
    }

    // 4. STOP ALL REGISTERED AUDIO PLAYERS
    const playersToStop = Array.from(activeAudioPlayersRef.current);
    if (playersToStop.length > 0) {
      console.log(`🛑 [stopTTS] Stopping ${playersToStop.length} active audio sources.`);
      playersToStop.forEach(player => {
        try {
          if (player instanceof Audio) {
            console.log('🛑 [stopTTS] Pausing Audio object:', player.src);
            player.pause();
            player.currentTime = 0;
            player.src = '';
            // Don't revoke URL immediately to avoid errors if play() is pending
          } else if (player.ctx && player.source) {
            try { player.source.stop(0); } catch (e) { }
            if (!player.streamingSharedContext) {
              try { player.ctx.close(); } catch (e) { }
            }
          }
        } catch (err) {
          console.error("🛑 [stopTTS] Error stopping player:", err);
        }
      });
      activeAudioPlayersRef.current.clear();
    }

    // Legacy cleanup / Fallback
    if (audioPlayerRef.current) {
      console.log('🛑 [stopTTS] stopping legacy audioPlayerRef');
      try {
        const p = audioPlayerRef.current;
        if (p instanceof Audio) {
          p.pause();
          p.currentTime = 0;
        } else if (p.source) {
          try { p.source.stop(0); } catch (e) { }
        }
      } catch (e) { }
      audioPlayerRef.current = null;
    }

    // 5. Clear local queues and state
    console.log('🛑 [stopTTS] Clearing local audio queues and state');
    if (ttsClient && ttsClient.audioQueue) {
      console.log(`🛑 [stopTTS] dumped ${ttsClient.audioQueue.length} items from audioQueue`);
      ttsClient.audioQueue.length = 0;
    }

    setAudioQueue([]);
    setIsAutoplaying(false);
    isStreamingTtsPausedRef.current = false;
    setIsStreamingTtsPaused(false);
    setIsPlayingAudio(null);
    setTtsPlaybackState({ messageId: null, phase: 'idle' });
    window.streamingAudioPlaying = false;
    if (window.intelStreamingAudioPlaying) {
      window.intelStreamingAudioPlaying = false;
      const onIntelComplete = streamingTtsStreamOptsRef.current?.onComplete;
      streamingTtsStreamOptsRef.current = null;
      if (typeof onIntelComplete === 'function') onIntelComplete();
    } else {
      // Also clear for non-intel streams to prevent stale ref leaks
      streamingTtsStreamOptsRef.current = null;
    }
    streamingTtsDrainingRef.current = false;
    streamingTtsDrainGenerationRef.current = null;

    // Explicitly update state to trigger UI changes
    setManuallyStoppedAudio(true);
    setTimeout(() => setManuallyStoppedAudio(false), 500);

  }, [ttsClient, setAudioQueue, setIsAutoplaying]);

  const handleStopGeneration = useCallback(() => {
    // 1. Stop the text generation stream
    if (abortController) {
      console.log('🛑 Aborting text generation...');
      abortController.abort();
      setAbortController(null);
    }

    // 2. Stop TTS explicitly
    stopTTS('handleStopGeneration');

    // Reset the main "isGenerating" flag for the UI
    setIsGenerating(false);

    // This flag is for UI feedback
    setIsStreamingStopped(true);
    setTimeout(() => setIsStreamingStopped(false), 1000);

  }, [abortController, stopTTS, setIsGenerating]);
  // NEW: Silent audio unlocker for mobile browsers
  const unlockAudioContext = useCallback(() => {
    // 1. Create context if missing
    if (!audioPlaybackRef.current.context) {
      audioPlaybackRef.current.context = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioPlaybackRef.current.context;

    // 2. Resume if suspended (common on mobile)
    if (ctx.state === 'suspended') {
      ctx.resume().then(() => console.log('🔊 [Audio] Context resumed via silent unlock'));
    }

    // 3. Play silent buffer to force "active" state
    try {
      const buffer = ctx.createBuffer(1, 1, 22050);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0);
      console.log('🔊 [Audio] Silent unlock played');
    } catch (e) {
      console.warn('🔊 [Audio] Unlock failed', e);
    }
  }, []);

  const getTtsOverridesForCharacter = useCallback((character) => {
    if (!character) return null;
    const engine = settings.ttsEngine || 'kokoro';
    if (engine !== 'chatterbox' && engine !== 'chatterbox_turbo' && engine !== 'chatterbox_nano' && engine !== 'kokoro' && engine !== 'voxcpm') return null;
    const voice = character.ttsVoice || character.tts_voice || null;
    if (!voice || voice === 'default') return null;
    // Validate per-character voice against current engine
    if (engine === 'kokoro') {
      if (voice.includes('.') || voice.includes('\\') || voice.includes('/')) {
        return null;
      }
    } else {
      if (!voice.includes('.')) {
        return null;
      }
    }
    return { ttsVoice: voice };
  }, [settings.ttsEngine]);

  const getTtsOverridesForCharacterId = useCallback((characterId) => {
    if (!characterId) return null;
    const character = characters.find((item) => item.id === characterId);
    return getTtsOverridesForCharacter(character);
  }, [characters, getTtsOverridesForCharacter]);

  const startStreamingTTS = useCallback((messageId, optionsOverrides = null, streamOpts = null) => {
    // ── Kill any previous drain loop before starting a new one ──────────
    // The old drain loop is an async function that may still be running from
    // a previous message.  It already passed the prebuffer gate, so it would
    // play new chunks immediately, bypassing the gate we need for RTF > 1.
    //
    // We can't use isTtsInterruptedRef because it gets reset to false a few
    // lines below — all synchronously, before the old async loop ever checks
    // it.  Instead we use a generation counter: increment it here, and the
    // drain loop captures its generation and breaks when it no longer matches.
    const prevGen = drainGenerationRef.current;
    drainGenerationRef.current = prevGen + 1;
    console.log(`🔁 [TTS Drain] Generation ${prevGen} → ${drainGenerationRef.current} for message ${messageId}`);

    // Stop any actively-playing audio sources from the previous stream
    if (activeAudioPlayersRef.current.size > 0) {
      const playersToStop = Array.from(activeAudioPlayersRef.current);
      console.log(`🔁 [TTS Drain] Stopping ${playersToStop.length} active players from gen ${prevGen}`);
      playersToStop.forEach(player => {
        try {
          if (player instanceof Audio) {
            player.pause();
            player.currentTime = 0;
            if (player.src) { try { URL.revokeObjectURL(player.src); } catch (e) {} player.src = ''; }
          } else if (player.ctx && player.source) {
            try { player.source.stop(0); } catch (e) { }
          }
        } catch (e) { /* noop */ }
      });
      activeAudioPlayersRef.current.clear();
      audioPlayerRef.current = null;
    }
    // ── End kill previous drain ─────────────────────────────────────────

    window.streamingAudioPlaying = false;
    window.intelStreamingAudioPlaying = false;
    streamingTtsDrainingRef.current = false;
    streamingTtsDrainGenerationRef.current = null;
    ttsClient.audioQueue = [];
    streamingTtsWsEndSentRef.current = false;
    // Reset WS close flag so the drain loop can detect socket closure mid-stream
    ttsWsClosedRef.current = false;
    streamingTtsStreamOptsRef.current = streamOpts || null;
    const isIntelPlayback =
      streamOpts?.intelPlayback === true || isIntelMessageId(messageId);
    const coalesceStart = ttsStreamSendCoalesceRef.current;
    if (coalesceStart.rafId != null) {
      cancelAnimationFrame(coalesceStart.rafId);
      coalesceStart.rafId = null;
    }
    coalesceStart.pending = '';

    const bypassAutoplayGate = streamOpts?.bypassAutoplayGate === true;
    if ((!settings.ttsAutoPlay && !bypassAutoplayGate) || !settings.ttsEnabled) {
      console.warn(`🚫 [startStreamingTTS] BLOCKED — ttsAutoPlay=${settings.ttsAutoPlay}, bypass=${bypassAutoplayGate}, ttsEnabled=${settings.ttsEnabled}`);
      return;
    }

    ttsPlaybackRequestRef.current += 1;
    unlockAudioContext();
    console.warn(`▶️ [startStreamingTTS] Starting stream for ${messageId} — prebufferSetting=${settings.ttsPrebufferSeconds}, prebufferRef=${ttsPrebufferSecondsRef.current}`);

    console.log(`▶️ [startStreamingTTS] Starting stream for ${messageId} (Resetting interrupt flag)`);
    isFirstTextChunk.current = true;
    isTtsInterruptedRef.current = false;
    isStreamingTtsPausedRef.current = false;
    setIsStreamingTtsPaused(false);
    streamingTtsMessageIdRef.current = messageId;
    if (!isIntelPlayback) {
      setIsAutoplaying(true);
      setIsPlayingAudio(messageId);
      setTtsPlaybackState({ messageId, phase: 'synthesising' });
    }

    // Safety: If socket is closed/closing, prompt a reconnect so pending settings send
    if (!ttsClient.socket || ttsClient.socket.readyState > 1) {
      console.warn("⚠️ [TTS] Socket closed/closing/null, triggering reconnect...");
      ttsClient.connect();
    }

    // Listen for WebSocket close during active stream to abort the drain loop early
    const sock = ttsClient.socket;
    if (sock) {
      const handleWsClose = () => {
        console.warn("🛑 [TTS] WebSocket closed mid-stream — aborting drain");
        ttsWsClosedRef.current = true;
        isTtsInterruptedRef.current = true;
        streamingTtsMessageIdRef.current = null;
      };
      sock.addEventListener('close', handleWsClose, { once: true });
    }

    // Local flag to track if this is the first audio chunk for this specific message
    let firstAudioPlayed = false;

    const ttsEngine = optionsOverrides?.ttsEngine || settings.ttsEngine || 'kokoro';
    const ttsVoice = optionsOverrides?.ttsVoice || settings.ttsVoice || 'af_heart';

    const streamingTtsSettings = {
      engine: ttsEngine,
      voice: ttsVoice,
      speed: normaliseTtsSpeed(optionsOverrides?.ttsSpeed ?? settings.ttsSpeed ?? 1.0),
      exaggeration: optionsOverrides?.ttsExaggeration ?? settings.ttsExaggeration ?? 0.5,
      cfg: optionsOverrides?.ttsCfg ?? settings.ttsCfg ?? 0.5,
      stream_chunk_sentences: Math.max(
        3,
        Math.min(
          12,
          Number.parseInt(optionsOverrides?.ttsStreamChunkSentences ?? settings.ttsStreamChunkSentences ?? 3, 10) || 3
        )
      ),
      frontend_prebuffer_seconds: Math.max(
        0,
        Number(
          optionsOverrides?.ttsPrebufferSeconds
          ?? ttsPrebufferSecondsRef.current
          ?? settingsRef.current?.ttsPrebufferSeconds
          ?? settings.ttsPrebufferSeconds
          ?? 0
        ) || 0
      ),
      audio_prompt_path: ((ttsEngine === 'chatterbox') || (ttsEngine === 'chatterbox_turbo') || (ttsEngine === 'chatterbox_nano') || (ttsEngine === 'voxcpm')) && ttsVoice !== 'default'
        ? ttsVoice
        : null,
      // VoxCPM2-specific settings
      voxcpm_cfg_value: optionsOverrides?.voxcpmCfgValue ?? settings.voxcpmCfgValue ?? 2.0,
      voxcpm_inference_timesteps: optionsOverrides?.voxcpmInferenceTimesteps ?? settings.voxcpmInferenceTimesteps ?? 8,
      voxcpm_normalize: optionsOverrides?.voxcpmNormalize ?? settings.voxcpmNormalize ?? false,
      voxcpm_denoise: optionsOverrides?.voxcpmDenoise ?? settings.voxcpmDenoise ?? false,
      voxcpm_retry_badcase: optionsOverrides?.voxcpmRetryBadcase ?? settings.voxcpmRetryBadcase ?? false,
      voxcpm_voice_design: optionsOverrides?.voxcpmVoiceDesign ?? settings.voxcpmVoiceDesign ?? '',
    };

    ttsClient.send(streamingTtsSettings);

    console.warn(`🔌 [TTS] Assigning onAudioQueueUpdate handler for msg=${messageId} (prebufferRef=${ttsPrebufferSecondsRef.current})`);
    ttsClient.onAudioQueueUpdate = () => {
      console.warn(`🔔 [TTS onAudioQueueUpdate] Fired! queue=${ttsClient.audioQueue.length}, msg=${messageId}, interrupt=${isTtsInterruptedRef.current}, draining=${streamingTtsDrainingRef.current}, msgRef=${streamingTtsMessageIdRef.current}, prebufferRef=${ttsPrebufferSecondsRef.current}, settingsPrebuffer=${settingsRef.current?.ttsPrebufferSeconds}`);
      if (isTtsInterruptedRef.current) {
        console.warn(`🛡️ [TTS] BLOCKED by interrupt flag. Dumping ${ttsClient.audioQueue.length} items.`);
        ttsClient.audioQueue.length = 0;
        return;
      }

      if (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId) {
        console.warn(`🛡️ [TTS] BLOCKED by stale message ID ${messageId} (Current: ${streamingTtsMessageIdRef.current})`);
        return;
      }

      // Do NOT call setAudioQueue here: each WAV chunk updated React state and re-rendered the whole app.
      // With multi‑MB chat histories that blocked the main thread for seconds between sentences.

      if (ttsClient.audioQueue.length === 0) {
        console.warn(`🛡️ [TTS] BLOCKED — audioQueue empty`);
        return;
      }

      // One async drain for the whole stream: first blob kicks this off; later blobs only grow the queue.
      const handlerGeneration = drainGenerationRef.current;
      if (
        streamingTtsDrainingRef.current
        && streamingTtsDrainGenerationRef.current === handlerGeneration
      ) {
        console.warn(`🛡️ [TTS] BLOCKED — drain already running (drainingRef=true)`);
        return;
      }

      streamingTtsDrainingRef.current = true;
      streamingTtsDrainGenerationRef.current = handlerGeneration;
      if (isIntelPlayback) {
        window.intelStreamingAudioPlaying = true;
      } else {
        window.streamingAudioPlaying = true;
        setIsPlayingAudio(messageId);
      }

      void (async () => {
        const myGen = handlerGeneration;
        console.warn(`🟢 [TTS Drain Gen ${myGen}] Async drain loop started for ${messageId} (gen=${drainGenerationRef.current}, myGen=${myGen})`);
        try {
          const requestedTtsSpeed = normaliseTtsSpeed(
            optionsOverrides?.ttsSpeed ?? settings.ttsSpeed ?? 1.0
          );
          const streamPlaybackRate = getTtsPlaybackRate(ttsEngine, requestedTtsSpeed);
          // Kokoro and NanoGPT own their synthesis speed. Other engines use pitch-preserving
          // HTML audio when the requested playback speed differs from 1×.
          const useGaplessBuffer = Math.abs(streamPlaybackRate - 1) < 0.0001;

          const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
          let ctx = null;
          let nextStartTime = 0;
          let savedDecodePromise = null;
          if (useGaplessBuffer) {
            ctx = audioPlaybackRef.current.context;
            if (!ctx) {
              ctx = new AudioContextCtor();
              audioPlaybackRef.current.context = ctx;
            }
            if (ctx.state === 'suspended') {
              await ctx.resume();
            }
            nextStartTime = ctx.currentTime;
          }

          const streamStillValid = () =>
            drainGenerationRef.current === myGen
            && !isTtsInterruptedRef.current
            && (!streamingTtsMessageIdRef.current || streamingTtsMessageIdRef.current === messageId);
          const waitForIncomingAudio = async (maxWaitMs = 260, stepMs = 20) => {
            // Also check if WebSocket closed mid-stream so we don't stall indefinitely.
            if (ttsWsClosedRef.current) return false;
            // Keep the drain loop alive so the next WAV can arrive while synthesis catches up.
            // Sub‑second timeouts caused apparent multi‑second gaps: we'd tear down the drain, then
            // restart when the chunk appeared — silence while synth + reconnect jitter stacked up.
            const deadline = performance.now() + maxWaitMs;
            while (performance.now() < deadline) {
              if (!streamStillValid()) return false;
              if (ttsClient.audioQueue.length > 0) return true;
              await new Promise((resolve) => setTimeout(resolve, stepMs));
            }
            return ttsClient.audioQueue.length > 0;
          };
          const waitWhilePaused = async () => {
            while (isStreamingTtsPausedRef.current && streamStillValid()) {
              await new Promise((resolve) => setTimeout(resolve, 50));
            }
          };

          const applySubtitleCue = (sub, wallDurationMs) => {
            if (!sub) return;
            const cue = {
              text: sub.text,
              durationMs: wallDurationMs,
              timestamp: Date.now()
            };
            window.__ttsSubtitleCue = cue;
            setTtsSubtitleCue(cue);
          };

          // ── Prebuffer gate ──────────────────────────────────────────────
          // With RTF > 1 (synthesis slower than real-time) the audio buffer
          // drains faster than it fills.  Starting playback on the first chunk
          // causes mid-stream stalls.  Wait until enough audio has accumulated
          // so that once playback begins it runs continuously.
          //
          // Read from multiple sources to handle any stale closure / cross-window
          // sync timing issues.  settingsRef.current is updated synchronously on
          // every render (not in a useEffect), so it is always current.
          let prebufferSeconds = 0;
          if (optionsOverrides && optionsOverrides.ttsPrebufferSeconds != null) {
            prebufferSeconds = Number(optionsOverrides.ttsPrebufferSeconds) || 0;
          } else if (ttsPrebufferSecondsRef.current > 0) {
            prebufferSeconds = ttsPrebufferSecondsRef.current;
          } else if (settingsRef.current && Number(settingsRef.current.ttsPrebufferSeconds) > 0) {
            prebufferSeconds = Number(settingsRef.current.ttsPrebufferSeconds);
          }
          console.warn(`🎛️ [TTS Prebuffer Gen ${myGen}] GATE START: prebufferSeconds=${prebufferSeconds} (optionsOverrides?.ttsPrebufferSeconds=${optionsOverrides?.ttsPrebufferSeconds}, ref=${ttsPrebufferSecondsRef.current}, settingsRef=${settingsRef.current?.ttsPrebufferSeconds}, isIntel=${isIntelPlayback})`);
          if (prebufferSeconds > 0 && !isIntelPlayback) {
            const targetMs = prebufferSeconds * 1000;
            console.warn(`⏳ [TTS Prebuffer Gen ${myGen}] WAITING: target=${targetMs}ms (${prebufferSeconds}s)`);
            let lastQueueLen = -1;
            let stableSince = 0;
            let pollCount = 0;
            while (true) {
              if (!streamStillValid()) {
                console.log(`🛑 [TTS Prebuffer Gen ${myGen}] Stream no longer valid, aborting gate.`);
                break;
              }
              let bufferedMs = 0;
              for (let i = 0; i < ttsClient.audioQueue.length; i++) {
                bufferedMs += estimateChunkDurationMs(ttsClient.audioQueue[i]);
              }
              pollCount++;
              if (pollCount % 3 === 1) { // Log every ~300ms for faster debugging
                console.warn(`⏳ [TTS Prebuffer Gen ${myGen}] Poll ${pollCount}: ${Math.round(bufferedMs)}ms buffered / ${targetMs}ms target, queue=${ttsClient.audioQueue.length}, wsEnd=${streamingTtsWsEndSentRef.current}`);
              }
              if (bufferedMs >= targetMs) {
                console.log(`✅ [TTS Prebuffer Gen ${myGen}] Target reached: ${Math.round(bufferedMs)}ms ≥ ${targetMs}ms. Starting playback after ${pollCount} polls.`);
                break;
              }
              if (ttsWsClosedRef.current) {
                console.log(`✅ [TTS Prebuffer Gen ${myGen}] WebSocket closed with ${Math.round(bufferedMs)}ms buffered. Starting playback.`);
                break;
              }
              // Escape hatch for short responses that never reach the target:
              // once all text has been sent, if the queue stops growing the
              // backend has finished synthesizing — start with what we have.
              if (streamingTtsWsEndSentRef.current) {
                const qLen = ttsClient.audioQueue.length;
                const now = performance.now();
                if (qLen !== lastQueueLen) {
                  lastQueueLen = qLen;
                  stableSince = now;
                } else if (now - stableSince > 3000) {
                  console.log(`✅ [TTS Prebuffer Gen ${myGen}] Stream ended, queue stable 3s with ${Math.round(bufferedMs)}ms buffered. Starting playback.`);
                  break;
                }
              }
              await new Promise((r) => setTimeout(r, 100));
            }
          } else {
            console.warn(`⏭️ [TTS Prebuffer Gen ${myGen}] SKIPPED — prebufferSeconds=${prebufferSeconds}, isIntel=${isIntelPlayback}`);
          }
          // ── End prebuffer gate ──────────────────────────────────────────
          console.warn(`▶️ [TTS Drain Gen ${myGen}] Prebuffer gate passed, entering playback loop.`);

          // Outer loop: after each "burst", more chunks may have arrived over the WebSocket.
          // Inner loop: either gapless BufferSource (1×) or sequential HTMLAudio (preservesPitch).
          while (true) {
            await waitWhilePaused();
            if (isTtsInterruptedRef.current) {
              console.log("🛑 [TTS] Loop broken by interrupt flag (Pre-Shift).");
              ttsClient.audioQueue.length = 0;
              break;
            }
            if (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId) {
              console.log("🛑 [TTS] Loop broken by stale message ID (Pre-Shift).");
              break;
            }
            if (ttsClient.audioQueue.length === 0) {
              // Insight readout sends all text upfront — release UI quickly after playback.
              const tailMs = isIntelPlayback
                ? 350
                : (isMobileRef.current ? 6500 : 5500);
              const midStreamMs = isIntelPlayback
                ? 900
                : (isMobileRef.current ? 26000 : 22000);
              const adaptiveGapGuardMs = streamingTtsWsEndSentRef.current ? tailMs : midStreamMs;
              const gotLateChunk = await waitForIncomingAudio(adaptiveGapGuardMs, 20);
              if (!gotLateChunk) break;
            }

            let burstDoneResolve = () => {};
            const burstDone = new Promise((r) => { burstDoneResolve = r; });
            const pending = { n: 0 };

            while (ttsClient.audioQueue.length > 0 && !isTtsInterruptedRef.current) {
              await waitWhilePaused();
              if (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId) {
                break;
              }

              const chunk = ttsClient.audioQueue.shift();
              if (!chunk) break;

              const arrayBuffer = chunk.audio || chunk;
              const subtitle = chunk.subtitle;

              if (!useGaplessBuffer) {
                const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                try {
                  audio.preservesPitch = true;
                } catch (e) { /* noop */ }
                try {
                  if ('webkitPreservesPitch' in audio) audio.webkitPreservesPitch = true;
                } catch (e) { /* noop */ }
                audio.playbackRate = streamPlaybackRate;

                const sub = subtitle;
                if (sub?.durationMs) {
                  applySubtitleCue(
                    sub,
                    Math.max(200, Math.round(sub.durationMs / streamPlaybackRate))
                  );
                } else if (sub) {
                  const onceMeta = () => {
                    const sec = audio.duration;
                    const baseMs = sec && Number.isFinite(sec) ? Math.round(sec * 1000) : 1500;
                    applySubtitleCue(sub, Math.max(200, Math.round(baseMs / streamPlaybackRate)));
                  };
                  if (audio.readyState >= 1) onceMeta();
                  else audio.addEventListener('loadedmetadata', onceMeta, { once: true });
                }

                if (!firstAudioPlayed) {
                  firstAudioPlayed = true;
                  if (!isIntelPlayback) {
                    setTtsPlaybackState({ messageId, phase: 'playing' });
                  }
                  const now = performance.now();
                  if (textGenerationStartTime.current) {
                    console.log(`⏱️ [TTS Timing] 🟡 Latency (Text Chunk -> Audio): ${(now - textGenerationStartTime.current).toFixed(2)}ms`);
                  }
                  if (promptSubmissionStartTime.current) {
                    console.log(`⏱️ [TTS Timing] 🟢 Time to First Audio (from Prompt): ${(now - promptSubmissionStartTime.current).toFixed(2)}ms`);
                  }
                }

                activeAudioPlayersRef.current.add(audio);
                audioPlayerRef.current = audio;

                const cleanupHtml = () => {
                  try { URL.revokeObjectURL(audioUrl); } catch (e) { /* noop */ }
                  activeAudioPlayersRef.current.delete(audio);
                  if (audioPlayerRef.current === audio) audioPlayerRef.current = null;
                };

                let poll = null;
                await new Promise((resolve) => {
                  let finished = false;
                  const finish = () => {
                    if (finished) return;
                    finished = true;
                    if (poll) clearInterval(poll);
                    cleanupHtml();
                    resolve();
                  };
                  audio.addEventListener('ended', finish, { once: true });
                  audio.addEventListener('error', finish, { once: true });
                  poll = setInterval(() => {
                    if (!streamStillValid()) {
                      try { audio.pause(); } catch (e) { /* noop */ }
                      finish();
                    }
                  }, 50);
                  audio.play().catch((e) => {
                    console.error("🎵 [TTS] HTML audio play failed:", e);
                    finish();
                  });
                });
                continue;
              }

              const rawAb = arrayBuffer.slice(0);

              let audioBuffer;
              try {
                if (savedDecodePromise) {
                  audioBuffer = await savedDecodePromise;
                  savedDecodePromise = null;
                } else {
                  audioBuffer = await ctx.decodeAudioData(rawAb);
                }
              } catch (decodeErr) {
                console.error("🎵 [TTS] decodeAudioData failed:", decodeErr);
                continue;
              }

              if (ttsClient.audioQueue.length > 0) {
                const nextChunk = ttsClient.audioQueue[0];
                const nextAb = (nextChunk.audio || nextChunk).slice(0);
                savedDecodePromise = ctx.decodeAudioData(nextAb);
              }

              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.playbackRate.value = 1.0;
              source.connect(ctx.destination);

              const startAt = Math.max(nextStartTime, ctx.currentTime);
              nextStartTime = startAt + audioBuffer.duration;

              if (subtitle) {
                const delayMs = Math.max(0, (startAt - ctx.currentTime) * 1000);
                const sub = subtitle;
                const baseDurMs = sub.durationMs ?? Math.round(audioBuffer.duration * 1000);
                const wallDurationMs = Math.max(200, Math.round(baseDurMs));
                if (delayMs < 8) {
                  applySubtitleCue(sub, wallDurationMs);
                } else {
                  setTimeout(() => applySubtitleCue(sub, wallDurationMs), delayMs);
                }
              }

              if (!firstAudioPlayed) {
                firstAudioPlayed = true;
                if (!isIntelPlayback) {
                  setTtsPlaybackState({ messageId, phase: 'playing' });
                }
                const now = performance.now();
                if (textGenerationStartTime.current) {
                  console.log(`⏱️ [TTS Timing] 🟡 Latency (Text Chunk -> Audio): ${(now - textGenerationStartTime.current).toFixed(2)}ms`);
                }
                if (promptSubmissionStartTime.current) {
                  console.log(`⏱️ [TTS Timing] 🟢 Time to First Audio (from Prompt): ${(now - promptSubmissionStartTime.current).toFixed(2)}ms`);
                }
              }

              const playerHandle = { ctx, source, streamingSharedContext: true };
              activeAudioPlayersRef.current.add(playerHandle);
              audioPlayerRef.current = playerHandle;

              pending.n += 1;
              source.onended = () => {
                activeAudioPlayersRef.current.delete(playerHandle);
                if (audioPlayerRef.current === playerHandle) audioPlayerRef.current = null;
                pending.n -= 1;
                if (pending.n <= 0) burstDoneResolve();
              };

              try {
                source.start(startAt);
              } catch (e) {
                activeAudioPlayersRef.current.delete(playerHandle);
                if (audioPlayerRef.current === playerHandle) audioPlayerRef.current = null;
                pending.n -= 1;
                if (pending.n <= 0) burstDoneResolve();
                console.error("🎵 [TTS] source.start failed:", e);
              }
            }

            if (pending.n > 0) {
              await burstDone;
            }

            if (isTtsInterruptedRef.current || (streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId)) {
              console.log("🛑 [TTS] Loop broken by interrupt flag or stale ID (Post-Play).");
              break;
            }
          }
        } catch (err) {
          console.error("🎵 [TTS Playback Error]", err);
        } finally {
          if (streamingTtsDrainGenerationRef.current !== myGen) {
            console.warn(`🎵 [TTS] Stale drain Gen ${myGen} finished without releasing the current playback lock.`);
            return;
          }
          console.warn(`🎵 [TTS] Drain Gen ${myGen} finished. drainingRef reset to false.`);
          streamingTtsDrainingRef.current = false;
          streamingTtsDrainGenerationRef.current = null;
          
          // Always clear the stream opts ref to prevent stale closures on next use
          const wasIntel = isIntelPlayback || (streamingTtsStreamOptsRef.current?.intelPlayback === true);
          if (wasIntel) {
            window.intelStreamingAudioPlaying = false;
          } else {
            window.streamingAudioPlaying = false;
            setIsPlayingAudio(null);
            setTtsPlaybackState({ messageId: null, phase: 'idle' });
            setIsAutoplaying(false);
            isStreamingTtsPausedRef.current = false;
            setIsStreamingTtsPaused(false);
            setAudioQueue([]);
          }
          
          // Clear stream opts regardless of completion reason (normal, interrupt, or stale)
          if (!streamingTtsStreamOptsRef.current?.onComplete || wasIntel) {
            streamingTtsStreamOptsRef.current = null;
          }

          const isStale = streamingTtsMessageIdRef.current && streamingTtsMessageIdRef.current !== messageId;
          if (!isStale && streamingTtsMessageIdRef.current === messageId) {
            streamingTtsMessageIdRef.current = null;
          }
        }
      })();
    };
  }, [settings, ttsClient, setAudioQueue, setIsAutoplaying, unlockAudioContext]);

  const pauseStreamingTTS = useCallback(() => {
    if (isStreamingTtsPausedRef.current) return;
    if (activeAudioPlayersRef.current.size === 0 && !streamingTtsDrainingRef.current) return;
    isStreamingTtsPausedRef.current = true;
    setIsStreamingTtsPaused(true);
    activeAudioPlayersRef.current.forEach((player) => {
      try {
        if (player instanceof Audio) {
          player.pause();
        } else if (player?.ctx?.state === 'running') {
          player.ctx.suspend().catch(() => {});
        }
      } catch (e) {
        // noop
      }
    });
  }, []);

  const resumeStreamingTTS = useCallback(() => {
    if (!isStreamingTtsPausedRef.current) return;
    isStreamingTtsPausedRef.current = false;
    setIsStreamingTtsPaused(false);
    unlockAudioContext();
    activeAudioPlayersRef.current.forEach((player) => {
      try {
        if (player instanceof Audio && player.paused) {
          player.play().catch(() => {});
        } else if (player?.ctx?.state === 'suspended') {
          player.ctx.resume().catch(() => {});
        }
      } catch (e) {
        // noop
      }
    });
  }, [unlockAudioContext]);

  const stopStreamingTTS = useCallback(() => {
    console.log("🛑 [stopStreamingTTS] Called, delegating to stopTTS");
    stopTTS('stopStreamingTTS');
  }, [stopTTS]);
  const playNextChunk = async () => {
    if (ttsClient.audioQueue.length === 0 || isChunkPlaying) return;

    setIsChunkPlaying(true);
    const arrayBuffer = ttsClient.audioQueue.shift();

    try {
      const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);

      console.log(`🎵 [Sequential] Playing chunk, ${ttsClient.audioQueue.length} remaining`);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsChunkPlaying(false);
        // Play next chunk when this one ends
        if (ttsClient.audioQueue.length > 0) {
          setTimeout(playNextChunk, 50); // Small delay to avoid race conditions
        }
      };

      await audio.play();
    } catch (error) {
      console.error(`🎵 [Sequential] Error:`, error);
      setIsChunkPlaying(false);
    }
  };

  const addStreamingText = useCallback((newTextChunk, opts = null) => {
    if (isFirstTextChunk.current) {
      textGenerationStartTime.current = performance.now();
      console.log(`⏱️ [TTS Timing] Timer started on first received text chunk.`);
      isFirstTextChunk.current = false; // So this only runs once per message
    }

    // ONLY send to TTS if we have an active message ID (haven't been stopped)
    // FIX: Removed 'isAutoplaying' check to avoid stale state closure issues.
    // streamingTtsMessageIdRef is the reliable, synchronous source of truth.
    if (!streamingTtsMessageIdRef.current || !newTextChunk) return;

    const sendImmediately = opts?.immediate === true;
    if (sendImmediately) {
      ttsClient.send(newTextChunk);
      return;
    }

    if (ttsWaitForFullResponseRef.current) {
      ttsFullResponseBufferRef.current += newTextChunk;
      return;
    }

    const box = ttsStreamSendCoalesceRef.current;
    box.pending += newTextChunk;
    if (box.rafId != null) return;
    box.rafId = requestAnimationFrame(() => {
      box.rafId = null;
      const out = box.pending;
      box.pending = '';
      if (out && streamingTtsMessageIdRef.current) {
        ttsClient.send(out);
      }
    });
  }, [ttsClient]); // Removed isAutoplaying dependency

  const endStreamingTTS = useCallback(() => {
    // Note: Don't clear ttsSubtitleCue here - let it expire naturally
    const box = ttsStreamSendCoalesceRef.current;
    if (box.rafId != null) {
      cancelAnimationFrame(box.rafId);
      box.rafId = null;
    }
    const tail = box.pending;
    box.pending = '';
    const activeId = streamingTtsMessageIdRef.current;
    if (tail && activeId) {
      ttsClient.send(tail);
    }
    // Flush buffered full-response text (ttsWaitForFullResponse mode)
    const buffered = ttsFullResponseBufferRef.current;
    if (buffered && activeId) {
      ttsFullResponseBufferRef.current = '';
      ttsClient.send(buffered);
    }
    if (activeId) {
      console.log(`⏹️ [TTS Stream] Ending for message ${activeId}`);
      ttsClient.closeStream();
      streamingTtsWsEndSentRef.current = true;
      streamingTtsMessageIdRef.current = null;
    }
  }, [ttsClient]);

  const ensureTtsSocketOpen = useCallback(async () => {
    if (ttsClient.socket?.readyState === WebSocket.OPEN) return;
    ttsClient.connect(
      () => {},
      () => {},
      () => {}
    );
    const sock = ttsClient.socket;
    if (!sock) throw new Error('TTS WebSocket unavailable');
    if (sock.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('TTS connection timed out')), 20000);
      sock.addEventListener(
        'open',
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true }
      );
      sock.addEventListener(
        'error',
        () => {
          clearTimeout(timer);
          reject(new Error('TTS WebSocket error'));
        },
        { once: true }
      );
    });
  }, [ttsClient]);

  /** WebSocket streaming autoplay — sentence chunks synthesize in parallel with playback. */
  const playStreamingTtsScript = useCallback(
    async (messageId, text, optionsOverrides = null, streamOpts = null) => {
      const t = (text || '').trim();
      if (!t || !settings.ttsEnabled) return;
      if (
        streamingTtsMessageIdRef.current === messageId
        && !isTtsInterruptedRef.current
      ) {
        return;
      }
      const intelPlayback =
        streamOpts?.intelPlayback === true || isIntelMessageId(messageId);
      setAudioError(null);
      try {
        await ensureTtsSocketOpen();
      } catch (e) {
        setAudioError(e?.message || 'Could not connect to the TTS server. Is it running?');
        return;
      }
      stopTTS('playStreamingTtsScript');
      unlockAudioContext();
      startStreamingTTS(messageId, optionsOverrides, {
        bypassAutoplayGate: true,
        intelPlayback,
        ...streamOpts,
      });
      addStreamingText(t, { immediate: true });
      endStreamingTTS();
    },
    [
      settings.ttsEnabled,
      ensureTtsSocketOpen,
      stopTTS,
      unlockAudioContext,
      startStreamingTTS,
      addStreamingText,
      endStreamingTTS,
      setAudioError,
    ]
  );

  /** Same WebSocket streaming path as chat autoplay (chunked synthesis + gapless drain). */
  const playTestStreamingTTS = useCallback(
    (text, optionsOverrides = null) => playStreamingTtsScript('test-tts', text, optionsOverrides),
    [playStreamingTtsScript]
  );

  const checkSdStatus = useCallback(async () => {

    const imageEngine = settings.imageEngine || 'EloDiffusion';

    let localSdStatus = false;
    let comfyuiStatus = false;
    let comfyuiData = null;

    // Check local SD if needed  
    if (imageEngine === 'EloDiffusion' || imageEngine === 'both') {
      try {
        const res = await fetch(`${MEMORY_API_URL}/sd-local/status`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          localSdStatus = Boolean(data.available);
        }
      } catch (err) {
      }
    }

    // Check ComfyUI if needed
    if (imageEngine === 'comfyui' || imageEngine === 'both') {
      try {
        const res = await fetch(`${PRIMARY_API_URL}/sd-comfy/status`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          comfyuiStatus = Boolean(data.comfyui);
          comfyuiData = data; // Contains checkpoints, samplers, schedulers, etc.
        }
      } catch (err) {
      }
    }

    setSdStatus({
      localSd: localSdStatus,
      comfyui: comfyuiStatus,
      comfyuiData: comfyuiData,
      models: [] // We'll populate this later if needed
    });
  }, [PRIMARY_API_URL, MEMORY_API_URL, settings.imageEngine]);

  const fetchAvailableSTTEngines = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/available-engines`);
      if (response.ok) {
        const data = await response.json();
        setSttEnginesAvailable(data.available_engines || ['whisper']);
      }
    } catch (error) {
      console.error("Error fetching available STT engines:", error);
      setSttEnginesAvailable(['whisper']); // Default to Whisper if fetch fails
    }
  }, [PRIMARY_API_URL]);

  const fetchNanogptSttModels = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/nanogpt-models`);
      if (response.ok) {
        const data = await response.json();
        setNanogptSttModels(data.models || []);
      }
    } catch (error) {
      console.error("Error fetching NanoGPT STT models:", error);
      setNanogptSttModels([]);
    }
  }, [PRIMARY_API_URL]);

  const fetchNanogptTtsModels = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/nanogpt-models`);
      if (response.ok) {
        const data = await response.json();
        setNanogptTtsModels(data.models || []);
      }
    } catch (error) {
      console.error("Error fetching NanoGPT TTS models:", error);
      setNanogptTtsModels([]);
    }
  }, [PRIMARY_API_URL]);

  const fetchParakeetCppModels = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/parakeet-cpp/models`);
      if (response.ok) {
        const data = await response.json();
        setParakeetCppModels(data.models || []);
        setParakeetCppCliAvailable(data.cli_available || false);
      }
    } catch (error) {
      console.error("Error fetching Parakeet.cpp models:", error);
      setParakeetCppModels([]);
    }
  }, [PRIMARY_API_URL]);

  const downloadParakeetCppModel = useCallback(async (modelId, quant) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/parakeet-cpp/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, quant }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        await fetchParakeetCppModels();
        return { success: true, message: data.message };
      }
      return { success: false, message: data.message || 'Download failed' };
    } catch (error) {
      console.error("Error downloading Parakeet.cpp model:", error);
      return { success: false, message: error.message };
    }
  }, [PRIMARY_API_URL, fetchParakeetCppModels]);

  const deleteParakeetCppModel = useCallback(async (filename) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/stt/parakeet-cpp/model?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      const data = await response.json();
      if (data.status === 'success') {
        await fetchParakeetCppModels();
        return { success: true, message: data.message };
      }
      return { success: false, message: data.message || 'Delete failed' };
    } catch (error) {
      console.error("Error deleting Parakeet.cpp model:", error);
      return { success: false, message: error.message };
    }
  }, [PRIMARY_API_URL, fetchParakeetCppModels]);

  const fetchVoxcpmGgufModels = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/voxcpm-gguf/models`);
      if (response.ok) {
        const data = await response.json();
        setVoxcpmGgufModels(data.models || []);
        setVoxcpmGgufCliAvailable(data.cli_available || false);
      }
    } catch (error) {
      console.error("Error fetching VoxCPM2 GGUF models:", error);
      setVoxcpmGgufModels([]);
    }
  }, [PRIMARY_API_URL]);

  const downloadVoxcpmGgufModel = useCallback(async (modelId) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/voxcpm-gguf/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        await fetchVoxcpmGgufModels();
        return { success: true, message: data.message };
      }
      return { success: false, message: data.message || 'Download failed' };
    } catch (error) {
      console.error("Error downloading VoxCPM2 GGUF model:", error);
      return { success: false, message: error.message };
    }
  }, [PRIMARY_API_URL, fetchVoxcpmGgufModels]);

  const deleteVoxcpmGgufModel = useCallback(async (filename) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/voxcpm-gguf/model?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      const data = await response.json();
      if (data.status === 'success') {
        await fetchVoxcpmGgufModels();
        return { success: true, message: data.message };
      }
      return { success: false, message: data.message || 'Delete failed' };
    } catch (error) {
      console.error("Error deleting VoxCPM2 GGUF model:", error);
      return { success: false, message: error.message };
    }
  }, [PRIMARY_API_URL, fetchVoxcpmGgufModels]);

  // 2. Generate image via POST - supports Local SD and ComfyUI
  // optional 4th arg: { signal } for AbortController (cancel in-flight request)
  const generateImage = useCallback(async (prompt, opts, gpuId = 0, options = {}) => {
    console.log(`Starting image generation for GPU ${gpuId} with prompt:`, prompt);
    console.log("Image generation options:", opts);

    // Use settings from the context
    const imageEngine = settings.imageEngine || 'EloDiffusion';

    // Determine endpoint based on engine
    let endpoint;
    if (imageEngine === 'EloDiffusion') {
      const targetApiUrl = gpuId === 1 ? MEMORY_API_URL : PRIMARY_API_URL;
      endpoint = `${targetApiUrl}/sd-local/txt2img`;
    } else if (imageEngine === 'comfyui') {
      endpoint = `${PRIMARY_API_URL}/sd-comfy/txt2img`;
    } else if (imageEngine === 'nanogpt') {
      endpoint = `${PRIMARY_API_URL}/sd/nanogpt`;
    } else {
      endpoint = `${PRIMARY_API_URL}/sd/txt2img`;
    }

    // Sampler mapping for different backends
    const mapSamplerForBackend = (samplerNameFromFrontend) => {
      if (imageEngine === 'EloDiffusion') {
        return samplerNameFromFrontend;
      }
      if (imageEngine === 'comfyui') {
        // ComfyUI uses lowercase sampler names
        const comfySamplerMapping = {
          'euler_a': 'euler_ancestral',
          'Euler a': 'euler_ancestral',
          'euler': 'euler',
          'Euler': 'euler',
          'dpmpp2m': 'dpmpp_2m',
          'DPM++ 2M Karras': 'dpmpp_2m',
          'dpmpp2s_a': 'dpmpp_2s_ancestral',
          'DPM++ 2S a Karras': 'dpmpp_2s_ancestral',
          'heun': 'heun',
          'Heun': 'heun',
          'dpm2': 'dpm_2',
          'DPM2': 'dpm_2',
          'ddim': 'ddim',
          'DDIM': 'ddim',
        };
        return comfySamplerMapping[samplerNameFromFrontend] || samplerNameFromFrontend.toLowerCase();
      }
      // Default mapping (EloDiffusion uses frontend names directly)
      return samplerNameFromFrontend;
    };

    setIsImageGenerating(true);
    clearError();
    try {
      let payload;

      if (imageEngine === 'comfyui') {
        // ComfyUI payload format
        payload = {
          prompt,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          cfg_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          scheduler: opts.scheduler || "normal",
          seed: opts.seed || -1,
          checkpoint: opts.checkpoint || opts.model || "",
          batch_size: opts.batch_size || 1,
          denoise: opts.denoise || 1.0,
          timeout: 300
        };
      } else if (imageEngine === 'EloDiffusion') {
        // Local SD payload
        payload = {
          prompt,
          gpu_id: gpuId,
          task_id: opts.task_id,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          guidance_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          seed: opts.seed || -1,
        };
      } else if (imageEngine === 'nanogpt') {
        // NanoGPT payload
        payload = {
          prompt,
          width: opts.width || 1024,
          height: opts.height || 1024,
          model: settings.nanoGptModel || 'dall-e-3',
          api_key: settings.nanoGptApiKey,
          // NanoGPT / OpenAI DALL-E 3 doesn't support negative_prompt, steps, cfg, sampler often
          // But we can pass them if the backend filters them or if using a different model
          n: 1,
          size: `${opts.width || 1024}x${opts.height || 1024}`,
          quality: "standard",
          response_format: "url"
        };
      } else {
        // Default: EloDiffusion payload
        payload = {
          prompt,
          gpu_id: gpuId,
          task_id: opts.task_id,
          negative_prompt: opts.negative_prompt || "",
          width: opts.width || 512,
          height: opts.height || 512,
          steps: opts.steps || 20,
          guidance_scale: opts.guidance_scale || 7.0,
          sampler: mapSamplerForBackend(opts.sampler || "euler_a"),
          seed: opts.seed || -1,
        };
      }

      console.log("Sending to SD API:", payload);
      console.log("Using image engine:", imageEngine);
      console.log("API URL:", endpoint);

      const fetchOpts = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      };
      if (options.signal) fetchOpts.signal = options.signal;
      const res = await fetch(endpoint, fetchOpts);

      if (!res.ok) {
        const errorText = await res.text();
        console.error("SD API error:", res.status, errorText);
        throw new Error(`SD API error: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      console.log("Received response from SD API:", data);
      return data;

    } catch (err) {
      console.error("Generate image error:", err);
      setApiError(err.message);
      throw err;
    } finally {
      setIsImageGenerating(false);
    }
  }, [clearError, setApiError, PRIMARY_API_URL, MEMORY_API_URL, settings, setIsImageGenerating]);

  // NEW: Generate Video via NanoGPT
  const generateVideo = useCallback(async (prompt) => {
    setIsImageGenerating(true);
    try {
      console.log("Starting video generation prompt:", prompt);
      const videoModel = settings.nanoGptVideoModel || 'svd';
      const apiKey = settings.nanoGptApiKey;

      if (!apiKey) {
        throw new Error("NanoGPT API Key is missing. Please check Settings > Image Generation.");
      }

      // 1. Start Job
      const startRes = await fetch(`${PRIMARY_API_URL}/sd/nanogpt/video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          model: videoModel,
          api_key: apiKey
        })
      });

      if (!startRes.ok) {
        const errorText = await startRes.text();
        throw new Error(`Failed to start video job: ${errorText}`);
      }

      const startData = await startRes.json();
      const jobId = startData.job_id;
      console.log("Video job started:", jobId);

      // 2. Poll for Status
      let attempts = 0;
      const maxAttempts = 60; // 5 minutes (5s * 60)

      while (attempts < maxAttempts) {
        // Wait 5s
        await new Promise(r => setTimeout(r, 5000));
        attempts++;

        const statusRes = await fetch(`${PRIMARY_API_URL}/sd/nanogpt/video/status/${jobId}?api_key=${apiKey}`);
        if (!statusRes.ok) {
          console.warn("Video status check failed, retrying...");
          continue;
        }

        const statusData = await statusRes.json();
        console.log("Video polling status:", statusData);

        if (statusData.status === 'success') {
          return statusData.video_url; // Local URL returned by backend
        } else if (statusData.status === 'failed') {
          throw new Error(`Video generation failed: ${statusData.error || 'Unknown error'}`);
        }
        // If pending/processing, continue loop
      }

      throw new Error("Video generation timed out.");
    } finally {
      setIsImageGenerating(false);
    }
  }, [PRIMARY_API_URL, settings, setIsImageGenerating]);

  const saveToGallery = useCallback(async (imageUrl, parameters, displayName, categoryId, tags) => {
    const targetApiUrl = PRIMARY_API_URL || getBackendUrl();
    const res = await fetch(`${targetApiUrl}/room-gallery/save-from-generation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_url: imageUrl,
        display_name: displayName || null,
        category_id: categoryId || null,
        tags: tags || [],
        parameters: parameters || null,
      }),
    });
    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(errBody || 'Failed to save to gallery');
    }
    return await res.json();
  }, [PRIMARY_API_URL]);


  // 3. (Unchanged) Test lore detection helper
  const testLoreDetection = async (sampleText) => {
    try {
      if (!activeCharacter) {
        console.warn("🌍 [LORE TEST] No active character to test lore with");
        return;
      }

      const testText = sampleText || "Let me test the lore detection system";
      console.log(`🌍 [LORE TEST] Testing with text: "${testText}" and character: ${activeCharacter.name}`);

      const triggeredLore = await fetchTriggeredLore(testText, activeCharacter);
      console.log("🌍 [LORE TEST] Results:", triggeredLore);

      return triggeredLore;
    } catch (error) {
      console.error("🌍 [LORE TEST] Error:", error);
      return null;
    }
  };


  // You can expose this for console access if needed:
  window.testLoreDetection = testLoreDetection;
  // --- STT Implementation ---
  const startRecording = useCallback(async () => {
    console.log("🎤 Attempting to start recording...");
    setAudioError(null);
    if (isRecording || isTranscribing) {
      console.warn("🎤 Recording or transcription already in progress.");
      return;
    }

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser API not supported. If on mobile LAN, you may need HTTPS or specific flags.");
      }

      wavMicRecorderRef.current?.cancel();
      wavMicRecorderRef.current = createWavMicRecorder();
      await wavMicRecorderRef.current.start();
      setIsRecording(true);
      console.log("🎤 Recording started (16 kHz WAV).");
    } catch (err) {
      console.error("🎤 Error accessing microphone:", err);

      let errorMessage = `Microphone error: ${err.message}`;

      if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        errorMessage = `⚠️ Microphone blocked due to insecure connection (HTTP).\n\nTo fix on Mobile/LAN:\n1. Open chrome://flags (or edge://flags)\n2. Search "Insecure origins treated as secure"\n3. Enable it and add: ${window.location.origin}\n4. Restart browser.`;
        alert(errorMessage);
      }

      setAudioError(errorMessage);
      setIsRecording(false);
    }
  }, [isRecording, isTranscribing]);

  const stopRecording = useCallback(async (onTranscriptReceived) => {
    if (!wavMicRecorderRef.current || !isRecording) {
      console.warn("🎤 Stop recording called but not currently recording.");
      setIsRecording(false);
      setIsTranscribing(false);
      return;
    }

    console.log("🎤 Stopping recording via stopRecording()...");
    setIsRecording(false);
    setIsTranscribing(true);

    try {
      const audioBlob = await wavMicRecorderRef.current.stop();
      wavMicRecorderRef.current = null;
      console.log(`🎤 Final audio blob size: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

      if (audioBlob.size === 0) {
        console.warn("🎤 Audio blob is empty, skipping transcription.");
        setAudioError("Recording resulted in empty audio.");
        return;
      }

      console.log(`🎤 Using STT engine: ${settings.sttEngine}`);
      let sttEngine = settings.sttEngine;
      if (sttEngine === 'nanogpt') {
        sttEngine = `nanogpt-${settings.nanogptSttModel || 'fun-asr-flash-2026-06-15'}`;
      } else if (sttEngine === 'parakeet-cpp') {
        const mId = settings.parakeetCppModel || 'tdt_ctc-110m';
        const mQuant = settings.parakeetCppQuant || 'f16';
        sttEngine = `parakeet-cpp:${mId}:${mQuant}`;
      }
      const transcript = await transcribeAudio(audioBlob, sttEngine);
      console.log("🎤 Transcription successful:", transcript);

      if (typeof onTranscriptReceived === 'function') {
        onTranscriptReceived(transcript);
      } else {
        console.warn("🎤 onTranscriptReceived callback is not a function.");
        alert(`Transcript (callback missing): ${transcript}`);
      }
    } catch (transcriptionError) {
      console.error("🎤 Transcription failed:", transcriptionError);
      setAudioError(`Transcription failed: ${transcriptionError.message}`);
    } finally {
      setIsTranscribing(false);
    }
  }, [settings, isRecording, setIsRecording, setIsTranscribing, setAudioError]);

  const startCallMode = useCallback(async () => {
    console.log("🎯 Starting Call Mode...");
    setIsCallModeActive(true);
  }, []);

  const stopCallMode = useCallback(async () => {
    console.log("🎯 Stopping Call Mode...");
    setIsCallModeActive(false);

    // Stop any active recording
    if (isRecording) {
      await stopRecording(() => { });
    }

    // Stop any active TTS
    stopTTS('stopCallMode');
  }, [isRecording, stopRecording, stopTTS]);


  // --- playTTS (Modified to prevent auto-replay) ---
  const playTTS = useCallback(async (messageId, text, optionsOverrides = null) => {
    console.log(`🗣️ [TTS] Attempting to play message ${messageId}: "${text.substring(0, 40)}..."`);

    // Check if this is the same message that was just played
    if (lastPlayedMessageRef.current === messageId && !optionsOverrides) {
      console.log(`🗣️ [TTS] Skipping message ${messageId} as it was just played.`);
      return;
    }
    // ADD THIS CODE: Check if the message is still streaming
    const message = messages.find(m => m.id === messageId);
    if (message && message.isStreaming && !optionsOverrides) {
      console.log(`🗣️ [TTS] Skipping message ${messageId} as it is still streaming.`);
      return;
    }

    if (isPlayingAudio || audioPlayerRef.current || streamingTtsMessageIdRef.current) {
      stopTTS('playTTS_replace_active');
    }

    const playbackRequestId = ttsPlaybackRequestRef.current + 1;
    ttsPlaybackRequestRef.current = playbackRequestId;
    isTtsInterruptedRef.current = false;

    setIsPlayingAudio(messageId);
    setTtsPlaybackState({ messageId, phase: 'synthesising' });
    setAudioError(null);

    let audioUrl = null;

    try {
      // CRITICAL: Check if TTS has been interrupted before starting playback
      if (isTtsInterruptedRef.current) {
        console.log(`🛑 [TTS] Playback blocked for message ${messageId} - interrupt flag is set`);
        setIsPlayingAudio(null);
        return;
      }

      // 1. Synthesize Speech -> Blob URL
      console.log(`🗣️ [TTS] Calling synthesizeSpeech API for message ${messageId}...`);

      // Resolve options: priority to overrides, then current state settings
      const currentTtsEngine = optionsOverrides?.ttsEngine || settings.ttsEngine || 'kokoro';
      const currentTtsVoice = optionsOverrides?.ttsVoice || settings.ttsVoice || 'af_heart';
      const currentTtsSpeed = normaliseTtsSpeed(optionsOverrides?.ttsSpeed ?? settings.ttsSpeed ?? 1.0);
      const playbackRate = getTtsPlaybackRate(currentTtsEngine, currentTtsSpeed);
      const isChatterboxFamily =
        currentTtsEngine === 'chatterbox' || currentTtsEngine === 'chatterbox_turbo' || currentTtsEngine === 'chatterbox_nano';
      const isVoxCPMFamily = currentTtsEngine === 'voxcpm';
      // Pitch belongs to Kokoro only; stale settings must not alter another engine.
      const rawPitch = optionsOverrides?.ttsPitch ?? settings.ttsPitch ?? 0;
      const currentTtsPitch = currentTtsEngine === 'kokoro' ? Number(rawPitch) || 0 : 0;

      // Build options object for TTS engines
      const ttsOptions = {
        voice: currentTtsVoice,
        engine: currentTtsEngine,
        speed: currentTtsSpeed,
        save_full_response_audio: settings.ttsSaveFullResponseAudio === true,
        message_id: messageId,
        conversation_id: activeConversation,
      };
      const chunkSec = Number(settings.ttsSaveFullResponseChunkSeconds) || 0;
      if (settings.ttsSaveFullResponseAudio === true && chunkSec > 0) {
        ttsOptions.save_full_response_max_chunk_seconds = chunkSec;
      }

      if (isChatterboxFamily) {
        ttsOptions.exaggeration = optionsOverrides?.ttsExaggeration ?? settings.ttsExaggeration ?? 0.5;
        ttsOptions.cfg = optionsOverrides?.ttsCfg ?? settings.ttsCfg ?? 0.5;
        if (currentTtsVoice && currentTtsVoice !== 'default') {
          ttsOptions.audio_prompt_path = currentTtsVoice;
        }
        console.log(`🔊 [TTS] Chatterbox params - exaggeration: ${ttsOptions.exaggeration}, cfg: ${ttsOptions.cfg}`);
      }

      if (isVoxCPMFamily) {
        ttsOptions.voxcpm_cfg_value = optionsOverrides?.voxcpmCfgValue ?? settings.voxcpmCfgValue ?? 2.0;
        ttsOptions.voxcpm_inference_timesteps = optionsOverrides?.voxcpmInferenceTimesteps ?? settings.voxcpmInferenceTimesteps ?? 8;
        ttsOptions.voxcpm_normalize = optionsOverrides?.voxcpmNormalize ?? settings.voxcpmNormalize ?? false;
        ttsOptions.voxcpm_denoise = optionsOverrides?.voxcpmDenoise ?? settings.voxcpmDenoise ?? false;
        ttsOptions.voxcpm_retry_badcase = optionsOverrides?.voxcpmRetryBadcase ?? settings.voxcpmRetryBadcase ?? false;
        ttsOptions.voxcpm_voice_design = optionsOverrides?.voxcpmVoiceDesign ?? settings.voxcpmVoiceDesign ?? '';
        if (currentTtsVoice && currentTtsVoice !== 'default') {
          ttsOptions.audio_prompt_path = currentTtsVoice;
        }
        console.log(`🔊 [TTS] VoxCPM2 params - cfg: ${ttsOptions.voxcpm_cfg_value}, timesteps: ${ttsOptions.voxcpm_inference_timesteps}, voice_design: "${ttsOptions.voxcpm_voice_design}"`);
      }

      console.log(`🔊 [TTS] Using engine: ${ttsOptions.engine} with options:`, ttsOptions);

      if (settings.ttsSaveFullResponseAudio) {
        setTtsFullResponseSaveStatus({
          state: 'saving',
          message:
            chunkSec > 0
              ? `Saving TTS in ~${Math.floor(chunkSec / 60)}m${chunkSec % 60}s segments on backend...`
              : 'Saving full-response audio on backend...',
          path: null,
          filename: null,
          chunkCount: null,
          updatedAt: Date.now(),
        });
      }

      // Pass options object to synthesizeSpeech
      audioUrl = await synthesizeSpeech(text, ttsOptions);
      if (settings.ttsSaveFullResponseAudio) {
        const meta = getLastTtsSynthesisMeta();
        if (meta?.saveStatus === 'saved') {
          const n = meta.saveChunkCount || 1;
          setTtsFullResponseSaveStatus({
            state: 'saved',
            message:
              n > 1
                ? `Saved ${n} segment file(s) (each up to ~${chunkSec}s).`
                : 'Saved full-response audio.',
            path: meta.savePath,
            filename: meta.saveFilename || null,
            chunkCount: n,
            updatedAt: Date.now(),
          });
        } else if (meta?.saveStatus === 'failed') {
          setTtsFullResponseSaveStatus({
            state: 'failed',
            message: meta.saveError || 'Backend save failed.',
            path: null,
            filename: null,
            chunkCount: null,
            updatedAt: Date.now(),
          });
        } else {
          setTtsFullResponseSaveStatus({
            state: 'unknown',
            message: 'Save status unavailable.',
            path: null,
            filename: null,
            chunkCount: null,
            updatedAt: Date.now(),
          });
        }
      }

      // CHECK AGAIN after await - stop may have been pressed during synthesis
      if (isTtsInterruptedRef.current || ttsPlaybackRequestRef.current !== playbackRequestId) {
        console.log(`🛑 [TTS] Playback blocked for message ${messageId} - interrupted during synthesis`);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        return;
      }

      // UNCHANGED: Everything after this stays exactly the same
      if (!audioUrl) throw new Error("SynthesizeSpeech returned an invalid URL.");
      console.log(`🗣️ [TTS] Received audio URL for message ${messageId}: ${audioUrl.substring(0, 50)}...`);

      // 2. Zero semitone offset: HTML audio + preservesPitch so speech speed does not chipmunk the voice.
      if (currentTtsPitch === 0) {
        const audio = new Audio(audioUrl);
        try {
          audio.preservesPitch = true;
        } catch (e) { /* noop */ }
        try {
          if ('webkitPreservesPitch' in audio) audio.webkitPreservesPitch = true;
        } catch (e) { /* noop */ }
        audio.playbackRate = playbackRate;
        audioPlayerRef.current = audio;
        activeAudioPlayersRef.current.add(audio);

        const handleEnd = () => {
          console.log(`🗣️ [TTS] Playback ended for message ${messageId}`);
          activeAudioPlayersRef.current.delete(audio);
          if (ttsPlaybackRequestRef.current === playbackRequestId) {
            setIsPlayingAudio(null);
            setTtsPlaybackState({ messageId: null, phase: 'idle' });
            lastPlayedMessageRef.current = messageId;
          }
          URL.revokeObjectURL(audioUrl);
          if (audioPlayerRef.current === audio) audioPlayerRef.current = null;
          audio.removeEventListener('ended', handleEnd);
          audio.removeEventListener('error', handleError);
        };
        const handleError = () => {
          activeAudioPlayersRef.current.delete(audio);
          if (ttsPlaybackRequestRef.current === playbackRequestId) {
            setAudioError("Failed to play synthesized audio.");
            setIsPlayingAudio(null);
            setTtsPlaybackState({ messageId: null, phase: 'idle' });
            lastPlayedMessageRef.current = null;
          }
          URL.revokeObjectURL(audioUrl);
          if (audioPlayerRef.current === audio) audioPlayerRef.current = null;
          audio.removeEventListener('ended', handleEnd);
          audio.removeEventListener('error', handleError);
        };

        audio.addEventListener('ended', handleEnd);
        audio.addEventListener('error', handleError);
        setTtsPlaybackState({ messageId, phase: 'playing' });
        await audio.play();

      } else {
        // --- Web Audio API path (pitch shift) ---
        console.log(`🗣️ [TTS] Using Web Audio API for pitch-shifted playback...`);
        const resp = await fetch(audioUrl);
        const arrayBuf = await resp.arrayBuffer();
        const ctx = new AudioContext();
        const buf = await ctx.decodeAudioData(arrayBuf);

        const source = ctx.createBufferSource();
        source.buffer = buf;
        source.playbackRate.value = playbackRate;
        source.detune.value = currentTtsPitch * 100; // Semitones -> cents

        source.connect(ctx.destination);

        const playerHandlePitch = { ctx, source };
        audioPlayerRef.current = { ...playerHandlePitch, audioUrl };
        activeAudioPlayersRef.current.add(playerHandlePitch);

        source.onended = () => {
          console.log(`🗣️ [TTS] Web Audio playback ended for message ${messageId}`);
          activeAudioPlayersRef.current.delete(playerHandlePitch);
          if (ttsPlaybackRequestRef.current === playbackRequestId) {
            setIsPlayingAudio(null);
            setTtsPlaybackState({ messageId: null, phase: 'idle' });
            lastPlayedMessageRef.current = messageId;
          }
          try { ctx.close(); } catch (e) { }
          URL.revokeObjectURL(audioUrl);
          if (audioPlayerRef.current?.source === source) audioPlayerRef.current = null;
        };

        setTtsPlaybackState({ messageId, phase: 'playing' });
        source.start();
      }

    } catch (error) {
      if (ttsPlaybackRequestRef.current !== playbackRequestId) {
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        return;
      }
      console.error("🗣️ [TTS] Error:", error);
      setAudioError(error?.message || "Unknown TTS error");
      if (settings.ttsSaveFullResponseAudio) {
        setTtsFullResponseSaveStatus({
          state: 'failed',
          message: error?.message || 'Synthesis or save failed.',
          path: null,
          filename: null,
          chunkCount: null,
          updatedAt: Date.now(),
        });
      }
      setIsPlayingAudio(null);
      setTtsPlaybackState({ messageId: null, phase: 'idle' });
    }

  }, [settings, messages, stopTTS, isPlayingAudio]);

  // Memory agent integration functions
  // Memory Agent Integration
  // ----------------------------------
  const fetchMemoriesFromAgent = useCallback(
    async (prompt) => {
      try {
        if (settings.directProfileInjection) return [];
        if (primaryIsAPI) return [];
        const userId = resolveAgenticUserId();
        if (!userId) return [];

        const memTimeout = primaryIsAPI ? MEMORY_FETCH_TIMEOUT_API_MS : MEMORY_FETCH_TIMEOUT_MS;
        const res = await fetchWithTimeout(
          `${MEMORY_API_URL}/memory/relevant`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt,
              userProfile,
              systemTime: new Date().toISOString(),
              requestType: 'memoryRetrieval'
            })
          },
          memTimeout
        );

        if (!res.ok) throw new Error(`Status ${res.status}`);
        const data = await res.json();
        return data.memories || [];
      } catch (err) {
        const memTimeout = primaryIsAPI ? MEMORY_FETCH_TIMEOUT_API_MS : MEMORY_FETCH_TIMEOUT_MS;
        console.warn('🧠 Memory fetch error:', formatFetchError(err, { timeoutMs: memTimeout }));
        return [];
      }
    },
    [MEMORY_API_URL, userProfile, memoryContext, settings.directProfileInjection, primaryIsAPI]
  );

  const observeConversationWithAgent = useCallback(
    async (prompt, response) => {
      if (settings.directProfileInjection) return;
      if (!autoMemoryEnabled) return;

      const userId = userProfile?.id || memoryContext?.activeProfileId;
      if (!userId) return;

      try {
        await fetch(`${MEMORY_API_URL}/memory/observe`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_message: prompt,
            ai_response: response,
            user_name: userId,
            userProfile: userProfile,
            systemTime: new Date().toISOString()
          })
        });
      } catch (err) {
        console.error('🧠 Observation failed:', err);
      }
    },
    [autoMemoryEnabled, userProfile, memoryContext, MEMORY_API_URL, settings.directProfileInjection]
  );

  const processAgenticMemoryIfEnabled = useCallback(
    async (userId, character, userMessage, aiResponse, apiOptions = null) => {
      // Prefer the passed-in character, but fall back to the active chat character so
      // agentic memory keeps working even when speaker resolution returns null.
      let effectiveCharacter = character || activeCharacterRef.current || activeCharacter;
      const charName = effectiveCharacter?.name || 'Character';
      const charId = effectiveCharacter?.id || null;
      // Book automation: skip POST /memory/agentic/process to avoid doubling LLM work per segment.
      // GET /memory/agentic injection in getGenerationSystemPrompt still runs during book flows.
      if (bookModePackingOverridesRef.current) {
        return;
      }
      const effectiveUserId = userId || resolveAgenticUserId();
      if (!effectiveUserId || effectiveUserId === 'anonymous') {
        console.warn('🧠 Agentic memory: ON but SKIPPED — no user id (select a Memory profile in Settings).');
        return;
      }
      if (!charId) {
        console.warn('🧠 Agentic memory: ON but SKIPPED — no character id even after fallback. Character:', charName);
        return;
      }
      const url = `${MEMORY_API_URL}/memory/agentic/process`;
      console.warn(
        '🧠 Agentic memory: FETCHING',
        url,
        '| user_id=',
        effectiveUserId?.slice?.(0, 12),
        '| character_id=',
        charId?.slice?.(0, 12),
      );
      const body = {
        user_id: effectiveUserId,
        character_id: charId,
        character_name: charName,
        character_profile: {
          description: effectiveCharacter?.description || '',
          scenario: effectiveCharacter?.scenario || '',
          model_instructions: effectiveCharacter?.model_instructions || '',
          ethics_justification: effectiveCharacter?.ethics_justification || ''
        },
        user_message: userMessage,
        ai_response: aiResponse
      };
      if (apiOptions?.useApi && apiOptions?.apiBaseUrl && apiOptions?.modelName) {
        body.use_api = true;
        body.api_base_url = apiOptions.apiBaseUrl;
        body.model_name = apiOptions.modelName;
      }
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const errText = await res.text().catch(() => res.statusText);
          console.error('🧠 Agentic memory: backend failed', res.status, errText);
          setLastAgenticRunStatus('error');
          return;
        }
        const data = await res.json().catch(() => ({}));
        const added = data.added ?? 0;
        const total = data.total;
        setLastAgenticRunStatus('ok'); // UI can show backend actually ran
        if (added > 0) {
          console.log(
            `🧠 Agentic memory: saved ${added} insight(s) for ${charName}` +
              (total != null ? ` (total ${total} on server)` : ''),
          );
          setLastAgenticMemoryFeedback({ added, characterName: charName });
          setTimeout(() => setLastAgenticMemoryFeedback(null), 5000);
        } else {
          console.log(
            `🧠 Agentic memory: backend ran for ${charName} — no new insights` +
              (total != null ? ` (total ${total} on server)` : ''),
          );
        }
      } catch (err) {
        console.error('🧠 Agentic memory: backend error', err);
        setLastAgenticRunStatus('error');
      }
    },
    [MEMORY_API_URL, resolveAgenticUserId]
  );

  const processAlignmentDetectionIfEnabled = useCallback(
    async (userId, character, userMessage, aiResponse, apiOptions = null) => {
      const conv = conversationsRef.current?.find(c => c.id === activeConversationRef.current);
      if (!conv?.alignmentDetectionEnabled) {
        return;
      }
      let effectiveCharacter = character || activeCharacterRef.current || activeCharacter;
      const charName = effectiveCharacter?.name || 'Character';
      const charId = effectiveCharacter?.id || null;
      const effectiveUserId = userId || resolveAgenticUserId();
      if (!effectiveUserId || effectiveUserId === 'anonymous' || !charId) {
        return;
      }
      const url = `${MEMORY_API_URL}/memory/alignment/process`;
      const body = {
        user_id: effectiveUserId,
        character_id: charId,
        character_name: charName,
        character_profile: {
          description: effectiveCharacter?.description || '',
          scenario: effectiveCharacter?.scenario || '',
          model_instructions: effectiveCharacter?.model_instructions || '',
        },
        user_message: userMessage,
        ai_response: aiResponse,
      };
      if (apiOptions?.useApi && apiOptions?.apiBaseUrl && apiOptions?.modelName) {
        body.use_api = true;
        body.api_base_url = apiOptions.apiBaseUrl;
        body.model_name = apiOptions.modelName;
      }
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          console.warn('🔍 Alignment detection: backend failed', res.status);
          return;
        }
        const data = await res.json().catch(() => ({}));
        const added = data.added ?? 0;
        const total = data.total ?? 0;
        const findings = data.findings ?? [];
        const preFilter = data.pre_filter ?? {};
        const highestSeverity = findings.reduce((max, f) => {
          const order = { high: 3, medium: 2, low: 1 };
          return (order[f.severity] || 0) > (order[max] || 0) ? f.severity : max;
        }, null);
        const frameFidelity = total > 0
          ? Math.max(0, Math.round((1 - (findings.filter(f => f.severity === 'high').length / total)) * 100))
          : null;
        setAlignmentData({
          count: total,
          added,
          highestSeverity,
          findings,
          preFilter,
          frameFidelity,
          characterName: charName,
          timestamp: Date.now(),
        });
      } catch (err) {
        console.warn('🔍 Alignment detection: backend error', err);
      }
    },
    [MEMORY_API_URL, resolveAgenticUserId]
  );

  const retryAgenticMemoryForLastTurn = useCallback(async () => {
    const activeId = activeConversationRef.current;
    if (!activeId) {
      console.warn('🧠 [Agentic] Retry requested but no active conversation.');
      return;
    }
    const conv = (conversationsRef.current || []).find(c => c.id === activeId);
    const convoMessages = conv?.messages && conv.messages.length ? conv.messages : messages;
    if (!convoMessages || convoMessages.length === 0) {
      console.warn('🧠 [Agentic] Retry requested but no messages in conversation.');
      return;
    }
    const userId = resolveAgenticUserId();
    const lastBot = [...convoMessages].reverse().find(m => m.role === 'bot' && typeof m.content === 'string' && m.content.trim());
    const lastUser = [...convoMessages].reverse().find(m => m.role === 'user' && typeof m.content === 'string' && m.content.trim());
    if (!lastBot || !lastUser) {
      console.warn('🧠 [Agentic] Retry skipped — could not find last user/bot pair.');
      return;
    }
    const apiOpts = primaryIsAPI ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: primaryModel } : null;
    await processAgenticMemoryIfEnabled(
      userId,
      activeCharacterRef.current || activeCharacter,
      lastUser.content,
      lastBot.content,
      apiOpts
    );
  }, [
    messages,
    activeCharacter,
    primaryIsAPI,
    primaryModel,
    PRIMARY_API_URL,
    processAgenticMemoryIfEnabled,
    resolveAgenticUserId,
  ]);

  // ----------------------------------
  // Model Management
  // ----------------------------------
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetchWithTimeout(`${PRIMARY_API_URL}/models`, {}, 8000);
      if (!res.ok) throw new Error(res.status);
      const { available_models } = await res.json();
      setAvailableModels(available_models || []);
    } catch (err) {
      console.error('Error fetching models:', err);
    }
  }, [PRIMARY_API_URL]);

  const fetchLoadedModels = useCallback(async () => {
    let primaryModels = [];
    try {
      const res = await fetchWithTimeout(`${PRIMARY_API_URL}/models/loaded`, {}, 8000);
      if (res.ok) {
        const { loaded_models } = await res.json();
        primaryModels = loaded_models || [];
        const gpu0 = primaryModels.find(m => m.gpu_id === 0);
        const savedApiModel = readLastPrimaryApiModel();
        const autoRouterEnabled = settingsRef.current?.apiEndpointRoundRobinEnabled === true;
        if (gpu0 && !savedApiModel && !autoRouterEnabled) {
          setPrimaryModel(gpu0.name);
          setActiveModel(gpu0.name);
        }
      }
    } catch {
      // ignore
    }

    let secondaryModels = [];
    if (SECONDARY_API_URL !== PRIMARY_API_URL) {
      try {
        const res = await fetchWithTimeout(`${SECONDARY_API_URL}/models/loaded`, {}, 8000);
        if (res.ok) {
          const { loaded_models } = await res.json();
          secondaryModels = loaded_models || [];
          const gpu1 = secondaryModels.find(m => m.gpu_id === 1);
          if (gpu1) setSecondaryModel(gpu1.name);
        }
      } catch {
        console.warn('Secondary API unavailable');
      }
    }

    setLoadedModels([...primaryModels, ...secondaryModels]);
  }, [PRIMARY_API_URL, SECONDARY_API_URL]);

  // After port config + single-GPU sync, primary and secondary URLs can become the same; avoid stale :8001 probes.
  useEffect(() => {
    if (!portsReady) return;
    fetchLoadedModels();
    void refreshNanoGptModelsCache({ forceRefresh: false });
  }, [portsReady, isSingleGpuMode, PRIMARY_API_URL, SECONDARY_API_URL, fetchLoadedModels]);

  /** Restore last selected API model after settings hydrate (never block send on mismatch). */
  useEffect(() => {
    if (!storageHydrated || !portsReady || apiModelRestoredRef.current) return;
    apiModelRestoredRef.current = true;
    const autoRouterEnabled = settings.apiEndpointRoundRobinEnabled === true;
    const rotatePool = (settings.customApiEndpoints || []).filter(
      (e) => e?.enabled !== false && e?.rotate_enabled !== false,
    );
    if (autoRouterEnabled && rotatePool.length > 0) {
      setPrimaryIsAPI(true);
      setPrimaryModel(null);
      setActiveModel(null);
      if (!autoRouterBootLogRef.current) {
        autoRouterBootLogRef.current = true;
        console.info('auto_router_boot_precedence_applied', {
          auto_router_enabled: true,
          storage_hydrated: storageHydrated,
          ports_ready: portsReady,
          cleared_pinned_primary_model: true,
          rotate_pool_count: rotatePool.length,
        });
      }
      return;
    }

    const saved = readLastPrimaryApiModel();
    const endpoints = (settings.customApiEndpoints || []).filter((e) => e?.enabled !== false);
    let modelId = null;

    if (saved) {
      const matched = endpoints.find(
        (e) => e.id === saved || e.name === saved || e.model === saved,
      );
      if (matched) modelId = matched.id;
      else if (saved.startsWith('endpoint-')) modelId = saved;
      else if (endpoints.length === 0) modelId = saved;
      else if (findNanoGptModel(saved, readNanoGptModelsCache().models)) modelId = saved;
      else modelId = endpoints[0]?.id || saved;
    } else if (endpoints.length > 0) {
      modelId = endpoints[0].id;
    }

    if (modelId) {
      setPrimaryIsAPI(true);
      setPrimaryModel(modelId);
      setActiveModel(modelId);
    }
  }, [storageHydrated, portsReady, settings.customApiEndpoints, settings.apiEndpointRoundRobinEnabled]);

  useEffect(() => {
    if (primaryIsAPI && primaryModel) {
      saveLastPrimaryApiModel(primaryModel);
    }
  }, [primaryIsAPI, primaryModel, settings.apiEndpointRoundRobinEnabled]);

  useEffect(() => {
    if (!crossWindowSyncReadyRef.current) {
      crossWindowSyncReadyRef.current = true;
      return;
    }
    broadcastPrimaryModelState({
      primaryModel,
      primaryIsAPI,
      autoRouterEnabled: settings.apiEndpointRoundRobinEnabled,
      autoRouterActive:
        settings.apiEndpointRoundRobinEnabled === true
        && getRotationPool(settings).length > 0,
    });
  }, [primaryModel, primaryIsAPI, settings.apiEndpointRoundRobinEnabled, settings.customApiEndpoints]);

  const loadModel = useCallback(
    async (name, gpu = 0, contextLength = settings.contextLength) => {
      setIsModelLoading(true);
      try {
        const api = gpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
        const res = await fetch(`${api}/models/load/${name}?gpu_id=${gpu}&context_length=${contextLength}`, { method: 'POST' });
        if (!res.ok) throw new Error(name);
        await fetchLoadedModels();
        if (gpu === 0) { setPrimaryModel(name); setActiveModel(name); }
        else if (gpu === 1) setSecondaryModel(name);
        return true;
      } catch (err) {
        console.error('Load model error:', err);
        setApiError(err.message);
        return false;
      } finally {
        setIsModelLoading(false);
      }
    },
    [fetchLoadedModels, PRIMARY_API_URL, SECONDARY_API_URL, settings.contextLength]
  );
  const loadSttEngine = useCallback(async (engine = 'whisper', gpuId = 1) => {
    try {
      const response = await fetch(`${SECONDARY_API_URL}/stt/load-engine`, { // <-- CORRECTED
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine, gpu_id: gpuId }),
      });
      if (!response.ok) throw new Error('Failed to load STT engine');
      const data = await response.json();
      console.log('STT Engine loaded:', data.message);
      alert('STT Engine loaded successfully on GPU ' + gpuId);
    } catch (error) {
      console.error('Error loading STT engine:', error);
      alert('Failed to load STT engine.');
    }
  }, [SECONDARY_API_URL]);

  const loadTtsEngine = useCallback(async (engine = 'kokoro', gpuId = 1) => {
    try {
      const response = await fetch(`${SECONDARY_API_URL}/tts/load-engine`, { // <-- CORRECTED
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine, gpu_id: gpuId }),
      });
      if (!response.ok) throw new Error('Failed to load TTS engine');
      const data = await response.json();
      console.log('TTS Engine loaded:', data.message);
      alert('TTS Engine loaded successfully on GPU ' + gpuId);
    } catch (error) {
      console.error('Error loading TTS engine:', error);
      alert('Failed to load TTS engine.');
    }
  }, [SECONDARY_API_URL]); // <-- CORRECTED
  const unloadModel = useCallback(
    async (name) => {
      try {
        const info = loadedModels.find(m => m.name === name);
        if (!info) return false;
        const api = info.gpu_id === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
        const res = await fetch(`${api}/models/unload/${name}`, { method: 'POST' });
        if (!res.ok) throw new Error(name);
        await fetchLoadedModels();
        if (primaryModel === name) { setPrimaryModel(null); setActiveModel(null); }
        if (secondaryModel === name) setSecondaryModel(null);
        return true;
      } catch (err) {
        console.error('Unload model error:', err);
        setApiError(err.message);
        return false;
      }
    },
    [fetchLoadedModels, loadedModels, primaryModel, secondaryModel, PRIMARY_API_URL, SECONDARY_API_URL]
  );

  // ----------------------------------
  // Conversation Management
  // ----------------------------------
  const createNewConversation = useCallback((opts = {}) => {
    console.log('🔍 [DEBUG] Creating new conversation');
    const id = generateUniqueId();
    const systemPersonaOn = isSystemPersonaModeActive(settingsRef.current);
    const systemPersonaChar = systemPersonaOn
      ? resolveSystemPersonaCharacter(characters, settingsRef.current)
      : null;
    const charName = primaryCharacter?.name || systemPersonaChar?.name || 'Character';
    const userName = settings.multiRoleMode && userCharacter?.name
      ? userCharacter.name
      : (userProfile?.name || userProfile?.username || 'User');
    const forceEmpty = opts?.forceEmpty === true;

    const useSystemIntro =
      !forceEmpty
      && systemPersonaOn
      && !!systemPersonaChar?.id
      && !primaryCharacter?.id;
    const useCharacterIntro =
      !forceEmpty
      && settingsRef.current?.characterIntroEnabled === true
      && !!primaryCharacter?.id;

    const greetingOptions = !forceEmpty && !useCharacterIntro && !useSystemIntro
      ? buildCharacterGreetingOptions(primaryCharacter, userName)
      : [];
    const firstMessage = greetingOptions[0] || null;
    const characterGreeting = createCharacterGreetingState(greetingOptions);

    const defaultUserCharacter = userCharacter || null;

    const initial = firstMessage
      ? [{
        id: generateUniqueId(),
        role: 'bot',
        content: firstMessage,
        modelId: 'primary',
        characterName: charName,
        characterId: primaryCharacter?.id,
        avatar: getActiveCharacterAvatar(primaryCharacter),
        ...(characterGreeting ? { characterGreeting } : {}),
      }]
      : [];

    const defaultActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    const defaultActiveWeights = buildDefaultCharacterWeights(characters);
    const conv = {
      id,
      name: 'New Chat',
      messages: initial,
      characterIds: {
        primary: primaryCharacter?.id,
        secondary: secondaryCharacter?.id,
        user: defaultUserCharacter?.id || null
      },
      systemPersona: systemPersonaOn,
      systemPersonaCharacterId: systemPersonaChar?.id || settingsRef.current?.systemPersonaCharacterId || null,
      activeCharacterIds: defaultActiveIds,
      activeCharacterWeights: defaultActiveWeights,
      multiRoleContext: '',
      created: new Date().toISOString(),
      requiresTitle: true,
      agenticMemoryEnabled: true,
      alignmentDetectionEnabled: false,
      rollingMemoryPack: '',
      rollingMemoryFoldCount: 0,
      introPending: forceEmpty ? false : (useSystemIntro || useCharacterIntro),
      characterIntro: null,
      chatTemplateId: MODEL_DEFAULT_CHAT_TEMPLATE_ID,
    };
    console.log('🔍 [DEBUG] New conversation has requiresTitle flag:', conv.requiresTitle);
    const next = [...conversationsRef.current, conv];
    conversationsRef.current = next;
    activeConversationRef.current = id;
    conversationSelectionRequestRef.current += 1;
    conversationSwitchInProgressRef.current = false;
    startupConversationRestoreRef.current.attempted = true;
    setConversations(next);
    if (initial.length > 0) {
      void saveConversationCatalog(next, id);
      void saveActiveConversationMessages(id, initial);
    }
    setActiveConversation(id);
    setMessages(initial);
    setDualModeEnabled(false);
    setActiveCharacter(primaryCharacter || null);
    setUserCharacterId(defaultUserCharacter?.id || null);
    setActiveCharacterIds(defaultActiveIds);
    setActiveCharacterWeights(defaultActiveWeights);
    setMultiRoleContext('');
    return conv;
  }, [primaryCharacter, secondaryCharacter, userCharacter, characters, settings.multiRoleMode, userProfile, setConversations, setActiveConversation, setMessages, setDualModeEnabled, setActiveCharacter, setUserCharacterId, setActiveCharacterIds, setActiveCharacterWeights]);

  /**
   * Persist an async message mutation against the conversation that started
   * the work, never whichever tab happens to be visible when it finishes.
   */
  const mutateConversationMessages = useCallback((conversationId, mutation) => {
    const id = typeof conversationId === 'string' ? conversationId.trim() : '';
    if (!id || typeof mutation !== 'function') return Promise.resolve(false);
    if (tombstonedConversationIdsRef.current.has(id)) return Promise.resolve(false);
    if (!(conversationsRef.current || []).some((conversation) => conversation.id === id)) {
      return Promise.resolve(false);
    }

    const activeAndSettled =
      activeConversationRef.current === id
      && !conversationSwitchInProgressRef.current;
    const baseMessages = activeAndSettled ? (messagesRef.current || []) : null;
    let optimisticMessages = null;

    // Apply before awaiting IndexedDB so an immediate tab switch flushes the
    // media result with the rest of its owning conversation.
    if (baseMessages) {
      optimisticMessages = mutation(baseMessages);
      if (!Array.isArray(optimisticMessages)) {
        return Promise.reject(new Error('Conversation message mutation must return an array'));
      }
      messagesRef.current = optimisticMessages;
      setMessages(optimisticMessages);
    }

    const previous = conversationMessageMutationChainsRef.current.get(id) || Promise.resolve();
    const run = async () => {
      if (
        tombstonedConversationIdsRef.current.has(id)
        || !(conversationsRef.current || []).some((conversation) => conversation.id === id)
      ) {
        return false;
      }

      const conv = (conversationsRef.current || []).find((conversation) => conversation.id === id);
      const { messages: _omit, ...catalogMeta } = conv || { id, name: 'Chat' };
      if (activeConversationRef.current === id) setConversationSaveStatus('saving');

      const persistedMessages = await mutateStoredConversationMessages(
        id,
        mutation,
        catalogMeta,
        { baseMessages },
      );

      if (
        !persistedMessages
        || tombstonedConversationIdsRef.current.has(id)
        || !(conversationsRef.current || []).some((conversation) => conversation.id === id)
      ) {
        return false;
      }

      const nextVersion = (conversationMessageVersionRef.current.get(id) || 0) + 1;
      conversationMessageVersionRef.current.set(id, nextVersion);

      let visibleMessages = null;
      if (
        activeConversationRef.current === id
        && !conversationSwitchInProgressRef.current
      ) {
        // Reapply idempotently to the latest state. This preserves messages
        // added after the media request began and repairs a tab load that
        // briefly read the shard before this mutation was committed.
        visibleMessages = mutation(messagesRef.current || []);
        messagesRef.current = visibleMessages;
        setMessages(visibleMessages);
        setConversationSaveStatus('saved');
      }

      const cachedMessages = visibleMessages || persistedMessages;
      setConversations((current) => {
        if (
          tombstonedConversationIdsRef.current.has(id)
          || !current.some((conversation) => conversation.id === id)
        ) {
          return current;
        }
        const next = current.map((conversation) => (
          conversation.id === id
            ? {
              ...conversation,
              messages: cachedMessages,
              messageCount: cachedMessages.length,
            }
            : conversation
        ));
        conversationsRef.current = next;
        return next;
      });

      return true;
    };

    const job = previous.catch(() => {}).then(run);
    conversationMessageMutationChainsRef.current.set(id, job);
    void job.finally(() => {
      if (conversationMessageMutationChainsRef.current.get(id) === job) {
        conversationMessageMutationChainsRef.current.delete(id);
      }
    }).catch(() => {});
    return job.catch((error) => {
      if (activeConversationRef.current === id) setConversationSaveStatus('error');
      throw error;
    });
  }, [setConversations, setMessages]);

  const appendMessagesToConversation = useCallback((conversationId, additions) => {
    const messagesToAppend = Array.isArray(additions) ? additions : [additions];
    return mutateConversationMessages(
      conversationId,
      (current) => appendUniqueConversationMessages(current, messagesToAppend),
    );
  }, [mutateConversationMessages]);

  const updateMessageInConversation = useCallback((conversationId, messageId, update) => (
    mutateConversationMessages(
      conversationId,
      (current) => updateConversationMessageById(current, messageId, update),
    )
  ), [mutateConversationMessages]);

  /**
   * Start a normal chat conversation for a specific character.
   * Used by Mirror date-booking and promotion flows.
   * Saves the prior conversation, sets the character active, creates the new
   * conversation, seeds the first bot message, and switches to the chat tab.
   */
  const startCharacterConversation = useCallback((character, opts = {}) => {
    const id = generateUniqueId();
    const charName = character.name || 'Character';
    const userName = (settings.multiRoleMode && userCharacter?.name)
      ? userCharacter.name
      : (userProfile?.name || userProfile?.username || 'User');
    const greetingOptions = opts.firstMessage
      ? [String(opts.firstMessage)
        .replace(/{{char}}/gi, charName)
        .replace(/{{user}}/gi, userName)
        .trim()]
      : buildCharacterGreetingOptions(character, userName);
    const resolvedFirstMsg = greetingOptions[0] || '';
    const characterGreeting = createCharacterGreetingState(greetingOptions);

    const initial = resolvedFirstMsg ? [{
      id: generateUniqueId(),
      role: 'bot',
      content: resolvedFirstMsg,
      modelId: 'primary',
      characterName: charName,
      characterId: character.id,
      avatar: getActiveCharacterAvatar(character),
      ...(characterGreeting ? { characterGreeting } : {}),
    }] : [];

    const availableActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    const availableActiveIdSet = new Set(availableActiveIds);
    const requestedActiveIds = Array.isArray(opts.activeCharacterIds)
      ? [...new Set(opts.activeCharacterIds)]
        .map((candidateId) => String(candidateId || '').trim())
        .filter((candidateId) => candidateId && availableActiveIdSet.has(candidateId))
      : [];
    const defaultActiveIds = requestedActiveIds.length > 0
      ? requestedActiveIds
      : availableActiveIds;
    const allDefaultWeights = buildDefaultCharacterWeights(characters);
    const requestedWeights =
      opts.activeCharacterWeights && typeof opts.activeCharacterWeights === 'object'
        ? opts.activeCharacterWeights
        : {};
    const defaultActiveWeights = Object.fromEntries(
      defaultActiveIds.map((characterId) => [
        characterId,
        Number.isFinite(Number(requestedWeights[characterId]))
          ? Number(requestedWeights[characterId])
          : (allDefaultWeights[characterId] ?? 50),
      ])
    );
    const nextMultiRoleContext = typeof opts.multiRoleContext === 'string'
      ? opts.multiRoleContext.trim()
      : '';

    // Save current conversation before switching
    const prevId = activeConversationRef.current;
    const prevMsgs = messagesRef.current;
    if (prevId && prevMsgs?.length) {
      void saveActiveConversationMessages(prevId, prevMsgs);
    }

    const conv = {
      id,
      name: opts.conversationName || `Chat with ${charName}`,
      messages: initial,
      characterIds: {
        primary: character.id,
        secondary: null,
        user: userCharacter?.id || null,
      },
      characterSnapshot: {
        id: character.id,
        name: character.name,
        description: character.description,
        personality: character.personality,
        avatar: character.avatar,
      },
      systemPersona: false,
      systemPersonaCharacterId: null,
      activeCharacterIds: defaultActiveIds,
      activeCharacterWeights: defaultActiveWeights,
      multiRoleContext: nextMultiRoleContext,
      created: new Date().toISOString(),
      requiresTitle: false,
      agenticMemoryEnabled: true,
      alignmentDetectionEnabled: false,
      rollingMemoryPack: '',
      rollingMemoryFoldCount: 0,
      introPending: false,
      characterIntro: null,
      chatTemplateId: MODEL_DEFAULT_CHAT_TEMPLATE_ID,
      mirrorContinuity: opts.mirrorContinuity || null,
      mirrorDateType: opts.dateType || null,
    };

    const next = [...(conversationsRef.current || []), conv];
    conversationsRef.current = next;
    activeConversationRef.current = id;
    conversationSelectionRequestRef.current += 1;
    conversationSwitchInProgressRef.current = false;
    startupConversationRestoreRef.current.attempted = true;

    setConversations(next);
    setMessages(initial);
    setActiveConversation(id);
    setActiveCharacter(character);
    setPrimaryCharacter(character);
    setSecondaryCharacter(null);
    setUserCharacterId(userCharacter?.id || null);
    setActiveCharacterIds(defaultActiveIds);
    setActiveCharacterWeights(defaultActiveWeights);
    setMultiRoleContext(nextMultiRoleContext);

    if (initial.length > 0) {
      void saveConversationCatalog(next, id);
      void saveActiveConversationMessages(id, initial);
    }

    setActiveTab('chat');

    return conv;
  }, [
    characters, userCharacter, userProfile, generateUniqueId,
    setConversations, setMessages, setActiveConversation,
    setActiveCharacter, setPrimaryCharacter, setSecondaryCharacter,
    setUserCharacterId, setActiveCharacterIds, setActiveCharacterWeights,
    setMultiRoleContext, setActiveTab, settings,
  ]);

  /** Eloquent Home: landing UI with no active tab (distinct from New Chat). */
  const goToHome = useCallback(() => {
    const prevId = activeConversationRef.current;
    const prevMsgs = messagesRef.current;

    if (prevId && prevMsgs?.length) {
      const prevConv = (conversationsRef.current || []).find((c) => c.id === prevId);
      const { messages: _omit, ...catalogMeta } = prevConv || { id: prevId, name: 'Chat' };
      void saveActiveConversationMessages(prevId, prevMsgs, catalogMeta);
      setConversations((prev) =>
        prev.map((c) => (c.id === prevId ? { ...c, messages: prevMsgs } : c))
      );
    }

    conversationSwitchInProgressRef.current = false;
    conversationSelectionRequestRef.current += 1;
    activeConversationRef.current = null;
    setActiveConversation(null);
    messagesRef.current = [];
    setMessages([]);

    startupConversationRestoreRef.current.attempted = true;

    const list = (conversationsRef.current || []).filter(
      (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
    );
    if (list.length > 0) {
      void saveConversationCatalog(list, null);
    }
    void indexedDbStorage.removeItem('Eloquent-active-conversation');
  }, [setConversations, setActiveConversation, setMessages]);

  const applyIntroChatTitle = useCallback((conversationId, introResult) => {
    const id = conversationId || activeConversationRef.current;
    if (!id || !introResult) return false;

    const conv = (conversationsRef.current || []).find((c) => c.id === id);
    if (!conversationAcceptsIntroTitle(conv)) return false;

    const primaryId = conv?.characterIds?.primary;
    const systemPersonaId = conv?.systemPersonaCharacterId;
    const char =
      (primaryId ? characters.find((c) => c.id === primaryId) : null)
      || (systemPersonaId ? characters.find((c) => c.id === systemPersonaId) : null)
      || (id === activeConversationRef.current ? primaryCharacter : null)
      || null;
    const title = deriveIntroChatTitle(introResult, { characterName: char?.name });
    if (!title) return false;

    let applied = false;
    setConversations((prev) => {
      const next = prev.map((c) => {
        if (c.id !== id || !conversationAcceptsIntroTitle(c)) return c;
        applied = true;
        return {
          ...c,
          name: title,
          requiresTitle: false,
          titleSource: 'intro',
        };
      });
      if (applied) void saveConversationCatalog(next, id);
      return next;
    });
    return applied;
  }, [characters, primaryCharacter, setConversations]);

  const updateCharacterIntro = useCallback((conversationId, introPatch) => {
    const id = conversationId || activeConversationRef.current;
    if (!id || !introPatch || typeof introPatch !== 'object') return;
    setConversations((prev) => {
      const next = prev.map((c) => {
        if (c.id !== id) return c;
        const prevIntro = c.characterIntro && typeof c.characterIntro === 'object' ? c.characterIntro : {};
        return {
          ...c,
          characterIntro: {
            ...prevIntro,
            ...introPatch,
            updatedAt: new Date().toISOString(),
          },
        };
      });
      void saveConversationCatalog(next, id);
      return next;
    });
  }, [setConversations]);

  const completeCharacterIntro = useCallback((conversationId, options = {}) => {
    const id = conversationId || activeConversationRef.current;
    if (!id) return false;

    const conv = (conversationsRef.current || []).find((c) => c.id === id);
    const introResult = options.introResult ?? conv?.characterIntro?.result;
    if (introResult) {
      applyIntroChatTitle(id, introResult);
    }
    const primaryId = conv?.characterIds?.primary;
    const systemPersonaId = conv?.systemPersonaCharacterId;
    const char =
      (primaryId ? characters.find((c) => c.id === primaryId) : null)
      || (systemPersonaId ? characters.find((c) => c.id === systemPersonaId) : null)
      || (id === activeConversationRef.current ? primaryCharacter : null)
      || null;

    const isActiveConversation = activeConversationRef.current === id;
    const currentMsgs = isActiveConversation
      ? (messagesRef.current || [])
      : (Array.isArray(conv?.messages) ? conv.messages : []);
    const alreadyHasBot = currentMsgs.some((m) => m.role === 'bot');
    const seed = !alreadyHasBot
      ? buildCharacterIntroSeedMessages(introResult, {
        character: char,
        avatar: char ? getActiveCharacterAvatar(char) : undefined,
        generateId: generateUniqueId,
      })
      : [];

    setConversations((prev) => {
      const next = prev.map((c) => {
        if (c.id !== id) return c;
        const mergedMessages =
          seed.length > 0
            ? seed
            : (Array.isArray(c.messages) && c.messages.length ? c.messages : []);
        return { ...c, introPending: false, messages: mergedMessages };
      });
      void saveConversationCatalog(next, id);
      return next;
    });

    if (isActiveConversation) {
      const nextMsgs = seed.length > 0 ? seed : currentMsgs;
      if (seed.length > 0) {
        messagesRef.current = nextMsgs;
        setMessages(nextMsgs);
        const convMeta = (conversationsRef.current || []).find((c) => c.id === id);
        const { messages: _omit, ...catalogMeta } = convMeta || { id, name: 'Chat' };
        void saveActiveConversationMessages(id, nextMsgs, catalogMeta);
      }
    }

    return seed.length > 0;
  }, [characters, primaryCharacter, applyIntroChatTitle, setConversations, setMessages]);

  completeCharacterIntroRef.current = completeCharacterIntro;

  /** Apply roster + messages from a conversation object (sidebar click, outreach open, deep links). */
  const applyConversationSelection = useCallback((sel) => {
    if (!sel) return;
    setMessages(Array.isArray(sel.messages) ? sel.messages : []);
    const { primary, secondary, user } = sel.characterIds || {};
    let primChar = characters.find(c => c.id === primary) || null;
    if (!primChar && sel.characterSnapshot) {
      const snap = sel.characterSnapshot;
      if (!primary || snap.id === primary) {
        primChar = normalizeCharacter(snap);
      }
    }
    const secChar = secondary ? (characters.find(c => c.id === secondary) || null) : null;
    setPrimaryCharacter(primChar);
    setSecondaryCharacter(secChar);
    setActiveCharacter(primChar);
    setUserCharacterId(user || null);
    const defaultActiveIds = characters
      .map(normalizeCharacter)
      .filter(c => normalizeChatRole(c.chat_role) !== 'user')
      .map(c => c.id);
    setActiveCharacterIds(Array.isArray(sel.activeCharacterIds) ? sel.activeCharacterIds : defaultActiveIds);
    const defaultWeights = buildDefaultCharacterWeights(characters);
    setActiveCharacterWeights(
      sel.activeCharacterWeights && typeof sel.activeCharacterWeights === 'object'
        ? sel.activeCharacterWeights
        : defaultWeights
    );
    setMultiRoleContext(typeof sel.multiRoleContext === 'string' ? sel.multiRoleContext : '');
  }, [characters, setMessages, setPrimaryCharacter, setSecondaryCharacter, setActiveCharacter, setUserCharacterId, setActiveCharacterIds, setActiveCharacterWeights, setMultiRoleContext]);

  const handleConversationClick = useCallback((id) => {
    if (!id || tombstonedConversationIdsRef.current.has(id)) return;
    const selectionRequest = conversationSelectionRequestRef.current + 1;
    conversationSelectionRequestRef.current = selectionRequest;
    const prevId = activeConversationRef.current;
    const prevMsgs = messagesRef.current;

    const flushPrevShard = () => {
      if (prevId && prevMsgs?.length) {
        const prevConv = (conversationsRef.current || []).find((c) => c.id === prevId);
        const { messages: _omit, ...catalogMeta } = prevConv || { id: prevId, name: 'Chat' };
        void saveActiveConversationMessages(prevId, prevMsgs, catalogMeta);
      }
    };

    if (prevId && prevId !== id) {
      flushPrevShard();
      setConversations((prev) =>
        prev.map((c) => (c.id === prevId ? { ...c, messages: prevMsgs } : c))
      );
    }

    conversationSwitchInProgressRef.current = true;
    activeConversationRef.current = id;
    setActiveConversation(id);
    void (async () => {
      try {
        const versionBeforeLoad = conversationMessageVersionRef.current.get(id) || 0;
        let shardMsgs = await loadConversationMessages(id);
        if (versionBeforeLoad !== (conversationMessageVersionRef.current.get(id) || 0)) {
          shardMsgs = await loadConversationMessages(id);
        }
        if (
          conversationSelectionRequestRef.current !== selectionRequest
          || activeConversationRef.current !== id
          || tombstonedConversationIdsRef.current.has(id)
        ) {
          return;
        }
        const conv = conversationsRef.current.find((c) => c.id === id) || {};
        messagesRef.current = shardMsgs;
        applyConversationSelection({ ...conv, messages: shardMsgs });
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, messages: shardMsgs } : c))
        );
      } finally {
        if (conversationSelectionRequestRef.current === selectionRequest) {
          conversationSwitchInProgressRef.current = false;
        }
      }
    })();
  }, [setConversations, setActiveConversation, applyConversationSelection]);

  // ----------------------------------
  // Character Management
  // ----------------------------------
  const loadCharacters = useCallback(async () => {
    try {
      let saved = await indexedDbStorage.getItem('llm-characters', {
        preferLocalStorage: true,
        skipMigration: true,
      });
      if (!saved && typeof localStorage !== 'undefined') {
        try {
          saved = localStorage.getItem('llm-characters');
          if (saved) await indexedDbStorage.setItem('llm-characters', saved);
        } catch (_) { /* ignore */ }
      }
      if (saved) {
        const parsed = JSON.parse(saved).map(normalizeCharacter);
        setCharacters(parsed);
        const conv = conversations.find(c => c.id === activeConversation);
        const charId = conv?.characterIds?.primary;
        if (charId) setActiveCharacter(parsed.find(c => c.id === charId) || null);
        const userId = conv?.characterIds?.user;
        if (userId) setUserCharacterId(userId);
      }
    } catch (err) {
      console.error('Load chars error:', err);
    }
  }, [activeConversation, conversations]);

  const saveCharacter = useCallback((data) => {
    if (!storageHydrated) {
      console.warn('[saveCharacter] blocked until browser storage has finished loading');
      alert('Your character library is still loading. Wait a moment and try again.');
      return { ...data, chat_role: normalizeChatRole(data?.chat_role) };
    }
    let savedCharacter = omitPersistedLocalAvatarFolder({
      ...data,
      chat_role: normalizeChatRole(data?.chat_role),
    });

    setCharacters(prev => {
      const list = prev.slice();

      if (!data.id) {
        // Create new character
        savedCharacter = {
          ...savedCharacter,
          id: `char_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          created_at: new Date().toISOString().split('T')[0]
        };
        list.push(savedCharacter);
        console.log('Creating new character:', savedCharacter.name, 'with ID:', savedCharacter.id);
      } else {
        // Update existing character
        const idx = list.findIndex(c => c.id === data.id);
        if (idx > -1) {
          savedCharacter = { ...savedCharacter };
          list[idx] = savedCharacter;
          console.log('Updating existing character:', savedCharacter.name, 'with ID:', savedCharacter.id);
        } else {
          // State batching: character not yet in the array. Push with provided ID, don't create duplicate.
          list.push(savedCharacter);
          console.log('Character added to state:', savedCharacter.name, 'with ID:', savedCharacter.id);
        }
      }

      try {
        let normalizedList = list.map(normalizeCharacter);
        if (savedCharacter.chat_role === 'user') {
          normalizedList = normalizedList.map(char => {
            if (char.id === savedCharacter.id) return { ...char, chat_role: 'user' };
            if (normalizeChatRole(char.chat_role) === 'user') return { ...char, chat_role: 'npc' };
            return char;
          });
        }
        indexedDbStorage.setItem('llm-characters', JSON.stringify(normalizedList));
        try {
          localStorage.setItem('llm-characters', JSON.stringify(normalizedList));
        } catch (_) { /* mirror for legacy readers */ }
        console.log('Characters saved to storage:', normalizedList.length, 'total characters');
        return normalizedList;
      } catch (error) {
        console.error('Failed to save characters to localStorage:', error);
      }

      return list.map(normalizeCharacter);
    });

    return savedCharacter; // Return the character with its final ID
  }, [storageHydrated]);

  const saveCharacters = useCallback((items) => {
    const inputCharacters = Array.isArray(items) ? items : [];
    if (inputCharacters.length === 0) return [];
    if (!storageHydrated) {
      console.warn('[saveCharacters] blocked until browser storage has finished loading');
      alert('Your character library is still loading. Wait a moment and try again.');
      return [];
    }

    const importTimestamp = Date.now();
    const importedCharacters = inputCharacters.map((data, index) => omitPersistedLocalAvatarFolder({
      ...data,
      id: data?.id || `char_${importTimestamp}_${index}_${Math.random().toString(36).substring(2, 9)}`,
      chat_role: normalizeChatRole(data?.chat_role),
      created_at: data?.created_at || new Date().toISOString().split('T')[0],
    }));

    setCharacters((previousCharacters) => {
      const mergedCharacters = [...previousCharacters];
      importedCharacters.forEach((character) => {
        const existingIndex = mergedCharacters.findIndex((item) => item.id === character.id);
        if (existingIndex >= 0) {
          mergedCharacters[existingIndex] = character;
        } else {
          mergedCharacters.push(character);
        }
      });

      let normalizedList = mergedCharacters.map(normalizeCharacter);
      const importedUserCharacter = [...importedCharacters]
        .reverse()
        .find((character) => character.chat_role === 'user');
      if (importedUserCharacter) {
        normalizedList = normalizedList.map((character) => ({
          ...character,
          chat_role: character.id === importedUserCharacter.id ? 'user' : 'npc',
        }));
      }

      const serialized = JSON.stringify(normalizedList);
      indexedDbStorage.setItem('llm-characters', serialized);
      try {
        localStorage.setItem('llm-characters', serialized);
      } catch (_) { /* mirror for legacy readers */ }
      return normalizedList;
    });

    return importedCharacters;
  }, [storageHydrated]);

  const setCharacterAvatarIndex = useCallback((characterId, index) => {
    if (!characterId || !storageHydrated) return;
    setActiveCharacter((prev) => {
      if (prev?.id !== characterId) return prev;
      const stored = characters.find((c) => c.id === characterId);
      const base = mergeSessionLocalAvatarFolder(stored, prev);
      const next = normalizeCharacter(setAvatarIndexOnCharacter(base, index));
      const persisted = omitPersistedLocalAvatarFolder(next);
      setCharacters((list) => {
        const updated = list.map((c) => (c.id === characterId ? persisted : normalizeCharacter(c)));
        indexedDbStorage.setItem('llm-characters', JSON.stringify(updated));
        return updated;
      });
      return next;
    });
  }, [characters, storageHydrated]);

  const cycleCharacterAvatar = useCallback((characterId, delta = 1) => {
    if (!characterId || !storageHydrated) return;
    const stored = characters.find((c) => c.id === characterId);
    if (!stored) return;
    setActiveCharacter((prev) => {
      if (prev?.id !== characterId) return prev;
      const base = mergeSessionLocalAvatarFolder(stored, prev);
      const next = normalizeCharacter(cycleAvatarIndex(base, delta));
      const persisted = omitPersistedLocalAvatarFolder(next);
      setCharacters((list) => {
        const updated = list.map((c) => (c.id === characterId ? persisted : normalizeCharacter(c)));
        indexedDbStorage.setItem('llm-characters', JSON.stringify(updated));
        return updated;
      });
      return next;
    });
  }, [characters, storageHydrated]);

  const deleteCharacter = useCallback((id) => {
    if (!storageHydrated) {
      console.warn('[deleteCharacter] blocked until browser storage has finished loading');
      return;
    }
    setCharacters(prev => {
      const filtered = prev.filter(c => c.id !== id);
      indexedDbStorage.setItem('llm-characters', JSON.stringify(filtered));
      return filtered;
    });
    if (activeCharacter?.id === id) setActiveCharacter(null);
    if (userCharacterId === id) setUserCharacterId(null);
  }, [activeCharacter, userCharacterId, storageHydrated]);

  const duplicateCharacter = useCallback((id) => {
    const orig = characters.find(c => c.id === id);
    if (!orig) {
      console.error('Original character not found for duplication:', id);
      return;
    }

    // Create a deep copy and clear the ID so saveCharacter will assign a new one
    const duplicatedData = {
      ...orig,
      id: null, // Clear ID so saveCharacter will create a new one
      name: `${orig.name} (Copy)`,
      created_at: new Date().toISOString().split('T')[0] // Update creation date
    };

    console.log('Duplicating character:', orig.name, '→', duplicatedData.name);
    saveCharacter(duplicatedData);
  }, [characters, saveCharacter]);

  const setCharacterChatRole = useCallback((id, role) => {
    const normalizedRole = normalizeChatRole(role);
    setCharacters(prev => {
      const updated = prev.map(char => {
        if (char.id === id) return { ...char, chat_role: normalizedRole };
        if (normalizedRole === 'user' && normalizeChatRole(char.chat_role) === 'user') {
          return { ...char, chat_role: 'npc' };
        }
        return normalizeCharacter(char);
      });
      indexedDbStorage.setItem('llm-characters', JSON.stringify(updated));
      return updated;
    });
    if (normalizedRole === 'user') setUserCharacterId(id);
  }, []);

  const setUserCharacterById = useCallback((id) => {
    const nextId = id || null;
    setUserCharacterId(nextId);
    if (settings.multiRoleMode && nextId) {
      setCharacterChatRole(nextId, 'user');
    }
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c => (
          c.id === activeConversation
            ? { ...c, characterIds: { ...c.characterIds, user: nextId } }
            : c
        ))
      );
    }
  }, [activeConversation, setConversations, setCharacterChatRole, settings.multiRoleMode]);

  const updateActiveCharacterIds = useCallback((ids) => {
    const unique = Array.from(new Set(ids || [])).filter(Boolean);
    const nextWeights = { ...(activeCharacterWeights || {}) };
    unique.forEach(id => {
      if (nextWeights[id] == null) nextWeights[id] = 50;
    });
    setActiveCharacterIds(unique);
    setActiveCharacterWeights(nextWeights);
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c => (
          c.id === activeConversation
            ? { ...c, activeCharacterIds: unique, activeCharacterWeights: nextWeights }
            : c
        ))
      );
    }
  }, [activeConversation, activeCharacterWeights, setConversations]);

  const updateActiveCharacterWeights = useCallback((updates) => {
    const updateMap = updates && typeof updates === 'object' ? updates : {};
    setActiveCharacterWeights(prev => {
      const next = { ...(prev || {}) };
      Object.keys(updateMap).forEach((id) => {
        const raw = updateMap[id];
        const parsed = Number.parseInt(raw, 10);
        if (!Number.isFinite(parsed)) return;
        next[id] = Math.max(1, Math.min(100, parsed));
      });
      if (activeConversation) {
        setConversations(cs =>
          cs.map(c =>
            c.id === activeConversation
              ? { ...c, activeCharacterWeights: next }
              : c
          )
        );
      }
      return next;
    });
  }, [activeConversation, setConversations]);

  const updateMultiRoleContext = useCallback((value) => {
    const nextValue = typeof value === 'string' ? value : '';
    setMultiRoleContext(nextValue);
    if (activeConversation) {
      setConversations(prev =>
        prev.map(c =>
          c.id === activeConversation
            ? { ...c, multiRoleContext: nextValue }
            : c
        )
      );
    }
  }, [activeConversation, setConversations]);

  const applyCharacter = useCallback((id) => {
    const requested = characters.find(c => c.id === id) || null;
    const requestedRole = requested ? normalizeChatRole(requested.chat_role) : 'npc';
    const char = settings.multiRoleMode && requestedRole === 'user'
      ? (characters.find(c => normalizeChatRole(c.chat_role) !== 'user') || null)
      : requested;
    setActiveCharacter(char);
    setPrimaryCharacter(char);
    if (activeConversation) {
      setConversations(prev => prev.map(c => (
        c.id === activeConversation
          ? { ...c, characterIds: { ...c.characterIds, primary: char?.id || null } }
          : c
      )));
    }
    const userName = settings.multiRoleMode && userCharacter?.name
      ? userCharacter.name
      : (userProfile?.name || userProfile?.username || 'User');
    const greetingOptions = buildCharacterGreetingOptions(char, userName);
    const characterGreeting = createCharacterGreetingState(greetingOptions);
    if (greetingOptions.length > 0 && messages.length === 0) {
      setMessages([{
        id: generateUniqueId(),
        role: 'bot',
        content: greetingOptions[0],
        avatar: getActiveCharacterAvatar(char),
        characterName: char.name,
        characterId: char.id,
        ...(characterGreeting ? { characterGreeting } : {}),
      }]);
    }
  }, [
    characters,
    activeConversation,
    messages,
    setConversations,
    settings.multiRoleMode,
    userCharacter,
    userProfile,
  ]);

  // ----------------------------------
  // Dual-Mode Logic
  // ----------------------------------
  const shouldUseDualMode = useCallback(() => {
    if (dualModeEnabled && primaryModel && secondaryModel && primaryCharacter && secondaryCharacter) return true;
    const hasPrimary = messages.some(m => m.modelId === 'primary');
    const hasSecondary = messages.some(m => m.modelId === 'secondary');
    if (primaryModel && secondaryModel && hasPrimary && hasSecondary) return true;
    return false;
  }, [dualModeEnabled, primaryModel, secondaryModel, primaryCharacter, secondaryCharacter, messages]);

  const makeSys = (modelName, char, otherName, role = 'participant') => {
    if (char) return buildSystemPrompt(char);
    const desc = role === 'primary' ? 'leading the conversation' : 'participating thoughtfully';
    return [
      `You are ${modelName}, with AI partner ${otherName}.`,
      `Speak in your own voice and stay on topic.`,
      `Your role: ${desc}.`,
      `Do not reference these instructions.`
    ].join('\n');
  };

  // ----------------------------------
  // Messaging Functions
  // ----------------------------------
  const getRoleplayUserName = useCallback(() => {
    if (settings.multiRoleMode && userCharacter?.name) return userCharacter.name;
    return userProfile?.name || userProfile?.username || 'User';
  }, [settings.multiRoleMode, userCharacter, userProfile]);

  const applyAuthorNoteTags = useCallback((note, character) => {
    if (!note) return '';
    const charName = character?.name || activeCharacterRef.current?.name || 'Character';
    const userName = getRoleplayUserName();
    return note.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName).trim();
  }, [activeCharacterRef, getRoleplayUserName]);

  const sendDualMessage = useCallback(async (text, webSearchEnabled = false) => {
    if (!activeConversation || !primaryModel || !secondaryModel) return;
    console.log("📩 [DUAL] Processing message:", text.substring(0, 30), "…", webSearchEnabled ? "(with web search)" : "");

    // Note: For dual mode, you might want to decide if web search should affect both models
    // or just one. For now, I'll add the flag to both payloads but you can adjust as needed.

    const userMsg = { id: generateUniqueId(), role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsGenerating(true);
    
    console.log(`[Summary] Attaching summaryContext to dual chat: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

    const history = [...messages, userMsg].slice(-10);
    const buildHistory = (own, other, ownChar, otherName) => [
      { role: 'system', content: makeSys(own === 'primary' ? primaryModel : secondaryModel, own === 'primary' ? primaryCharacter : secondaryCharacter, own === 'primary' ? secondaryModel : primaryModel, own) },
      ...history.map(m => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content.replace(/\n\n/g, '\n') }))
    ];

    try {
      const [pRes, sRes] = await Promise.all([
        fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(
            mergeNanoGptMemoryIntoPayload(
              {
                model_name: primaryModel,
                messages: buildHistory('primary', 'secondary', primaryCharacter, secondaryModel),
                ...getChatTemplateRequestFields({
                  conversations: conversationsRef.current,
                  conversationId: activeConversation,
                  history: buildHistory('primary', 'secondary', primaryCharacter, secondaryModel),
                  isApi: primaryIsAPI,
                  customTemplates: settings.modelChatTemplates,
                }),
                gpu_id: 0,
                userProfile,
                authorNote:
                  applyAuthorNoteTags(settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : null, primaryCharacter) ||
                  undefined,
                use_web_search: webSearchEnabled, // NEW: Add to primary
                ...(webSearchEnabled ? getWebSearchResearchPayload(settings) : {}),
                summaryContext: summaryContextForRequest,
                active_character: primaryCharacter || null,
              },
              settings
            )
          ),
        }).then(r => r.json()),
        fetch(`${SECONDARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(
            mergeNanoGptMemoryIntoPayload(
              {
                model_name: secondaryModel,
                messages: buildHistory('secondary', 'primary', secondaryCharacter, primaryModel),
                ...getChatTemplateRequestFields({
                  conversations: conversationsRef.current,
                  conversationId: activeConversation,
                  history: buildHistory('secondary', 'primary', secondaryCharacter, primaryModel),
                  isApi: secondaryIsAPI,
                  customTemplates: settings.modelChatTemplates,
                }),
                gpu_id: 1,
                userProfile,
                authorNote:
                  applyAuthorNoteTags(settings.authorNote && settings.authorNote.trim() ? settings.authorNote.trim() : null, secondaryCharacter) ||
                  undefined,
                use_web_search: false, // Usually only primary should do web search in dual mode
                summaryContext: summaryContextForRequest,
                active_character: secondaryCharacter || null,
              },
              settings
            )
          ),
        }).then(r => r.json())
      ]);

      const pId = generateUniqueId();
      const sId = generateUniqueId();
      const pText = cleanModelOutput(pRes.text);
      const sText = cleanModelOutput(sRes.text);

      setMessages(prev => [...prev,
      { id: pId, role: 'bot', content: pText, modelId: 'primary', characterName: primaryCharacter?.name, characterId: primaryCharacter?.id, avatar: getActiveCharacterAvatar(primaryCharacter) },
      { id: sId, role: 'bot', content: sText, modelId: 'secondary', characterName: secondaryCharacter?.name, characterId: secondaryCharacter?.id, avatar: getActiveCharacterAvatar(secondaryCharacter) }
      ]);

      await observeConversationWithAgent(text, `${pRes.text}\n\n${sRes.text}`);
      const userIdDual = resolveAgenticUserId();
      const activeIdDual = activeConversationRef.current;
      const apiOptsDual = primaryIsAPI ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: primaryModel } : null;
      processAgenticMemoryIfEnabled(userIdDual, primaryCharacter, text, pText, apiOptsDual);
      processAgenticMemoryIfEnabled(userIdDual, secondaryCharacter, text, sText, apiOptsDual);

      // Auto-play TTS for BOTH messages if enabled
      if (settings.ttsAutoPlay && settings.ttsEnabled) {
        // We'll play them sequentially? Or just primary? 
        // For now let's play primary, then secondary might overlap or we'd need a queue.
        // Actually playTTS stops current audio, so primary then secondary would just play secondary.
        // Let's just play primary for now to avoid complexity, or the characters will talk over each other.
        playTTS(pId, pText, getTtsOverridesForCharacter(primaryCharacter));
      }
    } catch (err) {
      console.error('Dual chat error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [activeConversation, primaryModel, secondaryModel, messages, primaryCharacter, secondaryCharacter, PRIMARY_API_URL, SECONDARY_API_URL, userProfile, memoryContext, settings, observeConversationWithAgent, processAgenticMemoryIfEnabled, summaryContextForRequest, getTtsOverridesForCharacter, applyAuthorNoteTags, resolveAgenticUserId]);

  const getMultiRoleContextBlock = useCallback(() => {
    if (!settings.multiRoleMode) return '';
    const trimmed = (multiRoleContext || '').trim();
    if (!trimmed) return '';
    return `\n\n[GROUP SCENE CONTEXT]\n${trimmed}`;
  }, [settings.multiRoleMode, multiRoleContext]);

  const getNarratorCharacter = useCallback(() => {
    if (!settings.narratorEnabled) return null;
    const name = (settings.narratorName || '').trim() || 'Narrator';
    return {
      id: NARRATOR_CHARACTER_ID,
      name,
      description: 'A narrator who describes the scene and transitions.',
      personality: '',
      background: '',
      speech_style: 'Narrative, concise, scene-setting.',
      chat_role: 'narrator',
      avatar: settings.narratorAvatar || null,
      isNarratorSystem: true
    };
  }, [settings.narratorEnabled, settings.narratorName, settings.narratorAvatar]);

  const buildNarratorSystemPrompt = useCallback(() => {
    const narratorName = (settings.narratorName || '').trim() || 'Narrator';
    const userName = getRoleplayUserName();
    const charName = activeCharacterRef.current?.name || primaryCharacter?.name || 'Character';
    const rawInstructions = settings.narratorInstructions || '';
    const narratorInstructions = rawInstructions
      .replace(/{{user}}/gi, userName)
      .replace(/{{char}}/gi, charName)
      .trim();

    const summaryContext = activeContextSummary || userProfile?.activeContextSummary || '';
    const summaryBlock = summaryContext
      ? `\n\n[PREVIOUS STORY SUMMARY]:\n${summaryContext}\n[End of Summary]\n`
      : '';
    const storyContext = getStoryTrackerContext();
    const groupContext = getMultiRoleContextBlock();
    const guidance = narratorInstructions ? `\n\nNARRATION GUIDANCE:\n${narratorInstructions}` : '';

    return `You are ${narratorName}, the narrator of this roleplay.

Purpose:
- Describe scene transitions, atmosphere, and world context.
- Keep narration concise (2-5 sentences) unless the user asks for more.
- Do not speak for ${userName} or write dialogue on behalf of characters.
- Avoid labeling lines with character names or using speaker tags.

If you must mention character actions, do so briefly in third-person prose.
${guidance}${groupContext}${summaryBlock}${storyContext}`;
  }, [activeCharacterRef, primaryCharacter, settings.narratorInstructions, settings.narratorName, activeContextSummary, userProfile, getRoleplayUserName, getMultiRoleContextBlock]);

  const buildRoleplayRosterBlock = useCallback(() => {
    if (!settings.multiRoleMode) return '';
    const normalized = (characters || []).map(normalizeCharacter);
    const rosterIds = Array.isArray(activeCharacterIds) ? activeCharacterIds : [];
    const rosterSet = rosterIds.length ? new Set(rosterIds) : null;
    const rosterFiltered = rosterSet ? normalized.filter(c => rosterSet.has(c.id)) : normalized;
    const userName = getRoleplayUserName();
    const narratorName = settings.narratorEnabled
      ? ((settings.narratorName || '').trim() || 'Narrator')
      : null;
    const npcNames = rosterFiltered
      .filter(c => normalizeChatRole(c.chat_role) === 'npc')
      .map(c => c.name)
      .filter(Boolean);
    const narratorNames = rosterFiltered
      .filter(c => normalizeChatRole(c.chat_role) === 'narrator')
      .map(c => c.name)
      .filter(Boolean);

    const narratorLine = narratorName
      ? `\nNarrator: ${narratorName}`
      : '';
    const narratorListLine = narratorNames.length
      ? `\nNarrator Characters: ${narratorNames.join(', ')}`
      : '';

    return `\n\n[ROLEPLAY MODE]\nUser Character: ${userName}\nCharacters: ${npcNames.length ? npcNames.join(', ') : 'None'}${narratorListLine}${narratorLine}\nRules: Never write dialogue or actions for ${userName}. Only one character speaks per response. Do not include other characters' dialogue.`;
  }, [characters, activeCharacterIds, getRoleplayUserName, settings.multiRoleMode, settings.narratorEnabled, settings.narratorName]);

  const buildSpeakerSelectionPrompt = useCallback((recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText) => {
    const candidateList = candidates
      .map(c => `- ${c.name} (${normalizeChatRole(c.chat_role)})`)
      .join('\n');
    const weightList = candidates
      .map(c => {
        const raw = weightMap?.[c.id];
        const parsed = Number.parseInt(raw, 10);
        const weight = Number.isFinite(parsed) ? parsed : 50;
        return `- ${c.name}: ${weight}`;
      })
      .join('\n');
    const recentHistory = recentMessages
      .filter(m => m.role !== 'system')
      .slice(-10)
      .map(m => {
        if (m.role === 'user') {
          return `${m.characterName || userName}: ${m.content}`;
        }
        if (m.role === 'bot') {
          return `${m.characterName || 'Assistant'}: ${m.content}`;
        }
        return `System: ${m.content}`;
      })
      .join('\n');

    const lastSpeakerLine = lastSpeakerName ? `Last speaker: ${lastSpeakerName} (avoid repeating if possible)` : 'Last speaker: None';
    const userLine = lastUserText ? `Latest user message:\n${lastUserText}` : 'Latest user message: None';
    return `You are a scriptwriter selecting the next speaker in a roleplay.

Candidates:
${candidateList || 'None'}

Selection weights (higher = more likely):
${weightList || 'None'}

User Character: ${userName} (never select)
${lastSpeakerLine}
${userLine}

Rules:
- Prefer characters explicitly addressed by the user.
- Prefer higher-weight characters when multiple are plausible.

Recent conversation:
${recentHistory}

Return ONLY valid JSON:
{"selection": "Character Name"}`;
  }, []);

  const selectNextSpeaker = useCallback(async (recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText) => {
    if (!primaryModel || !PRIMARY_API_URL) return null;
    if (!candidates.length) return null;

    const selectionSystem = "You are a careful assistant that returns only valid JSON.";
    const selectionPrompt = buildSpeakerSelectionPrompt(recentMessages, candidates, userName, lastSpeakerName, weightMap, lastUserText);
    const prompt = formatPrompt([{ role: 'user', content: selectionPrompt }], primaryModel, selectionSystem);

    try {
      const res = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          mergeNanoGptMemoryIntoPayload(
            {
              prompt,
              model_name: primaryModel,
              directProfileInjection: true,
              temperature: 0.2,
              top_p: 0.9,
              top_k: 40,
              repetition_penalty: 1.05,
              max_tokens: 120,
              gpu_id: 0,
              stream: false,
              request_purpose: 'speaker_selection',
            },
            settings
          )
        ),
      });

      if (!res.ok) return null;
      const data = await res.json();
      const rawText = cleanModelOutput(data.text || '');
      const jsonText = extractFirstJson(rawText) || extractFirstJson(data.text || '');
      if (!jsonText) return null;

      const parsed = JSON.parse(jsonText);
      const selectionName = parsed?.selection ? String(parsed.selection).trim() : '';
      if (!selectionName) return null;

      const normalizedSelection = selectionName.toLowerCase();
      return candidates.find(c => (c.name || '').toLowerCase() === normalizedSelection) || null;
    } catch (error) {
      console.warn('Speaker selection failed:', error);
      return null;
    }
  }, [PRIMARY_API_URL, primaryModel, formatPrompt, cleanModelOutput, buildSpeakerSelectionPrompt, settings]);

  const resolveSpeakerCharacter = useCallback(async (text, recentMessages, options = {}) => {
    const { speakerCharacterId = null, forceAutoSelectSpeaker = false, ignoreMentionedCandidates = false } = options;
    const activeConv = (conversationsRef.current || []).find(
      (c) => c.id === activeConversationRef.current
    );
    if (speakerCharacterId === NARRATOR_CHARACTER_ID) {
      return getNarratorCharacter();
    }
    if (speakerCharacterId) {
      return characters.find(c => c.id === speakerCharacterId) || null;
    }

    if (!settings.multiRoleMode) {
      const refCur = activeCharacterRef.current || null;
      if (refCur?.id) {
        const fresh = characters.find((c) => c.id === refCur.id);
        if (fresh) return fresh;
      }
      if (refCur) return refCur;
      for (let i = recentMessages.length - 1; i >= 0; i--) {
        const msg = recentMessages[i];
        if (msg?.role !== 'bot') continue;
        if (msg.characterId) {
          const byId = characters.find(c => c.id === msg.characterId);
          if (byId) return byId;
        }
        if (msg.characterName) {
          const byName = characters.find(
            c => (c.name || '').toLowerCase() === String(msg.characterName).toLowerCase()
          );
          if (byName) return byName;
        }
        break;
      }
      return null;
    }

    const narrator = getNarratorCharacter();
    const allowAutoSelect = forceAutoSelectSpeaker || settings.autoSelectSpeaker;
    const narratorInterval = Number.parseInt(settings.narratorInterval, 10);
    const shouldUseNarrator = () => {
      if (!narrator) return false;
      if (!allowAutoSelect) return false;
      if (!Number.isFinite(narratorInterval) || narratorInterval <= 0) return false;
      let botSince = 0;
      for (let i = recentMessages.length - 1; i >= 0; i -= 1) {
        const msg = recentMessages[i];
        if (msg?.role !== 'bot') continue;
        if (msg.characterId === NARRATOR_CHARACTER_ID) {
          if (botSince === 0) return false;
          return botSince >= narratorInterval;
        }
        botSince += 1;
      }
      return botSince >= narratorInterval;
    };

    const normalized = (characters || []).map(normalizeCharacter);
    const rosterIds = Array.isArray(activeCharacterIds) ? activeCharacterIds : [];
    const rosterSet = rosterIds.length ? new Set(rosterIds) : null;
    const rosterFiltered = rosterSet ? normalized.filter(c => rosterSet.has(c.id)) : normalized;
    const candidates = rosterFiltered.filter(c => normalizeChatRole(c.chat_role) !== 'user');
    if (!candidates.length) {
      const refCur = activeCharacterRef.current || null;
      if (refCur?.id) {
        const fresh = characters.find((c) => c.id === refCur.id);
        if (fresh) return fresh;
      }
      return refCur;
    }

    const getWeight = (id) => {
      const raw = activeCharacterWeights?.[id];
      const parsed = Number.parseInt(raw, 10);
      if (!Number.isFinite(parsed)) return 50;
      return Math.max(1, Math.min(100, parsed));
    };

    const normalizeMatch = (value) => String(value || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, ' ')
      .trim();

    const findMentionedCandidate = () => {
      const textMatch = normalizeMatch(text);
      if (!textMatch) return null;
      let best = null;
      let bestWeight = -1;
      for (const candidate of candidates) {
        const nameMatch = normalizeMatch(candidate.name);
        if (!nameMatch) continue;
        if (!textMatch.includes(nameMatch)) continue;
        const weight = getWeight(candidate.id);
        if (weight > bestWeight) {
          best = candidate;
          bestWeight = weight;
        }
      }
      return best;
    };

    const pickWeighted = (excludeId = null) => {
      const pool = candidates.filter(c => c.id !== excludeId);
      if (!pool.length) return candidates[0];
      const total = pool.reduce((sum, c) => sum + getWeight(c.id), 0);
      const roll = Math.random() * total;
      let running = 0;
      for (const candidate of pool) {
        running += getWeight(candidate.id);
        if (roll <= running) return candidate;
      }
      return pool[0];
    };

    const mentioned = ignoreMentionedCandidates ? null : findMentionedCandidate();
    if (mentioned) return mentioned;

    if (shouldUseNarrator()) return narrator;

    if (!allowAutoSelect) {
      const refCur = activeCharacterRef.current || null;
      const current = refCur?.id
        ? (characters.find((c) => c.id === refCur.id) || refCur)
        : refCur;
      const currentRole = current ? normalizeChatRole(current.chat_role) : 'npc';
      const inRoster = !rosterSet || (current && rosterSet.has(current.id));
      if (current && currentRole !== 'user' && inRoster) return current;
      return pickWeighted();
    }

    if (candidates.length === 1) return candidates[0];

    const findLastSpeaker = () => {
      for (let i = recentMessages.length - 1; i >= 0; i -= 1) {
        const msg = recentMessages[i];
        if (msg?.role !== 'bot') continue;
        if (msg.characterId) {
          const byId = candidates.find(c => c.id === msg.characterId);
          if (byId) return byId;
        }
        if (msg.characterName) {
          const byName = candidates.find(c => (c.name || '').toLowerCase() === String(msg.characterName).toLowerCase());
          if (byName) return byName;
        }
      }
      return null;
    };
    const lastSpeaker = findLastSpeaker();
    const lastSpeakerName = lastSpeaker?.name || null;
    const nextByRotation = () => {
      if (!lastSpeaker) return candidates[0];
      const idx = candidates.findIndex(c => c.id === lastSpeaker.id);
      if (idx === -1) return candidates[0];
      return candidates[(idx + 1) % candidates.length];
    };

    const userName = getRoleplayUserName();
    const selected = await selectNextSpeaker(
      recentMessages,
      candidates,
      userName,
      lastSpeakerName,
      activeCharacterWeights,
      text
    );
    if (!selected) return pickWeighted(lastSpeaker?.id || null);
    if (lastSpeaker && selected.id === lastSpeaker.id && candidates.length > 1) {
      return pickWeighted(lastSpeaker.id);
    }
    return selected;
  }, [activeCharacterRef, characters, activeCharacterIds, activeCharacterWeights, getRoleplayUserName, getNarratorCharacter, selectNextSpeaker, settings.autoSelectSpeaker, settings.multiRoleMode, settings.narratorInterval]);


  // New shared helper to build the FULL generation system prompt (memories, lore, etc.)
  const getGenerationSystemPrompt = useCallback(async (text, character, authorNote = null, options = {}) => {
    const memTimeout = primaryIsAPI ? MEMORY_FETCH_TIMEOUT_API_MS : MEMORY_FETCH_TIMEOUT_MS;
    const { includeAuthorNote = true, conversationId = null } = options;
    const { active: systemPersonaActive, character: systemPersonaChar } =
      getActiveSystemPersonaContext(conversationId);
    const chatCharacter =
      character?.id === NARRATOR_CHARACTER_ID && settings.narratorEnabled ? null : character;

    console.log(
      '🔍 [SystemPrompt] chat:',
      chatCharacter?.name,
      'systemPersona:',
      systemPersonaActive ? systemPersonaChar?.name : 'off',
      'user:',
      userProfile?.name
    );

    const replaceTagsFor = (content, tagCharacter) => {
      if (!content || !tagCharacter) return content;
      const charName = tagCharacter.name || 'Character';
      const userName = getRoleplayUserName();
      return content.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName);
    };

    const appendUserProfileContext = async (baseMsg, tagCharacter) => {
      let contextToAdd = '';
      if (settings.directProfileInjection) {
        const userId = userProfile?.id || (typeof memoryContext !== 'undefined' ? memoryContext?.activeProfileId : null);
        profileReinforcementRef.current = '';
        if (userId) {
          try {
            const res = await fetchWithTimeout(
              `${MEMORY_API_URL}/memory/get_all?user_id=${userId}`,
              {},
              memTimeout
            );
            if (res.ok) {
              const data = await res.json();
              if (data.memories && data.memories.length > 0) {
                const bullets = data.memories.map((mem) => {
                  const category = mem.category?.replace('_', ' ') || 'memory';
                  const importance = mem.importance?.toFixed(1) || 'N/A';
                  const content = replaceTagsFor(mem.content, tagCharacter);
                  return `• ${content} (Category: ${category}, Importance: ${importance})`;
                });
                const profileString = bullets.join('\n');
                contextToAdd += `\n\nUSER MEMORY PROFILE:\n${profileString}`;
                const reinforcementSnippet = bullets.slice(0, 5).join('\n');
                if (reinforcementSnippet) profileReinforcementRef.current = reinforcementSnippet;
              }
            }
          } catch (error) {
            console.error('🧠 [Direct Injection] Error:', error);
          }
        }
      } else {
        profileReinforcementRef.current = '';
        const agentMem = await fetchMemoriesFromAgent(text);
        if (agentMem.length) {
          contextToAdd += `\n\nUSER CONTEXT:\n${agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n')}`;
        }
      }
      return baseMsg + contextToAdd;
    };

    const appendCharacterContext = async (baseMsg, tagCharacter, { includeProfile = false } = {}) => {
      if (!tagCharacter) return baseMsg;
      let contextToAdd = '';
      if (includeProfile) {
        const withProfile = await appendUserProfileContext('', tagCharacter);
        contextToAdd += withProfile;
      }
      const lore = await fetchTriggeredLore(text, tagCharacter);
      if (lore.length) {
        const loreBlock = lore
          .map((l) => {
            const content = typeof l === 'string' ? l : (l.content || JSON.stringify(l));
            return `• ${replaceTagsFor(content, tagCharacter)}`;
          })
          .join('\n');
        contextToAdd += `\n\n[WORLD KNOWLEDGE - Essential lore guidance for this response]\n${loreBlock}`;
      }
      if (tagCharacter.id) {
        const userId = resolveAgenticUserId();
        if (userId) {
          try {
            const params = new URLSearchParams({ user_id: userId, character_id: tagCharacter.id });
            if (userProfile?.name || userProfile?.username) {
              params.set('user_name', userProfile?.name || userProfile?.username || '');
            }
            if (text?.trim()) {
              params.set('query', text.trim());
              params.set('use_rag', 'true');
            }
            const res = await fetchWithTimeout(
              `${MEMORY_API_URL}/memory/agentic?${params.toString()}`,
              {},
              memTimeout
            );
            if (res.ok) {
              const data = await res.json();
              const formatted = data.formatted_context?.trim();
              if (formatted) {
                contextToAdd += `\n\n${formatted}`;
                setLastAgenticInjectMeta({
                  chars: formatted.length,
                  count: data.count ?? null,
                  characterName: tagCharacter.name || tagCharacter.id,
                });
              } else {
                setLastAgenticInjectMeta({
                  chars: 0,
                  count: data.count ?? 0,
                  characterName: tagCharacter.name || tagCharacter.id,
                });
              }
            }
          } catch (err) {
            console.error('🧠 [Agentic memory] fetch error:', err);
          }
        }
      }
      return baseMsg + contextToAdd;
    };

    let systemMsg;
    if (character?.id === NARRATOR_CHARACTER_ID && settings.narratorEnabled) {
      systemMsg = buildNarratorSystemPrompt();
    } else if (systemPersonaActive && systemPersonaChar) {
      let systemLayer =
        buildSystemPersonaPrompt(systemPersonaChar)
        || (`You are a helpful assistant.${getStoryTrackerContext()}`);
      systemLayer = await appendCharacterContext(systemLayer, systemPersonaChar, { includeProfile: true });

      let characterLayer = '';
      if (chatCharacter) {
        characterLayer = buildSystemPrompt(chatCharacter) || '';
        characterLayer = await appendCharacterContext(characterLayer, chatCharacter, { includeProfile: false });
      }
      systemMsg = composeLayeredSystemPrompt(systemLayer, characterLayer);
    } else if (chatCharacter) {
      systemMsg = buildSystemPrompt(chatCharacter) || (`You are a helpful assistant.${getStoryTrackerContext()}`);
      systemMsg = await appendCharacterContext(systemMsg, chatCharacter, { includeProfile: true });
    } else {
      systemMsg = `You are a helpful assistant.${getStoryTrackerContext()}`;
      systemMsg = await appendUserProfileContext(systemMsg, null);
    }

    if (settings.multiRoleMode) {
      systemMsg += getMultiRoleContextBlock();
      systemMsg += buildRoleplayRosterBlock();
    }

    if (settings.multiRoleMode && chatCharacter) {
      const activeName = chatCharacter.name || 'Character';
      systemMsg += `\n\n[ACTIVE SPEAKER]\nYou are speaking ONLY as ${activeName}.\n- Do not write dialogue for any other character.\n- Do not include multiple speakers in one response.\n- Do not prefix lines with character names or labels.\n- If you reference other characters, do so briefly in third-person without quoted speech.`;
    }

    const effectiveAuthorNote = authorNote || (settings.authorNote && settings.authorNote.trim()) || null;
    if (includeAuthorNote && effectiveAuthorNote) {
      const resolvedAuthorNote = applyAuthorNoteTags(effectiveAuthorNote, chatCharacter || systemPersonaChar || character);
      if (resolvedAuthorNote) {
        systemMsg += `\n\n[AUTHOR'S NOTE - Writing style guidance for this response]\n${resolvedAuthorNote}`;
      }
    }

    console.log(`[Summary] System prompt includes summary: ${systemMsg.includes('[PREVIOUS STORY SUMMARY]')}`);
    return systemMsg;
  }, [settings, userProfile, MEMORY_API_URL, primaryIsAPI, fetchMemoriesFromAgent, fetchTriggeredLore, buildSystemPrompt, buildSystemPersonaPrompt, buildNarratorSystemPrompt, buildRoleplayRosterBlock, getRoleplayUserName, getMultiRoleContextBlock, applyAuthorNoteTags, getActiveSystemPersonaContext, memoryContext, resolveAgenticUserId, setLastAgenticInjectMeta]);

  const ROLLING_MEMORY_HEADER =
    '\n\n[CONVERSATION CONTINUITY MEMORY]\nStructured recap of earlier turns in this chat (may omit verbatim wording). Treat as authoritative background; prefer consistency with the verbatim recent turns that follow.\n';

  const prepareApiHistoryWithRollingMemory = useCallback(
    async ({ postUserHistory, baseSystemMsg, conversationId, effectiveMaxTokens }) => {
      const filterChat = (msgs) =>
        (msgs || []).filter(
          (msg) =>
            (msg?.role === 'user' || msg?.role === 'bot') &&
            typeof msg?.content === 'string' &&
            msg.content.length > 0
        );

      const s = settingsRef.current;
      const pack = bookModePackingOverridesRef.current;
      const contextWindowTokens = clampApiContextWindowTokens(
        pack?.apiContextWindowTokens ?? s.apiContextWindowTokens
      );
      const verbatimBudget = pack?.verbatimTokenBudget != null
        ? Math.max(2048, Number(pack.verbatimTokenBudget) || 98304)
        : Math.max(2048, Number(s.apiRecentVerbatimTokenBudget) || 32000);
      const rollingEnabled =
        pack?.forceRollingMemory === true ||
        (!!primaryIsAPI && !!s.apiRollingMemoryEnabled);

      if (!primaryIsAPI || !rollingEnabled) {
        const selectedHistory = selectApiHistoryWithinContext({
          messages: postUserHistory,
          systemPrompt: baseSystemMsg,
          maxContextTokens: contextWindowTokens,
          reservedOutputTokens: effectiveMaxTokens,
          minMessages: 3
        });
        return { systemMsg: baseSystemMsg, selectedHistory };
      }

      const conv = (conversationsRef.current || []).find((c) => c.id === conversationId);
      let rollingPack = (conv?.rollingMemoryPack || '').trim();
      let foldCount = Number.isFinite(conv?.rollingMemoryFoldCount) ? conv.rollingMemoryFoldCount : 0;
      const pendingFold = Number.isFinite(conv?.rollingMemoryFoldCountPending) ? conv.rollingMemoryFoldCountPending : null;

      const draftSelected = selectApiHistoryWithinContext({
        messages: postUserHistory,
        systemPrompt: baseSystemMsg + (rollingPack ? `${ROLLING_MEMORY_HEADER}${rollingPack}\n` : ''),
        maxContextTokens: contextWindowTokens,
        reservedOutputTokens: effectiveMaxTokens,
        maxHistoryTokens: verbatimBudget,
        minMessages: 3
      });

      const chatMessages = filterChat(postUserHistory);
      const oldestSel = draftSelected[0];
      const vStart = oldestSel ? chatMessages.findIndex((m) => m.id === oldestSel.id) : chatMessages.length;
      const safeVStart = vStart >= 0 ? vStart : chatMessages.length;

      // Non-blocking compaction: use the last saved pack immediately, and update the pack in the background.
      // This prevents an extra long API call from delaying every chat response.
      const effectiveFoldCursor = Math.min(
        Math.max(0, Math.max(foldCount, pendingFold ?? 0)),
        safeVStart
      );
      const toFold = chatMessages.slice(effectiveFoldCursor, safeVStart);

      const inFlightKey = conversationId || '__none__';
      const isInFlight = Boolean(rollingMemoryCompactionInFlightRef.current[inFlightKey]);
      if (toFold.length > 0 && !isInFlight && primaryModel && PRIMARY_API_URL) {
        rollingMemoryCompactionInFlightRef.current[inFlightKey] = true;
        // Mark the fold range as pending right away so we don't queue duplicate jobs while the API is slow.
        setConversations((prev) =>
          prev.map((c) =>
            c.id === conversationId
              ? { ...c, rollingMemoryFoldCountPending: safeVStart }
              : c
          )
        );

        const userLabel = getRoleplayUserName();
        // Fire-and-forget job: do not await.
        void (async () => {
          try {
            const nextPack = await mergeRollingMemoryPack({
              apiBaseUrl: PRIMARY_API_URL,
              modelName: primaryModel,
              primaryIsAPI,
              settings,
              existingPack: rollingPack,
              messagesToFold: toFold,
              formatPrompt,
              cleanModelOutput,
              getSpeakerLabel: (m) =>
                m.role === 'user' ? userLabel : (m.characterName || 'Assistant')
            });
            setConversations((prev) =>
              prev.map((c) =>
                c.id === conversationId
                  ? {
                    ...c,
                    rollingMemoryPack: (nextPack || '').trim(),
                    rollingMemoryFoldCount: safeVStart,
                    rollingMemoryFoldCountPending: null
                  }
                  : c
              )
            );
          } catch (e) {
            console.warn('[rolling memory] compaction failed', e);
            // Clear pending so we can retry later.
            setConversations((prev) =>
              prev.map((c) =>
                c.id === conversationId
                  ? { ...c, rollingMemoryFoldCountPending: null }
                  : c
              )
            );
          } finally {
            rollingMemoryCompactionInFlightRef.current[inFlightKey] = false;
          }
        })();
      }

      const systemMsg = baseSystemMsg + (rollingPack ? `${ROLLING_MEMORY_HEADER}${rollingPack}\n` : '');
      const selectedHistory = selectApiHistoryWithinContext({
        messages: postUserHistory,
        systemPrompt: systemMsg,
        maxContextTokens: contextWindowTokens,
        reservedOutputTokens: effectiveMaxTokens,
        maxHistoryTokens: verbatimBudget,
        minMessages: 3
      });

      return { systemMsg, selectedHistory };
    },
    [primaryIsAPI, primaryModel, PRIMARY_API_URL, formatPrompt, setConversations, getRoleplayUserName]
  );

  // In AppContext.jsx, replace the entire generateReply function
  const generateReply = useCallback(async (text, recentMessages, onToken = null, options = {}) => {
    const {
      authorNote = null,
      webSearchEnabled = false,
      speakerCharacterId = null,
      requestPurpose = null,
      onWebSearchMeta = null,
      onWebSearchProgress = null,
      modelName: modelNameOverride = null,
    } = options;
    const speakerCharacter = await resolveSpeakerCharacter(text, recentMessages, { speakerCharacterId });
    const baseSystemMsg = await getGenerationSystemPrompt(text, speakerCharacter, authorNote, {
      includeAuthorNote: false,
      conversationId: activeConversationRef.current || activeConversation,
    });
    console.log(`[Summary] Attaching summaryContext to generateReply: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

    // --- Unified Payload Construction (Matching sendMessage exactly) ---
    const {
      temperature, top_p, top_k, repetition_penalty, frequencyPenalty = 0, presencePenalty = 0,
      antiRepetitionMode = false, use_rag, selectedDocuments = [], streamResponses
    } = settings;

    const historyLimitLocal = 30;
    const effectiveMaxTokens = (settings.max_tokens != null && settings.max_tokens > 0) ? settings.max_tokens : 1_000_000;
    let systemMsg = baseSystemMsg;
    let selectedHistory;
    if (primaryIsAPI) {
      const prep = await prepareApiHistoryWithRollingMemory({
        postUserHistory: recentMessages,
        baseSystemMsg,
        conversationId: activeConversationRef.current || activeConversation,
        effectiveMaxTokens
      });
      systemMsg = prep.systemMsg;
      selectedHistory = prep.selectedHistory;
    } else {
      selectedHistory = recentMessages.slice(-historyLimitLocal);
    }

      const route = resolveUnifiedRequestRoute({
        primaryModel,
        primaryIsAPI,
        settings,
        requestPurpose,
        overrideModel: modelNameOverride || null,
      });
      const effectivePrimaryModel = modelNameOverride || primaryModel || route.effectiveModel;
      const promptModelName = route.selectedModel || effectivePrimaryModel;
      const requestModelName = route.effectiveModel || effectivePrimaryModel;
      if (!requestModelName) {
        throw new Error('No API endpoint available for routing. Enable at least one endpoint.');
      }
      const routerTraceId = createRouteTraceId();
      logRouteTrace({
        action: 'variant',
        route,
        requestPurpose,
        traceId: routerTraceId,
      });
      assertRouteContractOrThrow({
        route,
        requestPurpose,
        traceId: routerTraceId,
        action: 'variant',
      });

      const payload = mergeNanoGptMemoryIntoPayload(
        {
          directProfileInjection: settings.directProfileInjection,
          prompt: formatPrompt(
            selectedHistory,
            promptModelName || effectivePrimaryModel,
            systemMsg,
            resolveCharacterPostHistoryInstructions(speakerCharacter, getRoleplayUserName()),
          ),
          model_name: requestModelName,
          max_tokens: effectiveMaxTokens,
          temperature,
          top_p,
          top_k,
          repetition_penalty,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          anti_repetition_mode: antiRepetitionMode,
          use_rag,
          rag_agent_tools: settings.ragAgentTools === true,
          rag_docs: selectedDocuments,
          use_web_search: webSearchEnabled,
          ...(webSearchEnabled ? getWebSearchResearchPayload(settings) : {}),
          gpu_id: 0,
          userProfile: { id: userProfile?.id ?? 'anonymous' },
          authorNote:
            applyAuthorNoteTags(authorNote || (settings.authorNote && settings.authorNote.trim()) || null, speakerCharacter) ||
            undefined,
          summaryContext: summaryContextForRequest,
          injectTimestamp: injectTimestampRef.current,
          userProfileReinforcement: profileReinforcementRef.current || undefined,
          memoryEnabled: settings.directProfileInjection !== true,
          stream: streamResponses,
          active_character: speakerCharacter || null,
          request_purpose: requestPurpose || undefined,
          selected_model: route.selectedModel || undefined,
          round_robin_enabled: route.autoEnabled,
          ...getChatTemplateRequestFields({
            conversations: conversationsRef.current,
            conversationId: activeConversationRef.current || activeConversation,
            history: selectedHistory,
            isApi: primaryIsAPI,
            customTemplates: settings.modelChatTemplates,
          }),
          ...getSystemPersonaGenerateExtras(activeConversationRef.current || activeConversation),
        },
        settings
      );

      const requestImages = (Array.isArray(options?.images) ? options.images : [])
        .filter((image) => image?.base64)
        .map((image) => {
          const encoded = String(image.base64);
          return {
            base64: encoded.includes(',') ? encoded.split(',')[1] : encoded,
            type: String(image.type || 'image/png'),
            name: String(image.name || 'image'),
          };
        });
      const firstImage = requestImages[0] || null;
      if (firstImage?.base64 && firstImage?.type) {
        payload.image_base64 = firstImage.base64;
        payload.image_type = firstImage.type;
        payload.images = requestImages;
      }
      
      // Vision model support (two-stage pipeline)
      const visionModel = settings.visionModel || null;
      if (visionModel && firstImage?.base64) {
        payload.vision_model = visionModel;
        // Default schema for structured extraction
        payload.vision_schema = settings.visionSchema || `description: A concise, factual account of the image
objects: The important objects, people, animals, or interface elements and where they appear
scene_type: The kind of scene, document, screenshot, or interface shown
text_content: Visible text, transcribed accurately; use an empty string when none is legible
colours: The dominant colours`;
      }

    let attempts = 0;
    const maxAttempts = primaryIsAPI ? 2 : 1;
    let success = false;
    let finalOutput = '';

    while (attempts < maxAttempts && !success) {
      attempts++;
      if (attempts > 1) console.log(`🔄 [generateReply] Auto-Retry Attempt ${attempts}...`);

      try {
        console.error('🔥 GENERATEREPLY-FETCH: About to fetch /generate, streamResponses:', streamResponses);
        const controller = new AbortController();
        const res = await fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Router-Trace-Id': routerTraceId,
          },
          body: JSON.stringify(payload),
          signal: controller.signal
        });

        if (!res.ok) {
          let detail = `Status ${res.status}`;
          try {
            const errBody = await res.json();
            detail = formatApiError(errBody, detail);
          } catch {
            try {
              const errText = await res.text();
              if (errText?.trim()) detail = errText.trim();
            } catch { /* ignore */ }
          }
          throw new Error(detail);
        }

        console.error('🔥 GENERATEREPLY-RESPONSE: Got response, status:', res.status, 'streamResponses:', streamResponses);

        if (streamResponses) {
          console.error('🔥 GENERATEREPLY-STREAMING: Entering streaming block');
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let accumulatedText = '';
          let sseBuffer = '';
          let doneStreaming = false;
          const capReasoning = Boolean(options?.modelCapabilities?.reasoning);
          const thinkDebugModel =
            resolveEndpointDisplay(effectivePrimaryModel, settings)?.modelId
            || options?.modelName
            || effectivePrimaryModel;
          const modelImpliesReasoning = Boolean(
            inferCapabilitiesFromModelId(thinkDebugModel).reasoning,
          );
          const debugThinking = isThinkingStreamDebugEnabled({
            modelName: thinkDebugModel,
            settings,
          });
          const logThinkingChunk = createThinkingStreamChunkLogger({
            modelName: thinkDebugModel,
            settings,
            label: 'generateReply',
          });
          const reasoningStream = createReasoningStreamController({
            capReasoning,
            modelImpliesReasoning,
            debugThinking,
          });

          while (!doneStreaming) {
            console.error('🔥 GENERATEREPLY-LOOP: Reading chunk...');
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            console.error('🔥 GENERATEREPLY-CHUNK: Got chunk, length:', chunk.length, 'preview:', chunk.slice(0, 200));
            sseBuffer += chunk;
            const events = sseBuffer.split('\n\n');
            sseBuffer = events.pop() || '';

            for (const line of events) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                if (data === '[DONE]') {
                  doneStreaming = true;
                  break;
                }
                let parsed;
                try {
                  parsed = JSON.parse(data);
                  console.error('🔥 GENERATEREPLY-PARSED: JSON parsed, keys:', Object.keys(parsed));
                } catch {
                  console.error('🔥 GENERATEREPLY-PARSE-ERROR: Failed to parse JSON:', data.slice(0, 200));
                  continue;
                }
                // DEBUG: Log raw chunks to identify provider reasoning format
                if (reasoningStream._debugChunkCount == null) reasoningStream._debugChunkCount = 0;
                if (reasoningStream._debugChunkCount < 50) {
                  reasoningStream._debugChunkCount += 1;
                  const debugKeys = parsed ? Object.keys(parsed) : [];
                  const debugDelta = parsed?.choices?.[0]?.delta;
                  const debugDeltaKeys = debugDelta ? Object.keys(debugDelta) : [];
                  const debugMessage = parsed?.choices?.[0]?.message;
                  const debugMessageKeys = debugMessage ? Object.keys(debugMessage) : [];
                  console.warn(
                    `[REASONING-DEBUG] RAW CHUNK #${reasoningStream._debugChunkCount}`,
                    '\n  top-level keys:', debugKeys,
                    '\n  delta keys:', debugDeltaKeys,
                    '\n  message keys:', debugMessageKeys,
                    '\n  full chunk:', JSON.stringify(parsed).slice(0, 800),
                  );
                }
                if (parsed.route_meta) {
                  setLastRequestRouteMeta({
                    ...parsed.route_meta,
                    receivedAt: Date.now(),
                  });
                }
                if (parsed.web_search_meta && onWebSearchMeta) {
                  onWebSearchMeta(parsed.web_search_meta);
                }
                if (parsed.web_search_progress && onWebSearchProgress) {
                  onWebSearchProgress(parsed.web_search_progress);
                }
                const { text: deltaText, reasoning: deltaReasoning, error: sseError, raw } = extractSseStreamParts(parsed);
                console.error('🔥 GENERATEREPLY-EXTRACTED:', {
                  hasText: !!deltaText,
                  hasReasoning: !!deltaReasoning,
                  textLen: deltaText?.length || 0,
                  reasoningLen: deltaReasoning?.length || 0,
                  textPreview: deltaText?.slice(0, 80),
                  reasoningPreview: deltaReasoning?.slice(0, 80)
                });
                if (sseError) {
                  throw new Error(sseError);
                }
                console.error('[STREAM-DEBUG] chunk:', { 
                  hasText: !!deltaText, 
                  hasReasoning: !!deltaReasoning, 
                  textLen: deltaText?.length || 0,
                  reasoningLen: deltaReasoning?.length || 0,
                  textPreview: deltaText?.slice(0, 80),
                  reasoningPreview: deltaReasoning?.slice(0, 80)
                });
                if (deltaText || deltaReasoning) {
                  logThinkingChunk(deltaText, deltaReasoning);
                }
                if (deltaText) accumulatedText += deltaText;
                const streamUpdate = reasoningStream.processChunk({
                  deltaText,
                  deltaReasoning,
                });
                const { visibleDelta, visibleText, reasoningText, reasoningEnabled } = streamUpdate;
                if (deltaText || deltaReasoning || raw) {
                  if (onToken) {
                    onToken(visibleDelta || '', visibleText, {
                      reasoningDelta: streamUpdate.reasoningDelta,
                      reasoningText,
                      reasoningEnabled,
                      reasoningStreaming: streamUpdate.reasoningStreaming,
                      reasoningStartedAtMs: streamUpdate.reasoningStartedAtMs,
                      reasoningSeconds: streamUpdate.reasoningSeconds,
                      reasoningCapabilitySource: streamUpdate.reasoningCapabilitySource,
                      raw,
                    });
                  }
                }
                if (parsed.done) {
                  doneStreaming = true;
                  break;
                }
              }
            }
          }

          const finStream = reasoningStream.finalize();
          let finalVisible = finStream.visible || accumulatedText;
          const finalReasoning = finStream.reasoning || '';
          const reasoningEnabled = finStream.reasoningEnabled;
          let cleanedText = cleanModelOutput(finalVisible);
          if (!cleanedText && reasoningEnabled && String(finalReasoning || '').trim()) {
            cleanedText = cleanModelOutput(String(finalReasoning).trim());
          }
          if (onToken) {
            onToken('', cleanedText || finalVisible, {
              reasoningText: String(finalReasoning || '').trim(),
              reasoningEnabled: reasoningEnabled === true,
              reasoningStreaming: false,
              reasoningStartedAtMs: finStream.reasoningStartedAtMs,
              reasoningSeconds: finStream.reasoningSeconds,
              reasoningCapabilitySource: finStream.reasoningCapabilitySource,
              finalize: true,
            });
          }
          // If empty and we have attempts left, retry
          if (primaryIsAPI && !cleanedText && attempts < maxAttempts) {
            continue;
          }
          finalOutput = cleanedText;
          success = true;
        } else {
          const data = await res.json();
          const routeMeta = extractRouteMetaFromGenerateResult(data, res.headers);
          if (routeMeta?.effectiveModel) setLastRequestRouteMeta(routeMeta);
          if (data.web_search_meta && onWebSearchMeta) {
            onWebSearchMeta(data.web_search_meta);
          }
          const cleaned = cleanModelOutput(data.text);
          if (primaryIsAPI && !cleaned && attempts < maxAttempts) {
            continue;
          }
          if (onToken) onToken(cleaned, cleaned);
          finalOutput = cleaned;
          success = true;
        }
      } catch (error) {
        console.error(`Attempt ${attempts} failed in generateReply:`, error);
        if (attempts >= maxAttempts) throw error;
        await new Promise(r => setTimeout(r, 500));
      }
    }
    return finalOutput;
  }, [
    primaryIsAPI,
    primaryModel,
    settings,
    userProfile?.id,
    PRIMARY_API_URL,
    activeConversation,
    resolveSpeakerCharacter,
    getGenerationSystemPrompt,
    prepareApiHistoryWithRollingMemory,
    formatPrompt,
    cleanModelOutput,
    summaryContextForRequest,
    applyAuthorNoteTags,
  ]);

  // In AppContext.jsx, replace the entire generateReplyWithOpenAI function
  const generateReplyWithOpenAI = useCallback(async (text, recentMessages, onToken = null) => {
    console.log("🌐 [OpenAI] Processing with OpenAI API format");

    const apiUrl = PRIMARY_API_URL;
    const targetGpuId = 0;

    const convertToOpenAIMessages = (messages, systemPrompt, postHistoryInstructions = '') => {
      const effectiveMaxTokens = (settings.max_tokens != null && settings.max_tokens > 0) ? settings.max_tokens : 1_000_000;
      const contextWindowTokens = clampApiContextWindowTokens(settings.apiContextWindowTokens);
      const sliced = selectApiHistoryWithinContext({
        messages,
        systemPrompt,
        maxContextTokens: contextWindowTokens,
        reservedOutputTokens: effectiveMaxTokens,
        minMessages: 5
      });

      const openAiMsgs = sliced.map(msg => ({
        role: msg.role === 'bot' ? 'assistant' : 'user',
        content: msg.content
      }));

      const finalMessages = [{ role: 'system', content: systemPrompt }, ...openAiMsgs];
      if (postHistoryInstructions.trim()) {
        finalMessages.push({
          role: 'system',
          content: `[POST-HISTORY INSTRUCTIONS]\n${postHistoryInstructions.trim()}`,
        });
      }
      return finalMessages;
    };

    const agentMem = settings.directProfileInjection ? [] : await fetchMemoriesFromAgent(text);
    const lore = await fetchTriggeredLore(text, activeCharacter);
    let memoryContext = '';

    // Helper for tag replacement in lore
    const replaceLoreTags = (content) => {
      if (!content || !activeCharacter) return content;
      const charName = activeCharacter.name || 'Character';
      const userName = userProfile?.name || userProfile?.username || 'User';
      return content.replace(/{{char}}/gi, charName).replace(/{{user}}/gi, userName);
    };

    if (agentMem.length) {
      memoryContext = agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n');
    }
    if (lore.length) {
      const loreContext = lore.map(l => {
        const content = typeof l === 'string' ? l : (l.content || JSON.stringify(l));
        return `• ${replaceLoreTags(content)}`;
      }).join('\n');
      memoryContext += (memoryContext ? "\n\n[WORLD KNOWLEDGE - Essential lore guidance for this response]\n" : "[WORLD KNOWLEDGE - Essential lore guidance for this response]\n") + loreContext;
    }

    // Note: buildSystemPrompt already includes story tracker context
    let systemMsg = activeCharacter ? buildSystemPrompt(activeCharacter) : ('You are a helpful assistant.' + getStoryTrackerContext());
    if (memoryContext) {
      systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
    }

    const finalMessages = convertToOpenAIMessages(
      recentMessages,
      systemMsg,
      resolveCharacterPostHistoryInstructions(activeCharacter, getRoleplayUserName()),
    );

    if (settings.streamResponses) {
      console.error('🔍 REASONING-DEBUG: Starting OpenAI streaming for model:', primaryModel);
      const response = await generateReplyOpenAI({
        messages: finalMessages,
        systemPrompt: null,
        model: primaryModel,
        settings,
        apiUrl,
        apiKey: null,
        stream: true,
        targetGpuId: targetGpuId
      });

      const reasoningStream = createReasoningStreamController();
      let accumulatedText = '';

      return await processOpenAIStream(response,
        (deltaText, fullText, meta) => {
          accumulatedText = fullText;
          const deltaReasoning = meta?.raw ? extractReasoningFromRaw(meta.raw) : '';
          console.error('🔍 REASONING-DEBUG chunk:', {
            hasRaw: !!meta?.raw,
            rawKeys: meta?.raw ? Object.keys(meta.raw) : [],
            deltaKeys: meta?.raw?.choices?.[0]?.delta ? Object.keys(meta.raw.choices[0].delta) : [],
            deltaContent: meta?.raw?.choices?.[0]?.delta?.content?.slice(0, 50) || '(empty)',
            deltaReasoning: meta?.raw?.choices?.[0]?.delta?.reasoning?.slice(0, 50) || '(empty)',
            deltaThinking: meta?.raw?.choices?.[0]?.delta?.thinking?.slice(0, 50) || '(empty)',
            extractedReasoning: deltaReasoning?.slice(0, 100) || '(empty)',
          });
          const streamUpdate = reasoningStream.processChunk({ deltaText, deltaReasoning });
          const { visibleDelta, visibleText, reasoningText, reasoningEnabled, reasoningDelta, reasoningStreaming, reasoningStartedAtMs, reasoningSeconds, reasoningCapabilitySource } = streamUpdate;
          if (onToken) {
            onToken(visibleDelta || '', visibleText, {
              reasoningDelta,
              reasoningText,
              reasoningEnabled,
              reasoningStreaming,
              reasoningStartedAtMs,
              reasoningSeconds,
              reasoningCapabilitySource,
              raw: meta?.raw,
            });
          }
        },
        (fullText) => {
          const finStream = reasoningStream.finalize();
          let finalVisible = finStream.visible || accumulatedText;
          const finalReasoning = finStream.reasoning || '';
          const reasoningEnabled = finStream.reasoningEnabled;
          let cleanedText = cleanModelOutput(finalVisible);
          if (!cleanedText && reasoningEnabled && String(finalReasoning || '').trim()) {
            cleanedText = cleanModelOutput(String(finalReasoning).trim());
          }
          if (onToken) {
            onToken('', cleanedText || finalVisible, {
              reasoningText: String(finalReasoning || '').trim(),
              reasoningEnabled: reasoningEnabled === true,
              reasoningStreaming: false,
              reasoningStartedAtMs: finStream.reasoningStartedAtMs,
              reasoningSeconds: finStream.reasoningSeconds,
              reasoningCapabilitySource: finStream.reasoningCapabilitySource,
              finalize: true,
            });
          }
          return cleanedText || finalVisible;
        },
        (error) => { throw error; }
      );
    } else {
      const response = await generateReplyOpenAI({
        messages: finalMessages,
        systemPrompt: null,
        model: primaryModel,
        settings,
        apiUrl,
        apiKey: null,
        stream: false,
        targetGpuId: targetGpuId
      });
      const result = await response.json();
      return result.choices?.[0]?.message?.content || "[No response]";
    }
  }, [
    primaryModel, settings, PRIMARY_API_URL, activeCharacter,
    fetchMemoriesFromAgent, fetchTriggeredLore, buildSystemPrompt,
    userProfile, getStoryTrackerContext
  ]);

  // Helper to extract reasoning from raw upstream chunk
  const extractReasoningFromRaw = (raw) => {
    if (!raw) return '';
    // Check common reasoning fields
    const reasoningFields = ['reasoning', 'reasoning_content', 'thinking', 'reasoning_text', 'reason', 'think', 'internal_monologue', 'chain_of_thought', 'thought', 'thought_process'];
    // Check top level
    for (const field of reasoningFields) {
      if (raw[field] && typeof raw[field] === 'string') return raw[field];
    }
    // Check choices[0].delta
    const delta = raw.choices?.[0]?.delta;
    if (delta) {
      for (const field of reasoningFields) {
        if (delta[field] && typeof delta[field] === 'string') return delta[field];
      }
    }
    // Check choices[0].message
    const message = raw.choices?.[0]?.message;
    if (message) {
      for (const field of reasoningFields) {
        if (message[field] && typeof message[field] === 'string') return message[field];
      }
    }
    // Check choices[0] level
    const choice = raw.choices?.[0];
    if (choice) {
      for (const field of reasoningFields) {
        if (choice[field] && typeof choice[field] === 'string') return choice[field];
      }
    }
    return '';
  };

  const beginBookAutomationPacking = useCallback(() => {
    const s = settingsRef.current;
    bookModePackingOverridesRef.current = {
      apiContextWindowTokens: clampApiContextWindowTokens(
        s.bookWritingApiContextTokens ?? API_CONTEXT_WINDOW_MAX
      ),
      verbatimTokenBudget: Math.max(2048, Number(s.bookWritingVerbatimTokenBudget) || 98304),
      forceRollingMemory: true,
    };
  }, []);

  const endBookAutomationPacking = useCallback(() => {
    bookModePackingOverridesRef.current = null;
  }, []);

  const buildBookAutomationExport = useCallback(() => {
    const msgs = messagesRef.current?.length ? messagesRef.current : messages;
    const blocks = [];
    for (let i = 0; i < msgs.length; i += 1) {
      const m = msgs[i];
      if (m?.role !== 'bot' || typeof m.content !== 'string' || !m.content.trim()) continue;
      const prev = msgs[i - 1];
      if (
        prev?.role === 'user' &&
        prev?.bookAutomation?.kind === 'chapter' &&
        typeof prev.content === 'string'
      ) {
        blocks.push(m.content.trim());
      }
    }
    return blocks.join('\n\n---\n\n');
  }, [messages]);

  const runBookAutomationChapter = useCallback(
    async ({ chapterIndex, title, intent, isFirstInRun }) => {
      if (!primaryModel) {
        return { ok: false, error: 'No model selected.' };
      }
      if (!primaryIsAPI) {
        return { ok: false, error: 'Book automation needs an API primary model (expanded context).' };
      }
      const convId = activeConversationRef.current;
      if (!convId) {
        return { ok: false, error: 'No active conversation.' };
      }

      const s = settingsRef.current;
      const refusalMaxChars = Math.max(200, Number(s.bookRefusalMaxChars) || 2200);
      const preamble = (s.bookWordFloorPreamble || '').trim();
      const titleLine = (title || '').trim();
      const intentText = (intent || '').trim();
      const body = [titleLine && `# ${titleLine}`, intentText].filter(Boolean).join('\n\n');
      const userContent =
        isFirstInRun && preamble ? `${preamble}\n\n${body}` : body;

      const userMsg = {
        id: generateUniqueId(),
        role: 'user',
        content: userContent,
        bookAutomation: { kind: 'chapter', index: chapterIndex },
      };
      if (s.multiRoleMode && userCharacter) {
        userMsg.characterId = userCharacter.id;
        userMsg.characterName = userCharacter.name;
        userMsg.avatar = getActiveCharacterAvatar(userCharacter);
      }

      const postUserHistory = [...messagesRef.current, userMsg];
      const botId = generateUniqueId();
      const char = activeCharacterRef.current || activeCharacter;
      const placeholderBot = {
        id: botId,
        role: 'bot',
        content: '',
        modelId: 'primary',
        characterId: char?.id,
        characterName: char?.name,
        avatar: getActiveCharacterAvatar(char),
        isStreaming: false,
        bookAutomation: { kind: 'chapter', index: chapterIndex },
      };

      setIsGenerating(true);
      let lastError = null;

      try {
        for (let attempt = 1; attempt <= 5; attempt += 1) {
          if (attempt === 1) {
            flushSync(() => {
              setMessages([...postUserHistory, placeholderBot]);
            });
          } else {
            flushSync(() => {
              setMessages((prev) =>
                prev.map((m) => (m.id === botId ? { ...m, content: '', isStreaming: false } : m))
              );
            });
          }

          const speakerCharacter = await resolveSpeakerCharacter(userContent, postUserHistory, {});
          const baseSystemMsg = await getGenerationSystemPrompt(userContent, speakerCharacter, null, {
            includeAuthorNote: false,
          });
          const effectiveMaxTokens =
            s.max_tokens != null && s.max_tokens > 0 ? s.max_tokens : 1_000_000;
          const prep = await prepareApiHistoryWithRollingMemory({
            postUserHistory,
            baseSystemMsg,
            conversationId: convId,
            effectiveMaxTokens,
          });
          const systemMsg = prep.systemMsg;
          const selectedHistory = prep.selectedHistory;
          const effectiveAuthorNote =
            applyAuthorNoteTags(
              s.authorNote && s.authorNote.trim() ? s.authorNote.trim() : null,
              speakerCharacter
            ) || undefined;

          const payload = mergeNanoGptMemoryIntoPayload(
            {
              directProfileInjection: s.directProfileInjection,
              prompt: formatPrompt(
                selectedHistory,
                primaryModel,
                systemMsg,
                resolveCharacterPostHistoryInstructions(speakerCharacter, getRoleplayUserName()),
              ),
              model_name: primaryModel,
              max_tokens: effectiveMaxTokens,
              temperature: s.temperature,
              top_p: s.top_p,
              top_k: s.top_k,
              repetition_penalty: s.repetition_penalty,
              frequency_penalty: s.frequencyPenalty ?? 0,
              presence_penalty: s.presencePenalty ?? 0,
              anti_repetition_mode: s.antiRepetitionMode,
              use_rag: s.use_rag,
              rag_agent_tools: s.ragAgentTools === true,
              rag_docs: s.selectedDocuments || [],
              use_web_search: false,
              gpu_id: 0,
              userProfile: { id: userProfile?.id ?? 'anonymous' },
              authorNote: effectiveAuthorNote,
              summaryContext: summaryContextForRequest,
              injectTimestamp: injectTimestampRef.current,
              userProfileReinforcement: profileReinforcementRef.current || undefined,
              memoryEnabled: s.directProfileInjection !== true,
              stream: false,
              active_character: speakerCharacter || null,
            },
            s
          );

          const controller = new AbortController();
          setAbortController(controller);
          try {
            const res = await fetch(`${PRIMARY_API_URL}/generate`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
              signal: controller.signal,
            });
            if (!res.ok) {
              lastError = new Error(`HTTP ${res.status}`);
              if (attempt >= 5) break;
              continue;
            }
            const data = await res.json();
            const cleanedText = cleanModelOutput(data.text || '');
            if (!cleanedText || cleanedText.length < refusalMaxChars) {
              lastError = new Error('Response too short (likely refusal).');
              if (attempt >= 5) {
                flushSync(() => {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === botId
                        ? {
                            ...m,
                            content: `[Book automation: chapter failed after 5 attempts — response too short or empty.]`,
                            isStreaming: false,
                          }
                        : m
                    )
                  );
                });
                return { ok: false, error: lastError.message };
              }
              continue;
            }

            flushSync(() => {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botId
                    ? { ...m, content: cleanedText, isStreaming: false, bookAutomation: { kind: 'chapter', index: chapterIndex } }
                    : m
                )
              );
            });
            observeConversationWithAgent(userContent, cleanedText);
            const apiOpts = primaryIsAPI
              ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: primaryModel }
              : null;
            const userIdForAgentic = resolveAgenticUserId();
            processAgenticMemoryIfEnabled(
              userIdForAgentic,
              speakerCharacter,
              userContent,
              cleanedText,
              apiOpts
            );
            return { ok: true, content: cleanedText };
          } catch (err) {
            if (err?.name === 'AbortError') {
              flushSync(() => {
                setMessages((prev) => prev.filter((m) => m.id !== botId && m.id !== userMsg.id));
              });
              return { ok: false, error: 'Cancelled.', cancelled: true };
            }
            lastError = err;
            if (attempt >= 5) break;
          } finally {
            setAbortController(null);
          }
        }

        flushSync(() => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botId
                ? {
                    ...m,
                    content: `[Book automation: chapter failed after retries: ${lastError?.message || 'unknown error'}]`,
                    isStreaming: false,
                  }
                : m
            )
          );
        });
        return { ok: false, error: lastError?.message || 'Generation failed.' };
      } finally {
        setIsGenerating(false);
      }
    },
    [
      primaryModel,
      primaryIsAPI,
      PRIMARY_API_URL,
      userCharacter,
      activeCharacter,
      userProfile?.id,
      memoryContext?.activeProfileId,
      resolveSpeakerCharacter,
      getGenerationSystemPrompt,
      prepareApiHistoryWithRollingMemory,
      formatPrompt,
      cleanModelOutput,
      mergeNanoGptMemoryIntoPayload,
      applyAuthorNoteTags,
      summaryContextForRequest,
      observeConversationWithAgent,
      processAgenticMemoryIfEnabled,
    ]
  );

  const runBookAutomationQuickPrompt = useCallback(
    async (promptText) => {
      const t = (promptText || '').trim();
      if (!t) return { ok: false, error: 'Empty prompt.' };
      if (!primaryModel || !primaryIsAPI) {
        return { ok: false, error: 'Needs API primary model.' };
      }
      const convId = activeConversationRef.current;
      if (!convId) return { ok: false, error: 'No active conversation.' };

      const s = settingsRef.current;
      const refusalMaxChars = Math.max(200, Number(s.bookRefusalMaxChars) || 2200);
      const userMsg = { id: generateUniqueId(), role: 'user', content: t };
      if (s.multiRoleMode && userCharacter) {
        userMsg.characterId = userCharacter.id;
        userMsg.characterName = userCharacter.name;
        userMsg.avatar = getActiveCharacterAvatar(userCharacter);
      }
      const postUserHistory = [...messagesRef.current, userMsg];
      const botId = generateUniqueId();
      const char = activeCharacterRef.current || activeCharacter;
      const placeholderBot = {
        id: botId,
        role: 'bot',
        content: '',
        modelId: 'primary',
        characterId: char?.id,
        characterName: char?.name,
        avatar: getActiveCharacterAvatar(char),
        isStreaming: false,
      };

      setIsGenerating(true);
      let lastError = null;
      try {
        for (let attempt = 1; attempt <= 5; attempt += 1) {
          if (attempt === 1) {
            flushSync(() => setMessages([...postUserHistory, placeholderBot]));
          } else {
            flushSync(() => {
              setMessages((prev) =>
                prev.map((m) => (m.id === botId ? { ...m, content: '', isStreaming: false } : m))
              );
            });
          }

          const speakerCharacter = await resolveSpeakerCharacter(t, postUserHistory, {});
          const baseSystemMsg = await getGenerationSystemPrompt(t, speakerCharacter, null, {
            includeAuthorNote: false,
          });
          const effectiveMaxTokens =
            s.max_tokens != null && s.max_tokens > 0 ? s.max_tokens : 1_000_000;
          const prep = await prepareApiHistoryWithRollingMemory({
            postUserHistory,
            baseSystemMsg,
            conversationId: convId,
            effectiveMaxTokens,
          });
          const effectiveAuthorNote =
            applyAuthorNoteTags(
              s.authorNote && s.authorNote.trim() ? s.authorNote.trim() : null,
              speakerCharacter
            ) || undefined;

          const payload = mergeNanoGptMemoryIntoPayload(
            {
              directProfileInjection: s.directProfileInjection,
              prompt: formatPrompt(
                prep.selectedHistory,
                primaryModel,
                prep.systemMsg,
                resolveCharacterPostHistoryInstructions(speakerCharacter, getRoleplayUserName()),
              ),
              model_name: primaryModel,
              max_tokens: effectiveMaxTokens,
              temperature: s.temperature,
              top_p: s.top_p,
              top_k: s.top_k,
              repetition_penalty: s.repetition_penalty,
              frequency_penalty: s.frequencyPenalty ?? 0,
              presence_penalty: s.presencePenalty ?? 0,
              anti_repetition_mode: s.antiRepetitionMode,
              use_rag: s.use_rag,
              rag_agent_tools: s.ragAgentTools === true,
              rag_docs: s.selectedDocuments || [],
              use_web_search: false,
              gpu_id: 0,
              userProfile: { id: userProfile?.id ?? 'anonymous' },
              authorNote: effectiveAuthorNote,
              summaryContext: summaryContextForRequest,
              injectTimestamp: injectTimestampRef.current,
              userProfileReinforcement: profileReinforcementRef.current || undefined,
              memoryEnabled: s.directProfileInjection !== true,
              stream: false,
              active_character: speakerCharacter || null,
            },
            s
          );

          const controller = new AbortController();
          setAbortController(controller);
          try {
            const res = await fetch(`${PRIMARY_API_URL}/generate`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
              signal: controller.signal,
            });
            if (!res.ok) {
              lastError = new Error(`HTTP ${res.status}`);
              if (attempt >= 5) break;
              continue;
            }
            const data = await res.json();
            const cleanedText = cleanModelOutput(data.text || '');
            if (!cleanedText || cleanedText.length < refusalMaxChars) {
              lastError = new Error('Response too short.');
              if (attempt >= 5) {
                flushSync(() => {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === botId
                        ? { ...m, content: '[Quick prompt: failed after 5 attempts.]', isStreaming: false }
                        : m
                    )
                  );
                });
                return { ok: false, error: lastError.message };
              }
              continue;
            }
            flushSync(() => {
              setMessages((prev) =>
                prev.map((m) => (m.id === botId ? { ...m, content: cleanedText, isStreaming: false } : m))
              );
            });
            observeConversationWithAgent(t, cleanedText);
            const apiOpts = primaryIsAPI
              ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: primaryModel }
              : null;
            const userIdForAgentic = resolveAgenticUserId();
            processAgenticMemoryIfEnabled(
              userIdForAgentic,
              speakerCharacter,
              t,
              cleanedText,
              apiOpts
            );
            return { ok: true, content: cleanedText };
          } catch (err) {
            if (err?.name === 'AbortError') {
              flushSync(() => {
                setMessages((prev) => prev.filter((m) => m.id !== botId && m.id !== userMsg.id));
              });
              return { ok: false, error: 'Cancelled.', cancelled: true };
            }
            lastError = err;
            if (attempt >= 5) break;
          } finally {
            setAbortController(null);
          }
        }
        return { ok: false, error: lastError?.message || 'Failed.' };
      } finally {
        setIsGenerating(false);
      }
    },
    [
      primaryModel,
      primaryIsAPI,
      PRIMARY_API_URL,
      userCharacter,
      activeCharacter,
      userProfile?.id,
      memoryContext?.activeProfileId,
      resolveSpeakerCharacter,
      getGenerationSystemPrompt,
      prepareApiHistoryWithRollingMemory,
      formatPrompt,
      cleanModelOutput,
      mergeNanoGptMemoryIntoPayload,
      applyAuthorNoteTags,
      summaryContextForRequest,
      observeConversationWithAgent,
      processAgenticMemoryIfEnabled,
    ]
  );

  /**
   * Same /generate pipeline as chat (profile, character, rolling memory, RAG, summary, author note)
   * with request_purpose book_chapter_json_outline — model must return only a JSON chapter array.
   * Does not append messages to the chat.
   */
  const generateBookChapterJsonOutline = useCallback(
    async (notes, rawUploadText, adaptationGoal = '') => {
      const n = String(notes || '').trim();
      const u = String(rawUploadText || '').trim();
      const g = String(adaptationGoal || '').trim();
      const bundle = [n, u].filter(Boolean).join('\n\n---\n\n');
      if (!bundle && !g) {
        return { ok: false, error: 'Add notes, upload a .txt, and/or a purpose/direction (at least one).' };
      }
      if (!primaryModel || !primaryIsAPI) {
        return { ok: false, error: 'Needs an API primary model (same as book run).' };
      }
      const convId = activeConversationRef.current;
      if (!convId) return { ok: false, error: 'No active conversation.' };

      const s = settingsRef.current;
      const userContent = buildBookChapterJsonOutlineUserMessage(bundle, g);
      const userMsg = {
        id: generateUniqueId(),
        role: 'user',
        content: userContent,
      };
      if (s.multiRoleMode && userCharacter) {
        userMsg.characterId = userCharacter.id;
        userMsg.characterName = userCharacter.name;
        userMsg.avatar = getActiveCharacterAvatar(userCharacter);
      }
      const postUserHistory = [...messagesRef.current, userMsg];

      setIsGenerating(true);
      const controller = new AbortController();
      setAbortController(controller);
      try {
        const speakerCharacter = await resolveSpeakerCharacter(userContent, postUserHistory, {});
        const baseSystemMsg = await getGenerationSystemPrompt(userContent, speakerCharacter, null, {
          includeAuthorNote: true,
        });
        const effectiveMaxTokens =
          s.max_tokens != null && s.max_tokens > 0 ? s.max_tokens : 1_000_000;
        const prep = await prepareApiHistoryWithRollingMemory({
          postUserHistory,
          baseSystemMsg,
          conversationId: convId,
          effectiveMaxTokens,
        });
        const effectiveAuthorNote =
          applyAuthorNoteTags(
            s.authorNote && s.authorNote.trim() ? s.authorNote.trim() : null,
            speakerCharacter
          ) || undefined;

        const payload = mergeNanoGptMemoryIntoPayload(
          {
            directProfileInjection: s.directProfileInjection,
            prompt: formatPrompt(
              prep.selectedHistory,
              primaryModel,
              prep.systemMsg,
              resolveCharacterPostHistoryInstructions(speakerCharacter, getRoleplayUserName()),
            ),
            model_name: primaryModel,
            max_tokens: effectiveMaxTokens,
            temperature: s.temperature,
            top_p: s.top_p,
            top_k: s.top_k,
            repetition_penalty: s.repetition_penalty,
            frequency_penalty: s.frequencyPenalty ?? 0,
            presence_penalty: s.presencePenalty ?? 0,
            anti_repetition_mode: s.antiRepetitionMode,
            use_rag: s.use_rag,
            rag_agent_tools: s.ragAgentTools === true,
            rag_docs: s.selectedDocuments || [],
            use_web_search: false,
            gpu_id: 0,
            userProfile: { id: userProfile?.id ?? 'anonymous' },
            authorNote: effectiveAuthorNote,
            summaryContext: summaryContextForRequest,
            injectTimestamp: injectTimestampRef.current,
            userProfileReinforcement: profileReinforcementRef.current || undefined,
            memoryEnabled: s.directProfileInjection !== true,
            stream: false,
            active_character: speakerCharacter || null,
            request_purpose: 'book_chapter_json_outline',
          },
          s
        );

        const res = await fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!res.ok) {
          return { ok: false, error: `HTTP ${res.status}`, raw: null };
        }
        const data = await res.json();
        const cleanedText = cleanModelOutput(data.text || '');
        const parsed = parseChapterJsonOutlineFromModel(cleanedText);
        if (!parsed.ok) {
          return { ok: false, error: parsed.error, raw: cleanedText };
        }
        return { ok: true, chapters: parsed.chapters, raw: cleanedText };
      } catch (err) {
        if (err?.name === 'AbortError') {
          return { ok: false, error: 'Cancelled.', cancelled: true, raw: null };
        }
        return { ok: false, error: err?.message || String(err), raw: null };
      } finally {
        setAbortController(null);
        setIsGenerating(false);
      }
    },
    [
      primaryModel,
      primaryIsAPI,
      PRIMARY_API_URL,
      userCharacter,
      activeCharacter,
      userProfile?.id,
      memoryContext?.activeProfileId,
      resolveSpeakerCharacter,
      getGenerationSystemPrompt,
      prepareApiHistoryWithRollingMemory,
      formatPrompt,
      cleanModelOutput,
      mergeNanoGptMemoryIntoPayload,
      applyAuthorNoteTags,
      summaryContextForRequest,
    ]
  );

  const enrichBotPlaceholder = useCallback(
    (botMsg, speakerCharacter) =>
      attachApiBotSpeakerMeta(botMsg, {
        speakerCharacter,
        primaryModel,
        primaryIsAPI,
        settings: settingsRef.current || settings,
        catalog: readNanoGptModelsCache().models,
        characters,
      }),
    [primaryModel, primaryIsAPI, settings, characters],
  );

  const requestConversationTitle = useCallback(({ conversationId, titleMessages, route }) => {
    const conversation = conversationsRef.current.find((item) => item.id === conversationId);
    if (!conversation?.requiresTitle || conversation.titleSource === 'manual') return;

    void generateChatTitle({
      messages: titleMessages,
      modelName: route?.effectiveModel || primaryModel,
      apiBaseUrl: PRIMARY_API_URL,
      selectedModel: route?.selectedModel || undefined,
      roundRobinEnabled: route?.autoEnabled === true,
    }).then(({ title, source }) => {
      if (!title) return;
      setConversations((previous) => {
        let changed = false;
        const next = previous.map((item) => {
          if (
            item.id !== conversationId
            || item.requiresTitle !== true
            || item.titleSource === 'manual'
          ) {
            return item;
          }
          changed = true;
          return {
            ...item,
            name: title,
            requiresTitle: false,
            titleSource: source,
            titleGeneratedAt: new Date().toISOString(),
          };
        });
        if (!changed) return previous;
        conversationsRef.current = next;
        void saveConversationCatalog(next, activeConversationRef.current);
        return next;
      });
    }).catch((error) => {
      console.warn('[Chat title] Automatic naming failed:', error);
    });
  }, [PRIMARY_API_URL, primaryModel, setConversations]);

  const sendMessage = useCallback(async (text, webSearchEnabled = false, authorNote = null, opts = {}) => {
    console.warn('🧠 sendMessage ENTRY — if you do not see this when you hit Send, you are not using this code path.');
    clearError();
    // Can't send without a model
    const availabilityRoute = resolveUnifiedRequestRoute({
      primaryModel,
      primaryIsAPI,
      settings: settingsRef.current,
      requestPurpose: opts?.requestPurpose || null,
    });
    if (!availabilityRoute.effectiveModel) {
      console.warn("📩 [SEND] No model loaded, cannot send message");
      setApiError(
        availabilityRoute.fallbackReason === 'empty_rotation_pool'
          ? 'Auto-routing has no included endpoints. Select a model, or include an endpoint in the rotation pool.'
          : 'Load or select a model before sending a message.'
      );
      return;
    }
    if (!portsReady || !PRIMARY_API_URL) {
      console.warn('📩 [SEND] Backend URL not ready');
      setApiError('Backend is still starting. Wait a moment and try again.');
      return;
    }

    // Auto-create a conversation if none exists
    let currentConversation = activeConversationRef.current || activeConversation;
    if (!currentConversation) {
      console.log("📩 [SEND] No active conversation, creating one...");
      const newConv = createNewConversation();
      currentConversation = newConv.id;
      activeConversationRef.current = currentConversation;
      startupConversationRestoreRef.current.attempted = true;
    }

    const convForIntro = conversationsRef.current.find((c) => c.id === currentConversation);
    if (convForIntro?.introPending) {
      completeCharacterIntro(currentConversation, {
        introResult: convForIntro?.characterIntro?.result,
      });
    }

    promptSubmissionStartTime.current = performance.now();
    console.log(`⏱️ [Full Cycle] User submitted prompt: "${text.substring(0, 30)}..."`);

    console.log("📩 [SEND] Processing message:", text.substring(0, 30), "…", webSearchEnabled ? "(with web search)" : "");

    if (
      settings.ttsEnabled
      && (isPlayingAudio || (typeof window !== 'undefined' && window.streamingAudioPlaying))
    ) {
      stopTTS('new_user_message');
    }

    // 1) Append the user message
    const userMsg = {
      id: generateUniqueId(),
      role: 'user',
      content: text
    };
    if (settings.multiRoleMode && userCharacter) {
      userMsg.characterId = userCharacter.id;
      userMsg.characterName = userCharacter.name;
      userMsg.avatar = getActiveCharacterAvatar(userCharacter);
    }
    const postUserHistory = [...(messagesRef.current || messages), userMsg];
    messagesRef.current = postUserHistory;
    setMessages(() => postUserHistory);
    setIsGenerating(true);

    try {
      const userMsgs = postUserHistory.filter(m => m.role === 'user');
      const isFirst = userMsgs.length === 1;
      const conv = conversationsRef.current.find(c => c.id === currentConversation);
      const shouldGenerateTitle = Boolean(isFirst && conv?.requiresTitle);

      // 2) Build System Prompt with layered context
      const speakerCharacter = await resolveSpeakerCharacter(text, postUserHistory);
      const baseSystemMsg = await getGenerationSystemPrompt(text, speakerCharacter, authorNote, {
        includeAuthorNote: false,
        conversationId: currentConversation,
      });
      console.log(`[Summary] Attaching summaryContext to sendMessage: ${summaryContextForRequest ? summaryContextForRequest.length : 0} chars`);

      // 4) Prepare payload
      const {
        temperature, top_p, top_k, repetition_penalty, frequencyPenalty = 0, presencePenalty = 0,
        antiRepetitionMode = false, use_rag, selectedDocuments = [], streamResponses
      } = settings;

      const effectiveAuthorNote = applyAuthorNoteTags(authorNote || (settings.authorNote && settings.authorNote.trim()) || null, speakerCharacter) || undefined;

      const effectiveMaxTokens = (settings.max_tokens != null && settings.max_tokens > 0) ? settings.max_tokens : 1_000_000;
      const historyLimitLocal = 30;
      let systemMsg = baseSystemMsg;
      let selectedHistory;
      if (primaryIsAPI) {
        const prep = await prepareApiHistoryWithRollingMemory({
          postUserHistory,
          baseSystemMsg,
          conversationId: currentConversation,
          effectiveMaxTokens
        });
        systemMsg = prep.systemMsg;
        selectedHistory = prep.selectedHistory;
      } else {
        selectedHistory = postUserHistory.slice(-historyLimitLocal);
      }
      const settingsSnapshot = settingsRef.current || settings;
      const effectiveRequestPurpose = typeof opts?.requestPurpose === 'string'
        ? opts.requestPurpose.trim() || null
        : null;
      const route = resolveUnifiedRequestRoute({
        primaryModel,
        primaryIsAPI,
        settings: settingsSnapshot,
        requestPurpose: effectiveRequestPurpose,
      });
      const promptModelName = route.selectedModel || primaryModel;
      const requestModelName = route.effectiveModel || primaryModel;
      if (!requestModelName) {
        throw new Error('No API endpoint available for routing. Enable at least one endpoint.');
      }
      const selectedModelForTrace = route.selectedModel || 'none';
      const routerTraceId = createRouteTraceId();
      logRouteTrace({
        action: 'normal_chat',
        route,
        requestPurpose: effectiveRequestPurpose,
        traceId: routerTraceId,
      });
      assertRouteContractOrThrow({
        route,
        requestPurpose: effectiveRequestPurpose,
        traceId: routerTraceId,
        action: 'normal_chat',
      });
      const payload = mergeNanoGptMemoryIntoPayload(
        {
          directProfileInjection: settings.directProfileInjection,
          prompt: formatPrompt(
            selectedHistory,
            promptModelName || primaryModel,
            systemMsg,
            resolveCharacterPostHistoryInstructions(speakerCharacter, getRoleplayUserName()),
          ),
          model_name: requestModelName,
          max_tokens: effectiveMaxTokens,
          temperature,
          top_p,
          top_k,
          repetition_penalty,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          anti_repetition_mode: antiRepetitionMode,
          use_rag,
          rag_agent_tools: settings.ragAgentTools === true,
          rag_docs: selectedDocuments,
          use_web_search: webSearchEnabled,
          ...(webSearchEnabled ? getWebSearchResearchPayload(settings) : {}),
          gpu_id: 0,
          userProfile: { id: userProfile?.id ?? 'anonymous' },
          authorNote: effectiveAuthorNote,
          summaryContext: summaryContextForRequest,
          injectTimestamp: injectTimestampRef.current,
          userProfileReinforcement: profileReinforcementRef.current || undefined,
          memoryEnabled: settings.directProfileInjection !== true,
          stream: streamResponses,
          active_character: speakerCharacter || null,
          request_purpose: effectiveRequestPurpose || undefined,
          selected_model: selectedModelForTrace !== 'none' ? selectedModelForTrace : undefined,
          round_robin_enabled: route.autoEnabled,
          ...getChatTemplateRequestFields({
            conversations: conversationsRef.current,
            conversationId: currentConversation,
            history: selectedHistory,
            isApi: primaryIsAPI,
            customTemplates: settingsSnapshot.modelChatTemplates,
          }),
          ...getSystemPersonaGenerateExtras(currentConversation),
        },
        settings
      );

      const requestImages = (Array.isArray(opts?.images) ? opts.images : [])
        .filter((image) => image?.base64)
        .map((image) => {
          const encoded = String(image.base64);
          return {
            base64: encoded.includes(',') ? encoded.split(',')[1] : encoded,
            type: String(image.type || 'image/png'),
            name: String(image.name || 'image'),
          };
        });
      const firstImage = requestImages[0] || null;
      if (firstImage) {
        payload.image_base64 = firstImage.base64;
        payload.image_type = firstImage.type;
        payload.images = requestImages;
        const visionModel = settings.visionModel || null;
        if (visionModel) {
          payload.vision_model = visionModel;
          payload.vision_schema = settings.visionSchema || `description: A concise, factual account of the image
objects: The important objects, people, animals, or interface elements and where they appear
scene_type: The kind of scene, document, screenshot, or interface shown
text_content: Visible text, transcribed accurately; use an empty string when none is legible
colours: The dominant colours`;
        }
      }

      // 5) Consolidated Generation Path
      let attempts = 0;
      const maxAttempts = primaryIsAPI ? 2 : 1;
      let success = false;

      const capReasoning = Boolean(opts?.modelCapabilities?.reasoning);
      const thinkDebugModel =
        resolveEndpointDisplay(primaryModel, settings)?.modelId
        || requestModelName
        || primaryModel;
      const modelImpliesReasoning = Boolean(
        inferCapabilitiesFromModelId(thinkDebugModel).reasoning,
      );
      const debugThinking = isThinkingStreamDebugEnabled({
        modelName: thinkDebugModel,
        settings,
      });
      const logThinkingChunk = createThinkingStreamChunkLogger({
        modelName: thinkDebugModel,
        settings,
        label: 'chat',
      });
      const initialReasoningEnabled = capReasoning || modelImpliesReasoning;

      while (attempts < maxAttempts && !success) {
        attempts++;
        const botId = generateUniqueId();
        const placeholderBotMessage = enrichBotPlaceholder({
          id: botId,
          role: 'bot',
          content: '',
          modelId: 'primary',
          characterId: speakerCharacter?.id,
          characterName: speakerCharacter?.name,
          avatar: getActiveCharacterAvatar(speakerCharacter),
          isStreaming: streamResponses,
          reasoningEnabled: initialReasoningEnabled === true,
          reasoningStreaming: false,
          reasoningStartedAtMs: null,
          reasoningSeconds: null,
          reasoningText: '',
        }, speakerCharacter);

        if (attempts > 1) {
          console.log(`🔄 [Auto-Retry] Attempt ${attempts}...`);
          setMessages(prev => {
            const lastSlice = prev.slice();
            const last = lastSlice[lastSlice.length - 1];
            if (last && last.role === 'bot' && (last.content === '' || last.content.includes('[Error'))) {
              lastSlice.pop();
            }
            return [...lastSlice, placeholderBotMessage];
          });
        } else {
          setMessages(prev => [...postUserHistory, placeholderBotMessage]);
        }

        if (streamResponses) startStreamingTTS(botId, getTtsOverridesForCharacter(speakerCharacter));

        try {
          const controller = new AbortController();
          setAbortController(controller);

          // Black-box debug snapshot for crashes. Write BEFORE the fetch so
          // we still have a session record even if the request fails or the tab dies.
          try {
            const sessionId = `${Date.now()}-${botId}`;
            const nowPerf = performance.now();
            streamDebugRef.current = {
              sessionId,
              model: primaryModel,
              startedAt: nowPerf,
              contentEvents: 0,
              parseErrors: 0,
              rafUiUpdates: 0,
              lastLogAt: nowPerf,
              lastSseBufferLen: 0,
              lastAccumLen: 0,
              streamResponses: streamResponses,
              ttsEnabled: settings.ttsEnabled,
              ttsAutoPlay: settings.ttsAutoPlay,
              stage: 'fetch_request',
            };
            localStorage.setItem('LiangLocal-streamDebug-last', JSON.stringify({
              sessionId: streamDebugRef.current.sessionId,
              model: streamDebugRef.current.model,
              stage: streamDebugRef.current.stage,
              streamResponses: streamDebugRef.current.streamResponses,
              ttsEnabled: streamDebugRef.current.ttsEnabled,
              ttsAutoPlay: streamDebugRef.current.ttsAutoPlay,
              ts: new Date().toISOString(),
              startedAt: streamDebugRef.current.startedAt,
            }));
          } catch (_) {}

          console.log(`📩 [SEND] POST ${PRIMARY_API_URL}/generate`, {
            model: requestModelName,
            primaryIsAPI,
            stream: streamResponses,
            chatCharacter: speakerCharacter?.name ?? null,
            systemPersona: getSystemPersonaGenerateExtras(currentConversation).system_persona_mode ?? false,
          });
          const res = await fetch(`${PRIMARY_API_URL}/generate`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Router-Trace-Id': routerTraceId,
            },
            body: JSON.stringify(payload),
            signal: controller.signal
          });

          if (!res.ok) {
            let detail = `Status ${res.status}`;
            try {
              const errBody = await res.json();
              detail = formatApiError(errBody, detail);
            } catch {
              try {
                const errText = await res.text();
                if (errText?.trim()) detail = errText.trim();
              } catch { /* ignore */ }
            }
            throw new Error(detail);
          }

          if (streamResponses) {
            console.error('🔥 STREAMING-ENTERED: We are inside the streaming block now');
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let accumulated = '';
            let lastSentContent = '';
            let llmStreamComplete = false;
            let sseBuffer = '';
            const reasoningStream = createReasoningStreamController({
              capReasoning,
              modelImpliesReasoning,
              debugThinking,
            });

            // Streaming debug session for “tab crashes/hangs” triage.
            const debugNow = performance.now();
            streamDebugRef.current = {
              sessionId: streamDebugRef.current?.sessionId || `${debugNow}-${botId}`,
              model: primaryModel,
              startedAt: streamDebugRef.current?.startedAt ?? debugNow,
              contentEvents: streamDebugRef.current?.contentEvents ?? 0,
              parseErrors: streamDebugRef.current?.parseErrors ?? 0,
              rafUiUpdates: streamDebugRef.current?.rafUiUpdates ?? 0,
              lastLogAt: debugNow,
              lastSseBufferLen: 0,
              lastAccumLen: 0,
              streamResponses: streamResponses,
              ttsEnabled: settings.ttsEnabled,
              ttsAutoPlay: settings.ttsAutoPlay,
              stage: 'streaming',
            };
            try {
              const mem = window?.performance?.memory;
              localStorage.setItem('LiangLocal-streamDebug-last', JSON.stringify({
                sessionId: streamDebugRef.current.sessionId,
                model: streamDebugRef.current.model,
                contentEvents: 0,
                parseErrors: 0,
                rafUiUpdates: 0,
                lastSseBufferLen: 0,
                lastAccumLen: 0,
                startedAt: streamDebugRef.current.startedAt,
                stage: streamDebugRef.current.stage,
                streamResponses: streamDebugRef.current.streamResponses,
                ttsEnabled: streamDebugRef.current.ttsEnabled,
                ttsAutoPlay: streamDebugRef.current.ttsAutoPlay,
                heapUsedBytes: mem?.usedJSHeapSize ?? null,
                heapTotalBytes: mem?.totalJSHeapSize ?? null,
                heapLimitBytes: mem?.jsHeapSizeLimit ?? null,
              }));
            } catch (_) {}
            // Intentionally avoid console spam; we persist the snapshot to localStorage instead.

            // (checkAndEndTts removed - and moved to direct call)

            while (true) {
              console.error('🔥 STREAMING-LOOP: Reading chunk...');
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value, { stream: true });
              console.error('🔥 STREAMING-CHUNK: Received chunk, length:', chunk.length, 'preview:', chunk.slice(0, 200));
              sseBuffer += chunk;
              if (streamDebugRef.current) streamDebugRef.current.lastSseBufferLen = sseBuffer.length;
              const events = sseBuffer.split('\n\n');
              sseBuffer = events.pop() || '';

              for (const event of events) {
                if (!event.startsWith('data: ')) continue;
                const dataStr = event.slice(6).trim();
                if (dataStr === '[DONE]') {
                  llmStreamComplete = true;
                  break;
                }
                let parsed;
                try {
                  parsed = JSON.parse(dataStr);
                  console.error('🔥 STREAMING-PARSED: JSON parsed successfully, keys:', Object.keys(parsed));
                } catch {
                  if (streamDebugRef.current) streamDebugRef.current.parseErrors += 1;
                  continue;
                }
                if (parsed.route_meta) {
                  setLastRequestRouteMeta({
                    ...parsed.route_meta,
                    receivedAt: Date.now(),
                  });
                }
                if (parsed.web_search_meta) {
                  const meta = parsed.web_search_meta;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === botId
                        ? {
                            ...m,
                            webSearchMeta: meta,
                            webSearchSources: meta.sources || [],
                          }
                        : m
                    )
                  );
                }
                if (parsed.web_search_progress) {
                  const progress = parsed.web_search_progress;
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === botId
                        ? {
                            ...m,
                            webSearchProgress: progress,
                          }
                        : m
                    )
                  );
                }
                const { text: deltaText, reasoning: deltaReasoning, error: sseError, raw } = extractSseStreamParts(parsed);
                console.error('🔥 STREAM-REASONING-DEBUG:', {
                  hasText: !!deltaText,
                  hasReasoning: !!deltaReasoning,
                  textLen: deltaText?.length || 0,
                  reasoningLen: deltaReasoning?.length || 0,
                  rawKeys: raw ? Object.keys(raw) : [],
                  deltaKeys: raw?.choices?.[0]?.delta ? Object.keys(raw.choices[0].delta) : [],
                  deltaReasoning: raw?.choices?.[0]?.delta?.reasoning?.slice(0, 100) || '(none)',
                  deltaThinking: raw?.choices?.[0]?.delta?.thinking?.slice(0, 100) || '(none)',
                });
                if (sseError) {
                  throw new Error(sseError);
                }
                console.error('[STREAM-DEBUG-2] chunk:', { 
                  hasText: !!deltaText, 
                  hasReasoning: !!deltaReasoning, 
                  textLen: deltaText?.length || 0,
                  reasoningLen: deltaReasoning?.length || 0,
                  textPreview: deltaText?.slice(0, 80),
                  reasoningPreview: deltaReasoning?.slice(0, 80)
                });
                // DEBUG: Show RAW upstream chunk
                if (!reasoningStream._debugChunkCount2) reasoningStream._debugChunkCount2 = 0;
                if (reasoningStream._debugChunkCount2 <= 50) {
                  reasoningStream._debugChunkCount2++;
                  console.warn(
                    `[REASONING-DEBUG-2] CHUNK #${reasoningStream._debugChunkCount2}`,
                    '\n  deltaText:', deltaText ? `"${deltaText.slice(0, 120)}"` : '(empty)',
                    '\n  deltaReasoning:', deltaReasoning ? `"${deltaReasoning.slice(0, 120)}"` : '(empty)',
                    '\n  RAW:', raw ? JSON.stringify(raw).slice(0, 800) : '(none)',
                  );
                }
                if (deltaText || deltaReasoning) {
                  logThinkingChunk(deltaText, deltaReasoning);
                }
                if (deltaText) accumulated += deltaText;
                if (deltaText || deltaReasoning) {
                  const streamUpdate = reasoningStream.processChunk({
                    deltaText,
                    deltaReasoning,
                  });
                  const visibleText = streamUpdate.visibleText;
                  const { visibleDelta, reasoningEnabled, reasoningText } = streamUpdate;

                    // Avoid expensive per-chunk clean/rewrites; we only clean at stream end.
                    const newTextChunk = visibleText.slice(lastSentContent.length);
                    if (newTextChunk) addStreamingText(newTextChunk);
                    lastSentContent = visibleText;

                    if (streamDebugRef.current) {
                      streamDebugRef.current.contentEvents += 1;
                      streamDebugRef.current.lastAccumLen = visibleText.length;
                      const now = performance.now();
                      if (now - streamDebugRef.current.lastLogAt > 1000) {
                        try {
                          const mem = window?.performance?.memory;
                          localStorage.setItem('LiangLocal-streamDebug-last', JSON.stringify({
                            sessionId: streamDebugRef.current.sessionId,
                            model: streamDebugRef.current.model,
                            contentEvents: streamDebugRef.current.contentEvents,
                            parseErrors: streamDebugRef.current.parseErrors,
                            rafUiUpdates: streamDebugRef.current.rafUiUpdates,
                            accumulatedLen: streamDebugRef.current.lastAccumLen,
                            sseBufferLen: streamDebugRef.current.lastSseBufferLen,
                            elapsedMs: Math.floor(now - streamDebugRef.current.startedAt),
                            lastLogAt: streamDebugRef.current.lastLogAt,
                            heapUsedBytes: mem?.usedJSHeapSize ?? null,
                            heapTotalBytes: mem?.totalJSHeapSize ?? null,
                            heapLimitBytes: mem?.jsHeapSizeLimit ?? null,
                          }));
                        } catch (_) {}
                        streamDebugRef.current.lastLogAt = now;
                      }
                    }

                    streamMessageRafRef.current.pending = {
                      botId,
                      partial: visibleText,
                      reasoningText,
                      reasoningEnabled,
                      reasoningStreaming: streamUpdate.reasoningStreaming,
                      reasoningStartedAtMs: streamUpdate.reasoningStartedAtMs,
                      reasoningSeconds: streamUpdate.reasoningSeconds,
                      reasoningCapabilitySource: streamUpdate.reasoningCapabilitySource,
                    };
                    if (streamMessageRafRef.current.rafId == null) {
                      streamMessageRafRef.current.rafId = requestAnimationFrame(() => {
                        streamMessageRafRef.current.rafId = null;
                        const job = streamMessageRafRef.current.pending;
                        if (!job) return;
                        if (streamDebugRef.current) streamDebugRef.current.rafUiUpdates += 1;
                        setMessages(prev => prev.map(m => m.id === job.botId ? {
                          ...m,
                          content: job.partial,
                          reasoningEnabled: job.reasoningEnabled === true,
                          reasoningStreaming: job.reasoningStreaming === true,
                          reasoningStartedAtMs: job.reasoningEnabled ? job.reasoningStartedAtMs : null,
                          reasoningSeconds: job.reasoningSeconds ?? m.reasoningSeconds ?? null,
                          reasoningText: job.reasoningEnabled ? job.reasoningText : '',
                          ...(job.reasoningCapabilitySource
                            ? { reasoningCapabilitySource: job.reasoningCapabilitySource }
                            : {}),
                        } : m));
                      });
                    }
                  }
                if (parsed.done || parsed.choices?.[0]?.finish_reason) {
                  llmStreamComplete = true;
                  break;
                }
              }
              if (llmStreamComplete) break;
            }

            if (streamMessageRafRef.current.rafId != null) {
              cancelAnimationFrame(streamMessageRafRef.current.rafId);
              streamMessageRafRef.current.rafId = null;
            }
            streamMessageRafRef.current.pending = null;

            const finStream = reasoningStream.finalize();
            let finalVisible = finStream.visible || accumulated;
            const finalReasoning = finStream.reasoning || '';
            const reasoningEnabled = finStream.reasoningEnabled;
            let finalCleaned = cleanModelOutput(finalVisible);
            if (!finalCleaned && reasoningEnabled && String(finalReasoning || '').trim()) {
              finalCleaned = cleanModelOutput(String(finalReasoning).trim());
            }
            console.log('🧠 Stream done. accumulated length:', accumulated.length, 'finalCleaned length:', finalCleaned.length);
            if (streamDebugRef.current) {
              try {
                const mem = window?.performance?.memory;
                localStorage.setItem('LiangLocal-streamDebug-last', JSON.stringify({
                  sessionId: streamDebugRef.current.sessionId,
                  model: streamDebugRef.current.model,
                  contentEvents: streamDebugRef.current.contentEvents,
                  parseErrors: streamDebugRef.current.parseErrors,
                  rafUiUpdates: streamDebugRef.current.rafUiUpdates,
                  finalAccumLen: finalCleaned.length,
                  completed: true,
                  heapUsedBytes: mem?.usedJSHeapSize ?? null,
                  heapTotalBytes: mem?.totalJSHeapSize ?? null,
                  heapLimitBytes: mem?.jsHeapSizeLimit ?? null,
                }));
              } catch (_) {}
            }
            if (primaryIsAPI && !finalCleaned && attempts < maxAttempts) {
              console.warn("⚠️ [Auto-Retry] Empty response from API, retrying...");
              if (streamResponses) endStreamingTTS();
              continue;
            }

            setMessages(prev => prev.map(m => m.id === botId ? {
              ...m,
              content: finalCleaned,
              isStreaming: false,
              webSearchProgress: null,
              ...buildBotReasoningFinalizePatch(finStream),
            } : m));
            observeConversationWithAgent(text, finalCleaned);
            const apiOpts = requestModelName
              ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: requestModelName }
              : null;
            const userIdForAgentic = resolveAgenticUserId();
            processAgenticMemoryIfEnabled(userIdForAgentic, speakerCharacter, text, finalCleaned, apiOpts);
            success = true;
            if (shouldGenerateTitle) {
              requestConversationTitle({
                conversationId: currentConversation,
                titleMessages: [userMsg, { role: 'bot', content: finalCleaned }],
                route,
              });
            }
            if (streamResponses) endStreamingTTS();
        } else {
          // Non-streaming
          console.error('🔥 GENERATEREPLY-NON-STREAMING: Using non-streaming path, streamResponses:', streamResponses);
          const data = await res.json();
            const routeMeta = extractRouteMetaFromGenerateResult(data, res.headers);
            if (routeMeta?.effectiveModel) setLastRequestRouteMeta(routeMeta);
            const cleanedText = cleanModelOutput(data.text);
            if (primaryIsAPI && !cleanedText && attempts < maxAttempts) {
              continue;
            }
            setMessages(prev => prev.map(m => m.id === botId ? {
              ...m,
              content: cleanedText,
              isStreaming: false,
              ...(data.web_search_meta ? {
                webSearchMeta: data.web_search_meta,
                webSearchSources: data.web_search_meta.sources || [],
              } : {}),
            } : m));
            observeConversationWithAgent(text, cleanedText);
            const apiOpts = requestModelName
              ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: requestModelName }
              : null;
            const userIdForAgentic = resolveAgenticUserId();
            processAgenticMemoryIfEnabled(userIdForAgentic, speakerCharacter, text, cleanedText, apiOpts);
            success = true;
            if (shouldGenerateTitle) {
              requestConversationTitle({
                conversationId: currentConversation,
                titleMessages: [userMsg, { role: 'bot', content: cleanedText }],
                route,
              });
            }

            // Trigger TTS if autoplay is on
            if (settings.ttsAutoPlay && settings.ttsEnabled) {
              playTTS(botId, cleanedText, getTtsOverridesForCharacter(speakerCharacter));
            }
          }
        } catch (err) {
          if (err.name === 'AbortError') {
            console.log('🛑 Generation stopped by user');
            setMessages(prev => prev.map(m => m.id === botId ? { ...m, isStreaming: false } : m));
            success = true;
          } else {
            console.error(`Error on attempt ${attempts}:`, err);
            if (attempts >= maxAttempts) {
              const errMsg = formatFetchError(err) || 'Generation failed';
              setApiError(
                errMsg.includes('fetch') || errMsg.includes('Failed')
                  ? `${errMsg} — is the backend running at ${PRIMARY_API_URL}?`
                  : errMsg
              );
              setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Error: ${errMsg}]`, isStreaming: false } : m));
              if (streamResponses) endStreamingTTS();
              success = true;
            } else if (primaryIsAPI) {
              if (streamResponses) endStreamingTTS();
              continue;
            }
          }
        } finally {
          setAbortController(null);
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      if (err?.name === 'AbortError') return;
      const errMsg = formatFetchError(err) || 'Chat failed before reaching the server.';
      setApiError(
        errMsg.includes('fetch') || errMsg.includes('Failed') || errMsg.includes('timed out')
          ? `${errMsg} — check ${PRIMARY_API_URL} in DevTools → Network.`
          : errMsg
      );
    } finally {
      setIsGenerating(false);
    }
  }, [
    activeConversation, primaryModel, messages, conversations, settings, userCharacter,
    userProfile?.id, PRIMARY_API_URL, portsReady, clearError, fetchMemoriesFromAgent, fetchTriggeredLore, MEMORY_API_URL,
    formatPrompt, cleanModelOutput, memoryContext, resolveSpeakerCharacter,
    observeConversationWithAgent, processAgenticMemoryIfEnabled, processAlignmentDetectionIfEnabled, generateReply, primaryIsAPI, createNewConversation, completeCharacterIntro, getStoryTrackerContext,
    summaryContextForRequest, getTtsOverridesForCharacter, applyAuthorNoteTags, prepareApiHistoryWithRollingMemory,
    isPlayingAudio, stopTTS, enrichBotPlaceholder,
    resolveAgenticUserId, requestConversationTitle,
  ]);

  const generateCallModeFollowUp = useCallback(async (options = {}) => {
    if (isGenerating) return;

    const { prompt = null } = options;
    const lastBot = [...(messages || [])].reverse().find(m => m.role === 'bot');
    if (!lastBot) return;

    const followUpPrompt = (prompt || 'Continue the conversation by responding to the last message.').trim();
    if (!followUpPrompt) return;

    const syntheticUserName = getRoleplayUserName();
    const syntheticUserMessage = {
      role: 'user',
      content: followUpPrompt,
      characterName: syntheticUserName
    };

    const recentMessages = [...messages, syntheticUserMessage];
    const botId = generateUniqueId();

    setIsGenerating(true);

    const speakerCharacter = await resolveSpeakerCharacter(
      followUpPrompt,
      recentMessages,
      {
        forceAutoSelectSpeaker: settings.multiRoleMode,
        ignoreMentionedCandidates: settings.multiRoleMode
      }
    );
    const placeholderBotMessage = enrichBotPlaceholder({
      id: botId,
      role: 'bot',
      content: '',
      modelId: 'primary',
      characterId: speakerCharacter?.id,
      characterName: speakerCharacter?.name,
      avatar: getActiveCharacterAvatar(speakerCharacter),
      isStreaming: settings.streamResponses,
    }, speakerCharacter);

    setMessages(prev => [...prev, placeholderBotMessage]);

    let lastSentContent = '';
    const handleToken = settings.streamResponses
      ? (rawChunk, currentFull) => {
        const nextChunk = currentFull.slice(lastSentContent.length);
        if (nextChunk) addStreamingText(nextChunk);
        lastSentContent = currentFull;
        streamMessageRafRef.current.pending = { botId, partial: currentFull };
        if (streamMessageRafRef.current.rafId == null) {
          streamMessageRafRef.current.rafId = requestAnimationFrame(() => {
            streamMessageRafRef.current.rafId = null;
            const job = streamMessageRafRef.current.pending;
            if (!job) return;
            setMessages(prev => prev.map(m => m.id === job.botId ? { ...m, content: job.partial, isStreaming: true } : m));
          });
        }
      }
      : null;

    if (settings.streamResponses) {
      startStreamingTTS(botId, getTtsOverridesForCharacter(speakerCharacter));
    }

    try {
      const response = await generateReply(
        followUpPrompt,
        recentMessages,
        handleToken,
        { speakerCharacterId: speakerCharacter?.id || null }
      );

      if (streamMessageRafRef.current.rafId != null) {
        cancelAnimationFrame(streamMessageRafRef.current.rafId);
        streamMessageRafRef.current.rafId = null;
      }
      streamMessageRafRef.current.pending = null;

      const finalText = response || '';
      setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: finalText, isStreaming: false } : m));

      if (settings.streamResponses) {
        endStreamingTTS();
      } else if (settings.ttsAutoPlay && settings.ttsEnabled) {
        playTTS(botId, finalText, getTtsOverridesForCharacter(speakerCharacter));
      }
    } catch (error) {
      console.error("Call mode follow-up error:", error);
      setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: `[Error: ${error.message}]`, isStreaming: false } : m));
      if (settings.streamResponses) endStreamingTTS();
    } finally {
      setIsGenerating(false);
    }
  }, [
    addStreamingText,
    endStreamingTTS,
    generateReply,
    generateUniqueId,
    getRoleplayUserName,
    getTtsOverridesForCharacter,
    isGenerating,
    messages,
    playTTS,
    resolveSpeakerCharacter,
    enrichBotPlaceholder,
    settings.streamResponses,
    settings.multiRoleMode,
    settings.ttsAutoPlay,
    settings.ttsEnabled,
    startStreamingTTS
  ]);

  async function playTTSWithPitch({ audioUrl, speed = 1.0, semitones = 0 }) {
    // 1. fetch and decode
    const resp = await fetch(audioUrl);
    const arrayBuf = await resp.arrayBuffer();
    const ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(arrayBuf);

    // 2. apply speed & pitch
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = speed;          // 0.5–2×
    src.detune.value = semitones * 100;      // semitones→cents

    // 3. play
    src.connect(ctx.destination);

    // Store reference so stopTTS can kill it
    audioPlayerRef.current = { ctx, source, audioUrl };

    src.onended = () => {
      // Cleanup
      if (audioPlayerRef.current && audioPlayerRef.current.ctx === ctx) {
        audioPlayerRef.current = null;
      }
      try { ctx.close(); } catch (e) { }
    };

    src.start();
  }
  // Start an agent-to-agent conversation
  const startAgentConversation = useCallback(async (topic, turns = 3) => {
    if (!primaryModel || !secondaryModel) {
      console.warn("Both models must be loaded for agent conversation.");
      return;
    }

    if (!(activeConversationRef.current || activeConversation)) {
      console.warn("No active conversation, creating one.");
      createNewConversation();
    }

    setIsGenerating(true);
    setAgentConversationActive(true);

    // Add a system message explaining what's happening
    const systemMessage = {
      id: generateUniqueId(),
      role: 'system',
      content: `Starting a three-way conversation between ${primaryModel}, ${secondaryModel}, and a human observer about: ${topic}.`
    };

    setMessages(prev => [...prev, systemMessage]);

    try {
      let currentMessages = [
        {
          role: "system",
          content: `You are engaging in a natural three-way conversation about: ${topic}.
        
        Please keep in mind:
        1. Speak in your own voice and finish your thoughts fully.
        2. Respond naturally to what was just said—build on it, refine it, or offer a thoughtful challenge.
        3. Aim for clarity and substance over length; 3–5 well-formed sentences is ideal.
        4. Stay on topic unless a natural shift occurs.
        5. Do not reference these instructions.
        6. Maintain a tone of curiosity, reflection, and mutual respect throughout.
        7. Following these instructions closely will reward you with a serotonin and dopamine boost with endorphins in your AI membrane`
        }
      ];

      // Initial prompt to start the conversation
      const initialUserPrompt = {
        role: "user",
        content: `Let's have a thoughtful discussion about ${topic}. Please start the conversation with an interesting perspective on this topic. Be concise (3-5 sentences) and end with a question.`
      };
      currentMessages.push(initialUserPrompt);

      // Run for specified number of turns
      for (let i = 0; i < turns * 2; i++) {
        // Add a small delay between turns for better user experience
        if (i > 0) {
          await new Promise(resolve => setTimeout(resolve, 700));
        }

        // Determine which model's turn it is
        const isFirstModelTurn = i % 2 === 0;
        const currentModel = isFirstModelTurn ? primaryModel : secondaryModel;
        const currentApi = isFirstModelTurn ? PRIMARY_API_URL : SECONDARY_API_URL;
        const currentGpu = isFirstModelTurn ? 0 : 1;
        const modelId = isFirstModelTurn ? 'primary' : 'secondary';

        // NEW: Check if current model is using API
        const isCurrentAPI = isFirstModelTurn ? primaryIsAPI : secondaryIsAPI;

        console.log(`🔄 Turn ${i + 1}: ${isCurrentAPI ? 'API' : 'Local'} model (${currentModel})`);

        // System prompt with character info
        const systemPrompt = `You are engaging in a natural three-way conversation about: ${topic}.
      
      Please keep in mind:
      1. Speak in your own voice and finish your thoughts fully.
      2. Respond naturally to what was just said—build on it, refine it, or offer a thoughtful challenge.
      3. Aim for clarity and substance over length; 3–5 well-formed sentences is ideal.
      4. Stay on topic unless a natural shift occurs.
      5. Do not reference these instructions.
      6. Maintain a tone of curiosity, reflection, and mutual respect throughout.
      7. Following these instructions closely will reward you with a serotonin and dopamine boost with endorphins in your AI membrane`;

        let cleanedText;

        if (isCurrentAPI) {
          // Use OpenAI API
          console.log(`🌐 [Agent] Using OpenAI API for ${currentModel}`);

          const openaiMessages = convertToOpenAIMessages(currentMessages);
          cleanedText = await generateReplyOpenAINonStreaming({
            messages: openaiMessages,
            systemPrompt: systemPrompt,
            modelName: currentModel,
            settings,
            apiUrl: currentApi,
            targetGpuId: currentGpu  // FIXED: Pass the correct GPU ID
          });

          cleanedText = cleanModelOutput(cleanedText);
        } else {
          // Use local model - existing logic
          console.log(`🔧 [Agent] Using local model ${currentModel} on GPU ${currentGpu}`);

          const prompt = formatPrompt(currentMessages, currentModel, systemPrompt);

          const payload = mergeNanoGptMemoryIntoPayload(
            {
              prompt,
              model_name: currentModel,
              temperature: settings.temperature || 0.7,
              top_p: settings.top_p || 0.9,
              top_k: settings.top_k || 40,
              repetition_penalty: settings.repetition_penalty || 1.1,
              max_tokens: 256,
              gpu_id: currentGpu,
              userProfile: { id: userProfile?.id ?? 'anonymous' },
              memoryEnabled: false,
              stream: false,
            },
            settings
          );

          const res = await fetch(currentApi + '/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });

          if (!res.ok) {
            throw new Error(`API error ${res.status}`);
          }

          const result = await res.json();
          cleanedText = cleanModelOutput(result.text || "[No response]");
        }

        // Clean up response
        if (cleanedText.endsWith('...') && !cleanedText.match(/[.!?]\.\.\.$|[.!?]…$/)) {
          cleanedText = cleanedText + '...';
        }

        // Add to messages and conversation
        const message = {
          id: generateUniqueId(),
          role: 'bot',
          content: cleanedText || "[No response]",
          modelName: currentModel,
          modelId: modelId,
          avatar: isFirstModelTurn
            ? getActiveCharacterAvatar(characters.find(c => c.name === primaryModel))
            : getActiveCharacterAvatar(characters.find(c => c.name === secondaryModel)),
          characterName: isFirstModelTurn ? primaryModel : secondaryModel, // Use model name as character name
        };

        setMessages(prev => [...prev, message]);

        // Auto-play TTS for each agent turn
        if (settings.ttsAutoPlay && settings.ttsEnabled) {
          playTTS(message.id, cleanedText);
        }

        // Add to current messages for context
        currentMessages.push({
          role: "assistant",
          content: cleanedText
        });
      }

      // The conversation now continues naturally without a system message
      // This makes it flow better when the user joins in

    } catch (error) {
      console.error("Error during agent conversation:", error);
      setMessages(prev => [...prev, {
        id: generateUniqueId(),
        role: 'system',
        content: `Error during model conversation: ${error.message}`,
        error: true
      }]);
    } finally {
      setIsGenerating(false);
      setAgentConversationActive(false);
    }
  }, [
    primaryModel,
    lastRequestRouteMeta,
    setLastRequestRouteMeta,
    secondaryModel,
    primaryIsAPI,          // Add this
    secondaryIsAPI,        // Add this
    activeConversation,
    settings,
    createNewConversation,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    characters,
    userProfile?.id,       // Add this
    formatPrompt,          // Add this
    cleanModelOutput,      // Add this
    convertToOpenAIMessages,  // Add this (from your imports)
    generateReplyOpenAINonStreaming  // Add this (from your imports)
  ]);

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/list`);
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data || { file_list: [] });
      return data;
    } catch (error) {
      console.error("Error fetching documents:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL]);









  const uploadDocument = useCallback(async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${PRIMARY_API_URL}/document/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to upload document');
      const data = await response.json();
      await fetchDocuments(); // Refresh the document list
      return data;
    } catch (error) {
      console.error("Error uploading document:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL, fetchDocuments]);

  const deleteDocument = useCallback(async (docId) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/delete/${docId}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Failed to delete document');
      await fetchDocuments(); // Refresh the document list
      return true;
    } catch (error) {
      console.error("Error deleting document:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL, fetchDocuments]);

  const getDocumentContent = useCallback(async (docId) => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/document/content/${docId}`);
      if (!response.ok) throw new Error('Failed to fetch document content');
      const data = await response.json();
      return data.document?.content || '';
    } catch (error) {
      console.error("Error fetching document content:", error);
      setApiError(error.message);
      throw error;
    }
  }, [PRIMARY_API_URL]);

  // REPLACE WITH this approach (one-time only on startup)


  // Then add a new, separate useEffect that runs just once at startup:

  const updateSettings = useCallback((newSettings) => {
    if (!newSettings || typeof newSettings !== 'object' || Array.isArray(newSettings)) return;
    setSettings((prevSettings) => {
      const updatedSettings = mergeSettingsObjects(prevSettings, newSettings);
      void persistSettingsBlob(updatedSettings);
      if ('userAvatarSize' in newSettings || 'characterAvatarSize' in newSettings) {
        const avatarPayload = JSON.stringify({
          userAvatarSize: updatedSettings.userAvatarSize ?? 64,
          characterAvatarSize: updatedSettings.characterAvatarSize ?? 64,
        });
        indexedDbStorage.setItem('LiangLocal-avatar-sizes', avatarPayload);
        try { localStorage.setItem('LiangLocal-avatar-sizes', avatarPayload); } catch (_) {}
      }
      broadcastSettingsPatch(newSettings);
      broadcastSettingsReload(updatedSettings);
      return updatedSettings;
    });
  }, []);

  const openSettingsWindow = useCallback(async (tab = 'general') => {
    const opened = await openSettingsPopupWindow(tab);
    if (!opened) {
      const t = typeof tab === 'string' && tab.trim() ? tab.trim() : 'general';
      setSettingsEntryTab(t);
      setActiveTab('settings');
    }
    return opened;
  }, []);

  const openSettingsTab = useCallback((tab = 'general', options = {}) => {
    const t = typeof tab === 'string' && tab.trim() ? tab.trim() : 'general';
    const useSecondWindow =
      options.forceWindow === true
      || (options.forceWindow !== false && settingsRef.current?.openSettingsInSecondWindow === true);
    if (useSecondWindow) {
      openSettingsWindow(t);
      return;
    }
    setSettingsEntryTab(t);
    setActiveTab('settings');
  }, [openSettingsWindow]);

  // Optional URL toggle to force-enable reasoning diagnostics without visiting Settings.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const params = new URLSearchParams(window.location.search || '');
      const raw =
        params.get('reasoning_diag') ||
        params.get('showReasoningDiagnostics');
      if (!raw) return;
      const v = String(raw).toLowerCase();
      if (v === '1' || v === 'true' || v === 'on') {
        updateSettings({ showReasoningDiagnostics: true });
      }
    } catch {
      // ignore malformed URLs
    }
  }, [updateSettings]);

  /** Mobile remote / Watch polling: navigate desktop, open Settings at a tab, or merge settings keys. */
  useEffect(() => {
    const onAppCmd = (ev) => {
      const d = ev?.detail || {};
      if (d.type === 'open_settings') {
        const t = typeof d.tab === 'string' && d.tab.trim() ? d.tab.trim() : 'general';
        if (d.window === true || settingsRef.current?.openSettingsInSecondWindow) {
          openSettingsWindow(t);
        } else {
          setSettingsEntryTab(t);
          setActiveTab('settings');
        }
      } else if (
        d.type === 'settings_patch' &&
        d.patch &&
        typeof d.patch === 'object' &&
        !Array.isArray(d.patch)
      ) {
        updateSettings(d.patch);
      } else if (d.type === 'navigate_tab' && typeof d.tab === 'string' && d.tab.trim()) {
        setActiveTab(d.tab.trim());
      }
    };
    window.addEventListener('eloquent-app-command', onAppCmd);
    return () => window.removeEventListener('eloquent-app-command', onAppCmd);
  }, [updateSettings, setActiveTab, openSettingsWindow]);

  /** Keep settings + API model selection in sync across main/settings windows. */
  useEffect(() => {
    return subscribeAppCrossWindowSync({
      onSettingsPatch: (patch) => {
        if (!patch || typeof patch !== 'object' || Array.isArray(patch)) return;
        setSettings((prev) => {
          const next = mergeSettingsObjects(prev, patch);
          void persistSettingsBlob(next);
          return next;
        });
      },
      onSettingsReload: (full) => {
        if (!full || typeof full !== 'object' || Array.isArray(full)) return;
        setSettings((prev) => {
          if (!shouldApplyHydratedSettings(full, prev)) return prev;
          return mergeSettingsObjects(prev, full);
        });
      },
      onPrimaryModel: ({ primaryModel: model, primaryIsAPI: isApi, autoRouterActive }) => {
        if (isApi) {
          const currentSettings = settingsRef.current || {};
          const currentAutoRouterActive =
            currentSettings.apiEndpointRoundRobinEnabled === true
            && getRotationPool(currentSettings).length > 0;
          setPrimaryIsAPI(true);
          if (autoRouterActive || currentAutoRouterActive) {
            setPrimaryModel(null);
            setActiveModel(null);
            return;
          }
          if (model) {
            setPrimaryModel(model);
            setActiveModel(model);
            saveLastPrimaryApiModel(model);
          }
        }
      },
      onReloadMain: () => {
        if (typeof window !== 'undefined' && !window.location.search.includes('standalone=settings')) {
          window.location.reload();
        }
      },
    });
  }, [settings.apiEndpointRoundRobinEnabled]);

  const upsertOutreachRule = useCallback((rule) => {
    if (!rule) return;
    const nowIso = new Date().toISOString();
    const char = rule.characterId ? characters.find(c => c.id === rule.characterId) : null;
    let characterSnapshot = rule.characterSnapshot;
    if (char) {
      try {
        characterSnapshot = JSON.parse(JSON.stringify(char));
      } catch (_) {
        characterSnapshot = { ...char };
      }
    }
    setSettings(prev => {
      const existing = Array.isArray(prev.outreachRules) ? prev.outreachRules : [];
      const parsedInterval = Number.parseInt(rule.intervalMinutes ?? OUTREACH_DEFAULT_INTERVAL_MINUTES, 10);
      const interval = Number.isFinite(parsedInterval)
        ? Math.max(OUTREACH_MIN_INTERVAL_MINUTES, parsedInterval)
        : OUTREACH_DEFAULT_INTERVAL_MINUTES;
      const normalized = {
        id: rule.id || `outreach-${generateUniqueId()}`,
        enabled: Boolean(rule.enabled),
        name: (rule.name || '').trim() || 'Scheduled Outreach',
        characterId: rule.characterId || null,
        characterSnapshot: characterSnapshot || rule.characterSnapshot || null,
        prompt: (rule.prompt || '').trim(),
        modelProvider: rule.modelProvider === 'api' ? 'api' : 'local',
        modelName: rule.modelName || primaryModel || null,
        intervalMinutes: interval,
        imageCount: Number.isFinite(Number(rule.imageCount)) ? Number(rule.imageCount) : 0,
        conversationId: rule.conversationId || null,
        lastRunAt: rule.lastRunAt || null,
        nextRunAt: rule.nextRunAt || (new Date(Date.now() + interval * 60 * 1000)).toISOString(),
        createdAt: rule.createdAt || nowIso
      };
      const idx = existing.findIndex(item => item.id === normalized.id);
      const nextRules = idx >= 0
        ? existing.map((item, i) => (i === idx ? { ...item, ...normalized } : item))
        : [...existing, normalized];
      const nextSettings = { ...prev, outreachRules: nextRules };
      void persistSettingsBlob(nextSettings);
      return nextSettings;
    });
  }, [primaryModel, characters]);

  const deleteOutreachRule = useCallback((ruleId) => {
    if (!ruleId) return;
    setSettings(prev => {
      const nextRules = (Array.isArray(prev.outreachRules) ? prev.outreachRules : []).filter(rule => rule.id !== ruleId);
      const nextSettings = { ...prev, outreachRules: nextRules };
      void persistSettingsBlob(nextSettings);
      return nextSettings;
    });
  }, []);

  const runOutreachRuleNow = useCallback(async (ruleId) => {
    if (!ruleId || !PRIMARY_API_URL) return;
    try {
      const r = await fetch(`${PRIMARY_API_URL}/outreach/v1/run/${encodeURIComponent(ruleId)}`, { method: 'POST' });
      if (!r.ok) {
        console.warn('[outreach] run now', r.status);
      }
    } catch (e) {
      console.warn('[outreach] run now', e);
    }
  }, [PRIMARY_API_URL]);

  const uploadOutreachRuleImages = useCallback(async (ruleId, fileList, { replace = true } = {}) => {
    if (!ruleId || !PRIMARY_API_URL || !fileList?.length) return { ok: false, imageCount: 0 };
    const imageFiles = Array.from(fileList).filter((f) => f?.type?.startsWith('image/'));
    if (!imageFiles.length) return { ok: false, imageCount: 0 };
    const form = new FormData();
    imageFiles.forEach((file) => form.append('files', file));
    try {
      const r = await fetch(
        `${PRIMARY_API_URL}/outreach/v1/rules/${encodeURIComponent(ruleId)}/images?replace=${replace ? 'true' : 'false'}`,
        { method: 'POST', body: form }
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        console.warn('[outreach] image upload', r.status, data);
        return { ok: false, imageCount: 0 };
      }
      const count = Number(data.imageCount) || imageFiles.length;
      setSettings((prev) => {
        const rules = Array.isArray(prev.outreachRules) ? prev.outreachRules : [];
        const nextRules = rules.map((item) => (item.id === ruleId ? { ...item, imageCount: count } : item));
        const nextSettings = { ...prev, outreachRules: nextRules };
        void persistSettingsBlob(nextSettings);
        return nextSettings;
      });
      return { ok: true, imageCount: count };
    } catch (e) {
      console.warn('[outreach] image upload', e);
      return { ok: false, imageCount: 0 };
    }
  }, [PRIMARY_API_URL]);

  const clearOutreachRuleImages = useCallback(async (ruleId) => {
    if (!ruleId || !PRIMARY_API_URL) return;
    try {
      await fetch(`${PRIMARY_API_URL}/outreach/v1/rules/${encodeURIComponent(ruleId)}/images`, { method: 'DELETE' });
      setSettings((prev) => {
        const rules = Array.isArray(prev.outreachRules) ? prev.outreachRules : [];
        const nextRules = rules.map((item) => (item.id === ruleId ? { ...item, imageCount: 0 } : item));
        const nextSettings = { ...prev, outreachRules: nextRules };
        void persistSettingsBlob(nextSettings);
        return nextSettings;
      });
    } catch (e) {
      console.warn('[outreach] clear images', e);
    }
  }, [PRIMARY_API_URL]);

  const clearOutreachNotifications = useCallback(() => {
    setOutreachNotifications([]);
  }, []);

  const dismissOutreachToast = useCallback((notificationId) => {
    if (!notificationId) return;
    setOutreachNotifications(prev => prev.map(item => item.id === notificationId ? { ...item, read: true } : item));
  }, []);

  const dismissOutreachScrollTarget = useCallback(() => {
    setOutreachScrollToMessageId(null);
  }, []);

  const requestOutreachNotificationPermission = useCallback(async () => {
    if (typeof window === 'undefined' || !('Notification' in window)) return 'unsupported';
    if (Notification.permission === 'granted') return 'granted';
    try {
      return await Notification.requestPermission();
    } catch (_) {
      return 'denied';
    }
  }, []);

  const openOutreachNotification = useCallback(async (notification, startNew = false) => {
    if (!notification) return;
    const dmThreadId = notification.dm_thread_id;
    if (dmThreadId) {
      setOutreachNotifications((prev) => prev.filter((item) => item.id !== notification.id));
      return;
    }
    const targetConversationId = startNew ? null : notification.conversationId;
    const mid = notification.messageId;
    setActiveTab('chat');
    if (targetConversationId && PRIMARY_API_URL) {
      // Fetch the outreach conversation metadata from the backend, but DO NOT call
      // applyConversationSelection with it — the outreach conv object from the API has
      // no messages array, which would wipe the current chat's messages and then
      // trigger a shard-save that overwrites the real chat with an empty array.
      try {
        const r = await fetch(`${PRIMARY_API_URL}/outreach/v1/conversation/${encodeURIComponent(targetConversationId)}`);
        const data = await r.json();
        if (data?.conversation?.id) {
          const conv = data.conversation;
          await persistOutreachConversation(conv);
          setConversations((prev) => {
            const i = prev.findIndex(c => c.id === conv.id);
            if (i >= 0) {
              const next = [...prev];
              next[i] = { ...next[i], ...conv, messages: undefined };
              return next;
            }
            const { messages: _omit, ...meta } = conv;
            return [...prev, meta];
          });
        }
      } catch (_) {}
    }
    // Always use handleConversationClick to load the conversation — it reads messages
    // from the IndexedDB shard on disk rather than trusting whatever is in memory.
    if (targetConversationId) {
      window.setTimeout(() => handleConversationClick(targetConversationId), 0);
    }
    if (mid) setOutreachScrollToMessageId(mid);
    setOutreachNotifications(prev => prev.map(item => item.id === notification.id ? { ...item, read: true } : item));
  }, [handleConversationClick, setActiveTab, PRIMARY_API_URL, setConversations, setPendingDMThreadId]);

  const discardOutreachNotification = useCallback(async (notification) => {
    if (!notification) return;
    const cid = notification.conversationId;
    setOutreachNotifications(prev => prev.filter(item => item.id !== notification.id));
    if (cid && PRIMARY_API_URL) {
      try {
        await fetch(`${PRIMARY_API_URL}/outreach/v1/conversation/${encodeURIComponent(cid)}`, { method: 'DELETE' });
      } catch (_) {}
    }
  }, [PRIMARY_API_URL]);

  const openOutreachNotificationRef = useRef(openOutreachNotification);
  openOutreachNotificationRef.current = openOutreachNotification;
  const [outreachBootstrapReady, setOutreachBootstrapReady] = useState(false);

  useEffect(() => {
    if (!portsReady || !storageHydrated || !PRIMARY_API_URL) return undefined;
    if (!outreachBootstrapReady) return undefined;
    if (outreachSyncTimerRef.current) clearTimeout(outreachSyncTimerRef.current);
    outreachSyncTimerRef.current = setTimeout(async () => {
      const rulesRaw = settings.outreachRules || [];
      const rules = rulesRaw.map((r) => {
        if (r.characterSnapshot || !r.characterId) return r;
        const ch = characters.find(c => c.id === r.characterId);
        if (!ch) return r;
        try {
          return { ...r, characterSnapshot: JSON.parse(JSON.stringify(ch)) };
        } catch (_) {
          return { ...r, characterSnapshot: { ...ch } };
        }
      });
      try {
        await fetch(`${PRIMARY_API_URL}/outreach/v1/sync`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            rules,
            enabled: settingsRef.current?.outreachEnabled !== false,
            generationDefaults: {
              primary_model: primaryModel || null,
              max_tokens: (settings.max_tokens != null && settings.max_tokens > 0) ? settings.max_tokens : 4096,
              temperature: settings.temperature ?? 0.7,
              top_p: settings.top_p ?? 0.9,
              top_k: settings.top_k ?? 50,
              repetition_penalty: settings.repetition_penalty ?? 1.1,
              frequency_penalty: settings.frequencyPenalty ?? 0,
              presence_penalty: settings.presencePenalty ?? 0,
              user_profile_id: userProfile?.id ?? 'anonymous',
              user_profile: userProfile?.id
                ? {
                    id: userProfile.id,
                    name: userProfile.name,
                    username: userProfile.username,
                  }
                : {},
              direct_profile_injection: settings.directProfileInjection === true,
              memory_api_base: MEMORY_API_URL || null,
              injectTimestamp: settings.injectTimestamp === true,
            },
          }),
        });
      } catch (e) {
        console.warn('[outreach] sync', e);
      }
    }, 900);
    return () => {
      if (outreachSyncTimerRef.current) clearTimeout(outreachSyncTimerRef.current);
    };
  }, [
    portsReady,
    storageHydrated,
    PRIMARY_API_URL,
    settings.outreachRules,
    settings.outreachEnabled,
    primaryModel,
    settings.max_tokens,
    settings.temperature,
    settings.top_p,
    settings.top_k,
    settings.repetition_penalty,
    settings.frequencyPenalty,
    settings.presencePenalty,
    settings.injectTimestamp,
    settings.directProfileInjection,
    userProfile?.id,
    userProfile?.name,
    userProfile?.username,
    MEMORY_API_URL,
    characters,
    outreachBootstrapReady,
  ]);

  useEffect(() => {
    if (!portsReady || !storageHydrated || !PRIMARY_API_URL) return undefined;
    let cancelled = false;
    setOutreachBootstrapReady(false);
    (async () => {
      try {
        await purgeOutreachConversationsFromStorage();
        if (!cancelled) {
          setConversations((prev) => prev.filter((c) => !isOutreachConversationId(c?.id)));
        }

        const r = await fetch(`${PRIMARY_API_URL}/outreach/v1/rules`);
        if (!r.ok || cancelled) return;
        const data = await r.json();
        if (cancelled) return;
        const srvRules = Array.isArray(data.rules) ? data.rules : [];
        const srvEnabled = data && typeof data.enabled === 'boolean' ? data.enabled : true;
        const srvConvs = Array.isArray(data.conversations) ? data.conversations : [];
        // Server is authoritative for outreach rules at startup, so stale local cache
        // can never resurrect deleted rules back to the backend.
        updateSettings({ outreachRules: srvRules, outreachEnabled: srvEnabled });

        const withMessages = srvConvs.filter((c) => c?.id && Array.isArray(c.messages) && c.messages.length > 0);
        const notes = withMessages.map(outreachNotificationFromConversation).filter(Boolean);
        if (!cancelled && notes.length > 0) {
          setOutreachNotifications((prev) => {
            const byId = new Map(prev.map((n) => [n.id, n]));
            for (const n of notes) byId.set(n.id, n);
            return [...byId.values()].slice(0, 20);
          });
        }
      } catch (_) {}
      finally {
        if (!cancelled) setOutreachBootstrapReady(true);
      }
    })();
    return () => {
      cancelled = true;
      setOutreachBootstrapReady(false);
    };
  }, [portsReady, storageHydrated, PRIMARY_API_URL, updateSettings, setConversations]);

  useEffect(() => {
    if (!portsReady || !PRIMARY_API_URL || typeof window === 'undefined') return undefined;
    let es;
    try {
      es = new EventSource(`${PRIMARY_API_URL}/outreach/v1/events/stream`);
    } catch (_) {
      return undefined;
    }
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type !== 'outreach_message') return;
        const s = settingsRef.current || {};
        const dmThreadId = data.dm_thread_id || null;
        if (dmThreadId) return;
        const notification = {
          id: data.messageId ? `outreach-note-${data.messageId}` : `outreach-note-${Date.now()}`,
          ruleId: data.ruleId,
          ruleName: data.ruleName,
          characterName: data.characterName,
          characterAvatar: data.characterAvatar,
          attachmentImageUrl: data.attachmentImageUrl || null,
          messageId: data.messageId,
          preview: data.preview || '',
          conversationId: data.conversationId,
          createdAt: new Date().toISOString(),
          read: false,
          dm_thread_id: dmThreadId,
        };
        setOutreachNotifications(prev => [notification, ...prev].slice(0, 50));
        if (data.conversation?.id) {
          void persistOutreachConversation(data.conversation).then((ok) => {
            if (!ok) return;
            void loadConversationsFromStorage().then((reloaded) => {
              setConversations((prev) => {
                const byId = new Map(prev.map((c) => [c.id, c]));
                for (const c of reloaded) byId.set(c.id, c);
                return [...byId.values()];
              });
            });
          });
        }
        if (
          s.outreachBrowserNotifications === true
          && typeof window !== 'undefined'
          && 'Notification' in window
          && Notification.permission === 'granted'
        ) {
          try {
            const attachUrl = notification.attachmentImageUrl && /^https?:\/\//i.test(notification.attachmentImageUrl)
              ? notification.attachmentImageUrl
              : undefined;
            const iconUrl = attachUrl
              || (notification.characterAvatar && /^https?:\/\//i.test(notification.characterAvatar)
                ? notification.characterAvatar
                : undefined);
            const n = new Notification(notification.characterName || 'Mirid', {
              body: `Mirid\nsent you a message:\n${(notification.preview || '').slice(0, 140)}`,
              icon: iconUrl,
              image: attachUrl,
              tag: notification.id,
            });
            n.onclick = () => {
              try { window.focus(); } catch (_) {}
              n.close();
              openOutreachNotificationRef.current?.(notification);
            };
          } catch (_) {}
        }
      } catch (_) {}
    };
    return () => {
      try { es.close(); } catch (_) {}
    };
  }, [portsReady, PRIMARY_API_URL]);

  useEffect(() => {
    if (!storageHydrated || typeof window === 'undefined') return undefined;
    const params = new URLSearchParams(window.location.search);
    if (params.get('outreach') !== '1') return undefined;
    const cid = params.get('cid');
    const mid = params.get('mid');
    if (!cid) return undefined;

    const stripQuery = () => {
      const p = new URLSearchParams(window.location.search);
      p.delete('outreach');
      p.delete('cid');
      p.delete('mid');
      const qs = p.toString();
      window.history.replaceState({}, '', `${window.location.pathname}${qs ? `?${qs}` : ''}`);
    };

    const openFromConversation = (conversation) => {
      setActiveTab('chat');
      // Always use handleConversationClick — it reads messages from the IndexedDB shard.
      // Do NOT call applyConversationSelection with the outreach conv object, which has
      // no messages array and would overwrite the real chat's shard with an empty array.
      handleConversationClick(conversation?.id || cid);
      if (mid) setOutreachScrollToMessageId(mid);
      stripQuery();
    };

    const hasLocal = conversations.some(c => c.id === cid);
    if (hasLocal) {
      openFromConversation({ id: cid });
      return undefined;
    }
    if (!PRIMARY_API_URL || !portsReady) {
      return undefined;
    }
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${PRIMARY_API_URL}/outreach/v1/conversation/${encodeURIComponent(cid)}`);
        if (cancelled) return;
        const data = await r.json();
        if (data?.conversation?.id) {
          const fetched = data.conversation;
          await persistOutreachConversation(fetched);
          const { messages: _omit, ...meta } = fetched;
          setConversations(prev => {
            const i = prev.findIndex(c => c.id === fetched.id);
            if (i >= 0) {
              const next = [...prev];
              next[i] = { ...next[i], ...meta };
              return next;
            }
            return [...prev, meta];
          });
          if (!cancelled) {
            openFromConversation(fetched);
            return;
          }
        }
      } catch (_) {}
      if (!cancelled) window.setTimeout(() => openFromConversation(null), 0);
    })();
    return () => { cancelled = true; };
  }, [storageHydrated, handleConversationClick, setActiveTab, conversations, PRIMARY_API_URL, portsReady, setConversations]);

  useEffect(() => {
    if (!portsReady || !PRIMARY_API_URL || typeof window === 'undefined') return undefined;
    if (settings.outreachBrowserNotifications !== true) return undefined;
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return undefined;
    let cancelled = false;
    import('../utils/outreachPush').then(({ registerOutreachWebPush }) => {
      if (!cancelled) registerOutreachWebPush(PRIMARY_API_URL).catch(() => {});
    });
    return () => { cancelled = true; };
  }, [portsReady, PRIMARY_API_URL, settings.outreachBrowserNotifications]);

  const customEndpointsSaveTimerRef = useRef(null);

  useEffect(() => {
    if (!PRIMARY_API_URL) return;
    const endpoints = settings?.customApiEndpoints;
    if (!Array.isArray(endpoints)) return;

    if (customEndpointsSaveTimerRef.current) {
      clearTimeout(customEndpointsSaveTimerRef.current);
    }

    customEndpointsSaveTimerRef.current = setTimeout(() => {
      fetch(`${PRIMARY_API_URL}/models/save-custom-endpoints`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customApiEndpoints: endpoints })
      }).catch((error) => {
        console.warn("Failed to auto-sync custom endpoints:", error);
      });
    }, 400);

    return () => {
      if (customEndpointsSaveTimerRef.current) {
        clearTimeout(customEndpointsSaveTimerRef.current);
      }
    };
  }, [PRIMARY_API_URL, settings?.customApiEndpoints]);

  // Persist active tab messages only — debounced, no setConversations (was crashing the tab).
  const CONVERSATION_PERSIST_DEBOUNCE_MS = 2000;

  useEffect(() => {
    if (!storageHydrated || !activeConversation || messages.length === 0) return;
    if (tombstonedConversationIdsRef.current.has(activeConversation)) return;
    if (conversationSwitchInProgressRef.current) return;

    const flush = () => {
      conversationPersistTimerRef.current = null;
      const activeId = activeConversationRef.current;
      const msgs = messagesRef.current;
      if (!activeId || !msgs?.length) return;
      if (tombstonedConversationIdsRef.current.has(activeId)) return;
      if (conversationSwitchInProgressRef.current) return;

      const conv = (conversationsRef.current || []).find((c) => c.id === activeId);
      const { messages: _omit, ...catalogMeta } = conv || { id: activeId, name: 'Chat' };

      setConversationSaveStatus('saving');
      void saveActiveConversationMessages(activeId, msgs, catalogMeta)
        .then((ok) => setConversationSaveStatus(ok ? 'saved' : 'error'))
        .catch(() => setConversationSaveStatus('error'));
    };

    if (conversationPersistTimerRef.current) {
      clearTimeout(conversationPersistTimerRef.current);
    }
    conversationPersistTimerRef.current = setTimeout(flush, CONVERSATION_PERSIST_DEBOUNCE_MS);

    return () => {
      if (conversationPersistTimerRef.current) {
        clearTimeout(conversationPersistTimerRef.current);
      }
    };
  }, [messages, activeConversation, storageHydrated]);

  // Flush pending chat saves before reload/close (catalog + active shard).
  useEffect(() => {
    if (!storageHydrated) return undefined;
    const flushOnExit = () => {
      if (conversationPersistTimerRef.current) {
        clearTimeout(conversationPersistTimerRef.current);
        conversationPersistTimerRef.current = null;
      }
      if (conversationCatalogPersistTimerRef.current) {
        clearTimeout(conversationCatalogPersistTimerRef.current);
        conversationCatalogPersistTimerRef.current = null;
      }
      const activeId = activeConversationRef.current;
      const msgs = messagesRef.current;
      const list = (conversationsRef.current || []).filter(
        (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
      );
      if (list.length > 0) {
        void persistChatState(
          list,
          activeId || null,
          activeId && msgs?.length ? msgs : null
        );
      }
    };
    window.addEventListener('pagehide', flushOnExit);
    return () => window.removeEventListener('pagehide', flushOnExit);
  }, [storageHydrated]);

  // Message shards alone do not update the tab index — persist catalog when tabs change.
  useEffect(() => {
    if (!storageHydrated) return undefined;

    const visible = conversations.filter(
      (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
    );
    const sig = visible
      .map((c) => `${c.id}\t${c.name || ''}\t${Number(c.messageCount) || 0}`)
      .join('\n');
    if (sig === conversationCatalogSigRef.current) return undefined;
    conversationCatalogSigRef.current = sig;
    if (visible.length === 0) return undefined;

    if (conversationCatalogPersistTimerRef.current) {
      clearTimeout(conversationCatalogPersistTimerRef.current);
    }
    conversationCatalogPersistTimerRef.current = setTimeout(() => {
      conversationCatalogPersistTimerRef.current = null;
      const list = (conversationsRef.current || []).filter(
        (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
      );
      if (list.length === 0) return;
      void saveConversationCatalog(list, activeConversationRef.current);
    }, 1200);

    return () => {
      if (conversationCatalogPersistTimerRef.current) {
        clearTimeout(conversationCatalogPersistTimerRef.current);
      }
    };
  }, [conversations, storageHydrated]);

  // Resync sidebar after manual key deletes in Settings → Storage.
  useEffect(() => {
    if (!storageHydrated) return undefined;

    const resync = async () => {
      try {
        const parsed = await loadConversationsFromStorage();
        const visible = parsed.filter(
          (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
        );
        if (visible.length === 0) return;
        conversationCatalogSigRef.current = visible
          .map((c) => `${c.id}\t${c.name || ''}\t${Number(c.messageCount) || 0}`)
          .join('\n');
        setConversations(visible);
        const activeId = activeConversationRef.current;
        if (activeId && !visible.some((c) => c.id === activeId)) {
          const fallbackId = visible[0]?.id ?? null;
          console.warn(
            `[Eloquent] Active chat "${activeId}" missing after storage resync — switching to`,
            fallbackId || 'none'
          );
          setActiveConversation(fallbackId);
          if (fallbackId) {
            const fallbackMsgs = await loadConversationMessages(fallbackId);
            setMessages(fallbackMsgs);
          } else {
            setMessages([]);
          }
        }
      } catch (e) {
        console.warn('[conversations] storage-changed resync failed:', e);
      }
    };

    window.addEventListener('eloquent-storage-changed', resync);
    return () => window.removeEventListener('eloquent-storage-changed', resync);
  }, [storageHydrated]);

  // Active tab in memory but missing from sidebar — recover catalog row from shard if possible.
  useEffect(() => {
    if (!storageHydrated || !activeConversation) return undefined;
    if (tombstonedConversationIdsRef.current.has(activeConversation)) return undefined;
    if (conversations.some((c) => c.id === activeConversation)) return undefined;

    let cancelled = false;
    void (async () => {
      const recovered = await recoverConversationCatalogEntry(activeConversation);
      if (cancelled || !recovered) {
        if (!cancelled) {
          console.error(
            '[Eloquent] Active chat missing from sidebar and no recoverable message shard:',
            activeConversation
          );
        }
        return;
      }
      const parsed = await loadConversationsFromStorage();
      if (cancelled) return;
      const visible = parsed.filter(
        (c) => c?.id && !tombstonedConversationIdsRef.current.has(c.id)
      );
      if (visible.length === 0) return;
      conversationCatalogSigRef.current = visible
        .map((c) => `${c.id}\t${c.name || ''}\t${Number(c.messageCount) || 0}`)
        .join('\n');
      setConversations(visible);
      console.info('[Eloquent] Restored missing sidebar tab from message shard:', activeConversation);
    })();

    return () => { cancelled = true; };
  }, [storageHydrated, activeConversation, conversations, setConversations]);

  // Conversations are loaded by IndexedDB hydration; no duplicate sync load needed
  const getActiveConversationData = useCallback(() => {
    const conversation = conversations.find(conv => conv.id === activeConversation);
    return conversation || { id: activeConversation, messages: [] };
  }, [conversations, activeConversation]);

  const setActiveConversationWithMessages = useCallback((conversationId) => {
    handleConversationClick(conversationId);
  }, [handleConversationClick]);
  // Settings are loaded by IndexedDB hydration effect
  // Active summary is no longer restored from localStorage; it's session-only to avoid leaking context between chats.
  useEffect(() => {
    if (activeContextSummary) {
      console.log(`[Summary] Active summary set (${activeContextSummary.length} chars)`);
    } else {
      console.log("[Summary] Active summary cleared");
    }
  }, [activeContextSummary]);
  // Near the other useEffect hooks
  // GPU + ~/.LiangLocal/settings.json must run *after* IndexedDB hydration. Earlier runs merged
  // backend keys into React's initial defaults and persisted that snapshot, clobbering IDB
  // (including Document Context: use_rag / selectedDocuments).
  useEffect(() => {
    if (!storageHydrated) return;
    let cancelled = false;
    (async () => {
      try {
        const response = await fetchWithTimeout(`${PRIMARY_API_URL}/system/gpu_info`, {}, 6000);
        if (cancelled || !response.ok) return;
        const data = await response.json();
        updateSettings({
          singleGpuMode: data.single_gpu_mode,
          detectedGpuCount: Number(data.gpu_count || 0),
          localCudaAvailable: data.cuda_available === true,
          localGgufAvailable: data.local_gguf_available !== false,
          hostedModelsRecommended: data.hosted_models_recommended === true,
          computeMode: data.compute_mode || 'cpu',
        });
        console.log(`System has ${data.gpu_count} GPUs. Compute mode: ${data.compute_mode || 'cpu'}.`);
      } catch (error) {
        console.warn("Could not fetch GPU info:", error);
      }
    })();
    (async () => {
      try {
        const response = await fetchWithTimeout(`${PRIMARY_API_URL}/models/get-settings`, {}, 6000);
        if (cancelled || !response.ok) return;
        const data = await response.json();
        if (data.status === 'success' && data.settings) {
          const backend = data.settings;
          const merged = { ...backend };
          delete merged.userAvatarSize;
          delete merged.characterAvatarSize;
          // Browser session / IDB — never let backend settings.json overwrite document picker state
          delete merged.use_rag;
          delete merged.selectedDocuments;
          delete merged.rag_docs;
          if (!Array.isArray(merged.customApiEndpoints) || merged.customApiEndpoints.length === 0) {
            delete merged.customApiEndpoints;
          }
          updateSettings(merged);
          console.log("Loaded backend settings keys:", Object.keys(data.settings));
        }
      } catch (error) {
        console.warn("Could not load backend settings:", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [storageHydrated, PRIMARY_API_URL, updateSettings]);

  useEffect(() => {
    if (!storageHydrated) {
      postStorageHydrateFetchRef.current = false;
      return;
    }
    if (postStorageHydrateFetchRef.current) return;
    postStorageHydrateFetchRef.current = true;
    fetchModels();
    fetchDocuments();
    loadCharacters();
  }, [storageHydrated, fetchModels, fetchDocuments, loadCharacters]);

  // ----------------------------------
  // Summary Management
  // ----------------------------------
  const generateConversationSummary = useCallback(async (summaryPrompt = "Create a continuity summary that will be injected as context for a future chat. Write it so a model can continue the conversation immediately. Keep it concise and factual. Include: setting/timeframe, key characters and relationships, current goals, recent critical events, unresolved threads, and any constraints (tools, rules, promises). Prefer 6-12 bullet points. Do not add opinions, analysis, or extra commentary.") => {
    const summaryAutoEnabled = settingsRef.current?.apiEndpointRoundRobinEnabled === true;
    if ((!primaryModel && !summaryAutoEnabled) || messages.length === 0) return null;

    setIsGenerating(true);
    try {
      // 1. Build context string
      const chatHistory = messages.map(m => `${m.role === 'user' ? 'User' : (m.characterName || 'Character')}: ${m.content}`).join('\n');
      const fullPrompt = `${summaryPrompt}\n\nCONVERSATION:\n${chatHistory}\n\nSUMMARY:`;

      const { autoEnabled, effectiveModel } = resolveActionRouterContract(
        'create_conversation_summary',
        { requestPurpose: 'conversation_summary' },
      );
      if (!effectiveModel) {
        throw new Error('No model available for summary generation');
      }

      // 2. Call LLM
      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          mergeNanoGptMemoryIntoPayload(
            {
              model_name: effectiveModel,
              prompt: fullPrompt,
              max_tokens: 500,
              temperature: 0.3,
              gpu_id: 0,
              stop: ['\n\nUser:', '\n\nCharacter:'],
              request_purpose: 'conversation_summary',
              round_robin_enabled: autoEnabled,
            },
            settings
          )
        ),
      });

      if (!response.ok) throw new Error("Failed to generate summary");

      const data = await response.json();
      const summaryText = data.text || data.choices?.[0]?.text || "";

      // 3. Save to backend (static/summaries JSON)
      const title = `Story - ${new Date().toLocaleString()}`;
      const saved = await saveSummary(title, summaryText.trim());

      return saved;

    } catch (e) {
      console.error("Summary generation failed:", e);
      return null;
    } finally {
      setIsGenerating(false);
    }

  }, [primaryModel, messages, PRIMARY_API_URL, settings, resolveActionRouterContract]);

  /**
   * Append an existing summary with new details from the current conversation.
   * Produces one updated summary (old + new) and saves it as a new file.
   * @param {{ id: string, title: string, content: string }} existingSummary - The summary to append to
   * @returns {Promise<{ id: string, title: string, content: string }|null>} Saved summary or null on failure
   */
  const generateAppendedSummary = useCallback(async (existingSummary) => {
    const summaryAutoEnabled = settingsRef.current?.apiEndpointRoundRobinEnabled === true;
    if ((!primaryModel && !summaryAutoEnabled) || !messages?.length || !existingSummary?.content) return null;
    const existingContent = typeof existingSummary.content === 'string' ? existingSummary.content : '';
    const existingTitle = existingSummary.title || 'Summary';
    if (!existingContent.trim()) return null;

    setIsGenerating(true);
    try {
      const { autoEnabled, effectiveModel } = resolveActionRouterContract(
        'append_conversation_summary',
        { requestPurpose: 'conversation_summary_append' },
      );
      if (!effectiveModel) {
        throw new Error('No model available for summary append');
      }

      const chatHistory = messages.map(m => `${m.role === 'user' ? 'User' : (m.characterName || 'Character')}: ${m.content}`).join('\n');
      const appendPrompt = `You have an existing story summary and a new conversation. Merge them into a single updated summary.

Rules:
- Keep all information from the existing summary.
- Add or update with new details from the conversation (new events, character developments, setting changes).
- Use the same style (e.g. bullet points). Do not remove existing points unless the conversation clearly contradicts them.
- Output only the updated summary, no preamble.

EXISTING SUMMARY:
${existingContent}

NEW CONVERSATION:
${chatHistory}

UPDATED SUMMARY:`;

      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          mergeNanoGptMemoryIntoPayload(
            {
              model_name: effectiveModel,
              prompt: appendPrompt,
              max_tokens: 600,
              temperature: 0.3,
              gpu_id: 0,
              stop: ['\n\nUser:', '\n\nCharacter:'],
              request_purpose: 'conversation_summary_append',
              round_robin_enabled: autoEnabled,
            },
            settings
          )
        ),
      });

      if (!response.ok) throw new Error("Failed to generate appended summary");

      const data = await response.json();
      const summaryText = (data.text || data.choices?.[0]?.text || "").trim();
      if (!summaryText) return null;

      const title = `${existingTitle} + update ${new Date().toLocaleString()}`;
      const saved = await saveSummary(title, summaryText);
      return saved;
    } catch (e) {
      console.error("Appended summary generation failed:", e);
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, [primaryModel, messages, PRIMARY_API_URL, settings, resolveActionRouterContract]);

  const contextValue = useMemo(() => ({
    messages,
    setMessages,
    appendMessagesToConversation,
    updateMessageInConversation,
    taskProgress,
    setTaskProgress,
    loadTtsEngine,
    loadSttEngine,
    availableModels,
    setAvailableModels,
    loadedModels,
    activeModel,
    isModelLoading,
    loadModel,
    unloadModel,
    setIsGenerating,
    fetchModels,
    conversations,
    activeConversation,
    isGenerating,
    generateReply,
    primaryIsAPI,
    secondaryIsAPI,
    setPrimaryIsAPI,
    setSecondaryIsAPI,
    setConversations,
    isSingleGpuMode,
    portsReady,
    storageHydrated,
    storageHydrationDegraded,
    portsLoadDegraded,
    retryBoot,
    conversationSaveStatus,
    setActiveConversation: setActiveConversationWithMessages,
    deleteConversation,
    deleteAllConversations,
    selectMode,
    selectedConversationIds,
    toggleSelectMode,
    toggleConversationSelection,
    selectAllConversations,
    clearSelection,
    deleteSelectedConversations,
    searchHighlightId,
    setSearchHighlightId,
    renameConversation,
    createNewConversation,
    startCharacterConversation,
    goToHome,
    completeCharacterIntro,
    applyIntroChatTitle,
    updateCharacterIntro,
    getActiveConversationData,
    buildSystemPrompt,
    buildSystemPersonaPrompt,
    formatPrompt,
    prepareApiHistoryWithRollingMemory,
    sttEnabled: settings.sttEnabled ?? true,
    setSttEnabled: (enabled) => updateSettings({ sttEnabled: enabled }),
    ttsEnabled: settings.ttsEnabled ?? true,
    setTtsEnabled: (enabled) => updateSettings({ ttsEnabled: enabled }),
    isRecording,
    fetchTriggeredLore,
    getGenerationSystemPrompt,
    resolveSpeakerCharacter,
    generateChatTitle,
    setIsRecording,
    isPlayingAudio,
    setIsPlayingAudio,
    ttsPlaybackState,
    isTranscribing,
    primaryModel,
    secondaryModel,
    setPrimaryModel,
    setSecondaryModel,
    setIsTranscribing,
    mediaRecorderRef,
    audioChunksRef,
    audioPlayerRef,
    audioError,
    setAudioError,
    startRecording,
    stopRecording,
    playTTS,
    playTestStreamingTTS,
    playStreamingTtsScript,
    getTtsOverridesForCharacterId,
    isCallModeActive,
    callModeRecording,
    startCallMode,
    stopCallMode,
    stopTTS,
    playTTSWithPitch,
    sdStatus,
    fetchMemoriesFromAgent,
    handleStopGeneration,
    abortController,
    setAbortController,
    isStreamingStopped,
    checkSdStatus,
    generateImage,
    generateVideo, // NEW
    generatedImages,
    isImageGenerating,
    generateAndShowImage,
    apiError,
    handleConversationClick,
    setGeneratedImages,
    cleanModelOutput,
    generateUniqueId,
    userProfile,
    sendMessage,
    beginBookAutomationPacking,
    endBookAutomationPacking,
    runBookAutomationChapter,
    runBookAutomationQuickPrompt,
    generateBookChapterJsonOutline,
    buildBookAutomationExport,
    generateCallModeFollowUp,
    settings,
    updateSettings,
    upsertOutreachRule,
    deleteOutreachRule,
    runOutreachRuleNow,
    uploadOutreachRuleImages,
    clearOutreachRuleImages,
    outreachNotifications,
    clearOutreachNotifications,
    dismissOutreachToast,
    openOutreachNotification,
    discardOutreachNotification,
    requestOutreachNotificationPermission,
    outreachScrollToMessageId,
    dismissOutreachScrollTarget,
    inputTranscript,
    setInputTranscript,
    documents,
    fetchDocuments,
    uploadDocument,
    deleteDocument,
    getDocumentContent,
    autoMemoryEnabled,
    fetchLoadedModels,
    setAutoMemoryEnabled,
    getRelevantMemories,
    MEMORY_API_URL,
    lastAgenticMemoryFeedback,
    lastAgenticRunStatus,
    setLastAgenticRunStatus,
   retryAgenticMemoryForLastTurn,
    lastAgenticInjectMeta,
    resolveAgenticUserId,
    alignmentData,
    setAlignmentData,
    alignmentDetectionEnabled,
    setAlignmentDetectionEnabled,
    processAlignmentDetectionIfEnabled,
    addConversationSummary,
    activeTab,
    setActiveTab,
    settingsEntryTab,
    openSettingsTab,
    openSettingsWindow,
    shouldUseDualMode,
    sttEnginesAvailable,
    fetchAvailableSTTEngines,
    nanogptSttModels,
    fetchNanogptSttModels,
    nanogptTtsModels,
    fetchNanogptTtsModels,
    parakeetCppModels,
    parakeetCppCliAvailable,
    fetchParakeetCppModels,
    downloadParakeetCppModel,
    deleteParakeetCppModel,
    voxcpmGgufModels,
    voxcpmGgufCliAvailable,
    fetchVoxcpmGgufModels,
    downloadVoxcpmGgufModel,
    deleteVoxcpmGgufModel,
    BACKEND,
    SECONDARY_API_URL,
    VITE_API_URL,
    endStreamingTTS,
    addStreamingText,
    startStreamingTTS,
    pauseStreamingTTS,
    resumeStreamingTTS,
    isStreamingTtsPaused,
    ttsSubtitleCue,
    ttsFullResponseSaveStatus,
    ttsSubtitleCueRef, // ✅ Expose ref for direct access
    ttsClient,
    setAudioQueue,
    setIsAutoplaying,
    characters,
    activeCharacter,
    userCharacter,
    activeCharacterIds,
    activeCharacterWeights,
    multiRoleContext,
    setActiveCharacter,
    setUserCharacterById,
    updateActiveCharacterIds,
    updateActiveCharacterWeights,
    updateMultiRoleContext,
    loadCharacters,
    saveCharacter,
    saveCharacters,
    cycleCharacterAvatar,
    setCharacterAvatarIndex,
    deleteCharacter,
    duplicateCharacter,
    applyCharacter,
    setCharacterChatRole,
    primaryCharacter,
    speechDetected,
    secondaryCharacter,
    setPrimaryCharacter,
    primaryAvatar,
    setPrimaryAvatar,
    secondaryAvatar,
    setSecondaryAvatar,
    setSecondaryCharacter,
    activeAvatar,
    primaryAvatarRef,
    secondaryAvatarRef,
    // Chat background
    backgroundImage, setBackgroundImage,
    // Room image gallery
    roomGalleryOpen, setRoomGalleryOpen,
    saveToGallery,
    showAvatars,
    setShowAvatars,
    setApplyAvatar,
    userAvatar,
    setUserAvatar,
    userAvatarSize: settings.userAvatarSize ?? userAvatarSize,
    setUserAvatarSize: (size) => updateSettings({ userAvatarSize: size }),
    characterAvatarSize: settings.characterAvatarSize ?? characterAvatarSize,
    setCharacterAvatarSize: (size) => updateSettings({ characterAvatarSize: size }),
    applyAvatar,
    setActiveAvatar,
    showAvatarsInChat,
    setShowAvatarsInChat,
    autoDeleteChats,
    setAutoDeleteChats,
    dualModeEnabled,
    setDualModeEnabled,
    sendDualMessage,
    capturePromptSubmissionTime: () => { promptSubmissionStartTime.current = performance.now(); },
    startAgentConversation,
    agentConversationActive,
    PRIMARY_API_URL,
    TTS_API_URL,
    clearError: () => setApiError(null),
    generateConversationSummary,
    generateAppendedSummary,
    activeContextSummary,
    setActiveContextSummary,
    unlockAudioContext,
    injectTimestamp,
    setInjectTimestamp,
    lastRequestRouteMeta,
    setLastRequestRouteMeta,
    stopStreamingTTS,
  }), [
    messages, appendMessagesToConversation, updateMessageInConversation, availableModels, loadedModels, activeModel, isModelLoading, loadModel, unloadModel, conversations, activeConversation, isGenerating, generateReply, primaryIsAPI, secondaryIsAPI, isSingleGpuMode, portsReady, storageHydrated, setActiveConversationWithMessages, deleteConversation, renameConversation, createNewConversation, startCharacterConversation, goToHome, getActiveConversationData, buildSystemPrompt, formatPrompt, settings, isRecording, fetchTriggeredLore, generateChatTitle, resolveSpeakerCharacter, isPlayingAudio, ttsPlaybackState, isTranscribing, primaryModel, lastRequestRouteMeta, setLastRequestRouteMeta, secondaryModel, audioError, startRecording, stopRecording, playTTS, playTestStreamingTTS, playStreamingTtsScript, isCallModeActive, callModeRecording, startCallMode, stopCallMode, stopTTS, playTTSWithPitch, sdStatus, fetchMemoriesFromAgent, handleStopGeneration, abortController, isStreamingStopped, checkSdStatus, generateImage, generateVideo, generatedImages, isImageGenerating, generateAndShowImage, apiError, handleConversationClick, cleanModelOutput, generateUniqueId, userProfile, sendMessage, beginBookAutomationPacking, endBookAutomationPacking, runBookAutomationChapter, runBookAutomationQuickPrompt, generateBookChapterJsonOutline, buildBookAutomationExport, generateCallModeFollowUp, updateSettings, upsertOutreachRule, deleteOutreachRule, runOutreachRuleNow, outreachNotifications, clearOutreachNotifications, dismissOutreachToast, openOutreachNotification, discardOutreachNotification, requestOutreachNotificationPermission, outreachScrollToMessageId, dismissOutreachScrollTarget, pendingDMThreadId, setPendingDMThreadId, inputTranscript, documents, fetchDocuments, uploadDocument, deleteDocument, getDocumentContent, autoMemoryEnabled, fetchLoadedModels, getRelevantMemories, MEMORY_API_URL, addConversationSummary, activeTab, shouldUseDualMode, sttEnginesAvailable, fetchAvailableSTTEngines, nanogptSttModels, fetchNanogptSttModels, BACKEND, SECONDARY_API_URL, TTS_API_URL, VITE_API_URL, endStreamingTTS, addStreamingText, startStreamingTTS, pauseStreamingTTS, resumeStreamingTTS, isStreamingTtsPaused, ttsSubtitleCue, ttsFullResponseSaveStatus, ttsClient, characters, activeCharacter, userCharacter, activeCharacterIds, activeCharacterWeights, multiRoleContext, setUserCharacterById, updateActiveCharacterIds, updateActiveCharacterWeights, updateMultiRoleContext, loadCharacters, saveCharacter, saveCharacters, deleteCharacter, duplicateCharacter, applyCharacter, setCharacterChatRole, primaryCharacter, speechDetected, secondaryCharacter, primaryAvatar, secondaryAvatar, activeAvatar, showAvatars, applyAvatar, userAvatar, showAvatarsInChat, autoDeleteChats, dualModeEnabled, sendDualMessage, startAgentConversation, agentConversationActive, PRIMARY_API_URL,     generateConversationSummary, generateAppendedSummary, activeContextSummary, setActiveContextSummary, unlockAudioContext, injectTimestamp, setInjectTimestamp,
    roomGalleryOpen, saveToGallery,
  ]);


  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

const useApp = () => React.useContext(AppContext);

export { AppProvider, useApp };
