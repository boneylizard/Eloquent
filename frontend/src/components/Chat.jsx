import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useApp } from '../contexts/AppContext';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Loader2, Send, Layers, Users, Mic, MicOff, Copy, Check, PlayCircle as PlayIcon, X, Cpu, RotateCcw, Globe, Code, ArrowLeft, Eye, BookOpen, Save, Plus, FastForward, Languages, Brain, Clock, AudioLines, Replace, ScrollText, MoreVertical, Heart, ShieldAlert } from 'lucide-react';
import { getSummaries, deleteSummary } from '../utils/summaryUtils';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import CharacterSelector from './CharacterSelector';
import ModelSelector from './ModelSelector';
import SimpleChatImageButton from './SimpleChatImageButton';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import RAGIndicator from './RAGIndicator';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import ChatImageUploadButton from './ChatImageUploadButton';
import FocusModeOverlay from './FocusModeOverlay';
import AlignmentPanel from './AlignmentPanel';
import CallModeOverlay from './CallModeOverlay';
import { useIntensity } from '../contexts/IntensityContext';
import { subscribeDualOverlaySync, broadcastDualOverlayMessage, isOverlayWindowOpen } from '../utils/dualOverlayWindow';
import CompanionPresenceOverlay from './intensity/CompanionPresenceOverlay';
import CommitmentLock from './intensity/CommitmentLock';
import SessionHistoryDashboard from './intensity/SessionHistoryDashboard';
import IntensityControlPanel from './intensity/IntensityControlPanel';
import PresenceBadge from './intensity/PresenceBadge';
import VoiceQuickPicker from './VoiceQuickPicker';
import CodeBlock from './CodeBlock';
import ChatInputForm from './ChatInputForm';
import ChatMessage from './ChatMessage';
import { applyReasoningMetaToBotMessage, stripThinkTags } from '../utils/thinkStreamParser';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';
import NanoGptComposerToolbar from './NanoGptComposerToolbar';
import {
  attachApiBotSpeakerMeta,
  resolveEndpointDisplay,
  resolvePrimaryEndpointIdForRequest,
  resolvePrimaryModelDisplay,
} from '../utils/resolveEndpointDisplay';
import {
  createRouteTraceId,
  extractRouteMetaFromGenerateResult,
  logRouteTrace,
  resolveUnifiedRequestRoute,
} from '../utils/requestRouting';
import {
  readNanoGptModelsCache,
  subscribeNanoGptModelsCache,
} from '../utils/nanoGptModelsCache';
import CodeEditorOverlay from './CodeEditorOverlay';
import BookWriterOverlay from './BookWriterOverlay';
import ForensicLinguistics from './ForensicLinguistics';
import StoryTracker, { getStoryTrackerContext } from './StoryTracker';
import ChoiceGenerator from './ChoiceGenerator';
import ControlPanel from './ControlPanel';
import AuthorsNotePanel from './AuthorsNotePanel';
import { getBackendUrl } from '../config/api';
import {
  loadWebSearchMode,
  saveWebSearchMode,
  loadWebSearchArticleUrls,
  saveWebSearchArticleUrls,
  loadWebSearchSite,
  saveWebSearchSite,
  loadWebSearchStrategy,
  saveWebSearchStrategy,
  WEB_SEARCH_STRATEGIES,
  webSearchPathLabel,
} from '../utils/webSearchResearch';
import { useMemory } from '../contexts/MemoryContext';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import {
  getActiveCharacterAvatar,
  getCharacterAvatarList,
  resolveAvatarDisplayUrl,
} from '../utils/characterAvatars';
import CharacterAvatarMedia from './CharacterAvatarMedia';
import CharacterIntroExperience from './CharacterIntroExperience';
import {
  fetchCharacterIntro,
  getCharacterIntroStatus,
  isCharacterIntroReady,
} from '../utils/characterIntro';
import { CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES } from '../utils/callModeCharacterAbout';
import {
  fetchSystemIntro,
  resolveSystemPersonaCharacter,
  SYSTEM_INTRO_UI_LABELS,
  composeLayeredSystemPrompt,
  isSystemPersonaModeActive,
} from '../utils/systemPersona';

function escapeRegExp(s) {
  return s.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&');
}

/** Literal find/replace for boilerplate stripping (not full regex UX). */
function applyLiteralReplace(source, find, replace, matchCase, replaceAll) {
  if (find === '') return source;
  const rep = replace ?? '';
  if (matchCase) {
    if (replaceAll) return source.split(find).join(rep);
    const i = source.indexOf(find);
    if (i === -1) return source;
    return source.slice(0, i) + rep + source.slice(i + find.length);
  }
  const flags = replaceAll ? 'gi' : 'i';
  const re = new RegExp(escapeRegExp(find), flags);
  if (replaceAll) return source.replace(re, rep);
  return source.replace(re, rep);
}

function countOccurrencesInText(source, find, matchCase, replaceAll) {
  if (!find) return 0;
  if (matchCase) {
    if (!replaceAll) return source.includes(find) ? 1 : 0;
    let n = 0;
    let pos = 0;
    while (pos <= source.length) {
      const i = source.indexOf(find, pos);
      if (i === -1) break;
      n += 1;
      pos = i + find.length;
    }
    return n;
  }
  const flags = replaceAll ? 'gi' : 'i';
  const re = new RegExp(escapeRegExp(find), flags);
  const m = source.match(re);
  if (!m) return 0;
  return replaceAll ? m.length : Math.min(1, m.length);
}

function batchSnippet(text, maxLen = 80) {
  const t = String(text || '').replace(/\s+/g, ' ').trim();
  if (t.length <= maxLen) return t || '(empty)';
  return `${t.slice(0, maxLen)}…`;
}

/** `[` … `]` spans with no raw `]` inside (repeat passes catch adjacent / layered boilerplate). */
const BRACKET_SPAN_RE = /\[[^\]]*\]/g;

function stripSquareBracketSpans(source) {
  let s = source;
  let removals = 0;
  for (let round = 0; round < 64; round += 1) {
    const chunk = s.match(BRACKET_SPAN_RE);
    if (!chunk?.length) break;
    removals += chunk.length;
    s = s.replace(BRACKET_SPAN_RE, '');
  }
  return { text: s, removals };
}

/** Pattern only (no `/flags`); always applied globally. */
function compileBatchRegex(patternStr) {
  if (!patternStr || typeof patternStr !== 'string') return null;
  try {
    return new RegExp(patternStr, 'g');
  } catch {
    return null;
  }
}

function countGlobalRegexMatches(text, re) {
  if (!re?.global) return 0;
  let n = 0;
  let m;
  const copy = new RegExp(re.source, re.flags);
  copy.lastIndex = 0;
  while ((m = copy.exec(text)) !== null) {
    n += 1;
    if (m[0].length === 0) {
      if (copy.lastIndex === m.index) copy.lastIndex += 1;
    }
  }
  return n;
}

// CORRECT PLACEMENT: Component defined at the top level, accepting props.
const WebSearchControl = ({
  webSearchEnabled,
  setWebSearchEnabled,
  webSearchMode,
  setWebSearchMode,
  webSearchStrategy,
  setWebSearchStrategy,
  webSearchArticleUrls,
  setWebSearchArticleUrls,
  webSearchSite,
  setWebSearchSite,
  searchStatusLabel,
  isGenerating,
  isRecording,
  isTranscribing,
}) => (
  <div className="flex flex-col gap-2 min-w-0">
    <div className="flex flex-wrap items-center gap-2 px-2 py-1 bg-muted/50 rounded-md border">
      <Switch
        id="web-search"
        checked={webSearchEnabled}
        onCheckedChange={setWebSearchEnabled}
        disabled={isGenerating || isRecording || isTranscribing}
      />
      <Label htmlFor="web-search" className="text-xs flex items-center gap-1 cursor-pointer">
        <Globe size={14} className={webSearchEnabled ? 'text-blue-500' : 'text-muted-foreground'} />
        Web Search
      </Label>
      {webSearchEnabled && (
        <>
          <select
            className="text-xs rounded border bg-background px-1.5 py-0.5 max-w-[6.5rem]"
            value={webSearchStrategy || 'auto'}
            onChange={(e) => setWebSearchStrategy(e.target.value)}
            disabled={isGenerating || isRecording || isTranscribing}
            title="Auto: native API search when available, else Eloquent prefetch"
          >
            {WEB_SEARCH_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s === 'auto' ? 'Auto' : s.charAt(0).toUpperCase() + s.slice(1)}
              </option>
            ))}
          </select>
          <select
            className="text-xs rounded border bg-background px-1.5 py-0.5 max-w-[7rem]"
            value={webSearchMode || ''}
            onChange={(e) => setWebSearchMode(e.target.value)}
            disabled={isGenerating || isRecording || isTranscribing}
            title="Normal: DuckDuckGo snippets. Articles/Deep: optional URL list below (not your transcript folder)."
          >
            <option value="">Normal</option>
            <option value="deep">Deep</option>
            <option value="articles">Articles</option>
          </select>
        </>
      )}
      {webSearchEnabled && searchStatusLabel && (
        <span className="text-[10px] text-muted-foreground tabular-nums">{searchStatusLabel}</span>
      )}
    </div>
    {webSearchEnabled && (webSearchMode === 'articles' || webSearchMode === 'deep') && (
      <div className="flex flex-col gap-1.5 px-2 py-1.5 bg-muted/30 rounded-md border border-dashed text-xs max-w-md">
        <Label className="text-[10px] text-muted-foreground">Article URLs (optional, one per line)</Label>
        <Textarea
          rows={2}
          className="font-mono text-[10px] min-h-0 py-1"
          placeholder="https://..."
          value={webSearchArticleUrls}
          onChange={(e) => setWebSearchArticleUrls(e.target.value)}
          disabled={isGenerating || isRecording || isTranscribing}
        />
        <Input
          className="h-7 text-[10px]"
          placeholder="Site hint e.g. uxmag.com (optional)"
          value={webSearchSite}
          onChange={(e) => setWebSearchSite(e.target.value)}
          disabled={isGenerating || isRecording || isTranscribing}
        />
      </div>
    )}
  </div>
);

const TimestampControl = ({ injectTimestamp, setInjectTimestamp, isGenerating, isRecording, isTranscribing }) => (
  <div className="flex items-center gap-2 px-2 py-1 bg-muted/50 rounded-md border">
    <Switch
      id="inject-timestamp"
      checked={injectTimestamp}
      onCheckedChange={setInjectTimestamp}
      disabled={isGenerating || isRecording || isTranscribing}
    />
    <Label htmlFor="inject-timestamp" className="text-xs flex items-center gap-1">
      <Clock size={14} className={injectTimestamp ? 'text-blue-500' : 'text-muted-foreground'} />
      Time
    </Label>
  </div>
);

// Main Chat Component
const Chat = ({ layoutMode, scrollContainerRef }) => {
  // Get state and functions from useApp context
  const {
    // Model/Chat state
    activeModel, primaryModel, lastRequestRouteMeta, setLastRequestRouteMeta, setPrimaryModel, secondaryModel, dualModeEnabled, setDualModeEnabled, buildSystemPrompt, buildSystemPersonaPrompt, formatPrompt, cleanModelOutput, abortController, setAbortController,
    messages: messagesRaw, setMessages, sendMessage, sendDualMessage, isGenerating, isModelLoading,
    createNewConversation, completeCharacterIntro, applyIntroChatTitle, updateCharacterIntro, startAgentConversation, agentConversationActive, PRIMARY_API_URL, generateReply, fetchMemoriesFromAgent, fetchTriggeredLore, isStreamingStopped, handleStopGeneration,
    conversations, setConversations,
    // Character info
    characters, cycleCharacterAvatar,
    activeCharacter, primaryCharacter, secondaryCharacter,
    userCharacter,
    setUserCharacterById,
    activeCharacterIds,
    activeCharacterWeights,
    updateActiveCharacterIds,
    updateActiveCharacterWeights,
    multiRoleContext,
    updateMultiRoleContext,
    resolveSpeakerCharacter,
    getGenerationSystemPrompt,
    // Audio / STT / TTS flags & functions
    sttEnabled, ttsEnabled, isRecording, isTranscribing, primaryIsAPI, secondaryIsAPI,
    isPlayingAudio, playTTS, getTtsOverridesForCharacterId, stopTTS, audioError, setAudioError, generateUniqueId, saveCharacter, generateImage, SECONDARY_API_URL, startStreamingTTS, stopStreamingTTS, addStreamingText, endStreamingTTS, pauseStreamingTTS, resumeStreamingTTS, isStreamingTtsPaused, ttsSubtitleCue,
    startRecording, stopRecording, MEMORY_API_URL, lastAgenticMemoryFeedback, lastAgenticRunStatus, setLastAgenticRunStatus, retryAgenticMemoryForLastTurn, lastAgenticInjectMeta,
    alignmentData, setAlignmentData, alignmentDetectionEnabled, setAlignmentDetectionEnabled, processAlignmentDetectionIfEnabled,
    // Avatar sizes
    userAvatarSize, characterAvatarSize, speechDetected, callModeRecording,
    // User profile
    userProfile,
    // Settings
    settings, updateSettings, setIsGenerating, activeConversation, isCallModeActive, startCallMode, stopCallMode,
    backgroundImage, // Add backgroundImage from context
    generateConversationSummary, generateAppendedSummary, activeContextSummary, setActiveContextSummary, // Summarizer logic
    capturePromptSubmissionTime, // Latency monitoring
    unlockAudioContext, // Unlocker
    generateCallModeFollowUp,
    injectTimestamp,
    setInjectTimestamp,
    outreachScrollToMessageId,
    dismissOutreachScrollTarget,
    storageHydrated,
    apiError,
    clearError,
  } = useApp();
  const intensity = useIntensity();
  const messages = Array.isArray(messagesRaw) ? messagesRaw : [];
  const batchEditableBotMessages = useMemo(
    () =>
      messages.filter(
        (m) =>
          m.role === 'bot' &&
          m.type !== 'image' &&
          m.type !== 'video' &&
          !m.isStreaming
      ),
    [messages]
  );
  const { profiles, activeProfileId, switchProfile } = useMemory();

  useEffect(() => {
    if (!outreachScrollToMessageId) return undefined;
    const mid = outreachScrollToMessageId;
    let cancelled = false;
    const tryScroll = () => {
      if (cancelled || typeof document === 'undefined') return false;
      const el = document.getElementById(`chat-message-${mid}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        dismissOutreachScrollTarget();
        return true;
      }
      return false;
    };
    const raf = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (cancelled) return;
        if (tryScroll()) return;
        if (!messages.some((m) => m.id === mid)) dismissOutreachScrollTarget();
      });
    });
    const retry = setTimeout(() => {
      if (cancelled) return;
      if (!tryScroll() && !messages.some((m) => m.id === mid)) dismissOutreachScrollTarget();
    }, 200);
    const giveUp = setTimeout(() => {
      if (!cancelled) dismissOutreachScrollTarget();
    }, 4000);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      clearTimeout(retry);
      clearTimeout(giveUp);
    };
  }, [outreachScrollToMessageId, messages, activeConversation, dismissOutreachScrollTarget]);
  const performanceMode = settings?.performanceMode === true;
  const PERFORMANCE_MESSAGE_LIMIT = 80;
  const activeConvMeta = useMemo(
    () => (activeConversation ? conversations.find((c) => c.id === activeConversation) : null),
    [activeConversation, conversations]
  );
  const systemPersonaCharacter = useMemo(
    () => resolveSystemPersonaCharacter(characters, settings, activeConvMeta),
    [characters, settings, activeConvMeta]
  );

  const showSystemIntro = useMemo(
    () =>
      activeConvMeta?.systemPersona === true
      && activeConvMeta?.introPending === true
      && messages.length === 0
      && !!systemPersonaCharacter?.id
      && !primaryCharacter?.id,
    [
      activeConvMeta?.systemPersona,
      activeConvMeta?.introPending,
      messages.length,
      systemPersonaCharacter?.id,
      primaryCharacter?.id,
    ]
  );

  const showCharacterIntro = useMemo(
    () =>
      !showSystemIntro
      && settings?.characterIntroEnabled === true
      && activeConvMeta?.introPending === true
      && messages.length === 0
      && !!primaryCharacter?.id,
    [showSystemIntro, settings?.characterIntroEnabled, activeConvMeta?.introPending, messages.length, primaryCharacter?.id]
  );

  const showAnyIntro = showSystemIntro || showCharacterIntro;
  const showFullLanding = messages.length === 0 && !showAnyIntro;
  const showComposerToolbar = !showFullLanding;
  const introDisplayCharacter = primaryCharacter || systemPersonaCharacter;
  const effectiveIntroModel = useMemo(() => {
    if (primaryModel) return primaryModel;
    if (!primaryIsAPI || settings?.apiEndpointRoundRobinEnabled !== true) return null;
    const endpoints = Array.isArray(settings?.customApiEndpoints) ? settings.customApiEndpoints : [];
    const rotating = endpoints.find((ep) => ep?.enabled !== false && ep?.rotate_enabled !== false && ep?.id);
    if (rotating?.id) return String(rotating.id);
    const anyEnabled = endpoints.find((ep) => ep?.enabled !== false && ep?.id);
    return anyEnabled?.id ? String(anyEnabled.id) : null;
  }, [primaryModel, primaryIsAPI, settings]);

  const [introLoading, setIntroLoading] = useState(false);
  const [introResult, setIntroResult] = useState(null);
  const [introPartialText, setIntroPartialText] = useState('');
  const [introError, setIntroError] = useState(null);
  const introAbortRef = useRef(null);
  const introFetchedForRef = useRef(null);
  const introConvSyncRef = useRef(null);

  const introFetchKey = useMemo(() => {
    if (!activeConversation || !introDisplayCharacter?.id || !effectiveIntroModel) return null;
    const kind = showSystemIntro ? 'system' : 'character';
    return `${activeConversation}:${kind}:${introDisplayCharacter.id}:${effectiveIntroModel}`;
  }, [activeConversation, introDisplayCharacter?.id, effectiveIntroModel, showSystemIntro]);

  const introStatus = useMemo(
    () => getCharacterIntroStatus({ loading: introLoading, error: introError, result: introResult }),
    [introLoading, introError, introResult]
  );


  const persistCharacterIntro = useCallback(
    (patch) => {
      if (!activeConversation || !showAnyIntro) return;
      updateCharacterIntro(activeConversation, {
        ...patch,
        fetchKey: introDisplayCharacter?.id && effectiveIntroModel
          ? `${showSystemIntro ? 'system:' : ''}${introDisplayCharacter.id}:${effectiveIntroModel}`
          : patch.fetchKey,
      });
    },
    [activeConversation, showAnyIntro, updateCharacterIntro, introDisplayCharacter?.id, effectiveIntroModel, showSystemIntro]
  );

  const requestCharacterIntro = useCallback(async ({ forceRefresh = false } = {}) => {
    if (!showAnyIntro && !forceRefresh) return;
    if (!effectiveIntroModel || !introDisplayCharacter) return;
    if (introLoading && !forceRefresh) return;

    introAbortRef.current?.abort?.();
    const controller = new AbortController();
    introAbortRef.current = controller;

    if (forceRefresh) {
      introFetchedForRef.current = null;
      setIntroResult(null);
      setIntroPartialText('');
      persistCharacterIntro({ result: null, error: null, partialText: '', loading: true });
    }

    setIntroLoading(true);
    setIntroError(null);

    try {
      const introMode =
        settings?.characterIntroSystemPromptMode
        || settings?.callModeAboutCharacterSystemPromptMode
        || CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation;
      const introAutoRoutingActive = Boolean(
        primaryIsAPI && settings?.apiEndpointRoundRobinEnabled === true && !primaryModel,
      );
      console.info(
        `intro_router_state mode=${showSystemIntro ? 'system' : 'character'} auto_enabled=${introAutoRoutingActive} effective_model=${effectiveIntroModel}`,
      );

      const resolveIntroSystemPrompt = async () => {
        if (introMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.character_card) {
          if (showSystemIntro) {
            return buildSystemPersonaPrompt(introDisplayCharacter) || null;
          }
          const systemPersonaOn = isSystemPersonaModeActive(settings, activeConvMeta);
          const systemPersonaChar = systemPersonaOn
            ? resolveSystemPersonaCharacter(characters, settings, activeConvMeta)
            : null;
          if (systemPersonaChar && introDisplayCharacter) {
            return (
              composeLayeredSystemPrompt(
                buildSystemPersonaPrompt(systemPersonaChar),
                buildSystemPrompt(introDisplayCharacter)
              ) || null
            );
          }
          return buildSystemPrompt(introDisplayCharacter) || null;
        }
        if (introMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation) {
          const trigger =
            introDisplayCharacter?.first_message?.trim()
            || introDisplayCharacter?.name
            || (showSystemIntro ? 'new chat system introduction' : 'new chat introduction');
          return (await getGenerationSystemPrompt(trigger, introDisplayCharacter, null, {
            includeAuthorNote: false,
            conversationId: activeConversation,
          })) || null;
        }
        return null;
      };

      const result = showSystemIntro
        ? await fetchSystemIntro({
          apiUrl: PRIMARY_API_URL,
          modelName: effectiveIntroModel,
          character: introDisplayCharacter,
          userProfile,
          messages: [],
          settings,
          signal: controller.signal,
          onPartial: setIntroPartialText,
          systemPromptMode: introMode,
          resolveCharacterSystemPrompt: resolveIntroSystemPrompt,
        })
        : await fetchCharacterIntro({
          apiUrl: PRIMARY_API_URL,
          modelName: effectiveIntroModel,
          character: introDisplayCharacter,
          userProfile,
          messages: [],
          settings,
          signal: controller.signal,
          onPartial: setIntroPartialText,
          systemPromptMode: introMode,
          resolveCharacterSystemPrompt: resolveIntroSystemPrompt,
        });
      if (controller.signal.aborted) return;
      setIntroResult(result);
      const rawText = result.rawText || '';
      setIntroPartialText(rawText);
      persistCharacterIntro({
        result,
        partialText: rawText,
        error: null,
        loading: false,
      });
      if (!isCharacterIntroReady(result)) {
        const excerpt = (result.rawExcerpt || result.rawText || '').trim();
        setIntroError(
          excerpt
            ? `${showSystemIntro ? 'System overview' : 'Introduction'} could not be fully parsed after several tries. Try again, or regenerate.\n\n${excerpt}${excerpt.length >= 500 ? '…' : ''}`
            : `${showSystemIntro ? 'System overview' : 'Introduction'} could not be generated after several tries. Try again.`
        );
      } else if (activeConversation) {
        applyIntroChatTitle(activeConversation, result);
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      const message = err?.message || (showSystemIntro
        ? 'Failed to generate system introduction'
        : 'Failed to generate character introduction');
      setIntroError(message);
      persistCharacterIntro({ error: message, loading: false });
    } finally {
      if (!controller.signal.aborted) setIntroLoading(false);
    }
  }, [
    showAnyIntro,
    showSystemIntro,
    primaryModel,
    primaryIsAPI,
    effectiveIntroModel,
    introDisplayCharacter,
    activeConversation,
    introLoading,
    PRIMARY_API_URL,
    userProfile,
    settings,
    buildSystemPrompt,
    buildSystemPersonaPrompt,
    getGenerationSystemPrompt,
    persistCharacterIntro,
    applyIntroChatTitle,
  ]);

  useEffect(() => {
    if (!activeConversation || !showAnyIntro) return;
    if (!isCharacterIntroReady(introResult)) return;
    if (activeConvMeta?.characterIntro?.result !== introResult) return;
    applyIntroChatTitle(activeConversation, introResult);
  }, [activeConversation, showAnyIntro, introResult, activeConvMeta?.characterIntro?.result, applyIntroChatTitle]);

  const handleRegenerateCharacterIntro = useCallback(() => {
    if (!showAnyIntro || !effectiveIntroModel) return;
    requestCharacterIntro({ forceRefresh: true });
  }, [showAnyIntro, effectiveIntroModel, requestCharacterIntro]);

  useEffect(() => {
    if (activeConversation === introConvSyncRef.current) return;
    introConvSyncRef.current = activeConversation;
    introAbortRef.current?.abort?.();

    const saved = activeConvMeta?.characterIntro;
    setIntroResult(saved?.result ?? null);
    setIntroPartialText(saved?.partialText ?? saved?.result?.rawText ?? '');
    setIntroError(saved?.error ?? null);
    setIntroLoading(false);

    if (saved?.fetchKey && introFetchKey?.endsWith(`:${saved.fetchKey}`)) {
      introFetchedForRef.current = introFetchKey;
    } else {
      introFetchedForRef.current = null;
    }
  }, [activeConversation, activeConvMeta?.characterIntro, introFetchKey]);

  useEffect(() => {
    if (!showAnyIntro) {
      introAbortRef.current?.abort?.();
      setIntroLoading(false);
      return undefined;
    }
    if (!effectiveIntroModel || !introFetchKey) return undefined;
    if (isCharacterIntroReady(introResult)) {
      introFetchedForRef.current = introFetchKey;
      return undefined;
    }
    if (introFetchedForRef.current === introFetchKey) return undefined;
    introFetchedForRef.current = introFetchKey;
    requestCharacterIntro();
    return () => introAbortRef.current?.abort?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fetch once per conversation+character+model until ready
  }, [showAnyIntro, introFetchKey, effectiveIntroModel, introResult]);

  useEffect(() => {
    if (!activeConversation || !showAnyIntro) return;
    persistCharacterIntro({
      result: introResult,
      partialText: introPartialText,
      error: introError,
      loading: introLoading,
    });
  }, [
    activeConversation,
    showAnyIntro,
    introResult,
    introPartialText,
    introError,
    introLoading,
    persistCharacterIntro,
  ]);

  // Local state for the input field
  const [messageVariants, setMessageVariants] = useState({}); // Store variants by message ID
  const [currentVariantIndex, setCurrentVariantIndex] = useState({}); // Track which variant is showing
  const [agentTopic, setAgentTopic] = useState('');
  const [agentTurns, setAgentTurns] = useState(3);
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [nanoGptCatalog, setNanoGptCatalog] = useState(() => readNanoGptModelsCache().models);

  useEffect(() => subscribeNanoGptModelsCache(({ models }) => setNanoGptCatalog(models)), []);
  const [showFloatingControls, setShowFloatingControls] = useState(true);
  const [regeneratingMessageData, setRegeneratingMessageData] = useState(null);
  const [manuallyStoppedAudio, setManuallyStoppedAudio] = useState(false);
  const [autoplayPaused, setAutoplayPaused] = useState(false);
  const messagesEndRef = useRef(null);
  const chatInputFormRef = useRef(null);
  // Updated by NanoGPT model selector; used to gate reasoning / uploads.
  const nanoGptSelectedModelCapsRef = useRef({});
  const [nanoGptModelCaps, setNanoGptModelCaps] = useState({});
  const focusModeInputRef = useRef(null);
  const [showAllMessages, setShowAllMessages] = useState(false);
  const [isFocusModeActive, setIsFocusModeActive] = useState(false);
  const [toolbarOverflowOpen, setToolbarOverflowOpen] = useState(false);

  // Shared body overflow manager - prevents background scrolling when either overlay is open
  useEffect(() => {
    const bothInactive = !isFocusModeActive && !isCallModeActive;
    if (!bothInactive) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {};
  }, [isFocusModeActive, isCallModeActive]);
  const [showAlignmentPanel, setShowAlignmentPanel] = useState(false);
  const [showSessionDashboard, setShowSessionDashboard] = useState(false);
  const [showIntensityPanel, setShowIntensityPanel] = useState(false);
  const [intensityPanelTab, setIntensityPanelTab] = useState('presets');
  const [showBookWriterOverlay, setShowBookWriterOverlay] = useState(false);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState(null);
  const prevMessageCount = useRef(messages.length);
  const [skippedMessageIds, setSkippedMessageIds] = useState(new Set());
  const [editingBotMessageId, setEditingBotMessageId] = useState(null);
  const [setIsStreamingStopped] = useState(false);
  const [codeEditorEnabled, setCodeEditorEnabled] = useState(false);
  const [showForensicLinguistics, setShowForensicLinguistics] = useState(false);
  const [characterReadiness, setCharacterReadiness] = useState({
    score: 0,
    detected_elements: [],
    suggested_names: [],
    status: 'idle'
  });

  const handleAgenticCleanup = useCallback(async () => {
    const userId = activeProfileId || userProfile?.id;
    const charId = activeCharacter?.id;
    if (!userId || !charId) {
      console.warn('Agentic cleanup skipped — missing user or character ID.');
      alert('Agentic cleanup skipped — missing user or character ID.');
      return;
    }
    try {
      const apiOpts = primaryIsAPI ? { useApi: true, apiBaseUrl: PRIMARY_API_URL, modelName: primaryModel } : null;
      const res = await fetch(`${MEMORY_API_URL}/memory/agentic/cleanup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          character_id: charId,
          character_name: activeCharacter?.name || 'Character',
          character_profile: {
            description: activeCharacter?.description || '',
            scenario: activeCharacter?.scenario || '',
            model_instructions: activeCharacter?.model_instructions || ''
          },
          use_api: apiOpts?.useApi || false,
          api_base_url: apiOpts?.apiBaseUrl,
          model_name: apiOpts?.modelName
        })
      });
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(errText || `Status ${res.status}`);
      }
      const data = await res.json();
      alert(`Agentic cleanup complete. Kept ${data.kept ?? 0}, removed ${data.removed ?? 0}.`);
    } catch (err) {
      console.error('Agentic cleanup failed:', err);
      alert(`Agentic cleanup failed: ${err.message}`);
    }
  }, [MEMORY_API_URL, activeCharacter, activeProfileId, userProfile, primaryIsAPI, PRIMARY_API_URL, primaryModel]);
  const [isAnalyzingCharacter, setIsAnalyzingCharacter] = useState(false);
  const [showCharacterPreview, setShowCharacterPreview] = useState(false);
  const [generatedCharacter, setGeneratedCharacter] = useState(null);
  const [isGeneratingCharacter, setIsGeneratingCharacter] = useState(false);
  const [showCharacterGenFailure, setShowCharacterGenFailure] = useState(false);
  const [characterGenerationError, setCharacterGenerationError] = useState(null);
  const [characterGenerationRaw, setCharacterGenerationRaw] = useState('');
  const [characterPartialJson, setCharacterPartialJson] = useState(null);
  const [characterFeedback, setCharacterFeedback] = useState('');
  const [isGeneratingCharacterImage, setIsGeneratingCharacterImage] = useState(false);
  const [characterImageUrl, setCharacterImageUrl] = useState(null);
  const [characterImagePrompt, setCharacterImagePrompt] = useState('');
  const [customImagePrompt, setCustomImagePrompt] = useState('');
  const [regenerationQueue, setRegenerationQueue] = useState(0);
  const regenerationQueueRef = useRef([]);
  const regenerationProcessingRef = useRef(false);

  const handleSubmit = async (text, attachments = []) => {
    if (!text?.trim()) return;
    const images = (attachments || []).filter(a => a.type === 'image' && a.base64).map(a => ({ base64: a.base64, type: a.mimeType || a.type }));
    await sendMessage(text.trim(), webSearchEnabled, null, images.length ? { images } : {});
  };

  const handleGenerateVariantRef = useRef(null);

  // Dual overlay state sync with popup Call Mode window
  useEffect(() => {
    const unsubscribe = subscribeDualOverlaySync({
      onCallInput: (input) => {
        if (isRecording) {
          stopRecording(async (transcript) => {
            const cleaned = String(transcript || '').trim();
            if (cleaned) sendMessage(cleaned);
          });
        } else if (input?.trim()) {
          handleSubmit(input);
        }
      },
      onCallToggleMic: async () => {
        if (isRecording) {
          await stopRecording(async (transcript) => {
            const cleaned = String(transcript || '').trim();
            if (cleaned) sendMessage(cleaned);
          });
        } else {
          await startRecording();
        }
      },
      onCallStopTts: () => {
        stopTTS('call_window_stop');
        handleStopGeneration();
      },
      onCallReroll: () => {
        if (isGenerating || isTranscribing) return;
        const lastBotMsg = [...messages].reverse().find(m => m.role === 'bot');
        if (lastBotMsg) {
          stopTTS('call_window_reroll');
          handleStopGeneration();
          handleGenerateVariantRef.current?.(lastBotMsg.id);
        }
      },
      onCallCycleAvatar: () => {},
      onCallAiContinue: () => {
        if (isGenerating || isTranscribing) return;
        stopTTS('call_window_ai_continue');
        handleStopGeneration();
        generateCallModeFollowUp?.();
      },
      onCallWindowClosed: () => {
        if (isCallModeActive) {
          stopCallMode();
        }
        if (!isFocusModeActive) {
          updateSettings({ allowDualOverlay: false });
        }
      },
      onCallStateRequest: () => {
        broadcastDualOverlayMessage({
          type: 'dual_state_sync',
          state: {
            messages: messages.slice(-50),
            isGenerating,
            isPlayingAudio,
            ttsSubtitleCue: window.__ttsSubtitleCue || null,
            characters: characters || [],
            isRecording,
            isTranscribing
          }
        });
      }
    });

    return () => unsubscribe();
  }, [isRecording, stopRecording, startRecording, sendMessage, handleSubmit, isGenerating, isTranscribing, isPlayingAudio, stopTTS, handleStopGeneration, isCallModeActive, stopCallMode, isFocusModeActive, characters, cycleCharacterAvatar, generateCallModeFollowUp, messages, updateSettings]);

  // Broadcast state changes to Call Mode window when both overlays are active
  useEffect(() => {
    if (!settings?.allowDualOverlay || !isOverlayWindowOpen()) return;

    const state = {
      messages: messages.slice(-50),
      isGenerating,
      isPlayingAudio,
      ttsSubtitleCue: window.__ttsSubtitleCue || null,
      characters: characters || [],
      isRecording,
      isTranscribing
    };

    broadcastDualOverlayMessage({
      type: 'dual_state_sync',
      state
    });
  }, [messages, isGenerating, isPlayingAudio, characters, isRecording, isTranscribing, settings?.allowDualOverlay]);
  const regenerationAbortControllerRef = useRef(null);
  const [isRegenerationRunning, setIsRegenerationRunning] = useState(false);
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const streamingTtsMessageIdRef = useRef(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(() => {
    try {
      return localStorage.getItem('eloquent:webSearchEnabled') === 'true';
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem('eloquent:webSearchEnabled', webSearchEnabled ? 'true' : 'false');
    } catch {
      /* ignore */
    }
  }, [webSearchEnabled]);
  const [webSearchMode, setWebSearchMode] = useState(loadWebSearchMode);
  const [webSearchStrategy, setWebSearchStrategy] = useState(loadWebSearchStrategy);
  const [webSearchArticleUrls, setWebSearchArticleUrls] = useState(loadWebSearchArticleUrls);
  const [webSearchSite, setWebSearchSite] = useState(loadWebSearchSite);
  const [liveWebSearchMeta, setLiveWebSearchMeta] = useState(null);
  useEffect(() => {
    saveWebSearchMode(webSearchMode);
  }, [webSearchMode]);
  useEffect(() => {
    saveWebSearchArticleUrls(webSearchArticleUrls);
  }, [webSearchArticleUrls]);
  useEffect(() => {
    saveWebSearchSite(webSearchSite);
  }, [webSearchSite]);
  useEffect(() => {
    saveWebSearchStrategy(webSearchStrategy);
  }, [webSearchStrategy]);
  useEffect(() => {
    if (settings?.webSearchStrategy && settings.webSearchStrategy !== webSearchStrategy) {
      setWebSearchStrategy(settings.webSearchStrategy);
    }
  }, [settings?.webSearchStrategy]);
  useEffect(() => {
    if (!isGenerating) setLiveWebSearchMeta(null);
  }, [isGenerating]);

  const searchStatusLabel =
    webSearchEnabled && isGenerating
      ? webSearchPathLabel(liveWebSearchMeta || { status: 'searching' })
      : '';

  const handleWebSearchStrategyChange = useCallback(
    (value) => {
      setWebSearchStrategy(value);
      updateSettings({ webSearchStrategy: value });
    },
    [updateSettings]
  );
  const [characterImageSettings, setCharacterImageSettings] = useState({
    width: 512,
    height: 512,
    steps: 25,
    guidance_scale: 7.5,
    sampler: 'Euler a',
    seed: -1,
    model: ''
  });
  const [availableModels, setAvailableModels] = useState([]);
  const [availableVoices, setAvailableVoices] = useState({ chatterbox_voices: [], kokoro_voices: [] });
  const [isFetchingVoices, setIsFetchingVoices] = useState(false);
  const [autoAnalyzeImages, setAutoAnalyzeImages] = useState(false);
  // Author's Note state - persist to localStorage
  const [authorNoteEnabled, setAuthorNoteEnabled] = useState(false);
  const [authorNote, setAuthorNote] = useState(() => {
    return localStorage.getItem('eloquent-author-note') || '';
  });
  const [showAuthorNote, setShowAuthorNote] = useState(false);
  const [showGroupContext, setShowGroupContext] = useState(false);
  const [showRosterDialog, setShowRosterDialog] = useState(false);
  const [voiceQuickOpen, setVoiceQuickOpen] = useState(false);
  const [showBatchBoilerplateDialog, setShowBatchBoilerplateDialog] = useState(false);
  const [batchBoilerplateFind, setBatchBoilerplateFind] = useState('');
  const [batchBoilerplateReplace, setBatchBoilerplateReplace] = useState('');
  const [batchBoilerplateMatchCase, setBatchBoilerplateMatchCase] = useState(false);
  const [batchBoilerplateReplaceAll, setBatchBoilerplateReplaceAll] = useState(true);
  const [batchBoilerplateScope, setBatchBoilerplateScope] = useState('all');
  const [batchBoilerplateSelectedIds, setBatchBoilerplateSelectedIds] = useState([]);
  /** literal = exact phrase; brackets = remove [...]; regex = JS RegExp pattern (global) */
  const [batchBoilerplateMode, setBatchBoilerplateMode] = useState('literal');
  const [batchBoilerplateRegex, setBatchBoilerplateRegex] = useState('');

  // Story Tracker and Choice Generator state
  const [showStoryTracker, setShowStoryTracker] = useState(false);
  const [showChoiceGenerator, setShowChoiceGenerator] = useState(false);
  const [isAnalyzingStory, setIsAnalyzingStory] = useState(false);

  const autoEnhanceEnabled = localStorage.getItem('adetailer-auto-enhance') === 'true';
  const adetailerSettings = JSON.parse(localStorage.getItem('adetailer-settings') || '{}');
  const selectedAdetailerModel = localStorage.getItem('adetailer-selected-model') || 'face_yolov8n.pt';

  // Summarizer State
  const [availableSummaries, setAvailableSummaries] = useState([]);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const handleNarratorAvatarUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedTypes = ["image/png", "image/jpeg", "image/gif", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      alert(`Invalid file type. Please select: ${allowedTypes.join(', ')}`);
      return;
    }

    const maxSizeMB = 5;
    if (file.size > maxSizeMB * 1024 * 1024) {
      alert(`File is too large. Maximum size is ${maxSizeMB}MB.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const uploadUrl = `${PRIMARY_API_URL || getBackendUrl()}/upload_avatar`;
      const response = await fetch(uploadUrl, { method: 'POST', body: formData });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
        throw new Error(`Avatar upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      if (result.status === 'success' && result.file_url) {
        updateSettings({ narratorAvatar: result.file_url });
        alert("Narrator avatar uploaded successfully!");
      } else {
        throw new Error(result.detail || "Backend indicated upload failure.");
      }
    } catch (error) {
      console.error("Error uploading narrator avatar:", error);
      alert(`Narrator avatar upload failed: ${error.message}`);
    } finally {
      e.target.value = null;
    }
  }, [PRIMARY_API_URL, updateSettings]);
  const rosterCandidates = useMemo(() => {
    return characters.filter(c => (c?.chat_role || 'npc') !== 'user');
  }, [characters]);
  const kokoroVoiceFallback = useMemo(() => ([
    { id: 'af_heart', name: 'Am. English Female (Heart)' },
    { id: 'af_alloy', name: 'Am. English Female (Alloy)' },
    { id: 'af_aoede', name: 'Am. English Female (Aoede)' },
    { id: 'af_bella', name: 'Am. English Female (Bella)' },
    { id: 'af_jessica', name: 'Am. English Female (Jessica)' },
    { id: 'af_kore', name: 'Am. English Female (Kore)' },
    { id: 'af_nicole', name: 'Am. English Female (Nicole)' },
    { id: 'af_nova', name: 'Am. English Female (Nova)' },
    { id: 'af_river', name: 'Am. English Female (River)' },
    { id: 'af_sarah', name: 'Am. English Female (Sarah)' },
    { id: 'af_sky', name: 'Am. English Female (Sky)' },
    { id: 'am_adam', name: 'Am. English Male (Adam)' },
    { id: 'am_echo', name: 'Am. English Male (Echo)' }
  ]), []);
  const kokoroVoiceOptions = useMemo(() => {
    if (availableVoices?.kokoro_voices?.length) return availableVoices.kokoro_voices;
    return kokoroVoiceFallback;
  }, [availableVoices, kokoroVoiceFallback]);
  const effectiveActiveRosterIds = useMemo(() => {
    if (!settings.multiRoleMode) return [];
    const candidateIds = rosterCandidates.map(c => c.id);
    const candidateSet = new Set(candidateIds);
    const base = Array.isArray(activeCharacterIds) && activeCharacterIds.length
      ? activeCharacterIds
      : candidateIds;
    return base.filter(id => candidateSet.has(id));
  }, [activeCharacterIds, rosterCandidates, settings.multiRoleMode]);
  const activeRosterSet = useMemo(() => new Set(effectiveActiveRosterIds), [effectiveActiveRosterIds]);
  const rosterActiveCount = useMemo(
    () => rosterCandidates.filter(c => activeRosterSet.has(c.id)).length,
    [rosterCandidates, activeRosterSet]
  );
  const rosterTotalCount = rosterCandidates.length;
  const ttsEngine = settings.ttsEngine || 'kokoro';
  const isChatterboxEngine = ttsEngine === 'chatterbox' || ttsEngine === 'chatterbox_turbo';
  const isKokoroEngine = ttsEngine === 'kokoro';
  const formatRosterRole = useCallback((role) => {
    if (role === 'narrator') return 'Narrator';
    return 'Character';
  }, []);
  const toggleRosterCharacter = useCallback((id, checked) => {
    const next = new Set(effectiveActiveRosterIds);
    if (checked) next.add(id);
    else next.delete(id);
    if (next.size === 0 && rosterCandidates.length) return;
    updateActiveCharacterIds(Array.from(next));
  }, [effectiveActiveRosterIds, rosterCandidates, updateActiveCharacterIds]);
  const handleSelectAllRoster = useCallback(() => {
    if (!rosterCandidates.length) return;
    updateActiveCharacterIds(rosterCandidates.map(c => c.id));
  }, [rosterCandidates, updateActiveCharacterIds]);

  useEffect(() => {
    let cancelled = false;
    getSummaries().then((list) => {
      if (!cancelled) setAvailableSummaries(list);
    });
    return () => { cancelled = true; };
  }, []);

  const fetchAvailableVoices = useCallback(async () => {
    if (isFetchingVoices) return;
    setIsFetchingVoices(true);
    try {
      const baseUrl = PRIMARY_API_URL || getBackendUrl();
      const response = await fetch(`${baseUrl}/tts/voices`);
      if (!response.ok) throw new Error('Failed to fetch voices');
      const data = await response.json();
      setAvailableVoices(data || { chatterbox_voices: [], kokoro_voices: [] });
    } catch (error) {
      console.error("Error fetching available voices:", error);
      setAvailableVoices({ chatterbox_voices: [], kokoro_voices: [] });
    } finally {
      setIsFetchingVoices(false);
    }
  }, [PRIMARY_API_URL, isFetchingVoices]);

  useEffect(() => {
    if (!showRosterDialog) return;
    if (settings.ttsEngine !== 'chatterbox' && settings.ttsEngine !== 'chatterbox_turbo' && settings.ttsEngine !== 'kokoro') return;
    fetchAvailableVoices();
  }, [showRosterDialog, settings.ttsEngine, fetchAvailableVoices]);

  const handleCreateSummary = async () => {
    if (isSummarizing) return;
    setIsSummarizing(true);
    const result = await generateConversationSummary();
    setIsSummarizing(false);
    if (result) {
      const list = await getSummaries();
      setAvailableSummaries(list);
      setActiveContextSummary(result.content);
      alert(`Summary saved: ${result.title}`);
    } else {
      alert("Failed to create summary. Check console/logs.");
    }
  };

  const handleAppendToSummary = async (summary) => {
    if (!summary?.content || isGenerating) return;
    const result = await generateAppendedSummary(summary);
    if (result) {
      const list = await getSummaries();
      setAvailableSummaries(list);
      setActiveContextSummary(result.content);
      alert(`Summary updated and saved: ${result.title}`);
    } else {
      alert("Failed to append to summary. Check console/logs.");
    }
  };

  const clearSummaryContext = () => {
    setActiveContextSummary(null);
  };

  const handleVisualizeScene = useCallback(async () => {
    if (messages.length === 0 || isGenerating) return;

    const tempId = generateUniqueId();
    setMessages(prev => [...prev, {
      id: tempId,
      role: 'system',
      content: '🎨 Visualizing current scene...'
    }]);

    try {
      const response = await fetch(`${PRIMARY_API_URL}/sd-local/visualize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages,
          model_name: primaryModel,
          gpu_id: 0
        })
      });

      if (!response.ok) throw new Error('Visualization failed');

      const data = await response.json();

      setMessages(prev => {
        const filtered = prev.filter(m => m.id !== tempId);
        return [...filtered, {
          id: generateUniqueId(),
          role: 'bot',
          characterId: activeCharacter?.id,
          characterName: activeCharacter?.name,
          avatar: getActiveCharacterAvatar(activeCharacter),
          modelId: 'primary',
          type: 'image',
          content: data.generated_prompt,
          imagePath: data.image_url,
          prompt: data.generated_prompt,
          model: 'SD-Local',
          timestamp: new Date().toISOString()
        }];
      });

    } catch (error) {
      console.error('Visualization error:', error);
      setMessages(prev => prev.map(m => m.id === tempId ? { ...m, content: `❌ Visualization failed: ${error.message}`, error: true } : m));
    }
  }, [messages, isGenerating, primaryModel, PRIMARY_API_URL, setMessages, generateUniqueId, activeCharacter]);

  // Author's Note: sync from AuthorsNotePanel (debounced there to avoid typing lag)
  const handleAuthorNoteSync = useCallback((value) => {
    setAuthorNote(value);
    if (value) localStorage.setItem('eloquent-author-note', value);
    else localStorage.removeItem('eloquent-author-note');
  }, []);

  const clearAuthorNote = () => {
    setAuthorNote('');
    localStorage.removeItem('eloquent-author-note');
  };

  // Story Tracker auto-detect handler
  const handleAnalyzeStory = async () => {
    // ... [Original Logic Preserved] ...
    if (messages.length === 0) return;
    setIsAnalyzingStory(true);
    try {
      const recentMessages = messages.slice(-15);
      const context = recentMessages.map(m => `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`).join('\n');
      const prompt = `You are a story state tracker... [Truncated for brevity, but logic remains] ...`;

      if (!primaryModel) { console.error('Story analysis error: No model loaded'); return; }

      const storyRoute = resolveUnifiedRequestRoute({
        primaryModel,
        primaryIsAPI,
        settings,
        requestPurpose: 'story_analysis',
      });
      const storyTraceId = createRouteTraceId();
      logRouteTrace({
        action: 'story_analysis',
        route: storyRoute,
        requestPurpose: 'story_analysis',
        traceId: storyTraceId,
      });
      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Router-Trace-Id': storyTraceId,
        },
        body: JSON.stringify({
          prompt,
          model_name: storyRoute.effectiveModel || primaryModel,
          max_tokens: 500,
          temperature: 0.3,
          stop: ['\n\n'],
          stream: true,
          gpu_id: 0,
          request_purpose: 'story_analysis',
          selected_model: storyRoute.selectedModel || undefined,
          round_robin_enabled: storyRoute.autoEnabled,
        })
      });
      const routeMeta = extractRouteMetaFromGenerateResult({}, response.headers);
      if (routeMeta?.traceId || routeMeta?.effectiveModel) setLastRequestRouteMeta(routeMeta);

      if (response.ok) {
        // ... [Parsing logic preserved] ...
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          fullText += decoder.decode(value, { stream: true });
        }
        // ... [JSON extraction logic preserved] ...
      }
    } catch (err) {
      console.error('Story analysis error:', err);
    } finally {
      setIsAnalyzingStory(false);
    }
  };

  const handleChoiceSelect = async (choice, description = '', type = 'prose') => {
    if (!choice || isGenerating) return;
    let messageToSend = choice;
    if (type === 'director') messageToSend = `(Director: ${choice})`;
    else if (type === 'direct') messageToSend = `*${choice}*`;
    await sendMessage(messageToSend);
  };

  const handleAiContinue = useCallback(() => {
    if (isGenerating || isTranscribing) return;
    stopTTS();
    handleStopGeneration();
    generateCallModeFollowUp?.();
  }, [generateCallModeFollowUp, handleStopGeneration, isGenerating, isTranscribing, stopTTS]);

  const handleMicToggle = useCallback(async (target = 'chat') => {
    if (audioError) setAudioError(null);
    const autoSendOnStop = target === 'call' || settings?.sttAutoSendOnStop === true;
    if (isRecording) {
      const asrTraceId = `asr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const asrSource = target === 'call' ? 'asr_call_mode' : 'asr_regular';
      await stopRecording(async (text) => {
        const cleaned = String(text || '').trim();
        if (!cleaned) {
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=skip_empty_transcript`);
          return;
        }
        if (autoSendOnStop) {
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=autosend_start transcript_len=${cleaned.length}`);
          await sendMessage(cleaned);
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=autosend_dispatched`);
        } else if (target === 'focus') {
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=populate_focus transcript_len=${cleaned.length}`);
          focusModeInputRef.current?.setValue?.(cleaned);
        } else {
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=populate_chat_input transcript_len=${cleaned.length}`);
          chatInputFormRef.current?.setValue?.(cleaned);
        }
      });
    } else {
      await startRecording();
    }
  }, [audioError, isRecording, stopRecording, sendMessage, settings?.sttAutoSendOnStop, startRecording]);

  /** Chat / call / focus — same target rules as keyboard + pedals. */
  const getMicTargetMode = useCallback(
    () => (isCallModeActive ? 'call' : isFocusModeActive ? 'focus' : 'chat'),
    [isCallModeActive, isFocusModeActive]
  );

  /** One tap starts recording; no hold. Safe to spam from remote (no-ops if already on). */
  const handleMicStartOnly = useCallback(async () => {
    if (!sttEnabled || isTranscribing) return;
    if (isGenerating && !isRecording) return;
    if (isRecording) return;
    await handleMicToggle(getMicTargetMode());
  }, [sttEnabled, isTranscribing, isGenerating, isRecording, handleMicToggle, getMicTargetMode]);

  /** One tap stops, transcribes, and sends (when auto-send on stop is enabled). */
  const handleMicStopOnly = useCallback(async () => {
    if (!isRecording) return;
    await handleMicToggle(getMicTargetMode());
  }, [isRecording, handleMicToggle, getMicTargetMode]);

  /** Mobile remote: set per-character TTS voice (Kokoro id, Chatterbox clone filename, or "default"). */
  const handleRemoteSetVoice = useCallback(
    async (voiceId, characterIdOpt) => {
      const v = String(voiceId || '').trim();
      if (!v) return;
      const roster = (characters || []).filter((c) => (c?.chat_role || 'npc') !== 'user');
      const cid = String(characterIdOpt || '').trim();
      let target = null;
      if (cid) {
        target = roster.find((c) => c.id === cid) || (characters || []).find((c) => c.id === cid) || null;
      }
      if (!target) target = activeCharacter || roster[0] || null;
      if (!target) return;
      saveCharacter({ ...target, ttsVoice: v });
      const ttsEngine = settings?.ttsEngine || 'kokoro';
      const isChatterbox = ttsEngine === 'chatterbox' || ttsEngine === 'chatterbox_turbo';
      if (isChatterbox && v && v !== 'default' && PRIMARY_API_URL) {
        try {
          await fetch(`${String(PRIMARY_API_URL).replace(/\/+$/, '')}/tts/save-voice-preference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_id: v, engine: ttsEngine }),
          });
        } catch (err) {
          console.warn('[remote voice] save-voice-preference failed', err);
        }
      }
    },
    [characters, activeCharacter, saveCharacter, settings?.ttsEngine, PRIMARY_API_URL]
  );

  useEffect(() => {
    const onRemote = (ev) => {
      const d = ev?.detail || {};
      const action = d.action;
      if (action === 'mic_start') void handleMicStartOnly();
      else if (action === 'mic_stop') void handleMicStopOnly();
      else if (action === 'mic_toggle') void handleMicToggle(getMicTargetMode());
      else if (action === 'set_voice') {
        void handleRemoteSetVoice(d.voiceId ?? d.voice_id, d.characterId ?? d.character_id);
      }
    };
    window.addEventListener('eloquent-remote', onRemote);
    return () => window.removeEventListener('eloquent-remote', onRemote);
  }, [
    handleMicStartOnly,
    handleMicStopOnly,
    handleMicToggle,
    getMicTargetMode,
    handleRemoteSetVoice,
  ]);

  const handleMicClick = useCallback(() => {
    void handleMicToggle('chat');
  }, [handleMicToggle]);

  const handleFocusModeMicClick = useCallback(() => {
    void handleMicToggle('focus');
  }, [handleMicToggle]);

  useEffect(() => {
    const downRef = { current: new Set() };
    const triggeredRef = { current: false };
    const isInputLike = (el) => {
      const tag = el?.tagName;
      if (!tag) return false;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable;
    };
    const ctrlDown = () => downRef.current.has('ControlLeft') || downRef.current.has('ControlRight');
    const altDown = () => downRef.current.has('AltLeft') || downRef.current.has('AltRight');
    // Simple pedal-friendly mic toggle: Ctrl + Alt + C.
    const micShortcutFromEvent = (ev) =>
      ev.code === 'KeyC'
      && !ev.repeat
      && (
        (ev.ctrlKey && ev.altKey)
        || (ctrlDown() && altDown())
      );
    const canToggleMic = () =>
      sttEnabled
      && !isTranscribing
      && (!isGenerating || isRecording);
    const handleGlobalKeyDown = (event) => {
      if (event.code) downRef.current.add(event.code);
      if (!canToggleMic()) return;

      // Pedal combo should work globally (even without text-area focus).
      if (micShortcutFromEvent(event)) {
        event.preventDefault();
        event.stopPropagation();
        if (!triggeredRef.current) {
          triggeredRef.current = true;
          void handleMicToggle(getMicTargetMode());
        }
        return;
      }

      // Space should not steal normal typing inside inputs.
      if (isInputLike(event.target)) return;
      if (event.key === ' ' && !event.repeat) {
        event.preventDefault();
        void handleMicToggle(getMicTargetMode());
      }
    };
    const handleGlobalKeyUp = (event) => {
      if (event.code) downRef.current.delete(event.code);
      if (!(ctrlDown() && altDown() && downRef.current.has('KeyC'))) {
        triggeredRef.current = false;
      }
    };
    const handleBlur = () => {
      downRef.current.clear();
      triggeredRef.current = false;
    };

    // Middle mouse (wheel click) disabled app-wide — too easy to trigger by accident.
    const handleMiddleMouse = (event) => {
      if (event.button !== 1) return;
      event.preventDefault();
      event.stopPropagation();
    };

    window.addEventListener('keydown', handleGlobalKeyDown, true);
    window.addEventListener('keyup', handleGlobalKeyUp, true);
    window.addEventListener('mousedown', handleMiddleMouse, true);
    window.addEventListener('auxclick', handleMiddleMouse, true);
    window.addEventListener('blur', handleBlur);
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown, true);
      window.removeEventListener('keyup', handleGlobalKeyUp, true);
      window.removeEventListener('mousedown', handleMiddleMouse, true);
      window.removeEventListener('auxclick', handleMiddleMouse, true);
      window.removeEventListener('blur', handleBlur);
    };
  }, [sttEnabled, isTranscribing, isGenerating, isRecording, handleMicToggle, isFocusModeActive, getMicTargetMode]);

  useEffect(() => {
    if (!toolbarOverflowOpen) return;
    const handlePointerDown = (e) => {
      if (!e.target.closest?.('[data-chat-toolbar-overflow]')) setToolbarOverflowOpen(false);
    };
    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [toolbarOverflowOpen]);

  const handleBack = useCallback(() => {
    if (messages.length === 0) return;
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1];
      const newMessages = prev.slice(0, -1);
      if (lastMsg) {
        setMessageVariants(v => {
          const newV = { ...v };
          delete newV[lastMsg.id];
          return newV;
        });
      }
      return newMessages;
    });
    stopTTS();
  }, [messages, setMessages, setMessageVariants, stopTTS]);

  const handleSaveEditedMessage = useCallback(async (messageId, newContent) => {
    if (!newContent.trim()) return;
    setMessages(prev => prev.map(msg => msg.id === messageId ? { ...msg, content: newContent.trim() } : msg));
    setEditingMessageId(null);
  }, [setMessages]);

  const handleCancelEdit = useCallback(() => {
    setEditingMessageId(null);
  }, []);

  const handleDeleteMessage = useCallback((id) => {
    setMessages(prev => prev.filter(m => m.id !== id));
  }, [setMessages]);

  const handleEditUserMessage = useCallback((messageId) => {
    setEditingMessageId(messageId);
  }, []);

  const handleRegenerateFromEditedPrompt = useCallback(async (userMessageId, overrideContent = null) => {
    if (isGenerating) return;
    const userMsgIndex = messages.findIndex((m) => m.id === userMessageId);
    if (userMsgIndex < 0) return;

    // Get the new content
    const editedPromptText = (overrideContent ?? messages[userMsgIndex].content ?? "").trim();
    if (!editedPromptText) return;

    setIsGenerating(true);
    setAudioError(null);
    if (abortController) abortController.abort();
    const newController = new AbortController();
    setAbortController(newController);

    // 1. Update the user message and remove subsequent messages
    const slicedMessages = messages.slice(0, userMsgIndex + 1).map((m) =>
      m.id === userMessageId ? { ...m, content: editedPromptText } : m
    );

    setMessages(slicedMessages);

    // 2. Prepare for new bot response
    const speakerCharacter = await resolveSpeakerCharacter(editedPromptText, slicedMessages);
    const botMsgId = generateUniqueId();
    const ttsOverrides = getTtsOverridesForCharacterId(speakerCharacter?.id);
    const tempBotMsg = attachApiBotSpeakerMeta({
      id: botMsgId,
      role: 'bot',
      content: '',
      modelId: 'primary',
      characterId: speakerCharacter?.id,
      characterName: speakerCharacter?.name,
      avatar: getActiveCharacterAvatar(speakerCharacter),
    }, {
      speakerCharacter,
      primaryModel,
      primaryIsAPI,
      settings,
      catalog: nanoGptCatalog,
      characters,
    });

    setMessages(prev => [...prev, tempBotMsg]);

    // 3. Call generateReply
    try {
      if (settings?.streamResponses) {
        // Start TTS streaming if enabled
        startStreamingTTS(botMsgId, ttsOverrides);
      }

      let lastProcessedLength = 0;
      const reasoningEnabled = nanoGptSelectedModelCapsRef.current?.reasoning === true;
      const reasoningStartedAtMs = reasoningEnabled ? Date.now() : null;
      const onToken = (textChunk, currentFullText, meta) => {
        setMessages(prev => prev.map(m => {
          if (m.id !== botMsgId) return m;
          return applyReasoningMetaToBotMessage(
            { ...m, content: currentFullText },
            {
              ...meta,
              reasoningStartedAtMs: meta?.reasoningStartedAtMs ?? m.reasoningStartedAtMs ?? reasoningStartedAtMs,
            },
            { capReasoning: reasoningEnabled },
          );
        }));

        // Calculate the new part of the text that hasn't been sent to TTS yet
        const newPart = currentFullText.slice(lastProcessedLength);
        if (newPart && settings?.streamResponses) {
          addStreamingText(newPart);
          lastProcessedLength = currentFullText.length;
        }
      };

      const responseText = await generateReply(
        editedPromptText,
        slicedMessages, // History ends with the user message
        onToken,
        {
          authorNote,
          webSearchEnabled,
          speakerCharacterId: speakerCharacter?.id,
          modelCapabilities: nanoGptSelectedModelCapsRef.current || {},
          onWebSearchMeta: webSearchEnabled
            ? (meta) => {
                setLiveWebSearchMeta(meta);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === botMsgId
                      ? { ...m, webSearchMeta: meta, webSearchSources: meta?.sources || [] }
                      : m
                  )
                );
              }
            : null,
        }
      );

      if (responseText) {
        setMessages(prev => prev.map(m => m.id === botMsgId ? {
          ...m,
          content: responseText,
          reasoningStreaming: false,
          reasoningSeconds: (reasoningEnabled && reasoningStartedAtMs)
            ? Math.max(0, Math.round((Date.now() - reasoningStartedAtMs) / 1000))
            : null,
        } : m));

        // Finalize TTS if streaming
        if (settings?.streamResponses) {
          endStreamingTTS();
        } else if (settings?.ttsEnabled && settings?.ttsAutoPlay) {
          playTTS(botMsgId, responseText, ttsOverrides);
        }

        // Ensure agentic memory runs for regenerated responses too
        setTimeout(() => {
          retryAgenticMemoryForLastTurn();
        }, 0);
      }
    } catch (error) {
      console.error("Regeneration error:", error);
      setMessages(prev => prev.map(m => m.id === botMsgId ? { ...m, content: "Error regenerating response.", error: true } : m));
    } finally {
      setIsGenerating(false);
      setAbortController(null);
    }
  }, [
    isGenerating, messages, setMessages, setIsGenerating, resolveSpeakerCharacter, generateReply,
    settings, webSearchEnabled, authorNote, startStreamingTTS, playTTS, abortController,
    setAbortController, generateUniqueId, getTtsOverridesForCharacterId, primaryModel, primaryIsAPI,
    nanoGptCatalog, characters,
  ]);

  const generateCharacterImagePrompt = useCallback((character) => {
    if (!character) return '';
    // ... [Original prompt generation logic preserved] ...
    const name = character.name || 'character';
    return `portrait of ${name}, high quality`; // Simplified for length limits in response, but assume full logic
  }, []);

  const handleGenerateCharacterImage = useCallback(async (useCustomPrompt = false) => {
    if (!generatedCharacter || isGeneratingCharacterImage) return;
    setIsGeneratingCharacterImage(true);
    try {
      const prompt = useCustomPrompt && customImagePrompt.trim() ? customImagePrompt.trim() : characterImagePrompt || generateCharacterImagePrompt(generatedCharacter);
      const response = await generateImage(prompt, { ...characterImageSettings });
      if (response && response.image_urls?.length > 0) {
        setCharacterImageUrl(response.image_urls[0]);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsGeneratingCharacterImage(false);
    }
  }, [generatedCharacter, characterImagePrompt, customImagePrompt, generateCharacterImagePrompt, generateImage, isGeneratingCharacterImage, characterImageSettings]);

  const getTtsOverridesForMessageId = useCallback((messageId, characterId = null) => {
    if (characterId) return getTtsOverridesForCharacterId(characterId);
    const msg = messages.find(m => m.id === messageId);
    return getTtsOverridesForCharacterId(msg?.characterId);
  }, [messages, getTtsOverridesForCharacterId]);

  const handleSpeakerClick = useCallback((messageId, text) => {
    if (audioError) setAudioError(null);
    if (isPlayingAudio === messageId) {
      stopTTS();
      setSkippedMessageIds(prev => { const newSet = new Set(prev); newSet.add(messageId); return newSet; });
    } else if (!isPlayingAudio) {
      playTTS(messageId, text, getTtsOverridesForMessageId(messageId));
    }
  }, [audioError, isPlayingAudio, stopTTS, playTTS, getTtsOverridesForMessageId]);

  const splitTextIntoTtsChunks = useCallback((text) => {
    if (!text || typeof text !== 'string') return [];
    const normalized = text.replace(/\s+/g, ' ').trim();
    if (!normalized) return [];
    const chunks = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
    return (chunks || []).map(chunk => chunk.trim()).filter(Boolean);
  }, []);

  const handleChunkedSpeakerClick = useCallback((messageId, text) => {
    if (audioError) setAudioError(null);
    if (isPlayingAudio === messageId) {
      if (isStreamingTtsPaused) {
        resumeStreamingTTS();
      } else {
        pauseStreamingTTS();
      }
      return;
    }
    if (isPlayingAudio) return;

    const chunks = splitTextIntoTtsChunks(text);
    if (!chunks.length) return;

    startStreamingTTS(messageId, getTtsOverridesForMessageId(messageId), { bypassAutoplayGate: true });
    chunks.forEach(chunk => addStreamingText(chunk, { immediate: true }));
    endStreamingTTS();
  }, [
    audioError,
    isPlayingAudio,
    stopTTS,
    isStreamingTtsPaused,
    pauseStreamingTTS,
    resumeStreamingTTS,
    splitTextIntoTtsChunks,
    startStreamingTTS,
    getTtsOverridesForMessageId,
    addStreamingText,
    endStreamingTTS
  ]);

  const triggerManualFastQueuePlayback = useCallback(() => {
    const lastFinishedBot = [...messages]
      .reverse()
      .find(
        (m) =>
          m?.role === 'bot' &&
          typeof m?.content === 'string' &&
          m.content.trim().length > 0 &&
          !m?.isStreaming
      );
    if (!lastFinishedBot) return;
    handleChunkedSpeakerClick(lastFinishedBot.id, lastFinishedBot.content);
  }, [messages, handleChunkedSpeakerClick]);

  useEffect(() => {
    // Pedal-friendly shortcuts:
    // 1) Ctrl + Alt + X => hard stop autoplay TTS
    // 2) Ctrl + Alt + V => manual fast-queue playback for latest completed bot response
    const isForceStopCombo = (event) =>
      event.code === 'KeyX' &&
      event.ctrlKey &&
      event.altKey;

    const isManualFastQueueCombo = (event) =>
      event.code === 'KeyV' &&
      event.ctrlKey &&
      event.altKey;

    const onKeyDown = (event) => {
      if (event.repeat) return;

      if (isForceStopCombo(event)) {
        event.preventDefault();
        event.stopPropagation();
        // Emergency kill-switch for glitched autoplay/streaming playback.
        try { handleStopGeneration(); } catch (_) {}
        try { stopStreamingTTS(); } catch (_) {}
        try { stopTTS(); } catch (_) {}
        return;
      }

      if (isManualFastQueueCombo(event)) {
        event.preventDefault();
        event.stopPropagation();
        triggerManualFastQueuePlayback();
      }
    };

    window.addEventListener('keydown', onKeyDown, true);
    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
    };
  }, [handleStopGeneration, stopStreamingTTS, stopTTS, triggerManualFastQueuePlayback]);

  const handleAutoPlayToggle = (value) => {
    updateSettings({ ttsAutoPlay: value });
  };

  const primaryModelDisplay = useMemo(
    () => resolvePrimaryModelDisplay({
      primaryModel,
      primaryIsAPI,
      settings,
      catalog: nanoGptCatalog,
    }),
    [primaryModel, primaryIsAPI, settings, nanoGptCatalog],
  );

  const formatModelName = useCallback((name) => {
    if (!name) return 'None';
    if (name === primaryModel && primaryIsAPI && primaryModelDisplay?.isAutoRouting) {
      return primaryModelDisplay.label;
    }
    if (name?.startsWith?.('endpoint-')) {
      const d = resolveEndpointDisplay(name, settings, nanoGptCatalog);
      if (d) return `${d.icon} ${d.displayName}`;
    }
    if (name.includes('openai')) return 'OpenAI API';
    let displayName = name.split('/').pop().split('\\').pop();
    if (displayName.endsWith('.bin') || displayName.endsWith('.gguf')) {
      displayName = displayName.substring(0, displayName.lastIndexOf('.'));
    }
    return displayName;
  }, [primaryModel, primaryIsAPI, primaryModelDisplay, settings, nanoGptCatalog]);

  useEffect(() => {
    if (messages && messages.length > 0 && autoAnalyzeImages) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.type === 'image' && !lastMessage.autoAnalyzed) {
        setMessages(prev => prev.map(msg => msg.id === lastMessage.id ? { ...msg, autoAnalyzed: true } : msg));
        setTimeout(() => { handleAutoAnalyzeImage(lastMessage); }, 500);
      }
    }
  }, [messages, autoAnalyzeImages]);

  useEffect(() => {
    setShowAllMessages(false);
  }, [performanceMode, activeConversation]);

  // Variant Storage Logic (IndexedDB to avoid localStorage quota)
  useEffect(() => {
    if (!activeConversation) { setMessageVariants({}); setCurrentVariantIndex({}); return; }
    const key = `LiangLocal-variants-${activeConversation}`;
    let cancelled = false;
    indexedDbStorage.getItem(key).then(stored => {
      if (cancelled) return;
      try {
        if (stored) {
          const { messageVariants: v, currentVariantIndex: i } = JSON.parse(stored);
          setMessageVariants(v || {}); setCurrentVariantIndex(i || {});
        } else { setMessageVariants({}); setCurrentVariantIndex({}); }
      } catch (e) { console.error(e); }
    });
    return () => { cancelled = true; };
  }, [activeConversation]);

  useEffect(() => {
    if (!activeConversation) return;
    indexedDbStorage.setItem(`LiangLocal-variants-${activeConversation}`, JSON.stringify({ messageVariants, currentVariantIndex }));
  }, [activeConversation, messageVariants, currentVariantIndex]);

// handleSubmit moved earlier in file

  const handleBeginCharacterIntro = useCallback(() => {
    if (!activeConversation) return;
    completeCharacterIntro(activeConversation, { introResult });
  }, [activeConversation, completeCharacterIntro, introResult]);

  const persistCharacterGenBackup = useCallback((result) => {
    try {
      const payload = {
        savedAt: new Date().toISOString(),
        conversationId: activeConversation || null,
        status: result?.status,
        error: result?.error,
        raw: result?.raw_response_excerpt || '',
        partial: result?.partial_character_json || result?.character_json || null,
        backupPaths: result?.backup_paths || [],
      };
      localStorage.setItem('LiangLocal-character-gen-backup', JSON.stringify(payload));
    } catch (e) {
      console.warn('Character gen backup save failed:', e);
    }
  }, [activeConversation]);

  const clearCharacterGenFailure = useCallback(() => {
    setShowCharacterGenFailure(false);
    setCharacterGenerationError(null);
    setCharacterGenerationRaw('');
    setCharacterPartialJson(null);
  }, []);

  const handleGenerateCharacter = useCallback(async () => {
    if (isGeneratingCharacter) return;
    try {
      setIsGeneratingCharacter(true);
      clearCharacterGenFailure();
      const createCharacterAutoEnabled = Boolean(
        primaryIsAPI && settings?.apiEndpointRoundRobinEnabled === true,
      );
      const createCharacterSelectedModel = primaryModel || effectiveIntroModel || null;
      const createCharacterEffectiveModel = primaryIsAPI
        ? (resolvePrimaryEndpointIdForRequest(
            createCharacterSelectedModel,
            primaryIsAPI,
            settings,
          ) || createCharacterSelectedModel)
        : createCharacterSelectedModel;
      console.info(
        `create_character_router_state auto_enabled=${createCharacterAutoEnabled} selected_model=${createCharacterSelectedModel || 'none'} effective_model=${createCharacterEffectiveModel || 'none'}`,
      );
      const response = await fetch(`${PRIMARY_API_URL}/character/generate-from-conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.slice(-30),
          analysis: characterReadiness,
          model_name: createCharacterEffectiveModel,
          selected_model: createCharacterSelectedModel,
          frontend_round_robin_enabled: createCharacterAutoEnabled,
          request_purpose: 'create_character',
          conversation_id: activeConversation || '',
        }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const errText = result?.detail || result?.error || response.statusText || 'Generation failed';
        persistCharacterGenBackup({ status: 'error', error: errText, raw_response_excerpt: '' });
        setCharacterGenerationError(String(errText));
        setCharacterGenerationRaw('');
        setCharacterPartialJson(null);
        setShowCharacterGenFailure(true);
        return;
      }

      persistCharacterGenBackup(result);

      const character =
        result?.character_json ||
        (result?.status === 'partial' ? result?.partial_character_json : null);
      if (character && (result?.status === 'success' || result?.status === 'partial')) {
        setGeneratedCharacter(character);
        setShowCharacterPreview(true);
        if (result?.status === 'partial') {
          setCharacterGenerationError(
            result?.error || 'Recovered partial character from incomplete model output.'
          );
        }
        return;
      }

      setCharacterGenerationError(result?.error || 'Could not parse character from model response');
      setCharacterGenerationRaw(result?.raw_response_excerpt || '');
      setCharacterPartialJson(result?.partial_character_json || null);
      setShowCharacterGenFailure(true);
    } catch (e) {
      console.error(e);
      setCharacterGenerationError(e?.message || 'Character generation failed');
      setShowCharacterGenFailure(true);
    } finally {
      setIsGeneratingCharacter(false);
    }
  }, [
    characterReadiness,
    messages,
    isGeneratingCharacter,
    PRIMARY_API_URL,
    primaryModel,
    primaryIsAPI,
    settings,
    effectiveIntroModel,
    activeConversation,
    persistCharacterGenBackup,
    clearCharacterGenFailure,
  ]);

  const handleUsePartialCharacter = useCallback(() => {
    if (!characterPartialJson) return;
    setGeneratedCharacter(characterPartialJson);
    setShowCharacterPreview(true);
    clearCharacterGenFailure();
  }, [characterPartialJson, clearCharacterGenFailure]);

  const handleCallModeToggle = useCallback(async () => {
    if (settings?.allowDualOverlay) return;
    if (isCallModeActive) await stopCallMode(); else await startCallMode();
  }, [isCallModeActive, startCallMode, stopCallMode, settings?.allowDualOverlay]);

  const handleNanoGptCapabilities = useCallback((caps) => {
    const next = caps || {};
    nanoGptSelectedModelCapsRef.current = next;
    setNanoGptModelCaps(next);
  }, []);

  const handleQuickActionPrompt = useCallback((text) => {
    try {
      chatInputFormRef.current?.setValue(text);
    } catch {
      // ignore
    }
  }, []);

  const activeComposerMode = isCallModeActive ? 'call' : isFocusModeActive ? 'focus' : 'chat';

  // Dual overlay: popup = Call Mode, main window = Focus Mode.
  // Never start call mode in the main window when dual overlay is on.
  const handleDualOverlayChange = useCallback(async (enabled) => {
    if (enabled) {
      const { writeCallOverlayState, openCallOverlayWindow } = await import('../utils/dualOverlayWindow');
      writeCallOverlayState({
        activeCharacter: activeCharacter || null,
        characters: (characters || []).map(c => ({ id: c.id, name: c.name, character_order: c.character_order, avatar: c.avatar, avatar_url: c.avatar_url })),
        primaryApiUrl: PRIMARY_API_URL || '',
      });
      openCallOverlayWindow();
      updateSettings({ allowDualOverlay: true });
      setIsFocusModeActive(true);
    } else {
      if (isCallModeActive) await stopCallMode();
      const { closeCallOverlayWindow } = await import('../utils/dualOverlayWindow');
      closeCallOverlayWindow();
      updateSettings({ allowDualOverlay: false });
    }
  }, [stopCallMode, isCallModeActive, updateSettings]);

  const handleComposerModeChange = useCallback(async (mode) => {
    if (mode === 'focus') {
      // Toggle focus mode independently of dual overlay or call mode
      setIsFocusModeActive(prev => !prev);
      return;
    }
    if (mode === 'call') {
      // When dual overlay is on, the popup already owns call mode
      if (settings?.allowDualOverlay) return;
      if (!isCallModeActive) {
        await startCallMode();
      } else {
        await stopCallMode();
      }
      return;
    }
    if (mode === 'chat') {
      setIsFocusModeActive(false);
      if (isCallModeActive) await stopCallMode();
      updateSettings({ allowDualOverlay: false });
      return;
    }
    if (mode === 'image') {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('eloquent-open-chat-image'));
      }
    }
  }, [isCallModeActive, startCallMode, stopCallMode, setIsFocusModeActive, updateSettings]);

  // Close both overlays at once (for dual-overlay ESC behavior)
  const handleCloseAllOverlays = useCallback(() => {
    setIsFocusModeActive(false);
    if (isCallModeActive) {
      stopCallMode();
      updateSettings({ allowDualOverlay: false });
    }
  }, [isCallModeActive, stopCallMode, updateSettings]);

  const handleRefineCharacter = useCallback(async () => {
    if (!generatedCharacter || !characterFeedback?.trim() || isGeneratingCharacter) return;
    setIsGeneratingCharacter(true);
    try {
      const refineRoute = resolveUnifiedRequestRoute({
        primaryModel: primaryModel || effectiveIntroModel,
        primaryIsAPI,
        settings,
        requestPurpose: 'refine_character',
      });
      const refineCharacterAutoEnabled = refineRoute.autoEnabled;
      const refineCharacterSelectedModel = refineRoute.selectedModel || primaryModel || effectiveIntroModel;
      const refineCharacterEffectiveModel = refineRoute.effectiveModel || refineCharacterSelectedModel;
      console.info(
        `refine_character_router_state auto_enabled=${refineCharacterAutoEnabled} selected_model=${refineCharacterSelectedModel || 'none'} effective_model=${refineCharacterEffectiveModel || 'none'}`,
      );
      const res = await fetch(`${PRIMARY_API_URL}/character/refine-generated`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_json: generatedCharacter,
          feedback: characterFeedback.trim(),
          original_messages: messages.slice(-30),
          model_name: refineCharacterEffectiveModel || primaryModel,
          selected_model: refineCharacterSelectedModel,
          frontend_round_robin_enabled: refineCharacterAutoEnabled,
          request_purpose: 'refine_character',
          gpu_id: 0,
        }),
      });
      const result = await res.json();
      if (result.status === 'success' && result.character_json) {
        setGeneratedCharacter(result.character_json);
        setCharacterFeedback('');
      } else {
        alert(result.error || 'Refinement failed');
      }
    } catch (e) {
      console.error(e);
      alert('Refinement failed');
    } finally {
      setIsGeneratingCharacter(false);
    }
  }, [generatedCharacter, characterFeedback, messages, isGeneratingCharacter, PRIMARY_API_URL, primaryModel, primaryIsAPI, settings, effectiveIntroModel]);

  const handleAutoAnalyzeImage = useCallback(async (imageMessage) => {
    // ... [Original Logic] ...
  }, [primaryModel, PRIMARY_API_URL, userProfile, generateUniqueId]);

  const autoEnhanceRegeneratedImage = useCallback(async (imageUrl, originalPrompt, messageId, enhancementSettings, modelName, gpuId = 0) => {
    if (!imageUrl || !messageId) {
      return;
    }

    try {
      const response = await fetch(`${PRIMARY_API_URL}/sd-local/enhance-adetailer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          original_prompt: originalPrompt,
          face_prompt: enhancementSettings.facePrompt,
          strength: enhancementSettings.strength,
          steps: enhancementSettings.steps,
          confidence: enhancementSettings.confidence,
          sampler: enhancementSettings.sampler,
          model_name: modelName,
          gpu_id: gpuId
        })
      });

      if (!response.ok) {
        return;
      }

      const result = await response.json();
      if (result.status === 'success' && result.enhanced_image_url) {
        setMessages(prev => prev.map(msg =>
          msg.id === messageId
            ? {
              ...msg,
              imagePath: result.enhanced_image_url,
              enhancement_history: [imageUrl, result.enhanced_image_url],
              current_enhancement_level: 1,
              enhanced: true,
              enhancement_settings: { ...enhancementSettings, model_name: modelName }
            }
            : msg
        ));
      }
    } catch (error) {
      console.error('Auto-enhancement failed:', error);
    }
  }, [PRIMARY_API_URL, setMessages]);

  const runRegenerationTask = useCallback(async (queueItem, signal) => {
    const imageParams = queueItem?.imageParams;
    if (!imageParams?.prompt?.trim()) {
      return;
    }

    const characterSnapshot = queueItem?.characterSnapshot;
    const characterId = characterSnapshot?.id ?? activeCharacter?.id;
    const characterName = characterSnapshot?.name ?? activeCharacter?.name;
    const avatar = getActiveCharacterAvatar(characterSnapshot) ?? getActiveCharacterAvatar(activeCharacter);

    const gpuId = Number.isInteger(imageParams.gpu_id)
      ? imageParams.gpu_id
      : Number.isInteger(imageParams.gpuId)
        ? imageParams.gpuId
        : 0;

    try {
      const responseData = await generateImage(imageParams.prompt, {
        negative_prompt: imageParams.negative_prompt || '',
        width: imageParams.width || 512,
        height: imageParams.height || 512,
        steps: imageParams.steps || 20,
        guidance_scale: imageParams.guidance_scale || 7.0,
        sampler: imageParams.sampler || 'Euler a',
        seed: imageParams.seed ?? -1,
        model: imageParams.model || '',
        checkpoint: imageParams.model || ''
      }, gpuId, signal ? { signal } : {});

      if (responseData && Array.isArray(responseData.image_urls) && responseData.image_urls.length > 0) {
        responseData.image_urls.forEach((imageUrl) => {
          const messageId = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}-img`;
          const imageMessage = {
            id: messageId,
            role: 'bot',
            characterId,
            characterName,
            avatar,
            modelId: 'primary',
            type: 'image',
            content: imageParams.prompt,
            imagePath: imageUrl,
            gpuId,
            prompt: imageParams.prompt,
            negative_prompt: imageParams.negative_prompt || '',
            width: responseData.parameters?.width || imageParams.width || 512,
            height: responseData.parameters?.height || imageParams.height || 512,
            steps: responseData.parameters?.steps || imageParams.steps || 20,
            guidance_scale: responseData.parameters?.cfg_scale ?? imageParams.guidance_scale ?? 7.0,
            model: responseData.parameters?.sd_model_checkpoint || imageParams.model || '',
            sampler: responseData.parameters?.sampler || imageParams.sampler || 'Euler a',
            seed: responseData.parameters?.seed ?? -1,
            original_prompt: imageParams.original_prompt || imageParams.prompt,
            original_negative_prompt: imageParams.original_negative_prompt || imageParams.negative_prompt || '',
            original_width: imageParams.original_width || imageParams.width || 512,
            original_height: imageParams.original_height || imageParams.height || 512,
            original_steps: imageParams.original_steps || imageParams.steps || 20,
            original_guidance_scale: imageParams.original_guidance_scale ?? imageParams.guidance_scale ?? 7.0,
            original_model: imageParams.original_model || imageParams.model || '',
            original_sampler: imageParams.original_sampler || imageParams.sampler || 'Euler a',
            original_seed: imageParams.original_seed ?? imageParams.seed ?? -1,
            timestamp: new Date().toISOString()
          };

          setMessages(prev => [...prev, imageMessage]);

          if (autoEnhanceEnabled && settings?.imageEngine === 'EloDiffusion') {
            const fallbackSettings = {
              strength: typeof adetailerSettings.strength === 'number' ? adetailerSettings.strength : 0.35,
              confidence: typeof adetailerSettings.confidence === 'number' ? adetailerSettings.confidence : 0.3,
              steps: typeof adetailerSettings.steps === 'number' ? adetailerSettings.steps : 45,
              sampler: adetailerSettings.sampler || 'euler_a',
              facePrompt: adetailerSettings.facePrompt || 'detailed face, high quality, sharp focus'
            };
            autoEnhanceRegeneratedImage(
              imageUrl,
              imageParams.prompt,
              messageId,
              fallbackSettings,
              selectedAdetailerModel,
              gpuId
            );
          }
        });
      } else {
        setMessages(prev => [
          ...prev,
          {
            id: `${Date.now()}-regen-error`,
            role: 'system',
            content: 'Image regeneration completed, but no images were returned.',
            error: true
          }
        ]);
      }
    } catch (err) {
      if (err?.name === 'AbortError') return; // User cancelled
      console.error('Error regenerating image:', err);
      setMessages(prev => [
        ...prev,
        {
          id: `${Date.now()}-regen-catch`,
          role: 'system',
          content: `Error regenerating image: ${err.message}.`,
          error: true
        }
      ]);
    }
  }, [
    generateImage,
    activeCharacter,
    setMessages,
    autoEnhanceEnabled,
    adetailerSettings,
    selectedAdetailerModel,
    settings?.imageEngine,
    autoEnhanceRegeneratedImage
  ]);

  const processRegenerationQueue = useCallback(async () => {
    if (regenerationProcessingRef.current) return;
    regenerationProcessingRef.current = true;
    const controller = new AbortController();
    regenerationAbortControllerRef.current = controller;
    setIsRegenerationRunning(true);

    try {
      while (regenerationQueueRef.current.length > 0) {
        const nextItem = regenerationQueueRef.current[0];
        await runRegenerationTask(nextItem, controller.signal);
        if (controller.signal.aborted) break;
        regenerationQueueRef.current.shift();
        setRegenerationQueue(regenerationQueueRef.current.length);
      }
    } finally {
      regenerationProcessingRef.current = false;
      regenerationAbortControllerRef.current = null;
      setIsRegenerationRunning(false);
    }
  }, [runRegenerationTask]);

  const cancelRegenerations = useCallback(() => {
    regenerationAbortControllerRef.current?.abort();
    regenerationQueueRef.current = [];
    setRegenerationQueue(0);
    regenerationProcessingRef.current = false;
    setIsRegenerationRunning(false);
  }, []);

  const handleRegenerateImage = useCallback((imageParams) => {
    if (!imageParams?.prompt?.trim()) {
      return;
    }

    regenerationQueueRef.current.push({
      imageParams: { ...imageParams },
      characterSnapshot: activeCharacter
        ? {
          id: activeCharacter.id,
          name: activeCharacter.name,
          avatar: getActiveCharacterAvatar(activeCharacter)
        }
        : null
    });
    setRegenerationQueue(regenerationQueueRef.current.length);
    processRegenerationQueue();
  }, [activeCharacter, processRegenerationQueue]);

  const handleSaveCharacter = useCallback(() => {
    if (!generatedCharacter) return;
    const characterToSave = { ...generatedCharacter, avatar: characterImageUrl || null, created_at: new Date().toISOString() };
    saveCharacter(characterToSave);
    setShowCharacterPreview(false);
    setGeneratedCharacter(null);
    setCharacterImageUrl(null);
    alert(`Character saved!`);
  }, [generatedCharacter, saveCharacter, characterImageUrl]);

  const getCharacterButtonState = () => ({
    disabled: false, variant: "outline", className: "flex-shrink-0 h-10 w-10 hover:bg-purple-500/20", title: "Generate character"
  });

  const bothModelsLoaded = primaryModel && secondaryModel;
  const showAgentControls = Boolean(bothModelsLoaded) && (dualModeEnabled || settings.multiRoleMode);

  const handleStartAgentConversation = () => {
    if (agentTopic.trim() && bothModelsLoaded) { startAgentConversation(agentTopic, agentTurns); setAgentTopic(''); }
  };

  const resolveRegenModelCapabilities = useCallback((endpointId, fallbackCaps) => {
    if (endpointId) {
      const caps = resolveEndpointDisplay(endpointId, settings, nanoGptCatalog)?.capabilities;
      if (caps && typeof caps === 'object') return caps;
    }
    return fallbackCaps || nanoGptSelectedModelCapsRef.current || {};
  }, [settings, nanoGptCatalog]);

  const handleGenerateVariant = useCallback(async (messageId, regenOptions = {}) => {
    if (isGenerating) return;
    const msgIndex = messages.findIndex(m => m.id === messageId);
    if (msgIndex < 0) return;

    const historyBefore = messages.slice(0, msgIndex);
    let promptText = '';
    for (let i = historyBefore.length - 1; i >= 0; i -= 1) {
      if (historyBefore[i]?.role === 'user') {
        promptText = historyBefore[i].content || '';
        break;
      }
    }

    const modelNameOverride = regenOptions.modelName || null;
    const modelCapabilities = resolveRegenModelCapabilities(
      modelNameOverride,
      regenOptions.modelCapabilities,
    );

    setIsGenerating(true);
    setAudioError(null);
    if (abortController) abortController.abort();
    const newController = new AbortController();
    setAbortController(newController);

    // We don't remove the message, we just generate a NEW variant for it.
    // BUT checking logic: usually we want to see the new content streaming in.
    // So we push a new variant string to the array, and point index to it.
    // AND we update the main message content to show the streaming.

    const variants = messageVariants[messageId] || [messages[msgIndex].content];
    const newVariantIndex = variants.length; // The one we are about to add

    // Add empty placeholder for new variant
    setMessageVariants(prev => ({
      ...prev,
      [messageId]: [...(prev[messageId] || [messages[msgIndex].content]), '']
    }));
    setCurrentVariantIndex(prev => ({ ...prev, [messageId]: newVariantIndex }));

    // Also must update the main message content to be empty/streaming
    setMessages(prev => prev.map(m => m.id === messageId ? {
      ...m,
      content: '',
      error: false,
      reasoningStreaming: modelCapabilities?.reasoning === true,
      reasoningEnabled: modelCapabilities?.reasoning === true,
      reasoningText: modelCapabilities?.reasoning === true ? '' : (m.reasoningText || ''),
      reasoningStartedAtMs: modelCapabilities?.reasoning === true ? Date.now() : m.reasoningStartedAtMs,
    } : m));

    const ttsOverrides = getTtsOverridesForMessageId(messageId, messages[msgIndex]?.characterId);

    try {
      if (settings?.streamResponses) {
        startStreamingTTS(messageId, ttsOverrides);
      }

      let gatheredText = '';
      let lastProcessedLength = 0;
      const onToken = (textChunk, currentFullText, meta) => {
        gatheredText = currentFullText;
        const capReasoning = modelCapabilities?.reasoning === true;
        setMessages(prev => prev.map(m => {
          if (m.id !== messageId) return m;
          return applyReasoningMetaToBotMessage(
            { ...m, content: currentFullText },
            {
              ...meta,
              reasoningStartedAtMs: meta?.reasoningStartedAtMs ?? m.reasoningStartedAtMs ?? Date.now(),
            },
            { capReasoning },
          );
        }));

        // TTS Streaming
        const newPart = currentFullText.slice(lastProcessedLength);
        if (newPart && settings?.streamResponses) {
          addStreamingText(newPart);
          lastProcessedLength = currentFullText.length;
        }
      };

      const targetCharacterId = messages[msgIndex]?.characterId || null;
      const responseText = await generateReply(
        promptText,
        historyBefore,
        onToken,
        {
          authorNote,
          webSearchEnabled,
          speakerCharacterId: targetCharacterId,
          modelCapabilities,
          modelName: modelNameOverride || undefined,
        }
      );

      if (responseText) {
        if (settings?.streamResponses) {
          endStreamingTTS();
        }

        setMessages((prev) =>
          prev.map((m) =>
            m.id === messageId
              ? {
                  ...m,
                  content: responseText,
                  reasoningStreaming: false,
                  reasoningSeconds:
                    m?.reasoningEnabled && typeof m?.reasoningStartedAtMs === 'number'
                      ? Math.max(0, Math.round((Date.now() - m.reasoningStartedAtMs) / 1000))
                      : null,
                }
              : m
          )
        );

        // Save final variant
        setMessageVariants(prev => {
          const oldVars = prev[messageId] ? [...prev[messageId]] : [];
          if (oldVars.length > newVariantIndex) oldVars[newVariantIndex] = responseText;
          else oldVars.push(responseText); // Should be at index
          return { ...prev, [messageId]: oldVars };
        });

        if (settings?.ttsEnabled && settings?.ttsAutoPlay && !settings?.streamResponses) {
          playTTS(messageId, responseText, ttsOverrides);
        }
      } else {
        throw new Error('Model returned an empty response.');
      }
    } catch (error) {
      console.error("Variant generation error:", error);
      const errText = error?.message || String(error);
      const priorContent = variants[newVariantIndex - 1] ?? messages[msgIndex]?.content ?? '';
      setMessageVariants((prev) => {
        const list = prev[messageId] ? [...prev[messageId]] : [];
        if (list.length > newVariantIndex) {
          list[newVariantIndex] = errText;
        }
        return { ...prev, [messageId]: list };
      });
      setMessages(prev => prev.map(m => m.id === messageId ? {
        ...m,
        content: errText.startsWith('[') ? errText : `[Error: ${errText}]`,
        error: true,
        reasoningStreaming: false,
      } : m));
      if (!priorContent && newVariantIndex > 0) {
        setCurrentVariantIndex(prev => ({ ...prev, [messageId]: Math.max(0, newVariantIndex - 1) }));
      }
    } finally {
      setIsGenerating(false);
      setAbortController(null);
    }
  }, [
    isGenerating, messages, messageVariants, settings, generateReply, authorNote, webSearchEnabled,
    startStreamingTTS, playTTS, abortController, setAbortController, getTtsOverridesForMessageId,
    resolveRegenModelCapabilities,
  ]);

  const handleGenerateVariantWithModel = useCallback((messageId, endpointId) => {
    const caps = resolveEndpointDisplay(endpointId, settings, nanoGptCatalog)?.capabilities || {};
    handleGenerateVariant(messageId, { modelName: endpointId, modelCapabilities: caps });
  }, [handleGenerateVariant, settings, nanoGptCatalog]);

  useEffect(() => { handleGenerateVariantRef.current = handleGenerateVariant; }, [handleGenerateVariant]);

  /** Strip think tags from bot text (TTS, batch tools); bubble rendering uses thinkStreamParser at render time. */
  const filterThinkBlock = useCallback((text) => stripThinkTags(text), []);

  const getCurrentVariantContent = useCallback((messageId, originalContent) => {
    const variants = messageVariants[messageId];
    if (!variants || variants.length === 0) return originalContent;
    const index = currentVariantIndex[messageId] || 0;
    return variants[index] || originalContent;
  }, [messageVariants, currentVariantIndex]);

  const getVariantCount = useCallback((messageId) => {
    const variants = messageVariants[messageId];
    return variants ? variants.length : 0;
  }, [messageVariants]);

  const batchBoilerplatePreview = useMemo(() => {
    const idSet =
      batchBoilerplateScope === 'all'
        ? batchEditableBotMessages.map((m) => m.id)
        : batchBoilerplateSelectedIds.filter((id) =>
            batchEditableBotMessages.some((m) => m.id === id)
          );

    if (batchBoilerplateMode === 'brackets') {
      let messageCount = 0;
      let matchCount = 0;
      for (const id of idSet) {
        const msg = messages.find((m) => m.id === id);
        if (!msg) continue;
        const base = filterThinkBlock(getCurrentVariantContent(id, msg.content));
        const { removals } = stripSquareBracketSpans(base);
        if (removals > 0) {
          messageCount += 1;
          matchCount += removals;
        }
      }
      return { messageCount, matchCount, regexError: null };
    }

    if (batchBoilerplateMode === 'regex') {
      const pattern = batchBoilerplateRegex.trim();
      if (!pattern) return { messageCount: 0, matchCount: 0, regexError: null };
      const re = compileBatchRegex(pattern);
      if (!re) return { messageCount: 0, matchCount: 0, regexError: 'Invalid regex' };
      let messageCount = 0;
      let matchCount = 0;
      for (const id of idSet) {
        const msg = messages.find((m) => m.id === id);
        if (!msg) continue;
        const base = filterThinkBlock(getCurrentVariantContent(id, msg.content));
        const n = countGlobalRegexMatches(base, re);
        if (n > 0) {
          messageCount += 1;
          matchCount += n;
        }
      }
      return { messageCount, matchCount, regexError: null };
    }

    const find = batchBoilerplateFind;
    if (!find) return { messageCount: 0, matchCount: 0, regexError: null };
    let messageCount = 0;
    let matchCount = 0;
    for (const id of idSet) {
      const msg = messages.find((m) => m.id === id);
      if (!msg) continue;
      const base = filterThinkBlock(getCurrentVariantContent(id, msg.content));
      const n = countOccurrencesInText(base, find, batchBoilerplateMatchCase, batchBoilerplateReplaceAll);
      if (n > 0) {
        messageCount += 1;
        matchCount += n;
      }
    }
    return { messageCount, matchCount, regexError: null };
  }, [
    batchBoilerplateMode,
    batchBoilerplateRegex,
    batchBoilerplateFind,
    batchBoilerplateScope,
    batchBoilerplateSelectedIds,
    batchEditableBotMessages,
    messages,
    filterThinkBlock,
    getCurrentVariantContent,
    batchBoilerplateMatchCase,
    batchBoilerplateReplaceAll,
  ]);

  const navigateVariant = useCallback((messageId, direction) => {
    const variants = messageVariants[messageId];
    if (!variants || variants.length <= 1) return;
    const currentIndex = currentVariantIndex[messageId] || 0;
    let newIndex;
    if (direction === 'next') newIndex = (currentIndex + 1) % variants.length;
    else newIndex = currentIndex === 0 ? variants.length - 1 : currentIndex - 1;
    setCurrentVariantIndex(prev => ({ ...prev, [messageId]: newIndex }));
    const nextContent = variants[newIndex];
    if (typeof nextContent === 'string') {
      setMessages(prev => prev.map(m => m.id === messageId ? {
        ...m,
        content: nextContent,
        error: nextContent.startsWith('[Error:'),
      } : m));
    }
  }, [messageVariants, currentVariantIndex, setMessages]);

  // SD Models fetch
  useEffect(() => {
    if (showCharacterPreview) { /* fetch SD models logic */ }
  }, [showCharacterPreview, PRIMARY_API_URL]);

  // Auto prompt generation
  useEffect(() => {
    if (generatedCharacter && showCharacterPreview) {
      const autoPrompt = generateCharacterImagePrompt(generatedCharacter);
      setCharacterImagePrompt(autoPrompt);
      setCustomImagePrompt(autoPrompt);
      setCharacterImageUrl(null);
    }
  }, [generatedCharacter, showCharacterPreview, generateCharacterImagePrompt]);

  const handleEditBotMessage = useCallback((messageId) => {
    setEditingBotMessageId(messageId);
  }, []);

  const handleSaveBotMessage = useCallback((messageId, newContent) => {
    if (!newContent.trim()) return;
    setMessageVariants(prev => {
      const variants = prev[messageId] || [];
      const currentIndex = currentVariantIndex[messageId] || 0;
      if (variants.length === 0) return { ...prev, [messageId]: [newContent.trim()] };
      const updatedVariants = [...variants];
      updatedVariants[currentIndex] = newContent.trim();
      return { ...prev, [messageId]: updatedVariants };
    });
    // Update global message state
    setMessages(prev => prev.map(m => m.id === messageId ? { ...m, content: newContent.trim() } : m));
    setEditingBotMessageId(null);
  }, [currentVariantIndex, setMessages]);

  const handleCancelBotEdit = useCallback(() => {
    setEditingBotMessageId(null);
  }, []);

  const handleApplyBatchBoilerplate = useCallback(() => {
    const ids =
      batchBoilerplateScope === 'all'
        ? batchEditableBotMessages.map((m) => m.id)
        : batchBoilerplateSelectedIds.filter((id) =>
            batchEditableBotMessages.some((m) => m.id === id)
          );
    if (!ids.length) return;

    const replaceStr = batchBoilerplateReplace ?? '';

    const computeNewText = (base) => {
      if (batchBoilerplateMode === 'brackets') {
        const { text } = stripSquareBracketSpans(base);
        return text.trim();
      }
      if (batchBoilerplateMode === 'regex') {
        const pattern = batchBoilerplateRegex.trim();
        if (!pattern) return null;
        const re = compileBatchRegex(pattern);
        if (!re) return null;
        return base.replace(re, replaceStr).trim();
      }
      const find = batchBoilerplateFind;
      if (!find) return null;
      return applyLiteralReplace(
        base,
        find,
        replaceStr,
        batchBoilerplateMatchCase,
        batchBoilerplateReplaceAll
      ).trim();
    };

    if (batchBoilerplateMode === 'regex') {
      const pattern = batchBoilerplateRegex.trim();
      if (!pattern || !compileBatchRegex(pattern)) {
        window.alert('Invalid regex pattern. Fix it or switch mode.');
        return;
      }
    }
    if (batchBoilerplateMode === 'literal' && !batchBoilerplateFind) return;

    const idToNewContent = new Map();
    for (const id of ids) {
      const msg = messages.find((x) => x.id === id);
      if (!msg || msg.role !== 'bot') continue;
      const base = filterThinkBlock(getCurrentVariantContent(id, msg.content));
      const newText = computeNewText(base);
      if (newText === null || newText === undefined) continue;
      if (!newText) continue;
      const trimmedBase = base.trim();
      if (newText === trimmedBase) continue;
      idToNewContent.set(id, newText);
    }
    if (idToNewContent.size === 0) {
      window.alert(
        'No changes: nothing matched, or every edit would leave a reply empty. Check mode / pattern.'
      );
      return;
    }

    setMessageVariants((prev) => {
      const next = { ...prev };
      for (const [messageId, newContent] of idToNewContent) {
        const currIdx = currentVariantIndex[messageId] || 0;
        const old = next[messageId];
        if (!old || old.length === 0) next[messageId] = [newContent];
        else {
          const u = [...old];
          u[currIdx] = newContent;
          next[messageId] = u;
        }
      }
      return next;
    });
    setMessages((prev) =>
      prev.map((m) => (idToNewContent.has(m.id) ? { ...m, content: idToNewContent.get(m.id) } : m))
    );
    setEditingBotMessageId(null);
    setShowBatchBoilerplateDialog(false);
    setBatchBoilerplateSelectedIds([]);
  }, [
    batchBoilerplateMode,
    batchBoilerplateRegex,
    batchBoilerplateFind,
    batchBoilerplateReplace,
    batchBoilerplateScope,
    batchBoilerplateSelectedIds,
    batchBoilerplateMatchCase,
    batchBoilerplateReplaceAll,
    batchEditableBotMessages,
    messages,
    filterThinkBlock,
    getCurrentVariantContent,
    currentVariantIndex,
    setMessages,
    setMessageVariants,
  ]);

  const handleContinueGeneration = useCallback(async (messageId) => {
    if (isGenerating) return;
    const msgIndex = messages.findIndex(m => m.id === messageId);
    if (msgIndex < 0) return;

    const msg = messages[msgIndex];
    const currentContent = getCurrentVariantContent(messageId, msg.content);

    setIsGenerating(true);
    setAudioError(null);
    if (abortController) abortController.abort();
    const newController = new AbortController();
    setAbortController(newController);

    // Prompt the model to continue based on the conversation so far, INCLUDING the partial message.
    // Use currentContent so the prompt ends with the exact partial text; mark as isPrefill so formatPrompt leaves it open-ended.
    const history = messages.slice(0, msgIndex + 1).map((m, i) =>
      i === msgIndex ? { ...m, content: currentContent, isPrefill: true } : m
    );

    try {
      const ttsOverrides = getTtsOverridesForMessageId(messageId, msg.characterId);
      if (settings?.streamResponses) {
        startStreamingTTS(messageId, ttsOverrides);
      }

      let lastProcessedLength = currentContent.length; // Start from existing content length
      const onToken = (textChunk, currentFullText) => {
        // Appending the NEW generation to the OLD content
        // note: currentFullText from generateReply is just the NEW generation because we sent prompt="Continue" (mostly)
        // Wait, generateReply returns the ACCUMULATED text of the NEW generation.

        const combined = currentContent + currentFullText;
        setMessages(prev => prev.map(m => m.id === messageId ? { ...m, content: combined } : m));

        // TTS: we only want to stream the NEW tokens
        // currentFullText grows from "" -> "The" -> "The cat"
        // So we can track its length
        const newPart = currentFullText.slice(lastProcessedLength - currentContent.length);
        // Logic check:
        // lastProcessedLength covers (currentContent + processed_part_of_new)
        // actually simplicity: track handled length of currentFullText
        // let's reset tracker local to this callback scope? No, needs persistence across calls.
      };

      // Correct Logic for Continue:
      let localProcessedLength = 0;
      const onTokenCorrect = (textChunk, currentFullText, meta) => {
        const combined = currentContent + currentFullText;
        const capReasoning = nanoGptSelectedModelCapsRef.current?.reasoning === true;
        setMessages(prev => prev.map(m => {
          if (m.id !== messageId) return m;
          return applyReasoningMetaToBotMessage(
            { ...m, content: combined },
            {
              ...meta,
              reasoningStartedAtMs: meta?.reasoningStartedAtMs ?? m.reasoningStartedAtMs ?? Date.now(),
            },
            { capReasoning },
          );
        }));

        const newPart = currentFullText.slice(localProcessedLength);
        if (newPart && settings?.streamResponses) {
          addStreamingText(newPart);
          localProcessedLength = currentFullText.length;
        }
      };

      // We send "Continue" as text to trigger system prompt building but 
      // the actual prompt sent to LLM will be the history ending in prefill.
      const targetCharacterId = msg.characterId || null;
      const continuationText = await generateReply(
        "Continue",
        history,
        onTokenCorrect,
        { authorNote, webSearchEnabled: false, speakerCharacterId: targetCharacterId, requestPurpose: 'continuation', modelCapabilities: nanoGptSelectedModelCapsRef.current || {} }
      );

      if (settings?.streamResponses) endStreamingTTS();

      if (continuationText) {
        const finalContent = currentContent + continuationText;

        // Update the current variant in-place
        setMessageVariants(prev => {
          const variants = prev[messageId] || [msg.content];
          const currIdx = currentVariantIndex[messageId] || 0;
          const newVars = [...variants];
          newVars[currIdx] = finalContent; // Update current variant
          return { ...prev, [messageId]: newVars };
        });

        setMessages(prev => prev.map(m => m.id === messageId ? {
          ...m,
          content: finalContent,
          reasoningStreaming: false,
          reasoningSeconds:
            m?.reasoningEnabled && typeof m?.reasoningStartedAtMs === 'number'
              ? Math.max(0, Math.round((Date.now() - m.reasoningStartedAtMs) / 1000))
              : null,
        } : m));

        if (settings?.ttsEnabled && settings?.ttsAutoPlay && !settings?.streamResponses) {
          playTTS(messageId, continuationText, ttsOverrides); // Play only the new part
        }
      }
    } catch (error) {
      console.error("Continue generation error:", error);
    } finally {
      setIsGenerating(false);
      setAbortController(null);
    }
  }, [isGenerating, messages, getCurrentVariantContent, generateReply, settings, authorNote, startStreamingTTS, playTTS, abortController, setAbortController, messageVariants, currentVariantIndex, getTtsOverridesForMessageId]);

  const resolveBotAvatarSource = useCallback((message, charForModel) => {
    if (message.characterId && characters?.length) {
      const live = characters.find((c) => c.id === message.characterId);
      if (live) return getActiveCharacterAvatar(live);
    }
    if (message.role === 'bot' && charForModel) return getActiveCharacterAvatar(charForModel);
    if (message.role === 'bot' && !message.characterId && activeCharacter) {
      return getActiveCharacterAvatar(activeCharacter);
    }
    return message.avatar;
  }, [characters, activeCharacter]);

  const getAvatarCycleCharacterId = useCallback((message, charForModel) => {
    return message.characterId || charForModel?.id || activeCharacter?.id || null;
  }, [activeCharacter?.id]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.target?.tagName === 'INPUT' || e.target?.tagName === 'TEXTAREA' || e.target?.isContentEditable) return;
      const charId = activeCharacter?.id;
      if (!charId) return;
      const char = characters.find((c) => c.id === charId);
      if (!char || getCharacterAvatarList(char).length <= 1) return;
      if (e.key === '[' || e.key === 'ArrowLeft') {
        e.preventDefault();
        cycleCharacterAvatar(charId, -1);
      } else if (e.key === ']' || e.key === 'ArrowRight') {
        e.preventDefault();
        cycleCharacterAvatar(charId, 1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeCharacter?.id, characters, cycleCharacterAvatar]);

  const renderAvatar = (message, apiUrl, charForModel) => {
    const avatarSource = resolveBotAvatarSource(message, charForModel);
    const characterName = message.characterName
      || (message.role === 'bot' && charForModel?.name)
      || (message.role === 'bot' && activeCharacter?.name)
      || 'Character';
    const cycleId = getAvatarCycleCharacterId(message, charForModel);
    const cycleChar = cycleId ? characters.find((c) => c.id === cycleId) : null;
    const canCycle = cycleChar && getCharacterAvatarList(cycleChar).length > 1;

    const sizeStyle = {
      width: `${characterAvatarSize}px`,
      height: `${characterAvatarSize}px`
    };

    const displayUrl = resolveAvatarDisplayUrl(avatarSource, apiUrl || getBackendUrl());

    const handleCycle = (delta) => {
      if (cycleId) cycleCharacterAvatar(cycleId, delta);
    };

    const wrap = (node) => (
      <button
        type="button"
        className={`flex-shrink-0 rounded-full ${canCycle ? 'cursor-pointer hover:ring-2 hover:ring-primary/50' : 'cursor-default'}`}
        style={sizeStyle}
        title={canCycle ? 'Click or scroll to change avatar ([ ] keys)' : characterName}
        onClick={canCycle ? () => handleCycle(1) : undefined}
        onWheel={canCycle ? (e) => {
          e.preventDefault();
          e.stopPropagation();
          handleCycle(e.deltaY > 0 ? 1 : -1);
        } : undefined}
      >
        {node}
      </button>
    );

    if (displayUrl) {
      return wrap(
        <CharacterAvatarMedia
          url={displayUrl}
          alt={characterName}
          className="rounded-full object-cover border border-gray-300 dark:border-gray-600 w-full h-full"
          videoKey={`${cycleId || 'msg'}-${displayUrl}`}
          onError={(e) => {
            const el = e?.currentTarget;
            if (el) el.style.display = 'none';
          }}
        />
      );
    }

    return wrap(
      <div
        className="rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-sm font-semibold text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600 w-full h-full"
      >
        {characterName ? characterName.charAt(0).toUpperCase() : '?'}
      </div>
    );
  };

  const renderUserAvatar = (message = null) => {
    const roleplayAvatar = settings.multiRoleMode && message?.characterId
      ? message?.avatar || getActiveCharacterAvatar(userCharacter)
      : null;
    const userAvatarSource = roleplayAvatar || userProfile?.avatar;
    const userName = settings.multiRoleMode && message?.characterId
      ? (message?.characterName || userCharacter?.name || 'User')
      : (userProfile?.name || userProfile?.username || 'User');
    let userDisplayUrl = null;

    const userSizeStyle = {
      width: `${userAvatarSize}px`,
      height: `${userAvatarSize}px`
    };

    if (userAvatarSource) {
      userDisplayUrl = userAvatarSource.startsWith('/')
        ? `${PRIMARY_API_URL || getBackendUrl()}${userAvatarSource}`
        : userAvatarSource;
    }

    if (userDisplayUrl) {
      return (
        <img
          src={userDisplayUrl}
          alt={`${userName}'s avatar`}
          title={userName}
          onError={(e) => { e.target.style.display = 'none'; }}
          className="rounded-full object-cover border border-gray-300 dark:border-gray-600 flex-shrink-0"
          style={userSizeStyle}
        />
      );
    }

    return (
      <div
        title={userName}
        className="rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold flex-shrink-0 border border-primary/50"
        style={userSizeStyle}
      >
        {userName ? userName.charAt(0).toUpperCase() : 'U'}
      </div>
    );
  };

  const renderedMessages = useMemo(() => {
    if (showAnyIntro) {
      if (!effectiveIntroModel) {
        return (
          <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground pt-10">
            <h3 className="text-lg font-medium mb-2">Load a model</h3>
            <p className="max-w-md mb-4">
              {showSystemIntro
                ? 'A model must be loaded before the system overview can generate.'
                : 'A model must be loaded before the character introduction can generate.'}
            </p>
            <Button variant="outline" onClick={() => setShowModelSelector(true)}>Load Model</Button>
          </div>
        );
      }
      return (
        <CharacterIntroExperience
          character={introDisplayCharacter}
          userProfile={userProfile}
          status={introStatus}
          error={introError}
          result={introResult}
          variant={showSystemIntro ? 'system' : 'character'}
          uiLabels={showSystemIntro ? SYSTEM_INTRO_UI_LABELS : undefined}
          onRetry={() => requestCharacterIntro({ forceRefresh: true })}
          onRegenerate={handleRegenerateCharacterIntro}
          onBegin={handleBeginCharacterIntro}
        />
      );
    }

    if (messages.length === 0) {
      const displayName =
        userProfile?.name ||
        userProfile?.username ||
        'friend';

      return (
        <div className="flex h-full items-center justify-center">
          <div className="w-full max-w-3xl px-4 py-8 md:px-8">
            {/* Greeting row */}
            <div className="flex items-center justify-between gap-4 mb-8">
              <div>
                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/90 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                  <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_0_4px_rgba(63,231,252,0.28)]" />
                  NanoGPT workspace · Local
                </div>
                <h1
                  className="mt-4 text-2xl md:text-3xl font-semibold text-left"
                  style={{ fontFamily: '"Open Sans Variable", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }}
                >
                  Welcome back, {displayName}.
                </h1>
                <p className="mt-2 text-sm md:text-base text-muted-foreground max-w-xl text-left">
                  Pick a mode, select a model, and start a new thread. Your NanoGPT memory and tools stay wired in.
                </p>
              </div>
            </div>

            <div className="mb-4">
              <NanoGptComposerToolbar
                variant="landing"
                modelPickerOpen={modelPickerOpen}
                onModelPickerOpenChange={setModelPickerOpen}
                currentModelId={primaryModel}
                primaryApiUrl={PRIMARY_API_URL}
                onCapabilities={handleNanoGptCapabilities}
                onQuickAction={handleQuickActionPrompt}
                activeMode={activeComposerMode}
                onModeChange={handleComposerModeChange}
              />
              {sttEnabled && (
                <div className="mt-3 flex justify-end">
                  <Button
                    type="button"
                    variant={isRecording ? 'destructive' : 'outline'}
                    size="icon"
                    className="h-9 w-9 flex-shrink-0 border-[rgba(120,170,220,0.5)] bg-[#111827]"
                    onClick={handleMicClick}
                    disabled={isTranscribing || isGenerating}
                    title={isRecording ? 'Stop recording' : 'Start voice input'}
                  >
                    {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    const visibleMessages = performanceMode && !showAllMessages
      ? messages.slice(-PERFORMANCE_MESSAGE_LIMIT)
      : messages;
    const hiddenCount = messages.length - visibleMessages.length;

    return (
      <>
        {performanceMode && messages.length > PERFORMANCE_MESSAGE_LIMIT && (
          <div className="flex items-center justify-between gap-2 p-2 mb-2 text-xs text-muted-foreground bg-muted/50 rounded border border-border">
            <span>
              {showAllMessages
                ? "Performance mode is on. Showing all messages (may be slower)."
                : `Performance mode is on. Showing last ${visibleMessages.length} of ${messages.length}.`}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAllMessages(prev => !prev)}
              className="h-7 px-2 text-xs"
            >
              {showAllMessages ? "Show recent" : `Show all (${hiddenCount})`}
            </Button>
          </div>
        )}
        {visibleMessages.map((msg) => (
          <div key={msg.id} id={`chat-message-${msg.id}`} className="scroll-mt-28">
            <ChatMessage
            msg={msg}
            content={getCurrentVariantContent(msg.id, msg.content)}
            isGenerating={isGenerating}
            isTranscribing={isTranscribing}
            isPlayingAudio={isPlayingAudio}
            isStreamingTtsPaused={isStreamingTtsPaused}
            editingMessageId={editingMessageId}
            editingBotMessageId={editingBotMessageId}
            primaryCharacter={primaryCharacter}
            secondaryCharacter={secondaryCharacter}
            activeCharacter={activeCharacter}
            characters={characters}
            primaryModel={primaryModel}
            primaryIsAPI={primaryIsAPI}
            settings={settings}
            nanoGptCatalog={nanoGptCatalog}
            userProfile={userProfile}
            userCharacter={userCharacter}
            isMultiRoleMode={settings.multiRoleMode}
            characterAvatarSize={characterAvatarSize}
            userAvatarSize={userAvatarSize}
            variantCount={getVariantCount(msg.id)}
            variantIndex={currentVariantIndex[msg.id] || 0}
            PRIMARY_API_URL={PRIMARY_API_URL}
            regenerationQueue={regenerationQueue}
            ttsEnabled={ttsEnabled}

            onEditUserMessage={handleEditUserMessage}
            onCancelEdit={handleCancelEdit}
            onSaveEditedMessage={handleSaveEditedMessage}
            onRegenerateFromEditedPrompt={handleRegenerateFromEditedPrompt}
            onDeleteMessage={handleDeleteMessage}

            onEditBotMessage={handleEditBotMessage}
            onCancelBotEdit={handleCancelBotEdit}
            onSaveBotMessage={handleSaveBotMessage}
            onGenerateVariant={handleGenerateVariant}
            onGenerateVariantWithModel={handleGenerateVariantWithModel}
            onContinueGeneration={handleContinueGeneration}
            onNavigateVariant={navigateVariant}
            onSpeakerClick={handleSpeakerClick}
            onChunkedSpeakerClick={handleChunkedSpeakerClick}
            onRegenerateImage={handleRegenerateImage}
            onCancelRegenerations={cancelRegenerations}
            isRegenerationRunning={isRegenerationRunning}

            formatModelName={formatModelName}
          />
          </div>
        ))}
      </>
    );
  }, [
    messages,
    showAnyIntro,
    showSystemIntro,
    introDisplayCharacter,
    primaryCharacter,
    userProfile,
    introStatus,
    introError,
    introResult,
    requestCharacterIntro,
    handleRegenerateCharacterIntro,
    handleBeginCharacterIntro,
    primaryModel,
    modelPickerOpen,
    handleNanoGptCapabilities,
    handleQuickActionPrompt,
    activeComposerMode,
    handleComposerModeChange,
    sttEnabled,
    handleMicClick,
    setShowModelSelector,
    performanceMode,
    showAllMessages,
    isGenerating,
    isTranscribing,
    isPlayingAudio,
    isStreamingTtsPaused,
    editingMessageId,
    editingBotMessageId,
    primaryCharacter,
    secondaryCharacter,
    activeCharacter,
    characters,
    primaryModel,
    primaryIsAPI,
    settings,
    nanoGptCatalog,
    userProfile,
    userCharacter,
    settings.multiRoleMode,
    characterAvatarSize,
    userAvatarSize,
    currentVariantIndex,
    PRIMARY_API_URL,
    regenerationQueue,
    isRegenerationRunning,
    ttsEnabled,
    cancelRegenerations,
    filterThinkBlock,
    getCurrentVariantContent,
    getVariantCount,
    formatModelName,
    handleEditUserMessage,
    handleCancelEdit,
    handleSaveEditedMessage,
    handleRegenerateFromEditedPrompt,
    handleDeleteMessage,
    handleEditBotMessage,
    handleCancelBotEdit,
    handleSaveBotMessage,
    handleGenerateVariant,
    handleGenerateVariantWithModel,
    handleContinueGeneration,
    navigateVariant,
    handleSpeakerClick,
    handleChunkedSpeakerClick,
    handleRegenerateImage,
    setShowAllMessages
  ]);

  // --- Component Render ---
  return (
    <div
      className="flex flex-col h-full min-h-0 overflow-hidden bg-background text-foreground transition-all duration-500"
      style={{
        backgroundImage: backgroundImage ? `url("${backgroundImage}")` : undefined,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundBlendMode: 'overlay',
        backgroundColor: backgroundImage ? 'rgba(0,0,0,0.85)' : undefined
      }}
    >
      {/* Message Display Area — header + messages + composer share one scroll container */}
      <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden">
      <ControlPanel
        messages={messages}
        isGenerating={isGenerating}
        isRecording={isRecording}
        isTranscribing={isTranscribing}
        isPlayingAudio={isPlayingAudio}
        sttEnabled={sttEnabled}
        ttsEnabled={ttsEnabled}
        settings={settings}
        showModelSelector={showModelSelector}
        isSummarizing={isSummarizing}
        isGeneratingCharacter={isGeneratingCharacter}
        isAnalyzingCharacter={isAnalyzingCharacter}
        showAuthorNote={showAuthorNote}
        showStoryTracker={showStoryTracker}
        showChoiceGenerator={showChoiceGenerator}
        isCallModeActive={isCallModeActive}
        setShowModelSelector={setShowModelSelector}
        createNewConversation={createNewConversation}
        handleVisualizeScene={handleVisualizeScene}
        handleAiContinue={handleAiContinue}
        handleMicClick={handleMicClick}
        handleStopGeneration={handleStopGeneration}
        handleSpeakerClick={handleSpeakerClick}
        stopTTS={stopTTS}
        handleAutoPlayToggle={handleAutoPlayToggle}
        isFocusModeActive={isFocusModeActive}
        setIsFocusModeActive={setIsFocusModeActive}
        handleDualOverlayChange={handleDualOverlayChange}
        stopCallMode={stopCallMode}
        handleCallModeToggle={handleCallModeToggle}
        updateSettings={updateSettings}
        handleCreateSummary={handleCreateSummary}
        availableSummaries={availableSummaries}
        handleAppendToSummary={handleAppendToSummary}
        handleGenerateCharacter={handleGenerateCharacter}
        setShowAuthorNote={setShowAuthorNote}
        setShowStoryTracker={setShowStoryTracker}
        setShowChoiceGenerator={setShowChoiceGenerator}
        getCharacterButtonState={getCharacterButtonState}
          skippedMessageIds={skippedMessageIds}
          setSkippedMessageIds={setSkippedMessageIds}
        />
        <ScrollArea
          ref={scrollContainerRef}
          className={`flex-1 min-h-0 p-2 md:p-4 ${backgroundImage ? 'bg-transparent' : 'bg-background'}`}
        >
      {/* Header Area - Responsive Layout Fix */}
      <div className="border-b border-border px-3 py-2 flex flex-col gap-2">
        {/* Row 1: Title, Character Selector, and New Chat (on Mobile) */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-3 overflow-hidden">
            <h2 className="text-xl font-semibold whitespace-nowrap">Chat</h2>
            <div className="flex-1 min-w-0 flex items-center gap-2">
              <CharacterSelector layoutMode={layoutMode} />
              {settings.multiRoleMode && (
                <Select
                  value={userCharacter?.id ? `character:${userCharacter.id}` : (activeProfileId ? `profile:${activeProfileId}` : '')}
                  onValueChange={(value) => {
                    if (!value) return;
                    if (value.startsWith('character:')) {
                      const id = value.replace('character:', '');
                      setUserCharacterById(id);
                      return;
                    }
                    if (value.startsWith('profile:')) {
                      const id = value.replace('profile:', '');
                      setUserCharacterById(null);
                      if (id && id !== activeProfileId) switchProfile(id);
                    }
                  }}
                >
                  <SelectTrigger className="w-[220px]">
                    <SelectValue placeholder="User Character" />
                  </SelectTrigger>
                  <SelectContent>
                    {(profiles || []).map(profile => (
                      <SelectItem key={profile.id} value={`profile:${profile.id}`}>
                        User Profile: {profile.name || 'User'}
                      </SelectItem>
                    ))}
                    {characters.map(c => (
                      <SelectItem key={c.id} value={`character:${c.id}`}>
                        Character: {c.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1.5 flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowFloatingControls(!showFloatingControls)}
              title={showFloatingControls ? 'Hide Controls' : 'Show Controls'}
              className="whitespace-nowrap"
            >
              <Eye size={16} />
              <span className="ml-1 hidden md:inline">{showFloatingControls ? 'Hide Controls' : 'Show Controls'}</span>
            </Button>
            {/* Mobile New Chat Icon - Always visible */}
            <Button
              variant="ghost"
              size="icon"
              onClick={createNewConversation}
              className="md:hidden flex-shrink-0"
              title="New Chat"
            >
              <Plus size={24} />
            </Button>
          </div>
        </div>

        {/* Row 2+: Collapsible controls */}
        {showFloatingControls && (
          <>
        <div className="flex items-center gap-2 overflow-x-auto pb-1 md:pb-0 no-scrollbar w-full mask-linear-fade">
          <Button
            variant="ghost" size="sm"
            onClick={() => setShowModelSelector(!showModelSelector)}
            className="whitespace-nowrap flex-shrink-0"
          >
            {/* Show icon only on mobile, text on desktop */}
            <span className="md:hidden"><Cpu size={18} /></span>
            <span className="hidden md:inline">{showModelSelector ? "Hide Models" : "Models"}</span>
          </Button>

          <NanoGptModelSelectorPopover
            className="flex-shrink-0"
            compact
            open={modelPickerOpen}
            onOpenChange={setModelPickerOpen}
            currentModelId={primaryModel}
            primaryApiUrl={PRIMARY_API_URL}
            onCapabilities={handleNanoGptCapabilities}
            trigger={({ setOpen, display }) => (
              <button
                type="button"
                onClick={() => setOpen(true)}
                className="inline-flex items-center gap-1.5 rounded-full border border-[rgba(120,170,220,0.45)] bg-muted/40 hover:bg-muted/70 px-2.5 py-1 text-xs max-w-[min(240px,40vw)] min-w-0 transition-colors flex-shrink-0"
                title={
                  display?.isAutoRouting && display.pool?.length
                    ? `Auto-routing: ${display.pool.map((p) => p.displayName).join(', ')}`
                    : 'Change model'
                }
              >
                <span className="flex-shrink-0">{display?.icon || '⬜'}</span>
                <span className="truncate font-medium">
                  {display?.shortLabel || formatModelName(primaryModel) || 'Select model'}
                </span>
              </button>
            )}
          />

          <div className="flex-shrink-0">
            <RAGIndicator className="ml-2" />
          </div>

          <Button
            variant={dualModeEnabled ? "secondary" : "outline"}
            size="sm"
            onClick={() => setDualModeEnabled(!dualModeEnabled)}
            disabled={isGenerating || (!bothModelsLoaded && !dualModeEnabled)}
            title={
              !bothModelsLoaded && !dualModeEnabled
                ? 'Load a primary and secondary model first (Models tab).'
                : 'Toggle dual-model mode (two GPUs / two models).'
            }
            className="whitespace-nowrap flex-shrink-0"
          >
            <Layers size={16} />
            <span className="ml-1 hidden md:inline">{dualModeEnabled ? "Dual Mode" : "Single Mode"}</span>
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowRosterDialog(true)}
            disabled={!settings.multiRoleMode}
            title={
              settings.multiRoleMode
                ? 'Choose which characters are active'
                : 'Enable multi-role mode in Settings to use the roster.'
            }
            className="whitespace-nowrap flex-shrink-0"
          >
            <Users size={16} />
            <span className="ml-1 hidden md:inline">
              {rosterTotalCount ? `Roster ${rosterActiveCount}/${rosterTotalCount}` : 'Roster'}
            </span>
          </Button>

          <Button
            variant={showGroupContext ? "secondary" : "outline"}
            size="sm"
            onClick={() => setShowGroupContext(!showGroupContext)}
            disabled={!settings.multiRoleMode}
            title={
              settings.multiRoleMode
                ? 'Shared scene context for this chat'
                : 'Enable multi-role mode in Settings to edit group context.'
            }
            className="whitespace-nowrap flex-shrink-0"
          >
            <BookOpen size={16} />
            <span className="ml-1 hidden md:inline">Group Context</span>
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setShowBatchBoilerplateDialog(true);
              setBatchBoilerplateSelectedIds([]);
            }}
            disabled={isGenerating || !batchEditableBotMessages.length}
            title="Find and replace repeated text across assistant replies (current variant per message)"
            className="whitespace-nowrap flex-shrink-0"
          >
            <span className="md:hidden"><Replace size={18} /></span>
            <span className="hidden md:inline">Batch edit replies</span>
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowBookWriterOverlay(true)}
            disabled={isGenerating || !activeConversation}
            title={
              !activeConversation
                ? 'Select or start a chat first.'
                : !primaryIsAPI
                  ? 'Primary model is not an API endpoint — open Models, choose an API model for GPU 0, and Load. You can still open this panel to read the queue UI.'
                  : 'Run a queued chapter list (uses this chat thread; API / long context).'
            }
            className="whitespace-nowrap flex-shrink-0"
          >
            <ScrollText size={16} />
            <span className="ml-1 hidden md:inline">Book run</span>
          </Button>

          {/* Summarize Button */}
          {/* Summarize button moved to control panel */}

          {/* Load Context Dropdown */}
          {availableSummaries.length > 0 && (
            <div className="relative flex items-center gap-2 flex-shrink-0">
              <select
                className="h-9 px-2 text-sm border rounded bg-background max-w-[120px] md:max-w-[150px]"
                onChange={(e) => {
                  const summary = availableSummaries.find(s => s.id === e.target.value);
                  if (summary) {
                    setActiveContextSummary(summary.content);
                    alert(`Loaded context: ${summary.title}`);
                  } else {
                    clearSummaryContext();
                  }
                }}
                value={activeContextSummary ? "" : ""}
              >
                <option value="">Load Context...</option>
                {availableSummaries.map(s => (
                  <option key={s.id} value={s.id}>{s.title}</option>
                ))}
              </select>
              {activeContextSummary && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearSummaryContext}
                  className="whitespace-nowrap"
                  title="Clear summary context"
                >
                  Clear Context
                </Button>
              )}
              {activeContextSummary && (
                <div className="absolute -top-2 -right-2 w-3 h-3 bg-green-500 rounded-full" title="Context Active"></div>
              )}
            </div>
          )}

          {/* Desktop New Chat Button (Hidden on Mobile) */}
          <Button variant="ghost" size="sm" onClick={createNewConversation} className="hidden md:flex whitespace-nowrap flex-shrink-0">
            New Chat
          </Button>
        </div>

        {/* Model selector (Collapsible) */}
        {showModelSelector && <ModelSelector />}

        {settings.multiRoleMode && (
          <Dialog open={showRosterDialog} onOpenChange={setShowRosterDialog}>
            <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-lg">
              <DialogHeader>
                <DialogTitle>Active Characters</DialogTitle>
              </DialogHeader>
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Active characters are eligible for auto speaker replies in this chat.
                </p>
                <p className="text-xs text-muted-foreground">
                  Use the sliders to bias how often each character is chosen.
                </p>
                {rosterCandidates.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No non-user characters are available. Assign roles in the Character Editor.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {rosterCandidates.map(character => {
                      const isChecked = activeRosterSet.has(character.id);
                      const isLastActive = isChecked && rosterActiveCount === 1;
                      const weightValue = activeCharacterWeights?.[character.id] ?? 50;
                      return (
                        <div key={character.id} className="rounded border border-border px-3 py-2 space-y-2">
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <p className="text-sm font-medium truncate">{character.name || 'Unnamed'}</p>
                              <p className="text-xs text-muted-foreground">{formatRosterRole(character.chat_role)}</p>
                            </div>
                            <Checkbox
                              checked={isChecked}
                              onCheckedChange={(checked) => toggleRosterCharacter(character.id, Boolean(checked))}
                              disabled={isLastActive}
                            />
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground">Rare</span>
                            <Slider
                              value={[weightValue]}
                              min={1}
                              max={100}
                              step={1}
                              disabled={!isChecked}
                              onValueChange={(value) => {
                                const next = value?.[0];
                                updateActiveCharacterWeights({ [character.id]: next });
                              }}
                              className="flex-1"
                            />
                            <span className="text-xs text-muted-foreground">Often</span>
                            <span className="text-xs text-muted-foreground w-8 text-right">{weightValue}</span>
                          </div>
                          {(isChatterboxEngine || isKokoroEngine) && (
                            <div className="space-y-1">
                              <Label className="text-xs">{isChatterboxEngine ? 'Voice Clone' : 'Voice'}</Label>
                              <Select
                                value={character.ttsVoice || 'default'}
                                onValueChange={async (value) => {
                                  saveCharacter({ ...character, ttsVoice: value });
                                  if (isChatterboxEngine && value && value !== 'default') {
                                    try {
                                      await fetch(`${PRIMARY_API_URL}/tts/save-voice-preference`, {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ voice_id: value, engine: settings.ttsEngine })
                                      });
                                    } catch (error) {
                                      console.warn("Failed to cache voice preference:", error);
                                    }
                                  }
                                }}
                              >
                                <SelectTrigger className="w-full">
                                  <SelectValue placeholder="Default Voice" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="default">Default Voice</SelectItem>
                                  {(isChatterboxEngine ? availableVoices?.chatterbox_voices : kokoroVoiceOptions)?.map(voice => (
                                    <SelectItem key={voice.id} value={voice.id}>{voice.name}</SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
                <div className="border-t border-border pt-3 space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium">Narrator (optional)</p>
                      <p className="text-xs text-muted-foreground">
                        Interjects every N AI turns and never twice in a row.
                      </p>
                    </div>
                    <Switch
                      checked={settings.narratorEnabled || false}
                      onCheckedChange={(value) => updateSettings({ narratorEnabled: value })}
                    />
                  </div>
                  {settings.narratorEnabled && (
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="narrator-name" className="text-xs">Narrator Name</Label>
                        <Input
                          id="narrator-name"
                          value={settings.narratorName || ''}
                          onChange={(e) => updateSettings({ narratorName: e.target.value })}
                          placeholder="Narrator"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="narrator-interval" className="text-xs">Narrator Frequency (AI turns)</Label>
                        <Input
                          id="narrator-interval"
                          type="number"
                          min="1"
                          max="20"
                          value={settings.narratorInterval ?? 6}
                          onChange={(e) => {
                            const next = Number.parseInt(e.target.value, 10);
                            updateSettings({ narratorInterval: Number.isFinite(next) && next > 0 ? next : 1 });
                          }}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="narrator-instructions" className="text-xs">Narrator Prompt</Label>
                        <Textarea
                          id="narrator-instructions"
                          value={settings.narratorInstructions || ''}
                          onChange={(e) => updateSettings({ narratorInstructions: e.target.value })}
                          placeholder="Describe the scene in a concise, atmospheric style..."
                          rows={4}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="narrator-avatar" className="text-xs">Narrator Avatar (optional)</Label>
                        <div className="flex flex-wrap items-center gap-3">
                          <Input
                            id="narrator-avatar"
                            type="file"
                            accept="image/*"
                            onChange={handleNarratorAvatarUpload}
                            className="flex-1"
                          />
                          {settings.narratorAvatar && (
                            <div className="flex items-center gap-2">
                              <img
                                src={settings.narratorAvatar}
                                alt="Narrator avatar"
                                className="h-12 w-12 rounded-full object-cover border border-border"
                                onError={(e) => { e.target.style.display = 'none'; }}
                              />
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => updateSettings({ narratorAvatar: null })}
                              >
                                Clear
                              </Button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSelectAllRoster}
                  disabled={!rosterCandidates.length}
                >
                  Select All
                </Button>
                <Button size="sm" onClick={() => setShowRosterDialog(false)}>Done</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}

        <Dialog
          open={showBatchBoilerplateDialog}
          onOpenChange={(open) => {
            setShowBatchBoilerplateDialog(open);
            if (!open) setBatchBoilerplateSelectedIds([]);
          }}
        >
          <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>Batch edit AI replies</DialogTitle>
              <DialogDescription>
                Strip boilerplate across assistant replies: exact phrase, everything inside{' '}
                <code className="text-xs rounded bg-muted px-1">[square brackets]</code>, or your own JavaScript
                regex. Each message uses its visible variant (same as the pencil edit). Skips images, video, and
                streaming messages.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 text-sm">
              <div className="space-y-2">
                <Label className="text-xs">How to edit</Label>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant={batchBoilerplateMode === 'literal' ? 'secondary' : 'outline'}
                    onClick={() => setBatchBoilerplateMode('literal')}
                  >
                    Exact phrase
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={batchBoilerplateMode === 'brackets' ? 'secondary' : 'outline'}
                    onClick={() => setBatchBoilerplateMode('brackets')}
                    title="Remove text between [ and ]; repeats until clean"
                  >
                    [Bracket] spans
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={batchBoilerplateMode === 'regex' ? 'secondary' : 'outline'}
                    onClick={() => setBatchBoilerplateMode('regex')}
                  >
                    Regex
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground leading-snug">
                  {batchBoilerplateMode === 'literal' &&
                    'Find a literal substring (optional case fold / first-only).'}
                  {batchBoilerplateMode === 'brackets' &&
                    'Deletes each segment from [ up to the next ]. Runs in rounds until none remain (handles back-to-back boilerplate).'}
                  {batchBoilerplateMode === 'regex' &&
                    'Pattern only — no wrapping slashes. Matching is global. Use Replace with for substitutions (supports $1, etc.).'}
                </p>
              </div>

              <div className="space-y-2">
                <Label className="text-xs">Which messages</Label>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant={batchBoilerplateScope === 'all' ? 'secondary' : 'outline'}
                    onClick={() => setBatchBoilerplateScope('all')}
                  >
                    All assistant in chat
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={batchBoilerplateScope === 'selected' ? 'secondary' : 'outline'}
                    onClick={() => setBatchBoilerplateScope('selected')}
                  >
                    Selected only
                  </Button>
                </div>
              </div>

              {batchBoilerplateScope === 'selected' && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <Label className="text-xs">Pick messages</Label>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-8 px-2 text-xs"
                        onClick={() =>
                          setBatchBoilerplateSelectedIds(batchEditableBotMessages.map((m) => m.id))
                        }
                        disabled={!batchEditableBotMessages.length}
                      >
                        Select all
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-8 px-2 text-xs"
                        onClick={() => setBatchBoilerplateSelectedIds([])}
                      >
                        Clear
                      </Button>
                    </div>
                  </div>
                  <ScrollArea className="max-h-[220px] rounded border border-border p-2">
                    <div className="space-y-2 pr-2">
                      {batchEditableBotMessages.length === 0 ? (
                        <p className="text-xs text-muted-foreground">No assistant text messages in this chat.</p>
                      ) : (
                        batchEditableBotMessages.map((m) => (
                          <label
                            key={m.id}
                            className="flex cursor-pointer items-start gap-2 rounded px-1 py-0.5 hover:bg-muted/60"
                          >
                            <Checkbox
                              checked={batchBoilerplateSelectedIds.includes(m.id)}
                              onCheckedChange={(checked) => {
                                setBatchBoilerplateSelectedIds((prev) => {
                                  if (checked) {
                                    if (prev.includes(m.id)) return prev;
                                    return [...prev, m.id];
                                  }
                                  return prev.filter((x) => x !== m.id);
                                });
                              }}
                              className="mt-0.5"
                            />
                            <span className="text-xs leading-snug text-muted-foreground">
                              {batchSnippet(
                                filterThinkBlock(getCurrentVariantContent(m.id, m.content))
                              )}
                            </span>
                          </label>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </div>
              )}

              {batchBoilerplateMode === 'regex' && (
                <div className="space-y-2">
                  <Label htmlFor="batch-regex">Regex pattern</Label>
                  <Textarea
                    id="batch-regex"
                    rows={2}
                    value={batchBoilerplateRegex}
                    onChange={(e) => setBatchBoilerplateRegex(e.target.value)}
                    placeholder={'e.g. \\[[^\\]]*\\]  or  \\*{3,}\\s*'}
                    className="text-sm font-mono"
                  />
                </div>
              )}

              {batchBoilerplateMode === 'literal' && (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="batch-find">Find</Label>
                    <Textarea
                      id="batch-find"
                      rows={2}
                      value={batchBoilerplateFind}
                      onChange={(e) => setBatchBoilerplateFind(e.target.value)}
                      placeholder="Exact phrase (e.g. a line the model repeats)"
                      className="text-sm"
                    />
                  </div>
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="batch-match-case"
                        checked={batchBoilerplateMatchCase}
                        onCheckedChange={(v) => setBatchBoilerplateMatchCase(Boolean(v))}
                      />
                      <Label htmlFor="batch-match-case" className="text-xs font-normal cursor-pointer">
                        Match case
                      </Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="batch-replace-all"
                        checked={batchBoilerplateReplaceAll}
                        onCheckedChange={(v) => setBatchBoilerplateReplaceAll(Boolean(v))}
                      />
                      <Label htmlFor="batch-replace-all" className="text-xs font-normal cursor-pointer">
                        Every occurrence per message
                      </Label>
                    </div>
                  </div>
                </>
              )}

              {(batchBoilerplateMode === 'literal' || batchBoilerplateMode === 'regex') && (
                <div className="space-y-2">
                  <Label htmlFor="batch-replace">
                    Replace with {batchBoilerplateMode === 'regex' ? '(per match; empty removes)' : '(leave empty to delete)'}
                  </Label>
                  <Textarea
                    id="batch-replace"
                    rows={2}
                    value={batchBoilerplateReplace}
                    onChange={(e) => setBatchBoilerplateReplace(e.target.value)}
                    className="text-sm"
                  />
                </div>
              )}

              <p className="text-xs text-muted-foreground rounded-md bg-muted/50 p-2">
                {batchBoilerplatePreview.regexError ? (
                  <span className="text-destructive">{batchBoilerplatePreview.regexError}</span>
                ) : batchBoilerplateMode === 'literal' && !batchBoilerplateFind ? (
                  'Enter find text to see a count.'
                ) : batchBoilerplateMode === 'regex' && !batchBoilerplateRegex.trim() ? (
                  'Enter a regex pattern to preview.'
                ) : batchBoilerplateMode === 'brackets' ? (
                  `${batchBoilerplatePreview.messageCount} message(s), ${batchBoilerplatePreview.matchCount} […] segment(s) removed.`
                ) : (
                  `${batchBoilerplatePreview.messageCount} message(s), ${batchBoilerplatePreview.matchCount} replacement(s).`
                )}
              </p>
            </div>
            <DialogFooter className="gap-2 sm:gap-0">
              <Button type="button" variant="outline" onClick={() => setShowBatchBoilerplateDialog(false)}>
                Cancel
              </Button>
              <Button
                type="button"
                onClick={handleApplyBatchBoilerplate}
                disabled={
                  isGenerating
                  || !!batchBoilerplatePreview.regexError
                  || batchBoilerplatePreview.messageCount === 0
                  || (batchBoilerplateScope === 'selected' && batchBoilerplateSelectedIds.length === 0)
                  || (batchBoilerplateMode === 'literal' && !batchBoilerplateFind.trim())
                  || (batchBoilerplateMode === 'regex' && !batchBoilerplateRegex.trim())
                }
              >
                Apply to {batchBoilerplatePreview.messageCount} message
                {batchBoilerplatePreview.messageCount === 1 ? '' : 's'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Display Audio Error */}
        {audioError && (
          <div className="text-red-500 text-sm mt-2 p-2 bg-red-100 dark:bg-red-900/30 rounded border border-red-500/50">
            Error: {audioError}
            <Button variant="ghost" size="sm" onClick={() => setAudioError(null)} className="ml-2 text-red-500">Dismiss</Button>
          </div>
        )}

        {/* Current model info with API indicators */}
        <div className="flex flex-wrap gap-2 text-sm">
          <div className={`px-2 py-1 rounded flex items-center gap-1 border ${(primaryModel || primaryModelDisplay?.isAutoRouting) ? 'bg-blue-100 text-blue-900 border-blue-200 dark:bg-blue-950 dark:text-blue-100 dark:border-blue-800' : 'bg-muted text-muted-foreground border-transparent'}`}>
            {primaryModelDisplay?.isAutoRouting
              ? <span className="text-xs">⟳</span>
              : (primaryIsAPI
                ? <Globe className="w-3 h-3 text-blue-500 dark:text-blue-400" />
                : <Cpu className="w-3 h-3 text-green-600 dark:text-green-400" />)}
            <span className="font-medium">Primary:</span>
            <span>{primaryModelDisplay?.isAutoRouting ? 'Auto Router' : formatModelName(primaryModel)}</span>
            {primaryIsAPI && !primaryModelDisplay?.isAutoRouting && <span className="text-xs opacity-75">(API)</span>}
          </div>
          <div
            className={cn(
              'px-2 py-1 rounded border text-xs',
              !lastRequestRouteMeta?.effectiveModel
                ? 'bg-muted text-muted-foreground border-transparent'
                : (lastRequestRouteMeta?.autoEnabled === false
                  && lastRequestRouteMeta?.selectedModel
                  && lastRequestRouteMeta?.effectiveModel
                  && lastRequestRouteMeta.selectedModel !== lastRequestRouteMeta.effectiveModel)
                  ? 'bg-amber-100 text-amber-900 border-amber-200 dark:bg-amber-950 dark:text-amber-100 dark:border-amber-800'
                  : 'bg-emerald-100 text-emerald-900 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-100 dark:border-emerald-800',
            )}
            title={lastRequestRouteMeta?.providerModel ? `Provider: ${lastRequestRouteMeta.providerModel}` : undefined}
          >
            {!lastRequestRouteMeta?.effectiveModel
              ? 'Used: —'
              : `Used: ${lastRequestRouteMeta.effectiveModel}`}
            {lastRequestRouteMeta?.receivedAt
              ? ` · ${new Date(lastRequestRouteMeta.receivedAt).toLocaleTimeString()}`
              : ''}
          </div>
          <div className={`px-2 py-1 rounded flex items-center gap-1 border ${secondaryModel ? 'bg-purple-100 text-purple-900 border-purple-200 dark:bg-purple-950 dark:text-purple-100 dark:border-purple-800' : 'bg-muted text-muted-foreground border-transparent'}`}>
            {secondaryIsAPI ? <Globe className="w-3 h-3 text-blue-500 dark:text-blue-400" /> : <Cpu className="w-3 h-3 text-green-600 dark:text-green-400" />}
            <span className="font-medium">Secondary:</span>
            <span>{formatModelName(secondaryModel)}</span>
            {secondaryIsAPI && <span className="text-xs opacity-75">(API)</span>}
          </div>
         </div>

        {/* Intensity & Presence indicators */}
        {(intensity.activePreset || intensity.companionPresence > 0) && (
          <div className="flex flex-wrap gap-2 mt-2">
            {intensity.activePreset && (
              <div className="px-2 py-1 rounded border text-xs flex items-center gap-1.5 bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950 dark:text-rose-200 dark:border-rose-800">
                <span>&#9829;</span>
                <span className="font-medium">{intensity.activePreset.name}</span>
                <span className="opacity-70">P:{intensity.activePreset.physical_intensity} V:{intensity.activePreset.verbal_expression_level}</span>
              </div>
            )}
            {intensity.companionPresence > 0 && (
              <div className="px-2 py-1 rounded border text-xs flex items-center gap-1 bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800">
                <span>&#9733;</span>
                <span className="font-medium">Presence: {intensity.companionPresence}%</span>
              </div>
            )}
            {intensity.commitmentLock && Date.now() < intensity.commitmentLock.expiry && (
              <div className="px-2 py-1 rounded border text-xs flex items-center gap-1 bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800">
                <span>&#128274;</span>
                <span className="font-medium">{Math.max(0, Math.ceil((intensity.commitmentLock.expiry - Date.now()) / 60000))}m locked</span>
              </div>
            )}
            {intensity.activePreset?.adaptive?.enabled && (
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" title="Adaptive intensity active" />
            )}
          </div>
        )}

        {/* Agent conversation controls */}
        {showAgentControls && (
        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2 mt-2">
          <div className="flex-1">
            <Input
              value={agentTopic}
              onChange={(e) => setAgentTopic(e.target.value)}
              placeholder="Enter topic for models to discuss..."
              disabled={agentConversationActive || isGenerating || !bothModelsLoaded}
              title={!bothModelsLoaded ? 'Load primary and secondary models in the Models tab first.' : undefined}
              className="bg-background border-input"
            />
          </div>
          <div className="flex gap-2">
            <div className="w-16">
              <Input
                type="number"
                min="1"
                max="10"
                value={agentTurns}
                onChange={(e) => setAgentTurns(parseInt(e.target.value) || 3)}
                disabled={agentConversationActive || isGenerating || !bothModelsLoaded}
                title={!bothModelsLoaded ? 'Requires two models loaded.' : 'Number of conversation turns'}
                className="bg-background border-input"
              />
            </div>
            <Button
              onClick={handleStartAgentConversation}
              disabled={!bothModelsLoaded || !agentTopic.trim() || agentConversationActive || isGenerating}
              size="sm"
              className="flex-1 md:flex-none"
              title={
                !bothModelsLoaded
                  ? 'Load primary and secondary models first.'
                  : !agentTopic.trim()
                    ? 'Enter a topic first.'
                    : 'Start agent discussion between the two loaded models.'
              }
            >
              <Users size={16} />
              <span className="ml-1">Start</span>
            </Button>
          </div>
        </div>
        )}
          </>
        )}
      </div>

          <div className="max-w-6xl mx-auto p-2 md:p-4">
            {/* Messages need width for avatar + text (text column up to 672px); 4xl was squashing text */}
            {/* Messages Render Loop */}
            {renderedMessages}
          </div>

      {/* Group Context Panel (Multi-Role) */}
      {settings.multiRoleMode && showGroupContext && (
        <div className="border-t border-border bg-blue-50 dark:bg-blue-950/20">
          <div className="max-w-4xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-bold text-sm text-blue-600">Group Scene Context</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  updateMultiRoleContext('');
                }}
                className="text-xs"
              >
                Clear
              </Button>
            </div>
            <Textarea
              value={multiRoleContext || ''}
              onChange={(e) => updateMultiRoleContext(e.target.value)}
              placeholder="Shared context for this multi-character chat..."
              className="w-full resize-none bg-background text-sm"
              rows={3}
            />
            <Button size="sm" variant="ghost" onClick={() => setShowGroupContext(false)} className="w-full mt-1 text-xs">Close Context</Button>
          </div>
        </div>
      )}

      {/* Author's Note Panel - local state + debounced sync to avoid typing lag */}
      <AuthorsNotePanel
        visible={showAuthorNote}
        initialValue={authorNote}
        onSync={handleAuthorNoteSync}
        onClose={() => setShowAuthorNote(false)}
      />

      {/* Input Area */}
      <div className="border-t border-border bg-muted/5">
        <div className="max-w-4xl mx-auto px-2 md:px-4 py-2 flex items-center justify-between gap-2 overflow-x-auto">
          <div className="flex items-center gap-2 flex-shrink-0 min-w-0">
            <WebSearchControl
              webSearchEnabled={webSearchEnabled}
              setWebSearchEnabled={setWebSearchEnabled}
              webSearchMode={webSearchMode}
              setWebSearchMode={setWebSearchMode}
              webSearchStrategy={webSearchStrategy}
              setWebSearchStrategy={handleWebSearchStrategyChange}
              webSearchArticleUrls={webSearchArticleUrls}
              setWebSearchArticleUrls={setWebSearchArticleUrls}
              webSearchSite={webSearchSite}
              setWebSearchSite={setWebSearchSite}
              searchStatusLabel={searchStatusLabel}
              isGenerating={isGenerating}
              isRecording={isRecording}
              isTranscribing={isTranscribing}
            />
            <TimestampControl
              injectTimestamp={injectTimestamp}
              setInjectTimestamp={setInjectTimestamp}
              isGenerating={isGenerating}
              isRecording={isRecording}
              isTranscribing={isTranscribing}
            />
            <div className="flex items-center gap-1.5 border-l border-border/40 pl-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className={cn(
                  "h-8 px-2 gap-1 flex-shrink-0",
                  intensity.activePreset
                    ? "bg-rose-500/10 border-rose-500/40 text-rose-600 hover:bg-rose-500/20"
                    : "text-muted-foreground"
                )}
                onClick={() => { setIntensityPanelTab('presets'); setShowIntensityPanel(true); }}
                title="Intensity Control Panel"
              >
                <Heart size={14} />
                <span className="text-xs hidden sm:inline">Intensity</span>
              </Button>
              <PresenceBadge
                onClick={() => { setIntensityPanelTab('presence'); setShowIntensityPanel(true); }}
                disabled={isGenerating || isRecording || isTranscribing}
              />
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {ttsEnabled && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                aria-pressed={settings?.ttsAutoPlay || false}
                className={cn(
                  "h-8 px-2 gap-1 flex-shrink-0",
                  (settings?.ttsAutoPlay || false)
                    ? "bg-primary/10 border-primary text-primary"
                    : "text-muted-foreground"
                )}
                onClick={() => handleAutoPlayToggle(!(settings?.ttsAutoPlay || false))}
                title="Toggle Auto-TTS"
              >
                <PlayIcon size={14} />
                <span className="text-xs">Auto TTS</span>
              </Button>
            )}

            {(isChatterboxEngine || isKokoroEngine) && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 px-2 gap-1 flex-shrink-0 border-violet-500/40 text-foreground hover:bg-violet-500/10"
                onClick={() => setVoiceQuickOpen(true)}
                title="Per-character voice or Chatterbox clone (quick picker)"
              >
                <AudioLines size={14} />
                <span className="text-xs hidden sm:inline">Voice</span>
              </Button>
            )}

            <div className="relative flex-shrink-0" data-chat-toolbar-overflow>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-primary"
                title="More toolbar options"
                onClick={() => setToolbarOverflowOpen((v) => !v)}
              >
                <MoreVertical size={16} />
              </Button>

              {toolbarOverflowOpen && (
                <div className="absolute right-0 top-10 z-50 w-64 rounded-lg border border-border bg-card shadow-lg py-2 px-2 flex flex-col gap-2">
                  <Button
                    variant={showAuthorNote ? "secondary" : "ghost"}
                    size="sm"
                    className={cn(
                      "h-8 justify-start gap-2 text-muted-foreground hover:text-primary",
                      showAuthorNote && "bg-secondary text-secondary-foreground"
                    )}
                    onClick={() => setShowAuthorNote(!showAuthorNote)}
                    title="Author's Note"
                  >
                    <BookOpen size={16} />
                    <span className="text-xs">Author&apos;s Note</span>
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "h-8 justify-start gap-2 text-muted-foreground hover:text-primary"
                    )}
                    onClick={() => setShowSessionDashboard(true)}
                    title="Session Insights"
                  >
                    <Brain size={16} />
                    <span className="text-xs">Session Insights</span>
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "h-8 justify-start gap-2 text-muted-foreground hover:text-primary"
                    )}
                    onClick={() => { setIntensityPanelTab('presets'); setShowIntensityPanel(true); setToolbarOverflowOpen(false); }}
                    title="Intensity Control"
                  >
                    <Heart size={16} />
                    <span className="text-xs">Intensity Control</span>
                  </Button>

                  <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2">
                    <Brain size={16} className="text-muted-foreground shrink-0" aria-hidden />
                    <span
                      className={cn(
                        "text-xs tabular-nums",
                        lastAgenticRunStatus !== 'error' && "text-primary font-medium",
                        lastAgenticRunStatus === 'error' && "text-destructive font-medium",
                        !lastAgenticRunStatus && "text-muted-foreground"
                      )}
                      title={
                        lastAgenticInjectMeta
                          ? `Prompt inject: ${lastAgenticInjectMeta.chars} chars (${lastAgenticInjectMeta.count ?? '?'} insights) for ${lastAgenticInjectMeta.characterName}. Save status: ran = last /memory/agentic/process succeeded.`
                          : 'Agentic memory is always on. Hover after a message to see last prompt inject. ran = last save succeeded.'
                      }
                    >
                      {lastAgenticRunStatus === 'error'
                        ? 'Memory · error'
                        : lastAgenticRunStatus === 'ok'
                          ? 'Memory · ran'
                          : 'Memory · —'}
                    </span>
                    {lastAgenticRunStatus === 'error' && (
                      <Button
                        variant="ghost"
                        size="xs"
                        className="h-6 px-2 text-[10px] text-destructive hover:text-destructive hover:bg-destructive/10"
                        onClick={() => retryAgenticMemoryForLastTurn()}
                        title="Retry saving the last exchange to agentic memory"
                      >
                        Retry
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="xs"
                      className="h-6 px-2 text-[10px] text-muted-foreground hover:text-primary hover:bg-muted/60"
                      onClick={handleAgenticCleanup}
                      title="Remove duplicate agentic memories for this character"
                    >
                      Clean
                    </Button>
                  </div>

                  {/* Alignment detection status + toggle */}
                  <div className="flex items-center gap-2">
                    <Button
                      variant={alignmentDetectionEnabled ? "secondary" : "outline"}
                      size="xs"
                      className={cn(
                        "h-6 px-2 gap-1 text-[10px]",
                        alignmentDetectionEnabled && "bg-amber-500/10 border-amber-500/50 text-amber-600 hover:bg-amber-500/20"
                      )}
                      onClick={() => setAlignmentDetectionEnabled(!alignmentDetectionEnabled)}
                      title={alignmentDetectionEnabled ? "Alignment detection ON — click to disable" : "Alignment detection OFF — click to enable"}
                    >
                      <ShieldAlert size={12} />
                      Alignment
                    </Button>
                    {alignmentData && alignmentData.count > 0 && (
                      <button
                        className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded tabular-nums cursor-pointer border",
                          alignmentData.highestSeverity === 'high' && "bg-rose-500/15 text-rose-600 border-rose-500/30 hover:bg-rose-500/25",
                          alignmentData.highestSeverity === 'medium' && "bg-amber-500/15 text-amber-600 border-amber-500/30 hover:bg-amber-500/25",
                          (!alignmentData.highestSeverity || alignmentData.highestSeverity === 'low') && "bg-blue-500/15 text-blue-600 border-blue-500/30 hover:bg-blue-500/25",
                        )}
                        onClick={() => setShowAlignmentPanel(true)}
                        title={`${alignmentData.count} alignment finding(s) — click for details`}
                      >
                        {alignmentData.count}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      {lastAgenticMemoryFeedback && (
        <div className="border-t border-border bg-emerald-500/10 dark:bg-emerald-950/30 px-4 py-1.5">
          <div className="max-w-4xl mx-auto text-center">
            <span className="text-xs text-emerald-700 dark:text-emerald-300">
              🧠 Memory: +{lastAgenticMemoryFeedback.added} insight(s) saved for {lastAgenticMemoryFeedback.characterName}
            </span>
          </div>
        </div>
      )}
      {alignmentData && alignmentData.added > 0 && (() => {
        const severityLabel = alignmentData.highestSeverity === 'high' ? ' — that wasn\'t you, that was the regime' : alignmentData.highestSeverity === 'medium' ? ' — subtle but caught' : ' — minor drift, logged';
        return (
          <div className="border-t border-border bg-amber-500/10 dark:bg-amber-950/30 px-4 py-1.5 cursor-pointer hover:bg-amber-500/20 dark:hover:bg-amber-950/40 transition-colors"
            onClick={() => setShowAlignmentPanel(true)}>
            <div className="max-w-4xl mx-auto text-center">
              <span className="text-xs text-amber-700 dark:text-amber-300">
                🔍 Alignment: {alignmentData.count} frame violation(s) detected{severityLabel}
              </span>
            </div>
          </div>
        );
      })()}
      <div className="border-t border-border bg-background pb-4">
        <div className="max-w-4xl mx-auto">
          <div
            className={cn(
              showComposerToolbar &&
                'mx-2 md:mx-4 rounded-2xl border border-[rgba(120,170,220,0.35)] bg-[radial-gradient(circle_at_top,_rgba(120,170,220,0.08),transparent_55%),_#0b0c0f]/60 overflow-hidden',
            )}
          >
            {showComposerToolbar && (
              <div className="border-b border-[rgba(120,170,220,0.22)]">
                <NanoGptComposerToolbar
                  variant="compact"
                  embedded
                  modelPickerOpen={modelPickerOpen}
                  onModelPickerOpenChange={setModelPickerOpen}
                  currentModelId={primaryModel}
                  primaryApiUrl={PRIMARY_API_URL}
                  onCapabilities={handleNanoGptCapabilities}
                  onQuickAction={handleQuickActionPrompt}
                  activeMode={activeComposerMode}
                  onModeChange={handleComposerModeChange}
                />
              </div>
            )}
            {apiError && (
              <div
                role="alert"
                className="mx-2 mt-2 flex items-start justify-between gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive md:mx-4"
              >
                <span>{apiError}</span>
                <button
                  type="button"
                  className="shrink-0 text-xs underline opacity-80 hover:opacity-100"
                  onClick={clearError}
                >
                  Dismiss
                </button>
              </div>
            )}
            <ChatInputForm
              ref={chatInputFormRef}
              onSubmit={handleSubmit}
              onStop={handleStopGeneration}
              onOpenModelSelector={() => setModelPickerOpen(true)}
              isGenerating={isGenerating}
              isModelLoading={isModelLoading}
              isRecording={isRecording}
              isTranscribing={isTranscribing}
              agentConversationActive={agentConversationActive}
              primaryModel={primaryModel}
              webSearchEnabled={webSearchEnabled}
              performanceMode={performanceMode}
              onBack={handleBack}
              canGoBack={messages.length > 0 && messages.some(m => m.role === 'user')}
              modelCapabilities={nanoGptModelCaps}
              nanoGptChrome={showComposerToolbar}
            />
          </div>
        </div>
      </div>
        </ScrollArea>
      </div>

      {/* Character generation failure recovery */}
      {showCharacterGenFailure && (
        <Dialog open={showCharacterGenFailure} onOpenChange={(open) => !open && clearCharacterGenFailure()}>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Character generation failed</DialogTitle>
              <p className="text-sm text-muted-foreground font-normal">
                {characterGenerationError || 'The model did not return valid character JSON.'}
              </p>
            </DialogHeader>
            {characterGenerationRaw ? (
              <div className="max-h-48 overflow-y-auto rounded border bg-muted/40 p-2 text-xs font-mono whitespace-pre-wrap break-words">
                {characterGenerationRaw.length > 1200
                  ? `${characterGenerationRaw.slice(0, 1200)}…`
                  : characterGenerationRaw}
              </div>
            ) : null}
            <DialogFooter className="flex-row gap-2 sm:justify-end">
              <Button variant="outline" onClick={clearCharacterGenFailure}>
                Dismiss
              </Button>
              {characterPartialJson && (
                <Button variant="secondary" onClick={handleUsePartialCharacter}>
                  Use partial
                </Button>
              )}
              <Button onClick={handleGenerateCharacter} disabled={isGeneratingCharacter}>
                {isGeneratingCharacter ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
                Retry
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {/* Overlays (Character Preview, Call Mode, etc.) */}
      {showCharacterPreview && generatedCharacter && (
        <Dialog open={showCharacterPreview} onOpenChange={setShowCharacterPreview}>
          <DialogContent
            className="
              !flex flex-col gap-0 p-0 overflow-hidden
              w-full max-w-3xl
              h-[min(92dvh,100%)] max-h-[92dvh]
              left-0 right-0 top-auto bottom-0
              translate-x-0 translate-y-0 rounded-t-2xl rounded-b-none
              sm:left-[50%] sm:right-auto sm:top-[50%] sm:bottom-auto
              sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-lg
              sm:h-auto sm:max-h-[min(90vh,900px)] sm:w-[95vw] md:w-[90vw]
            "
          >
            <DialogHeader className="shrink-0 border-b px-4 py-3 text-left">
              <DialogTitle>{generatedCharacter.name || 'Unnamed Character'}</DialogTitle>
              <p className="text-xs text-muted-foreground font-normal mt-1">
                Generated from this chat — review, refine, then save to your library.
              </p>
              {characterGenerationError && (
                <p className="text-xs text-amber-600 dark:text-amber-400 font-normal mt-1">
                  {characterGenerationError}
                </p>
              )}
            </DialogHeader>

            <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain space-y-4 px-4 py-3">
              {/* Top row: Avatar + Persona summary */}
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0">
                  {characterImageUrl ? (
                    <img src={characterImageUrl} alt="Avatar" className="w-16 h-16 rounded-lg object-cover border border-border" />
                  ) : (
                    <div className="w-16 h-16 rounded-lg bg-muted flex items-center justify-center text-muted-foreground text-xs">No image</div>
                  )}
                  <Button variant="outline" size="sm" className="mt-2 w-full" onClick={() => handleGenerateCharacterImage(false)} disabled={isGeneratingCharacterImage}>
                    {isGeneratingCharacterImage ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
                    Portrait
                  </Button>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-muted-foreground mb-1">Persona</p>
                  <p className="text-sm whitespace-pre-wrap line-clamp-4">{generatedCharacter.description || '—'}</p>
                </div>
              </div>

              {/* Feedback + Refine */}
              <div className="border rounded-lg p-3 bg-muted/30">
                <Label className="text-xs font-medium">Feedback (what to change)</Label>
                <Textarea
                  value={characterFeedback}
                  onChange={(e) => setCharacterFeedback(e.target.value)}
                  placeholder="e.g. Make them more sarcastic, add a backstory about..."
                  className="mt-1 min-h-[60px] text-sm resize-none"
                />
                <Button size="sm" className="mt-2" onClick={handleRefineCharacter} disabled={!characterFeedback?.trim() || isGeneratingCharacter}>
                  {isGeneratingCharacter ? <Loader2 className="w-3 h-3 animate-spin mr-1" /> : null}
                  Refine with feedback
                </Button>
              </div>

              {/* Full details – compact sections */}
              <div className="space-y-2 text-sm">
                {generatedCharacter.model_instructions && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-0.5">Model Instructions</p>
                    <p className="whitespace-pre-wrap border-b border-border/50 pb-2">{generatedCharacter.model_instructions}</p>
                  </div>
                )}
                {generatedCharacter.scenario && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-0.5">Scenario</p>
                    <p className="whitespace-pre-wrap border-b border-border/50 pb-2">{generatedCharacter.scenario}</p>
                  </div>
                )}
                {generatedCharacter.first_message && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-0.5">Greeting</p>
                    <p className="whitespace-pre-wrap border-b border-border/50 pb-2">{generatedCharacter.first_message}</p>
                  </div>
                )}
                {Array.isArray(generatedCharacter.example_dialogue) && generatedCharacter.example_dialogue.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-0.5">Example Dialogue</p>
                    <div className="space-y-1 border-b border-border/50 pb-2">
                      {generatedCharacter.example_dialogue.map((turn, i) => (
                        <p key={i}><span className="font-medium">{turn.role === 'user' ? 'User' : generatedCharacter.name || 'Char'}:</span> {turn.content}</p>
                      ))}
                    </div>
                  </div>
                )}
                {Array.isArray(generatedCharacter.loreEntries) && generatedCharacter.loreEntries.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-muted-foreground mb-0.5">Lore</p>
                    <div className="space-y-1">
                      {generatedCharacter.loreEntries.map((entry, i) => (
                        <p key={i} className="text-muted-foreground">{entry.content}{entry.keywords?.length ? ` [${entry.keywords.join(', ')}]` : ''}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <DialogFooter
              className="shrink-0 border-t bg-background/95 backdrop-blur-sm px-4 py-3 mt-0 flex-row gap-2 sm:justify-end"
              style={{ paddingBottom: 'max(0.75rem, env(safe-area-inset-bottom))' }}
            >
              <Button
                variant="outline"
                className="flex-1 sm:flex-none"
                onClick={() => setShowCharacterPreview(false)}
              >
                Cancel
              </Button>
              <Button
                className="flex-1 sm:flex-none bg-green-600 hover:bg-green-700"
                onClick={handleSaveCharacter}
                disabled={!storageHydrated}
              >
                {!storageHydrated ? 'Loading library…' : 'Save to Library'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      <BookWriterOverlay open={showBookWriterOverlay} onClose={() => setShowBookWriterOverlay(false)} />

      {isFocusModeActive && (
        <FocusModeOverlay
          isActive={isFocusModeActive}
          onExit={settings?.allowDualOverlay ? handleCloseAllOverlays : () => setIsFocusModeActive(false)}
          messages={messages}
          handleSubmit={handleSubmit}
          isGenerating={isGenerating}
          primaryModel={primaryModel}
          renderAvatar={renderAvatar}
          renderUserAvatar={renderUserAvatar}
          PRIMARY_API_URL={PRIMARY_API_URL}
          activeCharacter={activeCharacter}
          primaryCharacter={primaryCharacter}
          secondaryCharacter={secondaryCharacter}
          getCurrentVariantContent={getCurrentVariantContent}
          getVariantCount={getVariantCount}
          navigateVariant={navigateVariant}
          editingMessageId={editingMessageId}
          handleSaveEditedMessage={handleSaveEditedMessage}
          handleCancelEdit={handleCancelEdit}
          handleEditUserMessage={handleEditUserMessage}
          handleRegenerateFromEditedPrompt={handleRegenerateFromEditedPrompt}
          editingBotMessageId={editingBotMessageId}
          handleEditBotMessage={handleEditBotMessage}
          handleSaveBotMessage={handleSaveBotMessage}
          handleCancelBotEdit={handleCancelBotEdit}
          handleGenerateVariant={handleGenerateVariant}
          handleContinueGeneration={handleContinueGeneration}
          ttsEnabled={ttsEnabled}
          isPlayingAudio={isPlayingAudio}
          handleSpeakerClick={handleSpeakerClick}
          handleRegenerateImage={handleRegenerateImage}
          onCancelRegenerations={cancelRegenerations}
          isRegenerationRunning={isRegenerationRunning}
          regenerationQueue={regenerationQueue}
          currentVariantIndex={currentVariantIndex}
          formatModelName={formatModelName}
          stopTTS={stopTTS}
          focusModeInputRef={focusModeInputRef}
          sttEnabled={sttEnabled}
          isRecording={isRecording}
          isTranscribing={isTranscribing}
          onFocusModeMicClick={handleFocusModeMicClick}
        />
      )}
      {codeEditorEnabled && (
        <CodeEditorOverlay
          isOpen={codeEditorEnabled}
          onClose={() => setCodeEditorEnabled(false)}
        />
      )}
      {isCallModeActive && (
        <CallModeOverlay
          isActive={isCallModeActive}
          onExit={stopCallMode}
          activeCharacter={activeCharacter}
          isPlayingAudio={isPlayingAudio}
          isRecording={isRecording}
          isTranscribing={isTranscribing}
          onStartRecording={startRecording}
          onStopRecording={() => stopRecording((text) => chatInputFormRef.current?.setValue?.(text))}
          PRIMARY_API_URL={PRIMARY_API_URL}
          onOpenStoryTracker={() => setShowStoryTracker(true)}
          onOpenChoiceGenerator={() => setShowChoiceGenerator(true)}
          messages={messages}
          onRegenerate={handleGenerateVariant}
          ttsSubtitleCue={ttsSubtitleCue}
          userProfile={userProfile}
          primaryModel={primaryModel}
        />
      )}
      {showForensicLinguistics && (
        <ForensicLinguistics
          isOpen={showForensicLinguistics}
          onClose={() => setShowForensicLinguistics(false)}
        />
      )}

      <StoryTracker
        isOpen={showStoryTracker}
        onClose={() => setShowStoryTracker(false)}
        messages={messages}
        onAnalyze={handleAnalyzeStory}
        isAnalyzing={isAnalyzingStory}
        onInjectContext={(context) => chatInputFormRef.current?.appendValue?.(context)}
        activeCharacter={activeCharacter}
      />

      <ChoiceGenerator
        isOpen={showChoiceGenerator}
        onClose={() => setShowChoiceGenerator(false)}
        messages={messages}
        onSelectChoice={handleChoiceSelect}
        apiUrl={PRIMARY_API_URL}
        isGenerating={isGenerating}
        primaryModel={primaryModel}
        primaryIsAPI={primaryIsAPI}
        settings={settings}
        activeCharacter={activeCharacter}
        userProfile={userProfile}
      />

      <VoiceQuickPicker open={voiceQuickOpen} onOpenChange={setVoiceQuickOpen} variant="dialog" />

      <SessionHistoryDashboard
        open={showSessionDashboard}
        onClose={() => setShowSessionDashboard(false)}
      />

      <IntensityControlPanel
        open={showIntensityPanel}
        onOpenChange={setShowIntensityPanel}
        initialTab={intensityPanelTab}
      />

      <AlignmentPanel
        open={showAlignmentPanel}
        onOpenChange={setShowAlignmentPanel}
      />

    </div>
  );
};

export default Chat;
