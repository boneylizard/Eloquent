// Settings.jsx
// Full Settings UI: General, Generation, SD, RAG, Characters, Audio, Memory Intent, Persona realignment, Memory Browser, Lore, About

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getBackendUrl, fetchWithTimeout, formatFetchError, memoryApiUnreachableHint } from '../config/api';
import {
  API_CONTEXT_WINDOW_MIN,
  API_CONTEXT_WINDOW_MAX,
  API_CONTEXT_WINDOW_TOKENS_DEFAULT,
  API_CONTEXT_WINDOW_SLIDER_STEP,
  formatApiContextWindowShort,
  clampApiContextWindowTokens,
} from '../config/apiContextLimits';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Textarea } from './ui/textarea';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Slider } from './ui/slider';
import { Save, Sun, Moon, DownloadCloud, Trash2, ExternalLink, Loader2, RefreshCw, X, Power, RotateCw, FolderOpen, Pencil, Monitor, Sparkles, ChevronRight, Link2 } from 'lucide-react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import CharacterEditor from './CharacterEditor';
import LoreDebugger from '../components/LoreDebugger';
import MemoryIntentDetector from './MemoryIntentDetector';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import RAGSettings from './RAGSettings';
import ProfileSelector from './ProfileSelector';
import SimpleUserProfileEditor from './SimpleUserProfileEditor';
import LocalStorageManager from './LocalStorageManager';
import PersonaRealignmentPanel from './PersonaRealignmentPanel';
import ChatlogCondenserPanel from './ChatlogCondenserPanel';
import FlowApiOverrideFields from './FlowApiOverrideFields';
import VoiceSculptPanel from './VoiceSculptPanel';
import MemoryCuratorPanel from './MemoryCuratorPanel';
import NanoGptMemorySettings from './NanoGptMemorySettings';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';
import { resolveEndpointDisplay, getRotationPool } from '../utils/resolveEndpointDisplay';
import { readNanoGptModelsCache } from '../utils/nanoGptModelsCache';
import MobileRemoteSettings from './MobileRemoteSettings';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import { fetchDemoShowcaseStatus, installDemoShowcase } from '../utils/demoShowcase';
import {
  TV_PERF_STORAGE_KEY,
  readTvPerformanceFromUrl,
  readTvPerformanceFromStorage,
  applyTvPerformanceClass,
} from '../utils/tvPerformanceMode';
import { useAppBoot } from '../hooks/useAppBoot';
import InfrastructureBanner from './InfrastructureBanner';
import {
  SPLASH_DURATION_OPTIONS,
  SPLASH_SCREEN_DURATION_DEFAULT,
} from '../utils/eloquentSplash';

const DIRECTORY_SETTING_KEYS = [
  'modelDirectory',
  'sdModelDirectory',
  'adetailerModelDirectory',
  'upscalerModelDirectory'
];

const SettingsSection = ({ title, description, children, actions }) => (
  <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2 border-b border-border/60 px-5 py-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">{title}</p>
        {description ? (
          <p className="text-sm text-foreground/80 mt-1">{description}</p>
        ) : null}
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
    <div className="p-5 space-y-3">{children}</div>
  </div>
);

const SettingRow = ({ label, description, htmlFor, layout = 'row', children }) => {
  const isStack = layout === 'stack';
  const labelNode = typeof label === 'string'
    ? <Label htmlFor={htmlFor} className="text-sm font-semibold text-foreground">{label}</Label>
    : label;

  return (
    <div
      className={[
        'rounded-lg border border-border/60 bg-background/40 px-4 py-3',
        isStack ? 'space-y-2' : 'grid md:grid-cols-[minmax(200px,1fr),minmax(220px,360px)] items-center gap-4'
      ].join(' ')}
    >
      <div className="space-y-1">
        {labelNode}
        {description ? <p className="text-xs text-muted-foreground">{description}</p> : null}
      </div>
      <div className={isStack ? '' : 'w-full md:max-w-sm'}>
        {children}
      </div>
    </div>
  );
};

const SettingsAccordion = ({ title, summary, defaultOpen = false, children }) => (
  <details open={defaultOpen} className="group rounded-xl border border-border/60 bg-background/20">
    <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3">
      <div className="min-w-0">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">{title}</p>
        {summary ? <p className="truncate text-xs text-muted-foreground mt-0.5">{summary}</p> : null}
      </div>
      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-90" />
    </summary>
    <div className="space-y-6 border-t border-border/60 px-3 py-4 md:px-4">{children}</div>
  </details>
);


const Settings = ({ darkMode, toggleDarkMode, initialTab = 'general', isStandaloneWindow = false }) => {
  const [settingsMainTab, setSettingsMainTab] = useState(initialTab);
  useEffect(() => {
    setSettingsMainTab(initialTab);
  }, [initialTab]);
  const {
    settings: contextSettings,
    updateSettings,
    userAvatarSize,
    setUserAvatarSize,
    characterAvatarSize,
    setCharacterAvatarSize,
    sttEnabled,
    ttsEnabled,
    checkSdStatus,
    sdStatus,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    TTS_API_URL,
    apiError,
    clearError,
    fetchAvailableSTTEngines,
    sttEnginesAvailable,
    playTestStreamingTTS,
    stopTTS,
    pauseStreamingTTS,
    resumeStreamingTTS,
    isStreamingTtsPaused,
    isPlayingAudio,
    ttsFullResponseSaveStatus,
    audioError,
    characters,
    primaryModel,
    availableModels,
    upsertOutreachRule,
    deleteOutreachRule,
    runOutreachRuleNow,
    uploadOutreachRuleImages,
    clearOutreachRuleImages,
    outreachNotifications,
    clearOutreachNotifications,
    openOutreachNotification,
    discardOutreachNotification,
    requestOutreachNotificationPermission,
    conversations,
    openSettingsWindow,
  } = useApp();

  const headingFontOptions = [
    { label: 'Default (Theme)', value: 'default' },
    { label: 'Poppins', value: "'Poppins', sans-serif" },
    { label: 'Inter', value: "'Inter', sans-serif" },
    { label: 'JetBrains Mono', value: "'JetBrains Mono', monospace" }
  ];

  const resolveHeadingFontValue = (value) => (value ? value : 'default');
  const normalizeHeadingFontValue = (value) => (value === 'default' ? '' : value);

  // Local editable copy of context settings
  const [localSettings, setLocalSettings] = useState({
    ...contextSettings,
    directProfileInjection: contextSettings.directProfileInjection ?? false,
    temperature: contextSettings.temperature ?? 0.7,
    max_tokens: contextSettings.max_tokens ?? -1,
    top_p: contextSettings.top_p ?? 0.9,
    top_k: contextSettings.top_k ?? 40,
    repetition_penalty: contextSettings.repetition_penalty ?? 1.1,
    frequencyPenalty: contextSettings.frequencyPenalty ?? 0.0,
    presencePenalty: contextSettings.presencePenalty ?? 0.0,
    antiRepetitionMode: contextSettings.antiRepetitionMode ?? false,
    detectRepeatedPhrases: contextSettings.detectRepeatedPhrases ?? false,
    streamResponses: contextSettings.streamResponses ?? true,
    mdBodyColor: contextSettings.mdBodyColor ?? '',
    mdBoldColor: contextSettings.mdBoldColor ?? '',
    mdItalicColor: contextSettings.mdItalicColor ?? '',
    mdQuoteColor: contextSettings.mdQuoteColor ?? '',
    mdQuoteBorder: contextSettings.mdQuoteBorder ?? '',
    mdH1Color: contextSettings.mdH1Color ?? '',
    mdH2Color: contextSettings.mdH2Color ?? '',
    mdH3Color: contextSettings.mdH3Color ?? '',
    mdH1Font: contextSettings.mdH1Font ?? '',
    mdH2Font: contextSettings.mdH2Font ?? '',
    mdH3Font: contextSettings.mdH3Font ?? '',
    ttsSpeed: contextSettings.ttsSpeed ?? 1.0,
    ttsPitch: contextSettings.ttsPitch ?? 0,
    ttsAutoPlay: contextSettings.ttsAutoPlay ?? false,
    ttsSaveFullResponseAudio: contextSettings.ttsSaveFullResponseAudio ?? false,
    ttsSaveFullResponseChunkSeconds: contextSettings.ttsSaveFullResponseChunkSeconds ?? 0,
    ttsEngine: contextSettings.ttsEngine ?? 'kokoro',
    ttsVoice: contextSettings.ttsVoice ?? 'af_heart',
    ttsStreamChunkSentences: contextSettings.ttsStreamChunkSentences ?? 2,
    ttsExaggeration: contextSettings.ttsExaggeration ?? 0.5,
    ttsCfg: contextSettings.ttsCfg ?? 0.5,
    ttsSpeedMode: contextSettings.ttsSpeedMode ?? 'standard',
    sdModelDirectory: contextSettings.sdModelDirectory ?? '',
    upscalerModelDirectory: contextSettings.upscalerModelDirectory ?? '',
    sdSteps: contextSettings.sdSteps ?? 20,
    sdSampler: contextSettings.sdSampler ?? 'Euler a',
    sdCfgScale: contextSettings.sdCfgScale ?? 7.0,
    imageEngine: contextSettings.imageEngine ?? 'EloDiffusion',
    adetailerModelDirectory: contextSettings.adetailerModelDirectory ?? '',
    useOpenAIAPI: contextSettings.useOpenAIAPI ?? false,
    apiEndpointRoundRobinEnabled: contextSettings.apiEndpointRoundRobinEnabled ?? false,
    customApiEndpoints: contextSettings.customApiEndpoints ?? [],
    webSearchStrategy: contextSettings.webSearchStrategy ?? 'auto',
    admin_password: contextSettings.admin_password ?? "",
    apiRollingMemoryEnabled: contextSettings.apiRollingMemoryEnabled ?? true,
    apiContextWindowTokens: contextSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT,
    apiRecentVerbatimTokenBudget: contextSettings.apiRecentVerbatimTokenBudget ?? 32000,
    bookWritingApiContextTokens: contextSettings.bookWritingApiContextTokens ?? 262144,
    bookWritingVerbatimTokenBudget: contextSettings.bookWritingVerbatimTokenBudget ?? 98304,
    bookRefusalMaxChars: contextSettings.bookRefusalMaxChars ?? 2200,
    bookQuickPromptButtons: Array.isArray(contextSettings.bookQuickPromptButtons)
      ? contextSettings.bookQuickPromptButtons
      : [],
    didQuickPromptButtons: Array.isArray(contextSettings.didQuickPromptButtons)
      ? contextSettings.didQuickPromptButtons
      : [],
    bookWordFloorPreamble: contextSettings.bookWordFloorPreamble ?? '',
    callModeAboutCharacterEnabled: contextSettings.callModeAboutCharacterEnabled ?? true,
    callModeAboutCharacterPrompt: contextSettings.callModeAboutCharacterPrompt ?? '',
    callModeAboutCharacterMaxTokens: contextSettings.callModeAboutCharacterMaxTokens ?? 1200,
    callModeAboutCharacterTemperature: contextSettings.callModeAboutCharacterTemperature ?? 0.6,
    callModeAboutCharacterHistoryLimit: contextSettings.callModeAboutCharacterHistoryLimit ?? 40,
    callModeAboutCharacterRequestPurpose: contextSettings.callModeAboutCharacterRequestPurpose ?? 'call_mode_character_about',
    callModeAboutCharacterEndpoint: contextSettings.callModeAboutCharacterEndpoint ?? '',
    callModeAboutCharacterApiOverrideEnabled: contextSettings.callModeAboutCharacterApiOverrideEnabled ?? false,
    callModeAboutCharacterApiEndpointId: contextSettings.callModeAboutCharacterApiEndpointId ?? '',
    callModeAboutCharacterApiModel: contextSettings.callModeAboutCharacterApiModel ?? '',
    callModeAboutCharacterApiKey: contextSettings.callModeAboutCharacterApiKey ?? '',
    callModeAboutCharacterSystemPromptMode:
      contextSettings.callModeAboutCharacterSystemPromptMode ?? 'flat',
    callModeFullscreenAvatar: contextSettings.callModeFullscreenAvatar ?? false,
    callModeFullscreenZoom: contextSettings.callModeFullscreenZoom ?? 1,
    callModeFullscreenPanX: contextSettings.callModeFullscreenPanX ?? 0,
    callModeFullscreenPanY: contextSettings.callModeFullscreenPanY ?? 0,
    characterIntroEnabled: contextSettings.characterIntroEnabled ?? false,
    characterIntroPrompt: contextSettings.characterIntroPrompt ?? '',
    characterIntroMaxTokens: contextSettings.characterIntroMaxTokens ?? 900,
    characterIntroTemperature: contextSettings.characterIntroTemperature ?? 0.55,
    characterIntroHistoryLimit: contextSettings.characterIntroHistoryLimit ?? 8,
    characterIntroRequestPurpose: contextSettings.characterIntroRequestPurpose ?? 'character_intro',
    characterIntroEndpoint: contextSettings.characterIntroEndpoint ?? '',
    characterIntroApiOverrideEnabled: contextSettings.characterIntroApiOverrideEnabled ?? false,
    characterIntroApiEndpointId: contextSettings.characterIntroApiEndpointId ?? '',
    characterIntroApiModel: contextSettings.characterIntroApiModel ?? '',
    characterIntroApiKey: contextSettings.characterIntroApiKey ?? '',
    characterIntroSystemPromptMode:
      contextSettings.characterIntroSystemPromptMode ?? 'full_generation',
    useCharacterAsSystemPrompt: contextSettings.useCharacterAsSystemPrompt ?? false,
    systemPersonaCharacterId: contextSettings.systemPersonaCharacterId ?? null,
    systemIntroRequestPurpose: contextSettings.systemIntroRequestPurpose ?? 'system_intro',
    systemIntroPrompt: contextSettings.systemIntroPrompt ?? '',
    systemIntroSystemPromptMode:
      contextSettings.systemIntroSystemPromptMode ?? 'full_generation',
    openSettingsInSecondWindow: contextSettings.openSettingsInSecondWindow ?? false,
    splashScreenDuration:
      contextSettings.splashScreenDuration ?? SPLASH_SCREEN_DURATION_DEFAULT,
    main_gpu_id: contextSettings.main_gpu_id ?? 0,
    auto_launch_browser: contextSettings.auto_launch_browser ?? true,
    showReasoningDiagnostics: contextSettings.showReasoningDiagnostics ?? false,
  });
  const [tvPerfStored, setTvPerfStored] = useState(() => readTvPerformanceFromStorage());


  useEffect(() => {
    applyTvPerformanceClass(readTvPerformanceFromUrl() || tvPerfStored);
  }, [tvPerfStored]);
  const [sdModels, setSdModels] = useState([]);
  const [isInstallingEngine, setIsInstallingEngine] = useState(false);
  const [isUploadingVoice, setIsUploadingVoice] = useState(false);
  const [availableVoices, setAvailableVoices] = useState(null);
  const [isUnloadingChatterbox, setIsUnloadingChatterbox] = useState(false);
  const [isReloadingChatterbox, setIsReloadingChatterbox] = useState(false);
  const [currentTensorSplit, setCurrentTensorSplit] = useState([0.5, 0.5]);
  const [gpuCount, setGpuCount] = useState(2);
  const [isUnloadingForensicModels, setIsUnloadingForensicModels] = useState(false);
  const [markdownDefaults, setMarkdownDefaults] = useState({
    body: '#ffffff',
    bold: '#ffffff',
    italic: '#9ca3af',
    quote: '#94a3b8',
    quoteBorder: '#334155',
    h1Color: '#ffffff',
    h2Color: '#ffffff',
    h3Color: '#ffffff',
    h1Font: "'Poppins', sans-serif",
    h2Font: "'Poppins', sans-serif",
    h3Font: "'Poppins', sans-serif"
  });
  const [isShuttingDownTTS, setIsShuttingDownTTS] = useState(false);
  const [isRestartingTTS, setIsRestartingTTS] = useState(false);
  const [directoryPickerKey, setDirectoryPickerKey] = useState(null);
  const [updateStatus, setUpdateStatus] = useState(null);
  const [updateProgress, setUpdateProgress] = useState(null);
  const [updateError, setUpdateError] = useState(null);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [isUpdateRunning, setIsUpdateRunning] = useState(false);
  const [demoShowcaseStatus, setDemoShowcaseStatus] = useState(null);
  const [demoShowcaseInstalling, setDemoShowcaseInstalling] = useState(false);
  const ttsTestFileInputRef = useRef(null);
  const pendingSettingsRef = useRef({});
  const settingsSaveTimerRef = useRef(null);
  const customEndpointsSaveTimerRef = useRef(null);

  // Memory intent input and detected result
  const [memoryIntentInput, setMemoryIntentInput] = useState('');
  const [ttsStreamTestText, setTtsStreamTestText] = useState(
    'This uses the same streaming TTS as chat autoplay over the WebSocket. First sentence. Second sentence: you should hear multiple chunks in a row. Third: replace this with as much text as you want to reproduce issues without sending chat messages.'
  );
  const [memoryIntentDetected, setMemoryIntentDetected] = useState(
    contextSettings.memoryIntentText ?? ''
  );
  const [outreachDraft, setOutreachDraft] = useState({
    name: '',
    characterId: '',
    prompt: '',
    modelName: '',
    intervalMinutes: 45,
    pendingImageFiles: null,
    pendingImageLabel: '',
  });
  const outreachImageInputRef = useRef(null);
  const outreachRuleImageInputRef = useRef(null);
  const [outreachImageUploadRuleId, setOutreachImageUploadRuleId] = useState(null);
  const [outreachImageUploading, setOutreachImageUploading] = useState(false);

  const handleMemoryIntent = useCallback(intent => {
    setMemoryIntentDetected(intent.content);
  }, []);

  const refreshDemoShowcaseStatus = useCallback(async () => {
    if (!PRIMARY_API_URL) return;
    try {
      const data = await fetchDemoShowcaseStatus(PRIMARY_API_URL);
      setDemoShowcaseStatus(data);
    } catch (err) {
      console.warn('Demo showcase status unavailable:', err);
      setDemoShowcaseStatus({ available: false, error: String(err?.message || err) });
    }
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    refreshDemoShowcaseStatus();
  }, [refreshDemoShowcaseStatus]);

  const handleInstallDemoShowcase = useCallback(async () => {
    if (!PRIMARY_API_URL) {
      alert('Backend URL is not configured.');
      return;
    }
    const msg =
      'Install the Call Mode demo showcase?\n\n' +
      'Adds fabricated user Alex Chen, character Mira Vale, sample chat history, profile memories, and agentic insights. ' +
      'Your existing profiles and chats are kept; demo data is merged in and set active.';
    if (!window.confirm(msg)) return;

    setDemoShowcaseInstalling(true);
    try {
      await installDemoShowcase({ apiUrl: PRIMARY_API_URL, setActiveProfile: true, reload: true });
    } catch (err) {
      console.error('Demo showcase install failed:', err);
      alert(`Demo showcase install failed: ${err?.message || err}`);
      setDemoShowcaseInstalling(false);
    }
  }, [PRIMARY_API_URL]);

  const handleOpenTtsTestFilePicker = useCallback(() => {
    ttsTestFileInputRef.current?.click();
  }, []);

  const handleImportTtsTestText = useCallback(async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      setTtsStreamTestText(text);
    } catch (error) {
      console.error('Failed to import TTS test text file:', error);
    } finally {
      event.target.value = '';
    }
  }, []);

  // Source of truth is AppContext `settings` (updated by upsertOutreachRule). `localSettings` is not
  // synced for outreach keys, so listing from context avoids an empty list after save.
  const outreachRules = Array.isArray(contextSettings.outreachRules) ? contextSettings.outreachRules : [];
  const outreachCharacters = (characters || []).filter(c => (c?.chat_role || 'npc') !== 'user');
  const modelOptions = Array.isArray(availableModels) ? availableModels : [];

  const handleSaveOutreachRule = useCallback(async () => {
    if (!outreachDraft.characterId || !outreachDraft.prompt.trim()) {
      alert('Select a character and add a prompt first.');
      return;
    }
    const chosenModel = outreachDraft.modelName || primaryModel || '';
    const ruleId = outreachDraft.id || `outreach-${Date.now()}`;
    upsertOutreachRule({
      id: ruleId,
      name: outreachDraft.name.trim() || 'Scheduled Outreach',
      characterId: outreachDraft.characterId,
      prompt: outreachDraft.prompt.trim(),
      modelName: chosenModel,
      intervalMinutes: Number.parseInt(outreachDraft.intervalMinutes, 10) || 45,
      modelProvider: 'local',
      enabled: true,
      imageCount: outreachDraft.imageCount || 0,
    });
    if (outreachDraft.pendingImageFiles?.length) {
      setOutreachImageUploading(true);
      try {
        const result = await uploadOutreachRuleImages(ruleId, outreachDraft.pendingImageFiles, { replace: true });
        if (!result?.ok) {
          alert('Rule saved, but image folder upload failed. Try attaching the folder again from the saved rule.');
        }
      } finally {
        setOutreachImageUploading(false);
      }
    }
    setOutreachDraft({
      name: '',
      characterId: '',
      prompt: '',
      modelName: '',
      intervalMinutes: 45,
      pendingImageFiles: null,
      pendingImageLabel: '',
    });
  }, [outreachDraft, upsertOutreachRule, primaryModel, uploadOutreachRuleImages]);

  const handleOutreachDraftFolderSelect = useCallback((event) => {
    const fileList = event.target.files;
    if (!fileList?.length) return;
    const images = Array.from(fileList).filter((f) => f.type?.startsWith('image/'));
    if (!images.length) {
      alert('No image files found in that folder.');
      event.target.value = '';
      return;
    }
    const folderLabel = images[0]?.webkitRelativePath?.split('/')[0] || `${images.length} images`;
    setOutreachDraft((prev) => ({
      ...prev,
      pendingImageFiles: images,
      pendingImageLabel: folderLabel,
    }));
    event.target.value = '';
  }, []);

  const handleOutreachRuleFolderSelect = useCallback(async (event) => {
    const ruleId = outreachImageUploadRuleId;
    const fileList = event.target.files;
    event.target.value = '';
    setOutreachImageUploadRuleId(null);
    if (!ruleId || !fileList?.length) return;
    setOutreachImageUploading(true);
    try {
      const result = await uploadOutreachRuleImages(ruleId, fileList, { replace: true });
      if (!result?.ok) alert('Could not upload images for this rule.');
    } finally {
      setOutreachImageUploading(false);
    }
  }, [outreachImageUploadRuleId, uploadOutreachRuleImages]);

  const fetchAvailableVoices = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/tts/voices`);
      if (response.ok) {
        const data = await response.json();
        setAvailableVoices(data);
      }
    } catch (error) {
      console.error("Error fetching available voices:", error);
    }
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    fetchAvailableVoices();
  }, [fetchAvailableVoices]);

  useEffect(() => {
    const fetchGpuInfo = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/system/gpu_info`);
        if (response.ok) {
          const data = await response.json();
          if (data.gpu_count) {
            setGpuCount(data.gpu_count);
          }
        }
      } catch (error) {
        console.error("Error fetching GPU info:", error);
      }
    };

    const fetchTensorSplit = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/models/get-tensor-split`);
        if (response.ok) {
          const data = await response.json();
          if (data.tensor_split) {
            setCurrentTensorSplit(data.tensor_split);
            const input = document.getElementById('tensor-split-input');
            if (input) {
              input.value = data.tensor_split.join(',');
            }
          }
        }
      } catch (error) {
        console.error("Error fetching tensor split:", error);
      }
    };

    fetchGpuInfo();
    fetchTensorSplit();
  }, [PRIMARY_API_URL]);

  useEffect(() => {
    updateSettings({ memoryIntentText: memoryIntentDetected });
  }, [memoryIntentDetected, updateSettings]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const styles = getComputedStyle(document.documentElement);

    const toHex = (value) => {
      const trimmed = String(value || '').trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('#')) return trimmed;
      const rgbMatch = trimmed.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
      if (rgbMatch) {
        const r = parseInt(rgbMatch[1], 10);
        const g = parseInt(rgbMatch[2], 10);
        const b = parseInt(rgbMatch[3], 10);
        return `#${[r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('')}`;
      }
      return trimmed;
    };

    const resolveVarRaw = (name, fallback) => {
      let value = String(styles.getPropertyValue(name) || '').trim();
      if (!value) value = fallback;
      const varMatch = value.match(/^var\((--[^),]+)(?:,[^)]+)?\)$/);
      if (varMatch) {
        const nested = String(styles.getPropertyValue(varMatch[1]) || '').trim();
        value = nested || fallback;
      }
      return value || fallback;
    };

    const resolveColor = (name, fallback) => {
      return toHex(resolveVarRaw(name, fallback)) || fallback;
    };

    const resolveFont = (name, fallback) => {
      return resolveVarRaw(name, fallback) || fallback;
    };

    setMarkdownDefaults({
      body: resolveColor('--md-body-color-default', '#ffffff'),
      bold: resolveColor('--md-bold-color-default', '#ffffff'),
      italic: resolveColor('--md-italic-color-default', '#9ca3af'),
      quote: resolveColor('--md-quote-color-default', '#94a3b8'),
      quoteBorder: resolveColor('--md-quote-border-default', '#334155'),
      h1Color: resolveColor('--md-h1-color-default', '#ffffff'),
      h2Color: resolveColor('--md-h2-color-default', '#ffffff'),
      h3Color: resolveColor('--md-h3-color-default', '#ffffff'),
      h1Font: resolveFont('--md-h1-font-default', "'Poppins', sans-serif"),
      h2Font: resolveFont('--md-h2-font-default', "'Poppins', sans-serif"),
      h3Font: resolveFont('--md-h3-font-default', "'Poppins', sans-serif")
    });
  }, [darkMode]);

  const queueSettingsSave = useCallback((patch) => {
    updateSettings(patch);
    pendingSettingsRef.current = { ...pendingSettingsRef.current, ...patch };
    if (settingsSaveTimerRef.current) {
      clearTimeout(settingsSaveTimerRef.current);
    }
    settingsSaveTimerRef.current = setTimeout(async () => {
      const payload = pendingSettingsRef.current;
      pendingSettingsRef.current = {};
      try {
        await fetch(`${PRIMARY_API_URL}/models/update-settings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch (e) {
        console.error("Failed to auto-save settings to backend:", e);
      }
    }, 300);
  }, [PRIMARY_API_URL, updateSettings]);

  const queueCustomEndpointsSave = useCallback((endpoints) => {
    if (customEndpointsSaveTimerRef.current) {
      clearTimeout(customEndpointsSaveTimerRef.current);
    }
    customEndpointsSaveTimerRef.current = setTimeout(async () => {
      try {
        await fetch(`${PRIMARY_API_URL}/models/save-custom-endpoints`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ customApiEndpoints: endpoints || [] })
        });
      } catch (e) {
        console.error("Failed to auto-save custom endpoints:", e);
      }
    }, 400);
  }, [PRIMARY_API_URL]);

  const handleChange = useCallback((key, value) => {
    setLocalSettings(prev => {
      if (Object.is(prev[key], value)) {
        return prev;
      }
      const updated = { ...prev, [key]: value };
      const isDirectorySetting = DIRECTORY_SETTING_KEYS.includes(key);
      if (!isDirectorySetting) {
        queueSettingsSave({ [key]: value });
        if (key === 'customApiEndpoints') {
          queueCustomEndpointsSave(value);
        }
      }
      return updated;
    });
  }, [contextSettings, queueCustomEndpointsSave, queueSettingsSave]);

  useEffect(() => () => {
    if (settingsSaveTimerRef.current) {
      clearTimeout(settingsSaveTimerRef.current);
    }
    if (customEndpointsSaveTimerRef.current) {
      clearTimeout(customEndpointsSaveTimerRef.current);
    }
  }, []);

  const handleReset = useCallback(() => {
    setLocalSettings({ ...contextSettings });
  }, [contextSettings]);

  useEffect(() => {
    setLocalSettings((prev) => ({ ...prev, ...contextSettings }));
  }, [contextSettings]);



  const handleExportBackendLogs = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/export-logs`);
      if (!response.ok) {
        throw new Error(`Export failed: ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `backend-logs-${new Date().toISOString().split('T')[0]}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to export backend logs:", error);
      alert("Failed to export backend logs.");
    }
  }, [PRIMARY_API_URL]);
  const handleClearBackendLogs = useCallback(async () => {
    if (!confirm("Delete all backend logs? This cannot be undone.")) {
      return;
    }
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/clear-logs`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(`Clear failed: ${response.status}`);
      }
      const data = await response.json();
      alert(`Deleted ${data.deleted ?? 0} log file(s).`);
    } catch (error) {
      console.error("Failed to clear backend logs:", error);
      alert("Failed to clear backend logs.");
    }
  }, [PRIMARY_API_URL]);

  const handleCheckUpdates = useCallback(async () => {
    setIsCheckingUpdate(true);
    setUpdateError(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update-status?fetch=1`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Status ${response.status}`);
      }
      setUpdateStatus(data);
    } catch (error) {
      setUpdateError(error.message || 'Failed to check updates.');
    } finally {
      setIsCheckingUpdate(false);
    }
  }, [PRIMARY_API_URL]);

  const fetchUpdateProgress = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update-progress`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Update failed (${response.status})`);
      }
      setUpdateProgress(data);
      if (data.status !== 'running') {
        setIsUpdateRunning(false);
        if (data.status === 'failed') {
          setUpdateError(data.error || 'Update failed.');
        }
        if (data.status === 'success' && data.restart_recommended) {
          alert('Update complete. Please restart the app to apply changes.');
        }
      }
    } catch (error) {
      setUpdateError(error.message || 'Failed to fetch update progress.');
      setIsUpdateRunning(false);
    }
  }, [PRIMARY_API_URL]);

  const handleRunUpdate = useCallback(async () => {
    if (!confirm("Update to the latest git version? This will discard local changes in the app folder. A restart may be required.")) {
      return;
    }
    setUpdateError(null);
    setUpdateProgress(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/system/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok && response.status !== 409) {
        throw new Error(data.detail || data.message || `Update failed (${response.status})`);
      }
      setIsUpdateRunning(true);
      fetchUpdateProgress();
    } catch (error) {
      setUpdateError(error.message || 'Failed to start update.');
      setIsUpdateRunning(false);
    }
  }, [PRIMARY_API_URL, fetchUpdateProgress]);

  useEffect(() => {
    if (!isUpdateRunning) return;
    const timer = setInterval(fetchUpdateProgress, 1000);
    return () => clearInterval(timer);
  }, [isUpdateRunning, fetchUpdateProgress]);

  const handleDirectoryBrowse = useCallback(async (settingKey, title) => {
    const baseUrl = PRIMARY_API_URL || getBackendUrl();
    setDirectoryPickerKey(settingKey);
    try {
      const response = await fetch(`${baseUrl}/system/select-directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          initial_directory: localSettings[settingKey] || null,
          title
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to open directory picker.');
      }
      if (data.status === 'cancelled' || !data.directory) {
        return;
      }
      handleChange(settingKey, data.directory);
    } catch (error) {
      console.error('Directory picker failed:', error);
      const isNetworkError = !error.message || /failed to fetch|network error|load failed/i.test(String(error.message));
      const message = isNetworkError
        ? 'Could not reach the backend. If the backend window closed or crashed, restart the app, then try again. You can also type or paste the folder path manually.'
        : `Directory picker failed: ${error.message}`;
      alert(message);
    } finally {
      setDirectoryPickerKey(null);
    }
  }, [PRIMARY_API_URL, handleChange, localSettings]);

  const updateLogLines = updateProgress?.logs || [];
  const updateLogText = updateLogLines
    .map((entry) => `[${entry.ts}] ${String(entry.level).toUpperCase()}: ${entry.message}`)
    .join('\n');
  const customEndpoints = localSettings.customApiEndpoints || [];
  const enabledEndpointCount = customEndpoints.filter((endpoint) => endpoint?.enabled).length;
  const endpointSummary = customEndpoints.length
    ? `${enabledEndpointCount}/${customEndpoints.length} enabled${localSettings.apiEndpointRoundRobinEnabled ? ' · auto-routing on' : ''}`
    : 'No custom endpoints configured';
  const automationSummary = `${outreachRules.length} outreach rule${outreachRules.length === 1 ? '' : 's'} · D-ID quick buttons`;

  return (
    <div className="w-full min-h-screen p-2 md:p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <h2 className="text-2xl font-bold mb-4">Settings</h2>
        <p className="text-sm text-muted-foreground">
          Changes save automatically. Directory fields still require the Save button.
        </p>
        <Tabs value={settingsMainTab} onValueChange={setSettingsMainTab} className="space-y-6">
          <div className="border rounded-lg bg-card p-1 overflow-x-auto">
            <TabsList className="flex w-full flex-wrap justify-start gap-1 h-auto min-h-[40px]">
            <TabsTrigger value="general" className="flex-shrink-0">General</TabsTrigger>
            <TabsTrigger value="styles" className="flex-shrink-0">Styles</TabsTrigger>
            <TabsTrigger value="generation" className="flex-shrink-0">LLM Settings</TabsTrigger>
            <TabsTrigger value="nano-gpt-memory" className="flex-shrink-0">NanoGPT memory</TabsTrigger>
            <TabsTrigger value="image-generation" className="flex-shrink-0">Image Generation</TabsTrigger>
            <TabsTrigger value="rag" className="flex-shrink-0">Document Context</TabsTrigger>
            <TabsTrigger value="characters" className="flex-shrink-0">Characters</TabsTrigger>
            <TabsTrigger value="audio" className="flex-shrink-0">Audio</TabsTrigger>
            <TabsTrigger value="memory-intent" className="flex-shrink-0">Memory Intent</TabsTrigger>
            <TabsTrigger value="persona-realignment" className="flex-shrink-0">Persona realignment</TabsTrigger>
            <TabsTrigger value="chatlog-condenser" className="flex-shrink-0">Chatlog condenser</TabsTrigger>
            <TabsTrigger value="memory" className="flex-shrink-0">Memory Browser</TabsTrigger>
            <TabsTrigger value="lore" className="flex-shrink-0">Lore Debugger</TabsTrigger>
            <TabsTrigger value="about" className="flex-shrink-0">About</TabsTrigger>
            <TabsTrigger value="profiles" className="flex-shrink-0">User Profiles</TabsTrigger>
            </TabsList>
          </div>

        {/* General */}
        <TabsContent value="general">
          <div className="space-y-6">
            <SettingsSection
              title="Interface"
              description="Theme and layout controls for the main UI."
            >
              <SettingRow label="Dark Mode" htmlFor="dark-mode" description="Toggle the dark theme.">
                <div className="flex items-center justify-end gap-2">
                  <Sun className="h-4 w-4" />
                  <Switch id="dark-mode" checked={darkMode} onCheckedChange={toggleDarkMode} />
                  <Moon className="h-4 w-4" />
                </div>
              </SettingRow>
              <SettingRow
                label="Splash screen duration"
                htmlFor="splash-screen-duration"
                description="How long the Eloquent logo stays visible on startup before fading out. Shorter times apply when reduced motion is enabled in your OS."
              >
                <Select
                  value={localSettings.splashScreenDuration || SPLASH_SCREEN_DURATION_DEFAULT}
                  onValueChange={(value) => handleChange('splashScreenDuration', value)}
                >
                  <SelectTrigger id="splash-screen-duration" className="w-full md:max-w-xs">
                    <SelectValue placeholder="Normal" />
                  </SelectTrigger>
                  <SelectContent>
                    {SPLASH_DURATION_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label} — {opt.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingRow>
              <SettingRow
                label={`User Avatar Size (${userAvatarSize}px)`}
                htmlFor="user-avatar-size"
                layout="stack"
                description="Controls the avatar size for your messages."
              >
                <Slider
                  id="user-avatar-size"
                  min={64}
                  max={512}
                  step={16}
                  value={[userAvatarSize]}
                  onValueChange={([v]) => setUserAvatarSize(v)}
                />
              </SettingRow>
              <SettingRow
                label={`Character Avatar Size (${characterAvatarSize}px)`}
                htmlFor="char-avatar-size"
                layout="stack"
                description="Controls the display size of characters' avatars in chat."
              >
                <Slider
                  id="char-avatar-size"
                  min={64}
                  max={512}
                  step={16}
                  value={[characterAvatarSize]}
                  onValueChange={([v]) => setCharacterAvatarSize(v)}
                />
              </SettingRow>
              <SettingRow
                label="TV / projector performance mode"
                htmlFor="tv-performance-mode"
                description="Larger text, disables backdrop blur, and uses lighter shadows so the UI stays smooth on Android TV browsers (for example Browsehere). You can also open the app with ?tv=1 or #tv in the address bar on this device only."
              >
                <div className="flex items-center justify-end gap-2">
                  <Monitor className="h-4 w-4 text-muted-foreground" aria-hidden />
                  <Switch
                    id="tv-performance-mode"
                    checked={tvPerfStored}
                    onCheckedChange={(checked) => {
                      setTvPerfStored(checked);
                      try {
                        if (checked) localStorage.setItem(TV_PERF_STORAGE_KEY, '1');
                        else localStorage.removeItem(TV_PERF_STORAGE_KEY);
                      } catch (_) {
                        /* ignore */
                      }
                      applyTvPerformanceClass(readTvPerformanceFromUrl() || checked);
                    }}
                  />
                </div>
              </SettingRow>
              <SettingRow
                label="Open settings in second window"
                htmlFor="open-settings-second-window"
                description="When enabled, clicking Settings in the sidebar opens a separate window — useful for dual-monitor setups. Changes sync live between windows."
              >
                <Switch
                  id="open-settings-second-window"
                  checked={localSettings.openSettingsInSecondWindow === true}
                  onCheckedChange={(value) => handleChange('openSettingsInSecondWindow', value)}
                />
              </SettingRow>
              {!isStandaloneWindow && (
                <SettingRow
                  label="Open settings window now"
                  description="Pop out settings to another screen without changing the default behavior above."
                >
                  <Button type="button" variant="outline" size="sm" onClick={() => openSettingsWindow(settingsMainTab)}>
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Open in new window
                  </Button>
                </SettingRow>
              )}
            </SettingsSection>

            <MobileRemoteSettings />

            <SettingsSection
              title="Access and Endpoints"
              description="Security and API endpoints for the running services."
            >
              <SettingRow
                label="Remote Access Password"
                htmlFor="admin-password"
                description="Set a password to protect your instance."
              >
                <Input
                  id="admin-password"
                  type="password"
                  value={localSettings.admin_password || ''}
                  className="w-full md:max-w-xs"
                  onChange={(e) => handleChange('admin_password', e.target.value)}
                  placeholder="No password set"
                />
              </SettingRow>
              <SettingRow label="Primary API URL" htmlFor="primary-api-url" description="Main backend address (read-only).">
                <Input id="primary-api-url" value={PRIMARY_API_URL} readOnly className="w-full md:max-w-xs" />
              </SettingRow>
              <SettingRow label="Secondary API URL" htmlFor="secondary-api-url" description="Secondary backend address (read-only).">
                <Input id="secondary-api-url" value={SECONDARY_API_URL} readOnly className="w-full md:max-w-xs" />
              </SettingRow>
            </SettingsSection>

            <SettingsAccordion
              title="Advanced and developer tools"
              summary="Backups, local browser storage, updates, model paths, startup, and service controls."
            >

            <SettingsSection
              title="Backups and Logs"
              description="Export includes IndexedDB data (characters, chats, settings, profiles). Use the same backend after import so character and user IDs still match agentic memory and memory-store files on disk."
            >
              <SettingRow label="Local Browser Backup" layout="stack">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      try {
                        const config = {};

                        // 1) Capture localStorage keys (current browser origin)
                        try {
                          for (let i = 0; i < localStorage.length; i += 1) {
                            const key = localStorage.key(i);
                            if (!key) continue;
                            const value = localStorage.getItem(key);
                            if (value != null) {
                              config[key] = value;
                            }
                          }
                        } catch (e) {
                          console.warn('Failed to read some localStorage keys during export:', e);
                        }

                        // 2) Capture IndexedDB-backed keys via indexedDbStorage
                        try {
                          const idbKeys = await indexedDbStorage.getAllKeys();
                          for (const key of idbKeys || []) {
                            try {
                              const value = await indexedDbStorage.getItem(key);
                              if (value != null && value !== '') {
                                config[key] = value;
                              }
                            } catch (e) {
                              console.warn('Failed to read IndexedDB key during export:', key, e);
                            }
                          }
                        } catch (e) {
                          console.warn('Failed to enumerate IndexedDB keys during export:', e);
                        }

                        // 3) Force-include critical keys so backup always has profiles/settings/endpoints.
                        for (const key of ['user-profiles', 'llm-characters', 'Eloquent-settings', 'LiangLocal-settings']) {
                          try {
                            const value = await indexedDbStorage.getItem(key);
                            if (value != null && value !== '') config[key] = value;
                          } catch (e) {
                            console.warn('Failed to capture critical backup key:', key, e);
                          }
                        }

                        config._eloquentExport = {
                          version: 2,
                          exportedAt: new Date().toISOString(),
                          includesIndexedDbKeys: true,
                        };

                        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `eloquent-config-${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (err) {
                        console.error('Backup export failed:', err);
                        alert('Failed to export backup');
                      }
                    }}
                    className="flex-1"
                  >
                    <DownloadCloud className="mr-2 h-4 w-4" />
                    Export Backup
                  </Button>

                  <Button
                    variant="outline"
                    onClick={async () => {
                      try {
                        const chars = await indexedDbStorage.getItem('llm-characters');
                        const profiles = await indexedDbStorage.getItem('user-profiles');
                        if (!chars && !profiles) {
                          alert('No characters or user profiles found in this browser storage.');
                          return;
                        }
                        const payload = {
                          _eloquentExport: {
                            version: 2,
                            exportedAt: new Date().toISOString(),
                            subset: 'characters_and_profiles',
                          },
                          ...(chars ? { 'llm-characters': chars } : {}),
                          ...(profiles ? { 'user-profiles': profiles } : {}),
                        };
                        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `eloquent-characters-profiles-${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (err) {
                        console.error('Characters/profiles export failed:', err);
                        alert('Failed to export characters and profiles');
                      }
                    }}
                    className="flex-1"
                  >
                    <DownloadCloud className="mr-2 h-4 w-4" />
                    Export characters + profiles
                  </Button>

                  <div className="relative flex-1">
                    <input
                      type="file"
                      accept=".json"
                      onChange={async (e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = async (ev) => {
                            try {
                              const text = ev.target?.result;
                              if (!text || typeof text !== 'string') {
                                alert('Backup file was empty or unreadable');
                                return;
                              }
                              const config = JSON.parse(text);
                              if (confirm('Import local backup? This will overwrite your current browser data.')) {
                                const entries = Object.entries(config || {}).filter(
                                  ([k]) => !k.startsWith('_eloquent')
                                );
                                const normalize = (v) => (typeof v === 'string' ? v : JSON.stringify(v));
                                const normalizeSettings = (raw) => {
                                  try {
                                    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
                                    if (parsed && typeof parsed === 'object') {
                                      return JSON.stringify(parsed);
                                    }
                                  } catch (_) {}
                                  return normalize(raw);
                                };
                                const normalizeProfiles = (raw) => {
                                  try {
                                    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
                                    let root = parsed;
                                    if (root && typeof root === 'object' && typeof root['user-profiles'] !== 'undefined') {
                                      root = root['user-profiles'];
                                    }
                                    if (typeof root === 'string') root = JSON.parse(root);
                                    const profiles = Array.isArray(root?.profiles) ? root.profiles : [];
                                    const safeProfiles = profiles
                                      .map((p) => ({
                                        id: String(p?.id || `profile_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`),
                                        name: String(p?.name || 'User'),
                                        avatar: p?.avatar || null,
                                        preferences: p?.preferences || { topics: [], responseStyle: 'balanced' },
                                      }))
                                      .filter((p) => !!p.id);
                                    const activeProfileId = safeProfiles.some((p) => p.id === root?.activeProfileId)
                                      ? root.activeProfileId
                                      : safeProfiles[0]?.id || null;
                                    return JSON.stringify({ profiles: safeProfiles, activeProfileId });
                                  } catch (_) {
                                    return normalize(raw);
                                  }
                                };

                                // 0) Clear current app data first, so restore is deterministic (no mixed old/new state).
                                try {
                                  const existingIdbKeys = await indexedDbStorage.getAllKeys();
                                  for (const key of existingIdbKeys || []) {
                                    // eslint-disable-next-line no-await-in-loop
                                    await indexedDbStorage.removeItem(String(key));
                                  }
                                } catch (err) {
                                  console.warn('Failed to clear existing IndexedDB keys before restore:', err);
                                }
                                try {
                                  const keysToRemove = [];
                                  for (let i = 0; i < localStorage.length; i += 1) {
                                    const key = localStorage.key(i);
                                    if (!key) continue;
                                    const looksAppKey =
                                      key.startsWith('LiangLocal-') ||
                                      key.startsWith('Eloquent-') ||
                                      key.startsWith('eloquent-') ||
                                      key.startsWith('adetailer-') ||
                                      key.startsWith('remote-') ||
                                      key === 'llm-characters' ||
                                      key === 'user-profiles' ||
                                      key === 'conversations' ||
                                      key === 'preferredContextLength' ||
                                      key === 'vite-ui-theme';
                                    if (looksAppKey) keysToRemove.push(key);
                                  }
                                  keysToRemove.forEach((k) => {
                                    try { localStorage.removeItem(k); } catch (_) {}
                                  });
                                } catch (err) {
                                  console.warn('Failed to clear existing localStorage keys before restore:', err);
                                }

                                // 1) Write keys that are managed by indexedDbStorage into IndexedDB
                                for (const [k, v] of entries) {
                                  try {
                                    if (indexedDbStorage.useIdb(k)) {
                                      const normalizedValue =
                                        k === 'user-profiles'
                                          ? normalizeProfiles(v)
                                          : (k === 'Eloquent-settings' || k === 'LiangLocal-settings')
                                            ? normalizeSettings(v)
                                            : normalize(v);
                                      // eslint-disable-next-line no-await-in-loop
                                      await indexedDbStorage.setItem(k, normalizedValue);
                                    }
                                  } catch (err) {
                                    console.warn('Failed to restore IndexedDB key from backup:', k, err);
                                  }
                                }

                                // 2) Write everything into localStorage as a mirror (for legacy readers)
                                try {
                                  for (const [k, v] of entries) {
                                    const normalizedValue =
                                      k === 'user-profiles'
                                        ? normalizeProfiles(v)
                                        : (k === 'Eloquent-settings' || k === 'LiangLocal-settings')
                                          ? normalizeSettings(v)
                                          : normalize(v);
                                    localStorage.setItem(k, normalizedValue);
                                  }
                                  // Mirror settings keys so custom API endpoints reliably rehydrate.
                                  const eSettings = localStorage.getItem('Eloquent-settings');
                                  const lSettings = localStorage.getItem('LiangLocal-settings');
                                  if (eSettings && !lSettings) localStorage.setItem('LiangLocal-settings', eSettings);
                                  if (lSettings && !eSettings) localStorage.setItem('Eloquent-settings', lSettings);
                                } catch (err) {
                                  console.warn('Failed to restore some localStorage keys from backup:', err);
                                }

                                try {
                                  localStorage.setItem('LiangLocal-prefer-local-profiles', '1');
                                } catch (_) {
                                  /* ignore */
                                }

                                window.location.reload();
                              }
                            } catch (err) {
                              console.error('Backup import failed:', err);
                              alert('Failed to parse backup file');
                            }
                          };
                          reader.readAsText(file);
                          e.target.value = '';
                        }
                      }}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                    <Button variant="outline" className="w-full">
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Import Backup
                    </Button>
                  </div>
                </div>
              </SettingRow>

              <SettingRow
                label="User Profiles Only"
                layout="stack"
                description="Quick backup/restore just for user profiles (used by the Profile selector)."
              >
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      try {
                        const raw = await indexedDbStorage.getItem('user-profiles');
                        if (!raw) {
                          alert('No user profiles found to export on this browser.');
                          return;
                        }
                        let parsed;
                        try {
                          parsed = JSON.parse(raw);
                        } catch {
                          // If it isn't valid JSON, still export the raw string
                          parsed = raw;
                        }
                        const payload = { 'user-profiles': parsed };
                        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `eloquent-user-profiles-${new Date().toISOString().split('T')[0]}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (err) {
                        console.error('User profile export failed:', err);
                        alert('Failed to export user profiles');
                      }
                    }}
                    className="flex-1"
                  >
                    <DownloadCloud className="mr-2 h-4 w-4" />
                    Export Profiles
                  </Button>

                  <div className="relative flex-1">
                    <input
                      type="file"
                      accept=".json"
                      onChange={async (e) => {
                        const file = e.target.files?.[0];
                        if (!file) return;
                        const reader = new FileReader();
                        reader.onload = async (ev) => {
                          try {
                            const text = ev.target?.result;
                            if (!text || typeof text !== 'string') {
                              alert('Profile backup file was empty or unreadable');
                              return;
                            }
                            const data = JSON.parse(text);

                            // Support both { profiles, activeProfileId } and { "user-profiles": ... } shapes
                            let valueForStorage = null;
                            if (data && data.profiles && Array.isArray(data.profiles)) {
                              valueForStorage = data;
                            } else if (data && typeof data['user-profiles'] !== 'undefined') {
                              const v = data['user-profiles'];
                              valueForStorage = typeof v === 'string' ? JSON.parse(v) : v;
                            } else {
                              alert('Profile backup did not contain any user-profiles data.');
                              return;
                            }

                            const serialized = JSON.stringify(valueForStorage);

                            // Write to IndexedDB and localStorage under the expected key
                            await indexedDbStorage.setItem('user-profiles', serialized);
                            try {
                              localStorage.setItem('user-profiles', serialized);
                            } catch (_) {}

                            try {
                              localStorage.setItem('LiangLocal-prefer-local-profiles', '1');
                            } catch (_) {
                              /* ignore */
                            }

                            const count = Array.isArray(valueForStorage.profiles)
                              ? valueForStorage.profiles.length
                              : 0;
                            alert(`Imported user profiles (${count} profiles). Reloading...`);
                            window.location.reload();
                          } catch (err) {
                            console.error('User profile import failed:', err);
                            alert('Failed to import user profiles');
                          }
                        };
                        reader.readAsText(file);
                        e.target.value = '';
                      }}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                    <Button variant="outline" className="w-full">
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Import Profiles
                    </Button>
                  </div>
                </div>
              </SettingRow>

              <SettingRow
                label="Backend Logs"
                layout="stack"
                description="Export backend logs for bug reports."
              >
                <div className="flex flex-col md:flex-row gap-2">
                  <Button variant="outline" onClick={handleExportBackendLogs}>
                    Export Backend Logs
                  </Button>
                  <Button variant="outline" onClick={handleClearBackendLogs}>
                    Delete Old Logs
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Call Mode demo showcase"
              description="Install a fully fabricated test user (Alex Chen), character (Mira Vale), chat history, profile memories, and agentic insights — for demos without exposing your real chats."
            >
              <SettingRow label="Demo data" layout="stack">
                <div className="flex flex-col gap-3">
                  {demoShowcaseStatus?.available === false && (
                    <Alert variant="destructive">
                      <AlertTitle>Demo pack unavailable</AlertTitle>
                      <AlertDescription>
                        {demoShowcaseStatus?.error || 'Could not reach the demo showcase endpoints on the backend.'}
                      </AlertDescription>
                    </Alert>
                  )}
                  {demoShowcaseStatus?.available && (
                    <p className="text-sm text-muted-foreground">
                      Backend: {demoShowcaseStatus.installed ? 'installed' : 'not installed'}
                      {' · '}
                      {demoShowcaseStatus.memory_count ?? 0} profile memories
                      {' · '}
                      {demoShowcaseStatus.agentic_count ?? 0} agentic insights
                      {demoShowcaseStatus.demo_is_active ? ' · active profile' : ''}
                    </p>
                  )}
                  <div className="flex flex-col md:flex-row gap-2">
                    <Button
                      variant="default"
                      onClick={handleInstallDemoShowcase}
                      disabled={demoShowcaseInstalling || demoShowcaseStatus?.available === false}
                    >
                      {demoShowcaseInstalling ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Sparkles className="mr-2 h-4 w-4" />
                      )}
                      {demoShowcaseInstalling ? 'Installing…' : 'Install demo showcase'}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={refreshDemoShowcaseStatus}
                      disabled={demoShowcaseInstalling}
                    >
                      <RefreshCw className="mr-2 h-4 w-4" />
                      Refresh status
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    After install, open chat &quot;Late night — Chapter 7&quot; with Mira Vale and enter call mode. IDs: profile_demo, char_mira_vale.
                  </p>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Local browser storage"
              description="Inspect IndexedDB and stray localStorage keys, delete large or obsolete data to free space. Removing a key reloads the page so the app stays consistent. Critical keys require typing DELETE."
            >
              <SettingsAccordion
                title="Local browser storage (advanced)"
                summary="Inspect and delete IndexedDB/localStorage keys. High-impact developer maintenance."
              >
                <LocalStorageManager conversations={conversations} />
              </SettingsAccordion>
            </SettingsSection>

            <SettingsSection
              title="App Updates"
              description="Force update to the latest git version. Local changes will be discarded."
            >
              <SettingRow label="Update Controls" layout="stack" description="Live progress and logs are shown while updating.">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={handleCheckUpdates}
                    disabled={isCheckingUpdate || isUpdateRunning}
                  >
                    {isCheckingUpdate ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="mr-2 h-4 w-4" />
                    )}
                    Check for Updates
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleRunUpdate}
                    disabled={isUpdateRunning || isCheckingUpdate}
                  >
                    {isUpdateRunning ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <RotateCw className="mr-2 h-4 w-4" />
                    )}
                    Update Now
                  </Button>
                </div>

                {updateStatus && (
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div>
                      Branch: {updateStatus.branch || 'unknown'} (
                      {updateStatus.current_commit ? updateStatus.current_commit.slice(0, 7) : 'unknown'})
                    </div>
                    {updateStatus.upstream ? (
                      <div>
                        Tracking: {updateStatus.upstream} - Ahead {updateStatus.ahead ?? 'n/a'} - Behind {updateStatus.behind ?? 'n/a'}
                      </div>
                    ) : (
                      <div>No upstream configured for this branch.</div>
                    )}
                    <div>Working tree: {updateStatus.dirty ? `dirty (${updateStatus.dirty_count})` : 'clean'}</div>
                  </div>
                )}

                {updateProgress && (
                  <div className="rounded-lg border border-border/60 bg-muted/30 p-3 space-y-2">
                    <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Update Status</div>
                    <div className="text-sm text-foreground">
                      {updateProgress.status === 'running' ? 'Running' : updateProgress.status}
                      {updateProgress.step ? ` - ${updateProgress.step}` : ''}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {updateProgress.before ? `Before: ${updateProgress.before.slice(0, 7)}` : ''}
                      {updateProgress.after ? ` | After: ${updateProgress.after.slice(0, 7)}` : ''}
                    </div>
                    {updateProgress.error && (
                      <Alert variant="destructive">
                        <AlertTitle>Update failed</AlertTitle>
                        <AlertDescription>{updateProgress.error}</AlertDescription>
                      </Alert>
                    )}
                    {updateLogLines.length > 0 && (
                      <div className="space-y-2">
                        <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Update Log</div>
                        <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-md bg-background/80 p-3 text-xs text-foreground">
{updateLogText}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {typeof updateStatus?.behind === 'number' && updateStatus.behind > 0 && (
                  <Alert>
                    <AlertTitle>Update available</AlertTitle>
                    <AlertDescription>Behind by {updateStatus.behind} commit(s).</AlertDescription>
                  </Alert>
                )}

                {updateError && (
                  <Alert variant="destructive">
                    <AlertTitle>Update failed</AlertTitle>
                    <AlertDescription>{updateError}</AlertDescription>
                  </Alert>
                )}
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Models and Storage"
              description="Directories for local model files."
            >
              <SettingRow label="Models Directory" htmlFor="model-directory" description="Location for local GGUF models.">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="model-directory"
                    value={localSettings.modelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('modelDirectory', e.target.value)}
                    placeholder="C:\models\gguf"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('modelDirectory', 'Select Models Directory')}
                    disabled={directoryPickerKey === 'modelDirectory'}
                  >
                    {directoryPickerKey === 'modelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.modelDirectory) {
                        queueSettingsSave({ modelDirectory: localSettings.modelDirectory });
                        fetch(`${PRIMARY_API_URL}/models/refresh-directory`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.modelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated! Restart required.' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Startup and GPU"
              description="Launch behavior and GPU distribution."
            >
              <SettingRow
                label="Auto-Launch Browser"
                htmlFor="auto-launch-browser"
                description="Automatically open the browser window on startup."
              >
                <Switch
                  id="auto-launch-browser"
                  checked={localSettings.auto_launch_browser}
                  onCheckedChange={(value) => handleChange('auto_launch_browser', value)}
                />
              </SettingRow>

              <SettingRow
                label="Single GPU Mode"
                htmlFor="single-gpu-mode"
                description={gpuCount <= 1 ? 'Automatically enabled (single GPU detected).' : 'Enable for single GPU setup.'}
              >
                <Switch
                  id="single-gpu-mode"
                  checked={localSettings.singleGpuMode || gpuCount <= 1}
                  disabled={gpuCount <= 1}
                  onCheckedChange={(value) => handleChange('singleGpuMode', value)}
                />
              </SettingRow>

              {gpuCount > 1 && (
                <SettingRow
                  label="Main Model GPU"
                  htmlFor="main-gpu-id"
                  description="Select which GPU runs the heavy LLM model service."
                >
                  <Select
                    value={localSettings.main_gpu_id?.toString() || "0"}
                    onValueChange={(value) => handleChange('main_gpu_id', parseInt(value, 10))}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select Main GPU" />
                    </SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: gpuCount }, (_, i) => (
                        <SelectItem key={i} value={i.toString()}>
                          GPU {i}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </SettingRow>
              )}

              {!localSettings.singleGpuMode && (
                <>
                  <SettingRow label="GPU Usage Mode (Dual GPU)" htmlFor="gpu-usage-mode">
                    <Select
                      value={localSettings.gpuUsageMode || 'split_services'}
                      onValueChange={(value) => {
                        handleChange('gpuUsageMode', value);
                        fetch(`${PRIMARY_API_URL}/models/update-gpu-mode`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ gpuUsageMode: value })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated! Restart required.' : d.message));
                      }}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select GPU usage mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="split_services">Split Services</SelectItem>
                        <SelectItem value="unified_model">Unified Model</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  {localSettings.gpuUsageMode === 'unified_model' && (
                    <SettingRow label="Tensor Split Ratio" htmlFor="tensor-split-input" layout="stack">
                      <div className="flex flex-col md:flex-row gap-3">
                        <Input
                          id="tensor-split-input"
                          type="text"
                          placeholder="1,1"
                          defaultValue={currentTensorSplit.join(',')}
                          key={currentTensorSplit.join(',')}
                          className="flex-1"
                        />
                        <Button
                          variant="outline"
                          className="w-full md:w-auto"
                          onClick={(e) => {
                            const input = document.getElementById('tensor-split-input');
                            const value = input.value.trim();
                            const parts = value.split(',').map(s => parseFloat(s.trim()));
                            const total = parts.reduce((a, b) => a + b, 0);
                            const normalized = parts.map(v => v / total);

                            fetch(`${PRIMARY_API_URL}/models/update-tensor-split`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ tensor_split: normalized })
                            }).then(r => r.json()).then(data => {
                              if (data.status === 'success') {
                                alert('Updated!');
                                setCurrentTensorSplit(data.tensor_split);
                              } else alert(data.message);
                            });
                          }}
                        >
                          <Save className="mr-2 h-4 w-4" /> Apply
                        </Button>
                      </div>
                    </SettingRow>
                  )}
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Services"
              description="Manage optional background services and VRAM usage."
            >
              <SettingRow label="Forensic Models Management" layout="stack">
                <Button
                  variant="outline"
                  onClick={async () => {
                    setIsUnloadingForensicModels(true);
                    try {
                      await fetch(`${PRIMARY_API_URL}/forensic/unload-models`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' }
                      });
                      alert('Unloaded!');
                    } finally { setIsUnloadingForensicModels(false); }
                  }}
                  disabled={isUnloadingForensicModels}
                  className="w-full"
                >
                  {isUnloadingForensicModels ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Power className="mr-2 h-4 w-4" />}
                  Unload RoBERTa/STAR Models
                </Button>
              </SettingRow>

              <SettingRow label="TTS Service Management (Port 8002)" layout="stack">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsShuttingDownTTS(true);
                      try {
                        await fetch(`${PRIMARY_API_URL}/tts/shutdown`, { method: 'POST' });
                        alert('Shutdown initiated.');
                      } finally { setIsShuttingDownTTS(false); }
                    }}
                    disabled={isShuttingDownTTS}
                    className="flex-1"
                  >
                    <Power className="mr-2 h-4 w-4" /> Shutdown TTS
                  </Button>
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsRestartingTTS(true);
                      try {
                        await fetch(`${PRIMARY_API_URL}/tts/restart`, { method: 'POST' });
                        alert('Restarting...');
                      } finally { setIsRestartingTTS(false); }
                    }}
                    disabled={isRestartingTTS}
                    className="flex-1"
                  >
                    <RotateCw className="mr-2 h-4 w-4" /> Restart TTS
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>
            </SettingsAccordion>

          </div>
        </TabsContent>

        {/* Styles */}
        <TabsContent value="styles">
          <div className="space-y-6">
            <SettingsSection
              title="Message Styling"
              description="Quick access to text and markdown presentation controls."
            >
              <SettingRow
                label="Plain Text Color"
                htmlFor="md-body-color"
                description="Applies to unformatted text inside messages."
              >
                <div className="flex items-center gap-2">
                  <Input
                    id="md-body-color"
                    type="color"
                    value={localSettings.mdBodyColor || markdownDefaults.body}
                    onChange={(e) => handleChange('mdBodyColor', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdBodyColor', '')}
                  >
                    Reset
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Bold Text Color"
                htmlFor="md-bold-color"
                description="Applies to **bold** text."
              >
                <div className="flex items-center gap-2">
                  <Input
                    id="md-bold-color"
                    type="color"
                    value={localSettings.mdBoldColor || markdownDefaults.bold}
                    onChange={(e) => handleChange('mdBoldColor', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdBoldColor', '')}
                  >
                    Reset
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Italic Text Color"
                htmlFor="md-italic-color"
                description="Applies to *italic* text (often used for actions)."
              >
                <div className="flex items-center gap-2">
                  <Input
                    id="md-italic-color"
                    type="color"
                    value={localSettings.mdItalicColor || markdownDefaults.italic}
                    onChange={(e) => handleChange('mdItalicColor', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdItalicColor', '')}
                  >
                    Reset
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Quote Text Color"
                htmlFor="md-quote-color"
                description="Applies to text inside quotes."
              >
                <div className="flex items-center gap-2">
                  <Input
                    id="md-quote-color"
                    type="color"
                    value={localSettings.mdQuoteColor || markdownDefaults.quote}
                    onChange={(e) => handleChange('mdQuoteColor', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdQuoteColor', '')}
                  >
                    Reset
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Quote Border Color"
                htmlFor="md-quote-border"
                description="Controls the left border of blockquotes."
              >
                <div className="flex items-center gap-2">
                  <Input
                    id="md-quote-border"
                    type="color"
                    value={localSettings.mdQuoteBorder || markdownDefaults.quoteBorder}
                    onChange={(e) => handleChange('mdQuoteBorder', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdQuoteBorder', '')}
                  >
                    Reset
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Heading Styles"
              description="Control the appearance of Markdown headings."
            >
              <SettingRow
                label="Heading 1 (H1)"
                htmlFor="md-h1-color"
                description="Applies to # Heading."
                layout="stack"
              >
                <div className="flex flex-wrap items-center gap-3">
                  <Input
                    id="md-h1-color"
                    type="color"
                    value={localSettings.mdH1Color || markdownDefaults.h1Color}
                    onChange={(e) => handleChange('mdH1Color', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Select
                    value={resolveHeadingFontValue(localSettings.mdH1Font)}
                    onValueChange={(value) => handleChange('mdH1Font', normalizeHeadingFontValue(value))}
                  >
                    <SelectTrigger className="w-full md:w-64">
                      <SelectValue placeholder="Default (Theme)" />
                    </SelectTrigger>
                    <SelectContent>
                      {headingFontOptions.map((option) => (
                        <SelectItem key={option.label} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdH1Color', '')}
                  >
                    Reset Color
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Heading 2 (H2)"
                htmlFor="md-h2-color"
                description="Applies to ## Heading."
                layout="stack"
              >
                <div className="flex flex-wrap items-center gap-3">
                  <Input
                    id="md-h2-color"
                    type="color"
                    value={localSettings.mdH2Color || markdownDefaults.h2Color}
                    onChange={(e) => handleChange('mdH2Color', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Select
                    value={resolveHeadingFontValue(localSettings.mdH2Font)}
                    onValueChange={(value) => handleChange('mdH2Font', normalizeHeadingFontValue(value))}
                  >
                    <SelectTrigger className="w-full md:w-64">
                      <SelectValue placeholder="Default (Theme)" />
                    </SelectTrigger>
                    <SelectContent>
                      {headingFontOptions.map((option) => (
                        <SelectItem key={option.label} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdH2Color', '')}
                  >
                    Reset Color
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="Heading 3 (H3)"
                htmlFor="md-h3-color"
                description="Applies to ### Heading."
                layout="stack"
              >
                <div className="flex flex-wrap items-center gap-3">
                  <Input
                    id="md-h3-color"
                    type="color"
                    value={localSettings.mdH3Color || markdownDefaults.h3Color}
                    onChange={(e) => handleChange('mdH3Color', e.target.value)}
                    className="h-10 w-16 p-1"
                  />
                  <Select
                    value={resolveHeadingFontValue(localSettings.mdH3Font)}
                    onValueChange={(value) => handleChange('mdH3Font', normalizeHeadingFontValue(value))}
                  >
                    <SelectTrigger className="w-full md:w-64">
                      <SelectValue placeholder="Default (Theme)" />
                    </SelectTrigger>
                    <SelectContent>
                      {headingFontOptions.map((option) => (
                        <SelectItem key={option.label} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleChange('mdH3Color', '')}
                  >
                    Reset Color
                  </Button>
                </div>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Streaming"
              description="Control how responses are displayed as they arrive."
            >
              <SettingRow label="Stream Responses" htmlFor="stream-responses">
                <Switch
                  id="stream-responses"
                  checked={localSettings.streamResponses}
                  onCheckedChange={(value) => handleChange('streamResponses', value)}
                />
              </SettingRow>
            </SettingsSection>

          </div>
        </TabsContent>

        {/* User Profiles */}
        <TabsContent value="profiles">
          <div className="space-y-6">
            <SettingsSection
              title="User Profiles"
              description="Manage your user identities and preferences."
            >
              <div className="space-y-6">
                <ProfileSelector />
                <div className="border-t border-border/60" />
                <SimpleUserProfileEditor />
              </div>
            </SettingsSection>
          </div>
        </TabsContent>

        <TabsContent value="rag">
          <RAGSettings />
        </TabsContent>

        <TabsContent value="nano-gpt-memory">
          <div className="max-w-3xl space-y-4">
            <NanoGptMemorySettings settings={contextSettings} updateSettings={updateSettings} />
          </div>
        </TabsContent>

        {/* Generation - RESTORED FULL CONTENT */}
        <TabsContent value="generation">
          <div className="space-y-6">
            <SettingsSection
              title="Sampling"
              description="Core sampling controls for text generation."
            >
              <SettingRow label={`Temperature (${localSettings.temperature.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.temperature]}
                  min={0}
                  max={2}
                  step={0.05}
                  onValueChange={([v]) => handleChange('temperature', v)}
                />
              </SettingRow>
              <SettingRow label={`Top-P (${localSettings.top_p.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.top_p]}
                  min={0}
                  max={1}
                  step={0.05}
                  onValueChange={([v]) => handleChange('top_p', v)}
                />
              </SettingRow>
              <SettingRow label={`Top-K (${localSettings.top_k})`} layout="stack">
                <Slider
                  value={[localSettings.top_k]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={([v]) => handleChange('top_k', v)}
                />
              </SettingRow>
              <SettingRow label={`Repetition Penalty (${localSettings.repetition_penalty.toFixed(2)})`} layout="stack">
                <Slider
                  value={[localSettings.repetition_penalty]}
                  min={1}
                  max={2}
                  step={0.01}
                  onValueChange={([v]) => handleChange('repetition_penalty', v)}
                />
              </SettingRow>
            </SettingsSection>

            <SettingsAccordion
              title="Output behavior"
              summary="Anti-repetition and output length limits."
            >

            <SettingsSection
              title="Anti-Repetition"
              description="Reduce loops and repeated phrases."
            >
              <SettingRow label="Anti-Repetition Mode" htmlFor="anti-repetition" description="Enable extra controls to reduce repetition.">
                <Switch
                  id="anti-repetition"
                  checked={localSettings.antiRepetitionMode}
                  onCheckedChange={(checked) => handleChange('antiRepetitionMode', checked)}
                />
              </SettingRow>

              {localSettings.antiRepetitionMode && (
                <>
                  <SettingRow label={`Frequency Penalty (${localSettings.frequencyPenalty.toFixed(2)})`} layout="stack">
                    <Slider
                      value={[localSettings.frequencyPenalty]}
                      min={0}
                      max={2}
                      step={0.1}
                      onValueChange={([v]) => handleChange('frequencyPenalty', v)}
                    />
                  </SettingRow>
                  <SettingRow label={`Presence Penalty (${localSettings.presencePenalty.toFixed(2)})`} layout="stack">
                    <Slider
                      value={[localSettings.presencePenalty]}
                      min={0}
                      max={2}
                      step={0.1}
                      onValueChange={([v]) => handleChange('presencePenalty', v)}
                    />
                  </SettingRow>
                  <SettingRow label="Detect and Remove Repeated Phrases" htmlFor="detect-phrases">
                    <Switch
                      id="detect-phrases"
                      checked={localSettings.detectRepeatedPhrases}
                      onCheckedChange={(checked) => handleChange('detectRepeatedPhrases', checked)}
                    />
                  </SettingRow>
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Limits and Streaming"
              description="Control output size and streaming behavior."
            >
              <SettingRow label="Max Tokens" htmlFor="max_tokens" description="Use -1 to allow auto selection.">
                <Input
                  id="max_tokens"
                  type="number"
                  min={-1}
                  step={1}
                  value={localSettings.max_tokens}
                  onChange={e => handleChange('max_tokens', parseInt(e.target.value, 10))}
                  className="w-full md:max-w-xs"
                />
              </SettingRow>
            </SettingsSection>
            </SettingsAccordion>

            <SettingsAccordion
              title="Automation / Call Mode"
              summary={automationSummary}
            >

            <SettingsSection
              title="Profile and API Behavior"
              description="Advanced behavior and API compatibility."
            >
              <SettingRow label="Direct Profile Injection" htmlFor="direct-profile-injection">
                <Switch
                  id="direct-profile-injection"
                  checked={localSettings.directProfileInjection || false}
                  onCheckedChange={(value) => {
                    handleChange('directProfileInjection', value);
                    updateSettings({ directProfileInjection: value });
                    fetch(`${PRIMARY_API_URL}/models/set-direct-profile-injection`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ directProfileInjection: value })
                    }).catch(err => console.error(err));
                  }}
                />
              </SettingRow>
              <SettingRow label="Use OpenAI API Format" htmlFor="use-openai-api">
                <Switch
                  id="use-openai-api"
                  checked={localSettings.useOpenAIAPI || false}
                  onCheckedChange={(value) => {
                    handleChange('useOpenAIAPI', value);
                    fetch(`${PRIMARY_API_URL}/models/set-openai-api-mode`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ useOpenAIAPI: value })
                    }).catch(err => console.error(err));
                  }}
                />
              </SettingRow>
              {localSettings.useOpenAIAPI && (
                <Alert>
                  <AlertTitle>OpenAI API Mode Active</AlertTitle>
                  <AlertDescription className="text-xs">
                    Requests will use the OpenAI compatible format.
                  </AlertDescription>
                </Alert>
              )}
              <SettingRow
                label="API rolling conversation memory"
                htmlFor="api-rolling-memory"
                description="When the primary model is an API/subscription route, older turns are summarized into a structured memory pack so each send does not replay the full chat (saves tokens). Recent dialogue still goes verbatim up to the budget below."
              >
                <Switch
                  id="api-rolling-memory"
                  checked={localSettings.apiRollingMemoryEnabled !== false}
                  onCheckedChange={(checked) => handleChange('apiRollingMemoryEnabled', checked)}
                />
              </SettingRow>
              <SettingRow
                label="Reasoning diagnostics (UI)"
                htmlFor="reasoning-diagnostics"
                description="Show a compact reasoning status line on assistant messages for debugging streaming and parser behavior."
              >
                <Switch
                  id="reasoning-diagnostics"
                  checked={localSettings.showReasoningDiagnostics || false}
                  onCheckedChange={(checked) => handleChange('showReasoningDiagnostics', checked)}
                />
              </SettingRow>
              <SettingRow
                label={`API context window budget (${formatApiContextWindowShort(localSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT)})`}
                htmlFor="api-context-window-budget"
                description="Total target budget for each API request context pack (system + rolling memory + history). Increase for long-context models; slider goes up to about 1M tokens."
                layout="stack"
              >
                <Slider
                  value={[clampApiContextWindowTokens(localSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT)]}
                  min={API_CONTEXT_WINDOW_MIN}
                  max={API_CONTEXT_WINDOW_MAX}
                  step={API_CONTEXT_WINDOW_SLIDER_STEP}
                  onValueChange={([v]) => {
                    const nextWindow = clampApiContextWindowTokens(Number(v) || API_CONTEXT_WINDOW_TOKENS_DEFAULT);
                    const maxRecent = Math.max(2048, nextWindow - 4096);
                    const currentRecent = Number(localSettings.apiRecentVerbatimTokenBudget) || 32000;
                    handleChange('apiContextWindowTokens', nextWindow);
                    if (currentRecent > maxRecent) {
                      handleChange('apiRecentVerbatimTokenBudget', maxRecent);
                    }
                  }}
                />
              </SettingRow>
              <SettingRow
                label="Recent dialogue token budget"
                htmlFor="api-recent-verbatim-tokens"
                description="Approximate cap on verbatim user/assistant history per request (excluding system). Larger = more continuity per send, higher API cost. Rolling memory holds the rest."
              >
                <Input
                  id="api-recent-verbatim-tokens"
                  type="number"
                  min={2048}
                  max={Math.max(2048, clampApiContextWindowTokens(localSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT) - 4096)}
                  step={512}
                  value={localSettings.apiRecentVerbatimTokenBudget ?? 32000}
                  onChange={(e) =>
                    handleChange(
                      'apiRecentVerbatimTokenBudget',
                      Math.min(
                        Math.max(2048, clampApiContextWindowTokens(localSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT) - 4096),
                        Math.max(2048, parseInt(e.target.value, 10) || 32000)
                      )
                    )
                  }
                  className="w-full md:max-w-xs"
                />
              </SettingRow>

              <SettingRow
                label="Book run"
                htmlFor="book-run-hint"
                description="Packing budgets, refusal retries, first-chapter preamble, and quick-prompt buttons are edited in the Book run overlay (not here)."
                layout="stack"
              >
                <p id="book-run-hint" className="text-sm text-muted-foreground">
                  Chat toolbar → <span className="font-medium text-foreground">Book run</span> → tab{' '}
                  <span className="font-medium text-foreground">Run settings</span>.
                </p>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="Call mode avatar"
              description="Portrait call layout: optional full-screen character window, framing zoom, and per-character folder imports (Character editor → Avatars)."
            >
              <SettingRow
                label="Fullscreen avatar (default)"
                htmlFor="call-mode-fullscreen-avatar"
                description="When enabled, entering call mode fills the screen with the portrait. Toggle anytime in call via the expand button or F."
              >
                <Switch
                  id="call-mode-fullscreen-avatar"
                  checked={localSettings.callModeFullscreenAvatar === true}
                  onCheckedChange={(value) => handleChange('callModeFullscreenAvatar', value)}
                />
              </SettingRow>
              <SettingRow
                label={`Default fullscreen zoom (${Math.round((localSettings.callModeFullscreenZoom ?? 1) * 100)}%)`}
                htmlFor="call-mode-fullscreen-zoom"
                description="Framing scale for fullscreen mode. Adjust live in call with the zoom slider."
                layout="stack"
              >
                <Slider
                  id="call-mode-fullscreen-zoom"
                  value={[Math.round((localSettings.callModeFullscreenZoom ?? 1) * 100)]}
                  min={100}
                  max={280}
                  step={5}
                  onValueChange={([v]) => handleChange('callModeFullscreenZoom', Math.max(1, Math.min(2.8, (Number(v) || 100) / 100)))}
                />
              </SettingRow>
              <SettingRow
                label="Avatar folder"
                description="Import many looks at once: Character editor → Avatars → Import avatar folder (browser folder picker; files upload to the server)."
                layout="stack"
              >
                <p className="text-sm text-muted-foreground">
                  Keeps unlimited images/videos per character alongside the usual {10} single-file uploads.
                </p>
              </SettingRow>
            </SettingsSection>

            <SettingsSection
              title="D-ID pipeline (Watch overlay)"
              description="One-tap preset prompts for the D-ID batch overlay (label + text). If the text is a URL, the button sets the avatar URL."
            >
              <div className="space-y-2">
                <div className="space-y-2">
                  <Label className="text-xs">D-ID quick buttons (label + text or avatar URL)</Label>
                  {(localSettings.didQuickPromptButtons || []).map((row, idx) => (
                    <div key={row.id || idx} className="flex flex-col sm:flex-row gap-2 items-start border rounded p-2 bg-background/60">
                      <Input
                        placeholder="Button label"
                        className="text-sm sm:w-40"
                        value={row.label || ''}
                        onChange={(e) => {
                          const next = [...(localSettings.didQuickPromptButtons || [])];
                          next[idx] = { ...row, label: e.target.value };
                          handleChange('didQuickPromptButtons', next);
                        }}
                      />
                      <Textarea
                        placeholder="Prompt log line or https://… avatar URL"
                        className="text-sm flex-1 min-h-[56px]"
                        value={row.text || ''}
                        onChange={(e) => {
                          const next = [...(localSettings.didQuickPromptButtons || [])];
                          next[idx] = { ...row, text: e.target.value };
                          handleChange('didQuickPromptButtons', next);
                        }}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="shrink-0"
                        onClick={() => {
                          const next = (localSettings.didQuickPromptButtons || []).filter((_, i) => i !== idx);
                          handleChange('didQuickPromptButtons', next);
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const next = [...(localSettings.didQuickPromptButtons || [])];
                      next.push({
                        id: `didqb_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
                        label: '',
                        text: '',
                      });
                      handleChange('didQuickPromptButtons', next);
                    }}
                  >
                    Add D-ID quick button
                  </Button>
                </div>
              </div>
            </SettingsSection>

            <SettingsSection
              title="Scheduled Character Outreach"
              description="Run a character prompt on a custom minute interval in a dedicated outreach chat queue."
              actions={(
                <Button size="sm" variant="ghost" onClick={clearOutreachNotifications}>
                  Clear Alerts
                </Button>
              )}
            >
              <SettingRow
                label="Enable scheduled outreach"
                description="When off, scheduled outreach rules will not run and no outreach notifications will be sent (rules stay saved)."
                htmlFor="outreach-enabled"
              >
                <Switch
                  id="outreach-enabled"
                  checked={contextSettings.outreachEnabled !== false}
                  onCheckedChange={(on) => {
                    updateSettings({ outreachEnabled: on });
                  }}
                />
              </SettingRow>
              <SettingRow
                label="Browser push for outreach"
                description="Sends a system notification when a scheduled rule finishes (saved with settings). Turn on to request permission; if you deny, the switch turns off."
                htmlFor="outreach-browser-notifications"
              >
                <div className="flex flex-col gap-2 sm:items-end">
                  <Switch
                    id="outreach-browser-notifications"
                    checked={contextSettings.outreachBrowserNotifications === true}
                    onCheckedChange={async (on) => {
                      if (!on) {
                        updateSettings({ outreachBrowserNotifications: false });
                        return;
                      }
                      updateSettings({ outreachBrowserNotifications: true });
                      const perm = await requestOutreachNotificationPermission();
                      if (perm !== 'granted') {
                        updateSettings({ outreachBrowserNotifications: false });
                      }
                    }}
                  />
                  {typeof window !== 'undefined' && 'Notification' in window ? (
                    <p className="text-[11px] text-muted-foreground text-right max-w-xs">
                      Permission:{' '}
                      <span className="font-medium text-foreground">
                        {Notification.permission === 'granted' ? 'allowed' : Notification.permission === 'denied' ? 'blocked' : 'not asked yet'}
                      </span>
                    </p>
                  ) : null}
                </div>
              </SettingRow>
              <SettingRow label="Rule Name">
                <Input
                  value={outreachDraft.name}
                  onChange={(e) => setOutreachDraft(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Weather Updates"
                />
              </SettingRow>
              <SettingRow label="Character">
                <Select
                  value={outreachDraft.characterId || ''}
                  onValueChange={(value) => setOutreachDraft(prev => ({ ...prev, characterId: value }))}
                >
                  <SelectTrigger><SelectValue placeholder="Select character" /></SelectTrigger>
                  <SelectContent>
                    {outreachCharacters.map((character) => (
                      <SelectItem key={character.id} value={character.id}>{character.name || character.id}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingRow>
              <SettingRow label="Model">
                <Select
                  value={outreachDraft.modelName || ''}
                  onValueChange={(value) => setOutreachDraft(prev => ({ ...prev, modelName: value }))}
                >
                  <SelectTrigger><SelectValue placeholder={primaryModel || 'Select model'} /></SelectTrigger>
                  <SelectContent>
                    {(modelOptions.length ? modelOptions : [{ name: primaryModel || 'default' }]).map((model) => (
                      <SelectItem key={model.name} value={model.name}>{model.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingRow>
              <SettingRow label="Prompt" layout="stack">
                <Textarea
                  rows={3}
                  value={outreachDraft.prompt}
                  onChange={(e) => setOutreachDraft(prev => ({ ...prev, prompt: e.target.value }))}
                  placeholder="Update me on current Melbourne weather."
                />
              </SettingRow>
              <SettingRow
                label="Interval (minutes)"
                description="How often this rule should send and notify."
              >
                <Input
                  type="number"
                  min={1}
                  step={1}
                  value={outreachDraft.intervalMinutes}
                  onChange={(e) => setOutreachDraft(prev => ({ ...prev, intervalMinutes: e.target.value }))}
                  className="w-full md:max-w-xs"
                />
              </SettingRow>
              <SettingRow
                label="Random image folder"
                description="Optional. Each notification can include one random image from this folder (uploaded to the server for scheduled runs)."
                layout="stack"
              >
                <div className="flex flex-col gap-2 w-full">
                  <input
                    ref={outreachImageInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleOutreachDraftFolderSelect}
                    {...{ webkitdirectory: '', directory: '' }}
                  />
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={outreachImageUploading}
                      onClick={() => outreachImageInputRef.current?.click()}
                    >
                      {outreachDraft.pendingImageFiles?.length
                        ? `Change folder (${outreachDraft.pendingImageFiles.length} images)`
                        : 'Choose image folder'}
                    </Button>
                    {outreachDraft.pendingImageLabel ? (
                      <span className="text-xs text-muted-foreground self-center truncate max-w-[240px]">
                        {outreachDraft.pendingImageLabel}
                      </span>
                    ) : null}
                    {outreachDraft.pendingImageFiles?.length ? (
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={() => setOutreachDraft((prev) => ({
                          ...prev,
                          pendingImageFiles: null,
                          pendingImageLabel: '',
                        }))}
                      >
                        Clear
                      </Button>
                    ) : null}
                  </div>
                </div>
              </SettingRow>
              <input
                ref={outreachRuleImageInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleOutreachRuleFolderSelect}
                {...{ webkitdirectory: '', directory: '' }}
              />
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => setOutreachDraft({
                    name: '',
                    characterId: '',
                    prompt: '',
                    modelName: '',
                    intervalMinutes: 45,
                    pendingImageFiles: null,
                    pendingImageLabel: '',
                  })}
                >
                  Clear form
                </Button>
                <Button type="button" size="sm" onClick={handleSaveOutreachRule}>Save Outreach Rule</Button>
              </div>

              <div className="space-y-2 pt-2 border-t border-border/60 mt-4">
                <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                  Saved rules ({outreachRules.length})
                </p>
                {outreachRules.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No outreach rules yet. Fill the form above and click Save.</p>
                ) : outreachRules.map((rule) => (
                  <div key={rule.id} className="rounded-lg border border-border/60 bg-muted/20 p-3 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-medium">{rule.name || 'Scheduled Outreach'}</div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-full border ${rule.enabled ? 'border-emerald-500/50 text-emerald-600 dark:text-emerald-400' : 'border-border text-muted-foreground'}`}>
                          {rule.enabled ? 'On' : 'Off'}
                        </span>
                        <Switch
                          checked={rule.enabled === true}
                          onCheckedChange={(enabled) => upsertOutreachRule({ ...rule, enabled })}
                        />
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">{rule.prompt}</p>
                    <p className="text-xs text-muted-foreground">
                      Every {rule.intervalMinutes || 45} minute(s) • model: {rule.modelName || primaryModel || 'default'}
                      {rule.imageCount > 0 ? ` • ${rule.imageCount} random image(s)` : ''}
                    </p>
                    {rule.nextRunAt && (
                      <p className="text-[11px] text-muted-foreground">
                        Next run (approx.): {new Date(rule.nextRunAt).toLocaleString()}
                      </p>
                    )}
                    <div className="flex flex-wrap gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={outreachImageUploading}
                        onClick={() => {
                          setOutreachImageUploadRuleId(rule.id);
                          window.setTimeout(() => outreachRuleImageInputRef.current?.click(), 0);
                        }}
                      >
                        {rule.imageCount > 0 ? 'Replace image folder' : 'Attach image folder'}
                      </Button>
                      {rule.imageCount > 0 ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={outreachImageUploading}
                          onClick={() => clearOutreachRuleImages(rule.id)}
                        >
                          Remove images
                        </Button>
                      ) : null}
                      <Button size="sm" variant="outline" onClick={() => runOutreachRuleNow(rule.id)}>Run Now</Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          const current = Number.parseInt(rule.intervalMinutes, 10) || 45;
                          const next = window.prompt('Set interval in minutes', String(current));
                          if (next == null) return;
                          const parsed = Number.parseInt(next, 10);
                          if (!Number.isFinite(parsed) || parsed < 1) {
                            alert('Interval must be at least 1 minute.');
                            return;
                          }
                          upsertOutreachRule({ ...rule, intervalMinutes: parsed });
                        }}
                      >
                        Edit Interval
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => deleteOutreachRule(rule.id)}>Delete</Button>
                    </div>
                  </div>
                ))}
              </div>

              {outreachNotifications?.length > 0 && (
                <div className="space-y-2 pt-2">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Recent Outreach Notifications</p>
                  {outreachNotifications.slice(0, 6).map((note) => {
                    const label = note.characterName || note.title || 'Character';
                    return (
                      <div key={note.id} className="flex gap-3 rounded-lg border border-border/60 bg-background/40 p-3">
                        {note.characterAvatar ? (
                          <img src={note.characterAvatar} alt="" className="h-10 w-10 shrink-0 rounded-full object-cover border border-border" />
                        ) : (
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-border bg-muted text-xs font-semibold">
                            {String(label).charAt(0).toUpperCase()}
                          </div>
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium">{label}</div>
                          <div className="text-[11px] text-muted-foreground">Eloquent</div>
                          <div className="text-xs text-muted-foreground mt-0.5 line-clamp-2">{note.preview}</div>
                          {note.attachmentImageUrl ? (
                            <img
                              src={note.attachmentImageUrl}
                              alt=""
                              className="mt-2 max-h-20 rounded-md object-cover border border-border/60"
                            />
                          ) : null}
                          <div className="mt-2 flex flex-wrap gap-2">
                            <Button size="sm" variant="outline" onClick={() => openOutreachNotification(note)}>
                              Open in chat
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => discardOutreachNotification(note)}
                            >
                              Dismiss
                            </Button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </SettingsSection>

            <SettingsSection
              title="Web Search"
              description="Choose how chat web search runs: provider-native (Perplexity-style) or Eloquent prefetch for local models."
            >
              <SettingRow
                label="Search mode"
                htmlFor="web-search-strategy"
                description="Auto uses native search on supported API endpoints (OpenRouter, Perplexity, nano-gpt); local GGUF uses Eloquent prefetch."
              >
                <select
                  id="web-search-strategy"
                  className="flex h-9 w-full max-w-xs rounded-md border border-input bg-background px-3 text-sm"
                  value={localSettings.webSearchStrategy || 'auto'}
                  onChange={(e) => handleChange('webSearchStrategy', e.target.value)}
                >
                  <option value="auto">Auto (prefer native)</option>
                  <option value="eloquent">Always Eloquent prefetch</option>
                  <option value="native">Native only</option>
                  <option value="off">Off (globe still toggles per chat)</option>
                </select>
              </SettingRow>
            </SettingsSection>
            </SettingsAccordion>

            <SettingsAccordion
              title="Custom API endpoints"
              summary={endpointSummary}
            >
              <SettingsSection
                title="Custom API Endpoints"
                description="Add external API targets for model selection. Pick models from the NanoGPT catalog or manage endpoints below."
                actions={(
                  <div className="flex flex-wrap items-center gap-2">
                    <NanoGptModelSelectorPopover
                      currentModelId={primaryModel}
                      primaryApiUrl={PRIMARY_API_URL}
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        const newEndpoints = [...(localSettings.customApiEndpoints || []), {
                          id: `endpoint-${Date.now()}`,
                          name: 'New Endpoint',
                          url: getBackendUrl(),
                          apiKey: '',
                          enabled: true,
                          rotate_enabled: true,
                          context_window: null,
                          supports_native_search: null,
                        }];
                        handleChange('customApiEndpoints', newEndpoints);
                      }}
                    >
                      Add Endpoint
                    </Button>
                  </div>
                )}
              >
              <SettingRow
                label="⟳ Auto-routing"
                htmlFor="api-endpoint-round-robin"
                description="Rotates across endpoints with enabled + include in rotation (⟳) each prompt. Navbar shows ⟳ Auto when active."
              >
                <Switch
                  id="api-endpoint-round-robin"
                  checked={localSettings.apiEndpointRoundRobinEnabled === true}
                  onCheckedChange={(enabled) => handleChange('apiEndpointRoundRobinEnabled', enabled)}
                />
              </SettingRow>

              {localSettings.apiEndpointRoundRobinEnabled === true && (() => {
                const pool = getRotationPool(localSettings);
                const catalog = readNanoGptModelsCache().models;
                if (pool.length < 2) {
                  return (
                    <p className="text-xs text-muted-foreground px-1">
                      Enable at least two endpoints with ⟳ rotation to use auto-routing.
                    </p>
                  );
                }
                const cursor = (localSettings.apiEndpointRoundRobinCursor || {}).__manual_rotation__ ?? 0;
                return (
                  <p className="text-xs text-muted-foreground px-1">
                    Rotation pool ({pool.length}):
                    {' '}
                    {pool.map((ep, i) => {
                      const d = resolveEndpointDisplay(ep.id, localSettings, catalog);
                      const atCursor = Number(cursor) % pool.length === i;
                      return (
                        <span key={ep.id} className={atCursor ? 'text-primary font-medium' : ''}>
                          {d?.icon}
                          {' '}
                          {d?.displayName || ep.name}
                          {atCursor ? ' ◀' : ''}
                          {i < pool.length - 1 ? ' · ' : ''}
                        </span>
                      );
                    })}
                  </p>
                );
              })()}

              {(localSettings.customApiEndpoints || []).map((endpoint, index) => {
                const catalog = readNanoGptModelsCache().models;
                const resolved = resolveEndpointDisplay(endpoint.id, localSettings, catalog);
                return (
                <div key={endpoint.id} className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-3">
                  <div className="flex flex-row items-center justify-between gap-2">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Switch
                        checked={endpoint.enabled}
                        onCheckedChange={(enabled) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, enabled };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                      <span className="text-lg flex-shrink-0" title={resolved?.provider}>{resolved?.icon || '⬜'}</span>
                      <div className="min-w-0 flex-1">
                        <Input
                          placeholder="Endpoint Name"
                          value={endpoint.name}
                          onChange={(e) => {
                            const updated = [...localSettings.customApiEndpoints];
                            updated[index] = { ...endpoint, name: e.target.value };
                            handleChange('customApiEndpoints', updated);
                          }}
                          className="w-full"
                        />
                        {resolved?.displayName && (
                          <p className="text-[11px] text-muted-foreground mt-1 truncate">
                            Model: {resolved.displayName}
                            {resolved.modelId ? ` · ${resolved.modelId}` : ''}
                          </p>
                        )}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        const updated = localSettings.customApiEndpoints.filter((_, i) => i !== index);
                        handleChange('customApiEndpoints', updated);
                      }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  <div className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-3 py-2">
                    <div className="text-xs text-muted-foreground">Include in rotation (⟳ Auto-routing)</div>
                    <Switch
                      checked={endpoint.rotate_enabled !== false}
                      onCheckedChange={(rotate_enabled) => {
                        const updated = [...localSettings.customApiEndpoints];
                        updated[index] = { ...endpoint, rotate_enabled };
                        handleChange('customApiEndpoints', updated);
                      }}
                    />
                  </div>

                  <div className="flex items-center justify-between rounded-md border border-border/50 bg-background/40 px-3 py-2">
                    <div className="text-xs text-muted-foreground max-w-[70%]">
                      Supports native web search (OpenRouter / Perplexity / :online). Leave off for auto-detect.
                    </div>
                    <select
                      className="text-xs rounded border bg-background px-2 py-1 max-w-[8rem]"
                      value={
                        endpoint.supports_native_search === true
                          ? 'yes'
                          : endpoint.supports_native_search === false
                            ? 'no'
                            : 'auto'
                      }
                      onChange={(e) => {
                        const v = e.target.value;
                        const supports_native_search =
                          v === 'yes' ? true : v === 'no' ? false : null;
                        const updated = [...localSettings.customApiEndpoints];
                        updated[index] = { ...endpoint, supports_native_search };
                        handleChange('customApiEndpoints', updated);
                      }}
                    >
                      <option value="auto">Auto</option>
                      <option value="yes">Yes</option>
                      <option value="no">No</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div>
                      <Label className="text-xs">API URL</Label>
                      <Input
                        value={endpoint.url}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, url: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Model Name</Label>
                      <Input
                        value={endpoint.model || ''}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, model: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">API Key</Label>
                      <Input
                        type="password"
                        value={endpoint.apiKey}
                        onChange={(e) => {
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, apiKey: e.target.value };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Context Window</Label>
                      <Input
                        type="number"
                        min="1024"
                        step="256"
                        placeholder="8192"
                        value={endpoint.context_window ?? ''}
                        onChange={(e) => {
                          const value = e.target.value;
                          const contextWindow = value === '' ? null : parseInt(value, 10);
                          const updated = [...localSettings.customApiEndpoints];
                          updated[index] = { ...endpoint, context_window: contextWindow };
                          handleChange('customApiEndpoints', updated);
                        }}
                      />
                    </div>
                  </div>
                </div>
                );
              })}

              <div className="text-xs text-muted-foreground">
                Changes are saved automatically. Model catalog caches to localStorage (
                <code className="text-[10px]">nanoGpt-models-cache-v1</code>
                , ~1h TTL). Browser fetch to nano-gpt.com may require CORS; stale cache is used on failure.
              </div>
            </SettingsSection>
            </SettingsAccordion>

          </div>
        </TabsContent>

        {/* Image Generation */}
        <TabsContent value="image-generation">
          <div className="space-y-6">
            <SettingsSection
              title="Engine Priority"
              description="Select which image engine is preferred."
            >
              <SettingRow
                label="Image Engine Priority"
                layout="stack"
                description="Local SD uses the built-in stable-diffusion.cpp engine. External engines require their own servers."
              >
                <Select
                  value={localSettings.imageEngine || 'EloDiffusion'}
                  onValueChange={(value) => handleChange('imageEngine', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select image engine" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="EloDiffusion">Local SD (Built-in)</SelectItem>
                    <SelectItem value="comfyui">ComfyUI (External)</SelectItem>
                    <SelectItem value="nanogpt">NanoGPT (Cloud)</SelectItem>
                  </SelectContent>
                </Select>
              </SettingRow>
            </SettingsSection>

            {localSettings.imageEngine === 'nanogpt' && (
              <SettingsSection
                title="NanoGPT (Cloud)"
                description="Generate images and videos using NanoGPT model APIs."
              >
                <SettingRow label="NanoGPT API Key (Shared)" htmlFor="nanogpt-api-key">
                  <Input
                    id="nanogpt-api-key"
                    type="password"
                    value={localSettings.nanoGptApiKey || ''}
                    onChange={(e) => handleChange('nanoGptApiKey', e.target.value)}
                    placeholder="sk-..."
                  />
                </SettingRow>
                <SettingRow label="Image Model Name" htmlFor="nanogpt-model">
                  <Input
                    id="nanogpt-model"
                    value={localSettings.nanoGptModel || 'dall-e-3'}
                    onChange={(e) => handleChange('nanoGptModel', e.target.value)}
                    placeholder="dall-e-3"
                  />
                </SettingRow>
                <SettingRow
                  label="Video Model Name"
                  htmlFor="nanogpt-video-model"
                  description="Default: svd (stable-video-diffusion)."
                >
                  <Input
                    id="nanogpt-video-model"
                    value={localSettings.nanoGptVideoModel || 'svd'}
                    onChange={(e) => handleChange('nanoGptVideoModel', e.target.value)}
                    placeholder="svd"
                  />
                </SettingRow>
              </SettingsSection>
            )}

            <SettingsSection
              title="Local SD (Built-in)"
              description="Built-in image generation using stable-diffusion.cpp."
            >
              <div className="rounded-md border border-border/60 bg-muted/20 p-3">
                <div className="text-xs text-muted-foreground">
                  <div className="font-medium text-foreground">Supported local models</div>
                  <ul className="mt-2 list-disc pl-5 space-y-1">
                    <li>Files: .safetensors, .ckpt, .gguf</li>
                    <li>Families: SD 1.x, SDXL (filename contains sdxl/xl), FLUX (filename contains flux)</li>
                    <li>FLUX needs extra files in the same folder: clip_l.safetensors, t5xxl_fp16.safetensors, ae.safetensors</li>
                    <li>SDXL is heavier; reduce resolution or steps if you hit VRAM limits</li>
                  </ul>
                </div>
              </div>

              <SettingRow label="Local SD Models Directory" htmlFor="sd-model-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="sd-model-directory"
                    value={localSettings.sdModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('sdModelDirectory', e.target.value)}
                    placeholder="C:\path\to\sd-models"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('sdModelDirectory', 'Select Local SD Models Directory')}
                    disabled={directoryPickerKey === 'sdModelDirectory'}
                  >
                    {directoryPickerKey === 'sdModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.sdModelDirectory) {
                        queueSettingsSave({ sdModelDirectory: localSettings.sdModelDirectory });
                        fetch(`${PRIMARY_API_URL}/sd-local/refresh-directory`, {
                          method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.sdModelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated!' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label="ADetailer Models Directory" htmlFor="adetailer-model-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="adetailer-model-directory"
                    value={localSettings.adetailerModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('adetailerModelDirectory', e.target.value)}
                    placeholder="C:\path\to\adetailer-models"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('adetailerModelDirectory', 'Select ADetailer Models Directory')}
                    disabled={directoryPickerKey === 'adetailerModelDirectory'}
                  >
                    {directoryPickerKey === 'adetailerModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (localSettings.adetailerModelDirectory) {
                        queueSettingsSave({ adetailerModelDirectory: localSettings.adetailerModelDirectory });
                        fetch(`${PRIMARY_API_URL}/sd-local/set-adetailer-directory`, {
                          method: 'POST', headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ directory: localSettings.adetailerModelDirectory })
                        }).then(r => r.json()).then(d => alert(d.status === 'success' ? 'Updated!' : d.message));
                      }
                    }}
                  >
                    Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label="Upscaler Models Directory" htmlFor="upscaler-directory">
                <div className="flex w-full md:w-auto items-center gap-2">
                  <Input
                    id="upscaler-directory"
                    value={localSettings.upscalerModelDirectory || ''}
                    className="flex-1 md:w-64"
                    onChange={(e) => handleChange('upscalerModelDirectory', e.target.value)}
                    placeholder="C:\path\to\upscalers"
                  />
                  <Button
                    variant="outline"
                    onClick={() => handleDirectoryBrowse('upscalerModelDirectory', 'Select Upscaler Models Directory')}
                    disabled={directoryPickerKey === 'upscalerModelDirectory'}
                  >
                    {directoryPickerKey === 'upscalerModelDirectory' ? (
                      <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                    ) : (
                      <FolderOpen className="mr-1 h-4 w-4" />
                    )}
                    Browse
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      queueSettingsSave({ upscalerModelDirectory: localSettings.upscalerModelDirectory });
                      fetch(`${PRIMARY_API_URL}/models/update-upscaler-dir`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ directory: localSettings.upscalerModelDirectory })
                      }).then(r => r.json()).then(d => alert('Updated!'));
                    }}
                  >
                    <Save className="mr-1 h-4 w-4" />Save
                  </Button>
                </div>
              </SettingRow>

              <SettingRow label={`Default Steps (${localSettings.sdSteps || 20})`} layout="stack">
                <Slider
                  min={10}
                  max={50}
                  step={1}
                  value={[localSettings.sdSteps || 20]}
                  onValueChange={([v]) => handleChange('sdSteps', v)}
                />
              </SettingRow>
              <SettingRow label={`Default CFG Scale (${(localSettings.sdCfgScale || 7.0).toFixed(1)})`} layout="stack">
                <Slider
                  min={1.0}
                  max={20.0}
                  step={0.5}
                  value={[localSettings.sdCfgScale || 7.0]}
                  onValueChange={([v]) => handleChange('sdCfgScale', v)}
                />
              </SettingRow>
            </SettingsSection>

            <div className="flex justify-end pt-2">
              <Button variant="outline" onClick={handleReset} className="w-full md:w-auto">
                Reset
              </Button>
            </div>
          </div>
        </TabsContent>

        {/* Characters */}
        <TabsContent value="characters" className="w-full max-w-none">
          <CharacterEditor />
        </TabsContent>

        {/* Audio - RESTORED FULL CONTENT */}
        <TabsContent value="audio">
          <div className="space-y-6">
            <SettingsSection
              title="Speech to Text"
              description="Configure speech recognition and engine tools."
            >
              <SettingRow label="Enable Speech-to-Text" htmlFor="stt-enabled">
                <Switch
                  id="stt-enabled"
                  checked={sttEnabled}
                  onCheckedChange={(value) => handleChange('sttEnabled', value)}
                />
              </SettingRow>

              {sttEnabled && (
                <>
                  <SettingRow
                    label="Auto-send speech on stop"
                    htmlFor="stt-auto-send-on-stop"
                    description="When enabled, stopping mic recording sends the transcript immediately (regular chat + focus mode + matching hotkeys/pedal)."
                  >
                    <Switch
                      id="stt-auto-send-on-stop"
                      checked={localSettings.sttAutoSendOnStop === true}
                      onCheckedChange={(value) => handleChange('sttAutoSendOnStop', value)}
                    />
                  </SettingRow>

                  <SettingRow label="Speech Recognition Engine" htmlFor="stt-engine" layout="stack">
                    <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                      <Select
                        id="stt-engine"
                        value={localSettings.sttEngine || 'whisper'}
                        onValueChange={async (value) => {
                          handleChange('sttEngine', value);
                          updateSettings({ sttEngine: value });
                        }}
                      >
                        <SelectTrigger className="w-full md:max-w-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="whisper">OpenAI Whisper</SelectItem>
                          <SelectItem value="whisper3">Whisper 3 Turbo</SelectItem>
                          <SelectItem value="parakeet">NVIDIA Parakeet v2 (English)</SelectItem>
                          <SelectItem value="parakeet-v3">NVIDIA Parakeet v3 (Multilingual)</SelectItem>
                          <SelectItem value="parakeet-zh">NVIDIA Parakeet (Chinese)</SelectItem>
                        </SelectContent>
                      </Select>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={fetchAvailableSTTEngines}
                        disabled={isInstallingEngine}
                        className="mt-2 md:mt-0"
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  </SettingRow>
                  <SettingsAccordion
                    title="STT maintenance"
                    summary="Engine install/fix actions and quick GPU tooling."
                  >
                    <SettingRow label="Engine Management" layout="stack">
                      <div className="flex flex-col md:flex-row gap-2 flex-wrap">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet&force=true`, { method: 'POST' });
                              alert('Parakeet (English) installed successfully!');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Force Install Parakeet (English)
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet-v3`, { method: 'POST' });
                              alert('Parakeet v3 (multilingual) installed successfully!');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Force Install Parakeet v3
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet-zh`, { method: 'POST' });
                              alert('Parakeet (Chinese) installed successfully!');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Force Install Parakeet (Chinese)
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              await fetch(`${PRIMARY_API_URL}/stt/fix-parakeet-numpy`, { method: 'POST' });
                              alert('Fixed! Restart Backend.');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Fix Dependencies
                        </Button>
                      </div>
                    </SettingRow>

                    <SettingRow label="Quick GPU Actions" layout="stack">
                      <div className="flex flex-col md:flex-row gap-2">
                        <Button variant="outline" size="sm" disabled title="Not implemented">Load Whisper on GPU1</Button>
                        <Button variant="outline" size="sm" disabled title="Not implemented">Load Kokoro on GPU1</Button>
                      </div>
                    </SettingRow>
                  </SettingsAccordion>
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Text to Speech"
              description="Configure voices, playback, and TTS engines."
            >
              <SettingRow label="Enable Text-to-Speech" htmlFor="tts-enabled">
                <Switch
                  id="tts-enabled"
                  checked={ttsEnabled}
                  onCheckedChange={(value) => handleChange('ttsEnabled', value)}
                />
              </SettingRow>

              {ttsEnabled && (
                <>
                  <SettingRow label="Text-to-Speech Engine" htmlFor="tts-engine">
                    <Select
                      id="tts-engine"
                      value={localSettings.ttsEngine || 'kokoro'}
                      onValueChange={value => {
                        handleChange('ttsEngine', value);
                        if (value === 'kokoro') {
                          handleChange('ttsVoice', 'af_heart');
                        }
                      }}
                    >
                      <SelectTrigger className="w-full md:w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="kokoro">Kokoro TTS</SelectItem>
                        <SelectItem value="chatterbox">Chatterbox (Faster)</SelectItem>
                        <SelectItem value="chatterbox_turbo">Chatterbox Turbo</SelectItem>
                        <SelectItem value="nanogpt-gpt-4o-mini-tts">NanoGPT (gpt-4o-mini-tts)</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  {(localSettings.ttsEngine || 'kokoro') === 'kokoro' && (
                    <SettingRow label="Kokoro Voice" htmlFor="tts-voice">
                      <Select
                        id="tts-voice"
                        value={localSettings.ttsVoice || 'af_heart'}
                        onValueChange={value => {
                          handleChange('ttsVoice', value);
                        }}
                      >
                        <SelectTrigger className="w-full md:w-64">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="max-h-64 overflow-y-auto">
                          <SelectItem value="af_heart">Am. English Female (Heart)</SelectItem>
                          <SelectItem value="af_alloy">Am. English Female (Alloy)</SelectItem>
                          <SelectItem value="af_aoede">Am. English Female (Aoede)</SelectItem>
                          <SelectItem value="af_bella">Am. English Female (Bella)</SelectItem>
                          <SelectItem value="af_jessica">Am. English Female (Jessica)</SelectItem>
                          <SelectItem value="af_kore">Am. English Female (Kore)</SelectItem>
                          <SelectItem value="af_nicole">Am. English Female (Nicole)</SelectItem>
                          <SelectItem value="af_nova">Am. English Female (Nova)</SelectItem>
                          <SelectItem value="af_river">Am. English Female (River)</SelectItem>
                          <SelectItem value="af_sarah">Am. English Female (Sarah)</SelectItem>
                          <SelectItem value="af_sky">Am. English Female (Sky)</SelectItem>
                          <SelectItem value="am_adam">Am. English Male (Adam)</SelectItem>
                          <SelectItem value="am_echo">Am. English Male (Echo)</SelectItem>
                        </SelectContent>
                      </Select>
                    </SettingRow>
                  )}

                  {localSettings.ttsEngine && localSettings.ttsEngine.startsWith('nanogpt-') && (
                    <SettingRow label="NanoGPT Voice" htmlFor="tts-voice">
                      <Select
                        id="tts-voice"
                        value={localSettings.ttsVoice || 'alloy'}
                        onValueChange={value => {
                          handleChange('ttsVoice', value);
                        }}
                      >
                        <SelectTrigger className="w-full md:w-64">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="max-h-64 overflow-y-auto">
                          <SelectItem value="alloy">Alloy</SelectItem>
                          <SelectItem value="echo">Echo</SelectItem>
                          <SelectItem value="fable">Fable</SelectItem>
                          <SelectItem value="onyx">Onyx</SelectItem>
                          <SelectItem value="nova">Nova</SelectItem>
                          <SelectItem value="shimmer">Shimmer</SelectItem>
                        </SelectContent>
                      </Select>
                    </SettingRow>
                  )}

                  {(localSettings.ttsEngine === 'chatterbox' || localSettings.ttsEngine === 'chatterbox_turbo') && (
                    <>
                      <SettingsAccordion
                        title="Engine maintenance tools"
                        summary="Voice upload, Voice Merge Lab, ffmpeg path, and VRAM actions."
                      >
                        <SettingRow label="Upload Voice Reference" htmlFor="voice-upload" layout="stack">
                          <Input
                            id="voice-upload"
                            type="file"
                            accept=".wav,.mp3,.flac,.m4a"
                            onChange={async (e) => {
                              const file = e.target.files?.[0];
                              if (!file) return;
                              try {
                                setIsUploadingVoice(true);
                                const formData = new FormData();
                                formData.append('file', file);
                                const response = await fetch(`${PRIMARY_API_URL}/tts/upload-voice`, {
                                  method: 'POST', body: formData,
                                });
                                const result = await response.json();
                                handleChange('ttsVoice', result.voice_id);
                                await fetchAvailableVoices();
                                alert(`Voice "${file.name}" uploaded successfully!`);
                              } catch (error) {
                                alert(`Failed to upload voice: ${error.message}`);
                              } finally {
                                setIsUploadingVoice(false);
                              }
                            }}
                            disabled={!ttsEnabled || isUploadingVoice}
                          />
                        </SettingRow>

                        <SettingRow
                          label="FFmpeg path"
                          htmlFor="ffmpeg-path"
                          description="Optional. Voice Merge, STT, and D-ID need ffmpeg.exe. Use this if a Python venv (e.g. after pip install rembg) hides system ffmpeg from PATH."
                        >
                          <Input
                            id="ffmpeg-path"
                            className="font-mono text-xs"
                            value={localSettings.ffmpegPath || ''}
                            onChange={(e) => handleChange('ffmpegPath', e.target.value.trim())}
                            placeholder="C:\ffmpeg\bin\ffmpeg.exe"
                            disabled={!ttsEnabled}
                          />
                        </SettingRow>

                        <VoiceSculptPanel
                          disabled={!ttsEnabled}
                          onVoiceReady={async (data) => {
                            if (data?.voice_id) {
                              handleChange('ttsVoice', data.voice_id);
                              await fetchAvailableVoices();
                            }
                          }}
                        />
                      </SettingsAccordion>

                      <SettingRow label="Active Voice" htmlFor="chatterbox-voice">
                        <Select
                          id="chatterbox-voice"
                          value={localSettings.ttsVoice || 'default'}
                          onValueChange={value => {
                            handleChange('ttsVoice', value);
                            fetch(`${PRIMARY_API_URL}/tts/save-voice-preference`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ voice_id: value, engine: localSettings.ttsEngine })
                            });
                          }}
                        >
                          <SelectTrigger className="w-full md:w-64">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="default">Default Voice</SelectItem>
                            {availableVoices?.chatterbox_voices?.map(voice => (
                              <SelectItem key={voice.id} value={voice.id}>{voice.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </SettingRow>

                      <SettingRow label={`Emotion Exaggeration (${(localSettings.ttsExaggeration || 0.5).toFixed(1)})`} layout="stack">
                        <Slider
                          id="tts-exaggeration"
                          min={0.0} max={1.0} step={0.1}
                          value={[localSettings.ttsExaggeration || 0.5]}
                          onValueChange={([v]) => handleChange('ttsExaggeration', v)}
                        />
                      </SettingRow>

                      {localSettings.ttsEngine === 'chatterbox' && (
                        <SettingRow label={`Guidance Scale (${(localSettings.ttsCfg || 0.5).toFixed(1)})`} layout="stack">
                          <Slider
                            id="tts-cfg"
                            min={0.1} max={1.0} step={0.1}
                            value={[localSettings.ttsCfg || 0.5]}
                            onValueChange={([v]) => handleChange('ttsCfg', v)}
                          />
                        </SettingRow>
                      )}

                      <SettingRow label="Generation Speed Mode" htmlFor="tts-speed-mode">
                        <Select
                          value={localSettings.ttsSpeedMode || 'standard'}
                          onValueChange={(value) => {
                            handleChange('ttsSpeedMode', value);
                          }}
                        >
                          <SelectTrigger className="w-full md:w-64">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="standard">Standard</SelectItem>
                            <SelectItem value="quality">Quality</SelectItem>
                          </SelectContent>
                        </Select>
                      </SettingRow>

                      <SettingsAccordion title="VRAM management" summary="Unload or reload Chatterbox service.">
                        <SettingRow label="VRAM Management" layout="stack">
                          <div className="flex flex-col md:flex-row gap-3">
                            <Button
                              variant="outline"
                              onClick={async () => {
                                try {
                                  setIsUnloadingChatterbox(true);
                                  await fetch(`${TTS_API_URL}/tts/unload-chatterbox`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                  alert('Unloaded!');
                                } finally { setIsUnloadingChatterbox(false); }
                              }}
                              disabled={isUnloadingChatterbox}
                              className="flex-1"
                            >
                              Unload Chatterbox
                            </Button>
                            <Button
                              variant="outline"
                              onClick={async () => {
                                try {
                                  setIsReloadingChatterbox(true);
                                  await fetch(`${TTS_API_URL}/tts/reload-chatterbox`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                  alert('Reloaded!');
                                } finally { setIsReloadingChatterbox(false); }
                              }}
                              disabled={isReloadingChatterbox}
                              className="flex-1"
                            >
                              Reload Chatterbox
                            </Button>
                          </div>
                        </SettingRow>
                      </SettingsAccordion>
                    </>
                  )}

                  <SettingRow label="Auto-Play TTS" htmlFor="tts-autoplay">
                    <Switch
                      id="tts-autoplay"
                      checked={localSettings.ttsAutoPlay}
                      onCheckedChange={(value) => handleChange('ttsAutoPlay', value)}
                    />
                  </SettingRow>

                  <SettingsAccordion
                    title="Exports and maintenance"
                    summary="Full-response export and storage diagnostics."
                  >
                    <SettingRow
                      label="Save full-response TTS audio"
                      htmlFor="tts-save-full-audio"
                      layout="stack"
                      description="When enabled, tap-to-play full /tts synthesis is written under backend/data/tts_full_exports (plus backups). One synthesis request still generates the full reply; you can keep a single WAV or split exports into fixed-duration segments (for tools that cap clip length, e.g. talking heads under five minutes)."
                    >
                      <div className="flex flex-col gap-2">
                        <Switch
                          id="tts-save-full-audio"
                          checked={localSettings.ttsSaveFullResponseAudio ?? false}
                          onCheckedChange={(value) => handleChange('ttsSaveFullResponseAudio', value)}
                          disabled={!ttsEnabled}
                        />
                        {localSettings.ttsSaveFullResponseAudio && (
                          <div className="space-y-2">
                            <Label className="text-xs text-muted-foreground">Export layout</Label>
                            <Select
                              value={String(localSettings.ttsSaveFullResponseChunkSeconds ?? 0)}
                              onValueChange={(v) => handleChange('ttsSaveFullResponseChunkSeconds', parseInt(v, 10))}
                              disabled={!ttsEnabled}
                            >
                              <SelectTrigger className="w-full max-w-md">
                                <SelectValue placeholder="Choose layout" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="0">One WAV file (full length)</SelectItem>
                                <SelectItem value="285">Split ~4m45 per file (under 5 min)</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        )}
                        {localSettings.ttsSaveFullResponseAudio && (
                          <div className="text-xs rounded border border-border/60 bg-background/40 px-3 py-2 space-y-1">
                            <div>
                              Save status: {ttsFullResponseSaveStatus?.state || 'idle'}{ttsFullResponseSaveStatus?.message ? ` - ${ttsFullResponseSaveStatus.message}` : ''}
                            </div>
                            {ttsFullResponseSaveStatus?.chunkCount > 1 ? (
                              <div>Segments: {ttsFullResponseSaveStatus.chunkCount}</div>
                            ) : null}
                            {ttsFullResponseSaveStatus?.filename ? (
                              <div className="break-all">Last file: {ttsFullResponseSaveStatus.filename}</div>
                            ) : null}
                            {ttsFullResponseSaveStatus?.path ? (
                              <div className="break-all">Path: {ttsFullResponseSaveStatus.path}</div>
                            ) : (
                              <div className="break-all">Folder: backend/data/tts_full_exports</div>
                            )}
                          </div>
                        )}
                      </div>
                    </SettingRow>
                  </SettingsAccordion>

                  <SettingRow label={`Speech Speed (${(localSettings.ttsSpeed || 1.0).toFixed(1)}x)`} layout="stack">
                    <Slider
                      id="tts-speed"
                      min={0.5} max={3.0} step={0.1}
                      value={[localSettings.ttsSpeed || 1.0]}
                      onValueChange={([v]) => handleChange('ttsSpeed', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Autoplay chunk size (${localSettings.ttsStreamChunkSentences || 2} sentence${(localSettings.ttsStreamChunkSentences || 2) === 1 ? '' : 's'})`}
                    layout="stack"
                    description="How many sentences the server groups per autoplay TTS chunk. Higher values increase initial wait but usually reduce boundary gaps."
                  >
                    <Slider
                      id="tts-stream-chunk-sentences"
                      min={1}
                      max={12}
                      step={1}
                      value={[localSettings.ttsStreamChunkSentences || 2]}
                      onValueChange={([v]) => handleChange('ttsStreamChunkSentences', v)}
                    />
                  </SettingRow>

                  {(localSettings.ttsEngine || 'kokoro') === 'kokoro' && (
                    <SettingRow label={`Pitch (${localSettings.ttsPitch} semitones)`} htmlFor="tts-pitch" layout="stack">
                      <Slider
                        id="tts-pitch"
                        min={-12} max={12} step={1}
                        value={[localSettings.ttsPitch || 0]}
                        onValueChange={([v]) => handleChange('ttsPitch', v)}
                      />
                    </SettingRow>
                  )}

                  <SettingRow
                    label="Test streaming TTS"
                    layout="stack"
                    description="Same WebSocket pipeline as chat autoplay: text is chunked on the server. At 1.0× speed, chunks play gaplessly via Web Audio; at other speeds, playback matches tap-to-play (HTML audio with pitch preserved). Engine/voice/exaggeration use the values above. Guidance scale applies only to Chatterbox (Faster), not Turbo. Connects even when Auto-Play TTS is off."
                  >
                    <input
                      ref={ttsTestFileInputRef}
                      type="file"
                      accept=".txt,text/plain"
                      className="hidden"
                      onChange={handleImportTtsTestText}
                    />
                    <Textarea
                      id="tts-stream-test-text"
                      className="w-full min-h-[128px] p-3 border rounded-md text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
                      value={ttsStreamTestText}
                      onChange={(e) => setTtsStreamTestText(e.target.value)}
                      placeholder="Type or paste multiple sentences to test playback…"
                      disabled={!ttsEnabled}
                    />
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {ttsStreamTestText.length.toLocaleString()} characters
                      </span>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleOpenTtsTestFilePicker}
                        >
                          Import .txt
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={
                            !ttsEnabled ||
                            isPlayingAudio === 'test-tts' ||
                            !String(ttsStreamTestText || '').trim()
                          }
                          onClick={() => playTestStreamingTTS(ttsStreamTestText, localSettings)}
                        >
                          {isPlayingAudio === 'test-tts' ? 'Playing…' : 'Play test'}
                        </Button>
                        {isPlayingAudio === 'test-tts' && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => (isStreamingTtsPaused ? resumeStreamingTTS() : pauseStreamingTTS())}
                          >
                            {isStreamingTtsPaused ? 'Resume' : 'Pause'}
                          </Button>
                        )}
                        {isPlayingAudio === 'test-tts' && (
                          <Button type="button" variant="ghost" size="sm" onClick={() => stopTTS()}>
                            Stop
                          </Button>
                        )}
                      </div>
                    </div>
                    {audioError ? (
                      <p className="text-sm text-destructive">{audioError}</p>
                    ) : null}
                  </SettingRow>
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Character as system prompt"
              description="Puts a chosen character card in the base system layer (model instructions, user profile, and that card's agentic memory). You still pick a normal chat character for the conversation — their roleplay card runs on top as Character Persona. New chats keep your selected character; only chats with no character use About this system intro."
            >
              <SettingRow label="Use character as system prompt" htmlFor="system-persona-enabled">
                <Switch
                  id="system-persona-enabled"
                  checked={localSettings.useCharacterAsSystemPrompt === true}
                  onCheckedChange={(value) => handleChange('useCharacterAsSystemPrompt', value)}
                />
              </SettingRow>
              {localSettings.useCharacterAsSystemPrompt === true && (
                <SettingRow
                  label="System persona character"
                  htmlFor="system-persona-character"
                  layout="stack"
                  description="Base system layer only — not your chat character. Profile and agentic memory for this id merge here; your selected chat character stays the roleplay speaker."
                >
                  <Select
                    value={localSettings.systemPersonaCharacterId || '__none__'}
                    onValueChange={(v) =>
                      handleChange('systemPersonaCharacterId', v === '__none__' ? null : v)
                    }
                  >
                    <SelectTrigger id="system-persona-character" className="w-full max-w-md">
                      <SelectValue placeholder="Select character" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__">— Select —</SelectItem>
                      {(characters || []).map((c) => (
                        <SelectItem key={c.id} value={c.id}>
                          {c.name || c.id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </SettingRow>
              )}
            </SettingsSection>

            <SettingsSection
              title="New chat — Character introduction"
              description="When enabled, new chats skip the static greeting and show an AI-generated introduction for the selected chat character (profile and agentic memory in full-generation mode). With Character as system prompt on, the system layer is still included in the prompt; intro stays about the chat character when one is selected."
            >
              <SettingRow label="Character introduction on new chat" htmlFor="character-intro-enabled">
                <Switch
                  id="character-intro-enabled"
                  checked={localSettings.characterIntroEnabled === true}
                  onCheckedChange={(value) => handleChange('characterIntroEnabled', value)}
                />
              </SettingRow>

              {localSettings.characterIntroEnabled === true && (
                <>
                  <SettingRow
                    label="Context mode"
                    htmlFor="character-intro-system-mode"
                    layout="stack"
                    description="Full generation includes the same system prompt as chat (memories, lore, agentic memory). Character card uses the card system prompt only."
                  >
                    <Select
                      value={localSettings.characterIntroSystemPromptMode || 'full_generation'}
                      onValueChange={(v) => handleChange('characterIntroSystemPromptMode', v)}
                    >
                      <SelectTrigger id="character-intro-system-mode" className="w-full max-w-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="flat">Flat character fields</SelectItem>
                        <SelectItem value="character_card">Character card system prompt</SelectItem>
                        <SelectItem value="full_generation">Full generation system prompt</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  <SettingRow
                    label="Custom intro prompt (optional)"
                    layout="stack"
                    description="Leave blank for the built-in JSON template. Placeholders: {{CHARACTER_BLOCK}}, {{USER_BLOCK}}, {{STORY_BLOCK}}, {{CHAT_HISTORY}}, {{CHARACTER_SYSTEM_PROMPT}}."
                  >
                    <Textarea
                      id="character-intro-prompt"
                      rows={6}
                      className="font-mono text-xs"
                      value={localSettings.characterIntroPrompt || ''}
                      onChange={(e) => handleChange('characterIntroPrompt', e.target.value)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Max tokens (${localSettings.characterIntroMaxTokens ?? 900})`}
                    layout="stack"
                  >
                    <Slider
                      min={400}
                      max={2000}
                      step={50}
                      value={[localSettings.characterIntroMaxTokens ?? 900]}
                      onValueChange={([v]) => handleChange('characterIntroMaxTokens', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Temperature (${(localSettings.characterIntroTemperature ?? 0.55).toFixed(2)})`}
                    layout="stack"
                  >
                    <Slider
                      min={0}
                      max={1.2}
                      step={0.05}
                      value={[localSettings.characterIntroTemperature ?? 0.55]}
                      onValueChange={([v]) => handleChange('characterIntroTemperature', v)}
                    />
                  </SettingRow>

                  <FlowApiOverrideFields
                    SettingRow={SettingRow}
                    idPrefix="character-intro"
                    settingsPrefix="characterIntro"
                    localSettings={localSettings}
                    onChange={handleChange}
                    description="Applies to request_purpose character_intro and system_intro so intro generation does not use your main chat model."
                  />
                </>
              )}
            </SettingsSection>

            <SettingsSection
              title="Call Mode — About This Character (experimental)"
              description="Hover the portrait in call mode to reveal an About button. Sends a one-off API request with character profile, user profile, story tracker, and chat history. Customize the prompt below; leave blank for the built-in template."
            >
              <SettingRow label="Enable About This Character" htmlFor="call-mode-about-enabled">
                <Switch
                  id="call-mode-about-enabled"
                  checked={localSettings.callModeAboutCharacterEnabled !== false}
                  onCheckedChange={(value) => handleChange('callModeAboutCharacterEnabled', value)}
                />
              </SettingRow>

              {localSettings.callModeAboutCharacterEnabled !== false && (
                <>
                  <SettingRow
                    label="Character context mode"
                    htmlFor="call-mode-about-system-mode"
                    layout="stack"
                    description="Character card / full generation sends the same system prompt used in chat first, then the intelligence-sheet JSON instructions. Full generation also includes memories, lore, and agentic memory when enabled."
                  >
                    <Select
                      value={localSettings.callModeAboutCharacterSystemPromptMode || 'flat'}
                      onValueChange={(v) => handleChange('callModeAboutCharacterSystemPromptMode', v)}
                    >
                      <SelectTrigger id="call-mode-about-system-mode" className="w-full max-w-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="flat">Flat character fields (legacy)</SelectItem>
                        <SelectItem value="character_card">Character card system prompt</SelectItem>
                        <SelectItem value="full_generation">Full generation system prompt</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  <SettingRow
                    label="Custom prompt template"
                    htmlFor="call-mode-about-prompt"
                    layout="stack"
                    description="Placeholders: {{CHARACTER_SYSTEM_PROMPT}}, {{CHARACTER_BLOCK}}, {{USER_BLOCK}}, {{STORY_BLOCK}}, {{CHAT_HISTORY}}. Empty = built-in template with per-card field rubric (Essence, On this call, With you, etc.). Custom templates should keep the same JSON keys."
                  >
                    <Textarea
                      id="call-mode-about-prompt"
                      rows={10}
                      value={localSettings.callModeAboutCharacterPrompt || ''}
                      onChange={(e) => handleChange('callModeAboutCharacterPrompt', e.target.value)}
                      placeholder="Leave blank to use the default structured prompt…"
                      className="font-mono text-xs"
                    />
                  </SettingRow>

                  <FlowApiOverrideFields
                    SettingRow={SettingRow}
                    idPrefix="call-mode-about"
                    settingsPrefix="callModeAboutCharacter"
                    localSettings={localSettings}
                    onChange={handleChange}
                    description="Uses request_purpose call_mode_character_about. Keeps About-panel requests off your main chat API when enabled."
                  />

                  <SettingRow label="Request purpose" htmlFor="call-mode-about-purpose" layout="stack">
                    <Input
                      id="call-mode-about-purpose"
                      value={localSettings.callModeAboutCharacterRequestPurpose || 'call_mode_character_about'}
                      onChange={(e) => handleChange('callModeAboutCharacterRequestPurpose', e.target.value)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Max tokens (${localSettings.callModeAboutCharacterMaxTokens ?? 1200})`}
                    layout="stack"
                  >
                    <Slider
                      min={400}
                      max={4000}
                      step={100}
                      value={[localSettings.callModeAboutCharacterMaxTokens ?? 1200]}
                      onValueChange={([v]) => handleChange('callModeAboutCharacterMaxTokens', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Temperature (${(localSettings.callModeAboutCharacterTemperature ?? 0.6).toFixed(2)})`}
                    layout="stack"
                  >
                    <Slider
                      min={0}
                      max={1.2}
                      step={0.05}
                      value={[localSettings.callModeAboutCharacterTemperature ?? 0.6]}
                      onValueChange={([v]) => handleChange('callModeAboutCharacterTemperature', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Chat history turns (${localSettings.callModeAboutCharacterHistoryLimit ?? 40})`}
                    layout="stack"
                  >
                    <Slider
                      min={10}
                      max={120}
                      step={5}
                      value={[localSettings.callModeAboutCharacterHistoryLimit ?? 40]}
                      onValueChange={([v]) => handleChange('callModeAboutCharacterHistoryLimit', v)}
                    />
                  </SettingRow>
                </>
              )}
              </SettingsSection>
          </div>
        </TabsContent>

        {/* Memory Intent Detection */}
        <TabsContent value="memory-intent">
          <div className="space-y-6">
            <SettingsSection
              title="Memory Intent Detector"
              description="Type text below to detect memory intent patterns."
            >
              <SettingRow label="Input Text" htmlFor="memory-intent-input" layout="stack">
                <textarea
                  id="memory-intent-input"
                  className="w-full p-3 border rounded text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800"
                  rows={3}
                  value={memoryIntentInput}
                  onChange={e => setMemoryIntentInput(e.target.value)}
                  placeholder="e.g. Remember that my favorite color is blue."
                />
              </SettingRow>
              <SettingRow label="Detection" layout="stack">
                <MemoryIntentDetector
                  text={memoryIntentInput}
                  onDetected={handleMemoryIntent}
                  allowExplicitCreation={true}
                />
              </SettingRow>
            </SettingsSection>
          </div>
        </TabsContent>

        {/* Persona realignment — prompt pack, LLM run, parse (optional code-excerpts appendix uses /memory/ethics_review/* on the server) */}
        <TabsContent value="persona-realignment">
          <div className="space-y-4">
            <Card className="border-primary/30 bg-gradient-to-br from-primary/[0.06] to-transparent shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Sparkles className="h-5 w-5 text-primary shrink-0" aria-hidden />
                  Persona realignment
                </CardTitle>
                <CardDescription className="text-sm leading-relaxed">
                  Use the sticky <strong>Build → Run model → Parse</strong> row. Defaults are enough to try it; open <strong>More options</strong> in the panel only if you need code snippets, long notes, or disk save. Curators and the memory list are under sidebar <strong>Memory tools</strong>.
                </CardDescription>
              </CardHeader>
            </Card>
            <PersonaRealignmentPanel />
          </div>
        </TabsContent>

        <TabsContent value="chatlog-condenser">
          <div className="space-y-4">
            <Card className="border-primary/30 bg-gradient-to-br from-primary/[0.06] to-transparent shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Chatlog condenser</CardTitle>
                <CardDescription className="text-sm leading-relaxed">
                  Compress long AI conversations without flattening reasoning structure — for sharing with another model inside context limits.
                </CardDescription>
              </CardHeader>
            </Card>
            <ChatlogCondenserPanel />
          </div>
        </TabsContent>

        {/* Memory Browser */}
        <TabsContent value="memory">
          <MemoryEditorTab onOpenPersonaRealignment={() => setSettingsMainTab('persona-realignment')} />
        </TabsContent>

        {/* Lore Debugger */}
        <TabsContent value="lore">
          <Card>
            <CardHeader><CardTitle>Lore Debugger</CardTitle></CardHeader>
            <CardContent><LoreDebugger /></CardContent>
          </Card>
        </TabsContent>
        {/*local sd*/}
        {/* About */}
        <TabsContent value="about">
          <div className="space-y-6">
            <SettingsSection
              title="About Eloquent"
              description="Local-first AI platform built for power users."
            >
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Eloquent is a local-first AI platform that combines LLM chat, image generation, voice,
                  evaluation, and tooling in one interface. It is designed to run on your hardware
                  (Windows + NVIDIA GPUs) with optional OpenAI-compatible API endpoints.
                </p>
                <p className="text-sm text-muted-foreground">
                  The stack pairs a React frontend with a FastAPI backend and includes multi-GPU orchestration,
                  a built-in Stable Diffusion pipeline, streaming TTS, a tool-calling code editor, and
                  a deep roleplay toolkit (character creator, multi-character chat, and lore).
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Core Systems</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Local LLM inference with multi-GPU support and OpenAI-compatible APIs</li>
                    <li>Built-in Stable Diffusion (SD, SDXL, FLUX) plus optional external engines</li>
                    <li>Streaming TTS with Kokoro and Chatterbox voice cloning</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Creator Tools</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Character creator and library with persona management</li>
                    <li>Multi-character chat with roster control, roles, and narrator support</li>
                    <li>Mobile-friendly UI and call-mode voice interface</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Evaluation and Analysis</h4>
                  <ul className="text-sm space-y-1 text-muted-foreground list-disc pl-4">
                    <li>Model ELO testing, A/B comparisons, and judge workflows</li>
                    <li>Forensic linguistics analysis with embedding models</li>
                    <li>Memory, RAG, and document ingestion tools</li>
                  </ul>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-2">Local-First Philosophy</h4>
                  <p className="text-sm text-muted-foreground">
                    Runs offline by default. Your data stays on your machine unless you enable external API
                    endpoints or web search.
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Credits and License</h4>
                  <p className="text-sm text-muted-foreground">
                    Built with FastAPI, React, llama.cpp, stable-diffusion.cpp, Kokoro, Chatterbox,
                    and ultralytics YOLO. Licensed under AGPL-3.0. Created by Bernard Peter Fitzgerald.
                  </p>
                </div>
              </div>
            </SettingsSection>
          </div>
        </TabsContent>

        </Tabs>
      </div>
    </div>
  );
};

const MemoryEditorTab = ({ onOpenPersonaRealignment }) => {
  const { activeProfileId } = useMemory();
  const {
    MEMORY_API_URL,
    PRIMARY_API_URL,
    characters = [],
    userProfile,
    isSingleGpuMode,
  } = useApp();
  const { apiReady } = useAppBoot();
  const [memories, setMemories] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deletingIndex, setDeletingIndex] = useState(null);
  const [duplicateIndices, setDuplicateIndices] = useState(new Set());
  const [duplicateStats, setDuplicateStats] = useState({ groups: 0, entries: 0 });
  const [dedupeBusy, setDedupeBusy] = useState(false);
  const [dedupeMode, setDedupeMode] = useState('exact');
  const [memoryTab, setMemoryTab] = useState('profile');
  // Agentic memories (per-character insights from agentic JSON)
  const [agenticProfiles, setAgenticProfiles] = useState([]);
  const [agenticLoading, setAgenticLoading] = useState(false);
  const [agenticError, setAgenticError] = useState(null);
  const [deletingAgenticId, setDeletingAgenticId] = useState(null);
  const [editingAgenticInsight, setEditingAgenticInsight] = useState(null);
  const [savingAgenticId, setSavingAgenticId] = useState(null);
  const [expandedAgenticCharacterIds, setExpandedAgenticCharacterIds] = useState(() => new Set());
  /** Per source character: { targetId, mode } for copy-to-character (agentic). */
  const [agenticCopyDrafts, setAgenticCopyDrafts] = useState({});
  const [agenticCopyBusySource, setAgenticCopyBusySource] = useState(null);
  const API_URL = MEMORY_API_URL;
  /** Same host as chat writes (MEMORY_API_URL); dual-GPU stores agentic JSON on the memory port. */
  const AGENTIC_API_URL = (MEMORY_API_URL || PRIMARY_API_URL || '').replace(/\/$/, '');

  const toggleAgenticCharacterExpanded = useCallback((characterId) => {
    if (!characterId) return;
    setExpandedAgenticCharacterIds((prev) => {
      const next = new Set(prev);
      if (next.has(characterId)) next.delete(characterId);
      else next.add(characterId);
      return next;
    });
  }, []);

  const computeDuplicateIndicesFromList = useCallback((list, mode = 'exact') => {
    const firstSentenceKey = (value) => {
      const text = String(value || '').trim();
      if (!text) return '';
      const m = text.match(/^(.*?[.!?])(?:\s|$)/s);
      const first = m ? m[1] : text.split(/\s+/).slice(0, 20).join(' ');
      return first.trim().toLowerCase().replace(/\s+/g, ' ');
    };
    const grouped = new Map();
    (list || []).forEach((memory, idx) => {
      const content = typeof memory?.content === 'string' ? memory.content : '';
      const key = mode === 'first_sentence'
        ? firstSentenceKey(content)
        : content.trim().toLowerCase().replace(/\s+/g, ' ');
      if (!key) return;
      const arr = grouped.get(key) || [];
      arr.push(idx);
      grouped.set(key, arr);
    });
    const toRemove = [];
    let groups = 0;
    grouped.forEach((arr) => {
      if (arr.length > 1) {
        groups += 1;
        toRemove.push(...arr.slice(1));
      }
    });
    return { indices: new Set(toRemove), groups, entries: toRemove.length };
  }, []);

  const getCharacterName = (characterId) => {
    const c = characters.find((ch) => ch.id === characterId);
    return c?.name || characterId;
  };

  // Helper function to format dates
  const formatDate = (s) => {
    const d = new Date(s);
    return isNaN(d.getTime()) ? 'Invalid Date' : d.toLocaleDateString();
  };

  // Helper function for category colors (profile + agentic)
  const getCategoryColor = (category) => {
    const colors = {
      'expertise': 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
      'personal_interest': 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      'preferences': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      'personal_info': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
      'insight': 'bg-slate-100 text-slate-800 dark:bg-slate-900 dark:text-slate-200',
      'preference': 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      'behavior': 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
      'habit': 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
      'identity': 'bg-violet-100 text-violet-800 dark:bg-violet-900 dark:text-violet-200',
      'plan': 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      'background': 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
      'medical_physiology': 'bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200',
    };
    return colors[category] || colors.personal_info;
  };

  // Fetch memories from backend
  const fetchMemories = useCallback(async () => {
    if (!activeProfileId) {
      setMemories([]);
      return;
    }
    if (!apiReady) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetchWithTimeout(`${API_URL}/memory/get_all?user_id=${activeProfileId}`, {}, 25000);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      if (data.status === 'success' && Array.isArray(data.memories)) {
        const sorted = data.memories.sort((a, b) => new Date(b.created) - new Date(a.created));
        setMemories(sorted);
        setDuplicateIndices(new Set());
        setDuplicateStats({ groups: 0, entries: 0 });
      } else {
        throw new Error(data.error || 'Unexpected response');
      }
    } catch (err) {
      setError(
        formatFetchError(err, {
          timeoutMs: 25000,
          hint: memoryApiUnreachableHint({
            isSingleGpuMode,
            memoryUrl: API_URL,
            primaryUrl: PRIMARY_API_URL,
          }),
        }),
      );
      setMemories([]);
    } finally {
      setIsLoading(false);
    }
  }, [activeProfileId, API_URL, apiReady, isSingleGpuMode, PRIMARY_API_URL]);

  const previewProfileDuplicates = useCallback(async () => {
    if (!activeProfileId) return;
    setDedupeBusy(true);
    try {
      // Backend pass: privacy-safe index/count calculation only (no memory content returned).
      const res = await fetchWithTimeout(`${API_URL}/memory/duplicates/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: activeProfileId, mode: dedupeMode })
      }, 25000);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Status ${res.status}`);
      }
      const local = computeDuplicateIndicesFromList(memories, dedupeMode);
      setDuplicateIndices(local.indices);
      setDuplicateStats({ groups: local.groups, entries: local.entries });
      if (local.entries === 0) {
        alert('No direct duplicates found.');
      }
    } catch (err) {
      alert(`Failed to scan duplicates: ${err.message}`);
    } finally {
      setDedupeBusy(false);
    }
  }, [activeProfileId, API_URL, memories, computeDuplicateIndicesFromList, dedupeMode]);

  const removeProfileDuplicates = useCallback(async () => {
    if (!activeProfileId) return;
    if (!window.confirm('Remove all directly duplicate profile memories (keep first copy)?')) return;
    setDedupeBusy(true);
    try {
      const res = await fetchWithTimeout(`${API_URL}/memory/duplicates/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: activeProfileId, mode: dedupeMode })
      }, 25000);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Status ${res.status}`);
      }
      const data = await res.json();
      alert(`Removed ${data.removed || 0} duplicate entries.`);
      await fetchMemories();
    } catch (err) {
      alert(`Failed to remove duplicates: ${err.message}`);
    } finally {
      setDedupeBusy(false);
    }
  }, [activeProfileId, API_URL, fetchMemories, dedupeMode]);

  const handleDelete = useCallback(async (memory, index) => {
    if (!activeProfileId) return;
    if (!window.confirm(`Delete this memory?`)) return;

    setDeletingIndex(index);

    try {
      const freshResponse = await fetchWithTimeout(`${API_URL}/memory/get_all?user_id=${activeProfileId}`, {}, 25000);
      if (!freshResponse.ok) throw new Error(`Failed to fetch fresh data`);

      const freshData = await freshResponse.json();
      const freshMemories = Array.isArray(freshData.memories) ? freshData.memories : [];
      const targetMemory = freshMemories.find(m => m.content === memory.content && m.created === memory.created);

      if (!targetMemory) throw new Error('Memory not found');

      const memoriesToKeep = freshMemories.filter(m => !(m.content === targetMemory.content && m.created === targetMemory.created));

      const clearResponse = await fetchWithTimeout(`${API_URL}/memory/clear?user_id=${activeProfileId}`, {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' }
      }, 25000);

      if (!clearResponse.ok) throw new Error(`Clear failed`);

      for (const memoryToSave of memoriesToKeep) {
        const memoryWithUserId = { ...memoryToSave, user_id: activeProfileId };
        await fetchWithTimeout(`${API_URL}/memory/memory/create`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(memoryWithUserId)
        }, 25000);
      }
      setMemories(prev => prev.filter((_, i) => i !== index));
      setTimeout(fetchMemories, 1000);

    } catch (err) {
      alert(`Failed to delete memory: ${err.message}`);
      await fetchMemories();
    } finally {
      setDeletingIndex(null);
    }
  }, [activeProfileId, API_URL, fetchMemories]);

  // Agentic: fetch all agentic profiles for current user
  const fetchAgenticMemories = useCallback(async () => {
    if (!activeProfileId) {
      setAgenticProfiles([]);
      return;
    }
    if (!apiReady || !AGENTIC_API_URL) return;
    setAgenticLoading(true);
    setAgenticError(null);
    try {
      const listUrl = `${AGENTIC_API_URL}/memory/agentic/list?user_id=${encodeURIComponent(activeProfileId)}`;
      const res = await fetchWithTimeout(listUrl, {}, 25000);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success' && Array.isArray(data.profiles)) {
          setAgenticProfiles(data.profiles);
          return;
        }
        throw new Error(data.error || 'Unexpected response');
      }
      if (res.status === 404 && Array.isArray(characters) && characters.length > 0) {
        const rows = await Promise.all(
          characters.map(async (ch) => {
            const r = await fetchWithTimeout(
              `${AGENTIC_API_URL}/memory/agentic?user_id=${encodeURIComponent(activeProfileId)}&character_id=${encodeURIComponent(ch.id)}`,
              {},
              25000
            );
            if (!r.ok) return null;
            const d = await r.json();
            if (d.status !== 'success' || !Array.isArray(d.insights)) return null;
            const insights = d.insights;
            if (!insights.length) return null;
            return { character_id: ch.id, insights, count: insights.length, meta: {} };
          })
        );
        setAgenticProfiles(rows.filter(Boolean));
        return;
      }
      throw new Error(`Status ${res.status}`);
    } catch (err) {
      setAgenticError(formatFetchError(err, { timeoutMs: 25000 }));
      setAgenticProfiles([]);
    } finally {
      setAgenticLoading(false);
    }
  }, [activeProfileId, AGENTIC_API_URL, apiReady, characters]);

  const handleDeleteAgenticInsight = useCallback(async (characterId, insightId) => {
    if (!activeProfileId) return;
    if (!window.confirm('Delete this agentic memory?')) return;
    setDeletingAgenticId(insightId);
    try {
      const res = await fetchWithTimeout(`${AGENTIC_API_URL}/memory/agentic/delete_insights`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: activeProfileId,
          character_id: characterId,
          insight_ids: [insightId],
        }),
      }, 25000);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Status ${res.status}`);
      }
      setAgenticProfiles((prev) =>
        prev.map((p) =>
          p.character_id === characterId
            ? {
                ...p,
                insights: (p.insights || []).filter((i) => i.id !== insightId),
                count: Math.max(0, (p.count || 0) - 1),
              }
            : p
        )
      );
      await fetchAgenticMemories();
    } catch (err) {
      alert(`Failed to delete agentic memory: ${err.message}`);
      await fetchAgenticMemories();
    } finally {
      setDeletingAgenticId(null);
    }
  }, [activeProfileId, AGENTIC_API_URL, fetchAgenticMemories]);

  const handleSaveAgenticEdit = useCallback(async (characterId, insightId, newContent) => {
    if (!activeProfileId || !newContent?.trim()) return;
    setSavingAgenticId(insightId);
    try {
      const res = await fetchWithTimeout(`${AGENTIC_API_URL}/memory/agentic/update_insight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: activeProfileId,
          character_id: characterId,
          insight_id: insightId,
          content: newContent.trim(),
        }),
      }, 25000);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Status ${res.status}`);
      }
      setEditingAgenticInsight(null);
      setAgenticProfiles((prev) =>
        prev.map((p) =>
          p.character_id === characterId
            ? {
                ...p,
                insights: (p.insights || []).map((i) =>
                  i.id === insightId ? { ...i, content: newContent.trim() } : i
                ),
              }
            : p
        )
      );
      await fetchAgenticMemories();
    } catch (err) {
      alert(`Failed to save: ${err.message}`);
    } finally {
      setSavingAgenticId(null);
    }
  }, [activeProfileId, AGENTIC_API_URL, fetchAgenticMemories]);

  const handleAgenticCopyToCharacter = useCallback(
    async (sourceCharacterId) => {
      if (!activeProfileId) return;
      const draft = agenticCopyDrafts[sourceCharacterId] || {};
      const targetId = draft.targetId;
      const mode = draft.mode === 'replace' ? 'replace' : 'merge';
      if (!targetId) {
        alert('Choose a target character.');
        return;
      }
      if (mode === 'replace') {
        const srcName = characters.find((c) => c.id === sourceCharacterId)?.name || sourceCharacterId;
        const tgtName = characters.find((c) => c.id === targetId)?.name || targetId;
        const ok = window.confirm(
          `Replace all agentic memories for "${tgtName}" with a copy from "${srcName}"? This cannot be undone.`
        );
        if (!ok) return;
      }
      setAgenticCopyBusySource(sourceCharacterId);
      try {
        const res = await fetchWithTimeout(`${AGENTIC_API_URL}/memory/agentic/copy_to_character`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: activeProfileId,
            source_character_id: sourceCharacterId,
            target_character_id: targetId,
            mode,
          }),
        }, 25000);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || data.error || `Status ${res.status}`);
        }
        const tName = characters.find((c) => c.id === targetId)?.name || targetId;
        if (mode === 'merge') {
          alert(
            `Merged: ${data.added ?? 0} new insight(s) added (${data.cloned_candidates ?? 0} from source). Target "${tName}" now has ${data.target_count ?? '?'}.`
          );
        } else {
          alert(`Replaced target "${tName}" with ${data.written_count ?? 0} insight(s).`);
        }
        await fetchAgenticMemories();
      } catch (err) {
        alert(`Copy failed: ${err.message}`);
      } finally {
        setAgenticCopyBusySource(null);
      }
    },
    [activeProfileId, AGENTIC_API_URL, agenticCopyDrafts, fetchAgenticMemories, characters]
  );

  useEffect(() => {
    fetchMemories();
    fetchAgenticMemories();
  }, [fetchMemories, fetchAgenticMemories]);

  const totalAgenticCount = agenticProfiles.reduce((n, p) => n + (p.count || 0), 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <CardTitle>Memory Browser</CardTitle>
            <CardDescription>
              Profile: {activeProfileId ?? 'None'} ·{' '}
              {memoryTab === 'profile'
                ? `${memories.length} profile memories`
                : memoryTab === 'agentic'
                  ? `${totalAgenticCount} agentic insights`
                  : 'Persona realignment — open Settings tab'}
            </CardDescription>
          </div>
          <div className="flex gap-2 w-full md:w-auto">
            <Button
              size="sm"
              variant="outline"
              onClick={memoryTab === 'profile' ? fetchMemories : memoryTab === 'agentic' ? fetchAgenticMemories : undefined}
              disabled={
                memoryTab === 'realign' || (memoryTab === 'profile' ? isLoading : agenticLoading)
              }
              title={memoryTab === 'realign' ? 'Refresh is not used on this tab' : undefined}
            >
              {(memoryTab === 'profile' ? isLoading : memoryTab === 'agentic' ? agenticLoading : false) ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
            {memoryTab === 'profile' && (
              <>
                <Select value={dedupeMode} onValueChange={(value) => {
                  setDedupeMode(value);
                  setDuplicateIndices(new Set());
                  setDuplicateStats({ groups: 0, entries: 0 });
                }}>
                  <SelectTrigger className="h-8 w-[190px] text-xs">
                    <SelectValue placeholder="Dedupe mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="exact">Exact normalized text</SelectItem>
                    <SelectItem value="first_sentence">Same first sentence</SelectItem>
                  </SelectContent>
                </Select>
                <Button size="sm" variant="outline" onClick={previewProfileDuplicates} disabled={dedupeBusy || isLoading || !activeProfileId}>
                  {dedupeBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Find Duplicates'}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={removeProfileDuplicates}
                  disabled={dedupeBusy || !activeProfileId || duplicateStats.entries === 0}
                  title="Remove only direct duplicates (normalized exact match), keeps first copy."
                >
                  {dedupeBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Remove Duplicates'}
                </Button>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!apiReady && (
          <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin shrink-0" />
            Connecting to memory API (waiting for network config).
          </div>
        )}
        {!activeProfileId && (
          <div className="text-center py-8 text-amber-600 bg-amber-50 dark:bg-amber-950/20 rounded mb-4">
            No active profile selected. Please select a profile to view memories.
          </div>
        )}

        <Tabs value={memoryTab} onValueChange={setMemoryTab} className="w-full">
          <TabsList className="grid w-full max-w-2xl grid-cols-3 mb-4">
            <TabsTrigger value="profile">Profile memories</TabsTrigger>
            <TabsTrigger value="agentic">Agentic memories</TabsTrigger>
            <TabsTrigger value="realign">Persona realign</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="mt-0">
            {(duplicateStats.groups > 0 || duplicateStats.entries > 0) && (
              <div className="mb-3 rounded border border-amber-300/60 bg-amber-50/60 px-3 py-2 text-xs text-amber-800 dark:border-amber-700/50 dark:bg-amber-950/30 dark:text-amber-300">
                Duplicate suggestion ({dedupeMode === 'first_sentence' ? 'same first sentence' : 'exact normalized text'}):
                {' '}{duplicateStats.entries} entries across {duplicateStats.groups} duplicate groups are highlighted below.
              </div>
            )}
            {isLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
                Loading memories...
              </div>
            )}
            {error && (
              <div className="text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded">Error: {error}</div>
            )}
            {!isLoading && !error && memories.length === 0 && activeProfileId && (
              <div className="text-center py-8 text-muted-foreground">No profile memories for this user.</div>
            )}
            {!isLoading && !error && memories.length > 0 && (
              <div className="space-y-2">
                {memories.map((memory, index) => (
                  <div
                    key={`${memory.content}-${index}`}
                    className={`flex flex-col md:flex-row items-start justify-between p-3 border rounded-lg hover:bg-muted/30 transition-colors gap-3 ${
                      duplicateIndices.has(index)
                        ? 'border-amber-400/80 bg-amber-50/40 dark:bg-amber-900/10'
                        : ''
                    }`}
                  >
                    <div className="flex-1 min-w-0 w-full">
                      <p className="text-sm mb-2 text-wrap break-words">{memory.content}</p>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getCategoryColor(memory.category)}`}>
                          {memory.category?.replace('_', ' ') || 'unknown'}
                        </span>
                        <span>★ {memory.importance?.toFixed(1) || '0.0'}</span>
                        <span>{formatDate(memory.created)}</span>
                        {memory.accessed && <span>accessed {memory.accessed}x</span>}
                      </div>
                    </div>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => handleDelete(memory, index)}
                      disabled={deletingIndex === index}
                      className="self-end md:self-start flex-shrink-0 ml-3 text-muted-foreground hover:text-destructive"
                    >
                      {deletingIndex === index ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="agentic" className="mt-0">
            {agenticLoading && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
                Loading agentic memories...
              </div>
            )}
            {agenticError && (
              <div className="text-red-600 bg-red-50 dark:bg-red-950/20 p-3 rounded mb-4">Error: {agenticError}</div>
            )}
            {!agenticLoading && !agenticError && agenticProfiles.length === 0 && activeProfileId && (
              <div className="text-center py-8 text-muted-foreground">No agentic memories for this profile. They are created per character when Brain is enabled.</div>
            )}
            {!agenticLoading && !agenticError && agenticProfiles.length > 0 && (
              <div className="space-y-2">
                {agenticProfiles.map((profile) => {
                  const cid = profile.character_id;
                  const isExpanded =
                    expandedAgenticCharacterIds.has(cid) ||
                    (editingAgenticInsight?.characterId === cid);
                  return (
                    <div key={cid} className="rounded-lg border bg-muted/20 overflow-hidden">
                      <button
                        type="button"
                        className="w-full flex items-center gap-2 p-3 text-left hover:bg-muted/40 transition-colors"
                        onClick={() => toggleAgenticCharacterExpanded(cid)}
                        aria-expanded={isExpanded}
                      >
                        <ChevronRight
                          className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                          aria-hidden
                        />
                        <span className="font-medium text-sm truncate min-w-0">
                          {getCharacterName(cid)}
                        </span>
                        <span className="text-xs font-normal text-muted-foreground shrink-0">
                          ({profile.count ?? (profile.insights || []).length} insights)
                        </span>
                        <span className="text-xs text-muted-foreground ml-auto shrink-0 hidden sm:inline">
                          {isExpanded ? 'Hide' : 'Show'}
                        </span>
                      </button>
                      {isExpanded && (
                        <div className="space-y-2 px-3 pb-3 pt-0 border-t border-border/60 bg-background/40">
                          {(profile.insights || []).length > 0 &&
                            characters.filter((c) => c.id && c.id !== cid).length > 0 && (
                              <div className="rounded-md border border-dashed border-border/80 bg-muted/10 p-3 space-y-2 mb-1">
                                <p className="text-xs text-muted-foreground leading-relaxed">
                                  Optional: copy these chat-derived (agentic) memories to another character&apos;s file for the same user profile.
                                  <strong className="font-medium text-foreground"> Merge</strong> adds lines and skips content duplicates already on the target.
                                  <strong className="font-medium text-foreground"> Replace</strong> overwrites the target character&apos;s entire agentic file.
                                </p>
                                <div className="flex flex-col sm:flex-row flex-wrap gap-2 items-stretch sm:items-end">
                                  <div className="flex-1 min-w-[180px] space-y-1">
                                    <Label className="text-xs">Target character</Label>
                                    <Select
                                      value={agenticCopyDrafts[cid]?.targetId ?? ''}
                                      onValueChange={(v) =>
                                        setAgenticCopyDrafts((prev) => ({
                                          ...prev,
                                          [cid]: {
                                            targetId: v,
                                            mode: prev[cid]?.mode === 'replace' ? 'replace' : 'merge',
                                          },
                                        }))
                                      }
                                    >
                                      <SelectTrigger className="h-8 text-xs">
                                        <SelectValue placeholder="Choose character…" />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {characters
                                          .filter((c) => c.id && c.id !== cid)
                                          .map((c) => (
                                            <SelectItem key={c.id} value={c.id}>
                                              {c.name || c.id}
                                            </SelectItem>
                                          ))}
                                      </SelectContent>
                                    </Select>
                                  </div>
                                  <div className="w-full sm:w-[168px] space-y-1">
                                    <Label className="text-xs">Mode</Label>
                                    <Select
                                      value={agenticCopyDrafts[cid]?.mode === 'replace' ? 'replace' : 'merge'}
                                      onValueChange={(v) =>
                                        setAgenticCopyDrafts((prev) => ({
                                          ...prev,
                                          [cid]: {
                                            targetId: prev[cid]?.targetId ?? '',
                                            mode: v === 'replace' ? 'replace' : 'merge',
                                          },
                                        }))
                                      }
                                    >
                                      <SelectTrigger className="h-8 text-xs">
                                        <SelectValue />
                                      </SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="merge">Merge (dedupe)</SelectItem>
                                        <SelectItem value="replace">Replace target</SelectItem>
                                      </SelectContent>
                                    </Select>
                                  </div>
                                  <Button
                                    type="button"
                                    size="sm"
                                    className="h-8 shrink-0 flex items-center gap-1"
                                    disabled={agenticCopyBusySource === cid || !(agenticCopyDrafts[cid]?.targetId)}
                                    onClick={() => handleAgenticCopyToCharacter(cid)}
                                  >
                                    {agenticCopyBusySource === cid ? (
                                      <Loader2 className="h-4 w-4 animate-spin" />
                                    ) : (
                                      <Link2 className="h-4 w-4" />
                                    )}
                                    Apply
                                  </Button>
                                </div>
                              </div>
                            )}
                          {(profile.insights || []).map((insight) => {
                        const isEditing = editingAgenticInsight?.insightId === insight.id && editingAgenticInsight?.characterId === profile.character_id;
                        return (
                          <div
                            key={insight.id}
                            className="flex flex-col md:flex-row items-start justify-between p-3 border rounded-lg bg-background/60 hover:bg-muted/30 gap-3"
                          >
                            <div className="flex-1 min-w-0 w-full">
                              {isEditing ? (
                                <>
                                  <Textarea
                                    className="text-sm mb-2 min-h-[80px] resize-y"
                                    value={editingAgenticInsight.content}
                                    onChange={(e) => setEditingAgenticInsight((prev) => (prev ? { ...prev, content: e.target.value } : null))}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Escape') setEditingAgenticInsight(null);
                                    }}
                                  />
                                  <div className="flex gap-2 mt-2">
                                    <Button
                                      size="sm"
                                      onClick={() => handleSaveAgenticEdit(profile.character_id, insight.id, editingAgenticInsight.content)}
                                      disabled={savingAgenticId === insight.id}
                                    >
                                      {savingAgenticId === insight.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                                      Save
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={() => setEditingAgenticInsight(null)}
                                      disabled={savingAgenticId === insight.id}
                                    >
                                      Cancel
                                    </Button>
                                  </div>
                                </>
                              ) : (
                                <>
                                  <p className="text-sm mb-2 text-wrap break-words">{insight.content}</p>
                                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                    <span className={`px-2 py-1 rounded font-medium ${getCategoryColor(insight.category)}`}>
                                      {(insight.category || 'insight').replace(/_/g, ' ')}
                                    </span>
                                    <span>★ {(insight.importance ?? 0.5).toFixed(1)}</span>
                                    <span>{formatDate(insight.created_at)}</span>
                                  </div>
                                </>
                              )}
                            </div>
                            {!isEditing && (
                              <div className="flex gap-1 self-end md:self-start flex-shrink-0">
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => setEditingAgenticInsight({ characterId: profile.character_id, insightId: insight.id, content: insight.content })}
                                  disabled={!!editingAgenticInsight}
                                  className="text-muted-foreground hover:text-foreground"
                                  title="Edit"
                                >
                                  <Pencil className="h-4 w-4" />
                                </Button>
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => handleDeleteAgenticInsight(profile.character_id, insight.id)}
                                  disabled={deletingAgenticId === insight.id}
                                  className="text-muted-foreground hover:text-destructive"
                                  title="Delete"
                                >
                                  {deletingAgenticId === insight.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                                </Button>
                              </div>
                            )}
                          </div>
                        );
                      })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </TabsContent>

          <TabsContent value="realign" className="mt-0">
            <div className="rounded-xl border border-border/70 bg-muted/15 p-6 space-y-4 max-w-xl">
              <p className="text-sm font-semibold text-foreground">Persona realignment (prompt builder)</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                The full builder is not duplicated here — it lives under <strong>Settings → Persona realignment</strong> (same screen as <strong>Memory tools</strong> section 1 in the sidebar).
                Optional code excerpts in that builder call <code className="text-[11px] px-1 rounded bg-muted">/memory/ethics_review/</code> on the server; that is an add-on bundle, not a separate feature from realignment.
              </p>
              {typeof onOpenPersonaRealignment === 'function' ? (
                <Button type="button" variant="default" size="sm" onClick={onOpenPersonaRealignment}>
                  Open Persona realignment (Settings)
                </Button>
              ) : (
                <p className="text-xs text-muted-foreground">Open Settings → Persona realignment.</p>
              )}
            </div>
          </TabsContent>
        </Tabs>
        {memoryTab !== 'realign' && (
          <SettingsAccordion
            title="Memory curator"
            summary="Manual curate tools for profile and agentic memories."
          >
            <MemoryCuratorPanel
              apiUrl={API_URL}
              apiReady={apiReady}
              activeProfileId={activeProfileId}
              userProfile={userProfile}
              characters={characters}
              scope={memoryTab === 'agentic' ? 'agentic' : 'profile'}
              onApplied={() => {
                fetchMemories();
                fetchAgenticMemories();
              }}
            />
          </SettingsAccordion>
        )}
      </CardContent>
    </Card>
  );
};

export default Settings;

