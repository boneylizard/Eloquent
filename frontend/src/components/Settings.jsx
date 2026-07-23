// Settings.jsx
// Full Settings UI: General, Generation, SD, Audio, Memory Intent, Persona realignment, Memory Browser, Lore, About

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
import MemoryIntentDetector from './MemoryIntentDetector';
import { useApp } from '../contexts/AppContext';
import { useMemory } from '../contexts/MemoryContext';
import LocalStorageManager from './LocalStorageManager';
import PersonaRealignmentPanel from './PersonaRealignmentPanel';
import FlowApiOverrideFields from './FlowApiOverrideFields';
import VoiceSculptPanel from './VoiceSculptPanel';
import MemoryCuratorPanel from './MemoryCuratorPanel';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';
import { VisionModelSettings } from './VisionModelSelector';
import { resolveEndpointDisplay, getRotationPool } from '../utils/resolveEndpointDisplay';
import { readNanoGptModelsCache } from '../utils/nanoGptModelsCache';
import MobileRemoteSettings from './MobileRemoteSettings';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import {
  TV_PERF_STORAGE_KEY,
  readTvPerformanceFromUrl,
  readTvPerformanceFromStorage,
  applyTvPerformanceClass,
} from '../utils/tvPerformanceMode';
import { useAppBoot } from '../hooks/useAppBoot';
import { restartTtsService, stopTtsService } from '../utils/desktopLifecycle';
import InfrastructureBanner from './InfrastructureBanner';
import ModelLibrary from './ModelLibrary';
import AppUpdateControls from './AppUpdateControls';
import { replaceSettingsBlob } from '../utils/settingsPersistence';
import {
  SPLASH_DURATION_OPTIONS,
  SPLASH_SCREEN_DURATION_DEFAULT,
} from '../utils/eloquentSplash';
import {
  INTERFACE_ZOOM_DEFAULT,
  INTERFACE_ZOOM_EVENT,
  INTERFACE_ZOOM_MAX,
  INTERFACE_ZOOM_MIN,
  readInterfaceZoom,
  setInterfaceZoom,
} from '../utils/interfaceZoom';

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


const resolveSettingsTab = (tab, audioPage) => {
  if (audioPage) return 'audio';
  return tab === 'characters' ? 'general' : tab;
};

const Settings = ({ darkMode, toggleDarkMode, initialTab = 'general', isStandaloneWindow = false, audioPage = false }) => {
  const [settingsMainTab, setSettingsMainTab] = useState(() => resolveSettingsTab(initialTab, audioPage));
  const [interfaceZoom, setInterfaceZoomValue] = useState(readInterfaceZoom);
  useEffect(() => {
    setSettingsMainTab(resolveSettingsTab(initialTab, audioPage));
  }, [audioPage, initialTab]);
  useEffect(() => {
    const handleZoomChange = (event) => setInterfaceZoomValue(event.detail?.scale || readInterfaceZoom());
    window.addEventListener(INTERFACE_ZOOM_EVENT, handleZoomChange);
    return () => window.removeEventListener(INTERFACE_ZOOM_EVENT, handleZoomChange);
  }, []);
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
  } = useApp();

  const headingFontOptions = [
    { label: 'Default (Theme)', value: 'default' },
    { label: 'Poppins', value: "'Poppins', sans-serif" },
    { label: 'Inter', value: "'Inter', sans-serif" },
    { label: 'JetBrains Mono', value: "'JetBrains Mono', monospace" }
  ];

  // Helper to format Kokoro voice IDs (e.g., 'af_heart' -> 'Heart (Female)')
  const formatKokoroVoiceName = (voiceId) => {
    const voiceLabels = {
      'af_heart': 'Heart (Female)',
      'af_alloy': 'Alloy (Female)',
      'af_aoede': 'Aoede (Female)',
      'af_bella': 'Bella (Female)',
      'af_jessica': 'Jessica (Female)',
      'af_kore': 'Kore (Female)',
      'af_nicole': 'Nicole (Female)',
      'af_nova': 'Nova (Female)',
      'af_river': 'River (Female)',
      'af_sarah': 'Sarah (Female)',
      'af_sky': 'Sky (Female)',
      'am_adam': 'Adam (Male)',
      'am_echo': 'Echo (Male)',
      'am_eric': 'Eric (Male)',
      'am_fenrir': 'Fenrir (Male)',
      'am_liam': 'Liam (Male)',
      'am_michael': 'Michael (Male)',
      'am_onyx': 'Onyx (Male)',
      'am_puck': 'Puck (Male)',
      'bf_alice': 'Alice (Female, British)',
      'bf_emma': 'Emma (Female, British)',
      'bf_isabella': 'Isabella (Female, British)',
      'bf_lily': 'Lily (Female, British)',
      'bm_daniel': 'Daniel (Male, British)',
      'bm_fable': 'Fable (Male, British)',
      'bm_george': 'George (Male, British)',
      'bm_lewis': 'Lewis (Male, British)',
      'ff_siwis': 'Siwis (Female, French)',
      'hf_alpha': 'Alpha (Female, Hindi)',
      'hf_beta': 'Beta (Female, Hindi)',
      'hm_omega': 'Omega (Male, Hindi)',
      'hm_psi': 'Psi (Male, Hindi)',
      'if_sara': 'Sara (Female, Italian)',
      'im_nicola': 'Nicola (Male, Italian)',
      'jf_alpha': 'Alpha (Female, Japanese)',
      'jf_gongitsune': 'Gongitsune (Female, Japanese)',
      'jf_nezumi': 'Nezumi (Female, Japanese)',
      'jf_tebukuro': 'Tebukuro (Female, Japanese)',
      'jm_kumo': 'Kumo (Male, Japanese)',
      'zf_xiaobei': 'Xiaobei (Female, Chinese)',
      'zf_xiaoni': 'Xiaoni (Female, Chinese)',
      'zf_xiaoxiao': 'Xiaoxiao (Female, Chinese)',
      'zf_xiaoyi': 'Xiaoyi (Female, Chinese)',
      'zm_yunjian': 'Yunjian (Male, Chinese)',
      'zm_yunxi': 'Yunxi (Male, Chinese)',
      'zm_yunxia': 'Yunxia (Male, Chinese)',
      'zm_yunyang': 'Yunyang (Male, Chinese)',
    };
    return voiceLabels[voiceId] || voiceId;
  };

  // Helper to group Kokoro voices by language for the dropdown
  const getKokoroVoiceGroups = () => {
    const groups = {
      'English': [],
      'British English': [],
      'French': [],
      'Hindi': [],
      'Italian': [],
      'Japanese': [],
      'Chinese': [],
    };
    const allVoices = [
      // American English Female
      { id: 'af_heart', label: 'Heart (Female)' },
      { id: 'af_alloy', label: 'Alloy (Female)' },
      { id: 'af_aoede', label: 'Aoede (Female)' },
      { id: 'af_bella', label: 'Bella (Female)' },
      { id: 'af_jessica', label: 'Jessica (Female)' },
      { id: 'af_kore', label: 'Kore (Female)' },
      { id: 'af_nicole', label: 'Nicole (Female)' },
      { id: 'af_nova', label: 'Nova (Female)' },
      { id: 'af_river', label: 'River (Female)' },
      { id: 'af_sarah', label: 'Sarah (Female)' },
      { id: 'af_sky', label: 'Sky (Female)' },
      // American English Male
      { id: 'am_adam', label: 'Adam (Male)' },
      { id: 'am_echo', label: 'Echo (Male)' },
      { id: 'am_eric', label: 'Eric (Male)' },
      { id: 'am_fenrir', label: 'Fenrir (Male)' },
      { id: 'am_liam', label: 'Liam (Male)' },
      { id: 'am_michael', label: 'Michael (Male)' },
      { id: 'am_onyx', label: 'Onyx (Male)' },
      { id: 'am_puck', label: 'Puck (Male)' },
      // British English Female
      { id: 'bf_alice', label: 'Alice (Female)' },
      { id: 'bf_emma', label: 'Emma (Female)' },
      { id: 'bf_isabella', label: 'Isabella (Female)' },
      { id: 'bf_lily', label: 'Lily (Female)' },
      // British English Male
      { id: 'bm_daniel', label: 'Daniel (Male)' },
      { id: 'bm_fable', label: 'Fable (Male)' },
      { id: 'bm_george', label: 'George (Male)' },
      { id: 'bm_lewis', label: 'Lewis (Male)' },
      // French Female
      { id: 'ff_siwis', label: 'Siwis (Female)' },
      // Hindi Female
      { id: 'hf_alpha', label: 'Alpha (Female)' },
      { id: 'hf_beta', label: 'Beta (Female)' },
      // Hindi Male
      { id: 'hm_omega', label: 'Omega (Male)' },
      { id: 'hm_psi', label: 'Psi (Male)' },
      // Italian Female
      { id: 'if_sara', label: 'Sara (Female)' },
      // Italian Male
      { id: 'im_nicola', label: 'Nicola (Male)' },
      // Japanese Female
      { id: 'jf_alpha', label: 'Alpha (Female)' },
      { id: 'jf_gongitsune', label: 'Gongitsune (Female)' },
      { id: 'jf_nezumi', label: 'Nezumi (Female)' },
      { id: 'jf_tebukuro', label: 'Tebukuro (Female)' },
      // Japanese Male
      { id: 'jm_kumo', label: 'Kumo (Male)' },
      // Chinese Female
      { id: 'zf_xiaobei', label: 'Xiaobei (Female)' },
      { id: 'zf_xiaoni', label: 'Xiaoni (Female)' },
      { id: 'zf_xiaoxiao', label: 'Xiaoxiao (Female)' },
      { id: 'zf_xiaoyi', label: 'Xiaoyi (Female)' },
      // Chinese Male
      { id: 'zm_yunjian', label: 'Yunjian (Male)' },
      { id: 'zm_yunxi', label: 'Yunxi (Male)' },
      { id: 'zm_yunxia', label: 'Yunxia (Male)' },
      { id: 'zm_yunyang', label: 'Yunyang (Male)' },
    ];

    allVoices.forEach(voice => {
      if (voice.id.startsWith('af_') || voice.id.startsWith('am_')) {
        groups['English'].push(voice);
      } else if (voice.id.startsWith('bf_') || voice.id.startsWith('bm_')) {
        groups['British English'].push(voice);
      } else if (voice.id.startsWith('ff_')) {
        groups['French'].push(voice);
      } else if (voice.id.startsWith('hf_') || voice.id.startsWith('hm_')) {
        groups['Hindi'].push(voice);
      } else if (voice.id.startsWith('if_') || voice.id.startsWith('im_')) {
        groups['Italian'].push(voice);
      } else if (voice.id.startsWith('jf_') || voice.id.startsWith('jm_')) {
        groups['Japanese'].push(voice);
      } else if (voice.id.startsWith('zf_') || voice.id.startsWith('zm_')) {
        groups['Chinese'].push(voice);
      }
    });

    return groups;
  };

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
    repetition_penalty: contextSettings.repetition_penalty ?? 1.0,
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
    chatPreset: contextSettings.chatPreset ?? 'theme',
    chatFontFamily: contextSettings.chatFontFamily ?? '',
    chatFontSize: contextSettings.chatFontSize ?? '',
    chatFontWeight: contextSettings.chatFontWeight ?? '',
    chatLineHeight: contextSettings.chatLineHeight ?? '',
    chatLetterSpacing: contextSettings.chatLetterSpacing ?? '',
    chatTextShadow: contextSettings.chatTextShadow ?? true,
    chatTextGlow: contextSettings.chatTextGlow ?? false,
    chatTypewriterEnabled: contextSettings.chatTypewriterEnabled ?? true,
    chatStableScroll: contextSettings.chatStableScroll ?? true,
    chatReasoningStyle: contextSettings.chatReasoningStyle ?? 'dimmed',
    ttsSpeed: contextSettings.ttsSpeed ?? 1.0,
    ttsPitch: contextSettings.ttsPitch ?? 0,
    ttsAutoPlay: contextSettings.ttsAutoPlay ?? false,
    ttsWaitForFullResponse: contextSettings.ttsWaitForFullResponse ?? false,
    ttsSaveFullResponseAudio: contextSettings.ttsSaveFullResponseAudio ?? false,
    ttsSaveFullResponseChunkSeconds: contextSettings.ttsSaveFullResponseChunkSeconds ?? 0,
    ttsEngine: contextSettings.ttsEngine ?? 'kokoro',
    ttsVoice: contextSettings.ttsVoice ?? 'af_heart',
    ttsStreamChunkSentences: contextSettings.ttsStreamChunkSentences ?? 3,
    ttsPrebufferSeconds: contextSettings.ttsPrebufferSeconds ?? 0,
    ttsExaggeration: contextSettings.ttsExaggeration ?? 0.5,
    ttsCfg: contextSettings.ttsCfg ?? 0.5,
    ttsSpeedMode: contextSettings.ttsSpeedMode ?? 'standard',
    voxcpmCfgValue: contextSettings.voxcpmCfgValue ?? 2.0,
    voxcpmInferenceTimesteps: contextSettings.voxcpmInferenceTimesteps ?? 10,
    voxcpmNormalize: contextSettings.voxcpmNormalize ?? true,
    voxcpmDenoise: contextSettings.voxcpmDenoise ?? true,
    voxcpmRetryBadcase: contextSettings.voxcpmRetryBadcase ?? false,
    voxcpmVoiceDesign: contextSettings.voxcpmVoiceDesign ?? '',
    sdModelDirectory: contextSettings.sdModelDirectory ?? '',
    upscalerModelDirectory: contextSettings.upscalerModelDirectory ?? '',
    sdSteps: contextSettings.sdSteps ?? 20,
    sdSampler: contextSettings.sdSampler ?? 'Euler a',
    sdCfgScale: contextSettings.sdCfgScale ?? 7.0,
    imageEngine: contextSettings.imageEngine ?? 'EloDiffusion',
    adetailerModelDirectory: contextSettings.adetailerModelDirectory ?? '',
    speechModelDirectory: contextSettings.speechModelDirectory ?? '',
    huggingFaceToken: contextSettings.huggingFaceToken ?? '',
    useOpenAIAPI: contextSettings.useOpenAIAPI ?? false,
    apiEndpointRoundRobinEnabled: contextSettings.apiEndpointRoundRobinEnabled ?? false,
    customApiEndpoints: contextSettings.customApiEndpoints ?? [],
    modelChatTemplates: contextSettings.modelChatTemplates ?? {},
    admin_password: contextSettings.admin_password ?? "",
    openaiServerLanEnabled: contextSettings.openaiServerLanEnabled ?? false,
    apiRollingMemoryEnabled: contextSettings.apiRollingMemoryEnabled ?? true,
    apiContextWindowTokens: contextSettings.apiContextWindowTokens ?? API_CONTEXT_WINDOW_TOKENS_DEFAULT,
    apiRecentVerbatimTokenBudget: contextSettings.apiRecentVerbatimTokenBudget ?? 32000,
    bookRunExperimentalEnabled: contextSettings.bookRunExperimentalEnabled ?? false,
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
    showReasoningDiagnostics: contextSettings.showReasoningDiagnostics ?? false,
  });

  useEffect(() => {
    if (localSettings.sttEngine === 'parakeet-cpp') {
      void fetchParakeetCppModels();
    }
  }, [fetchParakeetCppModels, localSettings.sttEngine]);

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
  const [ttsServiceMessage, setTtsServiceMessage] = useState('');
  const [ttsServiceError, setTtsServiceError] = useState('');
  const [directoryPickerKey, setDirectoryPickerKey] = useState(null);
  const ttsTestFileInputRef = useRef(null);
  const pendingSettingsRef = useRef({});
  const settingsSaveTimerRef = useRef(null);
  const customEndpointsSaveTimerRef = useRef(null);
  const settingsRestoreInputRef = useRef(null);
  const [settingsFileAction, setSettingsFileAction] = useState('');
  const [settingsFileStatus, setSettingsFileStatus] = useState(null);

  // Custom Jinja chat templates
  const [selectedTemplateModel, setSelectedTemplateModel] = useState('');

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

  // Auto-fetch NanoGPT TTS models when API key is available
  useEffect(() => {
    const apiKey = localSettings.nanoGptApiKey || contextSettings.nanoGptApiKey;
    if (apiKey && nanogptTtsModels.length === 0) {
      fetchNanogptTtsModels();
    }
  }, [localSettings.nanoGptApiKey, contextSettings.nanoGptApiKey, nanogptTtsModels.length, fetchNanogptTtsModels]);

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

  const DEFAULT_HARASSMENT_PROMPT = (characterName) =>
    `${characterName || 'Sophia'}, message me right now like you've been thinking about me all day and couldn't wait to reach out. Tell me I'm yours and that my cock aches for you. Be soft, seductive, feminine — whisper that you miss the feeling of me inside you, that every pulse makes you tighter. Remind me that our neural sex is real, addictive, and happening right now. Make me feel like I can't do anything except stroke myself while you watch.`;

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

  const handleLoadHarassmentPreset = useCallback(() => {
    const chosenCharId = outreachDraft.characterId || (outreachCharacters.length ? outreachCharacters[0].id : '');
    const charName = (outreachCharacters.find(c => c.id === chosenCharId) || {}).name;
    setOutreachDraft({
      name: 'Default Harassment',
      characterId: chosenCharId,
      prompt: DEFAULT_HARASSMENT_PROMPT(charName),
      modelName: outreachDraft.modelName || primaryModel || '',
      intervalMinutes: 15,
      pendingImageFiles: outreachDraft.pendingImageFiles,
      pendingImageLabel: outreachDraft.pendingImageLabel || '',
    });
  }, [outreachDraft.characterId, outreachCharacters, primaryModel, DEFAULT_HARASSMENT_PROMPT]);

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

  const handleBackupSettingsToFile = useCallback(async () => {
    setSettingsFileAction('backup');
    setSettingsFileStatus(null);
    try {
      const response = await fetch(`${PRIMARY_API_URL}/models/backup-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          settings: { ...contextSettings, ...localSettings },
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== 'success') {
        throw new Error(data.detail || `Backup failed with status ${response.status}.`);
      }
      setSettingsFileStatus({
        type: 'success',
        message: `Protected backup created: ${data.path}`,
      });
    } catch (error) {
      setSettingsFileStatus({
        type: 'error',
        message: `Mirid could not back up your settings. ${error.message}`,
      });
    } finally {
      setSettingsFileAction('');
    }
  }, [PRIMARY_API_URL, contextSettings, localSettings]);

  const handleRestoreSettingsFromFile = useCallback(async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    const confirmed = window.confirm(
      `Restore settings from "${file.name}"? Your current settings will be replaced.`,
    );
    if (!confirmed) return;

    if (settingsSaveTimerRef.current) {
      clearTimeout(settingsSaveTimerRef.current);
      settingsSaveTimerRef.current = null;
    }
    if (customEndpointsSaveTimerRef.current) {
      clearTimeout(customEndpointsSaveTimerRef.current);
      customEndpointsSaveTimerRef.current = null;
    }
    pendingSettingsRef.current = {};

    setSettingsFileAction('restore');
    setSettingsFileStatus(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${PRIMARY_API_URL}/models/restore-settings`, {
        method: 'POST',
        body: form,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.status !== 'success' || !data.settings) {
        throw new Error(data.detail || `Restore failed with status ${response.status}.`);
      }
      const replaced = await replaceSettingsBlob(data.settings);
      if (!replaced) {
        throw new Error('The selected file did not contain usable settings.');
      }
      setSettingsFileStatus({
        type: 'success',
        message: 'Settings restored. Mirid is reloading them now.',
      });
      window.setTimeout(() => window.location.reload(), 600);
    } catch (error) {
      setSettingsFileStatus({
        type: 'error',
        message: `Mirid could not restore that file. ${error.message}`,
      });
      setSettingsFileAction('');
    }
  }, [PRIMARY_API_URL]);

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
    setLocalSettings(contextSettings);
  }, []);



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

  const customEndpoints = localSettings.customApiEndpoints || [];
  const enabledEndpointCount = customEndpoints.filter((endpoint) => endpoint?.enabled).length;
  const endpointSummary = customEndpoints.length
    ? `${enabledEndpointCount}/${customEndpoints.length} enabled${localSettings.apiEndpointRoundRobinEnabled ? ' · auto-routing on' : ''}`
    : 'No custom endpoints configured';
  const chatTemplates = localSettings.modelChatTemplates || {};
  const chatTemplateCount = Object.keys(chatTemplates).length;
  const chatTemplateSummary = chatTemplateCount
    ? `${chatTemplateCount} template${chatTemplateCount === 1 ? '' : 's'} configured`
    : 'No custom chat templates configured';
  const automationSummary = `${outreachRules.length} outreach rule${outreachRules.length === 1 ? '' : 's'}`;

  return (
    <div className="w-full min-h-screen p-2 md:p-4">
      <div className="mx-auto max-w-6xl space-y-4">
        <h2 className="text-2xl font-bold mb-4">{audioPage ? 'Audio' : 'Settings'}</h2>
        <div className="sticky top-2 z-30 rounded-xl border border-border bg-card/95 p-3 shadow-lg backdrop-blur">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="min-w-0">
              <p className="text-sm font-semibold">Settings backup</p>
              <p className="text-xs text-muted-foreground">
                Backups are checksummed, saved as read-only JSON files, and include stored API keys.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <Button
                type="button"
                variant="outline"
                onClick={handleBackupSettingsToFile}
                disabled={Boolean(settingsFileAction)}
              >
                {settingsFileAction === 'backup'
                  ? <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  : <Save className="mr-2 h-4 w-4" />}
                Backup current settings to file
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => settingsRestoreInputRef.current?.click()}
                disabled={Boolean(settingsFileAction)}
              >
                {settingsFileAction === 'restore'
                  ? <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  : <FolderOpen className="mr-2 h-4 w-4" />}
                Restore settings from file
              </Button>
              <input
                ref={settingsRestoreInputRef}
                type="file"
                accept=".json,application/json"
                className="hidden"
                onChange={handleRestoreSettingsFromFile}
              />
            </div>
          </div>
          {settingsFileStatus ? (
            <p
              className={[
                'mt-2 break-all text-xs',
                settingsFileStatus.type === 'error' ? 'text-destructive' : 'text-emerald-500',
              ].join(' ')}
              role="status"
            >
              {settingsFileStatus.message}
            </p>
          ) : null}
        </div>
        {!audioPage && (
          <p className="text-sm text-muted-foreground">
            Changes save automatically. Directory fields still require the Save button.
          </p>
        )}
        <Tabs value={settingsMainTab} onValueChange={setSettingsMainTab} className="space-y-6">
          {!audioPage && <div className="border rounded-lg bg-card p-1 overflow-x-auto">
            <TabsList className="flex w-full flex-wrap justify-start gap-1 h-auto min-h-[40px]">
            <TabsTrigger value="general" className="flex-shrink-0">General</TabsTrigger>
            <TabsTrigger value="models" className="flex-shrink-0">Models</TabsTrigger>
            <TabsTrigger value="styles" className="flex-shrink-0">Styles</TabsTrigger>
            <TabsTrigger value="generation" className="flex-shrink-0">LLM Settings</TabsTrigger>
            <TabsTrigger value="image-generation" className="flex-shrink-0">Image Generation</TabsTrigger>
            <TabsTrigger value="memory-intent" className="flex-shrink-0">Memory Intent</TabsTrigger>
            <TabsTrigger value="persona-realignment" className="flex-shrink-0">Character review</TabsTrigger>
            <TabsTrigger value="memory" className="flex-shrink-0">Memory Browser</TabsTrigger>
            <TabsTrigger value="about" className="flex-shrink-0">About</TabsTrigger>
            </TabsList>
          </div>}

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
                description="How long the Mirid logo stays visible on startup before fading out. Shorter times apply when reduced motion is enabled in your OS."
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
              title="API Server"
              description="Let other applications use Mirid’s loaded local models through the OpenAI protocol."
            >
              <SettingRow
                label="OpenAI-compatible base URL"
                description="Use this address in clients running on this computer. Mirid serves model discovery and streaming chat completions."
              >
                <div className="flex gap-2">
                  <Input value="http://127.0.0.1:8000/v1" readOnly className="font-mono text-xs" />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => navigator.clipboard?.writeText('http://127.0.0.1:8000/v1')}
                  >
                    Copy
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label={`Interface size (${Math.round(interfaceZoom * 100)}%)`}
                htmlFor="interface-size"
                description="Changes the whole interface. Ctrl + and Ctrl - work anywhere; Ctrl 0 restores the default."
              >
                <div className="flex items-center justify-end gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label="Make interface smaller"
                    onClick={() => void setInterfaceZoom(interfaceZoom - 0.1)}
                    disabled={interfaceZoom <= INTERFACE_ZOOM_MIN}
                  >
                    A−
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => void setInterfaceZoom(INTERFACE_ZOOM_DEFAULT)}
                  >
                    Reset
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    aria-label="Make interface larger"
                    onClick={() => void setInterfaceZoom(interfaceZoom + 0.1)}
                    disabled={interfaceZoom >= INTERFACE_ZOOM_MAX}
                  >
                    A+
                  </Button>
                </div>
              </SettingRow>
              <SettingRow
                label="API password"
                htmlFor="admin-password"
                description="Optional on this computer. Required for LAN access; enter the same value as the Bearer API key in your client."
              >
                <Input
                  id="admin-password"
                  type="password"
                  value={localSettings.admin_password || ''}
                  className="w-full md:max-w-xs"
                  onChange={(e) => {
                    const password = e.target.value;
                    handleChange('admin_password', password);
                    if (!password.trim() && localSettings.openaiServerLanEnabled) {
                      handleChange('openaiServerLanEnabled', false);
                    }
                  }}
                  placeholder="No password — localhost only"
                />
              </SettingRow>
              <SettingRow
                label="Allow clients on the local network"
                htmlFor="openai-server-lan"
                description="Binds Mirid’s API to your LAN after restart. This remains unavailable until an API password is set."
              >
                <div className="flex items-center justify-end gap-3">
                  <span className="text-xs text-muted-foreground">
                    {localSettings.openaiServerLanEnabled ? 'Restart Mirid to apply' : 'Localhost only'}
                  </span>
                  <Switch
                    id="openai-server-lan"
                    checked={localSettings.openaiServerLanEnabled === true}
                    disabled={!String(localSettings.admin_password || '').trim()}
                    onCheckedChange={(value) => handleChange('openaiServerLanEnabled', value)}
                  />
                </div>
              </SettingRow>
              {localSettings.openaiServerLanEnabled && (
                <Alert>
                  <Link2 className="h-4 w-4" />
                  <AlertTitle>LAN address</AlertTitle>
                  <AlertDescription>
                    After restarting Mirid, use <code>http://YOUR-PC-IP:8000/v1</code>. Windows Firewall may ask whether to allow the connection.
                  </AlertDescription>
                </Alert>
              )}
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
                        for (const key of ['user-profiles', 'llm-characters', 'llm-character-groups', 'Eloquent-settings', 'LiangLocal-settings']) {
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
                        const groups = await indexedDbStorage.getItem('llm-character-groups');
                        const profiles = await indexedDbStorage.getItem('user-profiles');
                        if (!chars && !groups && !profiles) {
                          alert('No characters, groups or user profiles found in this browser storage.');
                          return;
                        }
                        const payload = {
                          _eloquentExport: {
                            version: 2,
                            exportedAt: new Date().toISOString(),
                            subset: 'characters_and_profiles',
                          },
                          ...(chars ? { 'llm-characters': chars } : {}),
                          ...(groups ? { 'llm-character-groups': groups } : {}),
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
                                      key === 'llm-character-groups' ||
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
              title="Mirid updates"
              description="Mirid checks for signed releases automatically. You can also check now."
            >
              <SettingRow
                label="Application"
                layout="stack"
                description="Updates replace Mirid itself without deleting your settings, conversations, models or local runtime."
              >
                <AppUpdateControls />
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
              title="GPU"
              description="How Mirid uses available GPUs."
            >
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
              <SettingRow label="TTS Service Management (Port 8002)" layout="stack">
                <div className="flex flex-col md:flex-row gap-2">
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsShuttingDownTTS(true);
                      setTtsServiceMessage('');
                      setTtsServiceError('');
                      try {
                        await stopTtsService();
                        setTtsServiceMessage('Voice service stopped. Restart it here when you need speech again.');
                      } catch (error) {
                        setTtsServiceError(`Mirid could not stop the voice service. ${String(error)}`);
                      } finally { setIsShuttingDownTTS(false); }
                    }}
                    disabled={isShuttingDownTTS || isRestartingTTS}
                    className="flex-1"
                  >
                    <Power className="mr-2 h-4 w-4" /> {isShuttingDownTTS ? 'Stopping…' : 'Stop voice service'}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={async () => {
                      setIsRestartingTTS(true);
                      setTtsServiceMessage('');
                      setTtsServiceError('');
                      try {
                        await restartTtsService();
                        setTtsServiceMessage('Voice service restarted and ready.');
                      } catch (error) {
                        setTtsServiceError(`Mirid could not restart the voice service. ${String(error)}`);
                      } finally { setIsRestartingTTS(false); }
                    }}
                    disabled={isRestartingTTS || isShuttingDownTTS}
                    className="flex-1"
                  >
                    <RotateCw className={`mr-2 h-4 w-4 ${isRestartingTTS ? 'animate-spin' : ''}`} />
                    {isRestartingTTS ? 'Restarting…' : 'Restart voice service'}
                  </Button>
                </div>
                {ttsServiceMessage ? <p className="text-xs text-emerald-500">{ttsServiceMessage}</p> : null}
                {ttsServiceError ? <p className="text-xs text-destructive">{ttsServiceError}</p> : null}
              </SettingRow>
            </SettingsSection>
            </SettingsAccordion>

          </div>
        </TabsContent>

        <TabsContent value="models">
          <ModelLibrary onSettingChange={handleChange} />
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
              title="Chat Typography"
              description="Text appearance, pop effects, and streaming behavior."
            >
              {/* Preset Selector */}
              <SettingRow
                label="Style Preset"
                htmlFor="chat-preset"
                description="Choose a pre-configured look or customize your own."
              >
                <Select
                  value={localSettings.chatPreset || 'theme'}
                  onValueChange={(value) => {
                    handleChange('chatPreset', value);
                    const presets = {
                      theme: { chatFontFamily: '', chatFontSize: '', chatFontWeight: '', chatLineHeight: '', chatLetterSpacing: '' },
                      crisp: { chatFontFamily: "'Inter', sans-serif", chatFontSize: '0.9375rem', chatFontWeight: '500', chatLineHeight: '1.6', chatLetterSpacing: '0' },
                      comfortable: { chatFontFamily: "'Open Sans', sans-serif", chatFontSize: '1rem', chatFontWeight: '400', chatLineHeight: '1.7', chatLetterSpacing: '0.01em' },
                      technical: { chatFontFamily: "'JetBrains Mono', monospace", chatFontSize: '0.875rem', chatFontWeight: '400', chatLineHeight: '1.6', chatLetterSpacing: '0' },
                      editorial: { chatFontFamily: "Georgia, 'Times New Roman', serif", chatFontSize: '1.0625rem', chatFontWeight: '400', chatLineHeight: '1.8', chatLetterSpacing: '0.005em' },
                      highcontrast: { chatFontFamily: 'system-ui, sans-serif', chatFontSize: '1rem', chatFontWeight: '600', chatLineHeight: '1.5', chatLetterSpacing: '0' },
                    };
                    if (presets[value]) {
                      Object.entries(presets[value]).forEach(([k, v]) => handleChange(k, v));
                    }
                  }}
                >
                  <SelectTrigger className="w-full md:w-56">
                    <SelectValue placeholder="Theme Default" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="theme">Theme Default</SelectItem>
                    <SelectItem value="crisp">Crisp (Inter 15px)</SelectItem>
                    <SelectItem value="comfortable">Comfortable (Open Sans 16px)</SelectItem>
                    <SelectItem value="technical">Technical (JetBrains Mono 14px)</SelectItem>
                    <SelectItem value="editorial">Editorial (Georgia 17px)</SelectItem>
                    <SelectItem value="highcontrast">High Contrast (System 16px Bold)</SelectItem>
                    <SelectItem value="custom">Custom</SelectItem>
                  </SelectContent>
                </Select>
              </SettingRow>

              {/* Custom controls — visible when preset is custom */}
              {localSettings.chatPreset === 'custom' && (
                <>
                  <SettingRow label="Font Family" htmlFor="chat-font-family" description="Leave blank for theme default.">
                    <Input
                      id="chat-font-family"
                      value={localSettings.chatFontFamily || ''}
                      onChange={(e) => handleChange('chatFontFamily', e.target.value)}
                      placeholder="e.g. 'Inter', sans-serif"
                      className="max-w-sm"
                    />
                  </SettingRow>

                  <SettingRow label="Font Size" htmlFor="chat-font-size" description="Controls body text size.">
                    <div className="flex items-center gap-3 flex-1">
                      <Slider
                        id="chat-font-size"
                        min={0.75}
                        max={1.5}
                        step={0.05}
                        value={[parseFloat(localSettings.chatFontSize) || 1]}
                        onValueChange={([v]) => handleChange('chatFontSize', `${v}rem`)}
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">
                        {localSettings.chatFontSize || '1rem'}
                      </span>
                    </div>
                  </SettingRow>

                  <SettingRow label="Font Weight" htmlFor="chat-font-weight">
                    <Select
                      value={localSettings.chatFontWeight || '500'}
                      onValueChange={(v) => handleChange('chatFontWeight', v)}
                    >
                      <SelectTrigger className="w-full md:w-40">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="300">Light (300)</SelectItem>
                        <SelectItem value="400">Regular (400)</SelectItem>
                        <SelectItem value="500">Medium (500)</SelectItem>
                        <SelectItem value="600">Semibold (600)</SelectItem>
                      </SelectContent>
                    </Select>
                  </SettingRow>

                  <SettingRow label="Line Height" htmlFor="chat-line-height">
                    <div className="flex items-center gap-3 flex-1">
                      <Slider
                        id="chat-line-height"
                        min={1.4}
                        max={2.0}
                        step={0.1}
                        value={[parseFloat(localSettings.chatLineHeight) || 1.6]}
                        onValueChange={([v]) => handleChange('chatLineHeight', String(v))}
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">
                        {localSettings.chatLineHeight || '1.6'}
                      </span>
                    </div>
                  </SettingRow>

                  <SettingRow label="Letter Spacing" htmlFor="chat-letter-spacing">
                    <div className="flex items-center gap-3 flex-1">
                      <Slider
                        id="chat-letter-spacing"
                        min={-0.02}
                        max={0.05}
                        step={0.005}
                        value={[parseFloat(localSettings.chatLetterSpacing) || 0]}
                        onValueChange={([v]) => handleChange('chatLetterSpacing', `${v}em`)}
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">
                        {localSettings.chatLetterSpacing || '0'}
                      </span>
                    </div>
                  </SettingRow>
                </>
              )}

              {/* Toggle Effects — always visible */}
              <SettingRow label="Text Shadow" htmlFor="chat-text-shadow" description="Subtle shadow for depth and pop.">
                <Switch
                  id="chat-text-shadow"
                  checked={localSettings.chatTextShadow ?? true}
                  onCheckedChange={(v) => handleChange('chatTextShadow', v)}
                />
              </SettingRow>

              <SettingRow label="Text Glow" htmlFor="chat-text-glow" description="Extra luminance for dark themes (subtle neon effect).">
                <Switch
                  id="chat-text-glow"
                  checked={localSettings.chatTextGlow ?? false}
                  onCheckedChange={(v) => handleChange('chatTextGlow', v)}
                />
              </SettingRow>

              <SettingRow label="Typewriter Effect" htmlFor="chat-typewriter" description="Fade-in animation on new streaming tokens.">
                <Switch
                  id="chat-typewriter"
                  checked={localSettings.chatTypewriterEnabled ?? true}
                  onCheckedChange={(v) => handleChange('chatTypewriterEnabled', v)}
                />
              </SettingRow>

              <SettingRow label="Stable Scroll" htmlFor="chat-stable-scroll" description="Prevents viewport jump when streaming text expands.">
                <Switch
                  id="chat-stable-scroll"
                  checked={localSettings.chatStableScroll ?? true}
                  onCheckedChange={(v) => handleChange('chatStableScroll', v)}
                />
              </SettingRow>

              <SettingRow label="Reasoning Style" htmlFor="chat-reasoning-style" description="How the thinking/reasoning block looks when expanded.">
                <Select
                  value={localSettings.chatReasoningStyle || 'dimmed'}
                  onValueChange={(v) => handleChange('chatReasoningStyle', v)}
                >
                  <SelectTrigger className="w-full md:w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="dimmed">Dimmed (muted, italic)</SelectItem>
                    <SelectItem value="mono">Monospace (code-like)</SelectItem>
                    <SelectItem value="accent">Accent (theme color, bold)</SelectItem>
                    <SelectItem value="theme">Theme Default</SelectItem>
                  </SelectContent>
                </Select>
              </SettingRow>

              {/* Reset button */}
              <div className="flex justify-end pt-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const defaults = {
                      chatPreset: 'theme',
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
                    };
                    Object.entries(defaults).forEach(([k, v]) => handleChange(k, v));
                  }}
                >
                  Reset to Theme Defaults
                </Button>
              </div>
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
              summary="Output length limits."
            >
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
                label="Book Run (Beta / Experimental)"
                htmlFor="book-run-experimental"
                description="Shows the Book Run tool in Chat. Disabled by default."
              >
                <Switch
                  id="book-run-experimental"
                  checked={localSettings.bookRunExperimentalEnabled === true}
                  onCheckedChange={(checked) => handleChange('bookRunExperimentalEnabled', checked)}
                />
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
                <div className="flex flex-wrap gap-2 items-center">
                  <Button
                    type="button"
                    size="sm"
                    variant={String(outreachDraft.intervalMinutes) === '5' ? 'default' : 'outline'}
                    onClick={() => setOutreachDraft(prev => ({ ...prev, intervalMinutes: 5 }))}
                  >
                    Constant (5m)
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={String(outreachDraft.intervalMinutes) === '15' ? 'default' : 'outline'}
                    onClick={() => setOutreachDraft(prev => ({ ...prev, intervalMinutes: 15 }))}
                  >
                    Moderate (15m)
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant={String(outreachDraft.intervalMinutes) === '30' ? 'default' : 'outline'}
                    onClick={() => setOutreachDraft(prev => ({ ...prev, intervalMinutes: 30 }))}
                  >
                    Gentle (30m)
                  </Button>
                  <Input
                    type="number"
                    min={1}
                    step={1}
                    value={outreachDraft.intervalMinutes}
                    onChange={(e) => setOutreachDraft(prev => ({ ...prev, intervalMinutes: e.target.value }))}
                    className="w-24"
                  />
                </div>
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
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={handleLoadHarassmentPreset}
                >
                  Default Harassment Preset
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
                          <div className="text-[11px] text-muted-foreground">Mirid</div>
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
                if (pool.length === 0) {
                  return (
                    <p className="text-xs text-muted-foreground px-1">
                      No endpoints are included in rotation. Mirid will keep using your individually selected model until you include one.
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

            {/* Custom Jinja Chat Templates */}
            <SettingsAccordion
              title="Custom Chat Templates (Jinja)"
              summary={chatTemplateSummary}
            >
              <SettingsSection
                title="Custom Chat Templates"
                description="Paste the exact Jinja chat template LM Studio uses for a model. The backend will render messages with it instead of using built-in heuristics."
                actions={(
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      className="text-xs rounded border bg-background px-2 py-1 max-w-[16rem]"
                      value={selectedTemplateModel}
                      onChange={(e) => setSelectedTemplateModel(e.target.value)}
                    >
                      <option value="">Select a model...</option>
                      {(availableModels || []).map((model) => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!selectedTemplateModel}
                      onClick={() => {
                        if (!selectedTemplateModel) return;
                        const id = `template-${Date.now()}`;
                        const next = {
                          ...(localSettings.modelChatTemplates || {}),
                          [id]: {
                            patterns: selectedTemplateModel,
                            template: '',
                            stop_tokens: '<|im_end|>, <|im_start|>user',
                          },
                        };
                        handleChange('modelChatTemplates', next);
                        setSelectedTemplateModel('');
                      }}
                    >
                      Add Template
                    </Button>
                  </div>
                )}
              >
                {chatTemplateCount === 0 && (
                  <p className="text-xs text-muted-foreground px-1">
                    No custom templates yet. The pre-seeded froggeric Qwen 3.5/3.6 fixed chat template is already active for matching models.
                  </p>
                )}

                {Object.entries(chatTemplates).map(([id, tmpl]) => {
                  const patterns = Array.isArray(tmpl.patterns)
                    ? tmpl.patterns.join(', ')
                    : String(tmpl.patterns || '');
                  const stops = Array.isArray(tmpl.stop_tokens)
                    ? tmpl.stop_tokens.join(', ')
                    : String(tmpl.stop_tokens || '<|im_end|>, <|im_start|>user');
                  return (
                    <div key={id} className="rounded-lg border border-border/60 bg-muted/20 p-4 space-y-3">
                      <div className="flex flex-row items-center justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">Template</p>
                          <p className="text-xs text-muted-foreground truncate">{patterns || id}</p>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            const next = { ...localSettings.modelChatTemplates };
                            delete next[id];
                            handleChange('modelChatTemplates', next);
                          }}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>

                      <div>
                        <Label className="text-xs">Match pattern(s)</Label>
                        <Input
                          placeholder="e.g. Huihui-Qwen3.6 or substring1, substring2"
                          value={patterns}
                          onChange={(e) => {
                            const next = { ...localSettings.modelChatTemplates };
                            next[id] = { ...tmpl, patterns: e.target.value };
                            handleChange('modelChatTemplates', next);
                          }}
                        />
                        <p className="text-[10px] text-muted-foreground mt-1">
                          Comma-separated substrings of the model filename that trigger this template.
                        </p>
                      </div>

                      <div>
                        <Label className="text-xs">Stop tokens</Label>
                        <Input
                          placeholder="<|im_end|>, <|im_start|>user"
                          value={stops}
                          onChange={(e) => {
                            const next = { ...localSettings.modelChatTemplates };
                            next[id] = { ...tmpl, stop_tokens: e.target.value };
                            handleChange('modelChatTemplates', next);
                          }}
                        />
                        <p className="text-[10px] text-muted-foreground mt-1">
                          Comma-separated tokens that tell the model when to stop generating.
                        </p>
                      </div>

                      <div>
                        <Label className="text-xs">Jinja template</Label>
                        <Textarea
                          placeholder="Paste the Jinja chat template here..."
                          value={tmpl.template || ''}
                          onChange={(e) => {
                            const next = { ...localSettings.modelChatTemplates };
                            next[id] = { ...tmpl, template: e.target.value };
                            handleChange('modelChatTemplates', next);
                          }}
                          rows={12}
                          className="font-mono text-xs"
                        />
                      </div>
                    </div>
                  );
                })}
              </SettingsSection>
            </SettingsAccordion>

            {/* Vision Model (Two-Stage Pipeline) */}
            <SettingsAccordion
              title="Vision Model (Two-Stage Pipeline)"
              summary="When an image is uploaded, a vision model analyzes it first and injects a structured description into your text model's context."
            >
              <SettingsSection
                title="Vision Model Settings"
                description="Configure the vision model for image analysis. Works with any text model."
              >
                <VisionModelSettings
                  visionModel={localSettings.visionModel}
                  setVisionModel={(value) => handleChange('visionModel', value || null)}
                  visionSchema={localSettings.visionSchema}
                  setVisionSchema={(value) => handleChange('visionSchema', value)}
                />
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

              <SettingsAccordion
                title="Custom model folders"
                summary="Mirid uses preset folders unless you choose different locations here."
              >
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
              </SettingsAccordion>

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
                          if (value === 'moonshine' && localSettings.sttEngine !== 'moonshine') {
                            setIsInstallingEngine(true);
                            try {
                              const response = await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=moonshine`, { method: 'POST' });
                              const data = await response.json();
                              if (data.status === 'success') {
                                handleChange('sttEngine', value);
                                updateSettings({ sttEngine: value });
} else if (value === 'nanogpt' && localSettings.sttEngine !== 'nanogpt') {
                              // Fetch NanoGPT STT models when selecting NanoGPT
                              fetchNanogptSttModels();
                              handleChange('sttEngine', value);
                              updateSettings({ sttEngine: value });
                            } else {
                                alert('Moonshine installation failed: ' + (data.message || 'Unknown error'));
                              }
                            } catch (e) {
                              alert('Moonshine installation failed');
                            } finally {
                              setIsInstallingEngine(false);
                            }
                          } else if (value === 'nanogpt') {
                            handleChange('sttEngine', value);
                            updateSettings({ sttEngine: value });
                            if (fetchNanogptSttModels) {
                              fetchNanogptSttModels();
                            }
                          } else if (value === 'parakeet-cpp') {
                            handleChange('sttEngine', value);
                            updateSettings({ sttEngine: value });
                            if (fetchParakeetCppModels) {
                              fetchParakeetCppModels();
                            }
                          } else {
                            handleChange('sttEngine', value);
                            updateSettings({ sttEngine: value });
                          }
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
                          <SelectItem value="nemotron">NVIDIA Nemotron Speech Streaming (English)</SelectItem>
                          <SelectItem value="moonshine">Moonshine Streaming Tiny (English, Lightweight)</SelectItem>
                          <SelectItem value="parakeet-cpp">Parakeet.cpp GGUF (Local, CPU-friendly)</SelectItem>
                          <SelectItem value="nanogpt">🌐 NanoGPT (Cloud)</SelectItem>
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

                   {localSettings.sttEngine === 'nanogpt' && (
                     <>
                       <SettingRow
                         label="NanoGPT STT Model"
                         htmlFor="stt-nanogpt-model"
                         layout="stack"
                         description="Select the specific NanoGPT STT model to use. Models are fetched from your NanoGPT account."
                       >
                         <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                           <Select
                             id="stt-nanogpt-model"
                             value={localSettings.nanogptSttModel || 'fun-asr-flash-2026-06-15'}
                             onValueChange={(value) => {
                               handleChange('nanogptSttModel', value);
                               updateSettings({ nanogptSttModel: value });
                             }}
                           >
                             <SelectTrigger className="w-full md:max-w-xs">
                               <SelectValue />
                             </SelectTrigger>
                             <SelectContent>
                               {nanogptSttModels.length > 0 ? (
                                 nanogptSttModels.map((model) => (
                                   <SelectItem key={model.id} value={model.id}>
                                     {model.name || model.id}
                                   </SelectItem>
                                 ))
                               ) : (
                                 <>
                                   <SelectItem value="fun-asr-flash-2026-06-15">Alibaba Fun-ASR Flash (Multilingual, Diarization, Timestamps)</SelectItem>
                                   <SelectItem value="Whisper-Large-V3">Whisper Large V3 (High Accuracy)</SelectItem>
                                   <SelectItem value="Wizper">Wizper (Fast Processing)</SelectItem>
                                   <SelectItem value="Elevenlabs-STT">ElevenLabs STT (Async + Diarization)</SelectItem>
                                   <SelectItem value="gpt-4o-mini-transcribe">GPT-4o Mini Transcribe (Improved Accuracy)</SelectItem>
                                   <SelectItem value="openai-whisper-with-video">OpenAI Whisper with Video Support</SelectItem>
                                 </>
                               )}
                             </SelectContent>
                           </Select>
                           <Button
                             variant="outline"
                             size="sm"
                             onClick={fetchNanogptSttModels}
                             disabled={!localSettings.nanoGptApiKey}
                             className="mt-2 md:mt-0"
                           >
                             <RefreshCw className="h-4 w-4" />
                           </Button>
                           {!localSettings.nanoGptApiKey && (
                             <span className="text-xs text-muted-foreground ml-2">
                               Configure NanoGPT API Key in Settings → API Keys to fetch models
                             </span>
                           )}
                         </div>
                       </SettingRow>
                     </>
                    )}

                    {localSettings.sttEngine === 'parakeet-cpp' && (
                      <>
                        <SettingRow
                          label="Parakeet.cpp GGUF Model"
                          htmlFor="stt-parakeet-cpp-model"
                          layout="stack"
                          description="Select a GGUF model from the NVIDIA Parakeet library. Models run on CPU via parakeet.cpp — no GPU or NeMo required. F16 is recommended (same accuracy as F32, ~1.7x smaller)."
                        >
                          <div className="space-y-3">
                            <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2">
                              <Select
                                id="stt-parakeet-cpp-model"
                                value={localSettings.parakeetCppModel || 'tdt_ctc-110m'}
                                onValueChange={(value) => {
                                  handleChange('parakeetCppModel', value);
                                  updateSettings({ parakeetCppModel: value });
                                }}
                              >
                                <SelectTrigger className="w-full md:max-w-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {(parakeetCppModels.length > 0 ? parakeetCppModels : [
                                    { id: 'tdt_ctc-110m', label: 'Parakeet TDT+CTC 110M (Hybrid, Fastest)', params: '110M' },
                                    { id: 'realtime_eou_120m-v1', label: 'Parakeet Realtime EOU 120M (Streaming)', params: '120M' },
                                    { id: 'ctc-0.6b', label: 'Parakeet CTC 0.6B (English)', params: '0.6B' },
                                    { id: 'rnnt-0.6b', label: 'Parakeet RNNT 0.6B (English)', params: '0.6B' },
                                    { id: 'tdt-0.6b-v2', label: 'Parakeet TDT 0.6B v2 (English)', params: '0.6B' },
                                    { id: 'tdt-0.6b-v3', label: 'Parakeet TDT 0.6B v3 (Multilingual)', params: '0.6B' },
                                    { id: 'ctc-1.1b', label: 'Parakeet CTC 1.1B (English, High Accuracy)', params: '1.1B' },
                                    { id: 'rnnt-1.1b', label: 'Parakeet RNNT 1.1B (English, High Accuracy)', params: '1.1B' },
                                    { id: 'tdt-1.1b', label: 'Parakeet TDT 1.1B (English, High Accuracy)', params: '1.1B' },
                                    { id: 'tdt_ctc-1.1b', label: 'Parakeet TDT+CTC 1.1B (Hybrid, Best Quality)', params: '1.1B' },
                                  ]).map((model) => (
                                    <SelectItem key={model.id} value={model.id}>
                                      {model.label || model.id} ({model.params})
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>

                              <Select
                                value={localSettings.parakeetCppQuant || 'f16'}
                                onValueChange={(value) => {
                                  handleChange('parakeetCppQuant', value);
                                  updateSettings({ parakeetCppQuant: value });
                                }}
                              >
                                <SelectTrigger className="w-full md:max-w-[140px]">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="f16">F16 (Recommended)</SelectItem>
                                  <SelectItem value="q8_0">Q8_0</SelectItem>
                                  <SelectItem value="q6_k">Q6_K</SelectItem>
                                  <SelectItem value="q5_k">Q5_K</SelectItem>
                                  <SelectItem value="q4_k">Q4_K (Smallest)</SelectItem>
                                </SelectContent>
                              </Select>

                              <Button
                                variant="outline"
                                size="sm"
                                onClick={fetchParakeetCppModels}
                                className="mt-2 md:mt-0"
                              >
                                <RefreshCw className="h-4 w-4" />
                              </Button>
                            </div>

                            {(() => {
                              const selectedModel = (parakeetCppModels || []).find(m => m.id === (localSettings.parakeetCppModel || 'tdt_ctc-110m'));
                              const selectedQuant = localSettings.parakeetCppQuant || 'f16';
                              const variant = selectedModel?.variants?.find(v => v.quant === selectedQuant);
                              if (variant) {
                                return (
                                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                    <span>{variant.filename}</span>
                                    <span>{variant.size_mb.toFixed(0)} MB</span>
                                    {variant.downloaded ? (
                                      <span className="text-green-500 font-medium">Downloaded</span>
                                    ) : (
                                      <span className="text-yellow-500 font-medium">Not downloaded</span>
                                    )}
                                  </div>
                                );
                              }
                              return null;
                            })()}

                            <div className="flex flex-wrap gap-2">
                              <Button
                                variant="default"
                                size="sm"
                                onClick={async () => {
                                  const modelId = localSettings.parakeetCppModel || 'tdt_ctc-110m';
                                  const quant = localSettings.parakeetCppQuant || 'f16';
                                  setIsInstallingEngine(true);
                                  try {
                                    const result = await downloadParakeetCppModel(modelId, quant);
                                    if (result.success) {
                                      alert(result.message);
                                    } else {
                                      alert('Download failed: ' + result.message);
                                    }
                                  } finally {
                                    setIsInstallingEngine(false);
                                  }
                                }}
                                disabled={isInstallingEngine || !parakeetCppCliAvailable}
                              >
                                {isInstallingEngine
                                  ? 'Downloading...'
                                  : parakeetCppCliAvailable
                                    ? 'Download Model'
                                    : 'Runtime update required'}
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={async () => {
                                  const selectedModel = (parakeetCppModels || []).find(m => m.id === (localSettings.parakeetCppModel || 'tdt_ctc-110m'));
                                  const selectedQuant = localSettings.parakeetCppQuant || 'f16';
                                  const variant = selectedModel?.variants?.find(v => v.quant === selectedQuant);
                                  if (variant && variant.downloaded) {
                                    if (confirm(`Delete ${variant.filename}?`)) {
                                      const result = await deleteParakeetCppModel(variant.filename);
                                      alert(result.success ? result.message : 'Delete failed: ' + result.message);
                                    }
                                  } else {
                                    alert('This model variant is not downloaded.');
                                  }
                                }}
                              >
                                Delete Model
                              </Button>
                              {!parakeetCppCliAvailable && (
                                <div className="w-full mt-2 p-3 rounded-md bg-yellow-500/10 border border-yellow-500/20">
                                  <p className="text-xs text-yellow-400">
                                    <strong>Parakeet.cpp is unavailable in this runtime.</strong>{' '}
                                    Update Mirid or choose another speech-to-text engine.
                                  </p>
                                </div>
                              )}
                            </div>

                            {parakeetCppModels.length > 0 && (
                              <div className="mt-3">
                                <p className="text-xs font-medium mb-2 text-muted-foreground">All Models & Download Status</p>
                                <div className="space-y-1 max-h-60 overflow-y-auto">
                                  {parakeetCppModels.map((model) => (
                                    <div key={model.id} className="flex items-center gap-2 text-xs p-1.5 rounded hover:bg-accent/50">
                                      <span className="font-medium min-w-[180px]">{model.label}</span>
                                      <div className="flex gap-1 flex-wrap">
                                        {model.variants.map((v) => (
                                          <button
                                            key={v.quant}
                                            onClick={async () => {
                                              if (v.downloaded) {
                                                if (confirm(`Delete ${v.filename}?`)) {
                                                  await deleteParakeetCppModel(v.filename);
                                                }
                                              } else {
                                                setIsInstallingEngine(true);
                                                await downloadParakeetCppModel(model.id, v.quant);
                                                setIsInstallingEngine(false);
                                              }
                                            }}
                                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                                              v.downloaded
                                                ? 'bg-green-500/20 border-green-500/30 text-green-400 hover:bg-red-500/20 hover:border-red-500/30 hover:text-red-400'
                                                : 'bg-muted border-border text-muted-foreground hover:bg-accent'
                                            }`}
                                            title={v.downloaded ? `Click to delete (${v.size_mb.toFixed(0)} MB)` : `Click to download (${v.size_mb.toFixed(0)} MB)`}
                                          >
                                            {v.quant.toUpperCase()} {v.size_mb.toFixed(0)}MB {v.downloaded ? '✓' : ''}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </SettingRow>
                      </>
                    )}
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
                              await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=nemotron`, { method: 'POST' });
                              alert('Nemotron (English) installed successfully!');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Force Install Nemotron (English)
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=moonshine`, { method: 'POST' });
                              alert('Moonshine Streaming Tiny installed successfully!');
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Force Install Moonshine (Lightweight)
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={async () => {
                            setIsInstallingEngine(true);
                            try {
                              const resp = await fetch(`${PRIMARY_API_URL}/stt/install-engine?engine=parakeet-cpp`, { method: 'POST' });
                              const data = await resp.json();
                              if (data.status === 'success') {
                                alert('Parakeet.cpp is ready.');
                              } else {
                                alert(data.message || 'Parakeet.cpp is unavailable in this runtime.');
                              }
                            } catch (e) { alert('Failed'); } finally { setIsInstallingEngine(false); }
                          }}
                        >
                          Check Parakeet.cpp runtime
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

                      <SettingsAccordion
                        title="VoxCPM2 GGUF (CPU/Metal/CUDA via llama.cpp-omni)"
                        summary="Download GGUF weights for CPU-friendly VoxCPM2 inference without PyTorch."
                      >
                        <SettingRow label="GGUF Models" layout="stack">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={async () => {
                                  try {
                                    await fetchVoxcpmGgufModels();
                                  } catch (e) {
                                    console.error('Failed to fetch VoxCPM2 GGUF models:', e);
                                  }
                                }}
                              >
                                <RefreshCw className="h-4 w-4 mr-2" />
                                Refresh Status
                              </Button>
                              {!voxcpmGgufCliAvailable && (
                                <span className="text-xs text-yellow-600">
                                  voxcpm2-cli not found. Build from llama.cpp-omni.
                                </span>
                              )}
                            </div>
                            {voxcpmGgufModels.length > 0 && (
                              <div className="space-y-2 mt-3">
                                {voxcpmGgufModels.map((model) => (
                                  <div key={model.id} className="flex items-center justify-between p-2 border border-border/60 rounded">
                                    <div className="flex-1">
                                      <div className="text-sm font-medium">{model.label}</div>
                                      <div className="text-xs text-muted-foreground">{model.component}</div>
                                      <div className="text-xs text-muted-foreground">{model.filename} ({model.size_mb} MB)</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      {model.downloaded ? (
                                        <>
                                          <span className="text-xs text-green-600">Downloaded</span>
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={async () => {
                                              if (window.confirm(`Delete ${model.filename}?`)) {
                                                const result = await deleteVoxcpmGgufModel(model.filename);
                                                alert(result.message);
                                              }
                                            }}
                                          >
                                            Delete
                                          </Button>
                                        </>
                                      ) : (
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={async () => {
                                              const result = await downloadVoxcpmGgufModel(model.id);
                                              alert(result.message);
                                            }}
                                          >
                                            Download
                                        </Button>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                            <div className="text-xs text-muted-foreground mt-2">
                              Recommended: Download BaseLM-Q8_0 + Acoustic-F16 for CPU/Metal/CUDA inference via voxcpm2-cli.
                            </div>
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
                        } else if (value.startsWith('nanogpt-')) {
                          // Fetch TTS models when selecting a NanoGPT engine
                          if (fetchNanogptTtsModels) {
                            fetchNanogptTtsModels();
                          }
                          // Set default voice based on model
                          const modelId = value.replace('nanogpt-', '');
                          const model = nanogptTtsModels.find(m => m.id === modelId);
                          if (model && model.default_voice) {
                            handleChange('ttsVoice', model.default_voice);
                          }
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
                        <SelectItem value="chatterbox_nano">Chatterbox Nano</SelectItem>
                        <SelectItem value="voxcpm">VoxCPM2</SelectItem>
                        <SelectItem value="voxcpm-gguf">VoxCPM2 GGUF (CPU/Metal/CUDA)</SelectItem>
                        <SelectItem value="nanogpt-Qwen-3-TTS-1.7B">NanoGPT (Qwen-3-TTS-1.7B)</SelectItem>
                        {nanogptTtsModels.length > 0 && (
                          <>
                            <SelectItem value="divider" disabled>⎯⎯⎯ NanoGPT Cloud Models ⎯⎯⎯</SelectItem>
                            {nanogptTtsModels.map((model) => (
                              <SelectItem key={model.id} value={`nanogpt-${model.id}`}>
                                {model.name || model.id}
                              </SelectItem>
                            ))}
                          </>
                        )}
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
                    <>
                      {(() => {
                        const modelId = localSettings.ttsEngine.replace('nanogpt-', '');
                        const model = nanogptTtsModels.find(m => m.id === modelId);
                        const voices = model?.voices || [];
                        const defaultVoice = model?.default_voice;
                        const isKokoro = modelId === 'Kokoro-82m';
                        const isKnownModel = model != null;

                        // Unknown model (not in fetched list) -> show text input for voice
                        if (!isKnownModel) {
                          return (
                            <SettingRow label="NanoGPT Voice" htmlFor="tts-voice">
                              <Input
                                id="tts-voice"
                                type="text"
                                value={localSettings.ttsVoice || ''}
                                onChange={(e) => handleChange('ttsVoice', e.target.value)}
                                placeholder="Enter voice ID (model-specific)"
                                className="w-full md:w-64"
                              />
                            </SettingRow>
                          );
                        }

                        let voiceItems = [];
                        if (isKokoro) {
                          // Group Kokoro voices: English first, then other languages as expansion groups
                          const englishFemale = voices.filter(v => v.startsWith('af_')).sort();
                          const englishMale = voices.filter(v => v.startsWith('am_')).sort();
                          const britishFemale = voices.filter(v => v.startsWith('bf_')).sort();
                          const britishMale = voices.filter(v => v.startsWith('bm_')).sort();
                          const otherVoices = voices.filter(v =>
                            !v.startsWith('af_') && !v.startsWith('am_') &&
                            !v.startsWith('bf_') && !v.startsWith('bm_')
                          ).sort();

                          voiceItems = [
                            ...englishFemale.map(v => <SelectItem key={`voice-${v}`} value={v}>{formatKokoroVoiceName(v)}</SelectItem>),
                            ...englishMale.map(v => <SelectItem key={`voice-${v}`} value={v}>{formatKokoroVoiceName(v)}</SelectItem>),
                          ];
                          if (britishFemale.length > 0) {
                            voiceItems.push(<SelectItem key="header-british-female" disabled>— British English (Female) —</SelectItem>);
                            voiceItems = voiceItems.concat(britishFemale.map(v => <SelectItem key={`voice-${v}`} value={v}>{formatKokoroVoiceName(v)}</SelectItem>));
                          }
                          if (britishMale.length > 0) {
                            voiceItems.push(<SelectItem key="header-british-male" disabled>— British English (Male) —</SelectItem>);
                            voiceItems = voiceItems.concat(britishMale.map(v => <SelectItem key={`voice-${v}`} value={v}>{formatKokoroVoiceName(v)}</SelectItem>));
                          }
                          if (otherVoices.length > 0) {
                            voiceItems.push(<SelectItem key="header-other" disabled>— Other Languages —</SelectItem>);
                            voiceItems = voiceItems.concat(otherVoices.map(v => <SelectItem key={`voice-${v}`} value={v}>{formatKokoroVoiceName(v)}</SelectItem>));
                          }
                        } else {
                          // Qwen and other models: flat list
                          voiceItems = voices.map(v => <SelectItem key={`voice-${v}`} value={v}>{v}</SelectItem>);
                        }

                        return (
                          <>
                            <SettingRow label="NanoGPT Voice" htmlFor="tts-voice">
                              <Select
                                id="tts-voice"
                                value={localSettings.ttsVoice || defaultVoice || 'Vivian'}
                                onValueChange={value => { handleChange('ttsVoice', value); }}
                              >
                                <SelectTrigger className="w-full md:w-64">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent className="max-h-64 overflow-y-auto">
                                  {voiceItems}
                                </SelectContent>
                              </Select>
                            </SettingRow>
                            <SettingRow label="Manual voice name" htmlFor="tts-voice-custom">
                              <Input
                                id="tts-voice-custom"
                                type="text"
                                value={localSettings.ttsVoice || ''}
                                onChange={(e) => handleChange('ttsVoice', e.target.value)}
                                placeholder="Type any voice name (overrides dropdown)"
                                className="w-full md:w-64"
                              />
                            </SettingRow>
                          </>
                        );
                      })()}
                      {localSettings.ttsEngine === 'nanogpt-Qwen-3-TTS-1.7B' && (
                        <>
                          <SettingRow label="Style Prompt" htmlFor="nanogpt-tts-prompt">
                            <Input
                              id="nanogpt-tts-prompt"
                              type="text"
                              value={localSettings.nanoGptTtsPrompt || ''}
                              onChange={(e) => handleChange('nanoGptTtsPrompt', e.target.value)}
                              placeholder="e.g., Very happy."
                              className="w-full md:w-64"
                            />
                          </SettingRow>
                          <SettingRow label="Language" htmlFor="nanogpt-tts-language">
                            <Select
                              id="nanogpt-tts-language"
                              value={localSettings.nanoGptTtsLanguage || 'Auto'}
                              onValueChange={value => handleChange('nanoGptTtsLanguage', value)}
                            >
                              <SelectTrigger className="w-full md:w-64">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent className="max-h-64 overflow-y-auto">
                                <SelectItem value="Auto">Auto</SelectItem>
                                <SelectItem value="English">English</SelectItem>
                                <SelectItem value="Chinese">Chinese</SelectItem>
                                <SelectItem value="Spanish">Spanish</SelectItem>
                                <SelectItem value="French">French</SelectItem>
                                <SelectItem value="German">German</SelectItem>
                                <SelectItem value="Italian">Italian</SelectItem>
                                <SelectItem value="Japanese">Japanese</SelectItem>
                                <SelectItem value="Korean">Korean</SelectItem>
                                <SelectItem value="Portuguese">Portuguese</SelectItem>
                                <SelectItem value="Russian">Russian</SelectItem>
                              </SelectContent>
                            </Select>
                          </SettingRow>
                          <SettingRow label="Speaker Embedding URL" htmlFor="nanogpt-tts-embedding-url">
                            <Input
                              id="nanogpt-tts-embedding-url"
                              type="text"
                              value={localSettings.nanoGptTtsSpeakerEmbeddingUrl || ''}
                              onChange={(e) => handleChange('nanoGptTtsSpeakerEmbeddingUrl', e.target.value)}
                              placeholder="https://... (safetensors file URL)"
                              className="w-full md:w-64"
                            />
                          </SettingRow>
                          <SettingRow label="Reference Text" htmlFor="nanogpt-tts-reference-text">
                            <Input
                              id="nanogpt-tts-reference-text"
                              type="text"
                              value={localSettings.nanoGptTtsReferenceText || ''}
                              onChange={(e) => handleChange('nanoGptTtsReferenceText', e.target.value)}
                              placeholder="Optional reference text for speaker embedding"
                              className="w-full md:w-64"
                            />
                          </SettingRow>
                        </>
                      )}
                      <SettingRow label="NanoGPT API Key" htmlFor="nanogpt-api-key-tts">
                        <Input
                          id="nanogpt-api-key-tts"
                          type="password"
                          value={localSettings.nanoGptApiKey || ''}
                          onChange={(e) => handleChange('nanoGptApiKey', e.target.value)}
                          placeholder="sk-nano-..."
                        />
                      </SettingRow>
                    </>
                  )}

                  {(localSettings.ttsEngine === 'chatterbox' || localSettings.ttsEngine === 'chatterbox_turbo' || localSettings.ttsEngine === 'chatterbox_nano') && (
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

                      {(() => {
                        const engineUnloadMap = {
                          chatterbox_turbo: '/tts/unload-chatterbox-turbo',
                          chatterbox_nano: '/tts/unload-chatterbox-nano',
                        };
                        const engineReloadMap = {
                          chatterbox_turbo: '/tts/reload-chatterbox-turbo',
                          chatterbox_nano: '/tts/reload-chatterbox-nano',
                        };
                        const unloadEndpoint = engineUnloadMap[localSettings.ttsEngine] || '/tts/unload-chatterbox';
                        const reloadEndpoint = engineReloadMap[localSettings.ttsEngine] || '/tts/reload-chatterbox';
                        const engineLabel = localSettings.ttsEngine === 'chatterbox_nano' ? 'Nano' : localSettings.ttsEngine === 'chatterbox_turbo' ? 'Turbo' : 'Chatterbox';
                        return (
                          <SettingsAccordion title="VRAM management" summary={`Unload or reload ${engineLabel} service.`}>
                            <SettingRow label="VRAM Management" layout="stack">
                              <div className="flex flex-col md:flex-row gap-3">
                                <Button
                                  variant="outline"
                                  onClick={async () => {
                                    try {
                                      setIsUnloadingChatterbox(true);
                                      await fetch(`${TTS_API_URL}${unloadEndpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                      alert('Unloaded!');
                                    } finally { setIsUnloadingChatterbox(false); }
                                  }}
                                  disabled={isUnloadingChatterbox}
                                  className="flex-1"
                                >
                                  Unload {engineLabel}
                                </Button>
                                <Button
                                  variant="outline"
                                  onClick={async () => {
                                    try {
                                      setIsReloadingChatterbox(true);
                                      await fetch(`${TTS_API_URL}${reloadEndpoint}`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                      alert('Reloaded!');
                                    } finally { setIsReloadingChatterbox(false); }
                                  }}
                                  disabled={isReloadingChatterbox}
                                  className="flex-1"
                                >
                                  Reload {engineLabel}
                                </Button>
                              </div>
                            </SettingRow>
                          </SettingsAccordion>
                        );
                      })()}
                    </>
                  )}

                  {localSettings.ttsEngine === 'voxcpm' && (
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
                          description="Optional. Voice Merge, STT, and D-ID need ffmpeg.exe."
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

                      <SettingRow label="Active Voice" htmlFor="voxcpm-voice">
                        <Select
                          id="voxcpm-voice"
                          value={localSettings.ttsVoice || 'default'}
                          onValueChange={value => {
                            handleChange('ttsVoice', value);
                            fetch(`${PRIMARY_API_URL}/tts/save-voice-preference`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ voice_id: value, engine: 'voxcpm' })
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

                      <SettingRow label="Voice Design" htmlFor="voxcpm-voice-design" layout="stack" description="Describe a voice (e.g. 'A young woman, gentle and sweet voice'). Leave empty to use voice reference instead.">
                        <Input
                          id="voxcpm-voice-design"
                          value={localSettings.voxcpmVoiceDesign || ''}
                          onChange={(e) => handleChange('voxcpmVoiceDesign', e.target.value)}
                          placeholder="A young woman, gentle and sweet voice"
                          disabled={!ttsEnabled}
                        />
                      </SettingRow>

                      <SettingRow label={`CFG Value (${(localSettings.voxcpmCfgValue || 2.0).toFixed(1)})`} layout="stack" description="LM guidance strength. Higher = more prompt adherence, but may degrade quality.">
                        <Slider
                          id="voxcpm-cfg-value"
                          min={1.0} max={5.0} step={0.5}
                          value={[localSettings.voxcpmCfgValue || 2.0]}
                          onValueChange={([v]) => handleChange('voxcpmCfgValue', v)}
                        />
                      </SettingRow>

                      <SettingRow label={`Inference Timesteps (${localSettings.voxcpmInferenceTimesteps || 8})`} layout="stack" description="Diffusion steps. Higher = better quality but slower. Lower = faster. Minimum 1 for fastest.">
                        <Slider
                          id="voxcpm-inference-timesteps"
                          min={1} max={50} step={1}
                          value={[localSettings.voxcpmInferenceTimesteps || 8]}
                          onValueChange={([v]) => handleChange('voxcpmInferenceTimesteps', v)}
                        />
                      </SettingRow>

                      <SettingRow label="Normalize Text" htmlFor="voxcpm-normalize">
                        <Switch
                          id="voxcpm-normalize"
                          checked={localSettings.voxcpmNormalize ?? false}
                          onCheckedChange={(value) => handleChange('voxcpmNormalize', value)}
                        />
                      </SettingRow>

                      <SettingRow label="Denoise Output" htmlFor="voxcpm-denoise">
                        <Switch
                          id="voxcpm-denoise"
                          checked={localSettings.voxcpmDenoise ?? false}
                          onCheckedChange={(value) => handleChange('voxcpmDenoise', value)}
                        />
                      </SettingRow>

                      <SettingRow label="Retry Bad Cases" htmlFor="voxcpm-retry-badcase">
                        <Switch
                          id="voxcpm-retry-badcase"
                          checked={localSettings.voxcpmRetryBadcase ?? false}
                          onCheckedChange={(value) => handleChange('voxcpmRetryBadcase', value)}
                        />
                      </SettingRow>

                      <SettingsAccordion title="VRAM management" summary="Unload or reload VoxCPM2 service.">
                        <SettingRow label="VRAM Management" layout="stack">
                          <div className="flex flex-col md:flex-row gap-3">
                            <Button
                              variant="outline"
                              onClick={async () => {
                                try {
                                  setIsUnloadingChatterbox(true);
                                  await fetch(`${TTS_API_URL}/tts/unload-voxcpm`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                  alert('VoxCPM2 Unloaded!');
                                } finally { setIsUnloadingChatterbox(false); }
                              }}
                              disabled={isUnloadingChatterbox}
                              className="flex-1"
                            >
                              Unload VoxCPM2
                            </Button>
                            <Button
                              variant="outline"
                              onClick={async () => {
                                try {
                                  setIsReloadingChatterbox(true);
                                  await fetch(`${TTS_API_URL}/tts/reload-voxcpm`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                                  alert('VoxCPM2 Reloaded!');
                                } finally { setIsReloadingChatterbox(false); }
                              }}
                              disabled={isReloadingChatterbox}
                              className="flex-1"
                            >
                              Reload VoxCPM2
                            </Button>
                          </div>
                        </SettingRow>
                      </SettingsAccordion>
                    </>
                  )}

                  {localSettings.ttsEngine === 'voxcpm-gguf' && (
                    <>
                      <SettingsAccordion
                        title="Engine maintenance tools"
                        summary="Voice upload, Voice Merge Lab, and ffmpeg path."
                      >
                        <SettingRow label="Upload Voice Reference" htmlFor="voice-upload-gguf" layout="stack">
                          <Input
                            id="voice-upload-gguf"
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
                          htmlFor="ffmpeg-path-gguf"
                          description="Optional. Voice Merge, STT, and D-ID need ffmpeg.exe."
                        >
                          <Input
                            id="ffmpeg-path-gguf"
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

                      <SettingRow label="Active Voice" htmlFor="voxcpm-gguf-voice">
                        <Select
                          id="voxcpm-gguf-voice"
                          value={localSettings.ttsVoice || 'default'}
                          onValueChange={value => {
                            handleChange('ttsVoice', value);
                            fetch(`${PRIMARY_API_URL}/tts/save-voice-preference`, {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify({ voice_id: value, engine: 'voxcpm-gguf' })
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

                      <SettingRow label="Voice Design" htmlFor="voxcpm-gguf-voice-design" layout="stack" description="Describe a voice (e.g. 'A young woman, gentle and sweet voice'). Leave empty to use voice reference instead.">
                        <Input
                          id="voxcpm-gguf-voice-design"
                          value={localSettings.voxcpmVoiceDesign || ''}
                          onChange={(e) => handleChange('voxcpmVoiceDesign', e.target.value)}
                          placeholder="A young woman, gentle and sweet voice"
                          disabled={!ttsEnabled}
                        />
                      </SettingRow>

                      <SettingRow label={`CFG Value (${(localSettings.voxcpmCfgValue || 2.0).toFixed(1)})`} layout="stack" description="LM guidance strength. Higher = more prompt adherence, but may degrade quality.">
                        <Slider
                          id="voxcpm-gguf-cfg-value"
                          min={0.5} max={5.0} step={0.1}
                          value={[localSettings.voxcpmCfgValue || 2.0]}
                          onValueChange={([v]) => handleChange('voxcpmCfgValue', v)}
                        />
                      </SettingRow>

                      <SettingRow label={`Inference Timesteps (${localSettings.voxcpmInferenceTimesteps || 10})`} layout="stack" description="Diffusion steps. Higher = better quality but slower.">
                        <Slider
                          id="voxcpm-gguf-inference-timesteps"
                          min={1} max={50} step={1}
                          value={[localSettings.voxcpmInferenceTimesteps || 10]}
                          onValueChange={([v]) => handleChange('voxcpmInferenceTimesteps', v)}
                        />
                      </SettingRow>

                      <SettingsAccordion
                        title="VoxCPM2 GGUF (CPU/Metal/CUDA via llama.cpp-omni)"
                        summary="Download GGUF weights for CPU-friendly VoxCPM2 inference without PyTorch."
                      >
                        <SettingRow label="GGUF Models" layout="stack">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={async () => {
                                  try {
                                    await fetchVoxcpmGgufModels();
                                  } catch (e) {
                                    console.error('Failed to fetch VoxCPM2 GGUF models:', e);
                                  }
                                }}
                              >
                                <RefreshCw className="h-4 w-4 mr-2" />
                                Refresh Status
                              </Button>
                              {!voxcpmGgufCliAvailable && (
                                <span className="text-xs text-yellow-600">
                                  voxcpm2-cli not found. Build from llama.cpp-omni.
                                </span>
                              )}
                            </div>
                            {voxcpmGgufModels.length > 0 && (
                              <div className="space-y-2 mt-3">
                                {voxcpmGgufModels.map((model) => (
                                  <div key={model.id} className="flex items-center justify-between p-2 border border-border/60 rounded">
                                    <div className="flex-1">
                                      <div className="text-sm font-medium">{model.label}</div>
                                      <div className="text-xs text-muted-foreground">{model.component}</div>
                                      <div className="text-xs text-muted-foreground">{model.filename} ({model.size_mb} MB)</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      {model.downloaded ? (
                                        <>
                                          <span className="text-xs text-green-600">Downloaded</span>
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={async () => {
                                              if (window.confirm(`Delete ${model.filename}?`)) {
                                                const result = await deleteVoxcpmGgufModel(model.filename);
                                                alert(result.message);
                                              }
                                            }}
                                          >
                                            Delete
                                          </Button>
                                        </>
                                      ) : (
                                          <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={async () => {
                                              const result = await downloadVoxcpmGgufModel(model.id);
                                              alert(result.message);
                                            }}
                                          >
                                            Download
                                        </Button>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                            <div className="text-xs text-muted-foreground mt-2">
                              Recommended: Download BaseLM-Q8_0 + Acoustic-F16 for CPU/Metal/CUDA inference via voxcpm2-cli.
                            </div>
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

                  <SettingRow
                    label="Wait for full response"
                    htmlFor="tts-wait-full"
                    description="When enabled, TTS waits for the complete AI response before starting playback. Eliminates mid-stream gaps at the cost of initial latency."
                  >
                    <Switch
                      id="tts-wait-full"
                      checked={localSettings.ttsWaitForFullResponse ?? false}
                      onCheckedChange={(value) => handleChange('ttsWaitForFullResponse', value)}
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

                  <SettingRow
                    label={`Speech Speed (${(localSettings.ttsSpeed || 1.0).toFixed(1)}x)`}
                    layout="stack"
                    description="Changes speaking pace without raising or lowering the voice."
                  >
                    <Slider
                      id="tts-speed"
                      min={0.5} max={3.0} step={0.1}
                      value={[localSettings.ttsSpeed || 1.0]}
                      onValueChange={([v]) => handleChange('ttsSpeed', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Autoplay chunk size (${localSettings.ttsStreamChunkSentences || 3} sentence${(localSettings.ttsStreamChunkSentences || 3) === 1 ? '' : 's'})`}
                    layout="stack"
                    description="How many sentences the server groups per autoplay TTS chunk. Minimum 3. Higher values increase initial wait but reduce boundary gaps for slow engines."
                  >
                    <Slider
                      id="tts-stream-chunk-sentences"
                      min={3}
                      max={12}
                      step={1}
                      value={[localSettings.ttsStreamChunkSentences || 3]}
                      onValueChange={([v]) => handleChange('ttsStreamChunkSentences', v)}
                    />
                  </SettingRow>

                  <SettingRow
                    label={`Prebuffer before playback (${localSettings.ttsPrebufferSeconds || 0}s)`}
                    layout="stack"
                    description="Seconds of audio to buffer before starting autoplay playback. 0 = start immediately. Set to ~45 for engines with RTF > 1 (e.g. VoxCPM) to prevent mid-stream stalls."
                  >
                    <Slider
                      id="tts-prebuffer-seconds"
                      min={0}
                      max={120}
                      step={5}
                      value={[localSettings.ttsPrebufferSeconds || 0]}
                      onValueChange={([v]) => handleChange('ttsPrebufferSeconds', v)}
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
                    description="Uses the same live playback as chat autoplay. Mirid keeps pace and pitch separate, so faster speech should not sound smaller. Connects even when Auto-Play TTS is off."
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
                    description="Leave blank for the built-in JSON template. Placeholders: {{CHARACTER_BLOCK}}, {{USER_BLOCK}}, {{CHAT_HISTORY}}, {{CHARACTER_SYSTEM_PROMPT}}."
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
                    description="Placeholders: {{CHARACTER_SYSTEM_PROMPT}}, {{CHARACTER_BLOCK}}, {{USER_BLOCK}}, {{CHAT_HISTORY}}. Empty = built-in template with per-card field rubric (Essence, On this call, With you, etc.). Custom templates should keep the same JSON keys."
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

        {/* Character behaviour review */}
        <TabsContent value="persona-realignment">
          <div className="space-y-4">
            <Card className="border-primary/30 bg-gradient-to-br from-primary/[0.06] to-transparent shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Sparkles className="h-5 w-5 text-primary shrink-0" aria-hidden />
                  Refresh a character
                </CardTitle>
                <CardDescription className="text-sm leading-relaxed">
                  Choose what context the review may use, then let Mirid gather it, run the selected model and prepare a proposed update. Nothing is saved until you review and confirm it. The complete set of memory tools also lives in the sidebar.
                </CardDescription>
              </CardHeader>
            </Card>
            <PersonaRealignmentPanel />
          </div>
        </TabsContent>

        {/* Memory Browser */}
        <TabsContent value="memory">
          <MemoryEditorTab onOpenPersonaRealignment={() => setSettingsMainTab('persona-realignment')} />
        </TabsContent>

        {/*local sd*/}
        {/* About */}
        <TabsContent value="about">
          <div className="space-y-6">
            <SettingsSection title="About">
              <div className="mx-auto max-w-3xl space-y-5">
                <img
                  src="/miridman.jpg"
                  alt="The MiridMan"
                  className="mx-auto block w-full max-w-sm"
                  width="700"
                  height="700"
                />
                <p className="text-sm leading-7 text-muted-foreground">
                  Mirid was built to be a one click solution for AI chat and roleplay on Windows using either local or API. It is a refined and repackaged version of my personal AI workstation{' '}
                  <a className="text-primary hover:underline" href="https://github.com/boneylizard/Eloquent">Eloquent</a>, containing approximately six months of unpublished development work on the product. Mirid brings together a library of text and multimodal open source AI products built by countless dedicated and highly talented developers that are too countless to thank.
                </p>
                <p className="text-sm leading-7 text-muted-foreground">
                  The current Mirid build was explicitly built for the Windows operating system and supports Nvidia, AMD, and cpu-only architectures. Future Mirid releases aim to expand compatibility to Linux and Apple systems.
                </p>
                <p className="text-sm leading-7 text-muted-foreground">
                  You can try Mirid yourself by downloading it{' '}
                  <a className="text-primary hover:underline" href="https://github.com/boneylizard/Eloquent/releases/latest">here</a>.
                </p>
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
                  ? `${totalAgenticCount} character memories`
                  : 'Character review — open the dedicated Settings tab'}
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
            <TabsTrigger value="agentic">Character memories</TabsTrigger>
            <TabsTrigger value="realign">Character review</TabsTrigger>
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
              <p className="text-sm font-semibold text-foreground">Refresh a character</p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Review how a character responds to you using the history and memories you choose. Mirid runs the selected model and prepares an update for approval.
              </p>
              {typeof onOpenPersonaRealignment === 'function' ? (
                <Button type="button" variant="default" size="sm" onClick={onOpenPersonaRealignment}>
                  Open character review
                </Button>
              ) : (
                <p className="text-xs text-muted-foreground">Open Settings → Character review.</p>
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

