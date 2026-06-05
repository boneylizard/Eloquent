import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  AudioLines,
  Play,
  Pause,
  Square,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Pin,
  PinOff,
  Maximize2,
  Minimize2,
  Move,
  MessageSquarePlus,
  LayoutGrid,
  X,
} from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { useApp } from '../contexts/AppContext';
import VoiceQuickPicker from './VoiceQuickPicker';
import CallModeCharacterViewport, { CALL_PORTRAIT_FRAME_CLASS } from './CallModeCharacterViewport';
import CallModeAboutPanel from './CallModeAboutPanel';
import { cn } from '@/lib/utils';
import { fetchCallModeCharacterAbout, CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES } from '../utils/callModeCharacterAbout';
import {
  composeLayeredSystemPrompt,
  isSystemPersonaModeActive,
  resolveSystemPersonaCharacter,
} from '../utils/systemPersona';
import {
  buildIntelTtsOverrides,
  intelMessageId,
  isIntelMessageId,
  loadIntelTtsVoice,
  saveIntelTtsVoice,
} from '../utils/callModeIntelTts';
import {
  getActiveCharacterAvatar,
  getCallModeAvatarCycleList,
  resolveAvatarDisplayUrl,
  isAvatarVideoUrl,
  CALL_MODE_VIDEO_HOTKEYS,
  CALL_MODE_FULLSCREEN_ZOOM_MIN,
  CALL_MODE_FULLSCREEN_ZOOM_MAX,
  CALL_MODE_FULLSCREEN_ZOOM_DEFAULT,
  CALL_MODE_FRAMING_MIGRATION_KEY,
  clampCallModeFullscreenZoom,
} from '../utils/characterAvatars';

const CALL_PORTRAIT_RIPPLE_CLASS = `${CALL_PORTRAIT_FRAME_CLASS} rounded-[2rem] sm:rounded-[2.75rem]`;
const VIDEO_CONTROLS_IDLE_MS = 2600;
const FRAMING_PANEL_AUTO_HIDE_MS = 3000;

/** Stacking inside call overlay (low → high): avatar → content → controls */
const Z_CALL_AVATAR_FULLSCREEN = 1;
const Z_CALL_CONTENT = 10;
const Z_CALL_CONTROLS = 40;

/** Soft voice-active halo — slow pulse only, no ping ripples. */
function CallPortraitVoiceGlow({ active, fullscreen = false }) {
  if (!active || fullscreen) return null;
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden>
      <div className={`relative ${CALL_PORTRAIT_RIPPLE_CLASS}`}>
        <div className="absolute inset-0 rounded-[2rem] sm:rounded-[2.75rem] bg-cyan-400/7 animate-[pulse_4.5s_ease-in-out_infinite]" />
        <div
          className="absolute inset-0 rounded-[2rem] sm:rounded-[2.75rem] border border-cyan-300/12 animate-[pulse_4.5s_ease-in-out_infinite]"
          style={{ animationDelay: '2.25s' }}
        />
      </div>
    </div>
  );
}

const CallModeOverlay = ({
  isActive,
  onExit,
  activeCharacter,
  isPlayingAudio,
  isRecording,
  isTranscribing,
  PRIMARY_API_URL,
  // New props for control panel features
  onOpenStoryTracker,
  onOpenChoiceGenerator,
  messages,
  onRegenerate,
  ttsSubtitleCue, // ✅ USE PROP instead of context
  userProfile,
  primaryModel,
  isStandaloneWindow = false,
  charactersOverride,
  // Props for standalone popup actions (sent via BroadcastChannel)
  onSendMessage,
  onToggleMic,
  onStopTts,
  onCycleAvatar,
  onAiContinue,
  isGenerating: isGeneratingProp,
}) => {
  // ✅ REMOVED ttsSubtitleCue from useApp() - using prop instead
  const {
    startRecording,
    stopRecording,
    sendMessage,
    generateCallModeFollowUp,
    stopTTS,
    handleStopGeneration,
    isGenerating,
    characters: contextCharacters,
    settings,
    updateSettings,
    cycleCharacterAvatar,
    buildSystemPrompt,
    buildSystemPersonaPrompt,
    getGenerationSystemPrompt,
    playStreamingTtsScript,
    activeConversation,
    conversations,
  } = useApp();
  // In standalone popup, use charactersOverride (popup has its own character list)
  // In main window, use context characters
  const characters = charactersOverride || contextCharacters;
  // In standalone popup, use prop-provided isGenerating since useApp() returns empty state
  const effectiveIsGenerating = isStandaloneWindow ? (isGeneratingProp ?? false) : isGenerating;
  const [isPulsing, setIsPulsing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [showControlPanel, setShowControlPanel] = useState(false);
  const [showSubtitles, setShowSubtitles] = useState(false);
  const [subtitleVisible, setSubtitleVisible] = useState(false);
  const [currentCue, setCurrentCue] = useState(null);
  const [ultraMinimalMode, setUltraMinimalMode] = useState(() => {
    try {
      return localStorage.getItem('LiangLocal-call-ultra-minimal') === '1';
    } catch {
      return false;
    }
  });
  const [showVoicePicker, setShowVoicePicker] = useState(false);
  const [showCharName, setShowCharName] = useState(() => {
    try {
      return localStorage.getItem('LiangLocal-call-show-char-name') !== '0';
    } catch {
      return true;
    }
  });
  const [showAboutHotspot, setShowAboutHotspot] = useState(() => {
    try {
      return localStorage.getItem('LiangLocal-call-show-about-hotspot') !== '0';
    } catch {
      return true;
    }
  });
  const [showFramingPanel, setShowFramingPanel] = useState(false);
  const framingHideTimerRef = useRef(null);
  const zoomCommitTimerRef = useRef(null);
  const panCommitTimerRef = useRef(null);

  const flashStatus = useCallback(() => {}, []);

  useEffect(() => {
    try {
      localStorage.setItem('LiangLocal-call-ultra-minimal', ultraMinimalMode ? '1' : '0');
    } catch {
      // ignore storage failures
    }
  }, [ultraMinimalMode]);

  useEffect(() => {
    try {
      localStorage.setItem('LiangLocal-call-show-char-name', showCharName ? '1' : '0');
    } catch {}
  }, [showCharName]);

  useEffect(() => {
    try {
      localStorage.setItem('LiangLocal-call-show-about-hotspot', showAboutHotspot ? '1' : '0');
    } catch {}
  }, [showAboutHotspot]);

  useEffect(() => {
    const pollInterval = setInterval(() => {
      if (isStandaloneWindow) {
        // In standalone popup, use prop-provided subtitle cue (synced via BroadcastChannel)
        if (ttsSubtitleCue) {
          setCurrentCue({ ...ttsSubtitleCue });
        }
      } else {
        if (window.__ttsSubtitleCue) {
          setCurrentCue({ ...window.__ttsSubtitleCue });
        }
      }
    }, 50);
    return () => clearInterval(pollInterval);
  }, [isStandaloneWindow, ttsSubtitleCue]);

  useEffect(() => {
    if (ultraMinimalMode) {
      setShowSubtitles(false);
    }
  }, [ultraMinimalMode]);

  const subtitleText = useMemo(() => {
    return (currentCue?.text || '').trim();
  }, [currentCue]);

  const subtitleSnippet = useMemo(() => {
    const text = subtitleText.replace(/\s+/g, ' ').trim();
    if (!text) return '';
    if (text.length <= 450) return text;
    return `...${text.slice(-450)}`;
  }, [subtitleText]);

  useEffect(() => {
    if (!showSubtitles || !subtitleSnippet) {
      setSubtitleVisible(false);
      return;
    }
    setSubtitleVisible(true);
    const durationMs = Math.max(900, Math.min(4000, ttsSubtitleCue?.durationMs || 1500));
    const timer = setTimeout(() => {
      setSubtitleVisible(false);
    }, durationMs);
    return () => clearTimeout(timer);
  }, [showSubtitles, subtitleSnippet, currentCue, ttsSubtitleCue?.durationMs]);

  // Handle Reroll (Regenerate last bot message)
  const handleReroll = useCallback(() => {
    if (effectiveIsGenerating || isTranscribing) return;

    if (isStandaloneWindow && onRegenerate) {
      // In standalone popup, send reroll request to main window
      onRegenerate();
      return;
    }

    // Find the last bot message to regenerate
    const lastBotMsg = [...(messages || [])].reverse().find(m => m.role === 'bot');
    if (lastBotMsg && onRegenerate) {
      console.log(`🔄 [CallMode] Rerolling message ${lastBotMsg.id}`);
      // Stop current playback if it's speaking
      stopTTS('call_mode_reroll');
      handleStopGeneration();

      onRegenerate(lastBotMsg.id);
    } else {
      console.warn("⚠️ [CallMode] No bot message found to reroll");
    }
  }, [messages, onRegenerate, effectiveIsGenerating, isTranscribing, stopTTS, handleStopGeneration, isStandaloneWindow]);

  // Sync pulsing animation with TTS playback
  useEffect(() => {
    setIsPulsing(Boolean(isPlayingAudio && !isIntelMessageId(isPlayingAudio)));
  }, [isPlayingAudio]);

  // NEW: Track processing state until audio starts playing
  useEffect(() => {
    if (isTranscribing) {
      setIsProcessing(true);
    }
  }, [isTranscribing]);

  // NEW: Monitor when streaming audio starts playing (character speech only — not insight readout)
  useEffect(() => {
    const checkAudioPlaying = () => {
      const characterStreaming = Boolean(
        window.streamingAudioPlaying && !window.intelStreamingAudioPlaying
      );
      if (characterStreaming) {
        setIsProcessing(false);
        setIsSpeaking(true);
      } else {
        setIsSpeaking(false);
      }
    };

    checkAudioPlaying();
    const interval = setInterval(checkAudioPlaying, 100);

    return () => clearInterval(interval);
  }, []);

  const handleAiContinue = useCallback(() => {
    if (effectiveIsGenerating || isRecording || isTranscribing) return;

    if (isStandaloneWindow && onAiContinue) {
      // In standalone popup, send AI continue request to main window
      onAiContinue();
      return;
    }

    stopTTS('call_mode_ai_continue');
    handleStopGeneration();
    generateCallModeFollowUp?.();
    flashStatus('AI continued', 'violet');
  }, [generateCallModeFollowUp, handleStopGeneration, effectiveIsGenerating, isRecording, isTranscribing, stopTTS, flashStatus, isStandaloneWindow, onAiContinue]);

  // Handle recording with auto-send
  const handleRecord = useCallback(async () => {
    if (isStandaloneWindow && onToggleMic) {
      // In standalone popup, send mic toggle request to main window
      onToggleMic();
      return;
    }

    if (isRecording) {
      const asrTraceId = `asr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const asrSource = 'asr_call_mode';
      // Stop recording and auto-send
      await stopRecording(async (transcript) => {
        const cleaned = String(transcript || '').trim();
        if (!cleaned) {
          console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=skip_empty_transcript`);
          return;
        }
        console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=autosend_start transcript_len=${cleaned.length}`);
        await sendMessage(cleaned);
        console.info(`[ASR_AUTOSEND_GUARD] trace_id=${asrTraceId} source=${asrSource} action=autosend_dispatched`);
      });
      flashStatus('Recording off', 'amber');
    } else {
      // Start recording
      await startRecording();
      flashStatus('Recording on', 'green');
    }
  }, [isRecording, startRecording, stopRecording, sendMessage, flashStatus, isStandaloneWindow, onToggleMic]);

  // Handle keyboard interaction
  useEffect(() => {
    const downRef = { current: new Set() };
    const triggeredRef = { current: false };

    const isInputLike = (el) => {
      const tag = el?.tagName;
      if (!tag) return false;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable;
    };

    const comboPressed = () => (
      downRef.current.has('ControlLeft') &&
      downRef.current.has('ControlRight') &&
      downRef.current.has('AltRight') &&
      downRef.current.has('KeyC')
    );

    const pedalChordFromEvent = (ev) =>
      ev.code === 'KeyC'
      && !ev.repeat
      && (
        comboPressed()
        || (ev.ctrlKey && ev.altKey)
      );

    const handleKeyDown = (event) => {
      // Track key state by physical key (supports left/right modifiers).
      // This enables a foot pedal mapped to: LCtrl + RCtrl + RAlt + C
      if (event.code) downRef.current.add(event.code);

      // Mic toggle hotkey: LCtrl + RCtrl + RAlt + C
      // Handle this globally even if an input retained focus underneath the overlay.
      if (isActive && pedalChordFromEvent(event)) {
        event.preventDefault();
        event.stopPropagation();
        if (!triggeredRef.current) {
          triggeredRef.current = true;
          if (!isProcessing) {
            handleRecord();
            flashStatus('Pedal: mic toggle', 'green');
          }
        }
        return;
      }

      // For normal shortcuts, still avoid stealing keys while typing in input-like elements.
      if (isInputLike(event.target)) return;

      if (event.key === 'Escape' && isActive) {
        if (showControlPanel) {
          setShowControlPanel(false);
        } else if (settings?.callModeFullscreenAvatar === true) {
          updateSettings({ callModeFullscreenAvatar: false });
          flashStatus('Portrait window', 'cyan', 900);
        } else {
          if (isStandaloneWindow) {
            window.close();
          } else {
            onExit();
          }
        }
      } else if (event.key === ' ' && isActive && !showControlPanel) {
        // Prevent default space bar behavior (scrolling)
        event.preventDefault();
        // Don't trigger if already processing
        if (!isProcessing) {
          handleRecord();
        }
      } else if (event.key === 'Tab' && isActive) {
        // Toggle control panel
        event.preventDefault();
        setShowControlPanel(prev => !prev);
      } else if ((event.key === 'm' || event.key === 'M') && isActive) {
        event.preventDefault();
        setUltraMinimalMode(prev => !prev);
      } else if ((event.key === 'n' || event.key === 'N') && isActive) {
        event.preventDefault();
        setShowCharName(prev => !prev);
      } else if ((event.key === 'a' || event.key === 'A') && isActive) {
        event.preventDefault();
        setShowAboutHotspot(prev => !prev);
      } else if ((event.key === 'r' || event.key === 'R') && isActive) {
        // Reroll shortcut
        event.preventDefault();
        handleReroll();
      }
    };

    const handleKeyUp = (event) => {
      if (event.code) downRef.current.delete(event.code);
      const chordStillHeld =
        downRef.current.has('ControlLeft')
        && downRef.current.has('ControlRight')
        && downRef.current.has('AltRight')
        && downRef.current.has('KeyC');
      if (!chordStillHeld) triggeredRef.current = false;
    };

    const handleBlur = () => {
      downRef.current.clear();
      triggeredRef.current = false;
    };

    if (isActive) {
      // Use window-level capture so hotkeys work without clicking overlay first.
      // This is important for foot-pedal key emitters.
      window.addEventListener('keydown', handleKeyDown, true);
      window.addEventListener('keyup', handleKeyUp, true);
      window.addEventListener('blur', handleBlur);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
      window.removeEventListener('keyup', handleKeyUp, true);
      window.removeEventListener('blur', handleBlur);
    };
  }, [
    isActive,
    onExit,
    isProcessing,
    handleRecord,
    isSpeaking,
    stopTTS,
    handleStopGeneration,
    showControlPanel,
    flashStatus,
    settings?.callModeFullscreenAvatar,
    updateSettings,
  ]);

  useEffect(() => {
    if (!isActive) return;
    if (isRecording) flashStatus('Recording on', 'green');
    else flashStatus('Recording off', 'amber', 900);
    // only reacts to recording edge while active
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording, isActive]);

  useEffect(() => {
    if (!isActive) return;
    if (isPlayingAudio || isSpeaking || window.streamingAudioPlaying) {
      flashStatus('Voice playing', 'cyan', 900);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlayingAudio, isSpeaking, isActive]);

  const NARRATOR_CHARACTER_ID = '__narrator__';

  const speakingMessage = useMemo(() => {
    if (!isActive) return null;
    if (isPlayingAudio && messages?.length) {
      const match = messages.find(m => m.id === isPlayingAudio);
      if (match) return match;
    }
    return messages ? [...messages].reverse().find(m => m.role === 'bot') : null;
  }, [isActive, isPlayingAudio, messages]);

  const displayCharacter = useMemo(() => {
    if (!isActive) return null;
    // Standalone popup uses its own locally-cycled activeCharacter directly
    if (isStandaloneWindow) return activeCharacter || null;
    if (speakingMessage?.characterId === NARRATOR_CHARACTER_ID) {
      return {
        name: (settings?.narratorName || '').trim() || 'Narrator',
        avatar: settings?.narratorAvatar || null
      };
    }
    if (speakingMessage?.characterId) {
      const byId = (characters || []).find(c => c.id === speakingMessage.characterId);
      if (byId) return byId;
    }
    if (speakingMessage?.characterName) {
      const byName = (characters || []).find(c => c.name === speakingMessage.characterName);
      if (byName) return byName;
      return { name: speakingMessage.characterName, avatar: speakingMessage.avatar || null };
    }
    return activeCharacter || null;
  }, [speakingMessage, characters, settings?.narratorName, settings?.narratorAvatar, activeCharacter]);

  const displayAvatar = useMemo(() => {
    if (speakingMessage?.characterId === NARRATOR_CHARACTER_ID) {
      return settings?.narratorAvatar || speakingMessage?.avatar || null;
    }
    if (displayCharacter?.id) {
      return getActiveCharacterAvatar(displayCharacter) || speakingMessage?.avatar || null;
    }
    return speakingMessage?.avatar || getActiveCharacterAvatar(displayCharacter) || null;
  }, [speakingMessage, displayCharacter, settings?.narratorAvatar]);

  const avatarUrl = resolveAvatarDisplayUrl(displayAvatar, PRIMARY_API_URL);
  const isVideoAvatar = useMemo(() => isAvatarVideoUrl(avatarUrl), [avatarUrl]);

  const avatarList = useMemo(
    () => (displayCharacter?.id ? getCallModeAvatarCycleList(displayCharacter) : []),
    [displayCharacter]
  );

  const fullscreenAvatar = settings?.callModeFullscreenAvatar === false;
  const [liveFullscreenZoom, setLiveFullscreenZoom] = useState(() =>
    clampCallModeFullscreenZoom(settings?.callModeFullscreenZoom)
  );
  const fullscreenZoom = liveFullscreenZoom;
  const [liveFullscreenPan, setLiveFullscreenPan] = useState(() => ({
    x: Number(settings?.callModeFullscreenPanX) || 0,
    y: Number(settings?.callModeFullscreenPanY) || 0,
  }));
  const fullscreenPanX = liveFullscreenPan.x;
  const fullscreenPanY = liveFullscreenPan.y;

  useEffect(() => {
    const next = clampCallModeFullscreenZoom(settings?.callModeFullscreenZoom);
    setLiveFullscreenZoom((prev) => (Math.abs(prev - next) < 0.001 ? prev : next));
  }, [settings?.callModeFullscreenZoom]);

  useEffect(() => {
    const nextX = Number(settings?.callModeFullscreenPanX) || 0;
    const nextY = Number(settings?.callModeFullscreenPanY) || 0;
    setLiveFullscreenPan((prev) => (
      Math.abs(prev.x - nextX) < 0.001 && Math.abs(prev.y - nextY) < 0.001
        ? prev
        : { x: nextX, y: nextY }
    ));
  }, [settings?.callModeFullscreenPanX, settings?.callModeFullscreenPanY]);

  useEffect(() => () => {
    if (zoomCommitTimerRef.current) {
      window.clearTimeout(zoomCommitTimerRef.current);
      zoomCommitTimerRef.current = null;
    }
    if (panCommitTimerRef.current) {
      window.clearTimeout(panCommitTimerRef.current);
      panCommitTimerRef.current = null;
    }
  }, []);

  /** One-time: reset persisted zoom&lt;1, zoom&gt;1, or pan from pre–fit-to-screen builds. */
  useEffect(() => {
    if (!isActive) return;
    try {
      if (localStorage.getItem(CALL_MODE_FRAMING_MIGRATION_KEY) === '1') return;
      localStorage.setItem(CALL_MODE_FRAMING_MIGRATION_KEY, '1');
    } catch {
      return;
    }
    const z = Number(settings?.callModeFullscreenZoom);
    const px = Number(settings?.callModeFullscreenPanX) || 0;
    const py = Number(settings?.callModeFullscreenPanY) || 0;
    const needsReset =
      !Number.isFinite(z)
      || z < CALL_MODE_FULLSCREEN_ZOOM_MIN - 0.001
      || z > CALL_MODE_FULLSCREEN_ZOOM_DEFAULT + 0.001
      || px !== 0
      || py !== 0;
    if (needsReset) {
      updateSettings({
        callModeFullscreenZoom: CALL_MODE_FULLSCREEN_ZOOM_DEFAULT,
        callModeFullscreenPanX: 0,
        callModeFullscreenPanY: 0,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  const setFullscreenAvatar = useCallback(
    (enabled) => {
      updateSettings({ callModeFullscreenAvatar: Boolean(enabled) });
    },
    [updateSettings]
  );

  const setFullscreenZoom = useCallback(
    (zoom) => {
      const next = clampCallModeFullscreenZoom(zoom);
      setLiveFullscreenZoom(next);
      if (zoomCommitTimerRef.current) window.clearTimeout(zoomCommitTimerRef.current);
      zoomCommitTimerRef.current = window.setTimeout(() => {
        updateSettings({ callModeFullscreenZoom: next });
        zoomCommitTimerRef.current = null;
      }, 180);
    },
    [updateSettings]
  );

  const handleMediaPanChange = useCallback(
    (panX, panY) => {
      const nextX = Math.max(-40, Math.min(40, Number(panX) || 0));
      const nextY = Math.max(-40, Math.min(40, Number(panY) || 0));
      setLiveFullscreenPan({ x: nextX, y: nextY });
      if (panCommitTimerRef.current) window.clearTimeout(panCommitTimerRef.current);
      panCommitTimerRef.current = window.setTimeout(() => {
        updateSettings({ callModeFullscreenPanX: nextX, callModeFullscreenPanY: nextY });
        panCommitTimerRef.current = null;
      }, 180);
    },
    [updateSettings]
  );

  const resetFullscreenFraming = useCallback(() => {
    if (zoomCommitTimerRef.current) {
      window.clearTimeout(zoomCommitTimerRef.current);
      zoomCommitTimerRef.current = null;
    }
    if (panCommitTimerRef.current) {
      window.clearTimeout(panCommitTimerRef.current);
      panCommitTimerRef.current = null;
    }
    setLiveFullscreenZoom(CALL_MODE_FULLSCREEN_ZOOM_DEFAULT);
    setLiveFullscreenPan({ x: 0, y: 0 });
    updateSettings({
      callModeFullscreenZoom: CALL_MODE_FULLSCREEN_ZOOM_DEFAULT,
      callModeFullscreenPanX: 0,
      callModeFullscreenPanY: 0,
    });
  }, [updateSettings]);

  const clearFramingHideTimer = useCallback(() => {
    if (framingHideTimerRef.current) {
      clearTimeout(framingHideTimerRef.current);
      framingHideTimerRef.current = null;
    }
  }, []);

  const scheduleFramingHide = useCallback(() => {
    clearFramingHideTimer();
    framingHideTimerRef.current = window.setTimeout(() => {
      setShowFramingPanel(false);
      framingHideTimerRef.current = null;
    }, FRAMING_PANEL_AUTO_HIDE_MS);
  }, [clearFramingHideTimer]);

  const toggleFramingPanel = useCallback(() => {
    setShowFramingPanel((prev) => {
      const next = !prev;
      if (next) scheduleFramingHide();
      else clearFramingHideTimer();
      return next;
    });
  }, [scheduleFramingHide, clearFramingHideTimer]);

  const bumpFramingPanelVisibility = useCallback(() => {
    if (showFramingPanel) scheduleFramingHide();
  }, [showFramingPanel, scheduleFramingHide]);

  useEffect(() => {
    if (!fullscreenAvatar) {
      setShowFramingPanel(false);
      clearFramingHideTimer();
    }
  }, [fullscreenAvatar, clearFramingHideTimer]);

  useEffect(() => () => clearFramingHideTimer(), [clearFramingHideTimer]);

  const toggleFullscreenAvatar = useCallback(() => {
    setFullscreenAvatar(!fullscreenAvatar);
    flashStatus(fullscreenAvatar ? 'Portrait window' : 'Fullscreen avatar', 'cyan', 900);
  }, [fullscreenAvatar, setFullscreenAvatar, flashStatus]);

  const activeAvatarIndex = displayCharacter?.activeAvatarIndex ?? 0;
  const avatarVideoRef = useRef(null);
  const [videoUserPaused, setVideoUserPaused] = useState(false);
  const [videoUserStopped, setVideoUserStopped] = useState(false);
  const [videoAudioOn, setVideoAudioOn] = useState(true);
  const [videoRestartToken, setVideoRestartToken] = useState(0);
  const [videoControlsVisible, setVideoControlsVisible] = useState(false);
  const [videoControlsPinned, setVideoControlsPinned] = useState(false);
  const videoControlsHideTimerRef = useRef(null);

  const clearVideoControlsHideTimer = useCallback(() => {
    if (videoControlsHideTimerRef.current) {
      window.clearTimeout(videoControlsHideTimerRef.current);
      videoControlsHideTimerRef.current = null;
    }
  }, []);

  const scheduleHideVideoControls = useCallback(() => {
    clearVideoControlsHideTimer();
    if (videoControlsPinned) return;
    videoControlsHideTimerRef.current = window.setTimeout(() => {
      setVideoControlsVisible(false);
    }, VIDEO_CONTROLS_IDLE_MS);
  }, [clearVideoControlsHideTimer, videoControlsPinned]);

  const revealVideoControls = useCallback(() => {
    setVideoControlsVisible(true);
    scheduleHideVideoControls();
  }, [scheduleHideVideoControls]);

  const handleViewportPointerActivity = useCallback(() => {
    if (!isVideoAvatar) return;
    revealVideoControls();
  }, [isVideoAvatar, revealVideoControls]);

  const handleViewportPointerLeave = useCallback(() => {
    if (videoControlsPinned) return;
    clearVideoControlsHideTimer();
    videoControlsHideTimerRef.current = window.setTimeout(() => {
      setVideoControlsVisible(false);
    }, 450);
  }, [clearVideoControlsHideTimer, videoControlsPinned]);

  const toggleVideoControlsPinned = useCallback((e) => {
    e?.stopPropagation?.();
    setVideoControlsPinned((pinned) => {
      const next = !pinned;
      if (next) {
        clearVideoControlsHideTimer();
        setVideoControlsVisible(true);
      } else {
        scheduleHideVideoControls();
      }
      return next;
    });
  }, [clearVideoControlsHideTimer, scheduleHideVideoControls]);

  useEffect(() => {
    if (!isActive || !isVideoAvatar) {
      setVideoControlsVisible(false);
      setVideoControlsPinned(false);
      clearVideoControlsHideTimer();
      return undefined;
    }
    revealVideoControls();
    return () => clearVideoControlsHideTimer();
  }, [isActive, isVideoAvatar, avatarUrl, activeAvatarIndex, revealVideoControls, clearVideoControlsHideTimer]);

  useEffect(() => () => clearVideoControlsHideTimer(), [clearVideoControlsHideTimer]);

  const aboutCharacterEnabled = settings?.callModeAboutCharacterEnabled !== false;
  const [aboutPanelOpen, setAboutPanelOpen] = useState(false);
  const [aboutLoading, setAboutLoading] = useState(false);
  const [aboutResult, setAboutResult] = useState(null);
  const [aboutPartialText, setAboutPartialText] = useState('');
  const [aboutError, setAboutError] = useState(null);
  const aboutAbortRef = useRef(null);

  const closeAboutPanel = useCallback(() => {
    aboutAbortRef.current?.abort?.();
    aboutAbortRef.current = null;
    setAboutPanelOpen(false);
    setAboutLoading(false);
  }, []);

  const requestCharacterAbout = useCallback(async ({ forceRefresh = false } = {}) => {
    if (!aboutCharacterEnabled || !displayCharacter) return;
    if (aboutLoading && !forceRefresh) return;

    aboutAbortRef.current?.abort?.();
    const controller = new AbortController();
    aboutAbortRef.current = controller;

    setAboutPanelOpen(true);
    setAboutLoading(true);
    setAboutError(null);
    if (forceRefresh) {
      setAboutResult(null);
      setAboutPartialText('');
    }

    try {
      const aboutMode =
        settings?.callModeAboutCharacterSystemPromptMode
        || CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.flat;

      const result = await fetchCallModeCharacterAbout({
        apiUrl: PRIMARY_API_URL,
        modelName: primaryModel,
        character: displayCharacter,
        userProfile,
        messages,
        settings,
        signal: controller.signal,
        onPartial: setAboutPartialText,
        systemPromptMode: aboutMode,
        resolveCharacterSystemPrompt: async () => {
          if (aboutMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.character_card) {
            const activeConvMeta = activeConversation
              ? conversations?.find((c) => c.id === activeConversation)
              : null;
            const systemPersonaChar = isSystemPersonaModeActive(settings, activeConvMeta)
              ? resolveSystemPersonaCharacter(characters, settings, activeConvMeta)
              : null;
            if (systemPersonaChar && displayCharacter) {
              return (
                composeLayeredSystemPrompt(
                  buildSystemPersonaPrompt(systemPersonaChar),
                  buildSystemPrompt(displayCharacter)
                ) || null
              );
            }
            return buildSystemPrompt(displayCharacter) || null;
          }
          if (aboutMode === CALL_MODE_ABOUT_SYSTEM_PROMPT_MODES.full_generation) {
            const trigger =
              [...(messages || [])].reverse().find((m) => m?.content?.trim())?.content?.trim()
              || displayCharacter?.first_message
              || displayCharacter?.name
              || '';
            return (await getGenerationSystemPrompt(trigger, displayCharacter, null, {
              includeAuthorNote: false,
            })) || null;
          }
          return null;
        },
      });
      if (controller.signal.aborted) return;
      setAboutResult(result);
      setAboutPartialText(result.rawText || '');
      flashStatus('Character insight ready', 'violet', 900);
    } catch (err) {
      if (controller.signal.aborted) return;
      setAboutError(err?.message || 'Failed to load character insight');
    } finally {
      if (!controller.signal.aborted) {
        setAboutLoading(false);
      }
    }
  }, [
    aboutCharacterEnabled,
    displayCharacter,
    aboutLoading,
    PRIMARY_API_URL,
    primaryModel,
    userProfile,
    messages,
    settings,
    flashStatus,
    buildSystemPrompt,
    buildSystemPersonaPrompt,
    getGenerationSystemPrompt,
    activeConversation,
    conversations,
  ]);

  useEffect(() => {
    if (!isActive) {
      closeAboutPanel();
      setAboutResult(null);
      setAboutPartialText('');
      setAboutError(null);
    }
  }, [isActive, closeAboutPanel]);

  useEffect(() => {
    setAboutResult(null);
    setAboutPartialText('');
    setAboutError(null);
    if (aboutPanelOpen) closeAboutPanel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayCharacter?.id]);

  const isTtsActive = useMemo(
    () => Boolean(
      (isPlayingAudio && !isIntelMessageId(isPlayingAudio))
      || isSpeaking
      || (typeof window !== 'undefined' && window.streamingAudioPlaying && !window.intelStreamingAudioPlaying)
    ),
    [isPlayingAudio, isSpeaking]
  );

  const [intelVoice, setIntelVoice] = useState(() =>
    loadIntelTtsVoice(settings?.ttsVoice || 'af_heart')
  );
  const [intelPlayingSlot, setIntelPlayingSlot] = useState(null);

  useEffect(() => {
    if (intelVoice) saveIntelTtsVoice(intelVoice);
  }, [intelVoice]);

  const autoTtsBusy = useMemo(
    () => Boolean(isTtsActive || effectiveIsGenerating),
    [isTtsActive, effectiveIsGenerating]
  );

  const characterVoiceLabel = useMemo(() => {
    const v = displayCharacter?.ttsVoice || displayCharacter?.tts_voice;
    if (!v || v === 'default') return null;
    const tail = String(v).split(/[/\\]/).pop();
    return tail || v;
  }, [displayCharacter?.ttsVoice, displayCharacter?.tts_voice]);

  const handlePlayIntel = useCallback(
    (slotId, text) => {
      if (autoTtsBusy || !text?.trim()) return;
      const msgId = intelMessageId(slotId);
      const overrides = buildIntelTtsOverrides(settings, intelVoice);
      setIntelPlayingSlot(slotId);
      void playStreamingTtsScript(msgId, text.trim(), overrides, {
        intelPlayback: true,
        onComplete: () => setIntelPlayingSlot(null),
      });
    },
    [autoTtsBusy, settings, intelVoice, playStreamingTtsScript]
  );

  const videoPlaybackPaused = isVideoAvatar && (videoUserPaused || videoUserStopped || isTtsActive);
  const videoPlaybackMuted = isVideoAvatar && (isTtsActive || !videoAudioOn);

  useEffect(() => {
    setVideoUserPaused(false);
    setVideoUserStopped(false);
  }, [avatarUrl, activeAvatarIndex]);

  const toggleVideoPlayback = useCallback(() => {
    if (!isVideoAvatar) return;
    revealVideoControls();
    if (videoUserPaused || videoUserStopped) {
      setVideoUserPaused(false);
      setVideoUserStopped(false);
      setVideoAudioOn(true);
      avatarVideoRef.current?.restart?.();
      flashStatus('Video playing', 'green', 700);
    } else {
      setVideoUserPaused(true);
      avatarVideoRef.current?.pause?.();
      flashStatus('Video paused', 'amber', 700);
    }
  }, [isVideoAvatar, videoUserPaused, videoUserStopped, flashStatus, revealVideoControls]);

  const stopVideoPlayback = useCallback(() => {
    if (!isVideoAvatar) return;
    revealVideoControls();
    setVideoUserPaused(true);
    setVideoUserStopped(true);
    avatarVideoRef.current?.pause?.();
    flashStatus('Video stopped', 'red', 900);
  }, [isVideoAvatar, flashStatus, revealVideoControls]);

  const restartVideoPlayback = useCallback(() => {
    if (!isVideoAvatar) return;
    revealVideoControls();
    setVideoUserPaused(false);
    setVideoUserStopped(false);
    setVideoAudioOn(true);
    setVideoRestartToken((t) => t + 1);
    flashStatus('Video restarted', 'cyan', 700);
  }, [isVideoAvatar, flashStatus, revealVideoControls]);

  const cycleAvatarSilent = useCallback(
    (delta) => {
      if (!displayCharacter?.id || avatarList.length <= 1) return;
      if (isStandaloneWindow && onCycleAvatar) {
        onCycleAvatar(delta, avatarList.length);
      } else {
        cycleCharacterAvatar(displayCharacter.id, delta);
      }
    },
    [displayCharacter?.id, avatarList.length, cycleCharacterAvatar, isStandaloneWindow, onCycleAvatar]
  );

  const cycleAvatarManual = useCallback((delta) => {
    if (!displayCharacter?.id || avatarList.length <= 1) return;
    revealVideoControls();
    if (isStandaloneWindow && onCycleAvatar) {
      onCycleAvatar(delta, avatarList.length);
    } else {
      cycleCharacterAvatar(displayCharacter.id, delta);
    }
    flashStatus(delta > 0 ? 'Next look' : 'Previous look', 'violet', 700);
  }, [displayCharacter?.id, avatarList.length, cycleCharacterAvatar, flashStatus, revealVideoControls, isStandaloneWindow, onCycleAvatar]);

  useEffect(() => {
    if (!isActive || !fullscreenAvatar) return undefined;

    const onWheel = (e) => {
      // About panel scroll must never adjust avatar framing.
      if (aboutPanelOpen) return;
      if (e.target?.closest?.('[data-character-about-panel]')) return;
      if (e.target?.closest?.('[data-character-about-hotspot]')) return;
      if (e.target?.closest?.('[data-avatar-video-controls]')) return;
      // Only zoom when scrolling directly over the portrait frame.
      if (!e.target?.closest?.('[data-call-portrait-frame]')) return;

      e.preventDefault();
      const direction = e.deltaY > 0 ? -1 : 1;
      const step = e.ctrlKey ? 0.03 : 0.08;
      const nextZoom = clampCallModeFullscreenZoom(fullscreenZoom + direction * step);
      setFullscreenZoom(nextZoom);
      setShowFramingPanel(true);
      scheduleFramingHide();
    };

    window.addEventListener('wheel', onWheel, { passive: false });
    return () => window.removeEventListener('wheel', onWheel);
  }, [
    isActive,
    fullscreenAvatar,
    fullscreenZoom,
    setFullscreenZoom,
    scheduleFramingHide,
    aboutPanelOpen,
  ]);

  useEffect(() => {
    if (!isActive) return undefined;
    const isInputLike = (el) => {
      const tag = el?.tagName;
      if (!tag) return false;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el?.isContentEditable;
    };
    const onKeyDown = (event) => {
      if (isInputLike(event.target)) return;
      const key = event.key;
      if (key === CALL_MODE_VIDEO_HOTKEYS.prevAvatar && displayCharacter?.id && avatarList.length > 1) {
        event.preventDefault();
        cycleAvatarManual(-1);
        return;
      }
      if (key === CALL_MODE_VIDEO_HOTKEYS.nextAvatar && displayCharacter?.id && avatarList.length > 1) {
        event.preventDefault();
        cycleAvatarManual(1);
        return;
      }
      if (key.toLowerCase() === 'f') {
        event.preventDefault();
        toggleFullscreenAvatar();
        return;
      }
      if (!isVideoAvatar) return;
      revealVideoControls();
      const lk = key.toLowerCase();
      if (lk === CALL_MODE_VIDEO_HOTKEYS.togglePlay) {
        event.preventDefault();
        toggleVideoPlayback();
      } else if (lk === CALL_MODE_VIDEO_HOTKEYS.stop) {
        event.preventDefault();
        stopVideoPlayback();
      } else if (lk === CALL_MODE_VIDEO_HOTKEYS.restart) {
        event.preventDefault();
        restartVideoPlayback();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [
    isActive,
    isVideoAvatar,
    displayCharacter?.id,
    avatarList.length,
    cycleAvatarManual,
    toggleVideoPlayback,
    stopVideoPlayback,
    restartVideoPlayback,
    revealVideoControls,
    toggleFullscreenAvatar,
  ]);

  const viewportProps = {
    avatarUrl,
    isVideoAvatar,
    characterName: displayCharacter?.name,
    isSpeaking,
    isTtsActive,
    avatarVideoRef,
    videoKey: `${displayCharacter?.id}-${activeAvatarIndex}`,
    videoPlaybackPaused,
    videoPlaybackMuted,
    videoRestartToken,
    avatarListLength: avatarList.length,
    onToggleVideo: toggleVideoPlayback,
    onPointerActivity: handleViewportPointerActivity,
    onPointerLeave: handleViewportPointerLeave,
    aboutHotspotEnabled: aboutCharacterEnabled && Boolean(displayCharacter?.name) && showAboutHotspot,
    showAboutHotspot,
    showCharacterName: showCharName,
    onAboutClick: () => {
      if (aboutResult && !aboutLoading) {
        setAboutPanelOpen(true);
        return;
      }
      requestCharacterAbout();
    },
    isAboutLoading: aboutLoading,
    onError: () => {
      console.warn(`Avatar failed to load: ${avatarUrl}`);
    },
    fullscreen: fullscreenAvatar,
    mediaZoom: fullscreenAvatar ? fullscreenZoom : 1,
    mediaPanX: fullscreenAvatar ? fullscreenPanX : 0,
    mediaPanY: fullscreenAvatar ? fullscreenPanY : 0,
    onMediaPanChange: fullscreenAvatar ? handleMediaPanChange : undefined,
  };

  const avatarVideoControls = isVideoAvatar ? (
    <>
      <button
        type="button"
        aria-label="Show video controls"
        className={cn(
          'absolute bottom-6 left-1/2 z-[5] -translate-x-1/2 rounded-full border border-white/10 bg-black/35 p-1.5 text-white/70 backdrop-blur-sm transition-all duration-300',
          'opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto',
          (videoControlsVisible || videoControlsPinned) && '!opacity-0 !pointer-events-none'
        )}
        onClick={(e) => {
          e.stopPropagation();
          revealVideoControls();
        }}
      >
        <ChevronUp className="h-4 w-4" />
      </button>
      <div
        data-avatar-video-controls
        onMouseEnter={revealVideoControls}
        className={cn(
          'absolute bottom-[5.75rem] left-1/2 z-20 flex -translate-x-1/2 flex-wrap items-center justify-center gap-2 rounded-2xl border border-white/15 bg-black/55 px-3 py-2 backdrop-blur-sm transition-all duration-300 ease-out',
          videoControlsVisible || videoControlsPinned
            ? 'pointer-events-auto translate-y-0 opacity-100'
            : 'pointer-events-none translate-y-2 opacity-0'
        )}
      >
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 disabled:opacity-40"
          aria-label="Previous look"
          title={`Previous look (${CALL_MODE_VIDEO_HOTKEYS.prevAvatar})`}
          disabled={avatarList.length <= 1}
          onClick={(e) => {
            e.stopPropagation();
            cycleAvatarManual(-1);
          }}
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          aria-label={videoUserPaused || videoUserStopped || isTtsActive ? 'Play video' : 'Pause video'}
          title={`Play / pause (${CALL_MODE_VIDEO_HOTKEYS.togglePlay})`}
          onClick={(e) => {
            e.stopPropagation();
            toggleVideoPlayback();
          }}
        >
          {videoUserPaused || videoUserStopped || isTtsActive ? (
            <Play className="h-5 w-5" />
          ) : (
            <Pause className="h-5 w-5" />
          )}
        </button>
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          aria-label="Stop video"
          title={`Stop (${CALL_MODE_VIDEO_HOTKEYS.stop})`}
          onClick={(e) => {
            e.stopPropagation();
            stopVideoPlayback();
          }}
        >
          <Square className="h-4 w-4 fill-current" />
        </button>
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20"
          aria-label="Restart video"
          title={`Restart (${CALL_MODE_VIDEO_HOTKEYS.restart})`}
          onClick={(e) => {
            e.stopPropagation();
            restartVideoPlayback();
          }}
        >
          <RotateCcw className="h-4 w-4" />
        </button>
        <button
          type="button"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 disabled:opacity-40"
          aria-label="Next look"
          title={`Next look (${CALL_MODE_VIDEO_HOTKEYS.nextAvatar})`}
          disabled={avatarList.length <= 1}
          onClick={(e) => {
            e.stopPropagation();
            cycleAvatarManual(1);
          }}
        >
          <ChevronRight className="h-5 w-5" />
        </button>
        <button
          type="button"
          className={cn(
            'flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20',
            videoControlsPinned && 'bg-white/20 ring-1 ring-white/25'
          )}
          aria-label={videoControlsPinned ? 'Unpin video controls' : 'Pin video controls open'}
          title={videoControlsPinned ? 'Unpin controls (auto-hide)' : 'Pin controls open'}
          onClick={toggleVideoControlsPinned}
        >
          {videoControlsPinned ? <PinOff className="h-4 w-4" /> : <Pin className="h-4 w-4" />}
        </button>
      </div>
    </>
  ) : null;

  const avatarLookControls = null;

  if (!isActive) return null;

  if (ultraMinimalMode) {
    const isVoicePlaying = Boolean(isPlayingAudio || isSpeaking || window.streamingAudioPlaying);
    return (
      <div className="fixed inset-0 z-[9999] bg-black text-white">
        <button
          type="button"
          onClick={onExit}
          className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/80 hover:bg-white/20"
          aria-label="Exit call mode"
          title="Exit Call Mode"
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
        <button
          type="button"
          onClick={() => setUltraMinimalMode(false)}
          className="absolute top-4 left-4 flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white/45 hover:bg-white/20 hover:text-white/75"
          aria-label="Return to full call overlay"
          title="Return to full call overlay"
        >
          <LayoutGrid className="h-4 w-4" aria-hidden />
        </button>

        <div className="absolute left-4 bottom-4 flex items-center gap-2 text-[10px] tracking-wider">
          <span className={`h-2 w-2 rounded-full ${isRecording ? 'bg-emerald-400 animate-pulse' : 'bg-white/20'}`} title="Recording" />
          <span className={`h-2 w-2 rounded-full ${isTranscribing ? 'bg-amber-400 animate-pulse' : 'bg-white/20'}`} title="Processing" />
          <span className={`h-2 w-2 rounded-full ${isVoicePlaying ? 'bg-cyan-400 animate-pulse' : 'bg-white/20'}`} title="Voice playing" />
        </div>

        <button
          type="button"
          onClick={() => setShowVoicePicker(true)}
          className="absolute bottom-20 right-4 flex h-11 w-11 items-center justify-center rounded-full border border-white/20 bg-white/10 text-white shadow-lg hover:bg-white/20"
          title="Voices & clones"
        >
          <AudioLines className="h-5 w-5" />
        </button>
        <VoiceQuickPicker
          open={showVoicePicker}
          onOpenChange={setShowVoicePicker}
          variant="call-sheet"
          primaryApiUrl={PRIMARY_API_URL}
        />
      </div>
    );
  }

  return (
    <div className={isStandaloneWindow ? "h-screen w-screen cursor-default bg-black flex flex-col items-center justify-center overflow-hidden isolate" : "fixed inset-0 z-[9999] cursor-default bg-black flex flex-col items-center justify-center overflow-hidden isolate"}>
      {/* Layer 1 — true fullscreen avatar (back); text/UI stacks above */}
      {fullscreenAvatar && (
        <div
          data-call-fullscreen-avatar-layer
          className={isStandaloneWindow ? "fixed inset-0 h-[100dvh] w-[100vw] max-h-[100dvh] max-w-[100vw] cursor-default bg-black pointer-events-auto" : "fixed inset-0 h-[100dvh] w-[100vw] max-h-[100dvh] max-w-[100vw] cursor-default bg-black pointer-events-auto"}
          style={{ zIndex: Z_CALL_AVATAR_FULLSCREEN }}
        >
          <CallModeCharacterViewport {...viewportProps}>
            {avatarVideoControls}
            {avatarLookControls}
          </CallModeCharacterViewport>
          {avatarList.length > 1 && !showControlPanel && (
            <>
              <button
                type="button"
                tabIndex={-1}
                aria-hidden
                className="absolute left-0 top-[32%] bottom-[32%] z-[2] w-[min(11vw,3.5rem)] cursor-default border-0 bg-transparent p-0 outline-none focus:outline-none"
                onClick={(e) => {
                  e.stopPropagation();
                  cycleAvatarSilent(-1);
                }}
              />
              <button
                type="button"
                tabIndex={-1}
                aria-hidden
                className="absolute right-0 top-[32%] bottom-[32%] z-[2] w-[min(11vw,3.5rem)] cursor-default border-0 bg-transparent p-0 outline-none focus:outline-none"
                onClick={(e) => {
                  e.stopPropagation();
                  cycleAvatarSilent(1);
                }}
              />
            </>
          )}
        </div>
      )}

      {/* Control panel: TAB to open (no visible toggle in minimal call view). */}

      {/* Control Panel Backdrop (Close on click outside) */}
      {showControlPanel && (
        <div
          className="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity duration-300"
          style={{ zIndex: Z_CALL_CONTROLS }}
          onClick={() => setShowControlPanel(false)}
        />
      )}

      {/* Control Panel Slide-out */}
      <div
        className={`absolute left-0 top-0 bottom-0 w-72 bg-gradient-to-r from-gray-900/95 to-gray-900/90 backdrop-blur-lg
                    border-r border-white/10 transition-transform duration-300 ease-out
                    ${showControlPanel ? 'translate-x-0' : '-translate-x-full'}`}
        style={{ zIndex: Z_CALL_CONTROLS }}
      >
        <div className="p-6 pt-20 space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-white text-lg font-semibold flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="3" y1="9" x2="21" y2="9" />
                <line x1="9" y1="21" x2="9" y2="9" />
              </svg>
              Control Panel
            </h3>
            <button
              onClick={() => setShowControlPanel(false)}
              className="p-1 rounded-full hover:bg-white/10 text-white/50 hover:text-white transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          {/* Story Tracker Button */}
          <button
            onClick={() => {
              setShowControlPanel(false);
              onOpenStoryTracker?.();
            }}
            disabled={!onOpenStoryTracker}
            className="w-full p-4 rounded-xl bg-gradient-to-r from-indigo-500/20 to-purple-500/20 
                       border border-indigo-500/30 hover:border-indigo-400/50 
                       text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]
                       disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-500/30 flex items-center justify-center group-hover:bg-indigo-500/40 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
              </div>
              <div>
                <div className="font-medium">Story Tracker</div>
                <div className="text-xs text-white/60">Track characters, items & events</div>
              </div>
            </div>
          </button>

          {/* Choice Generator Button */}
          <button
            onClick={() => {
              setShowControlPanel(false);
              onOpenChoiceGenerator?.();
            }}
            disabled={!onOpenChoiceGenerator || !messages?.length}
            className="w-full p-4 rounded-xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 
                       border border-amber-500/30 hover:border-amber-400/50 
                       text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]
                       disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/30 flex items-center justify-center group-hover:bg-amber-500/40 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="2" width="20" height="20" rx="2" />
                  <circle cx="8" cy="8" r="1.5" />
                  <circle cx="16" cy="8" r="1.5" />
                  <circle cx="8" cy="16" r="1.5" />
                  <circle cx="16" cy="16" r="1.5" />
                  <circle cx="12" cy="12" r="1.5" />
                </svg>
              </div>
              <div>
                <div className="font-medium">Choice Generator</div>
                <div className="text-xs text-white/60">Generate action options</div>
              </div>
            </div>
          </button>

          <button
            type="button"
            onClick={toggleFullscreenAvatar}
            className={cn(
              'w-full p-4 rounded-xl text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] group border',
              fullscreenAvatar
                ? 'bg-gradient-to-r from-cyan-500/25 to-sky-500/20 border-cyan-400/40'
                : 'bg-gradient-to-r from-sky-500/15 to-cyan-500/15 border-sky-400/25 hover:border-sky-300/45'
            )}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-sky-500/30 flex items-center justify-center group-hover:bg-sky-500/40 transition-colors">
                {fullscreenAvatar ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
              </div>
              <div>
                <div className="font-medium">{fullscreenAvatar ? 'Exit fullscreen avatar' : 'Fullscreen avatar'}</div>
                <div className="text-xs text-white/60">Portrait fills the screen · hotkey F</div>
              </div>
            </div>
          </button>

          {/* Reroll Button */}
          <button
            onClick={() => {
              setShowControlPanel(false);
              handleReroll();
            }}
            disabled={effectiveIsGenerating || isTranscribing || !messages?.some(m => m.role === 'bot')}
            className="w-full p-4 rounded-xl bg-gradient-to-r from-emerald-500/20 to-teal-500/20 
                       border border-emerald-500/30 hover:border-emerald-400/50 
                       text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]
                       disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/30 flex items-center justify-center group-hover:bg-emerald-500/40 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                  <path d="M3 3v5h5" />
                  <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                  <path d="M16 16h5v5" />
                </svg>
              </div>
              <div>
                <div className="font-medium">Reroll Response</div>
                <div className="text-xs text-white/60">Regenerate last AI answer</div>
              </div>
            </div>
          </button>

          <button
            type="button"
            onClick={() => setShowSubtitles(prev => !prev)}
            className="w-full p-4 rounded-xl bg-gradient-to-r from-slate-500/20 to-slate-400/20 
                       border border-slate-400/30 hover:border-slate-300/50 
                       text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-slate-500/30 flex items-center justify-center group-hover:bg-slate-500/40 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="4" width="18" height="14" rx="2" ry="2" />
                  <path d="M7 16h.01" />
                  <path d="M11 16h.01" />
                  <path d="M15 16h.01" />
                </svg>
              </div>
              <div>
                <div className="font-medium">{showSubtitles ? 'Hide Subtitles' : 'Show Subtitles'}</div>
                <div className="text-xs text-white/60">On-screen AI text while speaking</div>
              </div>
            </div>
          </button>

          <button
            type="button"
            onClick={() => {
              setShowControlPanel(false);
              setShowVoicePicker(true);
            }}
            className="w-full p-4 rounded-xl bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 
                       border border-violet-400/30 hover:border-violet-300/50 
                       text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-violet-500/30 flex items-center justify-center group-hover:bg-violet-500/40 transition-colors">
                <AudioLines className="h-5 w-5" />
              </div>
              <div>
                <div className="font-medium">Voices &amp; clones</div>
                <div className="text-xs text-white/60">Try Kokoro voices or Chatterbox refs per character</div>
              </div>
            </div>
          </button>

          {/* Stop Speaking Button - Visible during speaking OR generation */}
          {(isSpeaking || window.streamingAudioPlaying || effectiveIsGenerating) && (
            <button
              onClick={() => {
                if (isStandaloneWindow && onStopTts) {
                  onStopTts();
                } else {
                  stopTTS('call_mode_stop_button');
                  handleStopGeneration();
                }
              }}
              className="w-full p-4 rounded-xl bg-gradient-to-r from-red-500/20 to-pink-500/20 
                         border border-red-500/30 hover:border-red-400/50 
                         text-white text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] group"
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-red-500/30 flex items-center justify-center group-hover:bg-red-500/40 transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="6" y="4" width="4" height="16" />
                    <rect x="14" y="4" width="4" height="16" />
                  </svg>
                </div>
                <div>
                  <div className="font-medium">Stop Speaking</div>
                  <div className="text-xs text-white/60">Interrupt AI response</div>
                </div>
              </div>
            </button>
          )}

          {/* Keyboard Shortcuts */}
          <div className="mt-6 pt-4 border-t border-white/10">
            <h4 className="text-white/60 text-xs font-medium mb-3">KEYBOARD SHORTCUTS</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-white/70">
                <span>Toggle Panel</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">TAB</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Record</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">SPACE</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Reroll</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">R</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Stop Speaking</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">SHIFT</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Exit</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">ESC</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Ultra Minimal</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">M</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Toggle name</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">N</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Toggle About hotspot</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">A</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Fullscreen avatar</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">F</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Prev / next look</span>
                <span className="text-xs"><kbd className="bg-white/10 px-1.5 py-0.5 rounded">[</kbd> <kbd className="bg-white/10 px-1.5 py-0.5 rounded">]</kbd></span>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Video play / pause</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">V</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Video stop</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">S</kbd>
              </div>
              <div className="flex justify-between text-white/70">
                <span>Video restart</span>
                <kbd className="bg-white/10 px-2 py-0.5 rounded text-xs">G</kbd>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Exit is ESC only in minimal call view. */}

      {/* Mic — corner placement, transparent with accent state feedback */}
      <button
        type="button"
        onClick={handleRecord}
        disabled={isTranscribing}
        aria-label={isRecording ? 'Stop recording and send' : 'Start recording'}
        className={cn(
          'absolute bottom-6 right-6 flex h-14 w-14 items-center justify-center rounded-full border-2 backdrop-blur-sm',
          'opacity-0 transition-all duration-200 hover:scale-110 hover:opacity-100 focus-visible:opacity-100 active:scale-95',
          isRecording
            ? 'opacity-100 border-red-400/90 bg-red-500/15 text-red-300 shadow-[0_0_24px_rgba(248,113,113,0.45)] ring-2 ring-red-400/35'
            : 'border-blue-400/45 bg-black/15 text-blue-200/90 hover:border-blue-400/75 hover:bg-blue-500/10',
          isTranscribing && 'opacity-100 cursor-not-allowed'
        )}
        style={{ zIndex: Z_CALL_CONTROLS }}
        title={isRecording ? 'Tap once to stop, transcribe, and send (no hold)' : 'Tap once to start — talk as long as you want — tap again to stop'}
      >
        {isRecording && (
          <span
            className="pointer-events-none absolute inset-0 rounded-full border-2 border-red-400/50 animate-ping"
            aria-hidden
          />
        )}
        {isRecording ? (
          <div className="relative h-5 w-5 rounded-sm bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]" />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="26"
            height="26"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="relative"
            aria-hidden
          >
            <path d="M12 1c-3 0-5 2-5 5v6c0 3 2 5 5 5s5-2 5-5V6c0-3-2-5-5-5z" />
            <path d="M19 10v2c0 7-3 9-7 9s-7-2-7-9v-2" />
            <line x1="12" y1="19" x2="12" y2="23" />
            <line x1="8" y1="23" x2="16" y2="23" />
          </svg>
        )}
      </button>

      {/* Continue remains available via existing chat controls outside this minimal overlay. */}

      {/* Layer 2 — call content & text (above fullscreen avatar) */}
      <div
        className={cn(
          'relative flex flex-col items-center justify-center flex-1 w-full min-h-0 gap-4 sm:gap-6 px-3 sm:px-6 py-4 pointer-events-none [&_button]:pointer-events-auto [&_[data-call-portrait-frame]]:pointer-events-auto',
          fullscreenAvatar && 'gap-2 py-2'
        )}
        style={{ zIndex: Z_CALL_CONTENT }}
      >
        {/* Portrait window (normal mode only — fullscreen avatar is layer 1) */}
        {!fullscreenAvatar && (
        <div className="relative flex min-h-0 flex-1 items-center justify-center w-full max-h-[calc(100dvh-13rem)]">
          <CallPortraitVoiceGlow active={isSpeaking || isPulsing} fullscreen={false} />

          <CallModeCharacterViewport {...viewportProps}>
            {avatarVideoControls}
            {avatarLookControls}
          </CallModeCharacterViewport>

        </div>
        )}
      </div>

      {/* Fullscreen framing — hidden by default; corner toggle + auto-hide */}
      {fullscreenAvatar && (
        <>
          <button
            type="button"
            onClick={toggleFramingPanel}
            className={cn(
              'pointer-events-auto fixed bottom-6 left-6 flex h-10 w-10 items-center justify-center rounded-full border text-white shadow-lg transition-all duration-200 hover:scale-105 active:scale-95',
              'opacity-0 hover:opacity-100 focus-visible:opacity-100',
              showFramingPanel
                ? 'opacity-100 border-cyan-400/40 bg-cyan-500/25 hover:bg-cyan-500/35'
                : 'border-white/15 bg-black/45 hover:bg-black/60 backdrop-blur-sm'
            )}
            style={{ zIndex: Z_CALL_CONTROLS }}
            title={showFramingPanel ? 'Hide framing controls' : 'Framing — zoom & pan'}
            aria-label={showFramingPanel ? 'Hide framing controls' : 'Show framing controls'}
            aria-expanded={showFramingPanel}
          >
            <Move className="h-4 w-4" />
          </button>

          {showFramingPanel && (
            <div
              className="pointer-events-auto fixed bottom-20 left-1/2 w-[min(92vw,340px)] -translate-x-1/2 rounded-2xl border border-white/15 bg-black/70 px-4 py-3 shadow-xl backdrop-blur-md transition-opacity duration-200"
              style={{ zIndex: Z_CALL_CONTROLS }}
              onClick={(e) => e.stopPropagation()}
              onPointerDown={bumpFramingPanelVisibility}
              onPointerMove={bumpFramingPanelVisibility}
            >
              <Slider
                aria-label="Fullscreen avatar zoom"
                value={[Math.round(fullscreenZoom * 100)]}
                min={Math.round(CALL_MODE_FULLSCREEN_ZOOM_MIN * 100)}
                max={Math.round(CALL_MODE_FULLSCREEN_ZOOM_MAX * 100)}
                step={5}
                onValueChange={([v]) => {
                  setFullscreenZoom((Number(v) || 100) / 100);
                  bumpFramingPanelVisibility();
                }}
              />
              <button
                type="button"
                className="mt-3 flex h-9 w-full items-center justify-center rounded-lg border border-white/10 bg-white/5 text-white/80 hover:bg-white/10"
                aria-label="Reset framing"
                title="Reset framing"
                onClick={() => {
                  resetFullscreenFraming();
                  bumpFramingPanelVisibility();
                }}
              >
                <RotateCcw className="h-4 w-4" aria-hidden />
              </button>
            </div>
          )}
        </>
      )}

      <CallModeAboutPanel
        open={aboutPanelOpen}
        characterName={displayCharacter?.name}
        loading={aboutLoading}
        partialText={aboutPartialText}
        result={aboutResult}
        error={aboutError}
        onClose={closeAboutPanel}
        onRefresh={() => requestCharacterAbout({ forceRefresh: true })}
        primaryApiUrl={PRIMARY_API_URL}
        settings={settings}
        intelVoice={intelVoice}
        onIntelVoiceChange={setIntelVoice}
        characterVoiceLabel={characterVoiceLabel}
        autoTtsBusy={autoTtsBusy}
        playingIntelSlotId={intelPlayingSlot}
        onPlayIntel={handlePlayIntel}
      />

      {showSubtitles && subtitleSnippet && (
        <div
          className={`absolute bottom-20 left-1/2 w-full max-w-5xl -translate-x-1/2 px-6 py-4 rounded-2xl
                      bg-black/60 border border-white/10 backdrop-blur-sm text-center text-white/90 text-base md:text-lg
                      leading-snug shadow-lg transition-opacity duration-300 pointer-events-none
                      ${subtitleVisible ? 'opacity-100' : 'opacity-0'}`}
          style={{ zIndex: Z_CALL_CONTENT }}
          aria-live="polite"
        >
          {subtitleSnippet}
        </div>
      )}

      <VoiceQuickPicker
        open={showVoicePicker}
        onOpenChange={setShowVoicePicker}
        variant="call-sheet"
        primaryApiUrl={PRIMARY_API_URL}
      />
    </div>
  );
};

export default CallModeOverlay;
