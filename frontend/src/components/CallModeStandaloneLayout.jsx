import React, { useState, useEffect, useCallback } from 'react';
import CallModeOverlay from './CallModeOverlay';
import { subscribeDualOverlaySync, sendCallWindowClosed, clearCallOverlayState, sendCallStateRequest, sendCallInput, sendCallToggleMic, sendCallStopTts, sendCallReroll, sendCallAiContinue } from '../utils/dualOverlayWindow';
import { useTheme, ThemeProvider } from './ThemeProvider';

/** Synchronously load character data for standalone call mode */
function getStandaloneAppData() {
  try {
    // Try overlay state first (written by Chat.jsx when opening popup)
    const raw = localStorage.getItem('eloquent-dual-overlay-state');
    if (raw) {
      const parsed = JSON.parse(raw);
      
      let ac = null;
      if (parsed.ac) {
        try { ac = typeof parsed.ac === 'string' ? JSON.parse(parsed.ac) : parsed.ac; } catch(e) {}
      }
      
      const chs = Array.isArray(parsed.chs) ? parsed.chs : [];
      const api = typeof parsed.api === 'string' ? parsed.api : '';
      
      return { activeCharacter: ac, characters: chs, PRIMARY_API_URL: api };
    }
  } catch(e) {}

  // Fallback: try reading settings for PRIMARY_API_URL
  try {
    const keys = ['LiangLocal-settings', 'Eloquent-settings'];
    let primaryApiUrl = '';
    for (const key of keys) {
      const s = localStorage.getItem(key);
      if (s) { const parsed = JSON.parse(s); if (parsed.primaryApiUrl) primaryApiUrl = parsed.primaryApiUrl; break; }
    }
    return { activeCharacter: null, characters: [], PRIMARY_API_URL: primaryApiUrl };
  } catch(e) {}

  return { activeCharacter: null, characters: [], PRIMARY_API_URL: '' };
}

export default function CallModeStandaloneLayout() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <CallModeStandaloneContent />
    </ThemeProvider>
  );
}

function CallModeStandaloneContent() {
  const { theme, setTheme } = useTheme();
  
  // Read data synchronously from localStorage — same channel Chat.jsx writes to
  const appData = getStandaloneAppData();
  const [activeCharacter, setActiveCharacter] = useState(appData.activeCharacter || null);
  const [characters, setCharacters] = useState(appData.characters || []);
  const PRIMARY_API_URL = appData.PRIMARY_API_URL || '';

  // Live state sync via BroadcastChannel (messages, isGenerating, etc.)
  const [messages, setMessages] = useState([]);
  const [isPlayingAudio, setIsPlayingAudio] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [ttsSubtitleCue, setTtsSubtitleCue] = useState(null);
  
  useEffect(() => {
    const unsubscribe = subscribeDualOverlaySync({
      onCallStateRequest: () => {},
      onCallInput: (input) => { /* handled by main app */ },
      onCallToggleMic: async () => {},
      onCallStopTts: () => {},
      onDualStateSync: (state) => {
        if (state.messages !== undefined && state.messages.length > 0) setMessages(state.messages);
        if (state.isPlayingAudio !== undefined) setIsPlayingAudio(state.isPlayingAudio);
        if (state.isGenerating !== undefined) setIsGenerating(state.isGenerating);
        if (state.isRecording !== undefined) setIsRecording(state.isRecording);
        if (state.isTranscribing !== undefined) setIsTranscribing(state.isTranscribing);
        if (state.ttsSubtitleCue !== undefined) setTtsSubtitleCue(state.ttsSubtitleCue);
        if (state.characters !== undefined) setCharacters(state.characters);
      },
      onCallWindowClosed: () => {},
    });

    sendCallStateRequest(); // request initial state from main app
    return unsubscribe;
  }, []);

  const handleClose = useCallback(() => {
    sendCallWindowClosed();
    clearCallOverlayState();
    setTimeout(() => window.close(), 100);
  }, []);

  useEffect(() => {
    window.addEventListener('beforeunload', sendCallWindowClosed);
    return () => window.removeEventListener('beforeunload', sendCallWindowClosed);
  }, []);

  // Action callbacks that send messages to main window via BroadcastChannel
  const handleSendMessage = useCallback((text) => {
    sendCallInput(text);
  }, []);

  const handleToggleMic = useCallback(() => {
    sendCallToggleMic();
  }, []);

  const handleStopTts = useCallback(() => {
    sendCallStopTts();
  }, []);

  const handleReroll = useCallback(() => {
    sendCallReroll();
  }, []);

  const handleCycleAvatar = useCallback((delta, totalAvatars) => {
    // Cycle avatar locally — popup owns its own avatar state independently
    setActiveCharacter(prev => {
      if (!prev) return prev;
      const maxIndex = (totalAvatars || prev.avatar_folder_count || prev.avatars?.length || 1) - 1;
      const raw = (prev.activeAvatarIndex ?? 0) + delta;
      const nextIndex = raw < 0 ? maxIndex : raw > maxIndex ? 0 : raw;
      return { ...prev, activeAvatarIndex: nextIndex };
    });
  }, []);

  const handleAiContinue = useCallback(() => {
    sendCallAiContinue();
  }, []);

  return (
    <div className="min-h-screen bg-black text-white">
      <CallModeOverlay
        isActive={true} onExit={handleClose}
        activeCharacter={activeCharacter}
        isPlayingAudio={isPlayingAudio}
        isRecording={isRecording} isTranscribing={isTranscribing}
        PRIMARY_API_URL={PRIMARY_API_URL}
        charactersOverride={characters}
        onOpenStoryTracker={() => {}} onOpenChoiceGenerator={() => {}}
        messages={messages} onRegenerate={handleReroll}
        ttsSubtitleCue={ttsSubtitleCue}
        userProfile={{}} primaryModel=''
        isStandaloneWindow={true}
        onSendMessage={handleSendMessage}
        onToggleMic={handleToggleMic}
        onStopTts={handleStopTts}
        onCycleAvatar={handleCycleAvatar}
        onAiContinue={handleAiContinue}
        isGenerating={isGenerating}
      />
    </div>
  );
}
