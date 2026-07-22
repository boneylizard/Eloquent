import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Mic, PlayCircle as PlayIcon, X, Cpu, Focus, AudioLines, ScrollText, Lock } from 'lucide-react';
import ChatMessageItem from './ChatMessageItem';
import FocusModeInputForm from './FocusModeInputForm';
import VoiceQuickPicker from './VoiceQuickPicker';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';

const FocusModeOverlay = ({ 
  isActive, 
  onExit,
  messages,
  handleSubmit,
  isGenerating,
  primaryModel,
  renderAvatar,
  renderUserAvatar,
  PRIMARY_API_URL,
  primaryCharacter,
  secondaryCharacter,
  getCurrentVariantContent,
  getVariantCount,
  navigateVariant,
  editingMessageId,
  handleSaveEditedMessage,
  handleCancelEdit,
  handleEditUserMessage,
  handleRegenerateFromEditedPrompt,
  editingBotMessageId,
  handleEditBotMessage,
  handleSaveBotMessage,
  handleCancelBotEdit,
  handleGenerateVariant,
  handleContinueGeneration,
  ttsEnabled,
  isPlayingAudio,
  handleSpeakerClick,
  handleRegenerateImage,
  onCancelRegenerations,
  isRegenerationRunning,
  regenerationQueue,
  currentVariantIndex,
  formatModelName,
  stopTTS,
  focusModeInputRef,
  sttEnabled,
  isRecording,
  isTranscribing,
  onFocusModeMicClick,
  modelReady,
  ttsAutoPlay,
  onTtsAutoPlayChange,
  onTtsEnabledChange,
  onSttEnabledChange,
  onStopGeneration,
  apiError,
  audioError,
  onDismissApiError,
  onDismissAudioError,
}) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [voiceQuickOpen, setVoiceQuickOpen] = useState(false);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);
  const [userScrolledRecently, setUserScrolledRecently] = useState(false);
  const lastScrollTimeRef = useRef(0);
  const isNearBottomRef = useRef(true);
  const [showTypingIndicator, setShowTypingIndicator] = useState(false);
  const focusError = audioError || apiError;
  const focusErrorText = focusError?.message || String(focusError || '');

  // Handle ESC key to exit
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && isActive && !modelPickerOpen && !voiceQuickOpen) {
        onExit();
      }
    };

    if (isActive) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isActive, modelPickerOpen, onExit, voiceQuickOpen]);

  // Track user scroll behavior
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50; // 50px threshold
      
      isNearBottomRef.current = isAtBottom;
      
      // If user scrolled up, pause auto-scroll temporarily
      if (!isAtBottom) {
        setUserScrolledRecently(true);
        lastScrollTimeRef.current = Date.now();
        
        // Clear the flag after 3 seconds of no scrolling
        setTimeout(() => {
          if (Date.now() - lastScrollTimeRef.current >= 2900) {
            setUserScrolledRecently(false);
          }
        }, 3000);
      } else {
        // User is at bottom, enable auto-scroll
        setUserScrolledRecently(false);
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [isActive]);

  // Smart auto-scroll: only scroll if user hasn't manually scrolled recently
  useEffect(() => {
    if (!autoScrollEnabled) return;
    
    // Only auto-scroll if:
    // 1. Auto-scroll is enabled globally
    // 2. User hasn't manually scrolled recently
    // 3. User is near the bottom already
    if (!userScrolledRecently && isNearBottomRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScrollEnabled, userScrolledRecently]);

  // Typing indicator — shows when generating but no streaming content yet
  const isStreamingContent = useMemo(() => {
    const lastMsg = messages[messages.length - 1];
    if (!lastMsg) return false;
    const isStreaming = lastMsg?.isStreaming === true || lastMsg?.reasoningStreaming === true;
    return isStreaming && (lastMsg?.content?.length > 0 || lastMsg?.reasoningText?.length > 0);
  }, [messages]);

  useEffect(() => {
    if (isGenerating && !isStreamingContent && messages.length > 0) {
      const timer = setTimeout(() => setShowTypingIndicator(true), 300);
      return () => clearTimeout(timer);
    } else {
      setShowTypingIndicator(false);
    }
  }, [isGenerating, isStreamingContent, messages.length]);

  // STABLE LOGIC: Pre-calculate the last message ID for each character.
  // This message is the only one that will ever get an avatar.
const lastMessageAvatars = useMemo(() => {
    const lastMessageMap = new Map();
    if (!messages) return lastMessageMap;

    // Iterate backwards to find the last message for each character/model efficiently.
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role !== 'user' && msg.role !== 'system') {
            const characterId = msg.characterName || msg.modelId || 'assistant';
            if (!lastMessageMap.has(characterId)) {
                lastMessageMap.set(characterId, msg.id);
            }
        }
    }
    return lastMessageMap;
}, [messages]);

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex flex-col" style={{ background: 'radial-gradient(ellipse at 50% 30%, rgba(20,20,35,1), #000 70%)' }}>
      <div className="z-20 flex shrink-0 items-center gap-2 border-b border-[var(--chat-focus-border)] bg-black/55 px-2 py-2 backdrop-blur-md md:px-4">
        <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto no-scrollbar">
          <NanoGptModelSelectorPopover
            compact
            open={modelPickerOpen}
            onOpenChange={setModelPickerOpen}
            currentModelId={primaryModel}
            primaryApiUrl={PRIMARY_API_URL}
            trigger={({ setOpen, display }) => (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 max-w-[min(260px,55vw)] shrink-0 gap-1.5 border-[var(--chat-focus-border)] bg-[var(--chat-focus-surface)] text-[var(--chat-focus-text-bright)] hover:bg-[var(--chat-focus-surface-hover)]"
                onClick={() => setOpen(true)}
                disabled={isGenerating}
                title="Change model"
                aria-label="Change model"
              >
                <Cpu size={15} />
                <span className="truncate">{display?.shortLabel || display?.label || 'Select model'}</span>
              </Button>
            )}
          />

          <Button
            type="button"
            variant={ttsEnabled ? 'secondary' : 'ghost'}
            size="sm"
            className="h-9 shrink-0 gap-1.5 text-[var(--chat-focus-text-bright)]"
            onClick={() => {
              if (ttsEnabled) stopTTS?.('focus_tts_disabled');
              onTtsEnabledChange?.(!ttsEnabled);
            }}
            aria-pressed={ttsEnabled}
            title={ttsEnabled ? 'Turn text-to-speech off' : 'Turn text-to-speech on'}
          >
            <AudioLines size={15} />
            <span>TTS</span>
          </Button>

          <Button
            type="button"
            variant={ttsAutoPlay ? 'secondary' : 'ghost'}
            size="sm"
            className="h-9 shrink-0 gap-1.5 text-[var(--chat-focus-text-bright)]"
            onClick={() => onTtsAutoPlayChange?.(!ttsAutoPlay)}
            disabled={!ttsEnabled}
            aria-pressed={ttsAutoPlay}
            title={ttsAutoPlay ? 'Stop reading new replies automatically' : 'Read new replies automatically'}
          >
            <PlayIcon size={15} />
            <span>Auto TTS</span>
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-9 shrink-0 gap-1.5 text-[var(--chat-focus-text-bright)]"
            onClick={() => setVoiceQuickOpen(true)}
            disabled={!ttsEnabled}
            title="Choose voices"
          >
            <AudioLines size={15} />
            <span>Voice</span>
          </Button>

          <Button
            type="button"
            variant={sttEnabled ? 'secondary' : 'ghost'}
            size="sm"
            className="h-9 shrink-0 gap-1.5 text-[var(--chat-focus-text-bright)]"
            onClick={() => onSttEnabledChange?.(!sttEnabled)}
            disabled={isRecording || isTranscribing}
            aria-pressed={sttEnabled}
            title={sttEnabled ? 'Turn voice input off' : 'Turn voice input on'}
          >
            <Mic size={15} />
            <span>Voice input</span>
          </Button>

          {isPlayingAudio && (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              className="h-9 shrink-0 gap-1.5"
              onClick={() => stopTTS?.('focus_stop_audio')}
              title="Stop audio"
            >
              <X size={15} />
              <span>Stop audio</span>
            </Button>
          )}

          {isGenerating && (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              className="h-9 shrink-0 gap-1.5"
              onClick={onStopGeneration}
              title="Stop generating"
            >
              <X size={15} />
              <span>Stop</span>
            </Button>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-[var(--chat-focus-text-bright)] hover:bg-[var(--chat-focus-surface-hover)]"
            onClick={() => setAutoScrollEnabled(!autoScrollEnabled)}
            title={autoScrollEnabled ? 'Disable auto-scroll' : 'Enable auto-scroll'}
            aria-pressed={autoScrollEnabled}
          >
            {autoScrollEnabled ? <ScrollText size={18} /> : <Lock size={18} />}
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-9 w-9 text-[var(--chat-focus-text-bright)] hover:bg-[var(--chat-focus-surface-hover)]"
            onClick={onExit}
            title="Exit Focus Mode (ESC)"
            aria-label="Exit Focus Mode"
          >
            <X size={20} />
          </Button>
        </div>
      </div>

      <VoiceQuickPicker
        open={voiceQuickOpen}
        onOpenChange={setVoiceQuickOpen}
        variant="dialog"
        primaryApiUrl={PRIMARY_API_URL}
      />

      {focusErrorText && (
        <div className="flex shrink-0 items-start justify-between gap-3 border-b border-red-400/25 bg-red-950/35 px-3 py-2 text-sm text-red-100" role="alert">
          <span>{focusErrorText}</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0 text-red-100 hover:bg-red-400/15 hover:text-white"
            onClick={() => {
              if (audioError) onDismissAudioError?.();
              if (apiError) onDismissApiError?.();
            }}
            title="Dismiss error"
            aria-label="Dismiss error"
          >
            <X size={14} />
          </Button>
        </div>
      )}
      
      {/* User scroll status indicator */}
      {userScrolledRecently && (
        <div className="absolute top-14 right-3 px-3 py-1 rounded-full bg-[var(--chat-focus-indicator-bg)] text-[var(--chat-focus-indicator-text)] text-xs z-10 border border-[var(--chat-focus-indicator-border)] animate-in fade-in duration-200 md:top-20 md:right-6">
          Reading Mode — Auto-scroll paused
        </div>
      )}

      <div ref={messagesContainerRef} className="flex-1 min-h-0 overflow-y-auto p-3 md:p-6 messages-scroll-container">
        <div className="max-w-3xl mx-auto w-full">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="mb-6 focus-empty-glow">
                <Focus size={48} className="text-[var(--primary)] opacity-60" />
              </div>
              <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--chat-focus-text-bright)' }}>Focus Mode</h3>
              <p className="max-w-md" style={{ color: 'var(--chat-focus-text)' }}>Clean, distraction-free chat interface. Press <kbd className="inline-flex items-center justify-center rounded border border-[var(--chat-focus-border)] bg-[var(--chat-focus-surface)] px-1.5 py-0.5 text-[10px] font-mono">ESC</kbd> to exit.</p>
            </div>
          ) : (
            <>
{messages.map((msg) => {
  const characterId = msg.characterName || msg.modelId || 'assistant';
  const isLastMessage = lastMessageAvatars.get(characterId) === msg.id;

  return (
    <ChatMessageItem
      key={msg.id}
      msg={msg}
      isLastMessage={isLastMessage}
      renderAvatar={renderAvatar}
      renderUserAvatar={renderUserAvatar}
      PRIMARY_API_URL={PRIMARY_API_URL}
      primaryCharacter={primaryCharacter}
      secondaryCharacter={secondaryCharacter}
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
      onCancelRegenerations={onCancelRegenerations}
      isRegenerationRunning={isRegenerationRunning}
      regenerationQueue={regenerationQueue}
      getCurrentVariantContent={getCurrentVariantContent}
      getVariantCount={getVariantCount}
      navigateVariant={navigateVariant}
      currentVariantIndex={currentVariantIndex}
      formatModelName={formatModelName}
      isGenerating={isGenerating}
    />
  );
})}
              {showTypingIndicator && (
                <div className="flex items-start gap-3 my-4 animate-in fade-in duration-300">
                  <div className="flex items-center gap-1.5 rounded-2xl bg-muted border border-border px-4 py-3">
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

<div className="border-t border-[var(--chat-focus-border)] p-2 md:p-4 backdrop-blur-md" style={{ background: 'rgba(0,0,0,0.5)' }}>
    <div className="max-w-6xl mx-auto w-full">
        <FocusModeInputForm
            ref={focusModeInputRef}
            onSubmit={handleSubmit}
            isGenerating={isGenerating}
            modelReady={modelReady}
            sttEnabled={sttEnabled}
            isRecording={isRecording}
            isTranscribing={isTranscribing}
            onMicClick={onFocusModeMicClick}
        />
    </div>
</div>
    </div>
  );
};

export default React.memo(FocusModeOverlay);
