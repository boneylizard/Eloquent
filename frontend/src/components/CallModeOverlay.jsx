import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';

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
  onRegenerate
}) => {
  const { startRecording, stopRecording, sendMessage, stopTTS, handleStopGeneration, isGenerating } = useApp();
  const [isPulsing, setIsPulsing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechIntensity, setSpeechIntensity] = useState(0.5);
  const [rippleDelays, setRippleDelays] = useState([0, 0.2, 0.4, 0.6]);
  const [showControlPanel, setShowControlPanel] = useState(false);

  // Handle Reroll (Regenerate last bot message)
  const handleReroll = useCallback(() => {
    if (isGenerating || isTranscribing) return;

    // Find the last bot message to regenerate
    const lastBotMsg = [...(messages || [])].reverse().find(m => m.role === 'bot');
    if (lastBotMsg && onRegenerate) {
      console.log(`🔄 [CallMode] Rerolling message ${lastBotMsg.id}`);
      // Stop current playback if it's speaking
      stopTTS();
      handleStopGeneration();

      onRegenerate(lastBotMsg.id);
    } else {
      console.warn("⚠️ [CallMode] No bot message found to reroll");
    }
  }, [messages, onRegenerate, isGenerating, isTranscribing, stopTTS, handleStopGeneration]);

  // Sync pulsing animation with TTS playback
  useEffect(() => {
    setIsPulsing(!!isPlayingAudio);
  }, [isPlayingAudio]);

  // NEW: Track processing state until audio starts playing
  useEffect(() => {
    if (isTranscribing) {
      setIsProcessing(true);
    }
  }, [isTranscribing]);

  // NEW: Monitor when streaming audio starts playing
  useEffect(() => {
    const checkAudioPlaying = () => {
      if (window.streamingAudioPlaying) {
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

  // NEW: Randomize speech visual effects when speaking
  useEffect(() => {
    if (!isSpeaking) return;

    const randomizeEffects = () => {
      // Randomize speech intensity (0.15 to 0.4 for more subtle variation)
      setSpeechIntensity(0.15 + Math.random() * 0.25);

      // Randomize ripple delays (0 to 0.8s with some variation)
      setRippleDelays([
        0,
        0.1 + Math.random() * 0.3,
        0.2 + Math.random() * 0.4,
        0.3 + Math.random() * 0.5
      ]);
    };

    // Initial randomization
    randomizeEffects();

    // Randomize every 200-800ms to simulate speech rhythm changes
    const randomInterval = () => {
      const nextDelay = 200 + Math.random() * 600;
      setTimeout(() => {
        if (window.streamingAudioPlaying) {
          randomizeEffects();
          randomInterval();
        }
      }, nextDelay);
    };

    randomInterval();
  }, [isSpeaking]);

  // Handle recording with auto-send
  const handleRecord = useCallback(async () => {
    if (isRecording) {
      // Stop recording and auto-send
      await stopRecording(async (transcript) => {
        if (transcript && transcript.trim()) {
          await sendMessage(transcript.trim());
        }
      });
    } else {
      // Start recording
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording, sendMessage]);

  // Handle keyboard interaction
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check if user is in an input field (unlikely in call mode but just in case)
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;

      if (event.key === 'Escape' && isActive) {
        if (showControlPanel) {
          setShowControlPanel(false);
        } else {
          onExit();
        }
      } else if (event.key === ' ' && isActive && !showControlPanel) {
        // Prevent default space bar behavior (scrolling)
        event.preventDefault();
        // Don't trigger if already processing
        if (!isProcessing) {
          handleRecord();
        }
      } else if (event.key === 'Shift' && isActive) {
        // Stop TTS playback
        event.preventDefault();
        if (window.streamingAudioPlaying || isSpeaking) {
          stopTTS();
          handleStopGeneration();
        }
      } else if (event.key === 'Tab' && isActive) {
        // Toggle control panel
        event.preventDefault();
        setShowControlPanel(prev => !prev);
      } else if ((event.key === 'r' || event.key === 'R') && isActive) {
        // Reroll shortcut
        event.preventDefault();
        handleReroll();
      }
    };

    if (isActive) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'auto';
    };
  }, [isActive, onExit, isProcessing, handleRecord, isSpeaking, stopTTS, handleStopGeneration, showControlPanel]);

  if (!isActive) return null;

  const avatarUrl = activeCharacter?.avatar
    ? (activeCharacter.avatar.startsWith('http')
      ? activeCharacter.avatar
      : `${PRIMARY_API_URL}/static/${activeCharacter.avatar}`)
    : null;

  return (
    <div className="fixed inset-0 z-[9999] bg-black flex items-center justify-center">
      {/* Control Panel Toggle Button */}
      <button
        onClick={() => setShowControlPanel(!showControlPanel)}
        className={`absolute top-6 left-6 w-12 h-12 rounded-full flex items-center justify-center text-white transition-all duration-200 z-10
                   hover:scale-110 active:scale-95 ${showControlPanel ? 'bg-blue-500/50' : 'bg-white/10 hover:bg-white/20'}`}
        title="Control Panel (TAB)"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>

      {/* Control Panel Slide-out */}
      <div
        className={`absolute left-0 top-0 bottom-0 w-72 bg-gradient-to-r from-gray-900/95 to-gray-900/90 backdrop-blur-lg
                    border-r border-white/10 transition-transform duration-300 ease-out z-20
                    ${showControlPanel ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="p-6 pt-20 space-y-4">
          <h3 className="text-white text-lg font-semibold mb-4 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="3" y1="9" x2="21" y2="9" />
              <line x1="9" y1="21" x2="9" y2="9" />
            </svg>
            Control Panel
          </h3>

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

          {/* Reroll Button */}
          <button
            onClick={() => {
              setShowControlPanel(false);
              handleReroll();
            }}
            disabled={isGenerating || isTranscribing || !messages?.some(m => m.role === 'bot')}
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

          {/* Stop Speaking Button */}
          {(isSpeaking || window.streamingAudioPlaying) && (
            <button
              onClick={() => {
                stopTTS();
                handleStopGeneration();
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
            </div>
          </div>
        </div>
      </div>

      {/* Exit Button */}
      <button
        onClick={onExit}
        className="absolute top-6 right-6 w-12 h-12 rounded-full bg-white/10 hover:bg-white/20 
                   flex items-center justify-center text-white transition-all duration-200 z-10
                   hover:scale-110 active:scale-95"
        title="Exit Call Mode (ESC)"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M18 6L6 18" />
          <path d="M6 6l12 12" />
        </svg>
      </button>

      {/* Main Content */}
      <div className="flex flex-col items-center justify-center space-y-8 px-8">
        {/* Avatar with NEW Speaking Animation */}
        <div className="relative">
          {/* NEW: Speaking Animation - Breathing Glow + Audio Ripples */}
          {isSpeaking && (
            <>
              {/* Breathing Glow with randomized intensity - Enhanced visibility */}
              <div className="absolute inset-0 rounded-full">
                <div
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/25 to-cyan-400/25 animate-pulse scale-110"
                  style={{ opacity: speechIntensity * 0.8 }}
                />
                <div
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-300/15 to-white/15 animate-ping scale-115"
                  style={{ opacity: speechIntensity * 0.7 }}
                />
              </div>

              {/* Audio Ripples with randomized delays - Enhanced visibility */}
              <div className="absolute inset-0 rounded-full flex items-center justify-center">
                <div
                  className="absolute w-full h-full rounded-full border-2 border-blue-400/35 animate-ping"
                  style={{
                    animationDelay: `${rippleDelays[0]}s`,
                    opacity: speechIntensity * 0.7
                  }}
                />
                <div
                  className="absolute w-[110%] h-[110%] rounded-full border-2 border-blue-300/30 animate-ping"
                  style={{
                    animationDelay: `${rippleDelays[1]}s`,
                    opacity: speechIntensity * 0.65
                  }}
                />
                <div
                  className="absolute w-[120%] h-[120%] rounded-full border-2 border-cyan-400/25 animate-ping"
                  style={{
                    animationDelay: `${rippleDelays[2]}s`,
                    opacity: speechIntensity * 0.6
                  }}
                />
                <div
                  className="absolute w-[130%] h-[130%] rounded-full border-2 border-blue-200/20 animate-ping"
                  style={{
                    animationDelay: `${rippleDelays[3]}s`,
                    opacity: speechIntensity * 0.55
                  }}
                />
              </div>
            </>
          )}

          {/* OLD: Keep the original subtle animation for backwards compatibility, but only when not speaking */}
          {!isSpeaking && (
            <div className="absolute inset-0 rounded-full">
              <div className={`absolute inset-0 rounded-full border-2 border-blue-400/45 
                               ${isPulsing ? 'animate-ping' : ''} scale-125`}></div>
              {isPulsing && (
                <div className="absolute inset-0 rounded-full bg-blue-400/20 
                               animate-pulse scale-110"></div>
              )}
              {isPulsing && (
                <div className="absolute inset-0 rounded-full border-2 border-blue-300/60 
                               animate-ping scale-105"></div>
              )}
            </div>
          )}

          {/* Avatar Image - UNCHANGED */}
          <div className="relative w-80 h-80 rounded-full overflow-hidden border-4 border-white/30 shadow-2xl
                         transform transition-transform duration-300 hover:scale-105">
            {avatarUrl ? (
              <img
                src={avatarUrl}
                alt={activeCharacter?.name || "Character"}
                className="w-full h-full object-cover"
                onError={(e) => {
                  console.warn(`Avatar failed to load: ${avatarUrl}`);
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
            ) : null}

            {/* Fallback Avatar */}
            <div
              className="w-full h-full bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 
                        flex items-center justify-center text-white text-8xl font-bold"
              style={{ display: avatarUrl ? 'none' : 'flex' }}
            >
              {activeCharacter?.name?.charAt(0)?.toUpperCase() || 'A'}
            </div>
          </div>

          {/* Processing indicator - UNCHANGED */}
          {isTranscribing && (
            <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 
                           bg-blue-500 text-white px-3 py-1 rounded-full text-sm animate-pulse">
              Processing...
            </div>
          )}
        </div>

        {/* Character Name - UNCHANGED */}
        {activeCharacter?.name && (
          <h1 className="text-4xl font-bold text-white text-center tracking-wide">
            {activeCharacter.name}
          </h1>
        )}

        {/* Status Text - MODIFIED to use isProcessing instead of just isTranscribing */}
        <div className="text-center text-white/80 space-y-3 max-w-md">
          {isProcessing && (
            <p className="text-xl animate-pulse font-medium">Processing your message...</p>
          )}
          {isRecording && !isProcessing && (
            <p className="text-xl font-medium flex items-center justify-center gap-2">
              <span className="w-3 h-3 bg-red-400 rounded-full animate-pulse"></span>
              Recording...
            </p>
          )}
          {isSpeaking && !isRecording && !isProcessing && (
            <p className="text-xl font-medium flex items-center justify-center gap-2">
              <span className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse"></span>
              Speaking...
            </p>
          )}
          {!isRecording && !isProcessing && !isSpeaking && (
            <p className="text-xl font-medium">Ready for conversation</p>
          )}
        </div>
        {/* Record Button - UNCHANGED but with new styles */}
        <button
          onClick={handleRecord}
          disabled={isTranscribing}
          className={`w-20 h-20 rounded-full border-4 flex items-center justify-center text-white 
                 transition-all duration-200 transform hover:scale-110 active:scale-95
                 ${isRecording
              ? 'bg-red-500 border-red-300 hover:bg-red-600'
              : 'bg-blue-500 border-blue-300 hover:bg-blue-600'
            } ${isTranscribing ? 'opacity-50 cursor-not-allowed' : ''}`}
          title={isRecording ? "Stop & Send" : "Start Recording"}
        >
          {isRecording ? (
            <div className="relative">
              <div className="w-6 h-6 bg-white rounded"></div>
              {/* Subtle pulse overlay */}
              <div className="absolute inset-0 rounded-full border-2 border-white/30 animate-[pulse_3s_ease-in-out_infinite] opacity-40"></div>
            </div>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="32"
              height="32"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 1c-3 0-5 2-5 5v6c0 3 2 5 5 5s5-2 5-5V6c0-3-2-5-5-5z" />
              <path d="M19 10v2c0 7-3 9-7 9s-7-2-7-9v-2" />
              <line x1="12" y1="19" x2="12" y2="23" />
              <line x1="8" y1="23" x2="16" y2="23" />
            </svg>
          )}
        </button>

        {/* Instructions - UPDATED */}
        <div className="text-center text-white/60 space-y-1">
          <p className="text-sm">
            {isRecording ? "Click to stop & send message" : "Click to start recording"}
          </p>
          <p className="text-xs">
            Press <kbd className="bg-white/20 px-2 py-1 rounded text-xs">TAB</kbd> for controls • <kbd className="bg-white/20 px-2 py-1 rounded text-xs">ESC</kbd> to exit
          </p>
        </div>
      </div>
    </div>
  );
};

export default CallModeOverlay;
