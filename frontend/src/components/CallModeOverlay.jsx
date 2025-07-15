import React, { useState, useEffect, useCallback } from 'react';
import { useApp } from '../contexts/AppContext';

const CallModeOverlay = ({ 
  isActive, 
  onExit, 
  activeCharacter, 
  isPlayingAudio, 
  isRecording, 
  isTranscribing,
  PRIMARY_API_URL 
}) => {
  const { startRecording, stopRecording, sendMessage, stopTTS, handleStopGeneration } = useApp();
  const [isPulsing, setIsPulsing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechIntensity, setSpeechIntensity] = useState(0.5);
  const [rippleDelays, setRippleDelays] = useState([0, 0.2, 0.4, 0.6]);

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

  // Handle ESC key to exit and SPACE for recording toggle and SHIFT to stop audio
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && isActive) {
        onExit();
      } else if (event.key === ' ' && isActive) {
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
  }, [isActive, onExit, isProcessing, handleRecord, isSpeaking, stopTTS, handleStopGeneration]);

  if (!isActive) return null;

  const avatarUrl = activeCharacter?.avatar 
    ? (activeCharacter.avatar.startsWith('http') 
        ? activeCharacter.avatar 
        : `${PRIMARY_API_URL}/static/${activeCharacter.avatar}`)
    : null;

  return (
    <div className="fixed inset-0 z-[9999] bg-black flex items-center justify-center">
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
          <path d="M18 6L6 18"/>
          <path d="M6 6l12 12"/>
        </svg>
      </button>

      {/* Main Content */}
      <div className="flex flex-col items-center justify-center space-y-8 px-8">
        {/* Avatar with NEW Speaking Animation */}
        <div className="relative">
          {/* NEW: Speaking Animation - Breathing Glow + Audio Ripples */}
          {isSpeaking && (
            <>
              {/* Breathing Glow with randomized intensity - MORE SUBTLE */}
              <div className="absolute inset-0 rounded-full">
                <div 
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400/10 to-cyan-400/10 animate-pulse scale-110" 
                  style={{ opacity: speechIntensity * 0.5 }}
                />
                <div 
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-300/5 to-white/5 animate-ping scale-115" 
                  style={{ opacity: speechIntensity * 0.4 }}
                />
              </div>
              
              {/* Audio Ripples with randomized delays - MORE SUBTLE */}
              <div className="absolute inset-0 rounded-full flex items-center justify-center">
                <div 
                  className="absolute w-full h-full rounded-full border border-blue-400/15 animate-ping" 
                  style={{ 
                    animationDelay: `${rippleDelays[0]}s`,
                    opacity: speechIntensity * 0.4
                  }} 
                />
                <div 
                  className="absolute w-[110%] h-[110%] rounded-full border border-blue-300/10 animate-ping" 
                  style={{ 
                    animationDelay: `${rippleDelays[1]}s`,
                    opacity: speechIntensity * 0.35
                  }} 
                />
                <div 
                  className="absolute w-[120%] h-[120%] rounded-full border border-cyan-400/8 animate-ping" 
                  style={{ 
                    animationDelay: `${rippleDelays[2]}s`,
                    opacity: speechIntensity * 0.3
                  }} 
                />
                <div 
                  className="absolute w-[130%] h-[130%] rounded-full border border-blue-200/5 animate-ping" 
                  style={{ 
                    animationDelay: `${rippleDelays[3]}s`,
                    opacity: speechIntensity * 0.25
                  }} 
                />
              </div>
            </>
          )}

          {/* OLD: Keep the original subtle animation for backwards compatibility, but only when not speaking */}
          {!isSpeaking && (
            <div className="absolute inset-0 rounded-full">
              <div className={`absolute inset-0 rounded-full border-2 border-blue-400/30 
                               ${isPulsing ? 'animate-ping' : ''} scale-125`}></div>
              {isPulsing && (
                <div className="absolute inset-0 rounded-full bg-blue-400/10 
                               animate-pulse scale-110"></div>
              )}
              {isPulsing && (
                <div className="absolute inset-0 rounded-full border border-blue-300/50 
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
                <path d="M12 1c-3 0-5 2-5 5v6c0 3 2 5 5 5s5-2 5-5V6c0-3-2-5-5-5z"/>
                <path d="M19 10v2c0 7-3 9-7 9s-7-2-7-9v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
            )}
          </button>

          {/* Instructions - UNCHANGED */}
        <div className="text-center text-white/60 space-y-1">
          <p className="text-sm">
            {isRecording ? "Click to stop & send message" : "Click to start recording"}
          </p>
          <p className="text-xs">
            Press <kbd className="bg-white/20 px-2 py-1 rounded text-xs">ESC</kbd> to exit Call Mode
          </p>
        </div>
      </div>
    </div>
  );
};

export default CallModeOverlay;