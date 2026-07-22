import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Mic, MicOff, Volume2, VolumeX } from 'lucide-react';
import { useApp } from '../../contexts/AppContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { splitStreamingContent, nextStreamingTokenKey } from '../../utils/streamingText';

function formatReadTime(ts) {
  const diff = Date.now() - ts;
  if (diff < 60000) return 'Seen just now';
  if (diff < 3600000) return `Seen ${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `Seen ${Math.floor(diff / 3600000)}h ago`;
  return `Seen ${Math.floor(diff / 86400000)}d ago`;
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '0ms' }} />
      <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '150ms' }} />
      <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '300ms' }} />
    </span>
  );
}

export default function BreakoutChat({ character, conversationId, onSendMessage, onBookDate, messages, readReceipts = {}, isTyping = false }) {
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);

  const { startRecording, stopRecording, isRecording, isTranscribing, sttEnabled, playTTS, stopTTS, isPlayingAudio, settings, updateSettings } = useApp();

  const getVoiceOverrides = useCallback(() => {
    const voice = character?.ttsVoice || character?.tts_voice;
    if (voice && voice !== 'default') return { ttsVoice: voice };
    return null;
  }, [character]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput('');
    try {
      await onSendMessage(text);
    } finally {
      setSending(false);
    }
  }, [input, sending, onSendMessage]);

  const handleMicToggle = useCallback(async () => {
    if (isRecording) {
      await stopRecording(async (transcript) => {
        const cleaned = String(transcript || '').trim();
        if (cleaned) {
          setInput(cleaned);
          await onSendMessage(cleaned);
          setInput('');
        }
      });
    } else {
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording, onSendMessage]);

  const handlePlayMessage = useCallback(async (msgId, text) => {
    if (isPlayingAudio === msgId) {
      stopTTS();
      return;
    }
    await playTTS(msgId, text, getVoiceOverrides());
  }, [playTTS, stopTTS, isPlayingAudio, getVoiceOverrides]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const chatMsgs = messages || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {chatMsgs.length === 0 && !isTyping && (
          <div className="flex flex-col items-center justify-center h-full text-center text-xs text-muted-foreground space-y-2">
            <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-lg font-bold text-muted-foreground/40">
              {character.name?.[0] || '?'}
            </div>
            <p>Breakout room started with {character.name}.</p>
            <p>You have 30 minutes. Send a message to begin.</p>
          </div>
        )}
          {chatMsgs.map((msg, i) => {
          const isUser = msg.role === 'user';
          const msgId = msg.id || `bm-${i}`;
          const isThisPlaying = isPlayingAudio === msgId;
          const readTs = readReceipts[msgId];
          return (
            <div key={msgId} className={`flex items-end gap-2 ${isUser ? 'flex-row-reverse' : ''}`}>
              {!isUser && (
                <div className="w-7 h-7 rounded-full shrink-0 bg-muted flex items-center justify-center text-[9px] font-bold text-muted-foreground overflow-hidden">
                  {character.avatar ? (
                    <img src={character.avatar} alt="" className="w-full h-full object-cover" />
                  ) : (
                    character.name?.[0] || '?'
                  )}
                </div>
              )}
              <div className="max-w-[75%]">
                <div className={`px-3 py-2 text-xs leading-relaxed rounded-2xl ${
                  isUser
                    ? 'bg-primary text-primary-foreground rounded-br-md'
                    : 'bg-muted text-foreground rounded-bl-md'
                }`}>
                  {isUser ? (
                    msg.content
                  ) : msg.isStreaming && msg.content ? (
                    (() => {
                      const { completed, streaming } = splitStreamingContent(msg.content, true);
                      const tokenKey = nextStreamingTokenKey();
                      return (
                        <div className="chat-prose max-w-none break-words">
                          {completed && (
                            <ReactMarkdown remarkPlugins={[remarkGfm]} className="max-w-none">
                              {completed}
                            </ReactMarkdown>
                          )}
                          <span key={tokenKey} className="streaming-token">{streaming}</span>
                          <span className="blinking-cursor" aria-hidden="true" />
                        </div>
                      );
                    })()
                  ) : (
                    <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-sm chat-prose max-w-none break-words">
                      {msg.content}
                    </ReactMarkdown>
                  )}
                </div>
                {isUser && readTs && (
                  <div className="text-right mt-0.5 pr-1">
                    <span className="text-[9px] text-muted-foreground/50">{formatReadTime(readTs)}</span>
                  </div>
                )}
              </div>
              {!isUser && character?.voice_id && msg.content && !msg.isStreaming && (
                <button
                  onClick={() => handlePlayMessage(msgId, msg.content)}
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 transition-colors ${isThisPlaying ? 'bg-primary/20 text-primary' : 'hover:bg-muted text-muted-foreground'}`}
                  title={isThisPlaying ? 'Stop voice' : 'Play with voice'}
                >
                  {isThisPlaying ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                </button>
              )}
            </div>
          );
        })}

        {isTyping && (
          <div className="flex items-end gap-2">
            <div className="w-7 h-7 rounded-full shrink-0 bg-muted flex items-center justify-center overflow-hidden">
              {character.avatar ? (
                <img src={character.avatar} alt="" className="w-full h-full object-cover" />
              ) : (
                character.name?.[0] || '?'
              )}
            </div>
            <div className="bg-muted rounded-2xl rounded-bl-md px-4 py-2.5 flex items-center gap-2">
              <TypingDots />
              <span className="text-[10px] text-muted-foreground/60">typing...</span>
            </div>
          </div>
        )}

        {sending && !isTyping && (
          <div className="flex items-end gap-2">
            <div className="w-7 h-7 rounded-full shrink-0 bg-muted flex items-center justify-center overflow-hidden">
              {character.avatar ? (
                <img src={character.avatar} alt="" className="w-full h-full object-cover" />
              ) : (
                character.name?.[0] || '?'
              )}
            </div>
            <div className="bg-muted rounded-2xl rounded-bl-md px-3 py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border/30 px-3 py-2.5">
        <div className="flex items-center gap-2">
          {sttEnabled && (
            <button
              onClick={handleMicToggle}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors shrink-0 ${isRecording ? 'bg-red-500 text-white animate-pulse' : 'bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground'}`}
              title={isRecording ? 'Recording... tap to stop' : 'Voice input'}
            >
              {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>
          )}
          {settings?.ttsEnabled && isPlayingAudio && (
            <button
              onClick={() => stopTTS()}
              className="w-8 h-8 rounded-full flex items-center justify-center transition-colors shrink-0 bg-red-500/10 text-red-500 hover:bg-red-500/20"
              title="Stop audio"
            >
              <VolumeX className="w-4 h-4" />
            </button>
          )}
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Message ${character.name}...`}
            disabled={sending}
            className="flex-1 h-9 text-xs bg-muted border rounded-full px-4 outline-none focus:border-primary/50 transition-colors disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || sending}
            className="w-9 h-9 rounded-full bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40"
          >
            {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}