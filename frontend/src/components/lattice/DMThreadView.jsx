import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ArrowLeft, Send, Loader2, Volume2, VolumeX, Mic, MicOff } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';
import { useApp } from '../../contexts/AppContext';
import { getBackendUrl } from '../../config/api';

function formatTime(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return new Date(iso).toLocaleDateString();
}

function getThreadMessages(thread) {
  if (Array.isArray(thread?.messages) && thread.messages.length) return thread.messages;

  const content = thread?.last_message?.content?.trim();
  if (!content) return [];

  return [{
    id: `dm-msg-preview-${thread.id || 'thread'}`,
    role: thread.last_message.role || 'character',
    content,
    character_name: thread.character_name,
    created_at: thread.last_message.timestamp || thread.created_at,
  }];
}

export default function DMThreadView({ thread, onClose }) {
  const { sendDMMessage, closeDMThread } = usePool();
  const { playTTS, stopTTS, isPlayingAudio, startRecording, stopRecording, isRecording, isTranscribing, sttEnabled } = useApp();
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);
  const apiUrl = getBackendUrl();

  // Fetch full thread with messages from backend on open
  useEffect(() => {
    if (!thread?.id) return;
    setLoading(true);
    fetch(`${apiUrl}/lattice/dm-thread/${thread.id}`)
      .then(r => r.json())
      .then(data => {
        if (data.status === 'success' && data.thread) {
          setMessages(getThreadMessages(data.thread));
        } else {
          setMessages(getThreadMessages(thread));
        }
      })
      .catch(() => {
        setMessages(getThreadMessages(thread));
      })
      .finally(() => setLoading(false));
  }, [thread?.id, apiUrl]);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !thread?.id) return;
    setSending(true);
    setInput('');
    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: text, created_at: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    try {
      const result = await sendDMMessage(thread.id, text, 'user');
      if (result) {
        if (Array.isArray(result)) {
          setMessages(prev => [...prev, ...result.filter(m => m && m.role === 'character')]);
        } else if (result.role === 'character') {
          setMessages(prev => [...prev, result]);
        }
      }
    } catch { }
    setSending(false);
  }, [input, sending, thread?.id, sendDMMessage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ASR: toggle microphone recording
  const handleMicToggle = useCallback(async () => {
    if (isRecording) {
      await stopRecording(async (transcript) => {
        const cleaned = String(transcript || '').trim();
        if (cleaned) {
          setInput(cleaned);
          await sendDMMessage(thread.id, cleaned, 'user');
          setInput('');
        }
      });
    } else {
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording, sendDMMessage, thread?.id]);

  // TTS: play character message with voice
  const handlePlayMessage = useCallback(async (msgId, text) => {
    if (isPlayingAudio === msgId) {
      stopTTS();
      return;
    }
    const voice = thread?.character_snapshot?.ttsVoice || thread?.character_snapshot?.tts_voice;
    await playTTS(msgId, text, voice && voice !== 'default' ? { ttsVoice: voice } : null);
  }, [playTTS, stopTTS, isPlayingAudio, thread]);

  const character = thread?.character_snapshot || {
    name: thread?.character_name,
    avatar: thread?.character_avatar,
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border/30 bg-muted/30 shrink-0">
        <button onClick={onClose || closeDMThread} className="w-7 h-7 rounded-full hover:bg-muted flex items-center justify-center transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="w-7 h-7 rounded-full overflow-hidden bg-muted flex items-center justify-center text-[9px] font-bold shrink-0">
          {character.avatar ? (
            <img src={character.avatar} alt="" className="w-full h-full object-cover" />
          ) : (
            character.name?.[0] || '?'
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold truncate">{character.name || 'Character'}</div>
          <div className="text-[9px] text-muted-foreground">Direct Message</div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {loading && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        )}
        {!loading && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-xs text-muted-foreground space-y-2">
            <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-lg font-bold text-muted-foreground/40">
              {character.name?.[0] || '?'}
            </div>
            <p>DM with {character.name}. Send a message to begin.</p>
          </div>
        )}
        {messages.map((msg, i) => {
          const isUser = msg.role === 'user';
          const msgId = msg.id || `dm-msg-${i}`;
          const isThisPlaying = isPlayingAudio === msgId;
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
                  {msg.content}
                </div>
                <div className={`mt-0.5 ${isUser ? 'text-right pr-1' : 'text-left pl-1'}`}>
                  <span className="text-[9px] text-muted-foreground/50">
                    {msg.created_at ? formatTime(msg.created_at) : ''}
                  </span>
                </div>
              </div>
              {!isUser && msg.content && (
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
        {sending && (
          <div className="flex items-end gap-2">
            <div className="w-7 h-7 rounded-full shrink-0 bg-muted flex items-center justify-center text-[9px] font-bold text-muted-foreground overflow-hidden">
              {character.avatar ? (
                <img src={character.avatar} alt="" className="w-full h-full object-cover" />
              ) : (
                character.name?.[0] || '?'
              )}
            </div>
            <div className="px-3 py-2 text-xs rounded-2xl bg-muted text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: '300ms' }} />
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border/30 px-3 py-2.5 shrink-0">
        <div className="flex items-center gap-2">
          {sttEnabled && (
            <button
              onClick={handleMicToggle}
              disabled={isTranscribing || sending}
              className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-colors ${
                isRecording ? 'bg-red-500 text-white animate-pulse' : 'hover:bg-muted text-muted-foreground'
              } disabled:opacity-40`}
              title={isRecording ? 'Stop recording and send' : 'Speak'}
            >
              {isTranscribing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : isRecording ? (
                <MicOff className="w-4 h-4" />
              ) : (
                <Mic className="w-4 h-4" />
              )}
            </button>
          )}
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRecording ? 'Listening...' : `Message ${character.name}...`}
            disabled={sending || isRecording}
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
