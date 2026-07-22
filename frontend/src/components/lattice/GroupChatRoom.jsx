import React, { useState, useRef, useEffect, useCallback } from 'react';
import { X, Send, Loader2, Users, Volume2, VolumeX } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';
import { useApp } from '../../contexts/AppContext';
import { matchModelName } from '../../utils/modelDisplayNames';

const CHAR_COLORS = ['text-pink-400', 'text-red-400', 'text-purple-400', 'text-sky-400', 'text-emerald-400'];
const CHAR_BG = ['bg-pink-500/10', 'bg-red-500/10', 'bg-purple-500/10', 'bg-sky-500/10', 'bg-emerald-500/10'];

export default function GroupChatRoom({ onClose }) {
  const { activeGroupChat, sendGroupMessage, closeGroupChat } = usePool();
  const { playTTS, stopTTS, isPlayingAudio } = useApp();
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const bottomRef = useRef(null);

  const messages = activeGroupChat?.messages || [];
  const characters = activeGroupChat?.characters || [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput('');
    await sendGroupMessage(text, 'user');
    setSending(false);
  }, [input, sending, sendGroupMessage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handlePlayTTS = useCallback((msgId, text, ttsVoice) => {
    if (isPlayingAudio === msgId) { stopTTS(); return; }
    playTTS(msgId, text, ttsVoice && ttsVoice !== 'default' ? { ttsVoice } : null);
  }, [playTTS, stopTTS, isPlayingAudio]);

  if (!activeGroupChat?.active) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose || closeGroupChat} />
      <div className="relative bg-card border rounded-2xl w-full max-w-lg max-h-[90vh] shadow-2xl animate-in slide-in-from-bottom duration-300 flex flex-col">
        <div className="px-4 py-3 border-b border-border/30 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-semibold">Group Chat</span>
          </div>
          <div className="flex items-center gap-1.5">
            {characters.map((char, i) => (
              <div key={char.id} className="flex items-center gap-1 bg-muted/40 rounded-full pl-0.5 pr-2 py-0.5">
                <div className="w-5 h-5 rounded-full bg-muted overflow-hidden flex items-center justify-center text-[6px] font-bold">
                  {char.avatar ? <img src={char.avatar} alt="" className="w-full h-full object-cover" /> : char.name?.[0] || '?'}
                </div>
                <span className={`text-[9px] font-medium ${CHAR_COLORS[i % CHAR_COLORS.length]}`}>{char.name}</span>
              </div>
            ))}
            <button onClick={onClose || closeGroupChat} className="w-7 h-7 rounded-full hover:bg-muted flex items-center justify-center ml-1">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-xs text-muted-foreground">
              <Users className="w-8 h-8 mb-2 opacity-30" />
              <p>Group chat started!</p>
              {activeGroupChat.topic && <p className="text-[10px] mt-1 italic">Topic: {activeGroupChat.topic}</p>}
            </div>
          )}
          {messages.map((msg, i) => {
            const isUser = msg.role === 'user';
            const charIndex = characters.findIndex(c => c.name === msg.characterName);
            const colorClass = CHAR_COLORS[charIndex >= 0 ? charIndex % CHAR_COLORS.length : 0];
            const bgClass = CHAR_BG[charIndex >= 0 ? charIndex % CHAR_BG.length : 0];
            const msgId = msg.id || `gc-${i}`;
            const isThisPlaying = isPlayingAudio === msgId;
            const character = characters.find(c => c.name === msg.characterName);
            return (
              <div key={msgId} className={`flex items-end gap-2 ${isUser ? 'flex-row-reverse' : ''}`}>
                {!isUser && (
                  <div className="w-7 h-7 rounded-full bg-muted overflow-hidden flex items-center justify-center text-[9px] font-bold shrink-0">
                    {msg.characterAvatar || character?.avatar ? (
                      <img src={msg.characterAvatar || character?.avatar} alt="" className="w-full h-full object-cover" />
                    ) : (
                      msg.characterName?.[0] || '?'
                    )}
                  </div>
                )}
                <div className="max-w-[70%]">
                  {!isUser && msg.characterName && (
                    <div className={`text-[9px] font-semibold mb-0.5 ml-1 ${colorClass}`}>{msg.characterName}</div>
                  )}
                  <div className={`px-3 py-2 text-xs leading-relaxed rounded-2xl ${
                    isUser
                      ? 'bg-primary text-primary-foreground rounded-br-md'
                      : `${bgClass} text-foreground rounded-bl-md border border-border/20`
                  }`}>
                    {msg.content}
                  </div>
                  {!isUser && character?.voice_id && msg.content && (
                    <button
                      onClick={() => handlePlayTTS(msgId, msg.content, character.voice_id)}
                      className={`ml-1 mt-0.5 w-5 h-5 rounded flex items-center justify-center ${isThisPlaying ? 'text-primary' : 'text-muted-foreground hover:text-foreground'}`}
                    >
                      {isThisPlaying ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>

        {sending && (
          <div className="px-4 py-1.5 flex items-center gap-2 text-[10px] text-muted-foreground">
            <Loader2 className="w-3 h-3 animate-spin" />
            Characters are responding...
          </div>
        )}

        <div className="border-t border-border/30 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Message group...`}
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
    </div>
  );
}
