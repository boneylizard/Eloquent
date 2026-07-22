import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { X, Star, Send, Clock, Heart, Loader2 } from 'lucide-react';
import { usePool } from '../../contexts/PoolContext';

const ROUND_DURATION = 180000;

export default function SpeedDatingEvent({ onClose }) {
  const { speedDatingSession, rateSpeedDatingRound, closeSpeedDating, sendBreakoutMessage, bookDate } = usePool();
  const [roundTimeLeft, setRoundTimeLeft] = useState(ROUND_DURATION);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [showRating, setShowRating] = useState(false);
  const [starRating, setStarRating] = useState(0);
  const bottomRef = useRef(null);
  const timerRef = useRef(null);

  const session = speedDatingSession;
  const currentChar = session?.characters?.[session.currentRound];

  useEffect(() => {
    if (!session || session.complete) return;
    setMessages(session.messages || []);
  }, [session]);

  useEffect(() => {
    if (!session || session.complete || showRating) return;
    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - session.roundStartTime;
      const left = Math.max(0, ROUND_DURATION - elapsed);
      setRoundTimeLeft(left);
      if (left <= 0) {
        clearInterval(timerRef.current);
        setShowRating(true);
      }
    }, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [session, session?.currentRound, session?.roundStartTime, showRating]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !currentChar) return;
    setSending(true);
    setInput('');
    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: text, created_at: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const reply = await sendBreakoutMessage(currentChar, text, history, {});
      if (reply) {
        const charMsg = { id: `c-${Date.now()}`, role: 'assistant', content: reply, created_at: new Date().toISOString() };
        setMessages(prev => [...prev, charMsg]);
      }
    } catch {}
    setSending(false);
  }, [input, sending, currentChar, messages, sendBreakoutMessage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleRate = useCallback((rating) => {
    if (!currentChar?.id) return;
    rateSpeedDatingRound(currentChar.id, rating);
    setStarRating(0);
    setShowRating(false);
    setRoundTimeLeft(ROUND_DURATION);
  }, [currentChar, rateSpeedDatingRound]);

  const minutes = Math.floor(roundTimeLeft / 60000);
  const seconds = Math.floor((roundTimeLeft % 60000) / 1000);
  const timerPct = (roundTimeLeft / ROUND_DURATION) * 100;
  const timerColor = roundTimeLeft < 30000 ? 'text-red-400' : roundTimeLeft < 60000 ? 'text-amber-400' : 'text-emerald-400';

  if (!session) return null;

  if (session.complete) {
    const topCharId = session.topMatchId;
    const topChar = session.characters?.find(c => c.id === topCharId);
    const ratings = session.ratings || {};
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose || closeSpeedDating} />
        <div className="relative bg-card border rounded-2xl w-full max-w-md max-h-[90vh] overflow-y-auto shadow-2xl animate-in slide-in-from-bottom duration-300 p-6">
          <div className="text-center mb-6">
            <Heart className="w-10 h-10 text-red-400 mx-auto mb-2" />
            <h2 className="text-lg font-bold">Speed Dating Results</h2>
            <p className="text-xs text-muted-foreground mt-1">All rounds complete</p>
          </div>
          <div className="space-y-2">
            {session.characters?.map(char => {
              const rating = ratings[char.id] || 0;
              const isTop = char.id === topCharId;
              return (
                <div key={char.id} className={`flex items-center justify-between p-3 rounded-xl ${isTop ? 'bg-primary/10 border border-primary/30' : 'bg-muted/30'}`}>
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-muted overflow-hidden flex items-center justify-center text-xs font-bold">
                      {char.avatar ? <img src={char.avatar} alt="" className="w-full h-full object-cover" /> : char.name?.[0] || '?'}
                    </div>
                    <div>
                      <span className="text-sm font-semibold">{char.name}</span>
                      {isTop && <span className="text-[9px] text-primary ml-1.5">★ Best Match</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-0.5">
                    {[1,2,3,4,5].map(s => (
                      <Star key={s} className={`w-3.5 h-3.5 ${s <= rating ? 'text-amber-400 fill-amber-400' : 'text-muted-foreground/30'}`} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          {topChar && (
            <button
              onClick={() => { bookDate(topChar, 'casual'); onClose?.(); closeSpeedDating(); }}
              className="mt-6 w-full py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors"
            >
              Book Date with {topChar.name}
            </button>
          )}
          <button
            onClick={() => { onClose?.(); closeSpeedDating(); }}
            className="mt-2 w-full py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  if (showRating) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
        <div className="relative bg-card border rounded-2xl w-full max-w-sm shadow-2xl animate-in slide-in-from-bottom duration-300 p-6 text-center">
          <h3 className="text-lg font-bold mb-1">Rate {currentChar?.name}</h3>
          <p className="text-xs text-muted-foreground mb-4">How was your conversation?</p>
          <div className="flex justify-center gap-2 mb-6">
            {[1,2,3,4,5].map(s => (
              <button key={s} onClick={() => { handleRate(s); }} className="p-1 transition-transform hover:scale-110">
                <Star className={`w-8 h-8 ${s <= starRating ? 'text-amber-400 fill-amber-400' : 'text-muted-foreground/20'}`}
                  onMouseEnter={() => setStarRating(s)} onMouseLeave={() => setStarRating(0)} />
              </button>
            ))}
          </div>
          <button
            onClick={() => handleRate(Math.max(1, starRating))}
            disabled={starRating === 0}
            className="px-6 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-colors disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose || closeSpeedDating} />
      <div className="relative bg-card border rounded-2xl w-full max-w-lg max-h-[90vh] shadow-2xl animate-in slide-in-from-bottom duration-300 flex flex-col">
        <div className="px-4 py-3 border-b border-border/30 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground">Round</span>
              <span className="text-sm font-bold">{session.currentRound + 1} of {session.totalRounds}</span>
            </div>
            <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
            {currentChar && (
              <div className="flex items-center gap-1.5">
                <div className="w-6 h-6 rounded-full bg-muted overflow-hidden flex items-center justify-center text-[8px] font-bold">
                  {currentChar.avatar ? <img src={currentChar.avatar} alt="" className="w-full h-full object-cover" /> : currentChar.name?.[0] || '?'}
                </div>
                <span className="text-sm font-semibold">{currentChar.name}</span>
              </div>
            )}
          </div>
          <div className={`flex items-center gap-1 ${timerColor}`}>
            <Clock className="w-3.5 h-3.5" />
            <span className="text-xs font-mono font-bold">{minutes}:{seconds.toString().padStart(2, '0')}</span>
          </div>
        </div>
        <div className="h-1 bg-muted">
          <div className={`h-full transition-all duration-1000 ${timerColor.replace('text-', 'bg-')}`} style={{ width: `${timerPct}%` }} />
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center text-xs text-muted-foreground">
              <p>Chat with {currentChar?.name}!</p>
              <p className="text-[10px] mt-1">Timer is running — say something.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={msg.id || i} className={`flex items-end gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              {msg.role !== 'user' && (
                <div className="w-6 h-6 rounded-full bg-muted overflow-hidden flex items-center justify-center text-[8px] font-bold shrink-0">
                  {currentChar?.avatar ? <img src={currentChar.avatar} alt="" className="w-full h-full object-cover" /> : currentChar?.name?.[0] || '?'}
                </div>
              )}
              <div className="max-w-[75%]">
                <div className={`px-3 py-2 text-xs leading-relaxed rounded-2xl ${msg.role === 'user' ? 'bg-primary text-primary-foreground rounded-br-md' : 'bg-muted text-foreground rounded-bl-md'}`}>
                  {msg.content}
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <div className="border-t border-border/30 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
              placeholder={`Message ${currentChar?.name || '...'}...`} disabled={sending}
              className="flex-1 h-9 text-xs bg-muted border rounded-full px-4 outline-none focus:border-primary/50 transition-colors disabled:opacity-50" />
            <button onClick={handleSend} disabled={!input.trim() || sending}
              className="w-9 h-9 rounded-full bg-primary text-primary-foreground flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40">
              {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
