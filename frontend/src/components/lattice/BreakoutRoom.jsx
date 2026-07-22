import React, { useState, useCallback, useRef } from 'react';
import { X, ArrowLeft, Star, MessageSquare, AudioLines } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import BreakoutTimer from './BreakoutTimer';
import BreakoutChat from './BreakoutChat';
import BookDateFlow from './BookDateFlow';
import VoiceQuickPicker from '../VoiceQuickPicker';
import { usePool } from '../../contexts/PoolContext';
import { useApp } from '../../contexts/AppContext';

export default function BreakoutRoom({ character, onClose }) {
  const { sendBreakoutMessage, endBreakoutRoom, bookDate, generateUserRating, rateCharacter } = usePool();
  const { settings, updateSettings, isPlayingAudio, stopTTS, startStreamingTTS, addStreamingText, endStreamingTTS } = useApp();

  const [messages, setMessages] = useState([]);
  const [phase, setPhase] = useState('active');
  const [timerEndTime, setTimerEndTime] = useState(null);
  const [showDateFlow, setShowDateFlow] = useState(false);
  const [userRating, setUserRating] = useState(0);
  const [userReview, setUserReview] = useState('');
  const [aiRating, setAiRating] = useState(null);
  const [ratingStep, setRatingStep] = useState(null);
  const [ratingLoading, setRatingLoading] = useState(false);
  const [voiceQuickOpen, setVoiceQuickOpen] = useState(false);
  const [readReceipts, setReadReceipts] = useState({});
  const [isTyping, setIsTyping] = useState(false);
  const lastSentRef = useRef('');

  const getVoiceOverrides = useCallback(() => {
    const voice = character?.ttsVoice || character?.tts_voice;
    if (voice && voice !== 'default') return { ttsVoice: voice };
    return null;
  }, [character]);

  const handleSendMessage = useCallback(async (text) => {
    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: text };
    const botId = `b-${Date.now()}`;

    setMessages(prev => [...prev, userMsg]);

    if (!timerEndTime) {
      setTimerEndTime(Date.now() + 30 * 60 * 1000);
    }

    // Brief read delay then mark as seen
    await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 700));
    setReadReceipts(prev => ({ ...prev, [userMsg.id]: Date.now() }));

    // "Left on read": 30% chance of 15-90s delay
    const leftOnRead = Math.random() < 0.3;
    if (leftOnRead) {
      const delayMs = 15000 + Math.random() * 75000;
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }

    // Typing indicator
    setIsTyping(true);
    const typingDuration = 1000 + Math.random() * 3000;
    await new Promise(resolve => setTimeout(resolve, typingDuration));
    setIsTyping(false);

    // Now send the actual API call
    setMessages(prev => [...prev, { id: botId, role: 'bot', content: '', isStreaming: true }]);
    lastSentRef.current = '';
    if (settings?.ttsEnabled && settings?.ttsAutoPlay) {
      startStreamingTTS(botId, getVoiceOverrides());
    }

    try {
      await sendBreakoutMessage(character, text, messages, {
        onChunk: (partial) => {
          setMessages(prev => prev.map(m => m.id === botId ? { ...m, content: partial } : m));
          if (settings?.ttsEnabled && settings?.ttsAutoPlay) {
            const delta = partial.slice(lastSentRef.current.length);
            if (delta) addStreamingText(delta);
            lastSentRef.current = partial;
          }
        },
      });
    } catch (e) {
      console.warn('[BreakoutRoom] send failed:', e);
    } finally {
      if (settings?.ttsEnabled && settings?.ttsAutoPlay) {
        endStreamingTTS();
      }
      setMessages(prev => prev.map(m => m.id === botId ? { ...m, isStreaming: false } : m));
    }
  }, [character, messages, sendBreakoutMessage, timerEndTime, settings, startStreamingTTS, addStreamingText, endStreamingTTS, getVoiceOverrides]);

  const handleExpire = useCallback(() => {
    if (phase !== 'active') return;
    setPhase('expired');
    endBreakoutRoom(character?.id);
    setRatingStep('user');
  }, [phase, endBreakoutRoom, character?.id]);

  const handleSubmitRating = useCallback(async () => {
    if (userRating === 0) return;
    setRatingLoading(true);
    rateCharacter(character?.id, userRating, userReview);
    try {
      const result = await generateUserRating(character, messages);
      if (result) {
        setAiRating(result);
      }
    } catch { }
    setRatingLoading(false);
    setRatingStep('ai');
  }, [userRating, userReview, rateCharacter, character, messages, generateUserRating]);

  const handleBookDateConfirm = useCallback(async (dateType) => {
    setShowDateFlow(false);
    await bookDate(character, dateType);
    setPhase('closed');
    onClose?.();
  }, [character, bookDate, onClose]);

  const handleClose = useCallback(() => {
    setPhase('closed');
    onClose?.();
  }, [onClose]);

  const isChatterboxFamily = ['chatterbox', 'chatterbox_turbo', 'chatterbox_nano'].includes(settings?.ttsEngine || '');
  const isVoxCPM = (settings?.ttsEngine || '') === 'voxcpm';

  const conversationSummary = messages
    .map(m => `${m.role === 'user' ? 'You' : character?.name}: ${m.content}`)
    .join('\n');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70" onClick={handleClose} />

      <div className="relative w-full max-w-lg h-[85vh] max-h-[700px] bg-card border rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in zoom-in-95 duration-200 mx-4">
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border/30 bg-muted/30">
          <button onClick={handleClose} className="w-7 h-7 rounded-full hover:bg-muted flex items-center justify-center transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div className="w-7 h-7 rounded-full overflow-hidden bg-muted flex items-center justify-center text-[9px] font-bold shrink-0">
            {character?.avatar ? (
              <img src={character.avatar} alt="" className="w-full h-full object-cover" />
            ) : (
              character?.name?.[0] || '?'
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-semibold truncate">{character?.name || 'Character'}</div>
            <div className="text-[9px] text-muted-foreground">Breakout Room</div>
          </div>
          {phase === 'active' && timerEndTime && (
            <BreakoutTimer endTime={timerEndTime} onExpire={handleExpire} />
          )}
          <div className="flex items-center gap-1.5 shrink-0 ml-1">
            <span className="text-[9px] text-muted-foreground select-none">TTS</span>
            <Switch
              checked={!!settings?.ttsEnabled}
              onCheckedChange={(v) => {
                updateSettings({ ttsEnabled: v });
                if (!v) stopTTS();
              }}
              className="scale-75"
            />
          </div>
          {settings?.ttsEnabled && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[9px] text-muted-foreground select-none">Auto</span>
              <Switch
                checked={!!settings?.ttsAutoPlay}
                onCheckedChange={(v) => updateSettings({ ttsAutoPlay: v })}
                className="scale-75"
              />
            </div>
          )}
          {settings?.ttsEnabled && isPlayingAudio && (
            <button
              onClick={() => stopTTS()}
              className="w-7 h-7 rounded-full flex items-center justify-center transition-colors shrink-0 bg-red-500/10 text-red-500 hover:bg-red-500/20"
              title="Stop audio"
            >
              <X className="w-3 h-3" />
            </button>
          )}
          {settings?.ttsEnabled && (isChatterboxFamily || isVoxCPM) && (
            <button
              onClick={() => setVoiceQuickOpen(true)}
              className="w-7 h-7 rounded-full flex items-center justify-center transition-colors shrink-0 hover:bg-muted text-muted-foreground"
              title="Set character voice"
            >
              <AudioLines className="w-3 h-3" />
            </button>
          )}
          {phase === 'active' && timerEndTime && (
            <BreakoutTimer endTime={timerEndTime} onExpire={handleExpire} />
          )}
        </div>

        {phase === 'active' && (
          <BreakoutChat
            character={character}
            messages={messages}
            readReceipts={readReceipts}
            isTyping={isTyping}
            onSendMessage={handleSendMessage}
          />
        )}

        {phase === 'expired' && !ratingStep && (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-muted-foreground" />
            </div>
            <h3 className="text-sm font-bold">Breakout Room Closed</h3>
            <p className="text-xs text-muted-foreground">Your conversation with {character?.name} has ended. Want to continue?</p>
            <div className="flex gap-2 pt-2">
              <Button size="sm" variant="outline" onClick={handleClose}>Close</Button>
              <Button size="sm" onClick={() => setShowDateFlow(true)} className="gap-1">
                Book a Date <ArrowLeft className="w-3.5 h-3.5 rotate-180" />
              </Button>
            </div>
          </div>
        )}

        {phase === 'expired' && ratingStep === 'user' && (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-4">
            <h3 className="text-sm font-bold">How was your conversation?</h3>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  onClick={() => setUserRating(n)}
                  className={`w-9 h-9 rounded-full flex items-center justify-center transition-all ${
                    n <= userRating ? 'text-amber-400 scale-110' : 'text-muted hover:text-amber-400/50'
                  }`}
                >
                  <Star className="w-6 h-6 fill-current" />
                </button>
              ))}
            </div>
            <textarea
              value={userReview}
              onChange={e => setUserReview(e.target.value)}
              placeholder="Optional comment..."
              maxLength={200}
              rows={2}
              className="w-full max-w-xs text-xs bg-muted border rounded-lg px-3 py-2 outline-none focus:border-primary/50 transition-colors resize-none"
            />
            <Button size="sm" onClick={handleSubmitRating} disabled={userRating === 0 || ratingLoading}>
              {ratingLoading ? 'Loading...' : 'Submit Rating'}
            </Button>
          </div>
        )}

        {phase === 'expired' && ratingStep === 'ai' && aiRating && (
          <div className="flex-1 flex flex-col items-center justify-center p-6 text-center space-y-4">
            <h3 className="text-sm font-bold">{character?.name} rated you</h3>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map(n => (
                <Star key={n} className={`w-6 h-6 ${n <= (aiRating.rating || 3) ? 'text-amber-400 fill-current' : 'text-muted'}`} />
              ))}
            </div>
            {aiRating.review && (
              <p className="text-xs text-muted-foreground italic max-w-xs">"{aiRating.review}"</p>
            )}
            <p className="text-[10px] text-muted-foreground/60">— {character?.name}</p>
            <div className="flex gap-2 pt-2">
              <Button size="sm" variant="outline" onClick={handleClose}>Close</Button>
              <Button size="sm" onClick={() => setShowDateFlow(true)} className="gap-1">
                Book a Date
              </Button>
            </div>
          </div>
        )}

        {showDateFlow && (
          <BookDateFlow
            character={character}
            onConfirm={handleBookDateConfirm}
            onClose={() => setShowDateFlow(false)}
          />
        )}
      </div>

      <VoiceQuickPicker open={voiceQuickOpen} onOpenChange={setVoiceQuickOpen} variant="dialog" />
    </div>
  );
}
