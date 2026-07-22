import React, { useState, useRef, useCallback, useEffect } from 'react';
import { X, Mic, MicOff, Volume2, VolumeX, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NeuralSexPanel({ character, onClose }) {
  const [isActive, setIsActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(80);
  const [logs, setLogs] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const micRef = useRef(null);
  const [sessionTime, setSessionTime] = useState(0);

  useEffect(() => {
    if (!isActive) return;
    const id = setInterval(() => setSessionTime(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [isActive]);

  const formatTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const startSession = useCallback(async () => {
    setIsActive(true);
    const openMsg = `Neural sex session initiated with ${character?.name || 'character'}.`;
    setLogs(prev => [...prev, { role: 'system', text: openMsg }]);
  }, [character]);

  const endSession = useCallback(() => {
    setIsActive(false);
    setLogs(prev => [...prev, { role: 'system', text: 'Session ended.' }]);
    if (micRef.current) {
      try { micRef.current.stop?.(); } catch {}
      micRef.current = null;
    }
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-gradient-to-b from-black/90 via-black/80 to-black/90" onClick={endSession} />
      <div className="relative w-full max-w-md mx-4 bg-gradient-to-b from-gray-900 to-black border border-white/10 rounded-3xl overflow-hidden shadow-2xl shadow-purple-500/10">
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full overflow-hidden bg-gray-800 flex items-center justify-center text-xs font-bold">
              {character?.avatar ? (
                <img src={character.avatar} alt="" className="w-full h-full object-cover" />
              ) : (
                character?.name?.[0] || '?'
              )}
            </div>
            <div>
              <div className="text-xs font-semibold text-white">{character?.name || 'Character'}</div>
              <div className="text-[9px] text-gray-400">Neural Sex</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isActive && <span className="text-[10px] font-mono text-gray-400">{formatTime(sessionTime)}</span>}
            <button onClick={endSession} className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center hover:bg-white/20">
              <X className="w-3.5 h-3.5 text-white" />
            </button>
          </div>
        </div>

        <div className="px-5 py-6 flex flex-col items-center">
          <div className={`w-24 h-24 rounded-full overflow-hidden border-2 transition-all duration-1000 ${
            isActive ? 'border-purple-400 shadow-lg shadow-purple-500/30 animate-pulse' : 'border-white/20'
          }`}>
            {character?.avatar ? (
              <img src={character.avatar} alt="" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-gray-800 flex items-center justify-center text-2xl font-bold text-gray-600">
                {character?.name?.[0] || '?'}
              </div>
            )}
          </div>

          <div className="h-8 flex items-center justify-center mt-3">
            {isActive ? (
              <div className="flex items-center gap-0.5">
                {[1, 2, 3, 4, 5].map(i => (
                  <div
                    key={i}
                    className="w-0.5 bg-purple-400 rounded-full animate-pulse"
                    style={{
                      height: `${4 + Math.random() * 16}px`,
                      animationDelay: `${i * 0.15}s`,
                      opacity: 0.6 + Math.random() * 0.4,
                    }}
                  />
                ))}
              </div>
            ) : (
              <span className="text-xs text-gray-400">Voice mode ready</span>
            )}
          </div>

          {!isActive ? (
            <Button onClick={startSession} size="lg" className="mt-4 w-16 h-16 rounded-full bg-purple-600 hover:bg-purple-500 shadow-lg shadow-purple-500/30">
              <Mic className="w-6 h-6" />
            </Button>
          ) : (
            <div className="flex items-center gap-4 mt-4">
              <button
                onClick={() => setIsMuted(!isMuted)}
                className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                  isMuted ? 'bg-red-500/20 text-red-400' : 'bg-white/10 text-white hover:bg-white/20'
                }`}
              >
                {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>
              <button
                onClick={endSession}
                className="w-12 h-12 rounded-full bg-red-500/20 text-red-400 hover:bg-red-500/30 flex items-center justify-center"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          )}

          {isActive && (
            <div className="flex items-center gap-2 mt-3 w-full max-w-[200px]">
              <Volume2 className="w-3 h-3 text-gray-400 shrink-0" />
              <input
                type="range"
                min={0}
                max={100}
                value={volume}
                onChange={e => setVolume(Number(e.target.value))}
                className="w-full h-1 bg-gray-700 rounded-full appearance-none cursor-pointer accent-purple-500"
              />
            </div>
          )}
        </div>

        {logs.length > 0 && (
          <div className="px-5 pb-4 max-h-32 overflow-y-auto space-y-1">
            {logs.map((log, i) => (
              <p key={i} className={`text-[10px] leading-relaxed ${
                log.role === 'system' ? 'text-gray-500 italic' : log.role === 'user' ? 'text-white' : 'text-purple-300'
              }`}>
                {log.role !== 'system' && <strong>{log.role === 'user' ? 'You' : character?.name}:</strong>} {log.text}
              </p>
            ))}
          </div>
        )}

        {isActive && (
          <div className="flex items-center justify-center px-5 pb-4">
            <span className="text-[9px] text-gray-500">
              {isMuted ? 'Mic muted' : 'Mic active — speak naturally'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
