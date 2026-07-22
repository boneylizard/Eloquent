import React, { useState, useEffect, useCallback, useRef } from 'react';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';

const SECTION_GRADIENTS = {
  Intimate: 'from-pink-600/80 via-pink-800/60 to-black/90',
  Erotic: 'from-red-600/80 via-red-800/60 to-black/90',
  Experimental: 'from-purple-600/80 via-purple-800/60 to-black/90',
};

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function StoryViewer({ stories, initialIndex = 0, onClose, onMarkViewed }) {
  const [index, setIndex] = useState(initialIndex);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const timerRef = useRef(null);
  const story = stories[index];

  const advance = useCallback(() => {
    if (index < stories.length - 1) {
      setIndex(i => i + 1);
      setProgress(0);
      progressRef.current = 0;
    } else {
      onClose?.();
    }
  }, [index, stories.length, onClose]);

  const goBack = useCallback(() => {
    if (index > 0) {
      setIndex(i => i - 1);
      setProgress(0);
      progressRef.current = 0;
    }
  }, [index]);

  useEffect(() => {
    if (!story) return;
    onMarkViewed?.(story.id);
  }, [story?.id, onMarkViewed]);

  useEffect(() => {
    if (!story) return;
    const duration = 5000;
    const interval = 50;
    const step = interval / duration;
    progressRef.current = 0;
    timerRef.current = setInterval(() => {
      progressRef.current += step;
      setProgress(progressRef.current);
      if (progressRef.current >= 1) {
        clearInterval(timerRef.current);
        advance();
      }
    }, interval);
    return () => { clearInterval(timerRef.current); };
  }, [index, story, advance]);

  const handleClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < rect.width * 0.3) {
      goBack();
    } else if (x > rect.width * 0.7) {
      advance();
    }
  };

  if (!story) return null;

  const section = story.section || 'Intimate';
  const gradient = SECTION_GRADIENTS[section] || SECTION_GRADIENTS.Intimate;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <div className="absolute inset-0 bg-black" />
      <div
        className="relative w-full h-full sm:w-[420px] sm:h-[700px] sm:rounded-2xl overflow-hidden cursor-pointer select-none"
        onClick={handleClick}
      >
        <div className={`absolute inset-0 bg-gradient-to-b ${gradient}`} />

        <div className="absolute top-0 inset-x-0 flex gap-1 p-2 z-10">
          {stories.map((s, i) => (
            <div key={s.id} className="flex-1 h-0.5 rounded-full bg-white/20 overflow-hidden">
              <div
                className="h-full rounded-full bg-white transition-all duration-100 ease-linear"
                style={{
                  width: i < index ? '100%' : i === index ? `${progress * 100}%` : '0%',
                }}
              />
            </div>
          ))}
        </div>

        <button
          onClick={(e) => { e.stopPropagation(); onClose?.(); }}
          className="absolute top-3 right-3 z-10 w-8 h-8 rounded-full bg-black/40 flex items-center justify-center hover:bg-black/60 transition-colors"
        >
          <X className="w-4 h-4 text-white" />
        </button>

        <div className="absolute top-8 left-4 right-4 flex items-center gap-3 z-10">
          <div className="w-10 h-10 rounded-full overflow-hidden border-2 border-white/30 shrink-0">
            {story.character_avatar ? (
              <img src={story.character_avatar} alt="" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-white/20 flex items-center justify-center text-sm font-bold text-white">
                {story.character_name?.[0] || '?'}
              </div>
            )}
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-white drop-shadow truncate">
              {story.character_name}
            </div>
            <div className="text-[10px] text-white/60">{timeAgo(story.created_at)}</div>
          </div>
        </div>

        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 px-8 text-center z-10">
          <p className="text-lg sm:text-xl font-light leading-relaxed text-white drop-shadow-lg">
            {story.content}
          </p>
        </div>

        <div className="absolute bottom-6 inset-x-0 flex items-center justify-center gap-2 z-10">
          <button
            onClick={(e) => { e.stopPropagation(); goBack(); }}
            disabled={index === 0}
            className="w-8 h-8 rounded-full bg-black/30 flex items-center justify-center hover:bg-black/50 transition-colors disabled:opacity-20"
          >
            <ChevronLeft className="w-4 h-4 text-white" />
          </button>
          <span className="text-[10px] text-white/40">{index + 1} / {stories.length}</span>
          <button
            onClick={(e) => { e.stopPropagation(); advance(); }}
            disabled={index >= stories.length - 1}
            className="w-8 h-8 rounded-full bg-black/30 flex items-center justify-center hover:bg-black/50 transition-colors disabled:opacity-20"
          >
            <ChevronRight className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}