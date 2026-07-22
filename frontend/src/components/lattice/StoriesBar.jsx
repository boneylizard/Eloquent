import React, { useState } from 'react';
import StoryViewer from './StoryViewer';

const SECTION_RING_COLORS = {
  Intimate: { seen: 'border-pink-500/30', unseen: 'border-pink-500' },
  Erotic: { seen: 'border-red-500/30', unseen: 'border-red-500' },
  Experimental: { seen: 'border-purple-500/30', unseen: 'border-purple-500' },
};

function StoryRing({ story, viewed, onClick }) {
  const section = story.section || 'Intimate';
  const colors = SECTION_RING_COLORS[section] || SECTION_RING_COLORS.Intimate;
  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center gap-1 shrink-0 group"
    >
      <div className={`w-14 h-14 rounded-full p-0.5 border-2 transition-colors ${viewed ? colors.seen : colors.unseen}`}>
        <div className="w-full h-full rounded-full overflow-hidden bg-muted">
          {story.character_avatar ? (
            <img src={story.character_avatar} alt="" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-sm font-bold text-muted-foreground">
              {story.character_name?.[0] || '?'}
            </div>
          )}
        </div>
      </div>
      <span className="text-[9px] text-muted-foreground truncate max-w-[64px] text-center leading-tight">
        {story.character_name}
      </span>
    </button>
  );
}

export default function StoriesBar({ stories, viewedStoryIds, onMarkViewed, className = '' }) {
  const [viewerIndex, setViewerIndex] = useState(null);

  if (!stories?.length) return null;

  const uniqueStories = [];
  const seenNames = new Set();
  for (const s of stories) {
    if (!seenNames.has(s.character_name)) {
      seenNames.add(s.character_name);
      uniqueStories.push(s);
    }
  }

  return (
    <>
      <div className={`flex items-center gap-3 overflow-x-auto py-2 px-1 scrollbar-none ${className}`}>
        {uniqueStories.map((story) => {
          const viewed = viewedStoryIds?.has(story.id);
          const idx = stories.findIndex(s => s.id === story.id);
          return (
            <StoryRing
              key={story.id}
              story={story}
              viewed={viewed}
              onClick={() => setViewerIndex(idx)}
            />
          );
        })}
      </div>
      {viewerIndex !== null && (
        <StoryViewer
          stories={stories}
          initialIndex={viewerIndex}
          onClose={() => setViewerIndex(null)}
          onMarkViewed={onMarkViewed}
        />
      )}
    </>
  );
}