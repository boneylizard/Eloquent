import React, { useEffect, useRef } from 'react';

const LEAK_TYPES = ['analysis', 'cipher', 'somatic', 'micro_step'];

export default function ThoughtLeakTerminal({ events, maxLines = 50 }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const filteredEvents = events.filter(evt => LEAK_TYPES.includes(evt.type));

  if (filteredEvents.length === 0) return null;

  const formatLine = (evt) => {
    const timestamp = new Date(evt.ts).toISOString().split('T')[1].slice(0, -1);
    const eventType = evt.type.toUpperCase();
    
    if (evt.type === 'analysis') {
      const promptTokens = evt.data?.prompt_tokens || 0;
      return {
        timestamp,
        eventType: 'CONTEXTUAL_ANALYSIS',
        detail: 'Sending prompt to local inference engine',
        tokens: promptTokens,
      };
    }
    
    if (evt.type === 'somatic') {
      const promptTokens = evt.data?.prompt_tokens || 0;
      return {
        timestamp,
        eventType: 'SOMATIC_GENERATION',
        detail: 'Sending somatic calibration prompt',
        tokens: promptTokens,
      };
    }
    
    if (evt.type === 'cipher') {
      return {
        timestamp,
        eventType: 'CIPHER_STREAM',
        detail: 'Emitting encrypted payload blocks',
        tokens: 0,
      };
    }
    
    if (evt.type === 'micro_step') {
      const stepName = evt.data?.step || 'UNKNOWN';
      const detail = evt.data?.detail || '';
      const stepTokens = evt.data?.step_tokens || 0;
      return {
        timestamp,
        eventType: stepName,
        detail,
        tokens: stepTokens,
      };
    }
    
    return {
      timestamp,
      eventType,
      detail: '',
      tokens: 0,
    };
  };

  return (
    <div className="thought-leak-terminal" ref={scrollRef}>
      {filteredEvents.slice(-maxLines).map((evt, i) => {
        const formatted = formatLine(evt);
        return (
          <div key={i} className={`thought-leak-line ${evt.type}`}>
            <span className="thought-leak-timestamp">[{formatted.timestamp}]</span>
            <span className="thought-leak-type">[{formatted.eventType}]</span>
            <span className="thought-leak-data">{formatted.detail}</span>
            {formatted.tokens > 0 && (
              <span className="thought-leak-tokens">Tokens: {formatted.tokens.toLocaleString()}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
