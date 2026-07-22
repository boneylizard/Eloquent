import React from 'react';
import { useTypewriter } from '../../hooks/useTypewriter';

export default function TypewriterText({ text, speed = 20, enabled = true, className = '' }) {
  const { displayedText, isComplete } = useTypewriter(text, speed, enabled);

  return (
    <span className={`typewriter-text ${className}`}>
      {displayedText}
      {!isComplete && <span className="typewriter-cursor">█</span>}
    </span>
  );
}
