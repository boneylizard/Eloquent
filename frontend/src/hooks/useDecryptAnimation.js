import { useState, useEffect, useRef } from 'react';

const GLYPH_CHARSET = '0123456789ABCDEF█▓░▒◊○●';

export function useDecryptAnimation(targetText, duration = 1500, enabled = true) {
  const [displayText, setDisplayText] = useState('');
  const [isDecrypted, setIsDecrypted] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!enabled || !targetText) {
      setDisplayText(targetText || '');
      setIsDecrypted(true);
      return;
    }

    setDisplayText('');
    setIsDecrypted(false);
    
    const charCount = targetText.length;
    const stepTime = duration / charCount;
    const revealOrder = Array.from({ length: charCount }, (_, i) => i);
    
    for (let i = revealOrder.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [revealOrder[i], revealOrder[j]] = [revealOrder[j], revealOrder[i]];
    }

    let revealed = new Set();
    let currentText = Array(charCount).fill('').map(() => 
      GLYPH_CHARSET[Math.floor(Math.random() * GLYPH_CHARSET.length)]
    );

    intervalRef.current = setInterval(() => {
      if (revealed.size < charCount) {
        currentText = currentText.map((char, idx) => 
          revealed.has(idx) ? targetText[idx] : GLYPH_CHARSET[Math.floor(Math.random() * GLYPH_CHARSET.length)]
        );

        const nextIdx = revealOrder.find(idx => !revealed.has(idx));
        if (nextIdx !== undefined) {
          revealed.add(nextIdx);
          currentText[nextIdx] = targetText[nextIdx];
        }

        setDisplayText(currentText.join(''));
      } else {
        setIsDecrypted(true);
        clearInterval(intervalRef.current);
      }
    }, stepTime);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [targetText, duration, enabled]);

  return { displayText, isDecrypted };
}
