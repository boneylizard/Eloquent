import React, { useMemo } from 'react';

function decodeCipherBlock(block) {
  if (!block) return null;
  const CIPHER_RE = /⟦CIPHER:v(\d+):([0-9a-fA-F]+):([A-Za-z0-9+/=]*)⟧/;
  const match = CIPHER_RE.exec(block);
  if (!match) return null;

  const [, , , b64Layer] = match;

  try {
    const decoded = JSON.parse(atob(b64Layer));
    return decoded;
  } catch {
    return null;
  }
}

function scrambleToNoise(text) {
  const glyphs = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F',
    'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ',
    'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ', 'ﾄ',
    'ﾅ', 'ﾆ', 'ﾇ', 'ﾈ', 'ﾉ', 'ﾊ', 'ﾋ', 'ﾌ', 'ﾍ', 'ﾎ',
    'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ', 'ﾔ', 'ﾕ', 'ﾖ', 'ﾗ', 'ﾘ',
    'ﾙ', 'ﾚ', 'ﾛ', 'ﾜ', 'ｦ', 'ﾝ',
    ':', ';', '.', '!', '?', '-', '=', '+', '*', '#',
  ];

  return text.split('').map((char, i) => {
    const seed = (char.charCodeAt(0) * 31 + i * 7) % glyphs.length;
    return glyphs[seed];
  }).join('');
}

export default function CipherGlyphRow({ glyphs, block, phase }) {
  const cipherText = useMemo(() => {
    if (block) {
      const decoded = decodeCipherBlock(block);
      const source = decoded?.plan_text || block;
      return scrambleToNoise(source);
    }
    return null;
  }, [block]);

  if (!cipherText) return null;

  return (
    <div className="sanctuary-cipher-row">
      <div className="sanctuary-cipher-noise">
        {cipherText}
      </div>
      {phase && <span className="sanctuary-cipher-phase">{phase}</span>}
    </div>
  );
}
