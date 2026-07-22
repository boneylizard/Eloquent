/**
 * Cipher Block decoder (frontend mirror of backend/app/sanctuary/cipher.py).
 *
 * Format: ⟦CIPHER:v1:<hex_layer>:<b64_layer>⟧
 *
 * Used to decode cipher blocks for display in the Cognitive Glass.
 * The frontend never needs to ENCODE — it just passes the raw block back
 * to the backend on the next turn.
 */

const CIPHER_RE = /⟦CIPHER:v(\d+):([0-9a-fA-F]+):([A-Za-z0-9+/=]*)⟧/g;
const CIPHER_RE_SINGLE = /⟦CIPHER:v(\d+):([0-9a-fA-F]+):([A-Za-z0-9+/=]*)⟧/;

/**
 * Decode a cipher block string into a shadow state object.
 * Returns null if the block is malformed.
 */
export function decodeCipher(block) {
  if (!block) return null;
  const match = CIPHER_RE_SINGLE.exec(block);
  if (!match) return null;

  const [, versionStr, hexLayer, b64Layer] = match;

  const version = parseInt(versionStr, 10);
  if (isNaN(version)) return null;

  let turnCount = 0;
  try {
    turnCount = parseInt(hexLayer.slice(4, 12), 16);
    if (isNaN(turnCount)) turnCount = 0;
  } catch { turnCount = 0; }

  let heatIndex = 0.0;
  try {
    const hexPart = hexLayer.slice(12);
    if (hexPart.length >= 4) {
      heatIndex = parseInt(hexPart.slice(0, 4), 16) / 0xFFFF;
    }
  } catch { heatIndex = 0.0; }

  let remainder = {};
  try {
    const decoded = atob(b64Layer);
    remainder = JSON.parse(decoded);
  } catch { remainder = {}; }

  return {
    version,
    turn_count: turnCount,
    heat_index: heatIndex,
    ...remainder,
  };
}

/**
 * Find the first cipher block in a text string.
 * Returns the raw block string, or null.
 */
export function findCipherInText(text) {
  if (!text) return null;
  const match = CIPHER_RE_SINGLE.exec(text);
  return match ? match[0] : null;
}

/**
 * Remove all cipher blocks from a text string (used before rendering).
 */
export function stripCipherFromText(text) {
  if (!text) return text;
  return text.replace(CIPHER_RE, '').trim();
}

/**
 * Convert a cipher block into a visual glyph representation for the Cognitive Glass.
 * This is NOT the literal block — it's an abstracted visual that shows "activity"
 * without revealing the content.
 *
 * Returns an array of glyph descriptors.
 */
export function cipherToGlyphs(block) {
  if (!block) return [];
  const decoded = decodeCipher(block);
  if (!decoded) return [];

  const heat = decoded.heat_index ?? 0;
  const turn = decoded.turn_count ?? 0;
  const dominance = decoded.dominance_vector ?? 0.5;

  // Generate 8 glyphs whose appearance encodes state intensity
  const glyphs = [];
  const seed = turn * 31 + Math.floor(heat * 255);
  for (let i = 0; i < 8; i++) {
    const val = ((seed * (i + 1)) % 256) / 256;
    glyphs.push({
      index: i,
      intensity: 0.2 + (val * 0.6 * (0.5 + dominance * 0.5)),
      pulse: heat > 0.5,
      char: _GLYPH_CHARS[Math.floor(val * _GLYPH_CHARS.length) % _GLYPH_CHARS.length],
    });
  }
  return glyphs;
}

const _GLYPH_CHARS = ['◇', '◈', '◉', '◐', '◑', '◒', '◓', '◔', '◕', '◖', '◗', '◯'];
