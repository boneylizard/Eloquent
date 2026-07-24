/**
 * Shrink messages before IndexedDB — keeps URLs, drops embedded blobs and bloat.
 */

const PERSIST_STR_MAX = 2048;

function trimString(value, max = PERSIST_STR_MAX) {
  if (value == null) return value;
  const s = String(value);
  if (s.length <= max) return s;
  if (s.startsWith('data:')) return null;
  return `${s.slice(0, max)}…`;
}

/**
 * @param {object} msg
 * @returns {object}
 */
export function sanitizeMessageForStorage(msg) {
  if (!msg || typeof msg !== 'object') return msg;

  const out = { ...msg };

  delete out.image_base64;
  delete out.base64;
  delete out.previewUrl;

  if (out.avatar && String(out.avatar).startsWith('data:')) {
    out.avatar = null;
  } else {
    out.avatar = trimString(out.avatar, 512);
  }

  out.content = typeof out.content === 'string' ? trimString(out.content, 120_000) : out.content;
  out.reasoningText =
    typeof out.reasoningText === 'string' ? trimString(out.reasoningText, 120_000) : out.reasoningText;
  if (out.reasoningCapabilitySource != null) {
    out.reasoningCapabilitySource = trimString(out.reasoningCapabilitySource, 64);
  }
  out.imagePath = trimString(out.imagePath, 1024);

  if (Array.isArray(out.enhancement_history)) {
    const history = out.enhancement_history
      .map((entry) => trimString(entry, 1024))
      .filter(Boolean);
    if (out.imagePath && !history.includes(out.imagePath)) {
      history.push(out.imagePath);
    }
    out.enhancement_history = history;
    if (history.length > 0) {
      const currentPathIndex = history.indexOf(out.imagePath);
      const requestedLevel = Number(out.current_enhancement_level);
      out.current_enhancement_level = currentPathIndex >= 0
        ? currentPathIndex
        : Math.max(
          0,
          Math.min(Number.isFinite(requestedLevel) ? Math.trunc(requestedLevel) : 0, history.length - 1),
        );
    } else {
      out.current_enhancement_level = 0;
    }
  }

  if (Array.isArray(out.images)) {
    out.images = out.images
      .map((img) => {
        if (!img || typeof img !== 'object') return img;
        const copy = { ...img };
        delete copy.base64;
        delete copy.image_base64;
        if (copy.url && String(copy.url).startsWith('data:')) copy.url = null;
        return copy;
      })
      .filter(Boolean);
  }

  return out;
}

/**
 * @param {object[]} messages
 * @returns {object[]}
 */
export function sanitizeMessagesForStorage(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map(sanitizeMessageForStorage);
}
