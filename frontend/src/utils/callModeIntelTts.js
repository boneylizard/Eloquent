export const INTEL_VOICE_STORAGE_KEY = 'LiangLocal-call-intel-tts-voice';
export const INTEL_TAP_MODE_STORAGE_KEY = 'LiangLocal-call-intel-tap-mode';

/** How heading taps behave: listen (TTS), read (expand text), or both. */
export const INTEL_TAP_MODES = ['listen', 'read', 'both'];

export function loadIntelTapMode(fallback = 'both') {
  try {
    const stored = localStorage.getItem(INTEL_TAP_MODE_STORAGE_KEY);
    if (stored && INTEL_TAP_MODES.includes(stored)) return stored;
  } catch {
    /* private mode */
  }
  return fallback;
}

export function saveIntelTapMode(mode) {
  try {
    if (mode && INTEL_TAP_MODES.includes(mode)) {
      localStorage.setItem(INTEL_TAP_MODE_STORAGE_KEY, mode);
    }
  } catch {
    /* ignore */
  }
}

export const INTEL_MESSAGE_PREFIX = 'call-mode-intel:';

export function intelMessageId(slotId) {
  return `${INTEL_MESSAGE_PREFIX}${slotId}`;
}

export function isIntelMessageId(messageId) {
  return typeof messageId === 'string' && messageId.startsWith(INTEL_MESSAGE_PREFIX);
}

export function intelSlotFromMessageId(messageId) {
  if (!isIntelMessageId(messageId)) return null;
  return messageId.slice(INTEL_MESSAGE_PREFIX.length);
}

export function loadIntelTtsVoice(fallback = '') {
  try {
    const stored = localStorage.getItem(INTEL_VOICE_STORAGE_KEY);
    if (stored && stored.trim()) return stored.trim();
  } catch {
    /* private mode */
  }
  return fallback || '';
}

export function saveIntelTtsVoice(voice) {
  try {
    if (voice) localStorage.setItem(INTEL_VOICE_STORAGE_KEY, voice);
    else localStorage.removeItem(INTEL_VOICE_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/** TTS overrides for insight readout — uses narrator voice, not character card voice. */

/** Sentence-sized chunks for streaming autoplay (first sentence plays while rest synthesizes). */
export function splitTextIntoTtsChunks(text) {
  const normalized = String(text || '').replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  const chunks = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g);
  return (chunks || []).map((chunk) => chunk.trim()).filter(Boolean);
}

export function buildIntelTtsOverrides(settings, intelVoice) {
  const engine = settings?.ttsEngine || 'kokoro';
  const voice = intelVoice || settings?.ttsVoice || 'af_heart';
  const overrides = {
    ttsVoice: voice,
    ttsEngine: engine,
    ttsSpeed: Number(settings?.ttsSpeed ?? 1.0) || 1.0,
  };
  if (engine === 'chatterbox' || engine === 'chatterbox_turbo') {
    overrides.ttsExaggeration = settings?.ttsExaggeration ?? 0.5;
    overrides.ttsCfg = settings?.ttsCfg ?? 0.5;
  }
  return overrides;
}
