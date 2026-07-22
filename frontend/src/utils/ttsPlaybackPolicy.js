const MIN_TTS_SPEED = 0.25;
const MAX_TTS_SPEED = 4;

export function normaliseTtsSpeed(speed) {
  const parsed = Number(speed);
  if (!Number.isFinite(parsed) || parsed <= 0) return 1;
  return Math.max(MIN_TTS_SPEED, Math.min(MAX_TTS_SPEED, parsed));
}

export function engineSynthesisesTtsSpeed(engine) {
  const engineId = String(engine || '').toLowerCase();
  return engineId === 'kokoro' || engineId.startsWith('nanogpt-');
}

export function getTtsPlaybackRate(engine, requestedSpeed) {
  return engineSynthesisesTtsSpeed(engine) ? 1 : normaliseTtsSpeed(requestedSpeed);
}
