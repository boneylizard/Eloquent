export const FIRST_RUN_INTENT_KEY = 'mirid-first-run-intent-v1';
export const ROLEPLAY_THEME = 'faraday';

export const FIRST_RUN_PURPOSES = Object.freeze([
  {
    id: 'roleplay',
    label: 'Roleplay & storytelling',
    description: 'Characters, interactive stories and long-running fictional worlds.',
  },
  {
    id: 'sillytavern',
    label: 'Use Mirid with SillyTavern',
    description: 'Run text, speech, transcription and images behind SillyTavern.',
  },
  {
    id: 'classic',
    label: 'Mirid Classic',
    description: 'The complete workspace for chat, models, documents, voice and images.',
  },
]);

const PURPOSE_IDS = new Set(FIRST_RUN_PURPOSES.map((purpose) => purpose.id));
const LEGACY_PURPOSES = new Map([
  ['conversation', 'classic'],
  ['writing', 'classic'],
  ['voice-media', 'classic'],
  ['developer', 'classic'],
  ['everything', 'classic'],
]);

export function normaliseFirstRunIntent(value) {
  if (!value || typeof value !== 'object') return null;
  const purpose = LEGACY_PURPOSES.get(value.purpose) || value.purpose;
  if (!PURPOSE_IDS.has(purpose)) return null;
  const zoom = Number(value.interfaceZoom);
  return {
    version: 2,
    purpose,
    interfaceZoom: Number.isFinite(zoom) ? Math.min(2, Math.max(0.75, zoom)) : 1.1,
    chosenAt: typeof value.chosenAt === 'string' ? value.chosenAt : null,
  };
}

export function readFirstRunIntent(storage = globalThis?.localStorage) {
  if (!storage) return null;
  try {
    return normaliseFirstRunIntent(JSON.parse(storage.getItem(FIRST_RUN_INTENT_KEY) || 'null'));
  } catch {
    return null;
  }
}

export function writeFirstRunIntent(intent, storage = globalThis?.localStorage) {
  if (!storage) return null;
  const normalised = normaliseFirstRunIntent({
    ...intent,
    chosenAt: intent?.chosenAt || new Date().toISOString(),
  });
  if (!normalised) throw new Error('Choose a valid Mirid starting purpose.');
  storage.setItem(FIRST_RUN_INTENT_KEY, JSON.stringify(normalised));
  if (normalised.purpose === 'roleplay') {
    storage.setItem('vite-ui-theme', ROLEPLAY_THEME);
  }
  return normalised;
}
