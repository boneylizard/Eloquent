export const FIRST_RUN_INTENT_KEY = 'mirid-first-run-intent-v1';
export const ROLEPLAY_THEME = 'faraday';

export const FIRST_RUN_PURPOSES = Object.freeze([
  {
    id: 'roleplay',
    label: 'Roleplay & characters',
    description: 'Build, import and speak with persistent characters in their own worlds.',
  },
  {
    id: 'conversation',
    label: 'Everyday conversation',
    description: 'A capable private workspace for questions, thinking and ordinary chat.',
  },
  {
    id: 'writing',
    label: 'Writing & research',
    description: 'Long-form drafting, documents, memory and careful source work.',
  },
  {
    id: 'voice-media',
    label: 'Voice & media',
    description: 'Speech, transcription, images and more expressive conversations.',
  },
  {
    id: 'developer',
    label: 'Models & developer tools',
    description: 'Local inference, APIs, model testing and the machinery underneath.',
  },
  {
    id: 'everything',
    label: 'A bit of everything',
    description: 'Keep the full workspace visible and decide as you go.',
  },
]);

const PURPOSE_IDS = new Set(FIRST_RUN_PURPOSES.map((purpose) => purpose.id));

export function normaliseFirstRunIntent(value) {
  if (!value || typeof value !== 'object' || !PURPOSE_IDS.has(value.purpose)) return null;
  const zoom = Number(value.interfaceZoom);
  return {
    version: 1,
    purpose: value.purpose,
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
