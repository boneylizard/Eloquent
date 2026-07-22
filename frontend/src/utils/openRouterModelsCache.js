export const OPENROUTER_MODELS_CACHE_KEY = 'openrouter-models-cache-v1';
export const OPENROUTER_MODELS_CACHE_TTL_MS = 60 * 60 * 1000;
export const OPENROUTER_MODELS_API = 'https://openrouter.ai/api/v1/models?output_modalities=text';
export const OPENROUTER_FREE_MODEL_ID = 'openrouter/free';

const FREE_ROUTER = {
  id: OPENROUTER_FREE_MODEL_ID,
  name: 'Free Models Router',
  description: 'OpenRouter chooses an available free model for each request.',
  provider: 'openrouter',
  contextLength: null,
  capabilities: {},
  pricing: { prompt: '0', completion: '0' },
  free: true,
  isFreeRouter: true,
};

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function isZeroPrice(value) {
  return value !== null && value !== undefined && value !== '' && Number(value) === 0;
}

export function normalizeOpenRouterModel(model) {
  if (!model || typeof model !== 'object' || !model.id) return null;
  const id = String(model.id);
  const supported = Array.isArray(model.supported_parameters) ? model.supported_parameters : [];
  const architecture = model.architecture || model.raw?.architecture || {};
  const inputModalities = Array.isArray(architecture.input_modalities) ? architecture.input_modalities : [];
  const outputModalities = Array.isArray(architecture.output_modalities)
    ? architecture.output_modalities.map((modality) => String(modality).toLowerCase())
    : [];
  const pricing = model.pricing || {};
  const free = id === OPENROUTER_FREE_MODEL_ID
    || id.endsWith(':free')
    || (isZeroPrice(pricing.prompt) && isZeroPrice(pricing.completion));

  return {
    id,
    name: String(model.name || id),
    description: String(model.description || ''),
    provider: id.split('/')[0] || 'openrouter',
    contextLength: Number.isFinite(Number(model.context_length)) ? Number(model.context_length) : null,
    outputModalities,
    capabilities: {
      vision: inputModalities.includes('image'),
      tools: supported.includes('tools'),
      reasoning: supported.some((parameter) => String(parameter).includes('reasoning')),
    },
    pricing,
    free,
    isFreeRouter: id === OPENROUTER_FREE_MODEL_ID,
    raw: model,
  };
}

export function normalizeOpenRouterModels(payload) {
  const rows = Array.isArray(payload) ? payload : (Array.isArray(payload?.data) ? payload.data : []);
  const models = rows
    .map(normalizeOpenRouterModel)
    .filter(Boolean)
    .filter((model) => (
      model.isFreeRouter
      || model.outputModalities.length === 0
      || (model.outputModalities.length === 1 && model.outputModalities[0] === 'text')
    ));
  if (!models.some((model) => model.id === OPENROUTER_FREE_MODEL_ID)) {
    models.unshift({ ...FREE_ROUTER });
  }
  return models.sort((left, right) => {
    if (left.isFreeRouter !== right.isFreeRouter) return left.isFreeRouter ? -1 : 1;
    if (left.free !== right.free) return left.free ? -1 : 1;
    return left.name.localeCompare(right.name);
  });
}

export function readOpenRouterModelsCache() {
  if (typeof localStorage === 'undefined') return { models: [], savedAt: 0, stale: true };
  const cached = safeJsonParse(localStorage.getItem(OPENROUTER_MODELS_CACHE_KEY) || 'null');
  const savedAt = Number(cached?.savedAt);
  const models = normalizeOpenRouterModels(cached?.models || []);
  return {
    models: cached?.models ? models : [],
    savedAt: Number.isFinite(savedAt) ? savedAt : 0,
    stale: !Number.isFinite(savedAt) || Date.now() - savedAt >= OPENROUTER_MODELS_CACHE_TTL_MS,
  };
}

export function writeOpenRouterModelsCache(payload) {
  const models = normalizeOpenRouterModels(payload);
  localStorage.setItem(
    OPENROUTER_MODELS_CACHE_KEY,
    JSON.stringify({ savedAt: Date.now(), models }),
  );
  window.dispatchEvent(new Event('openrouter-models-cache-updated'));
  return models;
}

export async function refreshOpenRouterModelsCache({ forceRefresh = false, apiKey = '' } = {}) {
  const existing = readOpenRouterModelsCache();
  if (!forceRefresh && existing.models.length > 0 && !existing.stale) {
    return { ...existing, status: 'cached' };
  }

  try {
    const headers = apiKey.trim() ? { Authorization: `Bearer ${apiKey.trim()}` } : {};
    const response = await fetch(OPENROUTER_MODELS_API, { method: 'GET', headers });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const models = writeOpenRouterModelsCache(await response.json());
    return { models, status: 'ok', savedAt: Date.now() };
  } catch (error) {
    if (existing.models.length > 0) {
      return { ...existing, status: 'fallback', error };
    }
    return { models: [], status: 'error', savedAt: 0, error };
  }
}

export function subscribeOpenRouterModelsCache(listener) {
  const onStorage = (event) => {
    if (event.key === OPENROUTER_MODELS_CACHE_KEY) listener(readOpenRouterModelsCache());
  };
  const onCustom = () => listener(readOpenRouterModelsCache());
  window.addEventListener('storage', onStorage);
  window.addEventListener('openrouter-models-cache-updated', onCustom);
  return () => {
    window.removeEventListener('storage', onStorage);
    window.removeEventListener('openrouter-models-cache-updated', onCustom);
  };
}
