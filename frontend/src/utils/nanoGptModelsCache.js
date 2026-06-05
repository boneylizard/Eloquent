export const NANO_GPT_MODELS_CACHE_KEY = 'nanoGpt-models-cache-v1';
export const NANO_GPT_MODELS_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
export const NANO_GPT_MODELS_API = 'https://nano-gpt.com/api/models';

function safeJsonParse(str) {
  try {
    return JSON.parse(str);
  } catch {
    return null;
  }
}

/** Flat catalog entry used across UI. */
export function normalizeNanoGptModel(m) {
  if (!m || typeof m !== 'object') return null;
  const id = m.id || m.model || m.name;
  if (!id) return null;
  const name = String(m.name || m.label || m.display_name || id);
  const provider =
    m.provider
    || m.vendor
    || inferProviderFromName(id)
    || inferProviderFromName(name);
  const category = String(m.category || m.group || 'Models');
  return {
    id: String(id),
    name,
    provider: String(provider || ''),
    category,
    visible: m.visible !== false,
    api: String(m.api || m.type || 'chat'),
    capabilities: m.capabilities && typeof m.capabilities === 'object'
      ? m.capabilities
      : {
        reasoning: Boolean(m.reasoning),
        vision: Boolean(m.vision),
        pdf: Boolean(m.pdf),
      },
    raw: m,
  };
}

function inferProviderFromName(s) {
  const x = String(s || '').toLowerCase();
  if (x.includes('/')) return x.split('/')[0];
  return '';
}

/** Flatten NanoGPT /api/models payload shapes into a list of model objects. */
export function flattenNanoGptRawList(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.models)) return data.models;
  if (data?.models && typeof data.models === 'object') {
    return Object.values(data.models).flatMap((category) =>
      typeof category === 'object' && !Array.isArray(category)
        ? Object.values(category)
        : [],
    );
  }
  return [];
}

/** Normalize cache rows; unwrap stale nested category maps from older caches. */
function expandCachedModelEntries(rawList) {
  if (!Array.isArray(rawList)) return [];
  const out = [];
  for (const item of rawList) {
    const norm = normalizeNanoGptModel(item);
    if (norm) {
      out.push(norm);
      continue;
    }
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      for (const nested of Object.values(item)) {
        const n = normalizeNanoGptModel(nested);
        if (n) out.push(n);
      }
    }
  }
  return out;
}

export function readNanoGptModelsCache() {
  const cached = safeJsonParse(localStorage.getItem(NANO_GPT_MODELS_CACHE_KEY) || 'null');
  if (!cached?.models || !Array.isArray(cached.models)) {
    return { models: [], savedAt: 0, stale: true };
  }
  let models = expandCachedModelEntries(cached.models);
  if (!models.length) {
    models = expandCachedModelEntries(flattenNanoGptRawList({ models: cached.models }));
  }
  const savedAtRaw = cached.savedAt;
  const savedAt = Number(savedAtRaw);
  // If savedAt is missing or invalid, treat the cache as stale so we attempt a refresh.
  const age = Number.isFinite(savedAt) ? (Date.now() - savedAt) : Infinity;
  return {
    models,
    savedAt: Number.isFinite(savedAt) ? savedAt : 0,
    stale: age >= NANO_GPT_MODELS_CACHE_TTL_MS,
  };
}

export function writeNanoGptModelsCache(rawList) {
  const flat = flattenNanoGptRawList(
    Array.isArray(rawList) ? { models: rawList } : { models: rawList },
  );
  const normalized = expandCachedModelEntries(flat.length ? flat : (rawList || []));
  localStorage.setItem(
    NANO_GPT_MODELS_CACHE_KEY,
    JSON.stringify({ savedAt: Date.now(), models: normalized }),
  );
  notifyNanoGptModelsCacheUpdated();
  return normalized;
}

export function findNanoGptModel(modelId, models) {
  if (!modelId) return null;
  const norm = String(modelId).trim();
  return (models || []).find(
    (m) => m.id === norm || m.id === norm.replace(/^models\//, ''),
  ) || null;
}

/**
 * Fetch NanoGPT model catalog. Uses cache when fresh unless forceRefresh.
 * Network or server failures keep any existing cached catalog.
 */
export async function refreshNanoGptModelsCache({ forceRefresh = false } = {}) {
  const existing = readNanoGptModelsCache();
  if (!forceRefresh && existing.models.length > 0 && !existing.stale) {
    return { models: existing.models, status: 'cached', savedAt: existing.savedAt };
  }

  try {
    // Helpful log so users can see explicit retries in the console.
    // This is intentionally generic and does not assume CORS issues.
    console.info('Refreshing NanoGPT model catalog…', { forceRefresh });
    const res = await fetch(NANO_GPT_MODELS_API, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const rawList = flattenNanoGptRawList(data);
    const models = writeNanoGptModelsCache(rawList);
    return { models, status: 'ok', savedAt: Date.now() };
  } catch (e) {
    console.error('NanoGPT models fetch failed (using cached catalog when available):', e);
    if (existing.models.length > 0) {
      return { models: existing.models, status: 'fallback', savedAt: existing.savedAt, error: e };
    }
    return { models: [], status: 'error', savedAt: 0, error: e };
  }
}

/** Subscribe to cache updates (same-tab custom event + storage). */
export function subscribeNanoGptModelsCache(listener) {
  const onStorage = (ev) => {
    if (ev.key === NANO_GPT_MODELS_CACHE_KEY) listener(readNanoGptModelsCache());
  };
  const onCustom = () => listener(readNanoGptModelsCache());
  window.addEventListener('storage', onStorage);
  window.addEventListener('nanoGpt-models-cache-updated', onCustom);
  return () => {
    window.removeEventListener('storage', onStorage);
    window.removeEventListener('nanoGpt-models-cache-updated', onCustom);
  };
}

export function notifyNanoGptModelsCacheUpdated() {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new Event('nanoGpt-models-cache-updated'));
  }
}
