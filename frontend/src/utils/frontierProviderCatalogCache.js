const CACHE_KEY = 'mirid-frontier-provider-catalogues-v1';
const CACHE_TTL_MS = 60 * 60 * 1000;

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function credentialTag(apiKey) {
  let hash = 2166136261;
  for (const character of String(apiKey || '')) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function readAll() {
  if (typeof localStorage === 'undefined') return {};
  return safeJsonParse(localStorage.getItem(CACHE_KEY) || '{}') || {};
}

export function readFrontierProviderCatalog(providerId, apiKey) {
  const record = readAll()[providerId];
  const savedAt = Number(record?.savedAt);
  const matchesCredential = record?.credentialTag === credentialTag(apiKey);
  return {
    provider: providerId,
    baseUrl: matchesCredential ? String(record?.baseUrl || '') : '',
    models: matchesCredential && Array.isArray(record?.models) ? record.models : [],
    savedAt: matchesCredential && Number.isFinite(savedAt) ? savedAt : 0,
    stale: !matchesCredential || !Number.isFinite(savedAt) || Date.now() - savedAt >= CACHE_TTL_MS,
  };
}

export function writeFrontierProviderCatalog(providerId, apiKey, payload) {
  const catalogues = readAll();
  const record = {
    credentialTag: credentialTag(apiKey),
    savedAt: Date.now(),
    baseUrl: String(payload?.base_url || payload?.baseUrl || ''),
    models: Array.isArray(payload?.models) ? payload.models : [],
  };
  catalogues[providerId] = record;
  localStorage.setItem(CACHE_KEY, JSON.stringify(catalogues));
  window.dispatchEvent(new CustomEvent('mirid-provider-catalogue-updated', { detail: { providerId } }));
  return {
    provider: providerId,
    baseUrl: record.baseUrl,
    models: record.models,
    savedAt: record.savedAt,
    stale: false,
  };
}

export async function refreshFrontierProviderCatalog({
  providerId,
  apiKey,
  primaryApiUrl,
  forceRefresh = false,
}) {
  const existing = readFrontierProviderCatalog(providerId, apiKey);
  if (!forceRefresh && existing.models.length > 0 && !existing.stale) {
    return { ...existing, status: 'cached' };
  }
  const response = await fetch(`${primaryApiUrl}/provider-catalog/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider: providerId, api_key: apiKey }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || 'The provider model catalogue could not be read.');
  }
  return { ...writeFrontierProviderCatalog(providerId, apiKey, payload), status: 'ok' };
}

export function subscribeFrontierProviderCatalogs(listener) {
  const onStorage = (event) => {
    if (event.key === CACHE_KEY) listener();
  };
  const onCustom = () => listener();
  window.addEventListener('storage', onStorage);
  window.addEventListener('mirid-provider-catalogue-updated', onCustom);
  return () => {
    window.removeEventListener('storage', onStorage);
    window.removeEventListener('mirid-provider-catalogue-updated', onCustom);
  };
}
