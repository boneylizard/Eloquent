import { FRONTIER_PROVIDERS } from '../config/frontierProviders.js';

const BASE_PROVIDERS = [
  {
    id: 'nanogpt',
    label: 'NanoGPT',
    keySetting: 'nanoGptApiKey',
    baseUrl: 'https://nano-gpt.com/api/v1',
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    keySetting: 'openRouterApiKey',
    baseUrl: 'https://openrouter.ai/api/v1',
  },
];

export const HOSTED_MODEL_PROVIDERS = Object.freeze([
  ...BASE_PROVIDERS,
  ...FRONTIER_PROVIDERS.map((provider) => ({ ...provider })),
]);

export function getHostedModelProvider(providerId) {
  const normalisedId = String(providerId || '').trim().toLowerCase();
  return HOSTED_MODEL_PROVIDERS.find((provider) => provider.id === normalisedId) || {
    id: normalisedId || 'custom',
    label: normalisedId || 'Custom API',
    keySetting: '',
    baseUrl: '',
  };
}

export function getHostedProviderLabel(providerId) {
  return getHostedModelProvider(providerId).label;
}

export function getHostedProviderKey(settings, providerId) {
  const provider = getHostedModelProvider(providerId);
  return String(settings?.[provider.keySetting] || '').trim();
}

export function getConnectedHostedProviderIds(settings) {
  return HOSTED_MODEL_PROVIDERS
    .filter((provider) => provider.keySetting && String(settings?.[provider.keySetting] || '').trim())
    .map((provider) => provider.id);
}

export function annotateHostedModel(model, providerId, overrides = {}) {
  const provider = getHostedModelProvider(providerId);
  const modelId = String(model?.id || model?.model || '').trim();
  if (!modelId) return null;
  const modelProvider = String(
    model?.modelProvider
    || model?.model_provider
    || model?.provider
    || '',
  ).trim();
  return {
    ...model,
    id: modelId,
    name: String(model?.name || model?.display_name || modelId),
    description: String(model?.description || ''),
    contextLength: Number(model?.contextLength || model?.context_length || 0) || null,
    capabilities: model?.capabilities || {},
    hostProvider: provider.id,
    hostProviderLabel: provider.label,
    modelProvider,
    ...overrides,
    baseUrl: String(overrides.baseUrl || model?.baseUrl || provider.baseUrl || '').replace(/\/$/, ''),
  };
}

export function findHostedModelEndpoint(endpoints, providerId, modelId) {
  const normalisedProvider = String(providerId || '').trim().toLowerCase();
  const normalisedModel = String(modelId || '').trim().replace(/^models\//, '');
  return (endpoints || []).find((endpoint) => (
    String(endpoint?.provider || '').trim().toLowerCase() === normalisedProvider
    && String(endpoint?.model || '').trim().replace(/^models\//, '') === normalisedModel
  )) || null;
}

function endpointSlug(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72) || 'model';
}

export function upsertHostedModelEndpoint({
  endpoints = [],
  model,
  providerId,
  apiKey,
  baseUrl,
  billingMode,
}) {
  const hostedModel = annotateHostedModel(model, providerId, { baseUrl });
  if (!hostedModel) throw new Error('A provider model needs an ID.');
  const provider = getHostedModelProvider(providerId);
  const existing = findHostedModelEndpoint(endpoints, provider.id, hostedModel.id);
  const endpointId = existing?.id
    || (provider.id === 'openrouter' && hostedModel.id === 'openrouter/free'
      ? 'endpoint-openrouter-free'
      : `endpoint-${provider.id}-${endpointSlug(hostedModel.id)}`);
  const nextEndpoint = {
    ...existing,
    id: endpointId,
    name: `${provider.label} · ${hostedModel.name}`,
    url: hostedModel.baseUrl,
    apiKey: String(apiKey || '').trim(),
    model: hostedModel.id,
    provider: provider.id,
    provider_label: provider.label,
    model_provider: hostedModel.modelProvider,
    enabled: true,
    rotate_enabled: existing?.rotate_enabled !== false,
    context_window: hostedModel.contextLength || existing?.context_window || null,
  };
  if (billingMode) nextEndpoint.billing_mode = billingMode;
  return {
    endpointId,
    endpoint: nextEndpoint,
    endpoints: existing
      ? endpoints.map((endpoint) => (endpoint.id === existing.id ? nextEndpoint : endpoint))
      : [...endpoints, nextEndpoint],
  };
}

export function syncHostedProviderEndpointKey(endpoints, providerId, apiKey) {
  const normalisedProvider = String(providerId || '').trim().toLowerCase();
  const nextKey = String(apiKey || '').trim();
  return (endpoints || []).map((endpoint) => (
    String(endpoint?.provider || '').trim().toLowerCase() === normalisedProvider
      ? { ...endpoint, apiKey: nextKey }
      : endpoint
  ));
}
