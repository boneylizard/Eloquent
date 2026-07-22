import assert from 'node:assert/strict';
import test from 'node:test';

import {
  annotateHostedModel,
  getConnectedHostedProviderIds,
  syncHostedProviderEndpointKey,
  upsertHostedModelEndpoint,
} from './hostedModelProviders.js';

test('keeps the hosting provider separate from the model maker', () => {
  const model = annotateHostedModel({ id: 'anthropic/claude', provider: 'anthropic' }, 'openrouter');
  assert.equal(model.hostProvider, 'openrouter');
  assert.equal(model.hostProviderLabel, 'OpenRouter');
  assert.equal(model.modelProvider, 'anthropic');
});

test('connected provider keys determine which catalogues are usable', () => {
  assert.deepEqual(
    getConnectedHostedProviderIds({ nanoGptApiKey: 'nano-key', openAiApiKey: 'openai-key' }),
    ['nanogpt', 'openai'],
  );
});

test('selecting a hosted model creates one stable provider endpoint', () => {
  const first = upsertHostedModelEndpoint({
    endpoints: [],
    providerId: 'openrouter',
    apiKey: 'first-key',
    model: { id: 'anthropic/claude', name: 'Claude', provider: 'anthropic' },
  });
  const second = upsertHostedModelEndpoint({
    endpoints: first.endpoints,
    providerId: 'openrouter',
    apiKey: 'second-key',
    model: { id: 'anthropic/claude', name: 'Claude', provider: 'anthropic' },
  });

  assert.equal(second.endpoints.length, 1);
  assert.equal(second.endpointId, first.endpointId);
  assert.equal(second.endpoint.provider, 'openrouter');
  assert.equal(second.endpoint.model_provider, 'anthropic');
  assert.equal(second.endpoint.apiKey, 'second-key');
  assert.equal(second.endpoint.url, 'https://openrouter.ai/api/v1');
});

test('uses the provider URL when a catalogue omits its base URL', () => {
  const result = upsertHostedModelEndpoint({
    endpoints: [],
    providerId: 'nanogpt',
    apiKey: 'nano-key',
    baseUrl: undefined,
    billingMode: 'subscription',
    model: { id: 'zhipu/glm-roleplay-test', name: 'GLM Roleplay Test', provider: 'zhipu' },
  });

  assert.equal(result.endpoint.url, 'https://nano-gpt.com/api/v1');
  assert.equal(result.endpoint.billing_mode, 'subscription');
});

test('changing a provider key updates only that provider endpoints', () => {
  const endpoints = syncHostedProviderEndpointKey([
    { id: 'nano', provider: 'nanogpt', apiKey: 'old' },
    { id: 'router', provider: 'openrouter', apiKey: 'keep' },
  ], 'nanogpt', 'new');

  assert.equal(endpoints[0].apiKey, 'new');
  assert.equal(endpoints[1].apiKey, 'keep');
});
