import test from 'node:test';
import assert from 'node:assert/strict';
import {
  OPENROUTER_FREE_MODEL_ID,
  normalizeOpenRouterModel,
  normalizeOpenRouterModels,
} from './openRouterModelsCache.js';

test('normalises OpenRouter capabilities and free pricing', () => {
  const model = normalizeOpenRouterModel({
    id: 'example/model:free',
    name: 'Example',
    context_length: 131072,
    pricing: { prompt: '0', completion: '0' },
    architecture: { input_modalities: ['text', 'image'], output_modalities: ['text'] },
    supported_parameters: ['tools', 'reasoning'],
  });

  assert.equal(model.free, true);
  assert.equal(model.contextLength, 131072);
  assert.deepEqual(model.outputModalities, ['text']);
  assert.deepEqual(model.capabilities, { vision: true, tools: true, reasoning: true });
});

test('filters models that cannot produce text', () => {
  const models = normalizeOpenRouterModels({
    data: [
      { id: 'example/chat', architecture: { output_modalities: ['text'] } },
      { id: 'example/image', architecture: { output_modalities: ['image'] } },
      { id: 'example/music', architecture: { output_modalities: ['text', 'audio'] } },
    ],
  });

  assert.equal(models.some((model) => model.id === 'example/chat'), true);
  assert.equal(models.some((model) => model.id === 'example/image'), false);
  assert.equal(models.some((model) => model.id === 'example/music'), false);
});

test('always places the stable free router first', () => {
  const models = normalizeOpenRouterModels({
    data: [{ id: 'paid/model', name: 'Paid model', pricing: { prompt: '1', completion: '2' } }],
  });

  assert.equal(models[0].id, OPENROUTER_FREE_MODEL_ID);
  assert.equal(models[0].isFreeRouter, true);
  assert.equal(models[1].free, false);
});
