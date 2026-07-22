import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildChatTitlePrompt,
  cleanGeneratedChatTitle,
  createFallbackChatTitle,
  generateChatTitle,
} from './chatTitle.js';

describe('chat title formatting', () => {
  it('creates a concise fallback from the first message', () => {
    assert.equal(
      createFallbackChatTitle('Help me choose a small local vision model for screenshots and photographs.'),
      'Help me choose a small local vision model',
    );
  });

  it('cleans common model wrappers without flattening the title', () => {
    assert.equal(cleanGeneratedChatTitle('Title: “Choosing a Local Vision Model”'), 'Choosing a Local Vision Model');
    assert.equal(cleanGeneratedChatTitle('{"title":"A Door Left Open"}'), 'A Door Left Open');
    assert.equal(cleanGeneratedChatTitle('<think>Drafting.</think>\n**Moonlit Train Platform**'), 'Moonlit Train Platform');
  });

  it('builds the title from the first exchange', () => {
    const prompt = buildChatTitlePrompt([
      { role: 'user', content: 'Why did the lighthouse go dark?' },
      { role: 'bot', content: 'The keeper vanished during the storm.' },
    ]);
    assert.match(prompt, /Why did the lighthouse go dark/);
    assert.match(prompt, /keeper vanished during the storm/);
  });
});

describe('chat title request', () => {
  it('uses the active route and returns a generated title', async () => {
    let request = null;
    const result = await generateChatTitle({
      messages: [
        { role: 'user', content: 'Why did the lighthouse go dark?' },
        { role: 'bot', content: 'The keeper vanished during the storm.' },
      ],
      modelName: 'endpoint-openrouter-free',
      selectedModel: 'endpoint-openrouter-free',
      roundRobinEnabled: true,
      apiBaseUrl: 'http://localhost:8000/',
      fetchImpl: async (url, init) => {
        request = { url, payload: JSON.parse(init.body) };
        return { ok: true, json: async () => ({ text: 'The Vanished Lighthouse Keeper' }) };
      },
    });
    assert.equal(result.title, 'The Vanished Lighthouse Keeper');
    assert.equal(result.source, 'generated');
    assert.equal(request.url, 'http://localhost:8000/generate');
    assert.equal(request.payload.request_purpose, 'title_generation');
    assert.equal(request.payload.round_robin_enabled, true);
    assert.equal(request.payload.selected_model, 'endpoint-openrouter-free');
  });

  it('falls back to the first message when generation fails', async () => {
    const result = await generateChatTitle({
      messages: [{ role: 'user', content: 'Plan a winter garden for this balcony' }],
      modelName: 'local-model',
      fetchImpl: async () => ({ ok: false, status: 503 }),
    });
    assert.deepEqual(result, {
      title: 'Plan a winter garden for this balcony',
      source: 'fallback',
    });
  });
});
