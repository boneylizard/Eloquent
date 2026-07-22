import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { extractSseStreamParts } from './streamDelta.js';

describe('SSE stream part extraction', () => {
  it('extracts standard delta content', () => {
    const parts = extractSseStreamParts({ choices: [{ delta: { content: 'hello' } }] });
    assert.equal(parts.text, 'hello');
    assert.equal(parts.reasoning, '');
  });

  it('extracts named reasoning alongside content', () => {
    const parts = extractSseStreamParts({
      choices: [{ delta: { reasoning: 'think', content: 'answer' } }],
    });
    assert.equal(parts.text, 'answer');
    assert.equal(parts.reasoning, 'think');
  });

  it('joins OpenRouter reasoning_details entries', () => {
    const parts = extractSseStreamParts({
      choices: [{ delta: { reasoning_details: [{ text: 'step one ' }, { text: 'step two' }] } }],
    });
    assert.equal(parts.text, '');
    assert.equal(parts.reasoning, 'step one step two');
  });

  it('never treats serving metadata as reasoning', () => {
    // vLLM deployment identifiers stamped on OpenRouter chunks must never
    // surface in the thinking block (regression: doubled "vllm-..." leak).
    const parts = extractSseStreamParts({
      id: 'gen-123',
      provider: 'vllm-0.25.0-dp4-ep-d6f08423',
      system_fingerprint: 'vllm-0.25.0-dp4-ep-d6f08423',
      model: 'deepseek/deepseek-v4-flash',
      choices: [{ delta: { role: 'assistant', content: '' } }],
    });
    assert.equal(parts.text, '');
    assert.equal(parts.reasoning, '');
  });

  it('never treats unknown long string fields as reasoning', () => {
    const parts = extractSseStreamParts({
      x_custom_upstream_trace: 'a-very-long-identifier-without-spaces-12345',
      choices: [{ delta: { content: 'real answer' } }],
    });
    assert.equal(parts.text, 'real answer');
    assert.equal(parts.reasoning, '');
  });

  it('never treats token count fields as reasoning', () => {
    const parts = extractSseStreamParts({
      reasoning_tokens: '512',
      choices: [{ delta: { thinking_tokens: '128', content: 'answer' } }],
    });
    assert.equal(parts.text, 'answer');
    assert.equal(parts.reasoning, '');
  });
});
