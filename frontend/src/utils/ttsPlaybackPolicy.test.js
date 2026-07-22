import test from 'node:test';
import assert from 'node:assert/strict';

import {
  engineSynthesisesTtsSpeed,
  getTtsPlaybackRate,
  normaliseTtsSpeed,
} from './ttsPlaybackPolicy.js';

test('keeps TTS speed inside the supported playback range', () => {
  assert.equal(normaliseTtsSpeed(2.5), 2.5);
  assert.equal(normaliseTtsSpeed(99), 4);
  assert.equal(normaliseTtsSpeed(0), 1);
});

test('does not apply Kokoro or NanoGPT speed twice', () => {
  assert.equal(engineSynthesisesTtsSpeed('kokoro'), true);
  assert.equal(engineSynthesisesTtsSpeed('nanogpt-Kokoro-82m'), true);
  assert.equal(getTtsPlaybackRate('kokoro', 2), 1);
  assert.equal(getTtsPlaybackRate('nanogpt-Kokoro-82m', 2), 1);
});

test('uses browser playback speed for engines without synthesis speed', () => {
  assert.equal(engineSynthesisesTtsSpeed('chatterbox_turbo'), false);
  assert.equal(getTtsPlaybackRate('chatterbox_turbo', 1.7), 1.7);
});
