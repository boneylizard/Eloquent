import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FIRST_RUN_INTENT_KEY,
  ROLEPLAY_THEME,
  normaliseFirstRunIntent,
  readFirstRunIntent,
  writeFirstRunIntent,
} from './firstRunIntent.js';

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
  };
}

test('rejects unknown first-run purposes', () => {
  assert.equal(normaliseFirstRunIntent({ purpose: 'mystery' }), null);
});

test('roleplay intent selects the Faraday-inspired theme', () => {
  const storage = memoryStorage();
  const saved = writeFirstRunIntent({ purpose: 'roleplay', interfaceZoom: 1.2 }, storage);
  assert.equal(saved.purpose, 'roleplay');
  assert.equal(storage.getItem('vite-ui-theme'), ROLEPLAY_THEME);
  assert.deepEqual(readFirstRunIntent(storage), JSON.parse(storage.getItem(FIRST_RUN_INTENT_KEY)));
});

test('interface zoom is bounded before persistence', () => {
  const storage = memoryStorage();
  const saved = writeFirstRunIntent({ purpose: 'classic', interfaceZoom: 9 }, storage);
  assert.equal(saved.interfaceZoom, 2);
});

test('legacy general-purpose choices migrate to Mirid Classic', () => {
  assert.equal(normaliseFirstRunIntent({ purpose: 'writing' }).purpose, 'classic');
  assert.equal(normaliseFirstRunIntent({ purpose: 'everything' }).purpose, 'classic');
});
