import assert from 'node:assert/strict';
import test from 'node:test';

import * as indexedDbStorage from './indexedDbStorage.js';
import {
  disableRetiredAntiRepetition,
  mergeSettingsObjects,
  parseSettingsJson,
  replaceSettingsBlob,
} from './settingsPersistence.js';

function installLocalStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  globalThis.localStorage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
    key: (index) => [...values.keys()][index] ?? null,
    get length() {
      return values.size;
    },
  };
  return values;
}

test('normal settings writes mirror synchronously before IndexedDB finishes', async () => {
  const values = installLocalStorage();
  const serialized = JSON.stringify({ theme: 'dark', ttsEnabled: true });

  const write = indexedDbStorage.setItem('Eloquent-settings', serialized);

  assert.equal(values.get('Eloquent-settings'), serialized);
  await write;
});

test('explicit restore replaces stale settings in both browser mirrors', async () => {
  const values = installLocalStorage({
    'Eloquent-settings': JSON.stringify({ stale: true, theme: 'light' }),
    'LiangLocal-settings': JSON.stringify({ stale: true, theme: 'light' }),
  });

  const restored = { theme: 'dark', ttsEnabled: true };
  assert.equal(await replaceSettingsBlob(restored), true);

  assert.deepEqual(parseSettingsJson(values.get('Eloquent-settings')), restored);
  assert.deepEqual(parseSettingsJson(values.get('LiangLocal-settings')), restored);
});

test('empty patches cannot replace populated settings', () => {
  assert.deepEqual(
    mergeSettingsObjects({ theme: 'dark' }, {}),
    { theme: 'dark' },
  );
});

test('legacy anti-repetition settings are forced off', () => {
  assert.deepEqual(
    disableRetiredAntiRepetition({
      theme: 'dark',
      antiRepetitionMode: true,
      detectRepeatedPhrases: true,
      frequencyPenalty: 1.5,
      presencePenalty: 0.8,
    }),
    {
      theme: 'dark',
      antiRepetitionMode: false,
      detectRepeatedPhrases: false,
      frequencyPenalty: 0,
      presencePenalty: 0,
    },
  );
});
