import test from 'node:test';
import assert from 'node:assert/strict';

import {
  getAudioStartupDefaultsMigration,
  normaliseInstallerAudioProfile,
} from './installerAudioProfile.js';

test('normalises supported installer audio choices', () => {
  assert.deepEqual(
    normaliseInstallerAudioProfile({
      ttsEnabled: true,
      sttEnabled: true,
      ttsEngine: 'kokoro',
      sttEngine: 'whisper',
      nanoGptApiKey: ' key-value ',
    }),
    {
      ttsEnabled: true,
      sttEnabled: true,
      ttsEngine: 'kokoro',
      sttEngine: 'whisper',
      nanoGptApiKey: 'key-value',
    },
  );
});

test('drops unsupported installer audio choices', () => {
  assert.deepEqual(
    normaliseInstallerAudioProfile({
      ttsEnabled: false,
      ttsEngine: 'unknown-engine',
      sttEngine: '../../other',
    }),
    { ttsEnabled: false },
  );
});

test('enables audio once when upgrading from the old defaults', () => {
  assert.deepEqual(getAudioStartupDefaultsMigration({ sttEnabled: false, ttsEnabled: false }), {
    sttEnabled: true,
    ttsEnabled: true,
    audioStartupDefaultsVersion: 1,
  });
  assert.equal(getAudioStartupDefaultsMigration({ audioStartupDefaultsVersion: 1 }), null);
});
