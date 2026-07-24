import assert from 'node:assert/strict';
import test from 'node:test';

import { getConfig, SERVICE_ENDPOINTS_CHANGED_EVENT } from '../config/api.js';
import { restartMirid, restartTtsService, shutdownMirid, stopTtsService } from './desktopLifecycle.js';


test('desktop lifecycle controls fail clearly outside Tauri', async () => {
  for (const action of [restartMirid, shutdownMirid, stopTtsService, restartTtsService]) {
    await assert.rejects(action(), /available in the Mirid desktop app/);
  }
});

test('TTS restart applies and announces endpoints returned by the desktop host', async () => {
  const endpoints = {
    backend: 'http://127.0.0.1:18100',
    secondary: 'http://127.0.0.1:18100',
    tts: 'http://127.0.0.1:19102',
    backendPort: 18100,
    ttsPort: 19102,
  };
  const events = [];

  globalThis.isTauri = true;
  globalThis.window = {
    __TAURI_INTERNALS__: {
      async invoke(command) {
        assert.equal(command, 'restart_tts');
        return endpoints;
      },
    },
    dispatchEvent(event) {
      events.push(event);
      return true;
    },
  };

  try {
    assert.deepEqual(await restartTtsService(), endpoints);
    assert.deepEqual(getConfig(), endpoints);
    assert.deepEqual(window.__MIRID_SERVICE_ENDPOINTS__, endpoints);
    assert.equal(events.length, 1);
    assert.equal(events[0].type, SERVICE_ENDPOINTS_CHANGED_EVENT);
    assert.deepEqual(events[0].detail, endpoints);
  } finally {
    delete globalThis.window;
    delete globalThis.isTauri;
  }
});
