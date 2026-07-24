import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applyServiceEndpoints,
  getConfig,
  loadPortConfig,
  normalisePortConfig,
  SERVICE_ENDPOINTS_CHANGED_EVENT,
} from './api.js';

test('accepts dynamic desktop service endpoints and their selected ports', () => {
  assert.deepEqual(
    normalisePortConfig({
      backend: 'http://127.0.0.1:18100',
      secondary: 'http://127.0.0.1:18100',
      tts: 'http://127.0.0.1:18102',
      backendPort: 18100,
      ttsPort: 18102,
    }),
    {
      backend: 'http://127.0.0.1:18100',
      secondary: 'http://127.0.0.1:18100',
      tts: 'http://127.0.0.1:18102',
      backendPort: 18100,
      ttsPort: 18102,
    },
  );
});

test('uses the primary backend when a host omits a secondary endpoint', () => {
  assert.deepEqual(
    normalisePortConfig({
      backend: 'http://localhost:8000',
      tts: 'http://localhost:8002',
    }),
    {
      backend: 'http://localhost:8000',
      secondary: 'http://localhost:8000',
      tts: 'http://localhost:8002',
    },
  );
});

test('rejects incomplete or non-HTTP service configurations', () => {
  assert.equal(normalisePortConfig(null), null);
  assert.equal(normalisePortConfig({ backend: 'file:///tmp', tts: 'http://localhost:8002' }), null);
  assert.equal(normalisePortConfig({ backend: 'http://localhost:8000' }), null);
});

test('retries transient desktop endpoint failures instead of caching fixed defaults', async () => {
  const selectedEndpoints = {
    backend: 'http://127.0.0.1:18100',
    secondary: 'http://127.0.0.1:18100',
    tts: 'http://127.0.0.1:18102',
    backendPort: 18100,
    ttsPort: 18102,
  };
  let attempts = 0;
  const fakeWindow = {
    __TAURI_INTERNALS__: {
      async invoke(command) {
        assert.equal(command, 'get_service_endpoints');
        attempts += 1;
        if (attempts === 1) throw new Error('host state is not ready yet');
        if (attempts === 2) return null;
        return selectedEndpoints;
      },
    },
  };

  globalThis.window = fakeWindow;
  try {
    const loaded = await loadPortConfig();

    assert.equal(attempts, 3);
    assert.deepEqual(loaded, selectedEndpoints);
    assert.deepEqual(getConfig(), selectedEndpoints);
    assert.deepEqual(fakeWindow.__MIRID_SERVICE_ENDPOINTS__, selectedEndpoints);

    const descriptor = Object.getOwnPropertyDescriptor(
      fakeWindow,
      '__MIRID_SERVICE_ENDPOINTS__',
    );
    assert.equal(descriptor.set, undefined);
    assert.equal(descriptor.configurable, false);
    assert.equal(Object.isFrozen(fakeWindow.__MIRID_SERVICE_ENDPOINTS__), true);
    assert.throws(() => {
      fakeWindow.__MIRID_SERVICE_ENDPOINTS__.backend = 'http://localhost:8000';
    }, TypeError);

    await loadPortConfig();
    assert.equal(attempts, 3);
  } finally {
    delete globalThis.window;
  }
});

test('applies changed endpoints, updates the diagnostic hook and announces the payload', () => {
  const events = [];
  const fakeWindow = {
    dispatchEvent(event) {
      events.push(event);
      return true;
    },
  };
  globalThis.window = fakeWindow;

  try {
    const changed = {
      backend: 'http://127.0.0.1:18100',
      tts: 'http://127.0.0.1:19102',
      backendPort: 18100,
      ttsPort: 19102,
    };
    const applied = applyServiceEndpoints(changed);

    assert.deepEqual(applied, {
      ...changed,
      secondary: changed.backend,
    });
    assert.deepEqual(fakeWindow.__MIRID_SERVICE_ENDPOINTS__, applied);
    assert.equal(events.length, 1);
    assert.equal(events[0].type, SERVICE_ENDPOINTS_CHANGED_EVENT);
    assert.deepEqual(events[0].detail, applied);
  } finally {
    delete globalThis.window;
  }
});
