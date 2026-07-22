import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { closeCurrentMiridWindow, openMiridWindow } from './desktopWindows.js';

const originalWindow = globalThis.window;
const originalIsTauri = globalThis.isTauri;

afterEach(() => {
  globalThis.window = originalWindow;
  globalThis.isTauri = originalIsTauri;
});

describe('desktop window fallback', () => {
  it('opens a browser window outside Tauri', async () => {
    const marker = {};
    let received;
    globalThis.isTauri = false;
    globalThis.window = {
      open: (...args) => {
        received = args;
        return marker;
      },
    };

    const result = await openMiridWindow({
      label: 'test-window',
      url: '?standalone=test',
      browserFeatures: 'width=400,height=300',
    });

    assert.equal(result, marker);
    assert.deepEqual(received, ['?standalone=test', 'test-window', 'width=400,height=300']);
  });

  it('closes the browser window outside Tauri', async () => {
    let closed = false;
    globalThis.isTauri = false;
    globalThis.window = { close: () => { closed = true; } };

    assert.equal(await closeCurrentMiridWindow(), true);
    assert.equal(closed, true);
  });
});
