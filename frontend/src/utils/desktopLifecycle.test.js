import assert from 'node:assert/strict';
import test from 'node:test';

import { restartMirid, restartTtsService, shutdownMirid, stopTtsService } from './desktopLifecycle.js';


test('desktop lifecycle controls fail clearly outside Tauri', async () => {
  for (const action of [restartMirid, shutdownMirid, stopTtsService, restartTtsService]) {
    await assert.rejects(action(), /available in the Mirid desktop app/);
  }
});
