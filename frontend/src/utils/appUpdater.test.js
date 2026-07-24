import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createUpdateProgress,
  formatBytes,
  formatUpdateProgress,
  installAppUpdate,
  reduceUpdateProgress,
} from './appUpdater.js';

test('update progress reports percentage and download speed', () => {
  let progress = reduceUpdateProgress(
    createUpdateProgress(),
    { event: 'Started', data: { contentLength: 10_000 } },
    1_000,
  );
  progress = reduceUpdateProgress(
    progress,
    { event: 'Progress', data: { chunkLength: 2_500 } },
    2_000,
  );

  assert.equal(progress.percent, 25);
  assert.equal(progress.downloadedBytes, 2_500);
  assert.equal(progress.bytesPerSecond, 2_500);
  assert.match(formatUpdateProgress(progress), /^25%/);
});

test('update progress handles downloads without a content length', () => {
  let progress = reduceUpdateProgress(
    createUpdateProgress(),
    { event: 'Started', data: {} },
    1_000,
  );
  progress = reduceUpdateProgress(
    progress,
    { event: 'Progress', data: { chunkLength: 1_048_576 } },
    3_000,
  );

  assert.equal(progress.percent, null);
  assert.equal(formatUpdateProgress(progress), '1.0 MB · 512.0 KB/s');
});

test('byte formatting remains readable across common update sizes', () => {
  assert.equal(formatBytes(700 * 1024), '700.0 KB');
  assert.equal(formatBytes(32 * 1024 * 1024), '32.0 MB');
});

test('installed updates restart through Mirid lifecycle cleanup', async () => {
  const calls = [];
  const update = {
    downloadAndInstall: async (onEvent, options) => {
      calls.push(['install', options]);
      onEvent({ event: 'Finished', data: {} });
    },
  };

  await installAppUpdate(
    update,
    (progress) => calls.push(['progress', progress.phase]),
    async () => calls.push(['restart']),
  );

  assert.deepEqual(calls, [
    ['install', { timeout: 30 * 60_000 }],
    ['progress', 'installing'],
    ['restart'],
  ]);
});
