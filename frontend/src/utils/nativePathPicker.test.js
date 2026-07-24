import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { describe, it } from 'node:test';
import {
  buildBackendPickerRequest,
  buildNativeDialogOptions,
  normaliseNativePickerResult,
  openPathPicker,
} from './nativePathPicker.js';

describe('native path picker', () => {
  it('builds a directory-only native dialog with the current path', () => {
    assert.deepEqual(
      buildNativeDialogOptions({
        mode: 'directory',
        title: 'Choose image models',
        initialDirectory: ' C:\\Models\\Images ',
        multiple: true,
      }),
      {
        directory: true,
        multiple: false,
        title: 'Choose image models',
        defaultPath: 'C:\\Models\\Images',
      },
    );
  });

  it('preserves every selected file for Voice Sculpt multi-select', () => {
    assert.deepEqual(
      normaliseNativePickerResult(
        ['C:\\Voices\\one.wav', 'C:\\Voices\\two.flac'],
        { mode: 'file', multiple: true },
      ),
      {
        status: 'success',
        files: ['C:\\Voices\\one.wav', 'C:\\Voices\\two.flac'],
      },
    );
  });

  it('treats a closed native dialog as cancellation', () => {
    assert.deepEqual(
      normaliseNativePickerResult(null, { mode: 'file', multiple: true }),
      { status: 'cancelled' },
    );
  });

  it('uses the native adapter in Tauri without contacting the backend', async () => {
    let nativeOptions;
    let backendCalled = false;
    const result = await openPathPicker(
      {
        mode: 'directory',
        backendUrl: 'http://localhost:8000',
        title: 'Choose models',
        initialDirectory: 'C:\\Models',
      },
      {
        runningInTauri: true,
        nativeOpen: async (options) => {
          nativeOptions = options;
          return 'D:\\Mirid Models';
        },
        backendOpen: async () => {
          backendCalled = true;
          return { status: 'cancelled' };
        },
      },
    );

    assert.equal(backendCalled, false);
    assert.deepEqual(nativeOptions, {
      directory: true,
      multiple: false,
      title: 'Choose models',
      defaultPath: 'C:\\Models',
    });
    assert.deepEqual(result, {
      status: 'success',
      directory: 'D:\\Mirid Models',
    });
  });

  it('retains the backend picker outside Tauri', async () => {
    let request;
    const result = await openPathPicker(
      {
        mode: 'file',
        backendUrl: 'http://localhost:8000/',
        title: 'Choose references',
        initialDirectory: 'C:\\Voices',
        multiple: true,
      },
      {
        runningInTauri: false,
        nativeOpen: async () => {
          throw new Error('native dialog should not run');
        },
        backendOpen: async (nextRequest) => {
          request = nextRequest;
          return { status: 'success', files: ['C:\\Voices\\one.wav'] };
        },
      },
    );

    assert.equal(request.url, 'http://localhost:8000/system/select-file');
    assert.deepEqual(JSON.parse(request.init.body), {
      initial_directory: 'C:\\Voices',
      title: 'Choose references',
      multiple: true,
    });
    assert.deepEqual(result, {
      status: 'success',
      files: ['C:\\Voices\\one.wav'],
    });
  });

  it('builds the existing browser directory endpoint request', () => {
    const request = buildBackendPickerRequest({
      mode: 'directory',
      backendUrl: 'http://127.0.0.1:8000/',
      initialDirectory: '',
    });
    assert.equal(request.url, 'http://127.0.0.1:8000/system/select-directory');
    assert.deepEqual(JSON.parse(request.init.body), {
      initial_directory: null,
      title: null,
    });
  });

  it('grants open-dialog permission to the main and standalone settings windows', async () => {
    const defaultCapabilityUrl = new URL(
      '../../../src-tauri/capabilities/default.json',
      import.meta.url,
    );
    const settingsCapabilityUrl = new URL(
      '../../../src-tauri/capabilities/settings-dialog.json',
      import.meta.url,
    );
    const [defaultCapability, settingsCapability] = await Promise.all([
      readFile(defaultCapabilityUrl, 'utf8').then(JSON.parse),
      readFile(settingsCapabilityUrl, 'utf8').then(JSON.parse),
    ]);

    assert.deepEqual(defaultCapability.windows, ['main']);
    assert.ok(defaultCapability.permissions.includes('dialog:allow-open'));
    assert.ok(defaultCapability.permissions.includes('core:webview:allow-create-webview-window'));
    assert.ok(defaultCapability.permissions.includes('core:window:allow-close'));
    assert.equal(defaultCapability.permissions.includes('dialog:default'), false);
    assert.deepEqual(settingsCapability.windows, ['settings']);
    assert.deepEqual(
      settingsCapability.permissions,
      ['core:window:allow-close', 'dialog:allow-open'],
    );
  });
});
