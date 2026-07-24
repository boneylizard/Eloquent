import { isTauri } from '@tauri-apps/api/core';

const PICKER_MODES = new Set(['directory', 'file']);

const cleanPath = (value) => (
  typeof value === 'string' && value.trim() ? value.trim() : null
);

export function buildNativeDialogOptions({
  mode,
  title,
  initialDirectory,
  multiple = false,
  filters,
}) {
  if (!PICKER_MODES.has(mode)) {
    throw new Error(`Unsupported path picker mode: ${mode}`);
  }

  return {
    directory: mode === 'directory',
    multiple: mode === 'file' && Boolean(multiple),
    ...(cleanPath(title) ? { title: title.trim() } : {}),
    ...(cleanPath(initialDirectory) ? { defaultPath: initialDirectory.trim() } : {}),
    ...(Array.isArray(filters) && filters.length > 0 ? { filters } : {}),
  };
}

export function buildBackendPickerRequest({
  mode,
  backendUrl,
  title,
  initialDirectory,
  multiple = false,
}) {
  if (!PICKER_MODES.has(mode)) {
    throw new Error(`Unsupported path picker mode: ${mode}`);
  }
  if (!cleanPath(backendUrl)) {
    throw new Error('The backend address is unavailable.');
  }

  const endpoint = mode === 'directory' ? 'select-directory' : 'select-file';
  return {
    url: `${backendUrl.replace(/\/+$/, '')}/system/${endpoint}`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        initial_directory: cleanPath(initialDirectory),
        title: cleanPath(title),
        ...(mode === 'file' ? { multiple: Boolean(multiple) } : {}),
      }),
    },
  };
}

export function normaliseNativePickerResult(selection, { mode, multiple = false }) {
  const paths = (Array.isArray(selection) ? selection : [selection])
    .map(cleanPath)
    .filter(Boolean);

  if (paths.length === 0) return { status: 'cancelled' };
  if (mode === 'directory') {
    return { status: 'success', directory: paths[0] };
  }
  if (multiple) {
    return { status: 'success', files: paths };
  }
  return { status: 'success', file: paths[0] };
}

async function openNativeDialog(dialogOptions) {
  const { open } = await import('@tauri-apps/plugin-dialog');
  return open(dialogOptions);
}

async function openBackendDialog(options, fetcher) {
  const { url, init } = buildBackendPickerRequest(options);
  const response = await fetcher(url, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const fallback = options.mode === 'directory'
      ? 'The folder picker could not be opened.'
      : 'The file picker could not be opened.';
    throw new Error(data.detail || data.message || fallback);
  }
  return data;
}

/**
 * Open a path picker through the desktop shell when available. Plain browser
 * development retains the backend picker so the same controls still work.
 */
export async function openPathPicker(options, adapters = {}) {
  const runningInTauri = adapters.runningInTauri
    ?? (typeof window !== 'undefined' && isTauri());

  if (runningInTauri) {
    const nativeOpen = adapters.nativeOpen || openNativeDialog;
    const selection = await nativeOpen(buildNativeDialogOptions(options));
    return normaliseNativePickerResult(selection, options);
  }

  const backendOpen = adapters.backendOpen;
  if (backendOpen) {
    return backendOpen(buildBackendPickerRequest(options));
  }

  const fetcher = adapters.fetcher || globalThis.fetch;
  if (typeof fetcher !== 'function') {
    throw new Error('The browser cannot reach the backend picker.');
  }
  return openBackendDialog(options, fetcher);
}
