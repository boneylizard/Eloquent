/**
 * Save full-message TTS audio (REST /tts single response, not WebSocket streaming).
 * Uses File System Access API when available; otherwise falls back to download.
 */

import * as indexedDbStorage from './indexedDbStorage';

const DIR_HANDLE_KEY = 'LiangLocal-tts-full-audio-dir';

let cachedDirHandle = null;

function extensionForBlob(blob) {
  const t = (blob && blob.type) || '';
  if (t.includes('wav')) return '.wav';
  if (t.includes('mpeg') || t.includes('mp3')) return '.mp3';
  if (t.includes('ogg')) return '.ogg';
  if (t.includes('webm')) return '.webm';
  return '.wav';
}

function safeFilenamePart(messageId) {
  return String(messageId || 'msg').replace(/[^\w.-]+/g, '_').slice(-48);
}

export async function loadTtsSaveDirectoryHandle() {
  if (cachedDirHandle) return cachedDirHandle;
  try {
    const h = await indexedDbStorage.getItem(DIR_HANDLE_KEY);
    if (h && typeof h.queryPermission === 'function') {
      const perm = await h.queryPermission({ mode: 'readwrite' });
      if (perm !== 'granted') {
        const req = await h.requestPermission({ mode: 'readwrite' });
        if (req !== 'granted') return null;
      }
    }
    cachedDirHandle = h || null;
    return cachedDirHandle;
  } catch {
    return null;
  }
}

export async function storeTtsSaveDirectoryHandle(handle) {
  cachedDirHandle = handle || null;
  if (handle) {
    await indexedDbStorage.setItem(DIR_HANDLE_KEY, handle);
  } else {
    await indexedDbStorage.removeItem(DIR_HANDLE_KEY);
  }
}

export async function clearTtsSaveDirectoryForFullAudio() {
  cachedDirHandle = null;
  await indexedDbStorage.removeItem(DIR_HANDLE_KEY);
}

/** Must run from a user gesture (click). Returns handle or null. */
export async function pickTtsSaveDirectory() {
  if (typeof window === 'undefined' || !window.showDirectoryPicker) {
    return null;
  }
  try {
    const handle = await window.showDirectoryPicker();
    await storeTtsSaveDirectoryHandle(handle);
    return handle;
  } catch (e) {
    if (e && e.name === 'AbortError') return null;
    console.warn('[ttsFullResponseSave] pick folder failed:', e);
    return null;
  }
}

async function writeBlobToDirectory(dirHandle, blob, filename) {
  const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/**
 * @param {string} audioObjectUrl from synthesizeSpeech (blob URL)
 * @param {string} messageId chat message id
 */
export async function saveFullTtsResponseAudio(audioObjectUrl, messageId) {
  if (!audioObjectUrl || !audioObjectUrl.startsWith('blob:')) return;

  let blob;
  try {
    const resp = await fetch(audioObjectUrl);
    blob = await resp.blob();
  } catch (e) {
    console.warn('[ttsFullResponseSave] fetch blob failed:', e);
    return;
  }

  if (!blob || blob.size === 0) return;

  const ext = extensionForBlob(blob);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `tts-full-${stamp}-${safeFilenamePart(messageId)}${ext}`;

  let dir = cachedDirHandle || (await loadTtsSaveDirectoryHandle());

  if (dir && typeof dir.getFileHandle === 'function') {
    try {
      await writeBlobToDirectory(dir, blob, filename);
      console.log(`[ttsFullResponseSave] Saved ${filename}`);
      return;
    } catch (e) {
      console.warn('[ttsFullResponseSave] write failed, falling back to download:', e);
    }
  }

  if (typeof document !== 'undefined') {
    triggerDownload(blob, filename);
    console.log(`[ttsFullResponseSave] Downloaded ${filename} (no folder access or write failed)`);
  }
}
