/**
 * IndexedDB-backed storage for large app data (characters, conversations, settings, etc.)
 * to avoid localStorage quota limits (~5–10MB). IndexedDB typically allows 50MB+ per origin.
 *
 * Usage: use async getItem/setItem/removeItem for keys listed in IDB_KEYS and IDB_KEY_PREFIXES.
 * Run migrateFromLocalStorage() once on app load to copy existing localStorage data into IDB
 * and optionally clear localStorage for those keys to free space.
 */

import { coalesceSettingsWrite, isSettingsStorageKey } from './settingsPersistence';
import { coerceProfilesWrite, pickBestProfilesSource } from './userProfilesStorage';

const DB_NAME = 'LiangLocal';
const DB_VERSION = 1;
const STORE_NAME = 'keyvalue';

// Keys we store in IndexedDB (exact match)
const IDB_KEYS = new Set([
  'llm-characters',
  'Eloquent-conversations',
  'Eloquent-conversations-index',
  'Eloquent-conversations-storage-v',
  'Eloquent-conversations-deleted-ids',
  'Eloquent-active-conversation',
  'Eloquent-settings',
  'user-profiles',
  'eloquent-story-tracker',
  'Eloquent-model-elo-ratings',
  'Eloquent-analysis-auto-questions',
  'Eloquent-analysis-question-perspective',
  'Eloquent-saved-perspectives',
  'LiangLocal-settings',
  'preferredContextLength',
  'eloquent-author-note',
  'adetailer-settings',
  'adetailer-selected-model',
  'adetailer-auto-enhance',
  'local-upscaler-model',
  'Eloquent-backend-port',
  'vite-ui-theme',
  'user-memories', // legacy
  'conversations',
  'Eloquent-conversations-index',
  'LiangLocal-avatar-sizes', // dedicated persistence for avatar sizes (survives backend overwrite)
  'LiangLocal-tts-full-audio-dir', // FileSystemDirectoryHandle for full-response TTS backups

]);

// Key prefixes: any key starting with these is stored in IDB
const IDB_KEY_PREFIXES = [
  'Eloquent-conversation-',
  'LiangLocal-variants-',

];

/** Never read/write these via localStorage fallback — prevents deleted chats resurrecting. */
function isConversationStorageKey(key) {
  if (!key) return false;
  if (
    key === 'Eloquent-conversations'
    || key === 'Eloquent-conversations-index'
    || key === 'Eloquent-conversations-deleted-ids'
    || key === 'Eloquent-conversations-storage-v'
    || key === 'conversations'
  ) {
    return true;
  }
  return key.startsWith('Eloquent-conversation-');
}

/** True if this key is stored in IndexedDB (used by backup import/export). */
export function useIdb(key) {
  if (IDB_KEYS.has(key)) return true;
  return IDB_KEY_PREFIXES.some(prefix => key.startsWith(prefix));
}

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB not available'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
  });
  return dbPromise;
}

/**
 * Synchronous read for keys mirrored in localStorage (settings popup fast path).
 * @param {string} key
 * @returns {string|null}
 */
export function readLocalMirror(key) {
  if (!KEEP_IN_LOCAL_STORAGE.has(key)) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * @param {string} key
 * @param {{ preferLocalStorage?: boolean, skipMigration?: boolean }} [options]
 * @returns {Promise<string|null>}
 */
export async function getItem(key, options = {}) {
  const { preferLocalStorage = false, skipMigration = false } = options;
  if (!useIdb(key)) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }
  if (preferLocalStorage && key !== 'user-profiles') {
    const mirrored = readLocalMirror(key);
    if (mirrored != null) return mirrored;
  }
  if (!skipMigration) {
    await ensureStorageMigrated();
  }
  try {
    const db = await openDb();
    const idbResult = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
    if (key === 'user-profiles' && KEEP_IN_LOCAL_STORAGE.has(key)) {
      const mirrored = readLocalMirror(key);
      return pickBestProfilesSource(idbResult, mirrored).raw;
    }
    return idbResult;
  } catch (e) {
    if (isConversationStorageKey(key)) {
      console.error('[indexedDbStorage] getItem failed for conversation key (no localStorage fallback):', key, e);
      return null;
    }
    console.warn('[indexedDbStorage] getItem failed, falling back to localStorage:', key, e);
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }
}

/**
 * @param {string} key
 * @param {string} value
 * @returns {Promise<void>}
 */
export async function setItem(key, value) {
  if (isSettingsStorageKey(key)) {
    const coalesced = coalesceSettingsWrite(key, value);
    if (coalesced == null) {
      console.warn('[indexedDbStorage] Refusing settings write (empty or would clobber existing data):', key);
      return;
    }
    value = coalesced;
  }
  if (key === 'user-profiles') {
    const coalesced = coerceProfilesWrite(value);
    if (coalesced == null) {
      console.warn('[indexedDbStorage] Refusing user-profiles write (would clobber richer localStorage data)');
      return;
    }
    value = coalesced;
  }
  if (!useIdb(key)) {
    try {
      localStorage.setItem(key, value);
    } catch (e) {
      console.warn('[indexedDbStorage] localStorage setItem failed:', key, e);
    }
    return;
  }
  try {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(value, key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
    if (KEEP_IN_LOCAL_STORAGE.has(key)) {
      try {
        localStorage.setItem(key, value);
      } catch (_) { /* quota */ }
    }
    if (isConversationStorageKey(key)) {
      try {
        localStorage.removeItem(key);
      } catch (_) { /* ghost copy must not resurrect deleted tabs */ }
    }
  } catch (e) {
    if (isConversationStorageKey(key)) {
      console.error('[indexedDbStorage] setItem failed for conversation key (no localStorage fallback):', key, e);
      throw e;
    }
    console.warn('[indexedDbStorage] setItem failed, falling back to localStorage:', key, e);
    try {
      localStorage.setItem(key, value);
    } catch (_) {}
  }
}

/**
 * @param {string} key
 * @returns {Promise<void>}
 */
export async function removeItem(key) {
  try {
    localStorage.removeItem(key);
  } catch (_) {}
  if (!useIdb(key)) return;
  try {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  } catch (e) {
    console.warn('[indexedDbStorage] removeItem failed:', key, e);
  }
}

/**
 * @returns {Promise<string[]>}
 */
export async function getAllKeys() {
  try {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAllKeys();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return [];
  }
}

/**
 * @param {string} prefix
 * @returns {Promise<string[]>}
 */
export async function getKeysByPrefix(prefix) {
  const all = await getAllKeys();
  return all.filter(k => k.startsWith(prefix));
}

const MIGRATION_DONE_KEY = 'LiangLocal-idb-migrated';

let migrationPromise = null;

/** Run localStorage → IndexedDB migration once; safe to call from any getItem. */
export function ensureStorageMigrated(options = {}) {
  if (options.skipMigration) {
    return Promise.resolve({ copied: 0, cleared: 0 });
  }
  if (!migrationPromise) {
    migrationPromise = migrateFromLocalStorage(options).catch((e) => {
      console.warn('[indexedDbStorage] migration failed:', e);
      migrationPromise = null;
      return { copied: 0, cleared: 0 };
    });
  }
  return migrationPromise;
}

// Keys to keep in localStorage after migration (sync readers use them; we still write these to IDB)
const KEEP_IN_LOCAL_STORAGE = new Set([
  'user-profiles',
  'llm-characters',
  'LiangLocal-settings',
  'Eloquent-settings',
  'Eloquent-active-conversation',
  'eloquent-story-tracker',
  'LiangLocal-avatar-sizes',
  // ThemeProvider reads/writes localStorage synchronously; keep a copy here so migration does not strip it.
  'vite-ui-theme',
]);

/**
 * Copy known keys from localStorage into IndexedDB, then optionally remove from localStorage to free space.
 * Safe to call multiple times (skips if already run). Does not clear keys in KEEP_IN_LOCAL_STORAGE.
 * @param {{ clearLocalStorageAfterCopy?: boolean }} options
 * @returns {Promise<{ copied: number, cleared: number }>}
 */
export async function migrateFromLocalStorage(options = {}) {
  const {
    clearLocalStorageAfterCopy = true,
    skipMigration = false,
  } = options;
  if (skipMigration) {
    return { copied: 0, cleared: 0 };
  }
  try {
    if (localStorage.getItem(MIGRATION_DONE_KEY) === 'v1') {
      return { copied: 0, cleared: 0 };
    }
  } catch (_) {}

  let copied = 0;
  const pendingWrites = [];

  const skipMigrationKeys = new Set([
    'Eloquent-conversations',
    'Eloquent-conversations-index',
    'conversations',
  ]);

  const tryCopy = (key) => {
    if (skipMigrationKeys.has(key)) return;
    if (key.startsWith('Eloquent-conversation-')) return;
    if (key.startsWith('Eloquent-ban-')) return;
    try {
      const value = localStorage.getItem(key);
      if (value != null && value.length > 0) {
        pendingWrites.push(
          setItem(key, value).then(() => ({ key, value, ok: true })).catch(() => ({ key, value, ok: false }))
        );
        copied++;
      }
    } catch (_) {}
  };

  IDB_KEYS.forEach(tryCopy);

  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && IDB_KEY_PREFIXES.some(p => key.startsWith(p))) {
        tryCopy(key);
      }
    }
  } catch (_) {}

  const writeResults = await Promise.all(pendingWrites);
  const verifiedClear = [];
  for (const result of writeResults) {
    if (!result?.ok || !result.key) continue;
    if (KEEP_IN_LOCAL_STORAGE.has(result.key)) {
      if (isSettingsStorageKey(result.key) && result.value) {
        try {
          localStorage.setItem(result.key, result.value);
          if (result.key === 'Eloquent-settings') {
            localStorage.setItem('LiangLocal-settings', result.value);
          } else if (result.key === 'LiangLocal-settings') {
            localStorage.setItem('Eloquent-settings', result.value);
          }
        } catch (_) { /* quota */ }
      }
      continue;
    }
    try {
      const roundTrip = await getItem(result.key);
      if (roundTrip === result.value) {
        verifiedClear.push(result.key);
      } else {
        console.warn('[indexedDbStorage] migration verify failed, keeping localStorage copy:', result.key);
      }
    } catch (_) {
      console.warn('[indexedDbStorage] migration verify error, keeping localStorage copy:', result.key);
    }
  }

  if (clearLocalStorageAfterCopy && verifiedClear.length > 0) {
    verifiedClear.forEach(key => {
      try {
        localStorage.removeItem(key);
      } catch (_) {}
    });
  }

  try {
    localStorage.setItem(MIGRATION_DONE_KEY, 'v1');
  } catch (_) {}

  return { copied, cleared: clearLocalStorageAfterCopy ? verifiedClear.length : 0 };
}

/**
 * Rough UTF-8 byte length for quota visibility (same order of magnitude as disk use).
 * @param {string | null | undefined} s
 */
export function byteLengthUtf8(s) {
  if (s == null || s === '') return 0;
  try {
    return new TextEncoder().encode(s).length;
  } catch {
    return s.length * 2;
  }
}

/**
 * @param {string} key
 * @returns {{ kind: string, label?: string }}
 */
export function categorizeStorageKey(key) {
  if (key.startsWith('Eloquent-conversation-')) {
    return { kind: 'conversation-shard', convId: key.slice('Eloquent-conversation-'.length) };
  }
  if (key.startsWith('LiangLocal-variants-')) {
    return { kind: 'message-variants', convId: key.slice('LiangLocal-variants-'.length) };
  }
  if (key === 'Eloquent-conversations' || key === 'Eloquent-conversations-index') {
    return { kind: 'conversations-index' };
  }
  if (key === 'llm-characters') return { kind: 'characters' };
  if (key === 'user-profiles') return { kind: 'profiles' };
  if (key === 'Eloquent-settings' || key === 'LiangLocal-settings') return { kind: 'settings' };
  if (key.includes('memory') || key === 'eloquent-story-tracker') return { kind: 'memory' };
  return { kind: 'other' };
}

/**
 * One row per key in the LiangLocal IndexedDB store, sorted largest first.
 * @returns {Promise<Array<{ key: string, sizeBytes: number, kind: string, convId?: string }>>}
 */
export async function getStorageInventory() {
  const keys = await getAllKeys();
  const rows = await Promise.all(
    keys.map(async (key) => {
      const raw = await getItem(key);
      const sizeBytes = byteLengthUtf8(raw);
      const { kind, convId } = categorizeStorageKey(key);
      return { key, sizeBytes, kind, ...(convId ? { convId } : {}) };
    })
  );
  rows.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return rows;
}

/**
 * Keys that live only in localStorage (not routed through IndexedDB).
 * @returns {Array<{ key: string, sizeBytes: number }>}
 */
export function listNonIdbLocalStorageKeys() {
  const out = [];
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (!key) continue;
      if (useIdb(key)) continue;
      try {
        const v = localStorage.getItem(key);
        out.push({ key, sizeBytes: byteLengthUtf8(v) });
      } catch (_) {}
    }
  } catch (_) {}
  out.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return out;
}

/**
 * @returns {Promise<{ usage?: number, quota?: number } | null>}
 */
export async function getStorageQuotaInfo() {
  try {
    if (navigator.storage && navigator.storage.estimate) {
      const e = await navigator.storage.estimate();
      return {
        usage: typeof e.usage === 'number' ? e.usage : undefined,
        quota: typeof e.quota === 'number' ? e.quota : undefined
      };
    }
  } catch (_) {}
  return null;
}

export default {
  getItem,
  setItem,
  removeItem,
  getAllKeys,
  getKeysByPrefix,
  migrateFromLocalStorage,
  useIdb,
  byteLengthUtf8,
  categorizeStorageKey,
  getStorageInventory,
  listNonIdbLocalStorageKeys,
  getStorageQuotaInfo,
};
