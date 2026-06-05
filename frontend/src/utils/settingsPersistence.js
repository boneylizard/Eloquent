import * as indexedDbStorage from './indexedDbStorage';
import { SETTINGS_STORAGE_KEY } from './settingsCrossWindowSync';

export const SETTINGS_STORAGE_KEYS = [
  SETTINGS_STORAGE_KEY,
  'LiangLocal-settings',
];

/** Keys that must never be cleared or overwritten by chat recovery / migration cleanup. */
export const SETTINGS_PROTECTED_STORAGE_KEYS = new Set([
  ...SETTINGS_STORAGE_KEYS,
  'LiangLocal-avatar-sizes',
  'user-profiles',
  'llm-characters',
]);

export function isSettingsStorageKey(key) {
  return SETTINGS_STORAGE_KEYS.includes(key);
}

/**
 * @param {string|null|undefined} raw
 * @returns {Record<string, unknown>|null}
 */
export function parseSettingsJson(raw) {
  if (raw == null || typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed || trimmed === '{}' || trimmed === 'null') return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    return parsed;
  } catch {
    return null;
  }
}

/** @param {Record<string, unknown>|null|undefined} obj */
export function countMeaningfulSettingsKeys(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return 0;
  return Object.keys(obj).filter((k) => obj[k] !== undefined).length;
}

/** True when parsed settings have no user-facing keys (would wipe custom config if applied alone). */
export function isTrivialSettingsObject(obj) {
  return countMeaningfulSettingsKeys(obj) === 0;
}

/**
 * Shallow merge: patch keys overlay base. Refuses empty patch when base already has data.
 * @param {Record<string, unknown>} base
 * @param {Record<string, unknown>|null|undefined} patch
 * @param {{ allowEmptyPatch?: boolean }} [opts]
 */
export function mergeSettingsObjects(base, patch, opts = {}) {
  const safeBase = base && typeof base === 'object' && !Array.isArray(base) ? base : {};
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) return { ...safeBase };
  const patchKeys = countMeaningfulSettingsKeys(patch);
  const baseKeys = countMeaningfulSettingsKeys(safeBase);
  if (!opts.allowEmptyPatch && patchKeys === 0 && baseKeys > 0) return { ...safeBase };
  return { ...safeBase, ...patch };
}

/**
 * Synchronous read: localStorage mirror first (authoritative for settings UI), then legacy key.
 * @returns {Record<string, unknown>|null}
 */
export function readSettingsFromLocalStorageSync() {
  if (typeof localStorage === 'undefined') return null;
  for (const key of SETTINGS_STORAGE_KEYS) {
    try {
      const parsed = parseSettingsJson(localStorage.getItem(key));
      if (parsed) return parsed;
    } catch {
      /* try next key */
    }
  }
  return null;
}

/**
 * @param {{ preferLocalStorage?: boolean, skipMigration?: boolean }} [idbOpts]
 * @returns {Promise<Record<string, unknown>|null>}
 */
export async function readSettingsFromStorage(idbOpts = {}) {
  const fromLs = readSettingsFromLocalStorageSync();
  if (fromLs) return fromLs;

  for (const key of SETTINGS_STORAGE_KEYS) {
    try {
      const raw = await indexedDbStorage.getItem(key, {
        preferLocalStorage: true,
        ...idbOpts,
      });
      const parsed = parseSettingsJson(raw);
      if (parsed) return parsed;
    } catch {
      /* try next */
    }
  }
  return null;
}

/**
 * Whether hydrated/parsed settings should be applied to React state.
 * @param {Record<string, unknown>|null|undefined} parsed
 * @param {Record<string, unknown>} currentInMemory
 */
export function shouldApplyHydratedSettings(parsed, currentInMemory) {
  if (!parsed) return false;
  if (!isTrivialSettingsObject(parsed)) return true;
  return countMeaningfulSettingsKeys(currentInMemory) === 0;
}

/**
 * Merge avatar size sidecar into a settings object (hydration helper).
 * @param {Record<string, unknown>} target
 * @param {string|null|undefined} avatarSizesStr
 */
export function applyAvatarSizesToSettings(target, avatarSizesStr) {
  if (!avatarSizesStr) return target;
  try {
    const a = JSON.parse(avatarSizesStr);
    if (typeof a.userAvatarSize === 'number') target.userAvatarSize = a.userAvatarSize;
    if (typeof a.characterAvatarSize === 'number') target.characterAvatarSize = a.characterAvatarSize;
  } catch {
    /* ignore */
  }
  return target;
}

/**
 * Safe persist: never replace a rich on-disk blob with empty/partial JSON.
 * @param {Record<string, unknown>} nextFullState full merged settings object to save
 * @returns {Promise<boolean>} false if write was skipped
 */
export async function persistSettingsBlob(nextFullState) {
  if (!nextFullState || typeof nextFullState !== 'object') return false;

  let diskParsed = readSettingsFromLocalStorageSync();
  if (!diskParsed) {
    diskParsed = await readSettingsFromStorage({ preferLocalStorage: true, skipMigration: true });
  }

  const merged = mergeSettingsObjects(diskParsed || {}, nextFullState);
  if (isTrivialSettingsObject(merged)) {
    if (diskParsed && !isTrivialSettingsObject(diskParsed)) {
      console.warn('[settingsPersistence] Refusing to persist empty settings over existing data');
      return false;
    }
    if (countMeaningfulSettingsKeys(nextFullState) === 0) {
      console.warn('[settingsPersistence] Refusing to persist empty settings blob');
      return false;
    }
  }

  const str = JSON.stringify(merged);
  await safeSetSettingsRaw(str);
  return true;
}

/**
 * @param {string} serialized full JSON string
 * @returns {Promise<boolean>}
 */
export async function safeSetSettingsRaw(serialized) {
  const incoming = parseSettingsJson(serialized);
  if (!incoming) {
    const existing = readSettingsFromLocalStorageSync()
      || (await readSettingsFromStorage({ preferLocalStorage: true, skipMigration: true }));
    if (existing && !isTrivialSettingsObject(existing)) {
      console.warn('[settingsPersistence] Refusing empty settings write');
      return false;
    }
    if (!existing) return false;
  }

  let toWrite = serialized;
  if (incoming) {
    const existing = readSettingsFromLocalStorageSync()
      || (await readSettingsFromStorage({ preferLocalStorage: true, skipMigration: true }));
    if (existing && !isTrivialSettingsObject(existing)) {
      const merged = mergeSettingsObjects(existing, incoming);
      toWrite = JSON.stringify(merged);
    } else if (isTrivialSettingsObject(incoming)) {
      if (existing && !isTrivialSettingsObject(existing)) {
        console.warn('[settingsPersistence] Refusing trivial settings overwrite');
        return false;
      }
      if (!existing) return false;
    }
  }

  await indexedDbStorage.setItem(SETTINGS_STORAGE_KEY, toWrite);
  try {
    localStorage.setItem('LiangLocal-settings', toWrite);
  } catch {
    /* quota — primary key mirrored by indexedDbStorage.setItem */
  }
  return true;
}

/**
 * Merge incoming serialized settings with any on-disk copy before IndexedDB write.
 * Used by indexedDbStorage.setItem for settings keys.
 * @param {string} key
 * @param {string} value
 * @returns {string|null} null = refuse write
 */
export function coalesceSettingsWrite(key, value) {
  if (!isSettingsStorageKey(key)) return value;
  const incoming = parseSettingsJson(value);
  if (!incoming) {
    const existing = readSettingsFromLocalStorageSync();
    if (existing && !isTrivialSettingsObject(existing)) return null;
    return value;
  }
  if (isTrivialSettingsObject(incoming)) {
    const existing = readSettingsFromLocalStorageSync();
    if (existing && !isTrivialSettingsObject(existing)) return null;
    if (!existing) return null;
    return value;
  }
  const existing = readSettingsFromLocalStorageSync();
  if (existing && !isTrivialSettingsObject(existing)) {
    return JSON.stringify(mergeSettingsObjects(existing, incoming));
  }
  return value;
}
