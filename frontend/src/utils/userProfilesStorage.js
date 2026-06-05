/**

 * Helpers for user-profiles load/persist safety (localStorage mirror + IndexedDB).

 */



export function parseProfilesBlob(raw) {

  if (raw == null) return null;

  let parsed = raw;

  if (typeof parsed === 'string') {

    const t = parsed.trim();

    if (!t || t === 'null' || t === '{}') return null;

    try {

      parsed = JSON.parse(t);

    } catch {

      return null;

    }

  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;

  const profiles = Array.isArray(parsed.profiles) ? parsed.profiles : [];

  const valid = profiles.filter((p) => p && p.id);

  return {

    profiles: valid,

    activeProfileId: parsed.activeProfileId ?? null,

    count: valid.length,

  };

}



/** @param {string|null|undefined} raw */

export function countProfilesInBlob(raw) {

  return parseProfilesBlob(raw)?.count ?? 0;

}



/**

 * Prefer the source with more valid profiles (recovery when IDB was clobbered).

 * @returns {{ raw: string|null, source: string, idbCount: number, lsCount: number, restoreIdb: boolean }}

 */

export function pickBestProfilesSource(idbRaw, lsRaw) {

  const idbCount = countProfilesInBlob(idbRaw);

  const lsCount = countProfilesInBlob(lsRaw);



  if (lsCount > idbCount && lsRaw != null) {

    return {

      raw: typeof lsRaw === 'string' ? lsRaw : JSON.stringify(parseProfilesBlob(lsRaw)),

      source: 'localStorage',

      idbCount,

      lsCount,

      restoreIdb: idbCount < lsCount,

    };

  }

  if (idbCount > 0 && idbRaw != null) {

    return {

      raw: typeof idbRaw === 'string' ? idbRaw : JSON.stringify(parseProfilesBlob(idbRaw)),

      source: 'indexedDB',

      idbCount,

      lsCount,

      restoreIdb: false,

    };

  }

  if (lsCount > 0 && lsRaw != null) {

    return {

      raw: typeof lsRaw === 'string' ? lsRaw : JSON.stringify(parseProfilesBlob(lsRaw)),

      source: 'localStorage',

      idbCount,

      lsCount,

      restoreIdb: idbCount < lsCount,

    };

  }

  return { raw: null, source: 'none', idbCount, lsCount, restoreIdb: false };

}



/**

 * One-time boot recovery: scan every localStorage key matching *profile* (case-insensitive).

 * @returns {Array<{ key: string, raw: string, count: number }>}

 */

export function scanLocalStorageForProfileBackups() {

  const hits = [];

  if (typeof localStorage === 'undefined') return hits;

  try {

    for (let i = 0; i < localStorage.length; i += 1) {

      const key = localStorage.key(i);

      if (!key || !/profile/i.test(key)) continue;

      try {

        const raw = localStorage.getItem(key);

        if (!raw) continue;

        const count = countProfilesInBlob(raw);

        if (count > 0) hits.push({ key, raw, count });

      } catch {

        /* skip */

      }

    }

  } catch {

    /* ignore */

  }

  hits.sort((a, b) => b.count - a.count);

  return hits;

}



/**

 * Pick the richest profiles blob among primary IDB/LS sources and optional scan hits.

 * @param {{ idbRaw?: string|null, lsRaw?: string|null, scanHits?: Array<{ key: string, raw: string }> }} sources

 */

export function pickRichestProfilesSource({ idbRaw = null, lsRaw = null, scanHits = [] }) {

  let best = pickBestProfilesSource(idbRaw, lsRaw);

  for (const hit of scanHits) {

    if (!hit?.raw) continue;

    const candidate = pickBestProfilesSource(best.raw, hit.raw);

    if (countProfilesInBlob(candidate.raw) > countProfilesInBlob(best.raw)) {

      best = {

        ...candidate,

        source: hit.key === 'user-profiles' ? candidate.source : `localStorage:${hit.key}`,

        restoreIdb: true,

      };

    }

  }

  return best;

}



/**

 * @param {string} incomingValue serialized profiles JSON

 * @returns {string|null} null = refuse write

 */

export function coerceProfilesWrite(incomingValue) {

  const incomingCount = countProfilesInBlob(incomingValue);

  let diskRaw = null;

  try {

    if (typeof localStorage !== 'undefined') {

      diskRaw = localStorage.getItem('user-profiles');

    }

  } catch {

    /* ignore */

  }

  const diskCount = countProfilesInBlob(diskRaw);



  if (incomingCount === 0) {

    if (diskCount > 0) return null;

    return incomingValue;

  }

  if (diskCount > incomingCount) return null;

  return incomingValue;

}



/**

 * @param {{ profiles?: unknown[] }} memoryState

 * @param {'pending'|'loaded'|'empty'|'error'} profilesLoadStatus

 * @param {{ userMutated?: boolean, loadedBaselineCount?: number }} [opts]

 */

export function shouldPersistUserProfiles(memoryState, profilesLoadStatus, opts = {}) {

  const { userMutated = false, loadedBaselineCount = 0 } = opts;



  if (profilesLoadStatus === 'pending' || profilesLoadStatus === 'error') return false;



  const count = Array.isArray(memoryState?.profiles) ? memoryState.profiles.length : 0;

  if (count === 0) return false;



  if (profilesLoadStatus === 'empty' && !userMutated) return false;



  let diskRaw = null;

  try {

    if (typeof localStorage !== 'undefined') {

      diskRaw = localStorage.getItem('user-profiles');

    }

  } catch {

    /* ignore */

  }

  const diskCount = countProfilesInBlob(diskRaw);



  if (!userMutated && diskCount > count) {

    console.warn(

      '[MemoryContext] Refusing persist: localStorage has',

      diskCount,

      'profile(s), memory has',

      count,

    );

    return false;

  }

  if (!userMutated && loadedBaselineCount > 0 && count < loadedBaselineCount) {

    console.warn(

      '[MemoryContext] Refusing persist: would shrink below loaded baseline',

      loadedBaselineCount,

      '→',

      count,

    );

    return false;

  }

  if (profilesLoadStatus === 'empty' && diskCount > 0 && !userMutated) return false;



  return true;

}


