/** 
 * App boot configuration constants.
 */
export const PORT_CONFIG_TIMEOUT_MS = 4000;
export const STORAGE_HYDRATION_TIMEOUT_MS = 18000;
/** Settings popup: do not wait on main-window IDB migration / chat hydrate. */
export const SETTINGS_STANDALONE_STORAGE_TIMEOUT_MS = 6000;

/**
 * Race a promise against a timeout. Rejects with a clear Error on timeout.
 * */
export function withTimeout(promise, ms, label = 'operation') {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`)),
      ms
    );
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}
