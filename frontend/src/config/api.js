/**
 * Central API configuration
 * Reads service endpoints from the desktop host or ports.json.
 */

let portConfig = null;
let configLoaded = false;
let configLoadPromise = null;
let publishedEndpoints = null;

// Default ports
const DEFAULTS = {
  backend: 'http://localhost:8000',
  secondary: 'http://localhost:8000',
  tts: 'http://localhost:8002'
};

export const SERVICE_ENDPOINTS_CHANGED_EVENT = 'service-endpoints-changed';
const DESKTOP_ENDPOINTS_DIAGNOSTIC_KEY = '__MIRID_SERVICE_ENDPOINTS__';
const DESKTOP_RETRY_INITIAL_MS = 50;
const DESKTOP_RETRY_MAX_MS = 1000;

function isHttpEndpoint(value) {
  if (typeof value !== 'string' || !value.trim()) return false;
  try {
    const endpoint = new URL(value);
    return endpoint.protocol === 'http:' || endpoint.protocol === 'https:';
  } catch {
    return false;
  }
}

export function normalisePortConfig(candidate) {
  if (!candidate || typeof candidate !== 'object') return null;
  const backend = isHttpEndpoint(candidate.backend) ? candidate.backend : null;
  const tts = isHttpEndpoint(candidate.tts) ? candidate.tts : null;
  if (!backend || !tts) return null;
  return {
    backend,
    secondary: isHttpEndpoint(candidate.secondary) ? candidate.secondary : backend,
    tts,
    ...(Number.isInteger(candidate.backendPort) ? { backendPort: candidate.backendPort } : {}),
    ...(Number.isInteger(candidate.ttsPort) ? { ttsPort: candidate.ttsPort } : {}),
  };
}

function publishAppliedEndpoints(config) {
  publishedEndpoints = Object.freeze({ ...config });
  if (typeof window === 'undefined') return;

  const existing = Object.getOwnPropertyDescriptor(window, DESKTOP_ENDPOINTS_DIAGNOSTIC_KEY);
  if (existing) return;

  Object.defineProperty(window, DESKTOP_ENDPOINTS_DIAGNOSTIC_KEY, {
    get: () => publishedEndpoints,
    set: undefined,
    configurable: false,
    enumerable: false,
  });
}

export function applyServiceEndpoints(candidate, { emitEvent = true } = {}) {
  const config = normalisePortConfig(candidate);
  if (!config) return null;

  portConfig = config;
  configLoaded = true;
  publishAppliedEndpoints(config);

  if (
    emitEvent &&
    typeof window !== 'undefined' &&
    typeof window.dispatchEvent === 'function' &&
    typeof CustomEvent === 'function'
  ) {
    window.dispatchEvent(new CustomEvent(SERVICE_ENDPOINTS_CHANGED_EVENT, {
      detail: publishedEndpoints,
    }));
  }

  return portConfig;
}

function desktopRetryDelay(attempt) {
  const exponent = Math.min(Math.max(attempt - 1, 0), 5);
  return Math.min(DESKTOP_RETRY_INITIAL_MS * (2 ** exponent), DESKTOP_RETRY_MAX_MS);
}

async function loadDesktopPortConfig() {
  let attempt = 0;

  while (true) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const desktopConfig = normalisePortConfig(
        await invoke('get_service_endpoints')
      );
      if (!desktopConfig) {
        throw new Error('The desktop host returned an invalid service configuration.');
      }
      return desktopConfig;
    } catch (error) {
      attempt += 1;
      const retryInMs = desktopRetryDelay(attempt);
      if (attempt === 1 || attempt % 10 === 0) {
        console.warn(
          `Mirid could not read its desktop service endpoints; retrying in ${retryInMs}ms.`,
          error
        );
      }
      await new Promise((resolve) => setTimeout(resolve, retryInMs));
    }
  }
}

/**
 * Load the host's service endpoints. Desktop lookup keeps retrying instead of
 * silently caching fixed ports after a transient IPC failure.
 */
async function loadPortConfigUncached() {
  const inTauri =
    typeof window !== 'undefined' && !!window.__TAURI_INTERNALS__;

  if (inTauri) {
    const desktopConfig = await loadDesktopPortConfig();
    console.log('Loaded desktop service endpoints:', desktopConfig);
    return applyServiceEndpoints(desktopConfig);
  }

  try {
    const response = await fetchWithTimeout('/ports.json', {}, 4000);
    if (response.ok) {
      portConfig = normalisePortConfig(await response.json()) || { ...DEFAULTS };

      // Smart Hostname Override:
      // If we are accessing via a network IP/Hostname (not localhost),
      // force the API URLs to use that same hostname.
      // This solves issues where ports.json has a different IP (like a VPN IP 100.x)
      // than the one the user is actually using (like Wi-Fi 192.168.x).
      const currentHost = window.location.hostname;
      // Only rewrite API host when the UI is served as a plain web page from a
      // different host (e.g. opened on a phone via the machine's LAN IP). Inside
      // the Tauri desktop webview the backend is always local, and the webview
      // origin (asset.localhost / tauri://localhost) must NOT be used as the API
      // host — doing so sends requests to a host the backend doesn't answer on.
      if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
        const replaceHost = (url) => {
          try {
            const u = new URL(url);
            u.hostname = currentHost;
            return u.toString().replace(/\/$/, "");
          } catch (e) { return url; }
        };

        if (portConfig.backend) portConfig.backend = replaceHost(portConfig.backend);
        if (portConfig.secondary) portConfig.secondary = replaceHost(portConfig.secondary);
        if (portConfig.tts) portConfig.tts = replaceHost(portConfig.tts);
        console.log('🌍 Adapted API URLs to current host:', currentHost);
      }

      console.log('📌 Loaded port config:', portConfig);
    } else {
      console.log('📌 No ports.json found, using defaults');
      portConfig = { ...DEFAULTS };
    }
  } catch (e) {
    console.log('Could not load ports.json, using defaults');
    portConfig = { ...DEFAULTS };
  }

  return applyServiceEndpoints(portConfig);
}

export async function loadPortConfig() {
  if (configLoaded) return portConfig;
  if (configLoadPromise) return configLoadPromise;

  configLoadPromise = loadPortConfigUncached();
  try {
    return await configLoadPromise;
  } catch (error) {
    configLoadPromise = null;
    throw error;
  }
}

/**
 * Get the backend API URL
 * @param {boolean} isSingleGpuMode - If true, always returns primary backend
 */
export function getBackendUrl(isSingleGpuMode = false) {
  const config = portConfig || DEFAULTS;
  return config.backend;
}

/**
 * Get the secondary backend URL (for dual-GPU mode)
 * @param {boolean} isSingleGpuMode - If true, returns primary backend instead
 */
export function getSecondaryUrl(isSingleGpuMode = false) {
  const config = portConfig || DEFAULTS;
  return config.backend;
}

/**
 * Get the TTS service URL
 */
export function getTtsUrl() {
  const config = portConfig || DEFAULTS;
  return config.tts;
}

/**
 * Get the memory API URL (uses secondary in dual-GPU mode)
 * @param {boolean} isSingleGpuMode 
 */
export function getMemoryUrl(isSingleGpuMode = false) {
  return getBackendUrl(isSingleGpuMode);
}

// Synchronous getters for when you can't await (use after loadPortConfig has been called)
export function getConfig() {
  return portConfig || DEFAULTS;
}

/**
 * Read error text without assigning to err.message (some DOM/Abort errors use getter-only message).
 */
export function safeErrorMessage(err, fallback = 'Request failed') {
  if (err == null) return fallback;
  if (typeof err === 'string') return err || fallback;
  try {
    const msg = err.message;
    if (typeof msg === 'string' && msg.trim()) return msg;
  } catch {
    /* ignore getter-only message */
  }
  try {
    if (typeof err.detail === 'string' && err.detail.trim()) return err.detail;
    if (typeof err.install_hint === 'string' && err.install_hint.trim()) return err.install_hint;
  } catch {
    /* ignore */
  }
  try {
    const s = String(err);
    if (s && s !== '[object Object]') return s;
  } catch {
    /* ignore */
  }
  return fallback;
}

/** Hint when memory API (often secondary port) is unreachable after reinstall or dual-GPU setup. */
export function memoryApiUnreachableHint({
  isSingleGpuMode,
  memoryUrl,
  primaryUrl,
  settingsWindow = false,
} = {}) {
  const mem = memoryUrl || 'memory API';
  const primary = primaryUrl || 'the primary backend';
  if (settingsWindow || isSingleGpuMode) {
    return `Could not reach the memory API at ${mem} (using primary backend). Is the backend running on that host? Open ${primary} in the browser to verify.`;
  }
  return `Could not reach the memory API at ${mem}. In dual-GPU mode memory usually runs on the secondary port — start that process, or enable Single GPU mode in Settings (memory is also on ${primary}).`;
}

/**
 * User-friendly message for fetch failures (timeouts vs user cancel vs network).
 */
export function formatFetchError(err, { timeoutMs, hint } = {}) {
  if (!err) return hint || 'Request failed';
  const raw = safeErrorMessage(err);
  const isAbort =
    err.name === 'AbortError' ||
    err.name === 'TimeoutError' ||
    /timed out|aborted without reason/i.test(raw);
  if (isAbort) {
    const secs = timeoutMs ? Math.round(timeoutMs / 1000) : null;
    const base = secs ? `Request timed out after ${secs}s.` : 'Request was cancelled.';
    return hint ? `${base} ${hint}` : base;
  }
  if (/failed to fetch|network error|load failed/i.test(raw) && hint) {
    return `${raw} ${hint}`;
  }
  return raw || hint || 'Request failed';
}

/**
 * Fetch with AbortController timeout (avoids hanging spinners on mobile / wrong host).
 * Merges with caller-provided signal when present.
 */
export async function fetchWithTimeout(url, init = {}, timeoutMs = 25000) {
  const ctrl = new AbortController();
  const outer = init?.signal;
  if (outer) {
    if (outer.aborted) {
      throw new DOMException('The operation was aborted.', 'AbortError');
    }
    outer.addEventListener('abort', () => ctrl.abort(outer.reason), { once: true });
  }
  const id = setTimeout(() => {
    try {
      ctrl.abort(new DOMException(`Request timed out after ${timeoutMs}ms`, 'TimeoutError'));
    } catch {
      ctrl.abort();
    }
  }, timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } catch (err) {
    const wrapped = new Error(formatFetchError(err, { timeoutMs }));
    if (err && typeof err === 'object') {
      wrapped.cause = err;
    }
    throw wrapped;
  } finally {
    clearTimeout(id);
  }
}

