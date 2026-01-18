/**
 * Central API configuration
 * Reads ports from /ports.json (written by launch.py) or uses defaults
 */

let portConfig = null;
let configLoaded = false;

// Default ports
const DEFAULTS = {
  backend: 'http://localhost:8000',
  secondary: 'http://localhost:8001',
  tts: 'http://localhost:8002'
};

/**
 * Load port configuration from ports.json
 * Called once on app startup
 */
export async function loadPortConfig() {
  if (configLoaded) return portConfig;

  try {
    const response = await fetch('/ports.json');
    if (response.ok) {
      portConfig = await response.json();

      // Smart Hostname Override:
      // If we are accessing via a network IP/Hostname (not localhost),
      // force the API URLs to use that same hostname.
      // This solves issues where ports.json has a different IP (like a VPN IP 100.x)
      // than the one the user is actually using (like Wi-Fi 192.168.x).
      const currentHost = window.location.hostname;
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
      portConfig = DEFAULTS;
    }
  } catch (e) {
    console.log('📌 Could not load ports.json, using defaults');
    portConfig = DEFAULTS;
  }

  configLoaded = true;
  return portConfig;
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
  return isSingleGpuMode ? config.backend : config.secondary;
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
  return getSecondaryUrl(isSingleGpuMode);
}

// Synchronous getters for when you can't await (use after loadPortConfig has been called)
export function getConfig() {
  return portConfig || DEFAULTS;
}

