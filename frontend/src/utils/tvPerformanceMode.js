/** TV / projector / weak Android TV browsers (e.g. Browsehere): lighter compositing, larger type. */

export const TV_PERF_STORAGE_KEY = 'LiangLocal-tv-performance';
export const TV_PERF_CLASS = 'tv-performance';
export const TV_WEBVIEW_CLASS = 'tv-webview';

export function readTvPerformanceFromUrl() {
  try {
    const q = new URLSearchParams(window.location.search).get('tv');
    if (q === '1' || q === 'true' || q === 'yes') return true;
    if (window.location.hash === '#tv') return true;
  } catch (_) {
    /* ignore */
  }
  return false;
}

export function readTvPerformanceFromStorage() {
  try {
    return localStorage.getItem(TV_PERF_STORAGE_KEY) === '1';
  } catch (_) {
    return false;
  }
}

export function isTvPerformanceEnabled() {
  return readTvPerformanceFromUrl() || readTvPerformanceFromStorage();
}

export function applyTvPerformanceClass(enabled) {
  const el = document.documentElement;
  if (!el) return;
  const ua = typeof navigator !== 'undefined' ? String(navigator.userAgent || '') : '';
  const isAndroidTvUa =
    /Android/i.test(ua) &&
    (/TV/i.test(ua) || /AFT/i.test(ua) || /BRAVIA/i.test(ua) || /GoogleTV/i.test(ua));
  if (enabled) {
    el.classList.add(TV_PERF_CLASS);
    if (isAndroidTvUa) el.classList.add(TV_WEBVIEW_CLASS);
  } else {
    el.classList.remove(TV_PERF_CLASS);
    el.classList.remove(TV_WEBVIEW_CLASS);
  }
}

/** Apply class from URL (?tv=1, #tv) or localStorage. Call on startup. */
export function syncTvPerformanceFromUrlAndStorage() {
  applyTvPerformanceClass(isTvPerformanceEnabled());
  return isTvPerformanceEnabled();
}
