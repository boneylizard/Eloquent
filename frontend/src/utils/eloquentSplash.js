import { SETTINGS_STORAGE_KEY } from './settingsCrossWindowSync';

/** @typedef {'off' | 'fast' | 'normal' | 'long'} SplashScreenDuration */

export const SPLASH_DURATION_OPTIONS = [
  { value: 'off', label: 'Off', description: 'Hide as soon as the app is ready' },
  { value: 'fast', label: 'Fast', description: 'About 0.8 seconds on screen' },
  { value: 'normal', label: 'Normal', description: 'About 3 seconds on screen (default)' },
  { value: 'long', label: 'Long', description: 'About 5 seconds on screen' },
];

const SPLASH_MIN_MS = {
  off: 0,
  fast: 800,
  normal: 3000,
  long: 5000,
};

const SPLASH_MIN_MS_REDUCED_MOTION = {
  off: 0,
  fast: 400,
  normal: 1000,
  long: 1500,
};

export const SPLASH_SCREEN_DURATION_DEFAULT = 'normal';

function prefersReducedMotion() {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

/**
 * Read splash duration from mirrored localStorage settings (available before React hydrates).
 * @returns {SplashScreenDuration}
 */
export function readSplashScreenDurationSetting() {
  if (typeof localStorage === 'undefined') return SPLASH_SCREEN_DURATION_DEFAULT;
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return SPLASH_SCREEN_DURATION_DEFAULT;
    const parsed = JSON.parse(raw);
    const value = parsed?.splashScreenDuration;
    if (value === 'off' || value === 'fast' || value === 'normal' || value === 'long') {
      return value;
    }
  } catch {
    /* ignore */
  }
  return SPLASH_SCREEN_DURATION_DEFAULT;
}

/**
 * Minimum time the splash should stay visible after first paint (ms).
 * @param {SplashScreenDuration} [duration]
 */
export function getSplashMinDisplayMs(duration = readSplashScreenDurationSetting()) {
  const table = prefersReducedMotion() ? SPLASH_MIN_MS_REDUCED_MOTION : SPLASH_MIN_MS;
  return table[duration] ?? table[SPLASH_SCREEN_DURATION_DEFAULT];
}

export function getEloquentSplashStartMs() {
  if (typeof window !== 'undefined' && typeof window.__eloquentSplashStartMs === 'number') {
    return window.__eloquentSplashStartMs;
  }
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return Date.now();
}

function waitForReactPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(resolve);
    });
  });
}

function waitMs(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

/**
 * Dismiss splash after React has painted and the configured minimum display time has elapsed.
 * @param {() => void} dismiss
 */
export async function scheduleEloquentSplashDismiss(dismiss) {
  const minMs = getSplashMinDisplayMs();
  const startMs = getEloquentSplashStartMs();

  const minTimePromise =
    minMs <= 0
      ? Promise.resolve()
      : waitMs(Math.max(0, minMs - (performance.now() - startMs)));

  await Promise.all([waitForReactPaint(), minTimePromise]);
  dismiss();
}
