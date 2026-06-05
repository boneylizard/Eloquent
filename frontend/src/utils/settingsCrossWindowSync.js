/** Cross-window sync for settings + primary API model selection. */

export function isSettingsStandaloneWindow() {
  if (typeof window === 'undefined') return false;
  try {
    return new URLSearchParams(window.location.search).get('standalone') === 'settings';
  } catch {
    return false;
  }
}

export const SETTINGS_STORAGE_KEY = 'Eloquent-settings';
export const LAST_PRIMARY_API_MODEL_KEY = 'Eloquent-last-primary-api-model';
const CHANNEL_NAME = 'eloquent-app-sync';

let channel = null;

function getChannel() {
  if (typeof BroadcastChannel === 'undefined') return null;
  if (!channel) channel = new BroadcastChannel(CHANNEL_NAME);
  return channel;
}

export function saveLastPrimaryApiModel(modelId) {
  if (typeof localStorage === 'undefined' || !modelId) return;
  try {
    localStorage.setItem(LAST_PRIMARY_API_MODEL_KEY, String(modelId));
  } catch {
    /* ignore quota */
  }
}

export function readLastPrimaryApiModel() {
  if (typeof localStorage === 'undefined') return null;
  try {
    return localStorage.getItem(LAST_PRIMARY_API_MODEL_KEY);
  } catch {
    return null;
  }
}

export function broadcastSettingsPatch(patch) {
  if (!patch || typeof patch !== 'object') return;
  try {
    getChannel()?.postMessage({ type: 'settings_patch', patch, ts: Date.now() });
  } catch {
    /* ignore */
  }
}

export function broadcastSettingsReload(fullSettings) {
  try {
    getChannel()?.postMessage({ type: 'settings_reload', settings: fullSettings || null, ts: Date.now() });
  } catch {
    /* ignore */
  }
}

export function broadcastPrimaryModelState({ primaryModel, primaryIsAPI }) {
  try {
    getChannel()?.postMessage({
      type: 'primary_model',
      primaryModel: primaryModel ?? null,
      primaryIsAPI: Boolean(primaryIsAPI),
      ts: Date.now(),
    });
  } catch {
    /* ignore */
  }
}

export function requestMainWindowReload() {
  try {
    getChannel()?.postMessage({ type: 'reload_main', ts: Date.now() });
  } catch {
    /* ignore */
  }
}

/**
 * @param {{ onSettingsPatch?: (patch: object) => void, onSettingsReload?: (settings: object|null) => void, onPrimaryModel?: (payload: { primaryModel: string|null, primaryIsAPI: boolean }) => void, onReloadMain?: () => void }} handlers
 */
export function subscribeAppCrossWindowSync(handlers) {
  const ch = getChannel();

  const onStorage = (event) => {
    if (event.key !== SETTINGS_STORAGE_KEY || !event.newValue) return;
    try {
      const parsed = JSON.parse(event.newValue);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        handlers.onSettingsReload?.(parsed);
      }
    } catch {
      /* ignore corrupt cross-window payload — keep in-memory settings */
    }
  };

  const onMessage = (event) => {
    const data = event?.data || {};
    if (data.type === 'settings_patch') handlers.onSettingsPatch?.(data.patch || {});
    else if (data.type === 'settings_reload') handlers.onSettingsReload?.(data.settings ?? null);
    else if (data.type === 'primary_model') {
      handlers.onPrimaryModel?.({
        primaryModel: data.primaryModel ?? null,
        primaryIsAPI: Boolean(data.primaryIsAPI),
      });
    } else if (data.type === 'reload_main') handlers.onReloadMain?.();
  };

  window.addEventListener('storage', onStorage);
  ch?.addEventListener('message', onMessage);

  return () => {
    window.removeEventListener('storage', onStorage);
    ch?.removeEventListener('message', onMessage);
  };
}

export function buildSettingsWindowUrl(tab = 'general') {
  const t = typeof tab === 'string' && tab.trim() ? tab.trim() : 'general';
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}?standalone=settings&tab=${encodeURIComponent(t)}`;
}

export function openSettingsPopupWindow(tab = 'general') {
  const url = buildSettingsWindowUrl(tab);
  const features = 'width=1120,height=900,menubar=no,toolbar=no,location=no,status=no,resizable=yes,scrollbars=yes';
  return window.open(url, 'EloquentSettings', features);
}
