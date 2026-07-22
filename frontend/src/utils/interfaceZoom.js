export const INTERFACE_ZOOM_STORAGE_KEY = 'mirid-interface-zoom';
export const INTERFACE_ZOOM_EVENT = 'mirid-interface-zoom-changed';
export const INTERFACE_ZOOM_DEFAULT = 1.1;
export const INTERFACE_ZOOM_MIN = 0.75;
export const INTERFACE_ZOOM_MAX = 2;
export const INTERFACE_ZOOM_STEP = 0.1;

export function normaliseInterfaceZoom(value) {
  if (value == null || value === '') return INTERFACE_ZOOM_DEFAULT;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return INTERFACE_ZOOM_DEFAULT;
  const clamped = Math.min(INTERFACE_ZOOM_MAX, Math.max(INTERFACE_ZOOM_MIN, parsed));
  return Math.round(clamped * 100) / 100;
}

export function readInterfaceZoom() {
  if (typeof window === 'undefined') return INTERFACE_ZOOM_DEFAULT;
  try {
    return normaliseInterfaceZoom(window.localStorage.getItem(INTERFACE_ZOOM_STORAGE_KEY));
  } catch {
    return INTERFACE_ZOOM_DEFAULT;
  }
}

async function applyZoomToView(scale) {
  if (typeof window === 'undefined') return;
  if (window.__TAURI_INTERNALS__) {
    const { getCurrentWebview } = await import('@tauri-apps/api/webview');
    await getCurrentWebview().setZoom(scale);
    return;
  }
  document.documentElement.style.zoom = String(scale);
}

export async function setInterfaceZoom(value, { persist = true } = {}) {
  const scale = normaliseInterfaceZoom(value);
  if (typeof window !== 'undefined' && persist) {
    try {
      window.localStorage.setItem(INTERFACE_ZOOM_STORAGE_KEY, String(scale));
    } catch {
    }
  }

  try {
    await applyZoomToView(scale);
  } catch (error) {
    console.warn('[interface zoom] could not apply WebView zoom', error);
  }

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(INTERFACE_ZOOM_EVENT, { detail: { scale } }));
  }
  return scale;
}

export function installInterfaceZoom() {
  if (typeof window === 'undefined') return () => {};
  void setInterfaceZoom(readInterfaceZoom(), { persist: false });

  const handleKeyDown = (event) => {
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
    const key = event.key;
    let next = null;
    if (key === '+' || key === '=' || key === 'Add') {
      next = readInterfaceZoom() + INTERFACE_ZOOM_STEP;
    } else if (key === '-' || key === '_' || key === 'Subtract') {
      next = readInterfaceZoom() - INTERFACE_ZOOM_STEP;
    } else if (key === '0') {
      next = INTERFACE_ZOOM_DEFAULT;
    }
    if (next == null) return;
    event.preventDefault();
    void setInterfaceZoom(next);
  };

  window.addEventListener('keydown', handleKeyDown, { capture: true });
  return () => window.removeEventListener('keydown', handleKeyDown, { capture: true });
}
