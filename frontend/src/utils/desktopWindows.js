import { isTauri } from '@tauri-apps/api/core';

export async function openMiridWindow({
  label,
  url,
  title,
  width,
  height,
  x,
  y,
  fullscreen = false,
  browserFeatures = '',
}) {
  if (typeof window === 'undefined') return null;
  if (!isTauri()) return window.open(url, label, browserFeatures);

  try {
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
    const existing = await WebviewWindow.getByLabel(label);
    if (existing) await existing.close();

    return await new Promise((resolve) => {
      let settled = false;
      let timeoutId;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutId);
        resolve(value);
      };
      const child = new WebviewWindow(label, {
        url,
        title,
        width,
        height,
        ...(Number.isFinite(x) ? { x } : {}),
        ...(Number.isFinite(y) ? { y } : {}),
        fullscreen,
        resizable: true,
        focus: true,
      });
      timeoutId = setTimeout(() => finish(null), 1500);
      child.once('tauri://created', () => finish(child));
      child.once('tauri://error', () => finish(null));
    });
  } catch {
    return null;
  }
}

export async function closeCurrentMiridWindow() {
  if (typeof window === 'undefined') return false;
  if (isTauri()) {
    try {
      const { getCurrentWebviewWindow } = await import('@tauri-apps/api/webviewWindow');
      await getCurrentWebviewWindow().close();
      return true;
    } catch {}
  }
  window.close();
  return true;
}
