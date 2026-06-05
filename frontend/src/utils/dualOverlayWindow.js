/**
 * Dual Overlay popup window management and cross-window state sync.
 * 
 * Allows Call Mode to open in a separate browser window while Focus Mode
 * renders in the main Eloquent tab. Both windows sync state via BroadcastChannel.
 */

export function isCallOverlayWindow() {
  if (typeof window === 'undefined') return false;
  try {
    return new URLSearchParams(window.location.search).get('standalone') === 'call';
  } catch {
    return false;
  }
}

export const DUAL_OVERLAY_CHANNEL = 'eloquent-dual-overlay-sync';

const DUAL_OVERLAY_STORAGE_KEY = 'eloquent-dual-overlay-state';

/** Synchronous localStorage read/write — works across same-origin windows instantly */
export function writeCallOverlayState(state) {
  try {
    const payload = {};
    if (state?.activeCharacter) payload.ac = state.activeCharacter;
    if (state?.characters && state.characters.length > 0) {
      payload.chs = state.characters.map(c => ({
        id: c.id, name: c.name, character_order: c.character_order,
        avatar: c.avatar, avatar_url: c.avatar_url,
      }));
    }
    if (state?.primaryApiUrl) payload.api = state.primaryApiUrl;
    if (Object.keys(payload).length > 0) {
      localStorage.setItem(DUAL_OVERLAY_STORAGE_KEY, JSON.stringify(payload));
    }
  } catch(e) { console.warn('[dual-overlay] writeCallOverlayState failed:', e); }
}

export function readCallOverlayState() {
  try {
    const raw = localStorage.getItem(DUAL_OVERLAY_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch(e) { console.warn('[dual-overlay] readCallOverlayState failed:', e); return null; }
}

export function clearCallOverlayState() {
  try { localStorage.removeItem(DUAL_OVERLAY_STORAGE_KEY); } catch(e) {}
}

/** Build a simple state object from the main app's current scope */
export function buildMainAppState(activeCharacter, characters, primaryApiUrl) {
  return { activeCharacter, characters, primaryApiUrl };
}

let overlayChannel = null;

function getOverlayChannel() {
  if (typeof BroadcastChannel === 'undefined') return null;
  if (!overlayChannel) overlayChannel = new BroadcastChannel(DUAL_OVERLAY_CHANNEL);
  return overlayChannel;
}

export function buildCallOverlayWindowUrl() {
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}?standalone=call`;
}

function getWindowFeatures() {
  const screen = window.screen;
  
  // Default to full screen of current monitor
  const width = screen.availWidth;
  const height = screen.availHeight;
  const left = screen.availLeft;
  const top = screen.availTop;

  return `width=${width},height=${height},left=${left},top=${top},menubar=no,toolbar=no,location=no,status=no,resizable=yes,fullscreen=yes`;
}

let popupWindowRef = null;

export function openCallOverlayWindow() {
  if (typeof window === 'undefined') return null;
  
  // Close existing if any
  if (popupWindowRef && !popupWindowRef.closed) {
    try {
      popupWindowRef.close();
    } catch (e) {
      console.warn('Failed to close existing overlay window:', e);
    }
  }

  const url = buildCallOverlayWindowUrl();
  const features = getWindowFeatures();
  
  try {
    const popup = window.open(url, 'EloquentCallModeOverlay', features);
    if (popup) {
      popupWindowRef = popup;
      
      // Listen for popup closure
      const checkClosed = setInterval(() => {
        if (popup.closed) {
          clearInterval(checkClosed);
          popupWindowRef = null;
          broadcastDualOverlayMessage({ type: 'call_window_closed' });
        }
      }, 1000);
    }
    return popup;
  } catch (e) {
    console.error('Failed to open overlay window (blocked?):', e);
    return null;
  }
}

export function closeCallOverlayWindow() {
  const popup = popupWindowRef;
  if (popup && !popup.closed) {
    try {
      popup.close();
      broadcastDualOverlayMessage({ type: 'dual_close' });
    } catch (e) {
      console.warn('Failed to close overlay window:', e);
    }
  }
  clearCallOverlayState();
  popupWindowRef = null;
}

export function getOverlayWindow() {
  return popupWindowRef;
}

export function isOverlayWindowOpen() {
  return popupWindowRef && !popupWindowRef.closed;
}

export function broadcastDualOverlayMessage(message) {
  if (!message || typeof message !== 'object') return;
  try {
    getOverlayChannel()?.postMessage({ ...message, ts: Date.now() });
  } catch (e) {
    console.warn('Failed to broadcast dual overlay message:', e);
  }
}

export function sendCallStateRequest() {
  broadcastDualOverlayMessage({ type: 'call_state_request' });
}

export function sendCallInput(input) {
  broadcastDualOverlayMessage({ 
    type: 'call_input', 
    input,
    ts: Date.now() 
  });
}

export function sendCallToggleMic() {
  broadcastDualOverlayMessage({ 
    type: 'call_toggle_mic',
    ts: Date.now() 
  });
}

export function sendCallStopTts() {
  broadcastDualOverlayMessage({ 
    type: 'call_stop_tts',
    ts: Date.now() 
  });
}

export function sendCallReroll() {
  broadcastDualOverlayMessage({
    type: 'call_reroll',
    ts: Date.now()
  });
}

export function sendCallCycleAvatar(delta) {
  broadcastDualOverlayMessage({
    type: 'call_cycle_avatar',
    delta,
    ts: Date.now()
  });
}

export function sendCallAiContinue() {
  broadcastDualOverlayMessage({
    type: 'call_ai_continue',
    ts: Date.now()
  });
}

export function sendCallWindowClosed() {
  broadcastDualOverlayMessage({ 
    type: 'call_window_closed',
    ts: Date.now() 
  });
}

export function subscribeDualOverlaySync(handlers) {
  const ch = getOverlayChannel();

  const onMessage = (event) => {
    const data = event?.data || {};
    
    switch (data.type) {
      case 'call_state_request':
        handlers.onCallStateRequest?.();
        break;
      case 'call_input':
        handlers.onCallInput?.(data.input, data.ts);
        break;
      case 'call_toggle_mic':
        handlers.onCallToggleMic?.();
        break;
      case 'call_stop_tts':
        handlers.onCallStopTts?.();
        break;
      case 'call_reroll':
        handlers.onCallReroll?.();
        break;
      case 'call_cycle_avatar':
        handlers.onCallCycleAvatar?.(data.delta);
        break;
      case 'call_ai_continue':
        handlers.onCallAiContinue?.();
        break;
      case 'dual_state_sync':
        handlers.onDualStateSync?.(data.state);
        break;
      case 'dual_close':
        handlers.onDualClose?.();
        break;
      case 'call_window_closed':
        handlers.onCallWindowClosed?.();
        break;
      default:
        break;
    }
  };

  ch?.addEventListener('message', onMessage);

  return () => {
    ch?.removeEventListener('message', onMessage);
  };
}

export function setOverlayWindowRef(ref) {
  popupWindowRef = ref;
}