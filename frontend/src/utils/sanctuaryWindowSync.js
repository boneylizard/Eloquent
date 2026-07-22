/**
 * Sanctuary cross-window sync via BroadcastChannel.
 *
 * Enables the CognitiveGlass to open in a separate standalone window
 * with full two-way bidirectional sync between the main chat window
 * and the CognitiveGlass popup.
 *
 * Channel: sanctuary-sync
 * Messages:
 *   { type: "analysis", data: {...} }     — Step 1 result
 *   { type: "somatic",  data: {...} }     — Step 2 somatic payload
 *   { type: "cipher",   data: {...} }     — Cipher block (phase: preliminary/final)
 *   { type: "text",     data: {...} }     — Text delta
 *   { type: "done",     data: {...} }     — Turn complete + final state
 *   { type: "shadow_state", data: {...} } — Full shadow state snapshot
 *   { type: "reset_baseline", data: {...} } — Reset baseline request (main window action)
 *   { type: "baseline_reset", data: {...} } — Baseline reset complete -> main window
 *   { type: "end_turn" }                 — Turn ended
 *   { type: "start_turn" }               — Turn started
 *   { type: "hijack", data: {...} }      — Interface hijack state
 */

import { openMiridWindow } from './desktopWindows.js';

const CHANNEL_NAME = 'sanctuary-sync';

let mainChannel = null;
let glassChannel = null;
let channelInstances = {};

function getChannel(channelId, direction = 'main->glass') {
  if (typeof BroadcastChannel === 'undefined') return null;
  
  const key = `${channelId}-${direction}`;
  if (!channelInstances[key]) {
    channelInstances[key] = new BroadcastChannel(CHANNEL_NAME);
    channelInstances[key].addEventListener('messageerror', (e) => {
      console.warn(`[SanctuarySync] ${channelId} ${direction} channel messageerror:`, e);
    });
    channelInstances[key].addEventListener('message', (e) => {
      console.log(`[SanctuarySync] ${channelId} ${direction} received message:`, e.data);
    });
  }
  return channelInstances[key];
}

function getMainChannel(channelId = 'main') {
  return getChannel(channelId, 'main->glass');
}

function getGlassChannel(channelId = 'glass') {
  return getChannel(channelId, 'glass->main');
}

/**
 * Open the CognitiveGlass in a standalone popup window.
 * Uses the same origin so IndexedDB/localStorage are shared.
 */
export async function openCognitiveGlassWindow() {
  if (typeof window === 'undefined') return null;
  const url = `${window.location.origin}${window.location.pathname}?standalone=cognitive-glass`;
  return openMiridWindow({
    label: 'cognitive-glass',
    url,
    title: 'Mirid Cognitive Glass',
    width: 420,
    height: 800,
    x: 50,
    y: 50,
    browserFeatures: 'width=420,height=800,left=50,top=50',
  });
}

export async function openCognitiveGlassOnSecondScreen() {
  if (typeof window === 'undefined' || !window.screen || window.screenLeft === undefined) return openCognitiveGlassWindow();
  // Try to place on second screen if available
  try {
    const left = window.screenLeft < 0 ? Math.abs(window.screenLeft) + 200 : 
                 window.screenLeft > 1920 ? 200 : 2200;
    const url = `${window.location.origin}${window.location.pathname}?standalone=cognitive-glass`;
    return openMiridWindow({
      label: 'cognitive-glass',
      url,
      title: 'Mirid Cognitive Glass',
      width: 420,
      height: 800,
      x: left,
      y: 50,
      browserFeatures: `width=420,height=800,left=${left},top=50`,
    });
  } catch {
    return openCognitiveGlassWindow();
  }
}

export function isCognitiveGlassStandalone() {
  if (typeof window === 'undefined') return false;
  try {
    return new URLSearchParams(window.location.search).get('standalone') === 'cognitive-glass';
  } catch {
    return false;
  }
}

/** Broadcast message from main window to popup. */
export function broadcastToGlass(message, channelId = 'main') {
  try {
    getMainChannel(channelId)?.postMessage({ ...message, ts: Date.now(), direction: 'main->glass' });
  } catch (e) {
    // Channel might be closed
  }
}

/** Broadcast message from popup back to main window. */
export function broadcastToMain(message, channelId = 'glass') {
  try {
    getGlassChannel(channelId)?.postMessage({ ...message, ts: Date.now(), direction: 'glass->main' });
  } catch (e) {
    // Channel might be closed
  }
}

/** Subscribe to all sanctuary sync messages. Returns unsubscribe function. */
export function subscribeSanctuarySync(onMessage, channelId = 'main') {
  if (typeof BroadcastChannel === 'undefined') return () => {};
  const channel = getChannel(channelId, 'main->glass');
  if (!channel) return () => {};
  const handler = (event) => {
    try {
      onMessage(event.data || event);
    } catch (e) {
      console.warn('[SanctuarySync] handler error:', e);
    }
  };
  channel.addEventListener('message', handler);
  return () => {
    channel.removeEventListener('message', handler);
  };
}
