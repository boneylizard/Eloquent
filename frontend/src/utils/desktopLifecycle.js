import { invoke, isTauri } from '@tauri-apps/api/core';
import { applyServiceEndpoints } from '../config/api.js';


function requireDesktop() {
  if (!isTauri()) {
    throw new Error('This control is available in the Mirid desktop app.');
  }
}


export async function restartMirid() {
  requireDesktop();
  await invoke('restart_app');
}


export async function shutdownMirid() {
  requireDesktop();
  await invoke('shutdown_app');
}


export async function stopTtsService() {
  requireDesktop();
  await invoke('stop_tts');
}


export async function restartTtsService() {
  requireDesktop();
  const endpoints = await invoke('restart_tts');
  if (endpoints) {
    applyServiceEndpoints(endpoints);
  }
  return endpoints;
}
