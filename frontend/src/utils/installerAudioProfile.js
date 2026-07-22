const SUPPORTED_STT_ENGINES = new Set([
  'whisper',
  'whisper3',
  'parakeet',
  'parakeet-v3',
  'parakeet-zh',
  'nemotron',
  'moonshine',
  'parakeet-cpp',
  'nanogpt',
]);

const SUPPORTED_TTS_ENGINES = new Set([
  'kokoro',
  'chatterbox',
  'chatterbox_turbo',
  'chatterbox_nano',
  'voxcpm',
  'voxcpm-gguf',
  'nanogpt-Qwen-3-TTS-1.7B',
]);

export const AUDIO_STARTUP_DEFAULTS_VERSION = 1;

export function getAudioStartupDefaultsMigration(settings) {
  if (
    settings
    && typeof settings === 'object'
    && Number(settings.audioStartupDefaultsVersion) >= AUDIO_STARTUP_DEFAULTS_VERSION
  ) {
    return null;
  }
  return {
    ttsEnabled: true,
    sttEnabled: true,
    audioStartupDefaultsVersion: AUDIO_STARTUP_DEFAULTS_VERSION,
  };
}

export function normaliseInstallerAudioProfile(profile) {
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)) return null;

  const settings = {};
  if (typeof profile.ttsEnabled === 'boolean') settings.ttsEnabled = profile.ttsEnabled;
  if (typeof profile.sttEnabled === 'boolean') settings.sttEnabled = profile.sttEnabled;
  if (SUPPORTED_TTS_ENGINES.has(profile.ttsEngine)) settings.ttsEngine = profile.ttsEngine;
  if (SUPPORTED_STT_ENGINES.has(profile.sttEngine)) settings.sttEngine = profile.sttEngine;
  if (typeof profile.nanogptSttModel === 'string' && profile.nanogptSttModel.trim()) {
    settings.nanogptSttModel = profile.nanogptSttModel.trim();
  }
  if (typeof profile.nanoGptApiKey === 'string' && profile.nanoGptApiKey.trim()) {
    settings.nanoGptApiKey = profile.nanoGptApiKey.trim();
  }

  return Object.keys(settings).length ? settings : null;
}

function isTauriDesktop() {
  return typeof window !== 'undefined' && Boolean(window.__TAURI_INTERNALS__);
}

export async function readInstallerAudioProfile() {
  if (!isTauriDesktop()) return null;
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    const profile = await invoke('read_installer_audio_profile');
    return normaliseInstallerAudioProfile(profile);
  } catch (error) {
    console.warn('[InstallerAudio] Could not read installer audio choices.', error);
    return null;
  }
}

export async function clearInstallerAudioProfile() {
  if (!isTauriDesktop()) return;
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('clear_installer_audio_profile');
  } catch (error) {
    console.warn('[InstallerAudio] Could not clear the applied installer audio choices.', error);
  }
}
