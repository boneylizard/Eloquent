import { isTauri } from '@tauri-apps/api/core';
import { relaunch } from '@tauri-apps/plugin-process';
import { check } from '@tauri-apps/plugin-updater';

const CHECK_TIMEOUT_MS = 30_000;
const DOWNLOAD_TIMEOUT_MS = 30 * 60_000;

export function isAppUpdaterAvailable() {
  return typeof window !== 'undefined' && isTauri();
}

export async function checkForAppUpdate() {
  if (!isAppUpdaterAvailable()) return null;
  return check({ timeout: CHECK_TIMEOUT_MS });
}

export function createUpdateProgress() {
  return {
    phase: 'waiting',
    downloadedBytes: 0,
    totalBytes: null,
    percent: null,
    bytesPerSecond: 0,
    startedAt: null,
  };
}

export function reduceUpdateProgress(current, event, now = Date.now()) {
  if (!event || typeof event !== 'object') return current;

  if (event.event === 'Started') {
    const totalBytes = Number(event.data?.contentLength);
    return {
      ...current,
      phase: 'downloading',
      downloadedBytes: 0,
      totalBytes: Number.isFinite(totalBytes) && totalBytes > 0 ? totalBytes : null,
      percent: 0,
      bytesPerSecond: 0,
      startedAt: now,
    };
  }

  if (event.event === 'Progress') {
    const chunkLength = Number(event.data?.chunkLength);
    const downloadedBytes = current.downloadedBytes
      + (Number.isFinite(chunkLength) && chunkLength > 0 ? chunkLength : 0);
    const elapsedSeconds = current.startedAt
      ? Math.max((now - current.startedAt) / 1000, 0.001)
      : 0;
    const percent = current.totalBytes
      ? Math.min(100, Math.round((downloadedBytes / current.totalBytes) * 100))
      : null;
    return {
      ...current,
      phase: 'downloading',
      downloadedBytes,
      percent,
      bytesPerSecond: elapsedSeconds ? downloadedBytes / elapsedSeconds : 0,
    };
  }

  if (event.event === 'Finished') {
    return {
      ...current,
      phase: 'installing',
      percent: 100,
    };
  }

  return current;
}

export function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${Math.round(bytes)} B`;
}

export function formatUpdateProgress(progress) {
  if (!progress) return '';
  if (progress.phase === 'installing') return 'Download complete. Installing update…';
  if (progress.phase !== 'downloading') return '';

  const parts = [];
  if (progress.percent !== null) parts.push(`${progress.percent}%`);
  if (progress.totalBytes) {
    parts.push(`${formatBytes(progress.downloadedBytes)} of ${formatBytes(progress.totalBytes)}`);
  } else if (progress.downloadedBytes) {
    parts.push(formatBytes(progress.downloadedBytes));
  }
  if (progress.bytesPerSecond > 0) {
    parts.push(`${formatBytes(progress.bytesPerSecond)}/s`);
  }
  return parts.join(' · ');
}

export function formatUpdaterError(error, action = 'update Mirid') {
  const detail = String(error?.message || error || '').trim();
  if (!detail) return `Mirid could not ${action}.`;
  return `Mirid could not ${action}: ${detail}`;
}

export async function installAppUpdate(update, onProgress) {
  if (!update) throw new Error('No update is ready to install.');

  let progress = createUpdateProgress();
  await update.downloadAndInstall(
    (event) => {
      progress = reduceUpdateProgress(progress, event);
      onProgress?.(progress);
    },
    { timeout: DOWNLOAD_TIMEOUT_MS },
  );
  await relaunch();
}
