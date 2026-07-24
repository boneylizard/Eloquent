import { useCallback, useMemo } from 'react';
import { useApp } from '../contexts/AppContext';
import { isSettingsStandaloneWindow } from '../utils/settingsCrossWindowSync';

/**
 * Shared boot gate for Settings / memory panels.
 * apiReady = port config resolved (URLs usable). Does NOT block on chat IndexedDB hydration.
 * Optional backend/memory calls should fail fast with errors, not infinite spinners.
 */
export function useAppBoot() {
  const {
    portsReady,
    storageHydrated,
    storageHydrationDegraded,
    portsLoadDegraded,
    retryBoot,
    PRIMARY_API_URL,
    MEMORY_API_URL,
  } = useApp();

  const settingsStandalone = isSettingsStandaloneWindow();
  const apiReady = portsReady;

  const banner = useMemo(() => {
    if (portsLoadDegraded) {
      return {
        tone: 'warning',
        message: 'Mirid is still waiting for its local service endpoints. Model, memory, and voice features will remain paused until the desktop host responds.',
      };
    }
    if (!portsReady) {
      return {
        tone: 'loading',
        message: settingsStandalone
          ? 'Reading Mirid’s local service endpoints…'
          : 'Reading local service endpoints…',
      };
    }
    if (!storageHydrated && !settingsStandalone) {
      return {
        tone: 'loading',
        message: 'Loading local data (characters, chats, settings)…',
      };
    }
    if (!storageHydrated && settingsStandalone) {
      return {
        tone: 'loading',
        message: 'Syncing local settings from storage…',
      };
    }
    if (storageHydrationDegraded) {
      return {
        tone: 'warning',
        message: settingsStandalone
          ? 'Local settings sync took too long — API calls still work; retry or reload if fields look empty.'
          : 'Local data took too long to load — the app is usable but some chats may appear after refresh or when you open the chat tab.',
      };
    }
    return null;
  }, [portsReady, portsLoadDegraded, storageHydrated, storageHydrationDegraded, settingsStandalone]);

  const retry = useCallback(() => {
    retryBoot?.();
  }, [retryBoot]);

  return {
    apiReady,
    portsReady,
    storageHydrated,
    storageHydrationDegraded,
    portsLoadDegraded,
    banner,
    retry,
    PRIMARY_API_URL,
    MEMORY_API_URL,
  };
}
