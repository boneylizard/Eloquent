import React, { createContext, useContext, useEffect, useState, useMemo } from 'react';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import { isSettingsStandaloneWindow } from '../utils/settingsCrossWindowSync';

// Create theme context
const ThemeContext = createContext(null);

// Theme provider component
const ThemeProvider = ({
  children,
  defaultTheme = 'dark',
  storageKey = 'ui-theme',
  ...props
}) => {
  const [theme, setThemeState] = useState(() => {
    if (typeof window === 'undefined') {
      return defaultTheme; // SSR, return default
    }

    try {
      const storedTheme = localStorage.getItem(storageKey);
      if (storedTheme) {
        return storedTheme;
      }

      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
      }
    } catch (error) {
      console.error("Error accessing localStorage:", error);
      // Fallback to default if localStorage access fails (e.g., in some restricted environments)
    }

    return defaultTheme;
  });

  /** For IDB-backed keys: do not persist until we've tried to read IDB (otherwise we overwrite empty LS with a fallback theme). */
  const [themePersistReady, setThemePersistReady] = useState(
    () => typeof window === 'undefined' || !indexedDbStorage.useIdb(storageKey)
  );

  // Use useMemo to avoid unnecessary re-renders
  const setTheme = useMemo(() => {
    return (newTheme) => {
      setThemeState(newTheme);
    };
  }, []);

  // If this key was migrated to IndexedDB, localStorage may be empty — hydrate from IDB before persisting.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!indexedDbStorage.useIdb(storageKey)) {
      setThemePersistReady(true);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        if (!localStorage.getItem(storageKey)) {
          const idbOpts = isSettingsStandaloneWindow()
            ? { preferLocalStorage: true, skipMigration: true }
            : {};
          const fromIdb = await indexedDbStorage.getItem(storageKey, idbOpts);
          if (!cancelled && fromIdb) {
            setThemeState(fromIdb);
            try {
              localStorage.setItem(storageKey, fromIdb);
            } catch (_) {
              /* ignore */
            }
          }
        }
      } catch (e) {
        console.warn('[ThemeProvider] IDB theme hydrate failed:', e);
      } finally {
        if (!cancelled) setThemePersistReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [storageKey]);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const root = window.document.documentElement;
    root.dataset.theme = theme;

    // Legacy cleanup (ensure classes are gone so they don't fight tokens)
    root.classList.remove('light', 'dark', 'messenger', 'whatsapp');

    // Add 'dark' class if the theme is dark so Tailwind dark: modifiers work
    if (theme === 'dark' || theme === 'whatsapp' || theme === 'messenger' || theme === 'cyberpunk' || theme === 'nanogpt') {
      root.classList.add('dark');
    }
  }, [theme]);

  useEffect(() => {
    if (typeof window === 'undefined' || !themePersistReady) return;
    try {
      localStorage.setItem(storageKey, theme);
      if (indexedDbStorage.useIdb(storageKey)) {
        void indexedDbStorage.setItem(storageKey, theme);
      }
    } catch (error) {
      console.error("Error saving to localStorage:", error);
    }
  }, [theme, storageKey, themePersistReady]);

  // Use useMemo to avoid unnecessary re-renders
  const value = useMemo(() => ({
    theme,
    setTheme,
  }), [theme, setTheme]
  );

  return (
    <ThemeContext.Provider value={value} {...props}>
      {children}
    </ThemeContext.Provider>
  );
};

// Hook to use theme context
const useTheme = () => {
  const context = useContext(ThemeContext);

  if (context === null) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }

  return context;
};

export { ThemeProvider, useTheme };