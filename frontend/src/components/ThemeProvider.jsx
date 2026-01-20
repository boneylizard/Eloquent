import React, { createContext, useContext, useEffect, useState, useMemo } from 'react';

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

  // Use useMemo to avoid unnecessary re-renders
  const setTheme = useMemo(() => {
    return (newTheme) => {
      setThemeState(newTheme);
    };
  }, []);


  useEffect(() => {
    if (typeof window === 'undefined') return;

    const root = window.document.documentElement;
    root.dataset.theme = theme;

    // Legacy cleanup (ensure classes are gone so they don't fight tokens)
    root.classList.remove('light', 'dark', 'messenger', 'whatsapp');

    // Add 'dark' class if the theme is dark so Tailwind dark: modifiers work
    if (theme === 'dark' || theme === 'whatsapp' || theme === 'messenger' || theme === 'cyberpunk') {
      root.classList.add('dark');
    }

    try {
      localStorage.setItem(storageKey, theme);
    } catch (error) {
      console.error("Error saving to localStorage:", error);
    }
  }, [theme, storageKey]);

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