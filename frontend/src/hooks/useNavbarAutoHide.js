import { useCallback, useEffect, useRef, useState } from 'react';

const PIN_STORAGE_KEY = 'eloquent-navbar-pinned';
/** Scroll down this many px (accumulated) before hiding */
const HIDE_ACCUMULATED = 80;
/** Scroll up this many px (accumulated) before showing */
const SHOW_ACCUMULATED = 40;
const TOP_REVEAL = 12;
const DEBOUNCE_MS = 175;
const TRANSITION_MS = 300;

function readPinnedPreference() {
  try {
    return localStorage.getItem(PIN_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

/**
 * Hide the top navbar when the user scrolls down; reveal on scroll up.
 * @param {HTMLElement | null} scrollElement - scrollable chat/main container
 */
export function useNavbarAutoHide(scrollElement) {
  const [navbarHidden, setNavbarHidden] = useState(false);
  const [navbarPinned, setNavbarPinnedState] = useState(readPinnedPreference);
  const lastY = useRef(0);
  const scrollAccum = useRef(0);
  const ticking = useRef(false);
  const hiddenRef = useRef(false);
  const pinnedRef = useRef(navbarPinned);
  const transitioningUntil = useRef(0);
  const debounceTimer = useRef(null);
  const pendingHidden = useRef(null);

  pinnedRef.current = navbarPinned;
  hiddenRef.current = navbarHidden;

  const clearDebounce = useCallback(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
      debounceTimer.current = null;
    }
    pendingHidden.current = null;
  }, []);

  const commitHidden = useCallback(
    (next) => {
      if (next === hiddenRef.current) return;
      if (Date.now() < transitioningUntil.current) return;

      hiddenRef.current = next;
      setNavbarHidden(next);
      scrollAccum.current = 0;
      transitioningUntil.current = Date.now() + TRANSITION_MS;
    },
    [],
  );

  const scheduleHidden = useCallback(
    (next) => {
      if (pinnedRef.current) {
        clearDebounce();
        commitHidden(false);
        return;
      }
      if (next === hiddenRef.current) {
        clearDebounce();
        return;
      }
      if (pendingHidden.current === next && debounceTimer.current) return;

      pendingHidden.current = next;
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
      debounceTimer.current = setTimeout(() => {
        debounceTimer.current = null;
        pendingHidden.current = null;
        if (pinnedRef.current) {
          commitHidden(false);
          return;
        }
        if (Date.now() < transitioningUntil.current) return;
        commitHidden(next);
      }, DEBOUNCE_MS);
    },
    [clearDebounce, commitHidden],
  );

  const setNavbarPinned = useCallback(
    (value) => {
      setNavbarPinnedState(value);
      try {
        localStorage.setItem(PIN_STORAGE_KEY, value ? 'true' : 'false');
      } catch {
        /* ignore */
      }
      if (value) {
        clearDebounce();
        commitHidden(false);
      }
    },
    [clearDebounce, commitHidden],
  );

  useEffect(() => {
    const el = scrollElement;
    if (!el) return undefined;

    const onScroll = () => {
      if (ticking.current) return;
      if (Date.now() < transitioningUntil.current) return;

      ticking.current = true;
      requestAnimationFrame(() => {
        ticking.current = false;

        if (pinnedRef.current) {
          lastY.current = el.scrollTop;
          scrollAccum.current = 0;
          return;
        }

        const y = el.scrollTop;
        const delta = y - lastY.current;
        lastY.current = y;

        if (y <= TOP_REVEAL) {
          scrollAccum.current = 0;
          scheduleHidden(false);
          return;
        }

        if (delta > 0) {
          scrollAccum.current =
            scrollAccum.current > 0 ? scrollAccum.current + delta : delta;
          if (scrollAccum.current >= HIDE_ACCUMULATED) {
            scheduleHidden(true);
          }
        } else if (delta < 0) {
          scrollAccum.current =
            scrollAccum.current < 0 ? scrollAccum.current + delta : delta;
          if (scrollAccum.current <= -SHOW_ACCUMULATED) {
            scheduleHidden(false);
          }
        }
      });
    };

    el.addEventListener('scroll', onScroll, { passive: true });
    lastY.current = el.scrollTop;
    scrollAccum.current = 0;
    return () => {
      el.removeEventListener('scroll', onScroll);
      clearDebounce();
    };
  }, [scrollElement, clearDebounce, scheduleHidden]);

  const revealNavbar = useCallback(() => {
    clearDebounce();
    commitHidden(false);
    lastY.current = scrollElement?.scrollTop ?? 0;
    scrollAccum.current = 0;
  }, [scrollElement, clearDebounce, commitHidden]);

  useEffect(() => {
    const onReveal = () => revealNavbar();
    window.addEventListener('eloquent-navbar-reveal', onReveal);
    return () => window.removeEventListener('eloquent-navbar-reveal', onReveal);
  }, [revealNavbar]);

  useEffect(() => () => clearDebounce(), [clearDebounce]);

  const navbarCollapsed = navbarHidden && !navbarPinned;

  return {
    navbarHidden,
    navbarPinned,
    navbarCollapsed,
    setNavbarPinned,
    toggleNavbarPinned: () => setNavbarPinned(!navbarPinned),
    revealNavbar,
  };
}
