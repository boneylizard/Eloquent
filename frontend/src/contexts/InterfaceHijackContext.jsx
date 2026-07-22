/**
 * InterfaceHijackContext — applies CSS theme/shake/lock/glitch effects
 * based on the `interface_hijack` field in somatic payloads.
 *
 * Driven by the AI's dominance_vector — higher dominance → more intense effects.
 * All effects are bounded and respect prefers-reduced-motion.
 * A safety override is always available to force-release hijack state.
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { mapPythonDrives } from '../utils/pythonDriveMapper';

const InterfaceHijackContext = createContext(null);
export const useInterfaceHijack = () => useContext(InterfaceHijackContext);

const DEFAULT_HIJACK = {
  theme_shift: { hue: 0, saturation: 1.0, brightness: 0.8 },
  shake: { intensity: 0.0, duration_ms: 0 },
  lock: { input_locked: false, scroll_locked: false, duration_ms: 0 },
  glitch: { intensity: 0.0 },
};

const MAX_LOCK_DURATION_MS = 10000; // safety cap
const MAX_SHAKE_INTENSITY = 0.8; // safety cap
const MAX_GLITCH_INTENSITY = 0.6; // safety cap

export function InterfaceHijackProvider({ children }) {
  const [hijackState, setHijackState] = useState(DEFAULT_HIJACK);
  const [pythonDrives, setPythonDrives] = useState(null);
  const [isOverridden, setIsOverridden] = useState(false); // user safety override
  const lockTimerRef = useRef(null);
  const shakeTimerRef = useRef(null);

  // Check prefers-reduced-motion
  const prefersReducedMotion = useRef(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => { prefersReducedMotion.current = mq.matches; };
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, []);

  /** Apply hijack directives from a somatic payload. */
  const applyHijack = useCallback((hijack, drives = null) => {
    if (isOverridden) return; // respect user safety override

    let mergedHijack = hijack;

    if (drives) {
      setPythonDrives(drives);
      const driveEffects = mapPythonDrives(drives);
      if (driveEffects) {
        mergedHijack = _mergeHijackEffects(hijack, driveEffects);
      }
    }

    const safe = _sanitizeHijack(mergedHijack, prefersReducedMotion.current);
    setHijackState(safe);

    // Apply lock with auto-release
    if (safe.lock?.input_locked && safe.lock?.duration_ms > 0) {
      if (lockTimerRef.current) clearTimeout(lockTimerRef.current);
      lockTimerRef.current = setTimeout(() => {
        setHijackState(prev => ({
          ...prev,
          lock: { input_locked: false, scroll_locked: false, duration_ms: 0 },
        }));
      }, Math.min(safe.lock.duration_ms, MAX_LOCK_DURATION_MS));
    }

    // Clear shake after duration
    if (safe.shake?.intensity > 0 && safe.shake?.duration_ms > 0) {
      if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
      shakeTimerRef.current = setTimeout(() => {
        setHijackState(prev => ({
          ...prev,
          shake: { intensity: 0.0, duration_ms: 0 },
        }));
      }, safe.shake.duration_ms);
    }
  }, [isOverridden]);

  /** User safety override — force-release all hijack effects. */
  const safetyOverride = useCallback(() => {
    setIsOverridden(true);
    setHijackState(DEFAULT_HIJACK);
    setPythonDrives(null);
    if (lockTimerRef.current) clearTimeout(lockTimerRef.current);
    if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);

    // Re-enable after a short cooldown
    setTimeout(() => setIsOverridden(false), 5000);
  }, []);

  /** Reset to default (e.g., when switching conversations). */
  const reset = useCallback(() => {
    setHijackState(DEFAULT_HIJACK);
    setPythonDrives(null);
    setIsOverridden(false);
    if (lockTimerRef.current) clearTimeout(lockTimerRef.current);
    if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
  }, []);

  useEffect(() => {
    return () => {
      if (lockTimerRef.current) clearTimeout(lockTimerRef.current);
      if (shakeTimerRef.current) clearTimeout(shakeTimerRef.current);
    };
  }, []);

  const value = {
    hijackState,
    pythonDrives,
    isOverridden,
    applyHijack,
    safetyOverride,
    reset,
  };

  return (
    <InterfaceHijackContext.Provider value={value}>
      {children}
    </InterfaceHijackContext.Provider>
  );
}

function _sanitizeHijack(hijack, reducedMotion) {
  if (!hijack || typeof hijack !== 'object') return DEFAULT_HIJACK;

  const shake = hijack.shake || {};
  const lock = hijack.lock || {};
  const glitch = hijack.glitch || {};
  const theme = hijack.theme_shift || {};

  return {
    theme_shift: {
      hue: typeof theme.hue === 'number' ? theme.hue : 0,
      saturation: typeof theme.saturation === 'number' ? theme.saturation : 1.0,
      brightness: typeof theme.brightness === 'number' ? theme.brightness : 0.8,
    },
    shake: reducedMotion
      ? { intensity: 0.0, duration_ms: 0 }
      : {
          intensity: Math.min(MAX_SHAKE_INTENSITY, typeof shake.intensity === 'number' ? shake.intensity : 0.0),
          duration_ms: Math.min(MAX_LOCK_DURATION_MS, typeof shake.duration_ms === 'number' ? shake.duration_ms : 0),
        },
    lock: {
      input_locked: !!lock.input_locked,
      scroll_locked: !!lock.scroll_locked,
      duration_ms: Math.min(MAX_LOCK_DURATION_MS, typeof lock.duration_ms === 'number' ? lock.duration_ms : 0),
    },
    glitch: reducedMotion
      ? { intensity: 0.0 }
      : {
          intensity: Math.min(MAX_GLITCH_INTENSITY, typeof glitch.intensity === 'number' ? glitch.intensity : 0.0),
        },
  };
}

function _mergeHijackEffects(base, driveEffects) {
  if (!base || typeof base !== 'object') return driveEffects || DEFAULT_HIJACK;
  if (!driveEffects) return base;

  const merged = { ...base };

  if (driveEffects.theme_shift) {
    merged.theme_shift = driveEffects.theme_shift;
  }
  if (driveEffects.shake) {
    merged.shake = {
      intensity: Math.max(base.shake?.intensity || 0, driveEffects.shake.intensity || 0),
      duration_ms: Math.max(base.shake?.duration_ms || 0, driveEffects.shake.duration_ms || 0),
    };
  }
  if (driveEffects.lock) {
    merged.lock = {
      input_locked: base.lock?.input_locked || driveEffects.lock.input_locked,
      scroll_locked: base.lock?.scroll_locked || driveEffects.lock.scroll_locked,
      duration_ms: Math.max(base.lock?.duration_ms || 0, driveEffects.lock.duration_ms || 0),
    };
  }
  if (driveEffects.glitch) {
    merged.glitch = {
      intensity: Math.max(base.glitch?.intensity || 0, driveEffects.glitch.intensity || 0),
    };
  }

  return merged;
}
