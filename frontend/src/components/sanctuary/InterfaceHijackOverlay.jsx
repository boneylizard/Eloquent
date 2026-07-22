/**
 * InterfaceHijackOverlay — applies CSS theme/shake/lock/glitch effects
 * based on the interface_hijack directives from somatic payloads.
 *
 * Wraps the chat area with a filter layer that applies:
 *   theme_shift → hue-rotate / saturate / brightness filters
 *   shake       → CSS keyframe animation
 *   lock        → overlay blocking input + scroll
 *   glitch      → clip-path glitch animation
 *
 * A safety override button is always visible to force-release all effects.
 * All effects respect prefers-reduced-motion.
 */

import React, { useMemo } from 'react';
import { ShieldOff } from 'lucide-react';
import { useInterfaceHijack } from '../../contexts/InterfaceHijackContext';

export default function InterfaceHijackOverlay() {
  const { hijackState, pythonDrives, isOverridden, safetyOverride } = useInterfaceHijack();

  const filterStyle = useMemo(() => {
    const { theme_shift } = hijackState;
    if (!theme_shift) return undefined;
    const hue = theme_shift.hue || 0;
    const sat = theme_shift.saturation ?? 1.0;
    const bright = theme_shift.brightness ?? 0.8;
    if (hue === 0 && sat === 1.0 && bright === 0.8) return undefined;
    return {
      filter: `hue-rotate(${hue}deg) saturate(${sat}) brightness(${bright})`,
    };
  }, [hijackState.theme_shift]);

  const hasShake = hijackState.shake?.intensity > 0;
  const hasGlitch = hijackState.glitch?.intensity > 0;
  const isLocked = hijackState.lock?.input_locked;
  const hasAnyEffect = !!filterStyle || hasShake || hasGlitch || isLocked;

  if (!hasAnyEffect && !pythonDrives) return null;

  return (
    <>
      {/* Filter + shake + glitch layer */}
      <div
        className={`sanctuary-hijack-overlay ${hasShake ? 'sanctuary-hijack-shake' : ''} ${hasGlitch ? 'sanctuary-hijack-glitch' : ''}`}
        style={{
          ...filterStyle,
          ...(hasShake ? { '--shake-intensity': hijackState.shake.intensity } : {}),
        }}
      />

      {/* Lock overlay */}
      {isLocked && (
        <div className="sanctuary-hijack-lock-overlay">
          <span className="sanctuary-hijack-lock-text">
            ◈ locked ◈
          </span>
        </div>
      )}

      {/* Safety override button — always visible when any effect is active */}
      {hasAnyEffect && (
        <button
          className="sanctuary-safety-button"
          onClick={safetyOverride}
          title="Force-release all interface hijack effects"
        >
          <ShieldOff size={10} style={{ display: 'inline', marginRight: '0.2rem' }} />
          {isOverridden ? 'released' : 'override'}
        </button>
      )}
    </>
  );
}
