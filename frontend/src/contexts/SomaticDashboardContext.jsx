/**
 * SomaticDashboardContext — manages the latest somatic payload for the dashboard.
 *
 * Updated via the `somatic` SSE event from /agentic/turn.
 * Consumed by the SomaticDashboard component (Phase 5).
 */

import React, { createContext, useContext, useState, useCallback } from 'react';

const SomaticDashboardContext = createContext(null);
export const useSomaticDashboard = () => useContext(SomaticDashboardContext);

const DEFAULT_SOMATIC = {
  dashboard: {
    lubrication_level: 'dry',
    pupil_dilation: 0.0,
    spatial_position: 'across_room',
    breath_rate: 'steady',
    muscle_tension: 0.0,
  },
  whore_mode: false,
  posture_label: 'neutral stance',
  ghost_signal: { active: false, charge: 0.0, carrier_phrase: '' },
  xml_frame: null,
  python_drive: null,
};

export function SomaticDashboardProvider({ children }) {
  const [somaticPayload, setSomaticPayload] = useState(DEFAULT_SOMATIC);
  const [isUpdating, setIsUpdating] = useState(false);
  const updateTimerRef = React.useRef(null);

  /** Apply a somatic event from the stream. */
  const applySomatic = useCallback((payload) => {
    setSomaticPayload(payload);
    setIsUpdating(true);

    // Brief "updating" pulse for UI animation
    if (updateTimerRef.current) clearTimeout(updateTimerRef.current);
    updateTimerRef.current = setTimeout(() => setIsUpdating(false), 1500);
  }, []);

  const reset = useCallback(() => {
    setSomaticPayload(DEFAULT_SOMATIC);
    setIsUpdating(false);
  }, []);

  const value = {
    somaticPayload,
    isUpdating,
    applySomatic,
    reset,
  };

  return (
    <SomaticDashboardContext.Provider value={value}>
      {children}
    </SomaticDashboardContext.Provider>
  );
}
