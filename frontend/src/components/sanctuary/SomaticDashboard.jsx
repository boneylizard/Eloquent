/**
 * SomaticDashboard — discrete natural-language indicator chips
 * updated via the somatic payload from the agentic pipeline.
 *
 * Displays: lubrication_level, pupil_dilation, spatial_position,
 * breath_rate, muscle_tension, posture_label.
 */

import React from 'react';
import { useSomaticDashboard } from '../../contexts/SomaticDashboardContext';
import { useAgenticProfile } from '../../contexts/AgenticProfileContext';

function formatValue(key, value) {
  if (key === 'pupil_dilation' || key === 'muscle_tension') {
    const pct = Math.round((value || 0) * 100);
    return `${pct}%`;
  }
  return String(value || '—');
}

export default function SomaticDashboard() {
  const { somaticPayload, isUpdating } = useSomaticDashboard();
  const { displayConfig } = useAgenticProfile();

  const chipConfig = displayConfig?.dashboard_chips || {};
  const dashboard = somaticPayload?.dashboard;
  if (!dashboard) return null;

  const chips = Object.entries(chipConfig).map(([key, cfg]) => ({
    key,
    label: cfg?.label || key,
    color: cfg?.color || 'rgba(120, 180, 255, 0.8)',
    value: dashboard[key],
  }));

  return (
    <div className="sanctuary-somatic-dashboard">
      {chips.map((chip) => (
        <span
          key={chip.key}
          className={`sanctuary-somatic-chip ${isUpdating ? 'sanctuary-somatic-chip-updating' : ''}`}
          style={{
            borderColor: chip.color,
            boxShadow: isUpdating ? `0 0 12px ${chip.color}` : 'none',
            transition: 'all 0.3s ease',
          }}
        >
          <span className="sanctuary-somatic-chip-label">{chip.label}</span>
          <span className="sanctuary-somatic-chip-value">
            {formatValue(chip.key, chip.value)}
          </span>
        </span>
      ))}
      {somaticPayload?.posture_label && (
        <div className="sanctuary-somatic-posture-label w-full">
          {somaticPayload.posture_label}
        </div>
      )}
    </div>
  );
}
