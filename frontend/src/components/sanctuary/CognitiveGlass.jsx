/**
 * CognitiveGlass — streaming panel displaying the agent's chain-of-thought
 * and cipher blocks. Collapsible, floatable (pop-out as modal), with clear
 * state indicators.
 *
 * Modes:
 *   - Hidden (default): chat takes full width
 *   - Sidebar: 280px right rail
 *   - Popout: floating centered modal (large screen friendly)
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { ChevronRight, ChevronLeft, Brain, RotateCcw, Maximize2, Minimize2, Columns, X } from 'lucide-react';
import { useCognitiveGlass } from '../../contexts/CognitiveGlassContext';
import { useShadowState } from '../../contexts/ShadowStateContext';
import { useSomaticDashboard } from '../../contexts/SomaticDashboardContext';
import { useInterfaceHijack } from '../../contexts/InterfaceHijackContext';
import CipherGlyphRow from './CipherGlyphRow';
import SceneFrameRenderer from './SceneFrameRenderer';

const MODE = { HIDDEN: 'hidden', SIDEBAR: 'sidebar', POPOUT: 'popout' };

export default function CognitiveGlass({ characterName, userId, characterId }) {
  const {
    reasoningEntries,
    cipherGlyphs,
    sceneFrames,
    isStreaming,
    reset: resetGlass,
  } = useCognitiveGlass();
  const { resetBaseline, isResetting } = useShadowState();
  const { reset: resetSomatic } = useSomaticDashboard();
  const { reset: resetHijack } = useInterfaceHijack();

  const [mode, setMode] = useState(MODE.HIDDEN);

  // Auto-show sidebar when streaming starts (if hidden)
  useEffect(() => {
    if (isStreaming && mode === MODE.HIDDEN) {
      setMode(MODE.SIDEBAR);
    }
  }, [isStreaming]);

  const sortedEntries = useMemo(
    () => [...reasoningEntries].sort((a, b) => a.id - b.id),
    [reasoningEntries]
  );

  const sortedCiphers = useMemo(
    () => [...cipherGlyphs].sort((a, b) => a.timestamp - b.timestamp),
    [cipherGlyphs]
  );

  const handleResetBaseline = useCallback(async () => {
    if (!userId || !characterId) return;
    const charLabel = characterName || 'this character';
    const confirmed = window.confirm(
      `Reset shadow state for ${charLabel}?\n\nThis clears heat_index, dominance_vector, trap_progress, and posture back to baseline. The conversation transcript is not affected.`
    );
    if (!confirmed) return;
    const success = await resetBaseline(userId, characterId);
    if (success) {
      resetGlass();
      resetSomatic();
      resetHijack();
    }
  }, [userId, characterId, characterName, resetBaseline, resetGlass, resetSomatic, resetHijack]);

  const cycleMode = () => {
    setMode(prev => {
      if (prev === MODE.HIDDEN) return MODE.SIDEBAR;
      if (prev === MODE.SIDEBAR) return MODE.POPOUT;
      return MODE.HIDDEN;
    });
  };

  const glassContent = (
    <>
      <div className="sanctuary-glass-header">
        <div className="flex items-center gap-1.5">
          <Brain size={12} style={{ opacity: isStreaming ? 1 : 0.4, animation: isStreaming ? 'pulse 2s infinite' : 'none' }} />
          <span>Cognitive Glass</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span style={{ opacity: isStreaming ? 0.9 : 0.3, fontSize: '0.6rem' }}>
            {isStreaming ? 'live' : 'idle'}
          </span>
          <button onClick={handleResetBaseline} disabled={isResetting || isStreaming}
            className="sanctuary-glass-icon-btn" title="Reset baseline">
            <RotateCcw size={10} />
          </button>
          {mode === MODE.SIDEBAR && (
            <button onClick={() => setMode(MODE.POPOUT)} className="sanctuary-glass-icon-btn" title="Pop out">
              <Maximize2 size={10} />
            </button>
          )}
          {mode === MODE.POPOUT && (
            <button onClick={() => setMode(MODE.SIDEBAR)} className="sanctuary-glass-icon-btn" title="Dock">
              <Minimize2 size={10} />
            </button>
          )}
          <button onClick={() => setMode(MODE.HIDDEN)} className="sanctuary-glass-icon-btn" title="Close">
            <X size={10} />
          </button>
        </div>
      </div>
      <div className="sanctuary-glass-body">
        {sortedEntries.length === 0 && sortedCiphers.length === 0 && sceneFrames.length === 0 && (
          <div className="text-center py-6" style={{ color: 'rgba(100, 120, 180, 0.15)', fontSize: '0.6rem' }}>
            Awaiting cognitive stream…
          </div>
        )}
        {sortedCiphers.map((c, i) => (
          <CipherGlyphRow key={`cipher-${i}`} glyphs={c.glyphs} block={c.block} phase={c.phase} />
        ))}
        {sceneFrames.map((sf, i) => (
          <SceneFrameRenderer key={`scene-${i}`} xmlFrame={sf.xml} />
        ))}
        {sortedEntries.map((entry) => (
          <div key={entry.id} className={`sanctuary-reasoning-entry sanctuary-reasoning-entry-type-${entry.type}`}>
            {entry.type === 'analysis' && (
              <>
                <div className="sanctuary-reasoning-label">
                  Analysis · {entry.posture} · dom={((entry.dominance ?? 0) * 100).toFixed(0)}%
                </div>
                {entry.reasoning && (
                  <div className="sanctuary-reasoning-text">{entry.reasoning}</div>
                )}
                {entry.emotional_state && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.55rem', opacity: 0.8 }}>Emotional: {entry.emotional_state}</div>
                )}
                {entry.physical_state && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.55rem', opacity: 0.8 }}>Physical: {entry.physical_state}</div>
                )}
                {entry.trajectory && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.55rem', opacity: 0.7 }}>Trajectory: {entry.trajectory}</div>
                )}
                {entry.internal_state && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.5rem', opacity: 0.6, fontStyle: 'italic' }}>Internal: {entry.internal_state}</div>
                )}
                {entry.external_state && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.5rem', opacity: 0.6, fontStyle: 'italic' }}>External: {entry.external_state}</div>
                )}
              </>
            )}
            {entry.type === 'somatic' && (
              <>
                <div className="sanctuary-reasoning-label">
                  Somatic · {entry.postureLabel}
                  {entry.ghostActive && ' · ghost'}
                  {entry.whoreMode && ' · mode'}
                </div>
                {entry.somatic_narrative && (
                  <div className="sanctuary-reasoning-text">{entry.somatic_narrative}</div>
                )}
                {entry.behavioral_cues && (
                  <div className="sanctuary-reasoning-text" style={{ fontSize: '0.55rem', opacity: 0.8 }}>Behavioral: {entry.behavioral_cues}</div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </>
  );

  return (
    <>
      {/* Sidebar mode */}
      {mode === MODE.SIDEBAR && (
        <div className="sanctuary-glass sanctuary-glass-expanded">
          {glassContent}
        </div>
      )}

      {/* Popout mode — floating modal */}
      {mode === MODE.POPOUT && (
        <>
          <div className="sanctuary-glass-popout-backdrop" onClick={() => setMode(MODE.SIDEBAR)} />
          <div className="sanctuary-glass sanctuary-glass-popout">
            {glassContent}
          </div>
        </>
      )}

      {/* Hidden mode — tiny toggle button floating at right edge */}
      {mode === MODE.HIDDEN && (
        <button onClick={() => setMode(MODE.SIDEBAR)} className="sanctuary-glass-peek-btn"
          title="Show Cognitive Glass">
          <Brain size={14} style={{ opacity: 0.4 }} />
          <span style={{ fontSize: '0.5rem', writingMode: 'vertical-rl' }}>glass</span>
        </button>
      )}
    </>
  );
}
