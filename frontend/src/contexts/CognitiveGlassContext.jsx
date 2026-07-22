/**
 * CognitiveGlassContext — manages the reasoning stream and cipher glyphs
 * for the Cognitive Glass panel.
 *
 * Receives analysis reasoning, somatic posture labels, and cipher blocks.
 * The cipher blocks are stored as glyph rows for visual display.
 */

import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { cipherToGlyphs } from '../utils/cipher';

const CognitiveGlassContext = createContext(null);
export const useCognitiveGlass = () => useContext(CognitiveGlassContext);

export function CognitiveGlassProvider({ children }) {
  const [reasoningEntries, setReasoningEntries] = useState([]);
  const [cipherGlyphs, setCipherGlyphs] = useState([]);
  const [sceneFrames, setSceneFrames] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const entryIdRef = useRef(0);

  /** Start a new turn's reasoning stream. */
  const startTurn = useCallback(() => {
    setIsStreaming(true);
  }, []);

  /** Add an analysis reasoning entry. */
  const addAnalysisReasoning = useCallback((analysis) => {
    const entry = {
      ...analysis,
      id: ++entryIdRef.current,
      type: 'analysis',
      dominance: analysis.dominance_vector,
      reasoning: analysis.reasoning || '',
      timestamp: Date.now(),
    };
    setReasoningEntries(prev => [...prev.slice(-20), entry]); // keep last 20
  }, []);

  /** Add a somatic posture label entry. */
  const addSomaticLabel = useCallback((somatic) => {
    const entry = {
      ...somatic,
      id: ++entryIdRef.current,
      type: 'somatic',
      postureLabel: somatic.posture_label || '',
      whoreMode: somatic.whore_mode || false,
      ghostActive: somatic.ghost_signal?.active || false,
      timestamp: Date.now(),
    };
    setReasoningEntries(prev => [...prev.slice(-20), entry]);
    if (somatic.xml_frame) {
      setSceneFrames(prev => [...prev.slice(-5), { xml: somatic.xml_frame, timestamp: Date.now() }]);
    }
  }, []);

  /** Add a tactile outreach entry (pose, gesture, proximity). */
  const addTactileEntry = useCallback((tactile) => {
    if (!tactile) return;
    const entry = {
      id: ++entryIdRef.current,
      type: 'tactile',
      labelDisplay: tactile._label_display || 'Tactile Outreach',
      pose: tactile.pose || '',
      gestureNarrative: tactile.gesture_narrative || '',
      proximity: tactile.proximity || '',
      timestamp: Date.now(),
    };
    setReasoningEntries(prev => [...prev.slice(-20), entry]);
  }, []);

  /** Add a character signal entry (covert action, voice this turn). */
  const addSignalEntry = useCallback((signal) => {
    if (!signal) return;
    const entry = {
      id: ++entryIdRef.current,
      type: 'signal',
      labelDisplay: signal._label_display || 'Character Signal',
      covertAction: signal.covert_action || '',
      voiceThisTurn: signal.voice_this_turn || '',
      timestamp: Date.now(),
    };
    setReasoningEntries(prev => [...prev.slice(-20), entry]);
  }, []);

  /** Add cipher glyphs from a cipher event. */
  const addCipherGlyphs = useCallback(({ block, phase }) => {
    if (!block) return;
    const glyphs = cipherToGlyphs(block);
    setCipherGlyphs(prev => [...prev.slice(-8), { glyphs, block, phase, timestamp: Date.now() }]);
  }, []);

  /** Mark the turn as complete. */
  const endTurn = useCallback(() => {
    setIsStreaming(false);
  }, []);

  /** Clear all entries (e.g., when switching conversations). */
  const reset = useCallback(() => {
    setReasoningEntries([]);
    setCipherGlyphs([]);
    setSceneFrames([]);
    setIsStreaming(false);
    entryIdRef.current = 0;
  }, []);

  const value = {
    reasoningEntries,
    cipherGlyphs,
    sceneFrames,
    isStreaming,
    startTurn,
    addAnalysisReasoning,
    addSomaticLabel,
    addTactileEntry,
    addSignalEntry,
    addCipherGlyphs,
    endTurn,
    reset,
  };

  return (
    <CognitiveGlassContext.Provider value={value}>
      {children}
    </CognitiveGlassContext.Provider>
  );
}
