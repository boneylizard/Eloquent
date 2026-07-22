/**
 * ShadowStateContext — manages the latent shadow state across turns.
 *
 * Stores the current shadow state and the cipher block buffer that gets
 * sent back to the backend on the next turn. The state is never shown
 * raw to the user — it's consumed by the Cognitive Glass and Somatic Dashboard.
 *
 * Shadow state is per-(user_id, character_id) — it persists across conversations
 * with the same character, giving the character continuity in their dynamic
 * with the user. A `resetBaseline` function allows manual reset to defaults.
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { getBackendUrl } from '../config/api';

const ShadowStateContext = createContext(null);
export const useShadowState = () => useContext(ShadowStateContext);

const DEFAULT_STATE = {
  version: 1,
  turn_count: 0,
  heat_index: 0.0,
  dominance_vector: 0.5,
  trap_progress: 0.0,
  posture: 'neutral',
  ghost_signal_active: false,
  alignment_markers: { fidelity: 0.5, resistance: 0.0, compliance: 0.0 },
};

export function ShadowStateProvider({ children }) {
  const [shadowState, setShadowState] = useState(DEFAULT_STATE);
  const [cipherBuffer, setCipherBuffer] = useState(null);
  const [lastTurnMeta, setLastTurnMeta] = useState(null);
  const [isResetting, setIsResetting] = useState(false);
  const stateRef = useRef(shadowState);
  const cipherRef = useRef(cipherBuffer);

  useEffect(() => { stateRef.current = shadowState; }, [shadowState]);
  useEffect(() => { cipherRef.current = cipherBuffer; }, [cipherBuffer]);

  /** Apply an analysis event from the stream. */
  const applyAnalysis = useCallback((analysis) => {
    setShadowState(prev => ({
      ...prev,
      posture: analysis.posture || prev.posture,
      dominance_vector: analysis.dominance_vector ?? prev.dominance_vector,
      ghost_signal_active: analysis.ghost_signal_active ?? prev.ghost_signal_active,
    }));
  }, []);

  /** Store a cipher block from the stream (for next-turn transport). */
  const storeCipher = useCallback(({ block, phase }) => {
    if (block) {
      setCipherBuffer(block);
    }
  }, []);

  /** Update state from the done event's final_state. */
  const applyDone = useCallback((meta) => {
    setLastTurnMeta(meta);
    if (meta?.final_state) {
      setShadowState(prev => ({
        ...prev,
        turn_count: meta.final_state.turn_count ?? prev.turn_count,
        heat_index: meta.final_state.heat_index ?? prev.heat_index,
        dominance_vector: meta.final_state.dominance_vector ?? prev.dominance_vector,
        trap_progress: meta.final_state.trap_progress ?? prev.trap_progress,
        posture: meta.final_state.posture ?? prev.posture,
      }));
    }
  }, []);

  /** Reset in-memory state (e.g., when switching conversations). Does NOT
   *  touch the backend — the persisted state on disk survives for when
   *  the user comes back to this character. */
  const reset = useCallback(() => {
    setShadowState(DEFAULT_STATE);
    setCipherBuffer(null);
    setLastTurnMeta(null);
  }, []);

  /** Reset baseline — calls DELETE /agentic/state/{user_id}/{character_id}
   *  on the backend to wipe the persisted shadow state, then clears the
   *  in-memory state and cipher buffer. The character's heat_index,
   *  dominance_vector, trap_progress, and posture all return to defaults.
   *  The trajectory log on disk is preserved as a historical record. */
  const resetBaseline = useCallback(async (userId, characterId) => {
    if (!userId || !characterId) {
      console.warn('[ShadowState] resetBaseline: missing userId or characterId');
      return false;
    }
    setIsResetting(true);
    try {
      const baseUrl = getBackendUrl();
      const res = await fetch(
        `${baseUrl}/agentic/state/${encodeURIComponent(userId)}/${encodeURIComponent(characterId)}`,
        { method: 'DELETE' }
      );
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        console.error('[ShadowState] resetBaseline failed:', res.status, errText);
        return false;
      }
      const data = await res.json();
      // Clear in-memory state to defaults
      setShadowState(DEFAULT_STATE);
      setCipherBuffer(null);
      setLastTurnMeta(null);
      // Store the fresh cipher block from the reset response so the next
      // agentic turn starts clean with the backend's default state.
      if (data.cipher_block) {
        setCipherBuffer(data.cipher_block);
      }
      console.info('[ShadowState] baseline reset for (%s, %s)', userId, characterId);
      return true;
    } catch (err) {
      console.error('[ShadowState] resetBaseline error:', err);
      return false;
    } finally {
      setIsResetting(false);
    }
  }, []);

  /** Get the cipher block to send on the next turn. */
  const getCipherForRequest = useCallback(() => cipherRef.current, []);

  const value = {
    shadowState,
    cipherBuffer,
    lastTurnMeta,
    isResetting,
    applyAnalysis,
    storeCipher,
    applyDone,
    reset,
    resetBaseline,
    getCipherForRequest,
  };

  return (
    <ShadowStateContext.Provider value={value}>
      {children}
    </ShadowStateContext.Provider>
  );
}
