import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * Polls the Rust sidecar_status command so the UI knows whether Mirid's
 * backend and TTS processes are actually alive. TTS can crash silently —
 * this is how the developer panel detects it.
 */
export function useSidecarStatus(intervalMs = 2000) {
  const [status, setStatus] = useState({ backend: false, tts: false });
  const [available, setAvailable] = useState(true);
  const timer = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await invoke('sidecar_status');
        if (cancelled) return;
        setAvailable(true);
        setStatus({ backend: !!res.backend, tts: !!res.tts });
      } catch {
        if (!cancelled) setAvailable(false);
      }
    };

    poll();
    timer.current = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      if (timer.current) clearInterval(timer.current);
    };
  }, [intervalMs]);

  return { ...status, available };
}
