import { useEffect, useRef, useState, useCallback } from 'react';
import { getBackendUrl, getTtsUrl } from '../config/api';

/**
 * Polls the backend (and TTS) health endpoints and reports a live connection
 * state. Auto-retries forever so the UI can show "Connecting…" instead of a
 * one-shot "Failed to fetch". State: 'connecting' | 'online' | 'offline'.
 */
export function useBackendStatus() {
  const [backend, setBackend] = useState('connecting'); // 'connecting' | 'online' | 'offline'
  const [tts, setTts] = useState('connecting');
  const [lastChecked, setLastChecked] = useState(null);
  const [detail, setDetail] = useState('');
  const timer = useRef(null);
  const cancelled = useRef(false);

  const checkOne = useCallback(async (base, path) => {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), 4000);
    try {
      const res = await fetch(`${base}${path}`, { signal: ctrl.signal });
      return res.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(id);
    }
  }, []);

  const tick = useCallback(async () => {
    const backendOk = await checkOne(getBackendUrl(), '/health');
    const ttsOk = await checkOne(getTtsUrl(), '/health');
    if (cancelled.current) return;
    setBackend(backendOk ? 'online' : 'offline');
    setTts(ttsOk ? 'online' : 'offline');
    setDetail(
      `backend:${backendOk ? 'up' : 'down'} tts:${ttsOk ? 'up' : 'down'}`
    );
    setLastChecked(Date.now());
  }, [checkOne]);

  useEffect(() => {
    cancelled.current = false;
    tick();
    const loop = () => {
      timer.current = setTimeout(async () => {
        await tick();
        loop();
      }, 3000);
    };
    loop();
    return () => {
      cancelled.current = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [tick]);

  return { backend, tts, detail, lastChecked };
}
