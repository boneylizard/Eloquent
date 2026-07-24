import React, { useEffect, useState, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useBackendStatus } from '../hooks/useBackendStatus';
import { useSidecarStatus } from '../hooks/useSidecarStatus';
import { restartTtsService } from '../utils/desktopLifecycle';
import { getBackendUrl, getTtsUrl } from '../config/api';

/**
 * Developer console: full visibility into the desktop app's moving parts.
 * - Live backend + TTS connection state on the desktop host's selected ports
 * - Live sidecar process state (so you SEE when TTS crashes)
 * - Last first-run boot event (download/extract progress + errors)
 * - Rolling tail of the on-disk log file (pasteable to any AI agent)
 * - Restart TTS button
 *
 * Toggle with Ctrl+Shift+D (or the bug icon in the navbar).
 */
export default function DevDebugPanel({ open, onClose }) {
  const [boot, setBoot] = useState(null);
  const [appInfo, setAppInfo] = useState(null);
  const [logTail, setLogTail] = useState('');
  const [logError, setLogError] = useState('');
  const scrollRef = useRef(null);

  const conn = useBackendStatus();
  const sidecar = useSidecarStatus(2000);

  const refreshLog = useCallback(async () => {
    try {
      const tail = await invoke('read_log_tail', { lines: 300 });
      setLogTail(tail || '(empty)');
      setLogError('');
    } catch (e) {
      setLogError(String(e));
    }
  }, []);

  const restartTts = useCallback(async () => {
    try {
      await restartTtsService();
    } catch (e) {
      setLogError(`restart_tts failed: ${e}`);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    let unlisten = null;
    let cancelled = false;
    (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        unlisten = await listen('runtime-boot', (e) => {
          if (!cancelled) setBoot(e.payload);
        });
      } catch {
        /* ignore */
      }
      try {
        const info = await invoke('get_app_info');
        if (!cancelled) setAppInfo(info);
      } catch {
        /* ignore */
      }
      refreshLog();
    })();
    const id = setInterval(refreshLog, 2500);
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
      clearInterval(id);
    };
  }, [open, refreshLog]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logTail]);

  if (!open) return null;

  const Dot = ({ state }) => {
    const color =
      state === 'online' || state === true
        ? '#3ad17e'
        : state === 'connecting'
        ? '#e0b341'
        : '#e05260';
    return (
      <span
        style={{
          display: 'inline-block',
          width: 9,
          height: 9,
          borderRadius: 999,
          background: color,
          marginRight: 6,
          verticalAlign: 'middle',
        }}
      />
    );
  };

  const ttsDown = sidecar.available && !sidecar.tts;
  const backendDown = conn.backend === 'offline';
  const backendPort = new URL(getBackendUrl()).port;
  const ttsPort = new URL(getTtsUrl()).port;

  return (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.panel} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <strong>Developer Console</strong>
          <span style={styles.sub}>full backend / TTS visibility</span>
          <button style={styles.x} onClick={onClose}>
            ✕
          </button>
        </div>

        {(ttsDown || backendDown) && (
          <div style={styles.alert}>
            {backendDown && <div>⚠ Backend ({backendPort}) is DOWN</div>}
            {ttsDown && (
              <div>
                ⚠ TTS ({ttsPort}) CRASHED or stopped — click <b>Restart TTS</b>
              </div>
            )}
          </div>
        )}

        <div style={styles.grid}>
          <div style={styles.cell}>
            <div style={styles.label}>Backend {backendPort}</div>
            <div>
              <Dot state={conn.backend} />
              {conn.backend}
            </div>
          </div>
          <div style={styles.cell}>
            <div style={styles.label}>TTS {ttsPort} (process)</div>
            <div>
              <Dot state={sidecar.tts} />
              {sidecar.available ? (sidecar.tts ? 'alive' : 'DEAD') : 'n/a'}
            </div>
          </div>
          <div style={styles.cell}>
            <div style={styles.label}>First-run stage</div>
            <div>{boot ? `${boot.stage} (${boot.percent}%)` : '—'}</div>
          </div>
          <div style={styles.cell}>
            <div style={styles.label}>Runtime ready</div>
            <div>{appInfo ? String(appInfo.runtime_ready) : '—'}</div>
          </div>
        </div>

        {boot && boot.stage === 'error' && (
          <div style={styles.errBox}>ERROR: {boot.message}</div>
        )}

        <div style={styles.row}>
          <button style={styles.btn} onClick={restartTts}>
            Restart TTS
          </button>
          <button style={styles.btn} onClick={refreshLog}>
            Refresh log
          </button>
          {appInfo && (
            <span style={styles.path}>logs: {appInfo.log_dir}</span>
          )}
        </div>

        <div style={styles.logLabel}>Rolling log tail (pasteable to AI agents)</div>
        {logError && <div style={styles.errBox}>{logError}</div>}
        <pre ref={scrollRef} style={styles.log}>
          {logTail || 'loading…'}
        </pre>
      </div>
    </div>
  );
}

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    zIndex: 2147483001,
    background: 'rgba(0,0,0,0.55)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
    paddingTop: 40,
    fontFamily: 'Inter, system-ui, sans-serif',
  },
  panel: {
    width: 'min(860px, 94vw)',
    maxHeight: '86vh',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    background: '#0f1320',
    border: '1px solid rgba(120,140,255,0.25)',
    borderRadius: 12,
    color: '#dfe5ff',
    boxShadow: '0 24px 70px rgba(0,0,0,0.5)',
  },
  header: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 10,
    padding: '12px 16px',
    borderBottom: '1px solid rgba(255,255,255,0.08)',
  },
  sub: { fontSize: 12, color: '#8b95c8', fontWeight: 400 },
  x: {
    marginLeft: 'auto',
    background: 'transparent',
    border: 'none',
    color: '#9aa6d6',
    fontSize: 16,
    cursor: 'pointer',
  },
  alert: {
    margin: '10px 16px 0',
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(224,82,96,0.18)',
    border: '1px solid rgba(224,82,96,0.5)',
    color: '#ffc2c8',
    fontSize: 13,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 10,
    padding: '12px 16px',
  },
  cell: {
    background: 'rgba(255,255,255,0.04)',
    borderRadius: 8,
    padding: '8px 10px',
    fontSize: 13,
  },
  label: { fontSize: 11, color: '#8b95c8', marginBottom: 4 },
  row: { display: 'flex', alignItems: 'center', gap: 10, padding: '0 16px 8px' },
  btn: {
    background: 'rgba(108,140,255,0.18)',
    border: '1px solid rgba(108,140,255,0.4)',
    color: '#dfe5ff',
    borderRadius: 7,
    padding: '6px 12px',
    fontSize: 13,
    cursor: 'pointer',
  },
  path: { fontSize: 11, color: '#7f8ac0', wordBreak: 'break-all' },
  logLabel: { padding: '4px 16px', fontSize: 12, color: '#8b95c8' },
  errBox: {
    margin: '0 16px 8px',
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(224,82,96,0.15)',
    color: '#ffc2c8',
    fontSize: 12,
    whiteSpace: 'pre-wrap',
  },
  log: {
    margin: '0 16px 16px',
    padding: 12,
    flex: 1,
    overflow: 'auto',
    background: '#070a12',
    borderRadius: 8,
    fontSize: 11.5,
    lineHeight: 1.45,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    color: '#b9c4f0',
    whiteSpace: 'pre-wrap',
  },
};
