import React, { useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useBackendStatus } from '../hooks/useBackendStatus';
import FirstRunPurpose from './FirstRunPurpose';

const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes)) return '';
  if (bytes >= 1024 ** 3) return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / (1024 ** 2)).toFixed(0)} MB`;
  return `${Math.max(0, bytes / 1024).toFixed(0)} KB`;
};

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds)) return '';
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} sec`;
  return `${Math.ceil(seconds / 60)} min`;
};

export default function BackendGate({ children }) {
  const [boot, setBoot] = useState(null);
  const [ready, setReady] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [showHelp, setShowHelp] = useState(false);
  const [logTail, setLogTail] = useState('');
  const conn = useBackendStatus();
  const startRef = useRef(Date.now());
  const previewFirstRun = import.meta.env.DEV
    && typeof window !== 'undefined'
    && new URLSearchParams(window.location.search).get('preview') === 'first-run-purpose';

  useEffect(() => {
    if (typeof window === 'undefined' || !window.__TAURI_INTERNALS__) {
      setReady(true);
      return undefined;
    }
    let unlisten = null;
    let cancelled = false;
    void (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        const applyBootProgress = (payload) => {
          if (!cancelled && payload) setBoot(payload);
        };
        unlisten = await listen('runtime-boot', (event) => applyBootProgress(event.payload));
        applyBootProgress(await invoke('get_runtime_boot_status'));
      } catch {
      }
    })();
    const timer = window.setInterval(() => {
      const seconds = Math.floor((Date.now() - startRef.current) / 1000);
      setElapsed(seconds);
      if (seconds > 120) setShowHelp(true);
    }, 1000);
    return () => {
      cancelled = true;
      unlisten?.();
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (conn.backend === 'online') setReady(true);
  }, [conn.backend]);

  useEffect(() => {
    if (ready || typeof window === 'undefined' || !window.__TAURI_INTERNALS__) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const tail = await invoke('read_log_tail', { lines: 22 });
        if (!cancelled) setLogTail(tail || '');
      } catch {
      }
    };
    void poll();
    const timer = window.setInterval(poll, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [ready]);

  if (previewFirstRun) return <FirstRunPurpose />;
  if (ready) return children;

  const stage = boot?.stage;
  if (stage === 'awaiting_setup') {
    return (
      <FirstRunPurpose
        onBegin={() => setBoot({
          stage: 'starting',
          message: 'Preparing the verified runtime download.',
          percent: 0,
        })}
      />
    );
  }
  const isDownload = stage === 'download';
  const isExtract = stage === 'extract';
  const isError = stage === 'error';
  const percent = typeof boot?.percent === 'number' ? boot.percent : 0;

  let title = 'Starting Mirid';
  let subtitle = 'Preparing local services.';
  if (isDownload) {
    title = 'First-run setup';
    subtitle = 'Downloading Mirid’s local engine.';
  } else if (isExtract) {
    title = 'First-run setup';
    subtitle = 'Installing Mirid’s local engine.';
  } else if (stage === 'starting' || stage === 'done') {
    subtitle = boot?.message || 'Starting the local engine.';
  } else if (conn.backend === 'connecting') {
    subtitle = 'Starting the local engine.';
  } else if (conn.backend === 'offline') {
    subtitle = 'The local engine is not reachable yet.';
  }

  const downloadStatus = isDownload && Number.isFinite(boot?.downloaded_bytes)
    ? [
        `${formatBytes(boot.downloaded_bytes)} of ${formatBytes(boot.total_bytes)}`,
        Number.isFinite(boot.bytes_per_second) ? `${formatBytes(boot.bytes_per_second)}/s` : '',
        Number.isFinite(boot.eta_seconds) ? `about ${formatDuration(boot.eta_seconds)} remaining` : '',
      ].filter(Boolean).join(' · ')
    : '';
  const progressMessage = isDownload
    ? (downloadStatus || `${boot?.message || 'Downloading'} · ${percent}%`)
    : isExtract
      ? `${boot?.message || 'Installing'} · ${percent}% · ${elapsed}s elapsed`
      : `Local services are starting · ${elapsed}s`;
  const seenLogs = (logTail || '')
    .split('\n')
    .filter((line) => /backend:|tts:|Uvicorn|Application startup|ERROR|Error|Traceback|ImportError|ModuleNotFound/.test(line))
    .slice(-12);

  return (
    <div style={styles.backdrop}>
      <div style={styles.card}>
        <div style={styles.brand}>Mirid</div>
        <div style={styles.title}>{title}</div>
        <div style={styles.subtitle}>{subtitle}</div>

        {(isDownload || isExtract) ? (
          <div style={styles.track} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
            <div style={{ ...styles.fill, width: `${percent}%` }} />
          </div>
        ) : (
          <div style={styles.spinner} aria-label="Loading" />
        )}

        <div style={styles.message}>{progressMessage}</div>

        {(isDownload || isExtract) && (
          <div style={styles.contentsHint}>
            This installs Mirid’s engine, voice and image support. Chat models are separate. The download is about 3.3 GB and uses about 9 GB after extraction.
          </div>
        )}

        {isExtract && (
          <div style={styles.extractHint}>
            Keep Mirid open. A single large file can pause the file count while the percentage and elapsed time continue.
          </div>
        )}

        {seenLogs.length > 0 && <pre style={styles.log}>{seenLogs.join('\n')}</pre>}

        {isError && (
          <div style={styles.error}>
            Setup failed. {boot?.message}
            <div style={{ marginTop: 6 }}>Press <b>Ctrl+Shift+D</b> to read the full log.</div>
          </div>
        )}

        {showHelp && !isError && (
          <div style={styles.help}>
            First startup can take several minutes. Press <b>Ctrl+Shift+D</b> if the percentage and elapsed time both stop changing.
          </div>
        )}
      </div>
      <style>{`@keyframes mirid-spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    zIndex: 2147482999,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'radial-gradient(circle at 50% 30%, #1a1f33 0%, #0b0e1a 100%)',
    fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    color: '#e8ecff',
  },
  card: {
    width: 'min(480px, 88vw)',
    padding: '32px 34px',
    borderRadius: 16,
    background: 'rgba(20, 24, 40, 0.76)',
    border: '1px solid rgba(120, 140, 255, 0.18)',
    boxShadow: '0 20px 60px rgba(0,0,0,0.45)',
    backdropFilter: 'blur(10px)',
    textAlign: 'center',
  },
  brand: { fontSize: 26, fontWeight: 700, letterSpacing: '0.5px' },
  title: { marginTop: 4, fontSize: 14, fontWeight: 600 },
  subtitle: { marginTop: 3, marginBottom: 20, fontSize: 13, color: '#9aa6d6' },
  track: { width: '100%', height: 10, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden', marginBottom: 12 },
  fill: { height: '100%', borderRadius: 999, background: '#6c8cff', transition: 'width 0.25s ease' },
  spinner: { width: 34, height: 34, margin: '4px auto 14px', borderRadius: '50%', border: '3px solid rgba(255,255,255,0.12)', borderTopColor: '#6c8cff', animation: 'mirid-spin 0.9s linear infinite' },
  message: { minHeight: 18, fontSize: 13, color: '#c4ccf0' },
  contentsHint: { marginTop: 12, fontSize: 11.5, lineHeight: 1.45, color: '#aeb8e2' },
  extractHint: { marginTop: 10, fontSize: 11.5, lineHeight: 1.45, color: '#9aa6d6' },
  log: { marginTop: 14, maxHeight: 180, overflow: 'auto', width: '100%', textAlign: 'left', background: '#070a12', borderRadius: 8, padding: '8px 10px', fontSize: 10.5, lineHeight: 1.4, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace', color: '#b9c4f0', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  error: { marginTop: 12, fontSize: 12, color: '#ffc2c8', background: 'rgba(224,82,96,0.15)', borderRadius: 8, padding: '8px 12px', textAlign: 'left' },
  help: { marginTop: 14, fontSize: 12, color: '#7f8ac0', lineHeight: 1.5, textAlign: 'left' },
};
