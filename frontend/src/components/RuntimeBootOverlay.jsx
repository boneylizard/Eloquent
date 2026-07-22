import React, { useEffect, useState } from 'react';

/**
 * Full-screen overlay shown on first launch while the Tauri host downloads and
 * extracts the ML runtime from HuggingFace. Listens to the Rust-emitted
 * `runtime-boot` event ({ stage, message, percent, bytes_per_second, eta_seconds }). Renders nothing outside
 * of the Tauri desktop shell (i.e. plain browser use is unaffected).
 */
export default function RuntimeBootOverlay() {
  const [state, setState] = useState(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Only run inside the Tauri webview.
    if (typeof window === 'undefined' || !window.__TAURI_INTERNALS__) return;

    let unlisten = null;
    let cancelled = false;

    (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        unlisten = await listen('runtime-boot', (event) => {
          if (cancelled) return;
          const payload = event?.payload || {};
          setState({
            stage: payload.stage || '',
            message: payload.message || '',
            percent: typeof payload.percent === 'number' ? payload.percent : 0,
            downloadedBytes: typeof payload.downloaded_bytes === 'number' ? payload.downloaded_bytes : null,
            totalBytes: typeof payload.total_bytes === 'number' ? payload.total_bytes : null,
            bytesPerSecond: typeof payload.bytes_per_second === 'number' ? payload.bytes_per_second : null,
            etaSeconds: typeof payload.eta_seconds === 'number' ? payload.eta_seconds : null,
          });

          // Show the overlay for anything that isn't an instant "already ready".
          if (payload.stage === 'done') {
            // Backend is starting; give sidecars a moment then dismiss.
            setTimeout(() => !cancelled && setVisible(false), 1200);
          } else if (payload.stage === 'ready' && payload.percent === 100) {
            // Runtime already installed on a normal launch: don't flash the UI.
            // Only keep hidden if we never showed a download stage.
            setVisible((prev) => prev);
          } else {
            setVisible(true);
          }
        });
      } catch (err) {
        // If the event API isn't available, silently skip the overlay.
        console.warn('runtime-boot listener failed', err);
      }
    })();

    return () => {
      cancelled = true;
      if (typeof unlisten === 'function') unlisten();
    };
  }, []);

  if (!visible || !state) return null;

  const isError = state.stage === 'error';
  const isIndeterminate = state.stage === 'verify' || state.stage === 'starting';
  const pct = Math.max(0, Math.min(100, state.percent || 0));
  const speed = formatDownloadSpeed(state.bytesPerSecond);
  const eta = formatEta(state.etaSeconds);

  const stageLabel = {
    download: 'Downloading the runtime',
    verify: 'Verifying the download',
    extract: 'Installing the runtime',
    starting: 'Starting services',
    ready: 'Ready',
    done: 'Ready',
    error: 'Stopped',
  }[state.stage] || 'Preparing Mirid';

  return (
    <div style={styles.backdrop}>
      <div style={styles.card}>
        <div style={styles.title}>Mirid</div>
        <div style={styles.subtitle}>
          {isError ? 'Setup failed' : 'First-run setup'}
        </div>

        <div style={styles.stage}>{stageLabel}</div>

        <div style={styles.track}>
          <div
            style={{
              ...styles.fill,
              width: isIndeterminate ? '100%' : `${pct}%`,
              opacity: isIndeterminate ? 0.5 : 1,
              background: isError ? '#e05260' : '#6c8cff',
              animation: isIndeterminate ? 'eloquent-pulse 1.4s ease-in-out infinite' : 'none',
            }}
          />
        </div>

        <div style={styles.message}>
          {state.message}
          {!isIndeterminate && !isError ? `  (${pct}%)` : ''}
        </div>

        {state.stage === 'download' && speed && (
          <div style={styles.transfer}>
            {speed}{eta ? ` · ${eta} remaining` : ''}
          </div>
        )}

        {state.stage === 'download' && (
          <div style={styles.hint}>
            One-time download, roughly 3.1&nbsp;GB. Only on first launch.
          </div>
        )}
        {isError && (
          <div style={styles.hint}>
            Check your connection, then relaunch Mirid.
          </div>
        )}
      </div>

      <style>{`
        @keyframes eloquent-pulse {
          0% { transform: translateX(-30%); }
          50% { transform: translateX(0%); }
          100% { transform: translateX(30%); }
        }
      `}</style>
    </div>
  );
}

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    zIndex: 2147483000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'radial-gradient(circle at 50% 30%, #1a1f33 0%, #0b0e1a 100%)',
    fontFamily:
      'Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    color: '#e8ecff',
  },
  card: {
    width: 'min(460px, 86vw)',
    padding: '32px 34px',
    borderRadius: 16,
    background: 'rgba(20, 24, 40, 0.72)',
    border: '1px solid rgba(120, 140, 255, 0.18)',
    boxShadow: '0 20px 60px rgba(0,0,0,0.45)',
    backdropFilter: 'blur(10px)',
    textAlign: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: 700,
    letterSpacing: '0.5px',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
    color: '#9aa6d6',
    marginBottom: 22,
  },
  stage: {
    fontSize: 15,
    fontWeight: 600,
    marginBottom: 12,
  },
  track: {
    width: '100%',
    height: 10,
    borderRadius: 999,
    background: 'rgba(255,255,255,0.08)',
    overflow: 'hidden',
    marginBottom: 12,
  },
  fill: {
    height: '100%',
    borderRadius: 999,
    transition: 'width 0.25s ease',
  },
  message: {
    fontSize: 13,
    color: '#c4ccf0',
    minHeight: 18,
  },
  transfer: {
    marginTop: 6,
    fontSize: 13,
    fontVariantNumeric: 'tabular-nums',
    color: '#aeb9e8',
  },
  hint: {
    marginTop: 14,
    fontSize: 12,
    color: '#7f8ac0',
  },
};

function formatDownloadSpeed(bytesPerSecond) {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond < 0) return '';
  if (bytesPerSecond >= 1_000_000) {
    const megabytes = bytesPerSecond / 1_000_000;
    return `${megabytes >= 10 ? megabytes.toFixed(0) : megabytes.toFixed(1)} MB/s`;
  }
  return `${Math.round(bytesPerSecond / 1_000)} KB/s`;
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '';
  if (seconds < 60) return '<1 min';
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0 ? `${hours} hr ${remainingMinutes} min` : `${hours} hr`;
}
