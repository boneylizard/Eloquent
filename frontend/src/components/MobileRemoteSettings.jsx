import React, { useCallback } from 'react';
import { useVideoWatch } from '../contexts/VideoWatchContext';
import { Input } from './ui/input';
import { Switch } from './ui/switch';
import { Label } from './ui/label';
import { Smartphone } from 'lucide-react';

function defaultSessionId() {
  return `room-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

/**
 * Settings → General: Tailscale / LAN phone remote for Watch + Chat (remote.html).
 * Session id and enabled flag persist in localStorage via VideoWatchContext.
 */
export default function MobileRemoteSettings() {
  const {
    remoteSessionId,
    setRemoteSessionId,
    remoteEnabled,
    setRemoteEnabled,
    remoteLastSeenAt,
    remoteError,
  } = useVideoWatch();

  const remoteUrl = `${window.location.origin}/remote.html?session=${encodeURIComponent(
    remoteSessionId || ''
  )}`;

  const onRemoteEnabledChange = useCallback(
    (checked) => {
      if (checked) {
        if (!String(remoteSessionId || '').trim()) {
          setRemoteSessionId(defaultSessionId());
        }
        setRemoteEnabled(true);
      } else {
        setRemoteEnabled(false);
      }
    },
    [remoteSessionId, setRemoteSessionId, setRemoteEnabled]
  );

  return (
    <div className="rounded-2xl border border-border/70 bg-card/60 shadow-sm">
      <div className="flex flex-col gap-2 border-b border-border/60 px-5 py-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">
            Mobile remote (Tailscale)
          </p>
          <p className="text-sm text-foreground/80 mt-1">
            Listen for commands from <code className="text-xs">remote.html</code> on your phone (same
            Tailscale or LAN as this app). Session name and on/off are saved automatically in this
            browser.
          </p>
        </div>
      </div>
      <div className="p-5 space-y-3">
        <div className="space-y-2">
          <Label htmlFor="remote-session-id" className="text-xs flex items-center gap-1">
            <Smartphone className="h-3 w-3" aria-hidden />
            Session name
          </Label>
          <Input
            id="remote-session-id"
            placeholder="e.g. lounge-tv (or turn on below to auto-fill once)"
            value={remoteSessionId}
            onChange={(e) => {
              const v = e.target.value;
              setRemoteSessionId(v);
              if (!v.trim() && remoteEnabled) setRemoteEnabled(false);
            }}
            className="max-w-md"
          />
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border/60 bg-background/40 px-4 py-3">
          <div>
            <Label htmlFor="remote-enabled" className="text-sm font-semibold text-foreground">
              Listen for phone commands
            </Label>
            <p className="text-xs text-muted-foreground mt-0.5">
              Turn off when you do not need the phone remote.
            </p>
          </div>
          <Switch id="remote-enabled" checked={remoteEnabled} onCheckedChange={onRemoteEnabledChange} />
        </div>
        <div className="text-xs text-muted-foreground break-all">
          Open on phone:{' '}
          <a className="underline text-primary" href={remoteUrl} target="_blank" rel="noreferrer">
            {remoteUrl}
          </a>
        </div>
        <p className="text-xs text-muted-foreground">
          Remote: mic <strong>start</strong> / <strong>stop &amp; send</strong> (tap twice, no hold),
          stop TTS, chunk replay, plus Watch controls.
        </p>
        <p className="text-xs text-muted-foreground leading-relaxed">
          The phone page also has a <strong className="text-foreground">Desktop</strong> card: open Settings on a chosen tab, jump to Models/Characters/Documents, toggle light/dark, and send{' '}
          <code className="text-[10px]">settings_patch</code> for quick saved-setting toggles (e.g. STT/TTS).
        </p>
        {remoteLastSeenAt ? (
          <p className="text-xs text-emerald-600 dark:text-emerald-400">
            Last remote command: {new Date(remoteLastSeenAt).toLocaleTimeString()}
          </p>
        ) : null}
        {remoteError ? (
          <p className="text-xs text-destructive">Remote: {remoteError}</p>
        ) : null}
      </div>
    </div>
  );
}
