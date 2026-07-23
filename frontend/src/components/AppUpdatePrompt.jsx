import React, { useCallback, useEffect, useState } from 'react';
import { DownloadCloud, Loader2 } from 'lucide-react';

import { Button } from './ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import {
  checkForAppUpdate,
  formatUpdateProgress,
  formatUpdaterError,
  installAppUpdate,
  isAppUpdaterAvailable,
} from '../utils/appUpdater';

const AUTOMATIC_CHECK_DELAY_MS = 8_000;

export default function AppUpdatePrompt({ enabled }) {
  const [update, setUpdate] = useState(null);
  const [open, setOpen] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!enabled || !isAppUpdaterAvailable()) return undefined;

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const nextUpdate = await checkForAppUpdate();
        if (cancelled || !nextUpdate) {
          void nextUpdate?.close();
          return;
        }
        setUpdate(nextUpdate);
        setOpen(true);
      } catch (updateError) {
        console.warn('Automatic update check failed:', updateError);
      }
    }, AUTOMATIC_CHECK_DELAY_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [enabled]);

  const dismiss = useCallback(() => {
    if (installing) return;
    setOpen(false);
    void update?.close();
    setUpdate(null);
    setProgress(null);
    setError('');
  }, [installing, update]);

  const install = useCallback(async () => {
    if (!update) return;
    setInstalling(true);
    setError('');
    try {
      await installAppUpdate(update, setProgress);
    } catch (updateError) {
      setError(formatUpdaterError(updateError));
      setInstalling(false);
    }
  }, [update]);

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) dismiss();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Mirid {update?.version || ''} is ready</DialogTitle>
          <DialogDescription>
            Mirid will download the signed update and install it for you. Your
            settings, conversations, models and local runtime stay in place.
          </DialogDescription>
        </DialogHeader>

        {update?.body && (
          <div className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded-md border border-border/60 bg-muted/30 p-3 text-sm text-muted-foreground">
            {update.body}
          </div>
        )}

        {progress && (
          <p className="text-sm tabular-nums text-muted-foreground">
            {formatUpdateProgress(progress)}
          </p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}

        <DialogFooter>
          <Button variant="outline" onClick={dismiss} disabled={installing}>
            Later
          </Button>
          <Button onClick={install} disabled={installing}>
            {installing ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <DownloadCloud className="mr-2 h-4 w-4" />
            )}
            {installing ? 'Updating Mirid…' : 'Download and install'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
