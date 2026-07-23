import React, { useCallback, useEffect, useRef, useState } from 'react';
import { DownloadCloud, Loader2, RefreshCw } from 'lucide-react';

import { Button } from './ui/button';
import {
  checkForAppUpdate,
  formatUpdateProgress,
  formatUpdaterError,
  installAppUpdate,
  isAppUpdaterAvailable,
} from '../utils/appUpdater';

export default function AppUpdateControls() {
  const updateRef = useRef(null);
  const installingRef = useRef(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState(null);
  const [availableVersion, setAvailableVersion] = useState('');
  const updaterAvailable = isAppUpdaterAvailable();

  useEffect(() => () => {
    if (!installingRef.current) void updateRef.current?.close();
  }, []);

  const handleCheck = useCallback(async () => {
    setChecking(true);
    setError('');
    setStatus('');
    setAvailableVersion('');
    setProgress(null);
    try {
      void updateRef.current?.close();
      const update = await checkForAppUpdate();
      updateRef.current = update;
      if (!update) {
        setStatus('Mirid is up to date.');
        return;
      }
      setAvailableVersion(update.version);
      setStatus(`Mirid ${update.version} is ready.`);
    } catch (updateError) {
      setError(formatUpdaterError(updateError, 'check for updates'));
    } finally {
      setChecking(false);
    }
  }, []);

  const handleInstall = useCallback(async () => {
    if (!updateRef.current) return;
    installingRef.current = true;
    setInstalling(true);
    setError('');
    setStatus('Downloading the update…');
    try {
      await installAppUpdate(updateRef.current, (nextProgress) => {
        setProgress(nextProgress);
        if (nextProgress.phase === 'installing') {
          setStatus('Installing the update. Mirid will restart when it is ready.');
        }
      });
    } catch (updateError) {
      setError(formatUpdaterError(updateError));
      setStatus('');
      installingRef.current = false;
      setInstalling(false);
    }
  }, []);

  if (!updaterAvailable) {
    return (
      <p className="text-sm text-muted-foreground">
        Automatic updates are available in the installed Mirid desktop app.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row">
        <Button
          variant="outline"
          onClick={handleCheck}
          disabled={checking || installing}
        >
          {checking ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-4 w-4" />
          )}
          Check for updates
        </Button>
        {availableVersion && (
          <Button onClick={handleInstall} disabled={installing || checking}>
            {installing ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <DownloadCloud className="mr-2 h-4 w-4" />
            )}
            {installing ? 'Updating Mirid…' : `Update to ${availableVersion}`}
          </Button>
        )}
      </div>

      {status && <p className="text-sm text-foreground">{status}</p>}
      {progress && (
        <p className="text-sm tabular-nums text-muted-foreground">
          {formatUpdateProgress(progress)}
        </p>
      )}
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
