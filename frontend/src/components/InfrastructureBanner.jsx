import React from 'react';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Button } from './ui/button';
import { Loader2, RefreshCw, AlertTriangle } from 'lucide-react';

/**
 * Shown when port config or storage hydration is slow/degraded — never a silent infinite spinner.
 */
export default function InfrastructureBanner({ banner, onRetry, className = '' }) {
  if (!banner) return null;
  const isLoading = banner.tone === 'loading';

  return (
    <Alert
      className={[
        'mb-4',
        isLoading ? 'border-border/70 bg-muted/30' : 'border-amber-500/40 bg-amber-500/10',
        className,
      ].join(' ')}
    >
      {isLoading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
      )}
      <AlertTitle className="text-sm">
        {isLoading ? 'Starting up' : 'Limited mode'}
      </AlertTitle>
      <AlertDescription className="flex flex-wrap items-center gap-3 text-sm mt-1">
        <span className="flex-1 min-w-[200px]">{banner.message}</span>
        {onRetry && !isLoading ? (
          <Button type="button" variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
            Retry
          </Button>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}
