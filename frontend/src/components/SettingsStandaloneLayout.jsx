import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { useTheme } from './ThemeProvider';
import Settings from './Settings';
import { Button } from './ui/button';
import { RefreshCw } from 'lucide-react';
import { requestMainWindowReload } from '../utils/settingsCrossWindowSync';
import { useAppBoot } from '../hooks/useAppBoot';
import InfrastructureBanner from './InfrastructureBanner';

export default function SettingsStandaloneLayout() {
  const [searchParams] = useSearchParams();
  const { theme, setTheme } = useTheme();
  const { banner, retry } = useAppBoot();
  const initialTab = searchParams.get('tab') || 'general';

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-20 border-b border-border/70 bg-card/95 backdrop-blur-sm px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground">Eloquent</p>
          <h1 className="text-lg font-semibold">Settings</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Changes sync with the main window automatically. Reload main if something looks stale.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => requestMainWindowReload()}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Reload main window
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => window.close()}
          >
            Close
          </Button>
        </div>
      </header>
      <main className="p-4 max-w-6xl mx-auto">
        <InfrastructureBanner banner={banner} onRetry={retry} />
        <Settings
          darkMode={theme === 'dark'}
          toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          initialTab={initialTab}
          isStandaloneWindow
        />
      </main>
    </div>
  );
}
