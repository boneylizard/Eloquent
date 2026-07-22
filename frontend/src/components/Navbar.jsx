import React, { useEffect, useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { useTheme } from './ThemeProvider';
import { Button } from './ui/button';
import {
  Settings,
  Palette,
  Power,
  RotateCw,
  MoreVertical,
  Zap,
  Vote,
  Menu,
  Pin,
  PinOff,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { isModuleEnabled } from '../config/modules';
import { restartMirid, shutdownMirid } from '../utils/desktopLifecycle';
const EXTRA_TOOLS = [
  { id: 'modeltester', label: 'Model Tester', icon: Zap },
  ...(isModuleEnabled('elections') ? [{ id: 'election', label: 'Elections', icon: Vote }] : []),
];

const NAVBAR_HEIGHT_CLASS = 'h-12';

const Navbar = ({
  toggleSidebar,
  collapsed = false,
  pinned = false,
  onTogglePin,
  reduceMotion = false,
}) => {
  const [overflowOpen, setOverflowOpen] = useState(false);
  const [lifecycleAction, setLifecycleAction] = useState(null);

  const {
    setActiveTab,
    openSettingsTab,
  } = useApp();

  const { theme, setTheme } = useTheme();

  useEffect(() => {
    if (!overflowOpen) return;
    const onDoc = (e) => {
      if (!e.target.closest?.('[data-navbar-overflow]')) setOverflowOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [overflowOpen]);

  const handleRestart = async () => {
    if (!confirm('Restart Mirid? Any reply still being generated will stop.')) return;
    setLifecycleAction('restart');
    try {
      await restartMirid();
    } catch (error) {
      setLifecycleAction(null);
      alert(`Mirid could not restart. ${String(error)}`);
    }
  };

  const handleShutdown = async () => {
    if (!confirm('Shut down Mirid? Local services will stop and the window will close.')) return;
    setLifecycleAction('shutdown');
    try {
      await shutdownMirid();
    } catch (error) {
      setLifecycleAction(null);
      alert(`Mirid could not shut down. ${String(error)}`);
    }
  };

  const isNanoGpt = theme === 'nanogpt';

  return (
    <header
      className={cn(
        'fixed top-0 left-0 right-0 z-[60] border-b',
        NAVBAR_HEIGHT_CLASS,
        'bg-card/95 backdrop-blur-md supports-[backdrop-filter]:bg-card/90 shadow-sm',
        isNanoGpt &&
          'bg-background/95 border-border shadow-[0_8px_30px_rgba(8,6,18,0.45)]',
        !reduceMotion && 'transition-transform duration-300 ease-in-out',
        collapsed && '-translate-y-full pointer-events-none',
      )}
      aria-hidden={collapsed}
    >
      <div className={cn('flex items-center px-3 gap-2 min-w-0', NAVBAR_HEIGHT_CLASS)}>
        {/* Mobile sidebar */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden flex-shrink-0 h-9 w-9"
          onClick={toggleSidebar}
          title="Open navigation"
        >
          <Menu className="h-5 w-5" />
        </Button>

        {/* Brand */}
        <div className="flex items-center gap-2 flex-shrink-0 min-w-0">
          <img src="/eloquent_logo.png" alt="Mirid" className="h-8 w-8 dark:brightness-0 dark:invert" />
          <span
            className="font-bold text-lg hidden sm:inline truncate"
            style={{ fontFamily: 'IBM Plex Mono, monospace; font-weight: 500; letter-spacing: 0.22em; font-size: 0.72rem; text-transform: uppercase' }}
          >
            Mirid
          </span>
        </div>

        <div className="flex-1 min-w-0" />

        {onTogglePin && (
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              'h-9 w-9 flex-shrink-0',
              pinned && 'text-primary bg-primary/10',
            )}
            title={
              pinned
                ? 'Navbar pinned — unpin to auto-hide on scroll'
                : 'Pin navbar (stays visible while scrolling)'
            }
            onClick={onTogglePin}
            aria-pressed={pinned}
          >
            {pinned ? <Pin className="h-4 w-4" /> : <PinOff className="h-4 w-4" />}
          </Button>
        )}

        {/* Overflow ⋮ */}
        <div className="relative flex-shrink-0" data-navbar-overflow>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            title="More"
            onClick={() => setOverflowOpen((v) => !v)}
          >
            <MoreVertical className="h-5 w-5" />
          </Button>

          {overflowOpen && (
            <div className="absolute right-0 top-10 w-56 rounded-lg border bg-card shadow-lg py-1 z-50 animate-in fade-in slide-in-from-top-1">
              <button
                type="button"
                className="w-full px-3 py-2 text-left text-sm hover:bg-muted flex items-center gap-2"
                onClick={(e) => {
                  openSettingsTab('general', { forceWindow: e.shiftKey ? true : undefined });
                  setOverflowOpen(false);
                }}
              >
                <Settings className="h-4 w-4" />
                Settings
                <span className="ml-auto text-[10px] text-muted-foreground">⇧ new window</span>
              </button>

              <div className="px-3 py-2 border-t border-b">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                  <Palette className="h-3.5 w-3.5" />
                  Theme
                </div>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="w-full text-sm rounded border border-input bg-background px-2 py-1"
                >
                  <optgroup label="Base">
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="nanogpt">NanoGPT</option>
                  </optgroup>
                  <optgroup label="Chat">
                    <option value="faraday">Character Room</option>
                    <option value="whatsapp">WhatsApp</option>
                    <option value="messenger">Messenger</option>
                    <option value="claude">Claude</option>
                  </optgroup>
                  <optgroup label="Vibrant">
                    <option value="cyberpunk">Cyberpunk</option>
                    <option value="fallout">Fallout Terminal</option>
                  </optgroup>
                </select>
              </div>

              <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                Tools
              </div>
              {EXTRA_TOOLS.map((tool) => {
                const Icon = tool.icon;
                return (
                  <button
                    key={tool.id}
                    type="button"
                    className="w-full px-3 py-2 text-left text-sm hover:bg-muted flex items-center gap-2"
                    onClick={() => {
                      setActiveTab(tool.id);
                      setOverflowOpen(false);
                    }}
                  >
                    <Icon className="h-4 w-4" />
                    {tool.label}
                  </button>
                );
              })}

              <div className="border-t mt-1 pt-1">
                <button
                  type="button"
                  className="w-full px-3 py-2 text-left text-sm hover:bg-muted flex items-center gap-2"
                  onClick={() => {
                    handleRestart();
                    setOverflowOpen(false);
                  }}
                  disabled={Boolean(lifecycleAction)}
                >
                  <RotateCw className={cn('h-4 w-4', lifecycleAction === 'restart' && 'animate-spin')} />
                  {lifecycleAction === 'restart' ? 'Restarting…' : 'Restart'}
                </button>
                <button
                  type="button"
                  className="w-full px-3 py-2 text-left text-sm hover:bg-muted flex items-center gap-2 text-destructive"
                  onClick={() => {
                    handleShutdown();
                    setOverflowOpen(false);
                  }}
                  disabled={Boolean(lifecycleAction)}
                >
                  <Power className="h-4 w-4" />
                  {lifecycleAction === 'shutdown' ? 'Shutting down…' : 'Shut down'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Navbar;
