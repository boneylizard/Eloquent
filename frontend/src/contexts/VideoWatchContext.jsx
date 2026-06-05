import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';
import { useApp } from './AppContext';
import { Button } from '../components/ui/button';

const STORAGE_KEY = 'LiangLocal-watch-playlist-v1';
const AUDIO_PREFS_KEY = 'LiangLocal-watch-audio-v1';
const REMOTE_PREFS_KEY = 'LiangLocal-watch-remote-v1';

function readWatchPlaylistState() {
  if (typeof window === 'undefined') return { items: [], currentIndex: 0 };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { items: [], currentIndex: 0 };
    const p = JSON.parse(raw);
    const items = Array.isArray(p.items) && p.items.length ? p.items : [];
    let currentIndex =
      typeof p.currentIndex === 'number' && p.currentIndex >= 0 ? p.currentIndex : 0;
    if (items.length) currentIndex = Math.min(currentIndex, items.length - 1);
    else currentIndex = 0;
    return { items, currentIndex };
  } catch {
    return { items: [], currentIndex: 0 };
  }
}

function readAudioMuted() {
  if (typeof window === 'undefined') return true;
  try {
    const raw = localStorage.getItem(AUDIO_PREFS_KEY);
    if (!raw) return true;
    const p = JSON.parse(raw);
    if (typeof p.muted === 'boolean') return p.muted;
  } catch {
    /* ignore */
  }
  return true;
}

function readRemotePrefs() {
  if (typeof window === 'undefined') return { sessionId: '', enabled: false };
  try {
    const raw = localStorage.getItem(REMOTE_PREFS_KEY);
    if (!raw) return { sessionId: '', enabled: false };
    const p = JSON.parse(raw);
    return {
      sessionId: typeof p.sessionId === 'string' ? p.sessionId : '',
      enabled: typeof p.enabled === 'boolean' ? p.enabled : false,
    };
  } catch {
    return { sessionId: '', enabled: false };
  }
}

function makeId() {
  return `v-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

const VideoWatchContext = createContext(null);

export function useVideoWatch() {
  const ctx = useContext(VideoWatchContext);
  if (!ctx) throw new Error('useVideoWatch must be used within VideoWatchProvider');
  return ctx;
}

function MiniWatchBar() {
  const { setActiveTab } = useApp();
  const { current, goNext, goPrev, muted, setMuted } = useVideoWatch();
  if (!current) return null;
  return (
    <div className="flex items-center gap-1 px-2 py-1.5 bg-card text-xs text-foreground border-b border-border shrink-0">
      <Button type="button" variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={goPrev}>
        Prev
      </Button>
      <Button type="button" variant="outline" size="sm" className="h-7 px-2 text-xs" onClick={goNext}>
        Next
      </Button>
      <span className="truncate flex-1 min-w-0 px-1" title={current.title}>
        {current.title}
      </span>
      <Button
        type="button"
        variant={muted ? 'default' : 'outline'}
        size="sm"
        className="h-7 px-2 text-xs shrink-0"
        onClick={() => setMuted((m) => !m)}
      >
        {muted ? 'Muted' : 'Mute'}
      </Button>
      <Button type="button" size="sm" className="h-7 px-2 text-xs shrink-0" onClick={() => setActiveTab('watch')}>
        Watch
      </Button>
    </div>
  );
}

export function VideoWatchProvider({ children }) {
  const { activeTab, setActiveTab, PRIMARY_API_URL } = useApp();
  const watchInit = readWatchPlaylistState();
  const [items, setItems] = useState(watchInit.items);
  const [currentIndex, setCurrentIndex] = useState(watchInit.currentIndex);
  const [watchHostEl, setWatchHostEl] = useState(null);
  const [floatSlotEl, setFloatSlotEl] = useState(null);
  const [dockMini, setDockMini] = useState(true);
  const [muted, setMuted] = useState(readAudioMuted);
  const remoteInit = readRemotePrefs();
  const [remoteSessionId, setRemoteSessionId] = useState(remoteInit.sessionId);
  const [remoteEnabled, setRemoteEnabled] = useState(remoteInit.enabled);
  const [remoteLastId, setRemoteLastId] = useState(0);
  const [remoteLastSeenAt, setRemoteLastSeenAt] = useState(null);
  const [remoteError, setRemoteError] = useState('');
  const [portalEpoch, setPortalEpoch] = useState(0);
  const videoRef = useRef(null);
  const remoteLastIdRef = useRef(0);
  const offscreenHostRef = useRef(null);

  const bumpPortal = useCallback(() => setPortalEpoch((n) => n + 1), []);

  const current = items[currentIndex] || null;
  const currentSrc = current?.url || '';

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ items, currentIndex }));
    } catch {
      /* ignore */
    }
  }, [items, currentIndex]);

  useEffect(() => {
    try {
      localStorage.setItem(AUDIO_PREFS_KEY, JSON.stringify({ muted }));
    } catch {
      /* ignore */
    }
  }, [muted]);

  useEffect(() => {
    try {
      localStorage.setItem(
        REMOTE_PREFS_KEY,
        JSON.stringify({ sessionId: remoteSessionId, enabled: remoteEnabled })
      );
    } catch {
      /* ignore */
    }
  }, [remoteSessionId, remoteEnabled]);

  useEffect(() => {
    remoteLastIdRef.current = remoteLastId;
  }, [remoteLastId]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted, currentSrc]);

  const registerWatchHost = useCallback(
    (el) => {
      setWatchHostEl(el);
      bumpPortal();
    },
    [bumpPortal]
  );

  const registerFloatSlot = useCallback(
    (el) => {
      setFloatSlotEl(el);
      bumpPortal();
    },
    [bumpPortal]
  );

  const showFloat = activeTab !== 'watch' && dockMini && items.length > 0;

  const portalTarget = useMemo(() => {
    if (activeTab === 'watch' && watchHostEl) return watchHostEl;
    if (showFloat && floatSlotEl) return floatSlotEl;
    return offscreenHostRef.current;
  }, [activeTab, watchHostEl, showFloat, floatSlotEl, portalEpoch]);

  const goNext = useCallback(() => {
    setCurrentIndex((i) => (items.length ? (i + 1) % items.length : 0));
  }, [items.length]);

  const goPrev = useCallback(() => {
    setCurrentIndex((i) => (items.length ? (i - 1 + items.length) % items.length : 0));
  }, [items.length]);

  const addItem = useCallback((url, title) => {
    const u = (url || '').trim();
    if (!u) return;
    const t = (title || '').trim() || u.split('/').pop() || 'Video';
    setItems((prev) => [...prev, { id: makeId(), url: u, title: t }]);
  }, []);

  const removeItem = useCallback((id) => {
    setItems((prev) => {
      const idx = prev.findIndex((x) => x.id === id);
      const next = prev.filter((x) => x.id !== id);
      setCurrentIndex((ci) => {
        if (!next.length) return 0;
        if (idx < 0) return Math.min(ci, next.length - 1);
        if (ci === idx) return Math.min(idx, next.length - 1);
        if (ci > idx) return ci - 1;
        return ci;
      });
      return next;
    });
  }, []);

  const replacePlaylist = useCallback((nextItems, startIndex = 0) => {
    setItems(nextItems);
    setCurrentIndex(
      nextItems.length ? Math.min(Math.max(0, startIndex), nextItems.length - 1) : 0
    );
  }, []);

  const playIndex = useCallback((idx) => {
    setItems((prev) => {
      if (!prev.length) return prev;
      const i = ((idx % prev.length) + prev.length) % prev.length;
      setCurrentIndex(i);
      return prev;
    });
  }, []);

  const requestFullscreen = useCallback(() => {
    const el = videoRef.current;
    if (!el) return;
    const req = el.requestFullscreen || el.webkitRequestFullscreen;
    if (typeof req === 'function') req.call(el);
  }, []);

  const requestPip = useCallback(async () => {
    const el = videoRef.current;
    if (!el || !document.pictureInPictureEnabled) return;
    try {
      if (document.pictureInPictureElement === el) await document.exitPictureInPicture();
      else await el.requestPictureInPicture();
    } catch {
      /* ignore */
    }
  }, []);

  const applyRemoteCommand = useCallback(
    (cmd, payload = {}) => {
      const v = videoRef.current;
      const dispatchHotkey = (which) => {
        const map =
          which === 'x'
            ? { code: 'KeyX', key: 'x' }
            : which === 'v'
              ? { code: 'KeyV', key: 'v' }
              : { code: 'KeyX', key: 'x' };
        window.dispatchEvent(
          new KeyboardEvent('keydown', {
            ...map,
            ctrlKey: true,
            altKey: true,
            bubbles: true,
          })
        );
      };
      const fireHandsFree = (action) => {
        window.dispatchEvent(
          new CustomEvent('eloquent-remote', { detail: { action }, bubbles: true })
        );
      };
      switch (cmd) {
        case 'next':
          goNext();
          break;
        case 'prev':
          goPrev();
          break;
        case 'play':
          v?.play?.().catch(() => {});
          break;
        case 'pause':
          v?.pause?.();
          break;
        case 'toggle_play':
          if (!v) break;
          if (v.paused) v.play?.().catch(() => {});
          else v.pause?.();
          break;
        case 'mute_on':
          setMuted(true);
          break;
        case 'mute_off':
          setMuted(false);
          break;
        case 'mute_toggle':
          setMuted((m) => !m);
          break;
        case 'volume_up':
          if (v) v.volume = Math.min(1, Number(v.volume || 0) + 0.1);
          break;
        case 'volume_down':
          if (v) v.volume = Math.max(0, Number(v.volume || 0) - 0.1);
          break;
        case 'seek':
          if (v && Number.isFinite(Number(payload.seconds))) {
            const s = Number(payload.seconds);
            if (Number.isFinite(v.duration)) v.currentTime = Math.max(0, Math.min(v.duration, s));
            else v.currentTime = Math.max(0, s);
          }
          break;
        case 'fullscreen':
          requestFullscreen();
          break;
        case 'pip':
          requestPip();
          break;
        case 'watch_tab':
          setActiveTab('watch');
          break;
        case 'chat_tab':
          setActiveTab('chat');
          break;
        case 'ai_mic_start':
          setActiveTab('chat');
          setTimeout(() => fireHandsFree('mic_start'), 50);
          break;
        case 'ai_mic_stop':
          setActiveTab('chat');
          setTimeout(() => fireHandsFree('mic_stop'), 50);
          break;
        case 'ai_mic_toggle':
          setActiveTab('chat');
          setTimeout(() => fireHandsFree('mic_toggle'), 50);
          break;
        case 'ai_stop':
          setActiveTab('chat');
          setTimeout(() => dispatchHotkey('x'), 50);
          break;
        case 'ai_fast_queue':
          setActiveTab('chat');
          setTimeout(() => dispatchHotkey('v'), 50);
          break;
        case 'character_voice': {
          const voiceId = String(payload?.voice_id ?? payload?.voiceId ?? '').trim();
          if (!voiceId) break;
          const characterId = String(
            payload?.character_id ?? payload?.characterId ?? ''
          ).trim();
          setActiveTab('chat');
          setTimeout(() => {
            window.dispatchEvent(
              new CustomEvent('eloquent-remote', {
                detail: {
                  action: 'set_voice',
                  voiceId,
                  ...(characterId ? { characterId } : {}),
                },
                bubbles: true,
              })
            );
          }, 50);
          break;
        }
        case 'settings_open':
        case 'settings_tab':
          window.dispatchEvent(
            new CustomEvent('eloquent-app-command', {
              detail: {
                type: 'open_settings',
                tab: String(payload.tab || payload.section || 'general'),
              },
            })
          );
          break;
        case 'settings_patch': {
          const patch = payload.patch && typeof payload.patch === 'object' && !Array.isArray(payload.patch)
            ? payload.patch
            : null;
          if (patch) {
            window.dispatchEvent(
              new CustomEvent('eloquent-app-command', { detail: { type: 'settings_patch', patch } })
            );
          }
          break;
        }
        case 'app_tab':
        case 'desktop_tab': {
          const tab = String(payload.tab || '').trim();
          if (!tab) break;
          if (tab === 'settings') {
            window.dispatchEvent(
              new CustomEvent('eloquent-app-command', {
                detail: {
                  type: 'open_settings',
                  tab: String(payload.settings_tab || payload.settingsTab || 'general'),
                },
              })
            );
          } else {
            window.dispatchEvent(
              new CustomEvent('eloquent-app-command', { detail: { type: 'navigate_tab', tab } })
            );
          }
          break;
        }
        case 'theme_toggle':
          window.dispatchEvent(new CustomEvent('eloquent-app-command', { detail: { type: 'theme_toggle' } }));
          break;
        default:
          break;
      }
    },
    [goNext, goPrev, requestFullscreen, requestPip, setActiveTab]
  );

  useEffect(() => {
    if (!remoteEnabled || !remoteSessionId || !PRIMARY_API_URL) return undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const params = new URLSearchParams({
          session_id: remoteSessionId.trim(),
          after_id: String(remoteLastIdRef.current || 0),
        });
        const res = await fetch(`${PRIMARY_API_URL}/remote/v1/commands?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const cmds = Array.isArray(data?.commands) ? data.commands : [];
        for (const c of cmds) applyRemoteCommand(String(c.command || ''), c.payload || {});
        if (!cancelled) {
          if (typeof data?.last_id === 'number') setRemoteLastId(data.last_id);
          if (cmds.length > 0) setRemoteLastSeenAt(new Date().toISOString());
          setRemoteError('');
        }
      } catch (e) {
        if (!cancelled) setRemoteError(String(e?.message || e));
      }
    };
    tick();
    const id = setInterval(tick, 900);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [PRIMARY_API_URL, remoteEnabled, remoteSessionId, applyRemoteCommand]);

  const value = useMemo(
    () => ({
      items,
      currentIndex,
      current,
      currentSrc,
      dockMini,
      setDockMini,
      muted,
      setMuted,
      remoteSessionId,
      setRemoteSessionId,
      remoteEnabled,
      setRemoteEnabled,
      remoteLastSeenAt,
      remoteError,
      registerWatchHost,
      goNext,
      goPrev,
      addItem,
      removeItem,
      replacePlaylist,
      playIndex,
      videoRef,
      requestFullscreen,
      requestPip,
    }),
    [
      items,
      currentIndex,
      current,
      currentSrc,
      dockMini,
      registerWatchHost,
      muted,
      remoteSessionId,
      remoteEnabled,
      remoteLastSeenAt,
      remoteError,
      goNext,
      goPrev,
      addItem,
      removeItem,
      replacePlaylist,
      playIndex,
      requestFullscreen,
      requestPip,
    ]
  );

  const inFloat = showFloat && portalTarget === floatSlotEl;

  return (
    <VideoWatchContext.Provider value={value}>
      {children}
      <div
        ref={offscreenHostRef}
        className="fixed pointer-events-none opacity-0 w-px h-px overflow-hidden -z-10 bottom-0 right-0"
        aria-hidden
      />
      <div
        className={
          showFloat
            ? 'fixed bottom-4 right-4 z-[80] w-[min(100vw-2rem,400px)] rounded-lg border border-border bg-black shadow-xl overflow-hidden flex flex-col max-h-[45vh]'
            : 'hidden'
        }
        aria-hidden={!showFloat}
      >
        {showFloat ? <MiniWatchBar /> : null}
        {showFloat ? (
          <div
            ref={registerFloatSlot}
            className="relative flex-1 min-h-[140px] w-full bg-black"
          />
        ) : null}
      </div>
      {portalTarget
        ? createPortal(
            <video
              ref={videoRef}
              src={currentSrc || undefined}
              className={
                inFloat
                  ? 'absolute inset-0 w-full h-full object-contain'
                  : 'w-full h-full max-h-[min(70vh,720px)] bg-black object-contain'
              }
              controls
              playsInline
              muted={muted}
              onEnded={goNext}
            />,
            portalTarget
          )
        : null}
    </VideoWatchContext.Provider>
  );
}
