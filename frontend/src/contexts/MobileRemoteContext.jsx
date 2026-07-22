import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useApp } from './AppContext';

const REMOTE_PREFS_KEY = 'Mirid-mobile-remote-v1';
const LEGACY_REMOTE_PREFS_KEY = 'LiangLocal-watch-remote-v1';
const ALLOWED_REMOTE_TABS = new Set([
  'characters',
  'chat',
  'documents',
  'memory',
  'models',
  'modeltester',
  'pool',
  'settings',
]);

function readRemotePrefs() {
  if (typeof window === 'undefined') return { sessionId: '', enabled: false };
  try {
    const stored =
      localStorage.getItem(REMOTE_PREFS_KEY) ||
      localStorage.getItem(LEGACY_REMOTE_PREFS_KEY) ||
      '{}';
    const parsed = JSON.parse(stored);
    return {
      sessionId: typeof parsed.sessionId === 'string' ? parsed.sessionId : '',
      enabled: typeof parsed.enabled === 'boolean' ? parsed.enabled : false,
    };
  } catch {
    return { sessionId: '', enabled: false };
  }
}

const MobileRemoteContext = createContext(null);

export function useMobileRemote() {
  const context = useContext(MobileRemoteContext);
  if (!context) throw new Error('useMobileRemote must be used within MobileRemoteProvider');
  return context;
}

export function MobileRemoteProvider({ children }) {
  const { setActiveTab, PRIMARY_API_URL } = useApp();
  const initial = readRemotePrefs();
  const [remoteSessionId, setRemoteSessionId] = useState(initial.sessionId);
  const [remoteEnabled, setRemoteEnabled] = useState(initial.enabled);
  const [remoteLastId, setRemoteLastId] = useState(0);
  const [remoteLastSeenAt, setRemoteLastSeenAt] = useState(null);
  const [remoteError, setRemoteError] = useState('');
  const remoteLastIdRef = useRef(0);

  useEffect(() => {
    try {
      localStorage.setItem(
        REMOTE_PREFS_KEY,
        JSON.stringify({ sessionId: remoteSessionId, enabled: remoteEnabled })
      );
    } catch {}
  }, [remoteSessionId, remoteEnabled]);

  useEffect(() => {
    remoteLastIdRef.current = remoteLastId;
  }, [remoteLastId]);

  const applyRemoteCommand = useCallback(
    (command, payload = {}) => {
      const dispatchHotkey = (key) => {
        const mapped = key === 'v' ? { code: 'KeyV', key: 'v' } : { code: 'KeyX', key: 'x' };
        window.dispatchEvent(
          new KeyboardEvent('keydown', {
            ...mapped,
            ctrlKey: true,
            altKey: true,
            bubbles: true,
          })
        );
      };
      const dispatchChatAction = (action) => {
        window.dispatchEvent(
          new CustomEvent('eloquent-remote', { detail: { action }, bubbles: true })
        );
      };
      const openChatThen = (action) => {
        setActiveTab('chat');
        setTimeout(action, 50);
      };

      switch (command) {
        case 'chat_tab':
          setActiveTab('chat');
          break;
        case 'ai_mic_start':
          openChatThen(() => dispatchChatAction('mic_start'));
          break;
        case 'ai_mic_stop':
          openChatThen(() => dispatchChatAction('mic_stop'));
          break;
        case 'ai_mic_toggle':
          openChatThen(() => dispatchChatAction('mic_toggle'));
          break;
        case 'ai_stop':
          openChatThen(() => dispatchHotkey('x'));
          break;
        case 'ai_fast_queue':
          openChatThen(() => dispatchHotkey('v'));
          break;
        case 'character_voice': {
          const voiceId = String(payload?.voice_id ?? payload?.voiceId ?? '').trim();
          if (!voiceId) break;
          const characterId = String(payload?.character_id ?? payload?.characterId ?? '').trim();
          openChatThen(() => {
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
          });
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
          const settingsPatch =
            payload.patch && typeof payload.patch === 'object' && !Array.isArray(payload.patch)
              ? payload.patch
              : null;
          if (settingsPatch) {
            window.dispatchEvent(
              new CustomEvent('eloquent-app-command', {
                detail: { type: 'settings_patch', patch: settingsPatch },
              })
            );
          }
          break;
        }
        case 'app_tab':
        case 'desktop_tab': {
          const tab = String(payload.tab || '').trim();
          if (!ALLOWED_REMOTE_TABS.has(tab)) break;
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
          window.dispatchEvent(
            new CustomEvent('eloquent-app-command', { detail: { type: 'theme_toggle' } })
          );
          break;
        case 'chat_delete_last':
          openChatThen(() => dispatchChatAction('delete_last'));
          break;
        case 'chat_clear_input':
          openChatThen(() => dispatchChatAction('clear_input'));
          break;
        case 'chat_new':
          openChatThen(() => dispatchChatAction('new_conversation'));
          break;
        case 'chat_web_search_on':
          openChatThen(() => dispatchChatAction('web_search_on'));
          break;
        case 'chat_web_search_off':
          openChatThen(() => dispatchChatAction('web_search_off'));
          break;
        case 'chat_regenerate':
          openChatThen(() => dispatchChatAction('regenerate'));
          break;
        case 'chat_send_text':
          openChatThen(() => {
            window.dispatchEvent(
              new CustomEvent('eloquent-remote', {
                detail: { action: 'send_text', text: String(payload?.text || '').trim() },
                bubbles: true,
              })
            );
          });
          break;
        default:
          break;
      }
    },
    [setActiveTab]
  );

  useEffect(() => {
    if (!remoteEnabled || !remoteSessionId.trim() || !PRIMARY_API_URL) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const params = new URLSearchParams({
          session_id: remoteSessionId.trim(),
          after_id: String(remoteLastIdRef.current || 0),
        });
        const response = await fetch(`${PRIMARY_API_URL}/remote/v1/commands?${params.toString()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const commands = Array.isArray(data?.commands) ? data.commands : [];
        for (const item of commands) {
          applyRemoteCommand(String(item.command || ''), item.payload || {});
        }
        if (!cancelled) {
          if (typeof data?.last_id === 'number') setRemoteLastId(data.last_id);
          if (commands.length > 0) setRemoteLastSeenAt(new Date().toISOString());
          setRemoteError('');
        }
      } catch (error) {
        if (!cancelled) setRemoteError(String(error?.message || error));
      }
    };
    poll();
    const intervalId = setInterval(poll, 900);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [PRIMARY_API_URL, applyRemoteCommand, remoteEnabled, remoteSessionId]);

  const value = useMemo(
    () => ({
      remoteSessionId,
      setRemoteSessionId,
      remoteEnabled,
      setRemoteEnabled,
      remoteLastSeenAt,
      remoteError,
    }),
    [remoteEnabled, remoteError, remoteLastSeenAt, remoteSessionId]
  );

  return <MobileRemoteContext.Provider value={value}>{children}</MobileRemoteContext.Provider>;
}
