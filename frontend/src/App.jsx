// Description: Main application component that sets up the layout and theme context for the app.
import React, { lazy, Suspense, useState, useEffect, useCallback, useRef } from 'react';
import { cn } from '@/lib/utils';
import { useNavbarAutoHide } from './hooks/useNavbarAutoHide';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import './App.css';
// Correct named import for ThemeProvider and import useTheme
import { ThemeProvider, useTheme } from './components/ThemeProvider';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import Documents from './components/Documents';
import Settings from './components/Settings';
import { MemoryProvider } from './contexts/MemoryContext';
import MemoryPage from './pages/MemoryPage';
import DocsPage from './pages/DocsPage';
import UserProfilesPage from './pages/UserProfilesPage';
import SillyTavernSetupPage from './pages/SillyTavernSetupPage';
import { useApp, AppProvider } from './contexts/AppContext';

// Import components
import SimpleModelSelector from './components/SimpleModelSelector';
import CharacterManager from './components/CharacterManager';
import ModelTester from './components/ModelTester';
import { MobileRemoteProvider } from './contexts/MobileRemoteContext';
import { isModuleEnabled } from './config/modules';

import LoginOverlay from './components/LoginOverlay';
import OutreachNotificationStack from './components/OutreachNotificationStack';
import SettingsStandaloneLayout from './components/SettingsStandaloneLayout';
import RoomImageGalleryModal from './components/RoomImageGalleryModal';
import ProviderSetupDialog from './components/ProviderSetupDialog';
import RoleplayWelcomeDialog from './components/RoleplayWelcomeDialog';
import AppUpdatePrompt from './components/AppUpdatePrompt';
import { TRIGGER_LOGIN_EVENT } from './utils/auth-interceptor';

const ElectionTracker = __MIRID_INCLUDE_ELECTIONS__
  ? lazy(() => import('./components/ElectionTracker'))
  : null;
const PoolTab = null;

// Inner component to access theme context easily
function AppContent() {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const standalone = searchParams.get('standalone');
  const { theme, setTheme } = useTheme(); // Use the theme hook here
  const {
    activeTab,
    setActiveTab,
    settingsEntryTab,
    roomGalleryOpen,
    setRoomGalleryOpen,
    setBackgroundImage,
    openSettingsTab,
    primaryModel,
    settings,
    storageHydrated,
  } = useApp();
  // Default to closed on mobile (< 768px), open on desktop
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [chatHistoryOpen, setChatHistoryOpen] = useState(false);
  const [showLogin, setShowLogin] = useState(false);
  const [scrollContainer, setScrollContainer] = useState(null);
  const openChatHistory = useCallback(() => {
    setSidebarOpen(true);
    setChatHistoryOpen(true);
  }, []);
  const assignScrollContainer = useCallback((node) => {
    setScrollContainer(node);
  }, []);

  useEffect(() => {
    if (location.pathname === '/docs') setActiveTab('docs');
  }, [location.pathname, setActiveTab]);

  const modelSetupRedirectHandledRef = useRef(false);
  useEffect(() => {
    if (!storageHydrated || modelSetupRedirectHandledRef.current) return;
    modelSetupRedirectHandledRef.current = true;
    if (settings?.modelSetupRequired === true && !primaryModel) {
      openSettingsTab('models', { forceWindow: false });
    }
  }, [openSettingsTab, primaryModel, settings?.modelSetupRequired, storageHydrated]);

  const previewRoleplayWelcome = import.meta.env.DEV
    && new URLSearchParams(location.search).get('preview') === 'roleplay-welcome';
  const roleplayWelcomeOpen = previewRoleplayWelcome || Boolean(
    storageHydrated
    && settings?.providerSetupCompleted === true
    && settings?.primaryUse === 'roleplay'
    && settings?.roleplayIntroCompleted !== true
    && primaryModel,
  );

  const sillyTavernRedirectHandledRef = useRef(false);
  useEffect(() => {
    if (
      !storageHydrated
      || sillyTavernRedirectHandledRef.current
      || settings?.providerSetupCompleted !== true
      || settings?.primaryUse !== 'sillytavern'
      || settings?.sillyTavernSetupCompleted === true
    ) return;
    sillyTavernRedirectHandledRef.current = true;
    setActiveTab('sillytavern');
  }, [setActiveTab, settings?.primaryUse, settings?.providerSetupCompleted, settings?.sillyTavernSetupCompleted, storageHydrated]);

  useEffect(() => {
    if (roleplayWelcomeOpen) setTheme('faraday');
  }, [roleplayWelcomeOpen, setTheme]);

  const selectTab = useCallback((tab) => {
    setActiveTab(tab);
    if (tab === 'docs') {
      if (location.pathname !== '/docs') navigate('/docs');
    } else if (location.pathname === '/docs') {
      navigate('/');
    }
  }, [location.pathname, navigate, setActiveTab]);

  const openModelLibrary = useCallback(() => {
    if (location.pathname === '/docs') navigate('/');
    openSettingsTab('models');
  }, [location.pathname, navigate, openSettingsTab]);
  const {
    navbarCollapsed,
    navbarPinned,
    toggleNavbarPinned,
  } = useNavbarAutoHide(scrollContainer);

  const [reduceMotion, setReduceMotion] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const apply = () => setReduceMotion(mq.matches);
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, []);

  // Apply the layout mode as a class to the document body
  const [layoutMode, setLayoutMode] = useState('default'); // Default layout mode
  useEffect(() => {
    document.body.classList.remove('default', 'discord', 'whatsapp', 'messenger');
    document.body.classList.add(layoutMode);
    return () => {
      document.body.classList.remove(layoutMode);
    };
  }, [layoutMode]);

  // Listen for Login trigger event from global interceptor
  useEffect(() => {
    const handleTriggerLogin = () => {
      console.log("🔒 Login overlay event received");
      setShowLogin(true);
    };

    window.addEventListener(TRIGGER_LOGIN_EVENT, handleTriggerLogin);
    return () => {
      window.removeEventListener(TRIGGER_LOGIN_EVENT, handleTriggerLogin);
    };
  }, []);

  // Mobile remote: theme toggle (same as header theme control)
  useEffect(() => {
    const onAppCmd = (ev) => {
      if (ev?.detail?.type === 'theme_toggle') {
        setTheme(theme === 'dark' ? 'light' : 'dark');
      }
    };
    window.addEventListener('eloquent-app-command', onAppCmd);
    return () => window.removeEventListener('eloquent-app-command', onAppCmd);
  }, [theme, setTheme]);

  const handleLogin = (password) => {
    // Save to settings
    try {
      const saved = localStorage.getItem('Eloquent-settings') || '{}';
      const parsed = JSON.parse(saved);
      parsed.admin_password = password;
      localStorage.setItem('Eloquent-settings', JSON.stringify(parsed));

      setShowLogin(false);
      // Optional: Reload to retry requests
      window.location.reload();
    } catch (e) {
      console.error("Login save failed", e);
    }
  };

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <Chat
            layoutMode={layoutMode}
            scrollContainerRef={assignScrollContainer}
            onOpenChatHistory={openChatHistory}
          />
        );
      case 'documents':
        return <Documents />;
      case 'models':
        return <SimpleModelSelector />;
      case 'characters':
        return <CharacterManager />;
      case 'user-profiles':
        return <UserProfilesPage />;
      case 'audio':
        return <Settings
          darkMode={theme === 'dark'}
          toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          initialTab="audio"
          audioPage
        />;
      case 'election':
        return ElectionTracker && isModuleEnabled('elections') ? (
          <Suspense fallback={null}><ElectionTracker /></Suspense>
        ) : <Chat layoutMode={layoutMode} scrollContainerRef={assignScrollContainer} onOpenChatHistory={openChatHistory} />;
      case 'settings':
        // Pass theme state and toggle function to Settings
        return <Settings
          darkMode={theme === 'dark'}
          toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          initialTab={settingsEntryTab}
        />;
      case 'memory':
        return <MemoryPage />;
      case 'docs':
        return <DocsPage onLeave={() => selectTab('chat')} onOpenModelLibrary={openModelLibrary} />;
      case 'sillytavern':
        return <SillyTavernSetupPage />;
      case 'modeltester':
        return <ModelTester />;
      case 'pool':
        return PoolTab && isModuleEnabled('pool') ? (
          <Suspense fallback={null}><PoolTab /></Suspense>
        ) : <Chat layoutMode={layoutMode} scrollContainerRef={assignScrollContainer} onOpenChatHistory={openChatHistory} />;
      default:
        return (
          <Chat
            layoutMode={layoutMode}
            scrollContainerRef={assignScrollContainer}
            onOpenChatHistory={openChatHistory}
          />
        );
    }
  };

  // Get layout-specific classes
  const getLayoutClasses = () => {
    switch (layoutMode) {
      case 'discord':
        return 'bg-[#36393f] text-[#dcddde]';
      case 'whatsapp':
        return 'bg-[#efeae2] text-[#262626]';
      case 'messenger':
        return 'bg-[#ffffff] text-[#050505]';
      default:
        // Use CSS variables defined in ThemeProvider
        return 'bg-background text-foreground';
    }
  };

  if (standalone === 'settings') {
    return (
      <>
        <SettingsStandaloneLayout />
        <LoginOverlay isOpen={showLogin} onLogin={handleLogin} />
      </>
    );
  }

  const isChatTab = activeTab === 'chat' || activeTab == null;
  // Keep offset constant when auto-hiding: navbar slides over content via transform only.
  const navbarOffset = '3rem';

  return (
    <div
      className={cn('h-screen flex flex-col overflow-hidden', layoutMode)}
      style={{
        '--app-navbar-height': '3rem',
        '--app-navbar-offset': navbarOffset,
      }}
    >
      <Navbar
        toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        collapsed={navbarCollapsed}
        pinned={navbarPinned}
        onTogglePin={toggleNavbarPinned}
        reduceMotion={reduceMotion}
      />
      <div
        className={cn('flex flex-1 min-h-0 overflow-hidden', getLayoutClasses())}
        style={{ paddingTop: navbarOffset }}
      >
        <Sidebar
          isOpen={sidebarOpen}
          setIsOpen={setSidebarOpen}
          activeTab={activeTab}
          setActiveTab={selectTab}
          layoutMode={layoutMode}
          historyPanelOpen={chatHistoryOpen}
          setHistoryPanelOpen={setChatHistoryOpen}
        />
        <main
          ref={isChatTab ? undefined : assignScrollContainer}
          className={cn(
            'flex-1 min-h-0 flex flex-col',
            isChatTab ? 'min-h-0 overflow-hidden p-0' : 'overflow-y-auto overscroll-contain p-4',
          )}
        >
          {renderActiveComponent()}
        </main>
      </div>

      <OutreachNotificationStack />

      {!previewRoleplayWelcome && <ProviderSetupDialog />}

      <RoleplayWelcomeDialog
        open={roleplayWelcomeOpen}
        onOpenCharacters={() => selectTab('characters')}
      />

      <AppUpdatePrompt
        enabled={Boolean(storageHydrated)}
      />

      <LoginOverlay isOpen={showLogin} onLogin={handleLogin} />

      <RoomImageGalleryModal
        open={roomGalleryOpen}
        onOpenChange={setRoomGalleryOpen}
        onSelect={(url) => {
          setBackgroundImage(url);
        }}
      />
    </div>
  );
}


// Main App wrapper including Providers
function App() {
  return (
    <MemoryProvider>
      <AppProvider>
        <MobileRemoteProvider>
          <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
            <AppContent />
          </ThemeProvider>
        </MobileRemoteProvider>
      </AppProvider>
    </MemoryProvider>
  );
}

export default App;
