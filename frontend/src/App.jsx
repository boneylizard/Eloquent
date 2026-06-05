// Description: Main application component that sets up the layout and theme context for the app.
import React, { useState, useEffect, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { useNavbarAutoHide } from './hooks/useNavbarAutoHide';
import { useSearchParams } from 'react-router-dom';
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
import TranscriptCorpusPage from './pages/TranscriptCorpusPage';
import { useApp, AppProvider } from './contexts/AppContext';
import { IntensityProvider } from './contexts/IntensityContext';

// Import components
import SimpleModelSelector from './components/SimpleModelSelector';
import CharacterManager from './components/CharacterManager';
import ModelTester from './components/ModelTester';
import ForensicLinguistics from './components/ForensicLinguistics';
import CodeEditorOverlay from './components/CodeEditorOverlay';
import ElectionTracker from './components/ElectionTracker';
import ChessTab from './components/ChessTab';
import MarketSimTab from './components/MarketSimTab';
import WatchTab from './components/WatchTab';
import { VideoWatchProvider } from './contexts/VideoWatchContext';

import LoginOverlay from './components/LoginOverlay';
import OutreachNotificationStack from './components/OutreachNotificationStack';
import SettingsStandaloneLayout from './components/SettingsStandaloneLayout';
import CallModeStandaloneLayout from './components/CallModeStandaloneLayout';
import { TRIGGER_LOGIN_EVENT } from './utils/auth-interceptor';

// Inner component to access theme context easily
function AppContent() {
  const [searchParams] = useSearchParams();
  const standalone = searchParams.get('standalone');
  const { theme, setTheme } = useTheme(); // Use the theme hook here
  const { activeTab, setActiveTab, settingsEntryTab } = useApp();
  // Default to closed on mobile (< 768px), open on desktop
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [showLogin, setShowLogin] = useState(false);
  const [scrollContainer, setScrollContainer] = useState(null);
  const assignScrollContainer = useCallback((node) => {
    setScrollContainer(node);
  }, []);
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
          />
        );
      case 'documents':
        return <Documents />;
      case 'forensics':
        return <ForensicLinguistics onClose={() => setActiveTab('chat')} />;
      case 'models':
        return <SimpleModelSelector />;
      case 'characters':
        return <CharacterManager />;
      case 'election':
        return <ElectionTracker />;
      case 'chess':
        return <ChessTab />;
      case 'market-sim':
        return <MarketSimTab />;
      case 'watch':
        return <WatchTab />;
      case 'settings':
        // Pass theme state and toggle function to Settings
        return <Settings
          darkMode={theme === 'dark'}
          toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          initialTab={settingsEntryTab}
        />;
      case 'memory':
        return <MemoryPage />;
      case 'transcript-corpus':
        return <TranscriptCorpusPage />;
      case 'modeltester':
        return <ModelTester />;
      case 'codeeditor':
        return <CodeEditorOverlay isOpen={true} onClose={() => setActiveTab('chat')} />;
      default:
        return (
          <Chat
            layoutMode={layoutMode}
            scrollContainerRef={assignScrollContainer}
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

  if (standalone === 'call') {
    return <CallModeStandaloneLayout />;
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
          setActiveTab={setActiveTab}
          layoutMode={layoutMode}
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

      <LoginOverlay isOpen={showLogin} onLogin={handleLogin} />
    </div>
  );
}


// Main App wrapper including Providers
function App() {
  return (
    // MemoryProvider should likely wrap AppProvider if AppContext depends on MemoryContext
    <MemoryProvider>
      <AppProvider>
        <IntensityProvider>
          <VideoWatchProvider>
            {/* ThemeProvider wraps everything that needs theme context */}
            <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
              <AppContent /> {/* Render the inner component that uses the theme */}
            </ThemeProvider>
          </VideoWatchProvider>
        </IntensityProvider>
      </AppProvider>
    </MemoryProvider>
  );
}

export default App;
