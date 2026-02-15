// Description: Main application component that sets up the layout and theme context for the app.
import React, { useState, useEffect } from 'react';
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
import { useApp, AppProvider } from './contexts/AppContext';

// Import components
import SimpleModelSelector from './components/SimpleModelSelector';
import CharacterManager from './components/CharacterManager';
import ModelTester from './components/ModelTester';
import ForensicLinguistics from './components/ForensicLinguistics';
import CodeEditorOverlay from './components/CodeEditorOverlay';
import ElectionTracker from './components/ElectionTracker';

import LoginOverlay from './components/LoginOverlay';
import { TRIGGER_LOGIN_EVENT } from './utils/auth-interceptor';

// Inner component to access theme context easily
function AppContent() {
  const { theme, setTheme } = useTheme(); // Use the theme hook here
  const { activeTab, setActiveTab } = useApp(); // Get active tab from AppContext
  // Default to closed on mobile (< 768px), open on desktop
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768);
  const [showLogin, setShowLogin] = useState(false);

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
        return <Chat layoutMode={layoutMode} />;
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
      case 'settings':
        // Pass theme state and toggle function to Settings
        return <Settings
          darkMode={theme === 'dark'}
          toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          initialTab="general"
        />;
      case 'memory':
        return <MemoryPage />;
      case 'modeltester':
        return <ModelTester />;
      case 'codeeditor':
        return <CodeEditorOverlay isOpen={true} onClose={() => setActiveTab('chat')} />;
      default:
        return <Chat layoutMode={layoutMode} />;
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

  return (
    <div className={`min-h-screen flex flex-col ${layoutMode}`}>
      <Navbar
        toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        layoutMode={layoutMode}
        setLayoutMode={setLayoutMode}
      />
      <div className={`flex flex-1 overflow-hidden ${getLayoutClasses()}`}>
        <Sidebar
          isOpen={sidebarOpen}
          setIsOpen={setSidebarOpen}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          layoutMode={layoutMode}
        />
        <main className="flex-1 overflow-y-auto p-4">
          {renderActiveComponent()}
        </main>
      </div>

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
        {/* ThemeProvider wraps everything that needs theme context */}
        <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
          <AppContent /> {/* Render the inner component that uses the theme */}
        </ThemeProvider>
      </AppProvider>
    </MemoryProvider>
  );
}

export default App;
