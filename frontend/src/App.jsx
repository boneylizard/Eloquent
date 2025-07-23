// Description: Main application component that sets up the layout and theme context for the app.
//   <button
import React, { useState, useEffect } from 'react';
import './App.css';
// Correct named import for ThemeProvider and import useTheme
import { ThemeProvider, useTheme } from './components/ThemeProvider';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import ImageGen from './components/ImageGen';
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


// Inner component to access theme context easily
function AppContent() {
  const { theme, setTheme } = useTheme(); // Use the theme hook here
  const { activeTab, setActiveTab } = useApp(); // Get active tab from AppContext
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Apply the layout mode as a class to the document body
  const [layoutMode, setLayoutMode] = useState('default'); // Default layout mode
  useEffect(() => {
    document.body.classList.remove('default', 'discord', 'whatsapp', 'messenger');
    document.body.classList.add(layoutMode);
    return () => {
      document.body.classList.remove(layoutMode);
    };
  }, [layoutMode]);

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'chat':
        return <Chat layoutMode={layoutMode} />;
      case 'image':
        return <ImageGen />;
      case 'documents':
        return <Documents />;
      case 'forensics': // ADD THIS CASE
        return <ForensicLinguistics />;
      case 'models':
        return <SimpleModelSelector />;
      case 'characters':
        return <CharacterManager />;
      case 'settings':
        // Pass theme state and toggle function to Settings
        return <Settings
                  darkMode={theme === 'dark'}
                  toggleDarkMode={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                  initialTab="general" // Or keep whatever default you prefer
               />;
      case 'memory':
        return <MemoryPage />;
      case 'modeltester':
        return <ModelTester />;
      default:
        return <Chat layoutMode={layoutMode}/>;
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