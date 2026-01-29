import React from 'react';
import { useApp } from '../contexts/AppContext';
import { useTheme } from './ThemeProvider';
import { Button } from './ui/button';
import {
  Badge,
  Settings,
  MessageSquare,
  FileText,
  UserCircle,
  BookOpen,
  Zap,
  Search,
  Code,
  Fingerprint,
  Palette,
  Power,
  RotateCw
} from 'lucide-react';

const Navbar = ({ toggleSidebar }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);
  // Get necessary values from AppContext
  const {
    loadedModels,
    activeModel,
    activeTab,
    setActiveTab,
    PRIMARY_API_URL
  } = useApp();

  const { theme, setTheme } = useTheme();

  // Define the navigation items that mirror the sidebar
  const sidebarNavItems = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="mr-2 h-4 w-4" /> },
    { id: 'documents', label: 'Documents', icon: <FileText className="mr-2 h-4 w-4" /> },
    { id: 'characters', label: 'Characters', icon: <UserCircle className="mr-2 h-4 w-4" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="mr-2 h-4 w-4" /> },
    { id: 'memory', label: 'Memory', icon: <BookOpen className="mr-2 h-4 w-4" /> },
    { id: 'modeltester', label: 'Model Tester', icon: <Zap className="mr-2 h-4 w-4" /> },
    { id: 'codeeditor', label: 'Code Editor', icon: <Code className="mr-2 h-4 w-4" /> },
    { id: 'forensics', label: 'Forensics', icon: <Fingerprint className="mr-2 h-4 w-4" /> }
  ];

  return (
    <header className="border-b bg-card">
      <div className="container flex items-center h-16 px-4">
        {/* Brand/Logo - with more space */}
        <div className="flex items-center gap-3 mr-2 lg:mr-8 flex-shrink-0">
          <img
            src="/eloquent-logo.png"
            alt="Eloquent"
            className="h-9 w-9"
          />
          <span className="font-bold text-xl" style={{ fontFamily: 'Poppins, sans-serif' }}>Eloquent</span>
        </div>

        {/* Desktop Navigation - Hidden on Mobile */}
        <nav className="hidden lg:flex items-center gap-2 flex-1">
          {sidebarNavItems.map((item) => (
            <Button
              key={item.id}
              variant={activeTab === item.id ? 'secondary' : 'ghost'}
              className="px-3 py-1"
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              {item.label}
            </Button>
          ))}
        </nav>

        {/* Right Side: Status Indicators & Theme Toggle */}
        <div className="flex items-center gap-4">

          {/* System Controls */}
          <div className="flex items-center gap-1 border-r pr-2 mr-2">
            <Button
              variant="ghost"
              size="icon"
              title="Restart Application"
              className="text-muted-foreground hover:text-foreground"
              onClick={async () => {
                if (confirm('Are you sure you want to restart Eloquent?')) {
                  try {
                    await fetch(`${PRIMARY_API_URL}/system/restart`, { method: 'POST' });
                    // No alert needed as page will loose connection
                  } catch (e) {
                    console.error("Restart failed", e);
                  }
                }
              }}
            >
              <RotateCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Shutdown Application"
              className="text-muted-foreground hover:text-red-500"
              onClick={async () => {
                if (confirm('Are you sure you want to shutdown Eloquent?')) {
                  try {
                    await fetch(`${PRIMARY_API_URL}/system/shutdown`, { method: 'POST' });
                    alert("System shutting down. You can close this window.");
                  } catch (e) {
                    console.error("Shutdown failed", e);
                  }
                }
              }}
            >
              <Power className="h-4 w-4" />
            </Button>
          </div>

          {/* Theme Selector - Custom Styled or Standard Select */}
          <div className="flex items-center mr-2">
            <Palette className="w-4 h-4 mr-2 text-muted-foreground" />
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              className="bg-white text-black border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-gray-400 cursor-pointer shadow-sm"
            >
              <optgroup label="Base">
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </optgroup>
              <optgroup label="Chat">
                <option value="whatsapp">WhatsApp</option>
                <option value="messenger">Messenger</option>
                <option value="claude">Claude</option>
              </optgroup>
              <optgroup label="Vibrant">
                <option value="cyberpunk">Cyberpunk</option>
              </optgroup>
            </select>
          </div>

          {/* Model Status Badges */}
          {loadedModels?.length > 0 && (
            <Badge variant="outline">
              {loadedModels.length} Model{loadedModels.length !== 1 ? 's' : ''} Loaded
            </Badge>
          )}
          {activeModel && (
            <Badge variant="secondary">
              Active: {activeModel.split('/').pop().split('\\').pop().replace(/\.(bin|gguf)$/i, '')}
            </Badge>
          )}
        </div>
        {/* Mobile Menu Button - Visible on Mobile */}
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden ml-2"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </Button>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t bg-card p-4 space-y-2 absolute top-16 left-0 right-0 z-50 shadow-lg">
          {sidebarNavItems.map((item) => (
            <Button
              key={item.id}
              variant={activeTab === item.id ? 'secondary' : 'ghost'}
              className="w-full justify-start"
              onClick={() => {
                setActiveTab(item.id);
                setMobileMenuOpen(false);
              }}
            >
              {item.icon}
              <span className="ml-2">{item.label}</span>
            </Button>
          ))}
        </div>
      )}
    </header>
  );
};

export default Navbar;
