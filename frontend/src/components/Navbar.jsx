import React from 'react';
import { useApp } from '../contexts/AppContext';
import { useTheme } from './ThemeProvider';
import { Button } from './ui/button';
import {
  Badge,
  Settings,
  MessageSquare,
  Image,
  FileText,
  UserCircle,
  BookOpen,
  Zap,
  Search,
  Code,
  Fingerprint,
  Palette
} from 'lucide-react';

const Navbar = ({ toggleSidebar }) => {
  // Get necessary values from AppContext
  const {
    loadedModels,
    activeModel,
    activeTab,
    setActiveTab
  } = useApp();

  const { theme, setTheme } = useTheme();

  // Define the navigation items that mirror the sidebar
  const sidebarNavItems = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="mr-2 h-4 w-4" /> },
    { id: 'image', label: 'Image Gen', icon: <Image className="mr-2 h-4 w-4" /> },
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
        <div className="flex items-center gap-3 mr-8">
          <img
            src="/eloquent-logo.png"
            alt="Eloquent"
            className="h-9 w-9"
          />
          <span className="font-bold text-xl" style={{ fontFamily: 'Poppins, sans-serif' }}>Eloquent</span>
        </div>

        {/* Desktop Navigation */}
        <nav className="flex items-center gap-2 flex-1">
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
      </div>
    </header>
  );
};

export default Navbar;