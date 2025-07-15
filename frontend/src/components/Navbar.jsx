import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useApp } from '../contexts/AppContext'; // Import useApp hook
import { Button } from './ui/button';
import { ModeToggle } from './ModeToggle';
import {
  Menu,
  Badge,
  Bot,
  Settings, // Keep existing
  Box,
  Zap,
  // Add icons from Sidebar
  MessageSquare,
  Image,
  FileText,
  UserCircle,
  BookOpen
} from 'lucide-react';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from './ui/sheet';
import { Separator } from './ui/separator'; // Import Separator

const Navbar = ({ toggleSidebar, layoutMode, setLayoutMode }) => {
  // Get necessary values from AppContext
  const {
    loadedModels,
    activeModel,
    PRIMARY_API_URL,
    activeTab, // Get the active tab state
    setActiveTab // Get the function to set the active tab
  } = useApp();
  const location = useLocation();
  const [isShuttingDown, setIsShuttingDown] = useState(false);
  // Define the navigation items that mirror the sidebar
  const sidebarNavItems = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="mr-2 h-4 w-4" /> },
    { id: 'image', label: 'Image Gen', icon: <Image className="mr-2 h-4 w-4" /> },
    { id: 'documents', label: 'Documents', icon: <FileText className="mr-2 h-4 w-4" /> },
    { id: 'characters', label: 'Characters', icon: <UserCircle className="mr-2 h-4 w-4" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="mr-2 h-4 w-4" /> },
    { id: 'memory', label: 'Memory', icon: <BookOpen className="mr-2 h-4 w-4" /> },
    { id: 'modeltester', label: 'Model Tester', icon: <Zap className="mr-2 h-4 w-4" /> }
  ];

  // Define the original placeholder navigation items
  const placeholderNavItems = [
    { path: '/extensions', label: 'Extensions', icon: <Box className="mr-2 h-4 w-4" /> },
    { path: '/shutdown', label: 'Shutdown', icon: <Zap className="mr-2 h-4 w-4" /> }
  ];

// Replace the existing handleShutdown function with this:
const handleShutdown = async () => {
  if (isShuttingDown) return; // Prevent multiple clicks
  
  if (window.confirm('Are you sure you want to shut down the backend server? This will close the application.')) {
    setIsShuttingDown(true);
    
    try {
      console.log('🔴 Initiating shutdown...');
      
      // Call the shutdown endpoint
      const response = await fetch(`${PRIMARY_API_URL}/system/shutdown`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Shutdown initiated:', data.message);
        
        // Show success message
        alert('Shutdown initiated successfully. The server will stop in a few seconds.');
        
        // Optionally redirect or show a "server shutting down" page
        setTimeout(() => {
          window.location.href = 'about:blank'; // Close the tab
        }, 3000);
        
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
    } catch (error) {
      console.error('❌ Shutdown failed:', error);
      alert(`Failed to shutdown server: ${error.message}`);
      setIsShuttingDown(false); // Re-enable button on error
    }
  }
};

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

  {/* Desktop Navigation Items */}
  <nav className="hidden md:flex items-center gap-2 flex-1">
          {/* Sidebar Mirror Buttons */}
          {sidebarNavItems.map((item) => (
            <Button
              key={item.id}
              // Highlight based on activeTab state
              variant={activeTab === item.id ? 'secondary' : 'ghost'}
              className="px-3 py-1"
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              {item.label}
            </Button>
          ))}

          {/* Separator */}
          <Separator orientation="vertical" className="h-6 mx-2" />

          {/* Original Placeholder Buttons */}
          {placeholderNavItems.map((item) => (
            item.path === '/shutdown' ? (
              <Button
                key={item.path}
                variant="outline"
                className="px-3 py-1 text-red-500 hover:text-red-600"
                onClick={handleShutdown}
                disabled={isShuttingDown} // Disable while shutting down
              >
                {item.icon}
                {isShuttingDown ? 'Shutting Down...' : item.label}
              </Button>
            ) : (
              // These still use Link as they might represent actual routes
              <Link key={item.path} to={item.path}>
                <Button
                  // Highlight based on URL path for these placeholders
                  variant={location.pathname === item.path ? 'secondary' : 'ghost'}
                  className="px-3 py-1"
                >
                  {item.icon}
                  {item.label}
                </Button>
              </Link>
            )
          ))}
        </nav>

        {/* Right Side: Status Indicators + Theme + Layout Toggle + Mobile Menu */}
        <div className="flex items-center gap-4">
          {/* Model Status Badges */}
          {loadedModels?.length > 0 && (
            <Badge variant="outline" className="hidden sm:inline-flex"> {/* Hide on very small screens */}
              {loadedModels.length} Model{loadedModels.length !== 1 ? 's' : ''} Loaded
            </Badge>
          )}
          {activeModel && (
            <Badge variant="secondary" className="hidden sm:inline-flex"> {/* Hide on very small screens */}
              Active: {activeModel.split('/').pop().split('\\').pop().replace(/\.(bin|gguf)$/i, '')} {/* Shorten name */}
            </Badge>
          )}

          {/* Theme toggle */}
          <ModeToggle />

          {/* Layout toggle dropdown */}
          <select
            value={layoutMode}
            onChange={(e) => setLayoutMode(e.target.value)}
            className="px-2 py-1 border rounded text-sm bg-background"
            title="Change Layout Theme"
          >
            <option value="default">Default</option>
            <option value="whatsapp">WhatsApp</option>
            <option value="messenger">Messenger</option>
            <option value="discord">Discord</option>
          </select>

          {/* Mobile Menu */}
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="md:hidden">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64">
              <div className="p-4 border-b"> {/* Added border */}
                <h2 className="text-lg font-semibold">Menu</h2>
              </div>
              {/* Removed extra Separator here, nav adds spacing */}
              <nav className="p-4 space-y-2">
                {/* Sidebar Mirror Buttons */}
                {sidebarNavItems.map((item) => (
                   <Button
                     key={`mobile-${item.id}`}
                     variant={activeTab === item.id ? 'secondary' : 'ghost'}
                     className="w-full justify-start"
                     // Consider adding logic to close the sheet onClick
                     onClick={() => setActiveTab(item.id)}
                   >
                     {item.icon}
                     {item.label}
                   </Button>
                 ))}

                {/* Separator */}
                <Separator className="my-2" />

                {/* Original Placeholder Buttons */}
                {placeholderNavItems.map((item) => (
                  item.path === '/shutdown' ? (
                    <Button
                      key={`mobile-${item.path}`}
                      variant="outline"
                      className="w-full justify-start text-red-500 hover:text-red-600"
                      onClick={handleShutdown}
                    >
                      {item.icon}
                      {item.label}
                    </Button>
                  ) : (
                    <Link key={`mobile-${item.path}`} to={item.path}>
                      <Button
                        variant={location.pathname === item.path ? 'secondary' : 'ghost'}
                        className="w-full justify-start"
                        // Consider adding logic to close the sheet onClick
                      >
                        {item.icon}
                        {item.label}
                      </Button>
                    </Link>
                  )
                ))}
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
