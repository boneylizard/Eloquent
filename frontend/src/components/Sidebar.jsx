import { useEffect, useState } from 'react';
import { cn } from '../lib/utils';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import {
  MessageSquare,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  UserCircle,
  BookOpen,
  Cpu,
  Swords,
  TrendingUp
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from './ui/select';
import DeleteConfirmDialog from './ui/DeleteConfirmDialog';

const Sidebar = ({ isOpen, setIsOpen, activeTab, setActiveTab, layoutMode }) => {
  const { conversations, activeConversation, createNewConversation, setActiveConversation, setAvailableModels, availableModels, API_URL, deleteConversation, deleteAllConversations, autoDeleteChats, setAutoDeleteChats } = useApp();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);
  const [deleteDialogPosition, setDeleteDialogPosition] = useState({ x: 0, y: 0 });
  const [deleteAllDialogOpen, setDeleteAllDialogOpen] = useState(false);
  const [deleteAllPosition, setDeleteAllPosition] = useState({ x: 0, y: 0 });
  const [loadedModels, setLoadedModels] = useState([]);

  const [selectedGpu, setSelectedGpu] = useState("0");
  const [showAllConversations, setShowAllConversations] = useState(false);

  // Note: Keeping the useEffect in case it's used elsewhere, but models tab reference removed
  useEffect(() => {
    if (activeTab === 'models') {
      fetchModels();
    }
  }, [activeTab, selectedGpu]);

  const fetchModels = async () => {
    try {
      const response = await fetch(`http://localhost:800${selectedGpu}/models`);
      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
      const data = await response.json();
      setAvailableModels(data.available_models || []);
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  };

  const handleTabClick = (tab) => {
    setActiveTab(tab);
  };

  const handleToggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  const handleNewChat = () => {
    createNewConversation();
  };

  const handleConversationClick = (id) => {
    setActiveConversation(id);
    setActiveTab('chat');
  };

  // Add this effect to ensure dialog state respects autoDeleteChats setting
  useEffect(() => {
    // If autoDeleteChats becomes true, make sure the dialog is closed
    if (autoDeleteChats && deleteDialogOpen) {
      setDeleteDialogOpen(false);
    }
  }, [autoDeleteChats, deleteDialogOpen]);

  const navItems = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="w-5 h-5" /> },
    { id: 'documents', label: 'Documents', icon: <FileText className="w-5 h-5" /> },
    { id: 'characters', label: 'Characters', icon: <UserCircle className="w-5 h-5" /> },
    { id: 'chess', label: 'Chess', icon: <Swords className="w-5 h-5" /> },
    { id: 'market-sim', label: 'Market Simulator', icon: <TrendingUp className="w-5 h-5" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="w-5 h-5" /> },
    { id: 'memory', label: 'Memory', icon: <BookOpen className="w-5 h-5" /> }
  ];

  return (
    <>
      {isOpen && <div className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden" onClick={() => setIsOpen(false)} />}

      <div className={cn(
        "fixed inset-y-0 left-0 z-50 flex flex-col w-64 border-r transition-transform duration-300 ease-in-out md:relative md:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full",
        layoutMode === 'discord' ? "bg-[#2f3136] text-white border-gray-700" : "bg-card"
      )}>
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <img
              src="/eloquent-logo.png"
              alt="Eloquent"
              className="h-9 w-9"
            />
            <h2 className="text-lg font-bold" style={{ fontFamily: 'Poppins, sans-serif' }}>Eloquent</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={handleToggleSidebar} className="md:hidden">
            <ChevronLeft className="h-5 w-5" />
          </Button>
        </div>

        <div className="flex flex-col flex-1 overflow-hidden">
          {activeTab === 'chat' && (
            <>
              <div className="p-4 border-b space-y-2">
                <Button variant="default" className="w-full justify-start" onClick={handleNewChat}>
                  + New Chat
                </Button>
                {conversations.length > 0 && (
                  <Button
                    variant="outline"
                    className="w-full justify-start text-red-500 hover:text-red-700 hover:bg-red-50"
                    onClick={(e) => {
                      setDeleteAllPosition({ x: e.clientX, y: e.clientY });
                      setDeleteAllDialogOpen(true);
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                      <path d="M3 6h18"></path>
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                    </svg>
                    Delete All Chats
                  </Button>
                )}
              </div>

              <ScrollArea className="flex-1">
                <div className="p-2 space-y-1">
                  {/* Show conversations based on collapsed/expanded state */}
                  {(showAllConversations
                    ? [...conversations].sort((a, b) => new Date(b.created) - new Date(a.created))
                    : [...conversations].sort((a, b) => new Date(b.created) - new Date(a.created)).slice(0, 5)
                  ).map((conv) => (
                    <div key={conv.id} className="flex items-center w-full mb-1">
                      <Button
                        variant={conv.id === activeConversation ? "secondary" : "ghost"}
                        className={cn(
                          "flex-grow justify-start text-left items-start w-full py-2",
                          conv.id === activeConversation ? "font-medium" : "font-normal"
                        )}
                        onClick={() => handleConversationClick(conv.id)}
                        title={conv.name}
                      >
                        <MessageSquare className="mr-2 h-4 w-4 flex-shrink-0 mt-1" />
                        <span className="flex-grow break-words">{conv.name}</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 p-0 ml-1 text-red-500 hover:bg-red-100 hover:text-red-700"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (autoDeleteChats) {
                            deleteConversation(conv.id);
                          } else {
                            // Use clientX/clientY - actual mouse position on screen
                            setDeleteDialogPosition({
                              x: e.clientX,
                              y: e.clientY
                            });
                            setChatToDelete(conv);
                            setDeleteDialogOpen(true);
                          }
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 6h18"></path>
                          <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                          <path d="M8 6V4c0 1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                        </svg>
                      </Button>
                    </div>
                  ))}

                  {/* Show expand/collapse button only if there are more than 5 conversations */}
                  {conversations.length > 5 && (
                    <div className="px-2 py-1">
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full text-xs"
                        onClick={() => setShowAllConversations(!showAllConversations)}
                      >
                        {showAllConversations ? (
                          <>
                            <ChevronLeft className="w-3 h-3 mr-1" />
                            Show Recent Only
                          </>
                        ) : (
                          <>
                            <ChevronRight className="w-3 h-3 mr-1" />
                            Load More Chats ({conversations.length - 5})
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </>
          )}

          <div className="p-4 border-t mt-auto">
            {activeTab === 'models' && (
              <div className="mb-4">
                <Select onValueChange={setSelectedGpu} defaultValue={selectedGpu}>
                  <SelectTrigger className="w-full">
                    <Cpu className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="Select GPU" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">GPU 0</SelectItem>
                    <SelectItem value="1">GPU 1</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            <nav className="space-y-1">
              {navItems.map((item) => (
                <Button
                  key={item.id}
                  variant={activeTab === item.id ? "secondary" : "ghost"}
                  className="w-full justify-start"
                  onClick={() => handleTabClick(item.id)}
                >
                  {item.icon}
                  <span className="ml-2">{item.label}</span>
                </Button>
              ))}
            </nav>
          </div>
        </div>
      </div>

      {!isOpen && (
        <Button variant="outline" size="icon" className="fixed top-4 left-4 z-50 md:hidden" onClick={handleToggleSidebar}>
          <ChevronRight className="h-5 w-5" />
        </Button>
      )}

      {/* Render dialog at root level, outside all containers */}
      <DeleteConfirmDialog
        isOpen={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        onConfirm={(dontAskAgain) => {
          if (chatToDelete) {
            deleteConversation(chatToDelete.id);
            if (dontAskAgain) {
              setAutoDeleteChats(true);
            }
          }
          setChatToDelete(null);
          setDeleteDialogOpen(false);
        }}
        title={chatToDelete?.name || ''}
        position={deleteDialogPosition}
      />

      {/* Delete All confirmation dialog */}
      <DeleteConfirmDialog
        isOpen={deleteAllDialogOpen}
        onClose={() => setDeleteAllDialogOpen(false)}
        onConfirm={() => {
          deleteAllConversations();
          setDeleteAllDialogOpen(false);
        }}
        title={`all ${conversations.length} chats`}
        position={deleteAllPosition}
      />
    </>
  );
};

export default Sidebar;
