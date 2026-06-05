import { useCallback, useEffect, useState } from 'react';

import { cn } from '../lib/utils';

import { Button } from './ui/button';

import { ScrollArea } from './ui/scroll-area';

import {

  MessageSquare,

  FileText,

  Settings,

  Plus,

  ChevronLeft,

  ChevronRight,

  UserCircle,

  BookOpen,

  FolderSearch,

  Cpu,

  Swords,

  TrendingUp,

  Clapperboard,

  History,

  X,

} from 'lucide-react';

import { useApp } from '../contexts/AppContext';

import { isOutreachConversationId } from '../utils/conversationStorage';

import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from './ui/select';

import DeleteConfirmDialog from './ui/DeleteConfirmDialog';



const TrashIcon = () => (

  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">

    <path d="M3 6h18" />

    <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />

    <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />

  </svg>

);



function ConversationHistoryPanel({

  sidebarConversations,

  activeConversation,

  conversationSaveStatus,

  showAllConversations,

  setShowAllConversations,

  onNewChat,

  onConversationClick,

  onDeleteAllClick,

  onDeleteConversationClick,

  onClose,

  showCloseButton = false,

  className,

}) {

  const sorted = [...sidebarConversations].sort((a, b) => new Date(b.created) - new Date(a.created));

  const visible = showAllConversations ? sorted : sorted.slice(0, 5);



  return (

    <div className={cn('flex flex-col h-full min-h-0', className)}>

      <div className="flex items-center justify-between gap-2 px-3 py-3 border-b border-border shrink-0">

        <h2 className="text-sm font-semibold text-foreground tracking-wide">Chat history</h2>

        {showCloseButton && onClose && (

          <button

            type="button"

            onClick={onClose}

            title="Close chat history"

            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-[var(--nano-bg-hover)] hover:text-foreground border border-transparent hover:border-border transition-colors"

          >

            <X className="h-4 w-4" />

          </button>

        )}

      </div>



      <div className="p-3 space-y-2 shrink-0 border-b border-border/80">

        <Button

          variant="default"

          className="w-full justify-center bg-primary text-primary-foreground hover:brightness-110"

          onClick={onNewChat}

        >

          + New Chat

        </Button>

        {conversationSaveStatus === 'saving' && (

          <p className="text-xs text-muted-foreground px-1">Saving…</p>

        )}

        {conversationSaveStatus === 'saved' && (

          <p className="text-xs text-muted-foreground px-1">Saved</p>

        )}

        {conversationSaveStatus === 'error' && (

          <p className="text-xs text-red-400 px-1">Save failed</p>

        )}

        {sidebarConversations.length > 0 && (

          <Button

            variant="outline"

            className="w-full justify-center text-red-400 hover:text-red-200 hover:bg-red-500/10 border-red-500/40 bg-transparent"

            onClick={onDeleteAllClick}

          >

            <TrashIcon />

            <span className="ml-2">Delete All Chats</span>

          </Button>

        )}

      </div>



      <ScrollArea className="flex-1 min-h-0">

        <div className="p-2 space-y-1">

          {visible.map((conv) => (

            <div key={conv.id} className="flex items-center w-full mb-1 group">

              <Button

                variant={conv.id === activeConversation ? 'secondary' : 'ghost'}

                className={cn(

                  'flex-grow justify-start text-left items-start w-full py-2 rounded-xl',

                  conv.id === activeConversation

                    ? 'bg-secondary text-foreground font-medium border border-border'

                    : 'font-normal text-secondary-foreground hover:bg-[var(--nano-bg-hover)] hover:text-foreground'

                )}

                onClick={() => onConversationClick(conv.id)}

                title={conv.name}

              >

                <MessageSquare className="mr-2 h-4 w-4 flex-shrink-0 mt-1 opacity-70" />

                <span className="flex-grow break-words line-clamp-2">{conv.name}</span>

              </Button>

              <Button

                variant="ghost"

                size="icon"

                className="h-8 w-8 p-0 ml-1 text-red-400/90 hover:bg-red-500/15 hover:text-red-300 shrink-0"

                onClick={(e) => onDeleteConversationClick(e, conv)}

              >

                <TrashIcon />

              </Button>

            </div>

          ))}



          {sidebarConversations.length > 5 && (

            <div className="px-2 py-1">

              <Button

                variant="outline"

                size="sm"

                className="w-full text-xs border-border text-muted-foreground hover:bg-[var(--nano-bg-hover)] hover:text-foreground bg-transparent"

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

                    Load More Chats ({sidebarConversations.length - 5})

                  </>

                )}

              </Button>

            </div>

          )}



          {sidebarConversations.length === 0 && (

            <p className="text-xs text-[rgba(148,163,184,0.75)] px-3 py-4 text-center">No chats yet</p>

          )}

        </div>

      </ScrollArea>

    </div>

  );

}



const Sidebar = ({ isOpen, setIsOpen, activeTab, setActiveTab, layoutMode }) => {

  const { conversations, activeConversation, createNewConversation, goToHome, handleConversationClick, setAvailableModels, availableModels, deleteConversation, deleteAllConversations, autoDeleteChats, setAutoDeleteChats, openSettingsTab, conversationSaveStatus } = useApp();

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const [chatToDelete, setChatToDelete] = useState(null);

  const [deleteDialogPosition, setDeleteDialogPosition] = useState({ x: 0, y: 0 });

  const [deleteAllDialogOpen, setDeleteAllDialogOpen] = useState(false);

  const [deleteAllPosition, setDeleteAllPosition] = useState({ x: 0, y: 0 });

  const [loadedModels, setLoadedModels] = useState([]);



  const [selectedGpu, setSelectedGpu] = useState('0');

  const [showAllConversations, setShowAllConversations] = useState(false);

  const [historyPanelOpen, setHistoryPanelOpen] = useState(false);



  const sidebarConversations = conversations.filter((c) => !isOutreachConversationId(c?.id));



  // Note: Keeping the useEffect in case it's used elsewhere, but models tab reference removed

  useEffect(() => {

    if (activeTab === 'models') {

      fetchModels();

    }

  }, [activeTab, selectedGpu]);



  useEffect(() => {

    if (!historyPanelOpen) return undefined;

    const onKeyDown = (e) => {

      if (e.key === 'Escape') setHistoryPanelOpen(false);

    };

    window.addEventListener('keydown', onKeyDown);

    return () => window.removeEventListener('keydown', onKeyDown);

  }, [historyPanelOpen]);



  const fetchModels = async () => {

    try {

      const response = await fetch(`http://localhost:800${selectedGpu}/models`);

      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

      const data = await response.json();

      setAvailableModels(data.available_models || []);

    } catch (error) {

      console.error('Error fetching models:', error);

    }

  };



  const handleTabClick = (tab, event) => {

    if (tab === 'settings') {

      openSettingsTab('general', { forceWindow: event?.shiftKey === true ? true : undefined });

      return;

    }

    setActiveTab(tab);

  };



  const handleToggleSidebar = () => {

    setIsOpen(!isOpen);

  };



  const handleNewChat = useCallback(() => {

    setActiveTab('chat');

    createNewConversation();

    setHistoryPanelOpen(false);

  }, [createNewConversation, setActiveTab]);



  const onConversationClick = useCallback(

    (id) => {

      handleConversationClick(id);

      setActiveTab('chat');

      setHistoryPanelOpen(false);

    },

    [handleConversationClick, setActiveTab]

  );



  const onDeleteAllClick = useCallback((e) => {

    setDeleteAllPosition({ x: e.clientX, y: e.clientY });

    setDeleteAllDialogOpen(true);

  }, []);



  const onDeleteConversationClick = useCallback(

    (e, conv) => {

      e.stopPropagation();

      if (autoDeleteChats) {

        void deleteConversation(conv.id);

      } else {

        setDeleteDialogPosition({ x: e.clientX, y: e.clientY });

        setChatToDelete(conv);

        setDeleteDialogOpen(true);

      }

    },

    [autoDeleteChats, deleteConversation]

  );



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

    { id: 'watch', label: 'Watch', icon: <Clapperboard className="w-5 h-5" /> },

    { id: 'settings', label: 'Settings', icon: <Settings className="w-5 h-5" /> },

    { id: 'memory', label: 'Memory tools', icon: <BookOpen className="w-5 h-5" /> },

    { id: 'transcript-corpus', label: 'Transcript search', icon: <FolderSearch className="w-5 h-5" /> },

  ];



  const historyPanelProps = {

    sidebarConversations,

    activeConversation,

    conversationSaveStatus,

    showAllConversations,

    setShowAllConversations,

    onNewChat: handleNewChat,

    onConversationClick,

    onDeleteAllClick,

    onDeleteConversationClick,

  };



  return (

    <>

      {isOpen && <div className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden" onClick={() => setIsOpen(false)} />}



      {historyPanelOpen && (

        <div

          className="hidden md:block fixed inset-0 z-[44]"

          aria-hidden

          onClick={() => setHistoryPanelOpen(false)}

        />

      )}



      <div

        className={cn(

          'relative fixed inset-y-0 left-0 z-50 flex flex-col w-[75px] border-r transition-transform duration-300 ease-in-out md:relative md:translate-x-0',

          isOpen ? 'translate-x-0 max-md:w-72' : '-translate-x-full max-md:w-72',

          'bg-background text-foreground border-r-[0.8px] border-border'

        )}

      >

        {/* NanoGPT-style vertical nav */}

        <div className="flex flex-col items-center py-4 gap-6 flex-1 min-h-0">

          {/* Logo */}

          <button

            type="button"

            className="flex items-center justify-center h-10 w-10 rounded-2xl bg-card shadow-[0_12px_30px_rgba(0,0,0,0.65)] border border-border overflow-hidden"

            onClick={() => {

              setActiveTab('chat');

              goToHome();

            }}

            title="Eloquent Home"

          >

            <img src="/eloquent-logo.png" alt="Eloquent" className="h-7 w-7" />

          </button>

          {/* Main nav icons */}

          <div className="flex-1 flex flex-col items-center gap-3 min-h-0 overflow-y-auto">

            {navItems.map((item) => (

              <button

                key={item.id}

                type="button"

                onClick={(e) => handleTabClick(item.id, e)}

                title={item.label}

                className={cn(

                  'flex items-center justify-center h-10 w-10 rounded-2xl text-[13px] transition-all duration-200',

                  activeTab === item.id

                    ? 'bg-secondary text-foreground shadow-[0_10px_25px_rgba(0,0,0,0.65),0_0_14px_rgba(63,231,252,0.18)] border border-primary/50'

                    : 'bg-transparent text-muted-foreground border border-transparent hover:border-border hover:bg-[var(--nano-bg-hover)]'

                )}

              >

                {item.icon}

              </button>

            ))}

          </div>

          {/* Utility: history, new chat, settings */}

          <div className="flex flex-col items-center gap-3 pb-4 shrink-0">

            <button

              type="button"

              onClick={() => setHistoryPanelOpen((open) => !open)}

              title="Chat history"

              className={cn(

                'hidden md:flex items-center justify-center h-10 w-10 rounded-2xl transition-all duration-200',

                historyPanelOpen

                  ? 'bg-secondary text-foreground shadow-[0_10px_25px_rgba(0,0,0,0.65),0_0_14px_rgba(63,231,252,0.18)] border border-primary/50'

                  : 'bg-transparent text-muted-foreground border border-transparent hover:border-border hover:bg-[var(--nano-bg-hover)]'

              )}

            >

              <History className="h-5 w-5" />

            </button>

            {activeTab === 'chat' && (

              <button

                type="button"

                onClick={handleNewChat}

                title="New chat"

                className="flex items-center justify-center h-10 w-10 rounded-2xl bg-primary text-primary-foreground shadow-[0_10px_25px_rgba(17,24,39,0.9),0_0_16px_rgba(63,231,252,0.22)] hover:brightness-110 transition-all duration-150"

              >

                <Plus className="h-5 w-5" />

              </button>

            )}

            <button

              type="button"

              onClick={(e) => handleTabClick('settings', e)}

              title="Settings"

              className="flex items-center justify-center h-10 w-10 rounded-2xl bg-transparent text-muted-foreground border border-transparent hover:border-border hover:bg-[var(--nano-bg-hover)] transition-all duration-150"

            >

              <Settings className="h-5 w-5" />

            </button>

          </div>

        </div>



        {/* Desktop: flyout chat history panel (collapsed by default) */}

        {historyPanelOpen && (

          <aside

            className="hidden md:flex absolute left-full top-0 bottom-0 z-[55] w-[280px] flex-col bg-background border-r border-border shadow-[12px_0_40px_rgba(8,6,18,0.55)]"

            role="dialog"

            aria-label="Chat history"

          >

            <ConversationHistoryPanel

              {...historyPanelProps}

              onClose={() => setHistoryPanelOpen(false)}

              showCloseButton

            />

          </aside>

        )}



        {/* Mobile close button */}

        <Button

          variant="ghost"

          size="icon"

          onClick={handleToggleSidebar}

          className="md:hidden absolute top-4 right-[-40px] h-9 w-9 rounded-full bg-background border border-border text-muted-foreground shadow-[0_10px_25px_rgba(0,0,0,0.7)]"

        >

          <ChevronLeft className="h-5 w-5" />

        </Button>



        {/* Mobile: conversation list when nav drawer is open */}

        <div className="md:hidden mt-auto border-t border-border flex flex-col min-h-0 max-h-[55vh]">

          {activeTab === 'chat' && (

            <ConversationHistoryPanel {...historyPanelProps} className="min-h-0" />

          )}



          <div className="p-4 border-t mt-auto shrink-0">

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

                  variant={activeTab === item.id ? 'secondary' : 'ghost'}

                  className="w-full justify-start"

                  onClick={(e) => handleTabClick(item.id, e)}

                  title={item.id === 'settings' ? 'Shift+click: always open in new window' : undefined}

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

            void deleteConversation(chatToDelete.id);

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


