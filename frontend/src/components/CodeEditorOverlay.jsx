import React, { useState, useRef, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { getBackendUrl } from '../config/api';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import {
  X, Code, FileText, Loader2, FolderTree, Folder, FolderOpen,
  ChevronRight, ChevronDown, File, Search, RefreshCw, Settings,
  Send, Terminal, Image, Trash2, Plus, Zap, History, Menu
} from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

// ============================================================================
// DIRECTORY PICKER COMPONENT
// ============================================================================
const DirectoryPicker = ({ isOpen, onClose, onSelectDirectory, currentDir }) => {
  const [inputPath, setInputPath] = useState(currentDir || '');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!inputPath.trim()) return;

    setIsLoading(true);
    setError('');

    try {
      const baseUrl = import.meta.env.VITE_API_URL || getBackendUrl();
      const response = await fetch(`${baseUrl}/code_editor/set_base_dir?path=${encodeURIComponent(inputPath)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to set directory');
      }

      onSelectDirectory(inputPath);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-zinc-900 border-zinc-700">
        <DialogHeader>
          <DialogTitle className="text-zinc-100">Select Project Directory</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-zinc-300">Directory Path:</label>
            <Input
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder="C:\path\to\your\project"
              className="mt-1 bg-zinc-800 border-zinc-600 text-zinc-100"
            />
            {error && (
              <p className="text-sm text-red-400 mt-1">{error}</p>
            )}
          </div>
          <div className="text-xs text-zinc-400">
            Enter the absolute path to your project directory.
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} className="border-zinc-600">Cancel</Button>
          <Button onClick={handleSubmit} disabled={isLoading || !inputPath.trim()} className="bg-emerald-600 hover:bg-emerald-700">
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Select Directory'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

// ============================================================================
// FILE TREE COMPONENT
// ============================================================================
const FileTreeItem = ({ item, level = 0, onFileSelect, onToggleFolder }) => {
  if (!item) return null;

  const handleClick = () => {
    if (item.type === 'folder') {
      onToggleFolder(item);
    } else {
      onFileSelect(item);
    }
  };

  const handleDragStart = (e) => {
    if (item.type === 'file') {
      e.dataTransfer.setData('text/plain', item.name);
      e.dataTransfer.setData('application/json', JSON.stringify(item));
    }
  };

  return (
    <div>
      <div
        className={cn(
          "flex items-center gap-1 py-1 px-2 hover:bg-zinc-700/50 cursor-pointer text-sm",
          "transition-colors duration-150 text-zinc-300"
        )}
        style={{ paddingLeft: `${8 + level * 16}px` }}
        onClick={handleClick}
        draggable={item.type === 'file'}
        onDragStart={handleDragStart}
      >
        {item.type === 'folder' ? (
          <>
            {item.expanded ? (
              <ChevronDown className="h-3 w-3 text-zinc-500" />
            ) : (
              <ChevronRight className="h-3 w-3 text-zinc-500" />
            )}
            {item.expanded ? (
              <FolderOpen className="h-4 w-4 text-amber-500" />
            ) : (
              <Folder className="h-4 w-4 text-amber-500" />
            )}
          </>
        ) : (
          <>
            <div className="w-3" />
            <File className="h-4 w-4 text-zinc-400" />
          </>
        )}
        <span className="truncate flex-1 min-w-0">{item.name}</span>
      </div>
      {item.type === 'folder' && item.expanded && item.children && (
        <div>
          {item.children.map((child, index) => (
            <FileTreeItem
              key={index}
              item={child}
              level={level + 1}
              onFileSelect={onFileSelect}
              onToggleFolder={onToggleFolder}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// MAIN CODE EDITOR OVERLAY COMPONENT
// ============================================================================
const CodeEditorOverlay = ({ isOpen = true, onClose }) => {
  const { activeModel, primaryIsAPI, primaryModel } = useApp();
  const [input, setInput] = useState('');
  const [codeMessages, setCodeMessages] = useState([]);
  const [draggedFiles, setDraggedFiles] = useState([]);
  const [fileTree, setFileTree] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPath, setCurrentPath] = useState('');
  const [isLoadingTree, setIsLoadingTree] = useState(false);
  const [showDirectoryPicker, setShowDirectoryPicker] = useState(false);
  const [isCodeGenerating, setIsCodeGenerating] = useState(false);
  const [devstralStatus, setDevstralStatus] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  // Mobile View State: 'chat' | 'explorer' | 'sessions'
  const [mobileView, setMobileView] = useState('chat');
  
  const scrollAreaRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  const baseUrl = import.meta.env.VITE_API_URL || getBackendUrl();

  // ============================================================================
  // API HELPER
  // ============================================================================
  const apiCall = async (endpoint, options = {}) => {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API call failed: ${response.status}`);
    }

    return response.json();
  };

  // ============================================================================
  // DEVSTRAL STATUS CHECK
  // ============================================================================
  const checkDevstralStatus = async () => {
    const modelId = (primaryIsAPI && primaryModel) ? primaryModel : (typeof activeModel === 'string' ? activeModel : activeModel?.id);

    if (modelId && modelId.startsWith('endpoint-')) {
      setDevstralStatus({ devstral_loaded: true, is_api: true });
      return;
    }

    try {
      const status = await apiCall('/devstral/status');
      setDevstralStatus(status);
    } catch (error) {
      console.error('Failed to check Devstral status:', error);
      setDevstralStatus({ devstral_loaded: false });
    }
  };

  // ============================================================================
  // FILE TREE OPERATIONS
  // ============================================================================
  const loadFileTree = async () => {
    setIsLoadingTree(true);
    try {
      const result = await apiCall('/code_editor/get_tree');
      setFileTree(result.tree);
      setCurrentPath(result.base_path);
    } catch (error) {
      console.error('Error loading file tree:', error);
      setShowDirectoryPicker(true);
    } finally {
      setIsLoadingTree(false);
    }
  };

  const handleDirectorySelect = (newPath) => {
    setCurrentPath(newPath);
    loadFileTree();
  };

  const handleFileSelect = async (file) => {
    try {
      const result = await apiCall('/code_editor/read_file', {
        method: 'POST',
        body: JSON.stringify({ filepath: file.name })
      });

      const fileMessage = `📁 **${file.name}**\n\`\`\`\n${result.content}\n\`\`\``;

      setCodeMessages(prev => [...prev, {
        role: 'assistant',
        content: fileMessage,
        timestamp: Date.now(),
        tools_used: ['read_file']
      }]);
      
      // Auto-switch to chat on mobile when a file is selected
      setMobileView('chat');
    } catch (error) {
      console.error('Error reading file:', error);
    }
  };

  const handleToggleFolder = (folder) => {
    const toggleFolder = (items) => {
      return items.map(item => {
        if (item === folder) {
          return { ...item, expanded: !item.expanded };
        }
        if (item.children) {
          return { ...item, children: toggleFolder(item.children) };
        }
        return item;
      });
    };

    setFileTree(prev => prev ? toggleFolder([prev])[0] : null);
  };

  const filteredTree = (node, search) => {
    if (!search) return node;
    if (!node) return null;

    const searchLower = search.toLowerCase();

    if (node.type === 'file') {
      return node.name.toLowerCase().includes(searchLower) ? node : null;
    }

    if (node.children) {
      const filteredChildren = node.children
        .map(child => filteredTree(child, search))
        .filter(Boolean);

      if (filteredChildren.length > 0 || node.name.toLowerCase().includes(searchLower)) {
        return { ...node, expanded: true, children: filteredChildren };
      }
    }

    return null;
  };

  // ============================================================================
  // CHAT SUBMISSION
  // ============================================================================
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isCodeGenerating) return;

    let sessionId = activeSessionId;
    if (!sessionId) {
      sessionId = createNewSession(input);
    }

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: Date.now(),
      files: draggedFiles.length > 0 ? [...draggedFiles] : undefined,
      hasImage: !!imagePreview
    };

    const newMessages = [...codeMessages, userMessage];
    setCodeMessages(newMessages);
    updateActiveSession(newMessages);

    const currentInput = input;
    const currentImage = imagePreview;
    setInput('');
    setDraggedFiles([]);
    setImagePreview(null);
    setIsCodeGenerating(true);

    try {
      const conversationMessages = newMessages.map(msg => ({
        role: msg.role,
        content: msg.files
          ? `${msg.content}\n\nFiles in context: ${msg.files.join(', ')}`
          : msg.content
      }));

      const response = await apiCall('/devstral/chat', {
        method: 'POST',
        body: JSON.stringify({
          messages: conversationMessages,
          working_dir: currentPath,
          temperature: 0.15,
          max_tokens: 4096,
          image_base64: currentImage,
          auto_execute: true,
          model: (primaryIsAPI && primaryModel)
            ? primaryModel
            : ((typeof activeModel === 'string' ? activeModel : activeModel?.id) || 'devstral-small')
        })
      });

      let assistantContent = '';
      let toolsUsed = [];

      if (response.choices && response.choices[0]) {
        const message = response.choices[0].message;
        assistantContent = message.content || '';

        if (message.tool_calls && message.tool_calls.length > 0) {
          toolsUsed = message.tool_calls.map(tc => tc.function?.name).filter(Boolean);
        }
      }

      if (response.tool_results && response.tool_results.length > 0) {
        const toolResultsText = response.tool_results.map(tr => {
          const icon = tr.success ? '✅' : '❌';
          return `${icon} **${tr.name}**:\n\`\`\`\n${tr.result}\n\`\`\``;
        }).join('\n\n');

        assistantContent = assistantContent
          ? `${assistantContent}\n\n---\n\n**Tool Results:**\n${toolResultsText}`
          : toolResultsText;

        toolsUsed = response.tool_results.map(tr => tr.name);
      }

      const aiMessage = {
        role: 'assistant',
        content: assistantContent || "I've completed the requested operations.",
        timestamp: Date.now(),
        tools_used: toolsUsed
      };

      setCodeMessages(prev => {
        const next = [...prev, aiMessage];
        updateActiveSession(next);
        return next;
      });

    } catch (error) {
      console.error('❌ Devstral chat error:', error);
      const errorMessage = {
        role: 'assistant',
        content: `❌ Error: ${error.message}\n\nMake sure Devstral Small 2 24B is loaded.`,
        timestamp: Date.now(),
        tools_used: []
      };
      setCodeMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsCodeGenerating(false);
    }
  };

  // ============================================================================
  // IMAGE HANDLING
  // ============================================================================
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result.split(',')[1];
      setImagePreview(base64);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ============================================================================
  // DRAG AND DROP
  // ============================================================================
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedJSON = e.dataTransfer.getData('application/json');
    if (droppedJSON) {
      try {
        const fileData = JSON.parse(droppedJSON);
        if (!draggedFiles.includes(fileData.name)) {
          setDraggedFiles(prev => [...prev, fileData.name]);
        }
      } catch (err) {
        console.error('Error parsing dropped file data:', err);
      }
    }
  };

  const removeDraggedFile = (fileName) => {
    setDraggedFiles(prev => prev.filter(f => f !== fileName));
  };

  // ============================================================================
  // SESSION MANAGEMENT
  // ============================================================================
  const SESSIONS_KEY = 'devstral2Sessions';

  const loadSessions = () => {
    try {
      const raw = localStorage.getItem(SESSIONS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  };

  const saveSessions = (next) => {
    try {
      localStorage.setItem(SESSIONS_KEY, JSON.stringify(next));
    } catch { }
  };

  const createNewSession = (firstMessage) => {
    const id = `sess_${Date.now()}`;
    const title = (firstMessage || '').slice(0, 50) || 'New Session';
    const session = { id, title, createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
    const next = [session, ...sessions];
    setSessions(next);
    saveSessions(next);
    setActiveSessionId(id);
    setCodeMessages([]);
    return id;
  };

  const updateActiveSession = (messagesForSession) => {
    if (!activeSessionId) return;
    const next = sessions.map(s =>
      s.id === activeSessionId
        ? { ...s, messages: messagesForSession, updatedAt: Date.now() }
        : s
    );
    setSessions(next);
    saveSessions(next);
  };

  const loadSession = (id) => {
    const found = sessions.find(s => s.id === id);
    if (!found) return;
    setActiveSessionId(id);
    setCodeMessages(found.messages || []);
    setMobileView('chat'); // Auto-switch on mobile
  };

  const deleteSession = (id) => {
    const next = sessions.filter(s => s.id !== id);
    setSessions(next);
    saveSessions(next);
    if (activeSessionId === id) {
      setActiveSessionId(null);
      setCodeMessages([]);
    }
  };

  // ============================================================================
  // EFFECTS
  // ============================================================================
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [codeMessages, mobileView]); // Added mobileView dependency for re-layout

  useEffect(() => {
    if (isOpen) {
      loadFileTree();
      checkDevstralStatus();
    }
  }, [isOpen]);

  useEffect(() => {
    const loaded = loadSessions();
    setSessions(loaded);
    if (loaded.length > 0 && !activeSessionId) {
      setActiveSessionId(loaded[0].id);
      setCodeMessages(loaded[0].messages || []);
    }
  }, []);

  useEffect(() => {
    if (isOpen && inputRef.current && window.innerWidth > 768) {
      setTimeout(() => inputRef.current.focus(), 100);
    }
  }, [isOpen]);

  // ============================================================================
  // CODE BLOCK COMPONENT
  // ============================================================================
  const CodeBlock = ({ language, children }) => (
    <SyntaxHighlighter
      style={oneDark}
      language={language || 'text'}
      customStyle={{
        margin: 0,
        borderRadius: '0.375rem',
        fontSize: '0.8rem',
        background: '#1a1a2e'
      }}
    >
      {children}
    </SyntaxHighlighter>
  );

  if (!isOpen) return null;

  // ============================================================================
  // RENDER
  // ============================================================================
  return (
    <div className="fixed inset-0 bg-black/80 flex items-end md:items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-700 md:rounded-lg shadow-2xl w-full h-[100dvh] md:w-[95vw] md:h-[90vh] flex flex-col max-w-7xl">

        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-700 bg-zinc-800/50 md:rounded-t-lg shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-lg">
              <Terminal className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">Eloquent Vibe</h2>
              <p className="text-xs text-zinc-400 hidden sm:block">
                Powered by {(primaryIsAPI && primaryModel) ? primaryModel : (activeModel?.name || "Devstral Small 2 24B")}
              </p>
            </div>
            {devstralStatus && (
              <div className={cn(
                "hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full text-xs",
                devstralStatus.devstral_loaded
                  ? "bg-emerald-500/20 text-emerald-400"
                  : "bg-red-500/20 text-red-400"
              )}>
                <Zap className="h-3 w-3" />
                {devstralStatus.devstral_loaded ? 'Ready' : 'No Model'}
              </div>
            )}
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="text-zinc-400 hover:text-zinc-100">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden relative">

          {/* File Explorer */}
          <div className={cn(
            "border-r border-zinc-700 bg-zinc-800/30 flex-col",
            // Mobile: Full width if active, hidden otherwise
            // Desktop: Always flex, fixed width
            mobileView === 'explorer' ? 'flex w-full h-full' : 'hidden md:flex md:w-72'
          )}>
            <div className="p-3 border-b border-zinc-700 shrink-0">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium text-zinc-300">Explorer</h3>
                <div className="flex gap-1">
                  <Button variant="ghost" size="sm" onClick={() => setShowDirectoryPicker(true)} className="h-6 w-6 p-0 text-zinc-400">
                    <Settings className="h-3.5 w-3.5" />
                  </Button>
                  <Button variant="ghost" size="sm" onClick={loadFileTree} disabled={isLoadingTree} className="h-6 w-6 p-0 text-zinc-400">
                    <RefreshCw className={cn("h-3.5 w-3.5", isLoadingTree && "animate-spin")} />
                  </Button>
                </div>
              </div>

              <div className="relative">
                <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3 w-3 text-zinc-500" />
                <Input
                  placeholder="Search files..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-7 h-7 text-xs bg-zinc-800 border-zinc-600 text-zinc-300"
                />
              </div>

              <div className="mt-2 text-xs text-zinc-500 truncate bg-zinc-800/50 p-2 rounded" title={currentPath}>
                📁 {currentPath || 'No directory selected'}
              </div>
            </div>

            <ScrollArea className="flex-1">
              <div className="p-1 pb-16 md:pb-1">
                {isLoadingTree ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
                  </div>
                ) : fileTree ? (
                  <FileTreeItem
                    item={filteredTree(fileTree, searchTerm)}
                    onFileSelect={handleFileSelect}
                    onToggleFolder={handleToggleFolder}
                  />
                ) : (
                  <div className="text-center py-8 text-zinc-500">
                    <FolderTree className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No directory selected</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2 border-zinc-600"
                      onClick={() => setShowDirectoryPicker(true)}
                    >
                      Select Directory
                    </Button>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Sessions Sidebar */}
          <div className={cn(
            "border-r border-zinc-700 bg-zinc-800/20 flex-col",
            // Mobile: Full width if active, hidden otherwise
            // Desktop: Always flex, fixed width
            mobileView === 'sessions' ? 'flex w-full h-full' : 'hidden md:flex md:w-56'
          )}>
            <div className="p-3 border-b border-zinc-700 shrink-0">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-zinc-300">Sessions</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    createNewSession('');
                    setMobileView('chat');
                  }}
                  className="h-7 text-xs gap-1 border-zinc-600 bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
                >
                  <Plus className="h-3 w-3" />
                  New Chat
                </Button>
              </div>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-1 space-y-1 pb-16 md:pb-1">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={cn(
                      "group flex items-center justify-between p-2 rounded cursor-pointer text-sm transition-colors",
                      activeSessionId === session.id
                        ? "bg-emerald-600/30 text-emerald-300"
                        : "hover:bg-zinc-700/50 text-zinc-400"
                    )}
                  >
                    <div onClick={() => loadSession(session.id)} className="flex-1 min-w-0">
                      <div className="font-medium truncate">{session.title}</div>
                      <div className="text-xs opacity-70">
                        {new Date(session.updatedAt).toLocaleDateString()}
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); deleteSession(session.id); }}
                      className="h-5 w-5 p-0 text-zinc-500 hover:text-white hover:bg-zinc-700 rounded-full"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
                {sessions.length === 0 && (
                  <div className="text-center py-4 text-zinc-500 text-xs">
                    No sessions yet
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Chat Area */}
          <div className={cn(
            "flex-col min-w-0 bg-zinc-900",
            // Mobile: Full width if active, hidden otherwise
            // Desktop: Always flex, take remaining space
            mobileView === 'chat' ? 'flex flex-1 w-full' : 'hidden md:flex md:flex-1'
          )}>

            {/* Messages */}
            <ScrollArea className="flex-1 p-2 md:p-4" ref={scrollAreaRef}>
              <div className="space-y-4 max-w-4xl mx-auto pb-4">
                {codeMessages.length === 0 ? (
                  <div className="text-center py-8 md:py-16 text-zinc-500">
                    <div className="p-4 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-2xl w-16 h-16 md:w-20 md:h-20 mx-auto mb-6 flex items-center justify-center">
                      <Terminal className="h-8 w-8 md:h-10 md:w-10 text-emerald-400" />
                    </div>
                    <h3 className="text-lg md:text-xl font-semibold mb-2 text-zinc-300">Welcome to Eloquent Vibe</h3>
                    <p className="text-xs md:text-sm max-w-md mx-auto text-zinc-500 px-4">
                      Your local AI coding assistant.
                      Ask me to read files, write code, search your codebase, or run commands.
                    </p>
                    <div className="mt-6 flex flex-wrap gap-2 justify-center px-4">
                      {['List files in this directory', 'Search for TODO comments', 'Help me refactor this code'].map((suggestion) => (
                        <button
                          key={suggestion}
                          onClick={() => setInput(suggestion)}
                          className="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-400 rounded-full transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  codeMessages.map((message, index) => (
                    <div key={index} className={cn(
                      "flex gap-2 md:gap-3",
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    )}>
                      <div className={cn(
                        "max-w-[95%] md:max-w-[85%] rounded-xl p-3 md:p-4",
                        message.role === 'user'
                          ? 'bg-emerald-600/30 text-zinc-100 ml-8'
                          : 'bg-zinc-800 text-zinc-200 mr-8'
                      )}>
                        <div className="prose prose-sm max-w-none prose-invert text-xs md:text-sm">
                          <ReactMarkdown
                            components={{
                              code({ node, inline, className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '');
                                return !inline && match ? (
                                  <CodeBlock language={match[1]}>
                                    {String(children).replace(/\n$/, '')}
                                  </CodeBlock>
                                ) : (
                                  <code className="bg-zinc-700/50 px-1.5 py-0.5 rounded text-xs text-emerald-300" {...props}>
                                    {children}
                                  </code>
                                );
                              }
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                        {message.tools_used && message.tools_used.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-zinc-700/50 text-[10px] md:text-xs text-zinc-500 flex items-center gap-1">
                            <Terminal className="h-3 w-3" />
                            {message.tools_used.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}

                {isCodeGenerating && (
                  <div className="flex gap-3 justify-start">
                    <div className="bg-zinc-800 rounded-xl p-4 flex items-center gap-2 text-zinc-400">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm">Devstral is thinking...</span>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="border-t border-zinc-700 p-2 md:p-4 bg-zinc-800/30 shrink-0 mb-14 md:mb-0">
              
              {/* Attached Files/Image Preview */}
              {(draggedFiles.length > 0 || imagePreview) && (
                <div className="mb-2 flex flex-wrap gap-2">
                  {draggedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-1 bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded text-xs"
                    >
                      <FileText className="h-3 w-3" />
                      {file}
                      <button onClick={() => removeDraggedFile(file)} className="ml-1 hover:text-red-400">
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                  {imagePreview && (
                    <div className="relative">
                      <img
                        src={`data:image/png;base64,${imagePreview}`}
                        alt="Preview"
                        className="h-12 w-12 md:h-16 md:w-16 object-cover rounded border border-zinc-600"
                      />
                      <button
                        onClick={removeImage}
                        className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
              )}

              <form onSubmit={handleSubmit} className="flex gap-2">
                <div
                  className="flex-1"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <Textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask Devstral..."
                    className="resize-none min-h-[50px] md:min-h-[80px] max-h-32 bg-zinc-800 border-zinc-600 text-zinc-100 placeholder:text-zinc-500 text-sm"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                  />
                </div>
                <div className="flex flex-col gap-2">
                  {/* Vision Upload Button */}
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 border-zinc-600 text-zinc-400 hover:text-zinc-100 px-2 md:px-4"
                    title="Upload image for vision"
                  >
                    <Image className="h-4 w-4" />
                  </Button>

                  <Button
                    type="submit"
                    disabled={!input.trim() || isCodeGenerating}
                    className="flex-1 bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-700 hover:to-cyan-700 px-2 md:px-4"
                  >
                    {isCodeGenerating ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </form>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Bar (Visible only on mobile) */}
        <div className="md:hidden border-t border-zinc-800 bg-zinc-900 absolute bottom-0 left-0 right-0 h-14 flex items-center justify-around z-10 pb-safe">
          <button
            onClick={() => setMobileView('explorer')}
            className={cn(
              "flex flex-col items-center justify-center w-full h-full text-[10px] gap-1",
              mobileView === 'explorer' ? "text-emerald-400" : "text-zinc-500"
            )}
          >
            <FolderTree className="h-5 w-5" />
            <span>Explorer</span>
          </button>
          
          <button
            onClick={() => setMobileView('chat')}
            className={cn(
              "flex flex-col items-center justify-center w-full h-full text-[10px] gap-1",
              mobileView === 'chat' ? "text-emerald-400" : "text-zinc-500"
            )}
          >
            <Terminal className="h-5 w-5" />
            <span>Chat</span>
          </button>
          
          <button
            onClick={() => setMobileView('sessions')}
            className={cn(
              "flex flex-col items-center justify-center w-full h-full text-[10px] gap-1",
              mobileView === 'sessions' ? "text-emerald-400" : "text-zinc-500"
            )}
          >
            <FileText className="h-5 w-5" />
            <span>Sessions</span>
          </button>
        </div>

        {/* Directory Picker Modal */}
        <DirectoryPicker
          isOpen={showDirectoryPicker}
          onClose={() => setShowDirectoryPicker(false)}
          onSelectDirectory={handleDirectorySelect}
          currentDir={currentPath}
        />
      </div>
    </div>
  );
};

export default CodeEditorOverlay;