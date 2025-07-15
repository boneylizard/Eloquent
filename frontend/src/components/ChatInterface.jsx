import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'; // Or choose another theme
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useApp } from '../contexts/AppContext';
import { cn } from '@/lib/utils'; // Assuming you have this utility for class names
import { Copy, Trash2, Edit2, Check, X } from 'lucide-react'; // Import icons
import { send } from 'process';

// Style for code blocks
const CodeBlock = React.memo(({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeText = String(children).replace(/\n$/, '');
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(codeText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500); // Reset after 1.5 seconds
    });
  }, [codeText]);

  return !inline && match ? (
    <div className="relative group my-4 rounded-md bg-[#282c34] text-sm">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-600">
        <span className="text-gray-400 text-xs font-sans">{match[1]}</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-gray-400 hover:text-white"
          onClick={handleCopy}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </Button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={match[1]}
        PreTag="div"
        {...props}
      >
        {codeText}
      </SyntaxHighlighter>
    </div>
  ) : (
    <code className={cn("font-mono text-sm bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded", className)} {...props}>
      {children}
    </code>
  );
});

const ChatInterface = () => {
  const {
    messages = [], // Default to empty array
    setMessages,
    sendMessage,
    isGenerating,
    activeConversation,
    deleteConversation, // Added for potential future use
    renameConversation, // Added for potential future use
    activeCharacter, // Get active character for avatar/name
    // Add other necessary context values if needed
  } = useApp();

  const [inputMessage, setInputMessage] = useState('');
  const scrollAreaRef = useRef(null);
  const textareaRef = useRef(null);

  // State for editing messages
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [editText, setEditText] = useState('');



  const handleSendMessage = useCallback(() => {
    const trimmedMessage = inputMessage.trim();
    if (!trimmed || isGenerating) return;

    if (dualModeEnabled && primaryModel && secondaryModel) {
      senDualMessage(trimmedMessage); // Function to handle dual model messages
    } else {
      // Send message to the bot
      sendMessage(trimmedMessage);
    }
      setInputMessage(''); // Clear input after sending
    }, [sendMessage, inputMessage, isGenerating, sendDualMessage, dualModeEnabled, primaryModel, secondaryModel]);

  const handleKeyDown = useCallback((event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Prevent default Enter behavior (new line)
      handleSendMessage();
    }
  }, [handleSendMessage]);

  // Handle starting edit
  const handleEdit = (message) => {
    setEditingMessageId(message.id);
    setEditText(message.content);
  };

  // Handle saving edit
  const handleSaveEdit = (messageId) => {
    setMessages(prevMessages =>
      prevMessages.map(msg =>
        msg.id === messageId ? { ...msg, content: editText } : msg
      )
    );
    setEditingMessageId(null);
    setEditText('');
    // TODO: Potentially re-send context or trigger regeneration if needed after edit?
  };

  // Handle canceling edit
  const handleCancelEdit = () => {
    setEditingMessageId(null);
    setEditText('');
  };

  // Handle deleting a message
  const handleDeleteMessage = (messageId) => {
      if (window.confirm("Are you sure you want to delete this message?")) {
          setMessages(prevMessages => prevMessages.filter(msg => msg.id !== messageId));
          // Note: This only deletes from the current view.
          // If conversation history is persisted, ensure it's updated there too.
      }
  };

useEffect(() => {
  if (!isGenerating && !isRecording && !isTranscribing) {
  const inputRef = textareaRef.current;
 inputRef.current?.focus();

 inputRef.current?.focus({ preventScroll: true });
}
}, [isGenerating, isRecording, isTranscribing]);

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900">
      {/* Message Display Area */}
      <ScrollArea ref={scrollAreaRef} className="flex-grow p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex items-start space-x-2",
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              {/* Bot Avatar */}
              {message.role === 'bot' && (
                <div className="flex-shrink-0">
                  {/* --- AVATAR IMAGE --- */}
                  {message.avatar ? (
                    <img
                      src={message.avatar}
                      alt={message.characterName || 'Bot'}
                      onError={(e) => { e.target.style.display = 'none'; }}
                      // --- ADDED/MODIFIED className for sizing ---
                      // --- You can adjust w-8 h-8 (32px) to w-10 h-10 (40px) or other Tailwind sizes ---
                      className="w-14 h-14 rounded-full mr-2 flex-shrink-0 object-cover border border-gray-300 dark:border-gray-600"
                      // --- End of added/modified className ---
                    />
                  ) : (
                    // Placeholder if no avatar
                    <div className="w-14 h-14 rounded-full mr-2 flex-shrink-0 bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-sm font-semibold text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600">
                      {message.characterName ? message.characterName.charAt(0).toUpperCase() : 'B'}
                    </div>
                  )}
                </div>
              )}

              {/* Message Content Bubble */}
              <div
                className={cn(
                  "max-w-[75%] p-3 rounded-lg shadow-sm",
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100',
                  message.role === 'system' && 'bg-yellow-100 dark:bg-yellow-900 text-yellow-900 dark:text-yellow-100 text-xs italic text-center w-full max-w-full',
                  message.error && 'bg-red-100 dark:bg-red-900 text-red-900 dark:text-red-100'
                )}
              >
                {/* Bot Name/Model */}
                {message.role === 'bot' && (
                  <p className="text-xs font-semibold mb-1 text-gray-600 dark:text-gray-400">
                    {message.characterName || message.modelName || 'Bot'}
                    {message.modelId && ` (${message.modelId})`} {/* Show primary/secondary */}
                  </p>
                )}

                {/* Editable Message Content */}
                {editingMessageId === message.id ? (
                  <div className="space-y-2">
                    <Textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      className="w-full text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded"
                      rows={Math.max(3, editText.split('\n').length)} // Adjust rows based on content
                    />
                    <div className="flex space-x-2 justify-end">
                      <Button variant="ghost" size="sm" onClick={() => handleSaveEdit(message.id)} className="text-green-600 hover:text-green-700">
                        <Check size={16} />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={handleCancelEdit} className="text-red-600 hover:text-red-700">
                        <X size={16} />
                      </Button>
                    </div>
                  </div>
                ) : (
                  // Display Message Content (Markdown)
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{ code: CodeBlock }}
                    className="prose prose-sm dark:prose-invert max-w-none" // prose classes for markdown styling
                  >
                    {message.content}
                  </ReactMarkdown>
                )}

                {/* Message Actions (Copy, Edit, Delete) - Show on hover */}
                 {!editingMessageId && (message.role === 'user' || message.role === 'bot') && (
                    <div className="message-actions opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex justify-end space-x-1 mt-1">
                        <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400" title="Copy"
                            onClick={() => navigator.clipboard.writeText(message.content)}>
                            <Copy size={12} />
                        </Button>
                        {/* Allow editing user messages and non-generating bot messages */}
                        {message.role === 'user' || (message.role === 'bot' && !isGenerating) ? (
                             <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400" title="Edit"
                                onClick={() => handleEdit(message)}>
                                <Edit2 size={12} />
                            </Button>
                        ) : null}
                         <Button variant="ghost" size="icon" className="h-5 w-5 text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400" title="Delete"
                            onClick={() => handleDeleteMessage(message.id)}>
                            <Trash2 size={12} />
                        </Button>
                    </div>
                 )}
              </div>
              <div className="flex items-center mb-2">
  <Switch 
    id="use_rag" 
    checked={settings.use_rag} 
    onCheckedChange={(checked) => updateSettings({use_rag: checked})}
  />
  <Label htmlFor="use_rag" className="ml-2">Use documents for context</Label>
</div>
              {/* User Avatar (Optional - Placeholder) */}
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-semibold text-sm flex-shrink-0 ml-2 border border-blue-600">
                  {/* Placeholder - could use user profile initial later */}
                  U
                </div>
              )}
            </div>
          ))}
          {/* Loading Indicator */}
          {isGenerating && (
            <div className="flex items-center justify-start space-x-2">
                {/* Placeholder for Bot Avatar during generation */}
                <div className="w-8 h-8 rounded-full mr-2 flex-shrink-0 bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-sm font-semibold text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600">
                    {activeCharacter ? activeCharacter.name.charAt(0).toUpperCase() : 'B'}
                </div>
               <div className="p-3 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
                    <div className="flex space-x-1 items-center">
                        <span className="text-xs font-semibold mb-1 text-gray-600 dark:text-gray-400">
                            {activeCharacter?.name || 'Bot'} is thinking
                        </span>
                        <div className="typing-indicator">
                            <span className="dot"></span>
                            <span className="dot"></span>
                            <span className="dot"></span>
                        </div>
                    </div>
                </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="relative flex items-end space-x-2">
          <Textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="flex-grow resize-none overflow-hidden pr-16 rounded-md border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white"
            rows={1} // Start with 1 row, auto-expands
            style={{ minHeight: '40px', maxHeight: '200px' }} // Set min/max height
            disabled={isGenerating}
          />
          <Button
            onClick={handleSendMessage}
            disabled={isGenerating || !inputMessage.trim()}
            className="absolute bottom-2 right-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md px-4 py-1.5 disabled:opacity-50"
          >
            Send
          </Button>
        </div>
      </div>

      {/* Basic CSS for Typing Indicator */}
      <style jsx>{`
        .typing-indicator { display: inline-flex; align-items: center; margin-left: 4px; }
        .dot { width: 5px; height: 5px; background-color: currentColor; border-radius: 50%; margin: 0 2px; animation: typing 1s infinite ease-in-out; }
        .dot:nth-child(1) { animation-delay: 0s; }
        .dot:nth-child(2) { animation-delay: 0.1s; }
        .dot:nth-child(3) { animation-delay: 0.2s; }
        @keyframes typing {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
        .message-actions { /* Ensure actions container is positioned correctly if needed */ }
      `}</style>
    </div>
  );
};

export default ChatInterface;
