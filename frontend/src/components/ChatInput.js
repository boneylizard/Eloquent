// --- ChatInput.jsx - Enhanced with Memory Intent Detection ---

import React, { useState, useRef, useEffect } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import MemoryIntentDetector from './MemoryIntentDetector';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Send, Mic, MicOff } from 'lucide-react';

const ChatInput = ({ onSendMessage, isGenerating, onStartRecording, onStopRecording, isRecording }) => {
  const [message, setMessage] = useState('');
  const [hasMemoryIntent, setHasMemoryIntent] = useState(false);
  const textareaRef = useRef(null);
  const { detectMemoryIntent } = useMemory();
  
  useEffect(() => {
    // Focus the textarea when the component mounts
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);
  
  const handleInputChange = (e) => {
    const text = e.target.value;
    setMessage(text);
    
    // Check for memory intent
    setHasMemoryIntent(detectMemoryIntent(text));
  };
  
  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const handleSendMessage = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !isGenerating) {
      onSendMessage(trimmedMessage);
      setMessage('');
      setHasMemoryIntent(false);
    }
  };
  
  return (
    <div className="chat-input-container">
      <div className="relative">
        <Textarea
          ref={textareaRef}
          value={message}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          className={`min-h-[60px] resize-none pr-14 ${hasMemoryIntent ? 'border-yellow-500 focus-visible:ring-yellow-500' : ''}`}
          rows={1}
          disabled={isGenerating}
        />
        
        <div className="absolute right-2 bottom-2 flex gap-2">
          {onStartRecording && onStopRecording && (
            <Button
              size="icon"
              variant="ghost"
              onClick={isRecording ? onStopRecording : onStartRecording}
              disabled={isGenerating}
              className={isRecording ? "text-red-500 animate-pulse" : ""}
            >
              {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
            </Button>
          )}
          
          <Button
            size="icon"
            onClick={handleSendMessage}
            disabled={!message.trim() || isGenerating}
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
      
      {/* Memory intent detector below the input */}
      {hasMemoryIntent && (
        <div className="mt-1">
          <MemoryIntentDetector 
            text={message}
            onDetected={() => {}}  // Could add notifications or suggestions here
          />
        </div>
      )}
    </div>
  );
};

export default ChatInput;