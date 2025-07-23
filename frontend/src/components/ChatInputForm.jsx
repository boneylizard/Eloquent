import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';
import ChatImageUploadButton from './ChatImageUploadButton';

const ChatInputForm = ({ 
    onSubmit, 
    isGenerating, 
    isModelLoading, 
    isRecording, 
    isTranscribing,
    agentConversationActive, 
    primaryModel,
    webSearchEnabled,
    inputValue,
    setInputValue
}) => {
  const inputRef = useRef(null);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmedValue = inputValue.trim();
    if (trimmedValue) {
      onSubmit(trimmedValue);
      setInputValue('');
    }
  };
  
  // Auto-focus logic moved here
  useEffect(() => {
    if (!isGenerating && !isRecording && !isTranscribing) {
      inputRef.current?.focus({ preventScroll: true });
    }
  }, [isGenerating, isRecording, isTranscribing]);
useEffect(() => {
  const textarea = inputRef.current;
  if (!textarea) return;
  
  // Reset height to get accurate scrollHeight
  textarea.style.height = 'auto';
  
  // Calculate line height dynamically
  const computedStyle = window.getComputedStyle(textarea);
  const lineHeight = parseInt(computedStyle.lineHeight) || 24; // fallback to 24px
  
  const minHeight = lineHeight; // 1 line
  const maxHeight = lineHeight * 16; // 16 lines exactly
  
  const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
  textarea.style.height = newHeight + 'px';
  
  // Show scrollbar only when we hit 16 lines
  if (textarea.scrollHeight > maxHeight) {
    textarea.style.overflowY = 'auto';
  } else {
    textarea.style.overflowY = 'hidden';
  }
}, [inputValue]);
  const isDisabled = !primaryModel || isGenerating || isModelLoading || agentConversationActive || isRecording || isTranscribing;
  const placeholderText = 
      !primaryModel ? "Load a model first" :
      isRecording ? "Recording... Click mic to stop" :
      isTranscribing ? "Transcribing..." :
      isGenerating ? "Generating response..." :
      webSearchEnabled ? "Type a message (web search enabled)..." :
      "Type a message or click mic...";

  return (
    <form className="border-t border-border p-4 pb-12 flex items-center gap-2 bg-background" onSubmit={handleSubmit}>
      <div className="relative flex-1">
<Textarea
  ref={inputRef}
  value={inputValue}
  onChange={(e) => setInputValue(e.target.value)}
  onKeyDown={handleKeyDown}
  placeholder={placeholderText}
  disabled={isDisabled}
  className="flex-1 resize-none border-input bg-background pr-20"
  rows={1}
  style={{ 
    minHeight: '24px', 
    overflowY: 'hidden',
    transition: 'height 0.1s ease'
  }}
/>
        <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
          <SimpleChatImageButton />
          <ChatImageUploadButton />
        </div>
      </div>
      <Button
        type="submit"
        disabled={!inputValue.trim() || isDisabled}
        size="icon"
        className="h-10 w-10"
      >
        {isGenerating ? <Loader2 className="animate-spin" size={18}/> : <Send size={18}/>}
      </Button>
    </form>
  );
};

export default ChatInputForm;