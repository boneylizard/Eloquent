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

  const isDisabled = !primaryModel || isGenerating || isModelLoading || agentConversationActive || isRecording || isTranscribing;
  const placeholderText = 
      !primaryModel ? "Load a model first" :
      isRecording ? "Recording... Click mic to stop" :
      isTranscribing ? "Transcribing..." :
      isGenerating ? "Generating response..." :
      webSearchEnabled ? "Type a message (web search enabled)..." :
      "Type a message or click mic...";

  return (
    <form className="border-t border-border p-4 flex items-center gap-2 bg-background" onSubmit={handleSubmit}>
      <div className="relative flex-1">
        <Textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isDisabled}
          className="flex-1 resize-none border-input bg-background pr-20" // Increased padding for buttons
          rows={1}
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