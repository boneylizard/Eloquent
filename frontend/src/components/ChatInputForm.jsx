import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, ArrowLeft } from 'lucide-react';
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
  setInputValue,
  performanceMode,
  onBack,
  canGoBack
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

  useEffect(() => {
    if (!isGenerating && !isRecording && !isTranscribing) {
      inputRef.current?.focus({ preventScroll: true });
    }
  }, [isGenerating, isRecording, isTranscribing]);

  useEffect(() => {
    if (performanceMode) return;
    const textarea = inputRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const computedStyle = window.getComputedStyle(textarea);
    const lineHeight = parseInt(computedStyle.lineHeight) || 24;
    const minHeight = lineHeight;
    const maxHeight = lineHeight * 16;
    const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
    textarea.style.height = newHeight + 'px';

    if (textarea.scrollHeight > maxHeight) {
      textarea.style.overflowY = 'auto';
    } else {
      textarea.style.overflowY = 'hidden';
    }
  }, [inputValue, performanceMode]);

  const isDisabled = !primaryModel || isGenerating || isModelLoading || agentConversationActive || isRecording || isTranscribing;
  const placeholderText =
    !primaryModel ? "Load a model first" :
      isRecording ? "Recording..." :
        isTranscribing ? "Transcribing..." :
          isGenerating ? "Generating..." :
            webSearchEnabled ? "Message (Web)..." :
              "Message...";

  return (
    // Adjusted padding: p-2 on mobile, p-4 on desktop. Reduced bottom padding.
    <form className="border-t border-border p-2 pb-6 md:p-4 md:pb-8 flex items-end gap-2 bg-background transition-all duration-200" onSubmit={handleSubmit}>
      <div className="relative flex-1">
        <Textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isDisabled}
          className="flex-1 resize-none border-input bg-background pr-16 md:pr-20 text-base py-3"
          rows={performanceMode ? 3 : 1}
          style={{
            minHeight: performanceMode ? '96px' : '44px', // Taller touch target
            height: performanceMode ? '96px' : undefined,
            overflowY: performanceMode ? 'auto' : 'hidden',
            transition: performanceMode ? 'none' : 'height 0.1s ease'
          }}
        />
        <div className="absolute right-1 bottom-1.5 flex gap-1">
          <SimpleChatImageButton />
          <ChatImageUploadButton />
        </div>
      </div>

      {canGoBack && (
        <Button
          type="button"
          variant="outline"
          onClick={onBack}
          disabled={isGenerating || isRecording || isTranscribing}
          size="icon"
          className="h-11 w-11 flex-shrink-0"
          title="Undo"
        >
          <ArrowLeft size={20} />
        </Button>
      )}

      <Button
        type="submit"
        disabled={!inputValue.trim() || isDisabled}
        size="icon"
        className="h-11 w-11 flex-shrink-0"
      >
        {isGenerating ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
      </Button>
    </form>
  );
};

export default ChatInputForm;
