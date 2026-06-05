import React, { useMemo, useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, ArrowLeft, Plus, X, StopCircle } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';
import ChatImageUploadButton from './ChatImageUploadButton';
import { cn } from '@/lib/utils';

// Local input state so typing never re-renders parent Chat (fixes lag as chat grows).
// Parent can set/append value via ref for STT and StoryTracker inject.
const ChatInputForm = forwardRef(({
  onSubmit,
  onStop,
  onOpenModelSelector,
  isGenerating,
  isModelLoading,
  isRecording,
  isTranscribing,
  agentConversationActive,
  primaryModel,
  webSearchEnabled,
  performanceMode,
  onBack,
  canGoBack,
  modelCapabilities,
  nanoGptChrome = false,
}, ref) => {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);
  const [attachments, setAttachments] = useState([]); // { id, name, type, base64 }
  const [plusOpen, setPlusOpen] = useState(false);
  const imageInputRef = useRef(null);
  const fileInputRef = useRef(null);

  useImperativeHandle(ref, () => ({
    setValue(text) {
      setInputValue(String(text ?? ''));
    },
    appendValue(text) {
      setInputValue(prev => prev + (prev ? '\n\n' : '') + (text ?? ''));
    }
  }), []);

  const handleKeyDown = (e) => {
    if (e.key === '/' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const empty = !inputValue || inputValue.trim().length === 0;
      if (empty && typeof onOpenModelSelector === 'function') {
        e.preventDefault();
        onOpenModelSelector();
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSubmit = (e) => {
    e?.preventDefault?.();
    const trimmedValue = inputValue.trim();
    if (!trimmedValue && attachments.length === 0) return;
    onSubmit(trimmedValue, attachments);
    setInputValue('');
    setAttachments([]);
    setPlusOpen(false);
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

  const isDisabled = isGenerating || isModelLoading || agentConversationActive || isRecording || isTranscribing;
  const placeholderText =
    isRecording ? "Recording..." :
        isTranscribing ? "Transcribing..." :
          isGenerating ? "Generating..." :
            webSearchEnabled ? "Message (Web)..." :
              "Message...";

  const capabilities = modelCapabilities && typeof modelCapabilities === 'object' ? modelCapabilities : {};
  const canVision = capabilities.vision === true;
  const canPdf = capabilities.pdf === true || capabilities['pdf-upload'] === true;
  const canUploadAny = canVision || canPdf;

  const fileToDataUrl = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  const addFiles = async (fileList) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const next = [];
    for (const f of files) {
      if (f.type && f.type.startsWith('image/')) {
        const base64 = await fileToDataUrl(f);
        next.push({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`, name: f.name, type: f.type, base64 });
      } else if (canPdf && (f.type === 'application/pdf' || f.name?.toLowerCase().endsWith('.pdf'))) {
        // Minimal placeholder: keep as "file" attachment without upload wiring yet.
        next.push({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`, name: f.name, type: f.type || 'application/octet-stream', base64: null });
      }
    }
    if (next.length) setAttachments((prev) => [...prev, ...next]);
  };

  const attachmentChips = useMemo(() => {
    if (!attachments.length) return null;
    return (
      <div className="mb-2 flex flex-wrap gap-2">
        {attachments.map((a) => (
          <span
            key={a.id}
            className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-background/80 px-3 py-1 text-xs text-foreground"
            title={a.name}
          >
            <span className="max-w-[220px] truncate">{a.name}</span>
            <button
              type="button"
              className="rounded-full p-0.5 text-muted-foreground hover:text-foreground"
              onClick={() => setAttachments((prev) => prev.filter((x) => x.id !== a.id))}
              aria-label="Remove attachment"
            >
              <X size={14} />
            </button>
          </span>
        ))}
      </div>
    );
  }, [attachments]);

  return (
    // Adjusted padding: p-2 on mobile, p-4 on desktop. Reduced bottom padding.
    <form
      className={cn(
        'p-2 md:p-3 flex items-end gap-2 transition-all duration-200',
        nanoGptChrome ? 'border-0 bg-transparent' : 'border-t border-border bg-background',
      )}
      onSubmit={handleSubmit}
    >
      <div className="relative flex-1">
        {attachmentChips}
        <Textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={async (e) => {
            const items = Array.from(e.clipboardData?.items || []);
            const images = items.filter((it) => it.kind === 'file' && it.type && it.type.startsWith('image/'));
            if (!images.length) return;
            if (!canVision) return;
            e.preventDefault();
            const files = images.map((it) => it.getAsFile()).filter(Boolean);
            await addFiles(files);
          }}
          onDrop={async (e) => {
            const files = Array.from(e.dataTransfer?.files || []);
            if (!files.length) return;
            if (!canUploadAny) return;
            e.preventDefault();
            await addFiles(files);
          }}
          onDragOver={(e) => {
            if (!canUploadAny) return;
            if (e.dataTransfer?.types?.includes?.('Files')) e.preventDefault();
          }}
          placeholder={placeholderText}
          disabled={isDisabled}
          className="flex-1 resize-none border-input bg-background pr-16 md:pr-20 text-base py-2"
          rows={performanceMode ? 3 : 1}
          style={{
            minHeight: performanceMode ? '96px' : '40px',
            height: performanceMode ? '96px' : undefined,
            overflowY: performanceMode ? 'auto' : 'hidden',
            transition: performanceMode ? 'none' : 'height 0.1s ease'
          }}
        />
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = e.target.files;
            if (files?.length) void addFiles(files);
            e.target.value = '';
          }}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept={canPdf ? '.pdf' : undefined}
          multiple
          className="hidden"
          onChange={(e) => {
            const files = e.target.files;
            if (files?.length) void addFiles(files);
            e.target.value = '';
          }}
        />

        <div className="absolute right-1 bottom-1.5 flex gap-1">
          {/* Legacy buttons kept for now (minimal regression) */}
          <SimpleChatImageButton />
          <ChatImageUploadButton />

          {/* NanoGPT-style "+" popover (upwards) */}
          <div className="relative">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 rounded-full p-0"
              title="Add attachments"
              onClick={() => setPlusOpen((v) => !v)}
              disabled={isDisabled}
            >
              <Plus className="h-4 w-4" />
            </Button>
            {plusOpen && (
              <div className="absolute bottom-10 right-0 z-50 w-56 rounded-xl border border-border bg-background shadow-lg p-1">
                <button
                  type="button"
                  className={`w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-muted ${canPdf ? '' : 'opacity-50 cursor-not-allowed'}`}
                  onClick={() => {
                    if (!canPdf) return;
                    fileInputRef.current?.click();
                    setPlusOpen(false);
                  }}
                >
                  Add files
                </button>
                <button
                  type="button"
                  className={`w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-muted ${canVision ? '' : 'opacity-50 cursor-not-allowed'}`}
                  onClick={() => {
                    if (!canVision) return;
                    imageInputRef.current?.click();
                    setPlusOpen(false);
                  }}
                >
                  Add images
                </button>
                <button
                  type="button"
                  className="w-full rounded-lg px-3 py-2 text-left text-sm opacity-50 cursor-not-allowed"
                  onClick={() => {}}
                >
                  Import from Drive (stub)
                </button>
                <button
                  type="button"
                  className="w-full rounded-lg px-3 py-2 text-left text-sm opacity-50 cursor-not-allowed"
                  onClick={() => {}}
                >
                  Upload conversation JSON (stub)
                </button>
                <button
                  type="button"
                  className="w-full rounded-lg px-3 py-2 text-left text-sm opacity-50 cursor-not-allowed"
                  onClick={() => {}}
                >
                  Memory toggle (stub)
                </button>
              </div>
            )}
          </div>
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
        type={isGenerating && typeof onStop === 'function' ? 'button' : 'submit'}
        onClick={isGenerating && typeof onStop === 'function' ? onStop : undefined}
        disabled={(!inputValue.trim() && attachments.length === 0) || isDisabled}
        size="icon"
        className="h-11 w-11 flex-shrink-0"
      >
        {isGenerating
          ? (typeof onStop === 'function'
            ? <StopCircle size={20} />
            : <Loader2 className="animate-spin" size={20} />)
          : <Send size={20} />}
      </Button>
    </form>
  );
});

ChatInputForm.displayName = 'ChatInputForm';

export default ChatInputForm;
