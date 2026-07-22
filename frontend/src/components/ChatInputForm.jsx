import React, { useMemo, useState, useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, ArrowLeft, Plus, X, StopCircle, ImagePlus } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';
import { cn } from '@/lib/utils';
import { useApp } from '@/contexts/AppContext';

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
  const [attachments, setAttachments] = useState([]); // { id, kind, name, type, base64 }
  const [plusOpen, setPlusOpen] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const imageInputRef = useRef(null);
  const fileInputRef = useRef(null);
  const { settings } = useApp();
  const hasAutoRoute = settings?.apiEndpointRoundRobinEnabled === true
    && (settings?.customApiEndpoints || []).some((endpoint) => endpoint?.enabled !== false && endpoint?.rotate_enabled !== false);
  const hasUsableModel = Boolean(primaryModel) || hasAutoRoute;

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
    if (!hasUsableModel) {
      onOpenModelSelector?.();
      return;
    }
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
    const maxHeight = Math.min(lineHeight * 16, window.innerWidth < 768 ? lineHeight * 8 : lineHeight * 16);
    const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
    textarea.style.height = newHeight + 'px';

    if (textarea.scrollHeight > maxHeight) {
      textarea.style.overflowY = 'auto';
    } else {
      textarea.style.overflowY = 'hidden';
    }
  }, [inputValue, performanceMode]);

  const isDisabled = !hasUsableModel || isGenerating || isModelLoading || agentConversationActive || isRecording || isTranscribing;
  const placeholderText =
    !hasUsableModel ? "Choose a model to start" :
      isRecording ? "Recording..." :
        isTranscribing ? "Transcribing..." :
          isGenerating ? "Generating..." :
            webSearchEnabled ? "Message (Web)..." :
              "Message...";

  const visionModel = settings?.visionModel || null;

  const capabilities = modelCapabilities && typeof modelCapabilities === 'object' ? modelCapabilities : {};
  const canVision = capabilities.vision === true;
  const canAttachImages = canVision || Boolean(visionModel);
  const canPdf = capabilities.pdf === true || capabilities['pdf-upload'] === true;
  const canUploadAny = canAttachImages || canPdf;

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
        if (!canAttachImages) continue;
        const base64 = await fileToDataUrl(f);
        next.push({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`, kind: 'image', name: f.name, type: f.type, base64 });
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
      <div className={cn('relative flex-1 rounded-lg', isDraggingFiles && 'ring-2 ring-primary ring-offset-2 ring-offset-background')}>
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
            if (!canAttachImages) return;
            e.preventDefault();
            const files = images.map((it) => it.getAsFile()).filter(Boolean);
            await addFiles(files);
          }}
          onDrop={async (e) => {
            const files = Array.from(e.dataTransfer?.files || []);
            if (!files.length) return;
            if (!canUploadAny) return;
            e.preventDefault();
            setIsDraggingFiles(false);
            await addFiles(files);
          }}
          onDragOver={(e) => {
            if (!canUploadAny) return;
            if (e.dataTransfer?.types?.includes?.('Files')) {
              e.preventDefault();
              setIsDraggingFiles(true);
            }
          }}
          onDragLeave={() => setIsDraggingFiles(false)}
          placeholder={placeholderText}
          disabled={isDisabled}
          className="flex-1 resize-none border-input bg-background pr-16 md:pr-20 text-base py-2 focus-visible:ring-2 focus-visible:ring-[var(--chat-input-focus-ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-background transition-shadow"
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

        <div className="absolute right-1 bottom-1.5 flex gap-1 items-center">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 rounded-full p-0"
            title={canAttachImages ? `Attach one or more images${visionModel ? ` · ${visionModel}` : ''}` : 'Choose a vision model before attaching images'}
            aria-label="Attach images"
            onClick={() => imageInputRef.current?.click()}
            disabled={isDisabled || !canAttachImages}
          >
            <ImagePlus className="h-4 w-4" />
          </Button>

          {/* Image generation remains separate from ordinary attachments. */}
          <SimpleChatImageButton />

          {canPdf && <div className="relative">
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
              <div className="absolute bottom-10 right-0 z-50 w-56 rounded-xl border border-border bg-background shadow-lg p-1 animate-in fade-in slide-in-from-bottom-2 zoom-in-95 duration-150">
                <button
                  type="button"
                  className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-muted"
                  onClick={() => {
                    fileInputRef.current?.click();
                    setPlusOpen(false);
                  }}
                >
                  Attach PDF files
                </button>
              </div>
            )}
          </div>}
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
        className={cn("h-11 w-11 flex-shrink-0 transition-all duration-150", !isGenerating && "active:scale-95 hover:brightness-110")}
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
