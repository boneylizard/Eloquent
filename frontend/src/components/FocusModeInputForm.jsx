import React, { useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, Mic, MicOff, ImagePlus, X } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';

const FocusModeInputForm = forwardRef(
    (
        {
            onSubmit,
            isGenerating,
            modelReady,
            sttEnabled,
            isRecording,
            isTranscribing,
            onMicClick,
        },
        ref
    ) => {
        const [inputValue, setInputValue] = useState('');
        const [attachments, setAttachments] = useState([]);
        const [isDraggingFiles, setIsDraggingFiles] = useState(false);
        const inputRef = useRef(null);
        const imageInputRef = useRef(null);

        const addImages = async (fileList) => {
            const files = Array.from(fileList || []).filter((file) => file.type?.startsWith('image/'));
            if (!files.length) return;
            const next = await Promise.all(files.map((file) => new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve({
                    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
                    kind: 'image',
                    name: file.name,
                    type: file.type || 'image/png',
                    base64: String(reader.result || ''),
                });
                reader.onerror = reject;
                reader.readAsDataURL(file);
            })));
            setAttachments((current) => [...current, ...next]);
        };

        useImperativeHandle(
            ref,
            () => ({
                setValue(text) {
                    setInputValue(String(text ?? ''));
                },
                appendValue(text) {
                    setInputValue((prev) => prev + (prev ? '\n\n' : '') + (text ?? ''));
                },
            }),
            []
        );

        const handleKeyDown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                localHandleSubmit(e);
            }
        };

        const localHandleSubmit = (e) => {
            e.preventDefault();
            const trimmedValue = inputValue.trim();
            if (trimmedValue || attachments.length) {
                onSubmit(trimmedValue, attachments);
                setInputValue('');
                setAttachments([]);
            }
        };

        useEffect(() => {
            if (!isGenerating && !isRecording && !isTranscribing) {
                inputRef.current?.focus();
            }
        }, [isGenerating, isRecording, isTranscribing]);

        useEffect(() => {
            const textarea = inputRef.current;
            if (!textarea) return;
            textarea.style.height = 'auto';
            const computed = window.getComputedStyle(textarea);
            const lineHeight = parseInt(computed.lineHeight) || 20;
            const maxHeight = lineHeight * 6;
            const newHeight = Math.min(Math.max(textarea.scrollHeight, lineHeight), maxHeight);
            textarea.style.height = newHeight + 'px';
            textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
        }, [inputValue]);

        const isDisabled = !modelReady || isGenerating || isRecording || isTranscribing;
        const placeholderText =
            !modelReady ? 'Select a model first'
                : isRecording ? 'Recording...'
                    : isTranscribing ? 'Transcribing...'
                        : isGenerating ? 'Generating response...'
                            : 'Type a message...';

        return (
            <form className="flex flex-col gap-2 md:flex-row md:items-center" onSubmit={localHandleSubmit}>
                <div className={`relative flex-1 rounded-md ${isDraggingFiles ? 'ring-2 ring-[var(--primary)] ring-offset-2 ring-offset-black' : ''}`}>
                    {attachments.length > 0 && (
                        <div className="mb-2 flex flex-wrap gap-2">
                            {attachments.map((attachment) => (
                                <span key={attachment.id} className="inline-flex items-center gap-2 rounded-full border border-[var(--chat-focus-border)] bg-black/40 px-3 py-1 text-xs text-[var(--chat-focus-text-bright)]">
                                    <span className="max-w-[220px] truncate">{attachment.name}</span>
                                    <button type="button" aria-label={`Remove ${attachment.name}`} onClick={() => setAttachments((current) => current.filter((item) => item.id !== attachment.id))}>
                                        <X size={13} />
                                    </button>
                                </span>
                            ))}
                        </div>
                    )}
                    <Textarea
                        ref={inputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onPaste={(event) => {
                            const files = Array.from(event.clipboardData?.items || [])
                                .filter((item) => item.kind === 'file' && item.type?.startsWith('image/'))
                                .map((item) => item.getAsFile())
                                .filter(Boolean);
                            if (!files.length) return;
                            event.preventDefault();
                            void addImages(files);
                        }}
                        onDrop={(event) => {
                            const files = Array.from(event.dataTransfer?.files || []);
                            if (!files.some((file) => file.type?.startsWith('image/'))) return;
                            event.preventDefault();
                            setIsDraggingFiles(false);
                            void addImages(files);
                        }}
                        onDragOver={(event) => {
                            if (!event.dataTransfer?.types?.includes?.('Files')) return;
                            event.preventDefault();
                            setIsDraggingFiles(true);
                        }}
                        onDragLeave={() => setIsDraggingFiles(false)}
                        placeholder={placeholderText}
                        disabled={isDisabled}
                        className="flex-1 resize-none bg-[var(--chat-focus-surface)] border-[var(--chat-focus-border)] text-[var(--chat-focus-text-bright)] placeholder:text-[var(--chat-focus-text)] pr-20 focus-visible:border-[var(--primary)] transition-colors max-h-[40vh] md:max-h-[200px]"
                        rows={1}
                    />
                    <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        className="hidden"
                        onChange={(event) => {
                            if (event.target.files?.length) void addImages(event.target.files);
                            event.target.value = '';
                        }}
                    />
                    <div className="absolute right-2 top-2 flex gap-1">
                        <Button type="button" variant="ghost" size="icon" className="h-8 w-8" title="Attach one or more images" aria-label="Attach images" onClick={() => imageInputRef.current?.click()} disabled={isDisabled}>
                            <ImagePlus size={17} />
                        </Button>
                        <SimpleChatImageButton />
                    </div>
                </div>
                <div className="flex gap-2 shrink-0 self-end md:self-center">
                    {sttEnabled && (
                        <Button
                            type="button"
                            variant={isRecording ? 'destructive' : 'ghost'}
                            size="icon"
                            className={`h-10 w-10 shrink-0 transition-all duration-200 ${isRecording ? 'animate-pulse ring-2 ring-destructive/40' : 'bg-[var(--chat-focus-surface)] text-[var(--chat-focus-text-bright)] hover:bg-[var(--chat-focus-surface-hover)]'}`}
                            onClick={onMicClick}
                            disabled={isTranscribing || !modelReady || isGenerating}
                            title={isRecording ? 'Stop recording' : 'Voice input'}
                        >
                            {isTranscribing ? (
                                <Loader2 className="animate-spin" size={18} />
                            ) : isRecording ? (
                                <MicOff size={18} />
                            ) : (
                                <Mic size={18} />
                            )}
                        </Button>
                    )}
                    <Button type="submit" disabled={(!inputValue.trim() && attachments.length === 0) || isDisabled} size="icon" className="h-10 w-10 bg-[var(--primary)] text-primary-foreground hover:brightness-110 active:scale-95 transition-all duration-150">
                        {isGenerating ? <Loader2 className="animate-spin" size={18}/> : <Send size={18}/>}
                    </Button>
                </div>
            </form>
        );
    }
);

FocusModeInputForm.displayName = 'FocusModeInputForm';

export default FocusModeInputForm;
