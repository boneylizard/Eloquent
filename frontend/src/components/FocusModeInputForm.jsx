import React, { useState, useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, Mic, MicOff } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';
import ChatImageUploadButton from './ChatImageUploadButton';

const FocusModeInputForm = forwardRef(
    (
        {
            onSubmit,
            isGenerating,
            primaryModel,
            sttEnabled,
            isRecording,
            isTranscribing,
            onMicClick,
        },
        ref
    ) => {
        const [inputValue, setInputValue] = useState('');
        const inputRef = useRef(null);

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
            if (trimmedValue) {
                onSubmit(trimmedValue);
                setInputValue('');
            }
        };

        const isDisabled =
            !primaryModel || isGenerating || isRecording || isTranscribing;
        const placeholderText =
            !primaryModel ? 'Load a model first'
                : isRecording ? 'Recording...'
                    : isTranscribing ? 'Transcribing...'
                        : isGenerating ? 'Generating response...'
                            : 'Type a message...';

        useEffect(() => {
            if (!isGenerating && !isRecording && !isTranscribing) {
                inputRef.current?.focus();
            }
        }, [isGenerating, isRecording, isTranscribing]);

        return (
            <form className="flex items-center gap-2" onSubmit={localHandleSubmit}>
                <div className="relative flex-1">
                    <Textarea
                        ref={inputRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={placeholderText}
                        disabled={isDisabled}
                        className="flex-1 resize-none bg-gray-900/50 border-gray-600 text-white pr-20"
                        rows={1}
                    />
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                        <SimpleChatImageButton />
                        <ChatImageUploadButton />
                    </div>
                </div>
                {sttEnabled && (
                    <Button
                        type="button"
                        variant={isRecording ? 'destructive' : 'secondary'}
                        size="icon"
                        className={`h-10 w-10 shrink-0 ${isRecording ? 'animate-pulse ring-2 ring-destructive/40' : 'bg-white/10 text-white hover:bg-white/20 border-0'}`}
                        onClick={onMicClick}
                        disabled={isTranscribing || !primaryModel || isGenerating}
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
                <Button type="submit" disabled={!inputValue.trim() || isDisabled} size="icon" className="h-10 w-10 bg-blue-600 hover:bg-blue-700">
                    {isGenerating ? <Loader2 className="animate-spin" size={18}/> : <Send size={18}/>}
                </Button>
            </form>
        );
    }
);

FocusModeInputForm.displayName = 'FocusModeInputForm';

export default FocusModeInputForm;
