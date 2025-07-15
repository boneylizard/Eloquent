import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send } from 'lucide-react';
import SimpleChatImageButton from './SimpleChatImageButton';
import ChatImageUploadButton from './ChatImageUploadButton';

const FocusModeInputForm = ({ onSubmit, isGenerating, primaryModel }) => {
    const [inputValue, setInputValue] = useState('');
    const inputRef = useRef(null);

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

    useEffect(() => {
        if (!isGenerating) {
            inputRef.current?.focus();
        }
    }, [isGenerating]);

    return (
        <form className="flex items-center gap-2" onSubmit={localHandleSubmit}>
            <div className="relative flex-1">
                <Textarea 
                    ref={inputRef}
                    value={inputValue} 
                    onChange={(e) => setInputValue(e.target.value)} 
                    onKeyDown={handleKeyDown} 
                    placeholder={!primaryModel ? "Load a model first" : isGenerating ? "Generating response..." : "Type a message..."} 
                    disabled={!primaryModel || isGenerating} 
                    className="flex-1 resize-none bg-gray-900/50 border-gray-600 text-white pr-20" // Padding for buttons
                    rows={1} 
                />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                    <SimpleChatImageButton />
                    <ChatImageUploadButton />
                </div>
            </div>
            <Button type="submit" disabled={!inputValue.trim() || !primaryModel || isGenerating} size="icon" className="h-10 w-10 bg-blue-600 hover:bg-blue-700">
                {isGenerating ? <Loader2 className="animate-spin" size={18}/> : <Send size={18}/>}
            </Button>
        </form>
    );
};

export default FocusModeInputForm;