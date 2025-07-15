import React, { useState, useEffect, useRef, useMemo } from 'react';
import { cn } from '@/lib/utils';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, Layers, Users, Mic, MicOff, Copy, Check, PlayCircle as PlayIcon, X, Cpu, RotateCcw, Globe, Phone, PhoneOff, Focus } from 'lucide-react';;
import CodeBlock from './CodeBlock';
import SimpleChatImageButton from './SimpleChatImageButton';
import ChatImageUploadButton from './ChatImageUploadButton';
import ChatMessageItem from './ChatMessageItem'; // ADD THIS IMPORT
import FocusModeInputForm from './FocusModeInputForm';

const FocusModeOverlay = ({ 
  isActive, 
  onExit,
  messages,
  handleSubmit,
  isGenerating,
  primaryModel,
  renderAvatar,
  renderUserAvatar,
  PRIMARY_API_URL,
  primaryCharacter,
  secondaryCharacter,
  getCurrentVariantContent,
  getVariantCount,
  navigateVariant,
  editingMessageId,
  editingMessageContent,
  setEditingMessageContent,
  handleSaveEditedMessage,
  handleCancelEdit,
  handleEditUserMessage,
  handleRegenerateFromEditedPrompt,
  editingBotMessageId,
  editingBotMessageContent,
  handleEditBotMessage,
  handleSaveBotMessage,
  handleCancelBotEdit,
  handleGenerateVariant,
  handleContinueGeneration,
  ttsEnabled,
  isPlayingAudio,
  handleSpeakerClick,
  handleRegenerateImage,
  regenerationQueue,
  currentVariantIndex,
  formatModelName,
  stopTTS
}) => {
  const messagesEndRef = useRef(null);
  useEffect(() => {
  const handleKeyDown = (event) => {
    if (event.key === 'Shift' && isPlayingAudio && stopTTS) {
      event.preventDefault();
      stopTTS();
    }
  };

  if (isActive) {
    document.addEventListener('keydown', handleKeyDown);
  }

  return () => {
    document.removeEventListener('keydown', handleKeyDown);
  };
}, [isActive, isPlayingAudio, stopTTS]);
  // Handle ESC key to exit
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && isActive) {
        onExit();
      }
    };

    if (isActive) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'auto';
    };
  }, [isActive, onExit]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages]);

  // STABLE LOGIC: Pre-calculate the last message ID for each character.
  // This message is the only one that will ever get an avatar.
const lastMessageAvatars = useMemo(() => {
    const lastMessageMap = new Map();
    if (!messages) return lastMessageMap;

    // Iterate backwards to find the last message for each character/model efficiently.
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role !== 'user' && msg.role !== 'system') {
            const characterId = msg.characterName || msg.modelId || 'assistant';
            if (!lastMessageMap.has(characterId)) {
                lastMessageMap.set(characterId, msg.id);
            }
        }
    }
    return lastMessageMap;
}, [messages]);

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 z-[9999] bg-black flex flex-col">
      <button onClick={onExit} className="absolute top-6 right-6 w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-all duration-200 z-10" title="Exit Focus Mode (ESC)">
        <X size={20} />
      </button>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-6xl mx-auto w-full">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
              <h3 className="text-lg font-medium mb-2">Focus Mode</h3>
              <p className="max-w-md">Clean, distraction-free chat interface</p>
            </div>
          ) : (
            <>
{messages.map((msg) => {
  const characterId = msg.characterName || msg.modelId || 'assistant';
  const isLastMessage = lastMessageAvatars.get(characterId) === msg.id;

  return (
    <ChatMessageItem
      key={msg.id}
      msg={msg}
      isLastMessage={isLastMessage}
      renderAvatar={renderAvatar}
      renderUserAvatar={renderUserAvatar}
      PRIMARY_API_URL={PRIMARY_API_URL}
      primaryCharacter={primaryCharacter}
      secondaryCharacter={secondaryCharacter}
      editingMessageId={editingMessageId}
      editingMessageContent={editingMessageContent}
      setEditingMessageContent={setEditingMessageContent}
      handleSaveEditedMessage={handleSaveEditedMessage}
      handleCancelEdit={handleCancelEdit}
      handleEditUserMessage={handleEditUserMessage}
      handleRegenerateFromEditedPrompt={handleRegenerateFromEditedPrompt}
      editingBotMessageId={editingBotMessageId}
      editingBotMessageContent={editingBotMessageContent}
      handleEditBotMessage={handleEditBotMessage}
      handleSaveBotMessage={handleSaveBotMessage}
      handleCancelBotEdit={handleCancelBotEdit}
      handleGenerateVariant={handleGenerateVariant}
      handleContinueGeneration={handleContinueGeneration}
      ttsEnabled={ttsEnabled}
      isPlayingAudio={isPlayingAudio}
      handleSpeakerClick={handleSpeakerClick}
      handleRegenerateImage={handleRegenerateImage}
      regenerationQueue={regenerationQueue}
      getCurrentVariantContent={getCurrentVariantContent}
      getVariantCount={getVariantCount}
      navigateVariant={navigateVariant}
      currentVariantIndex={currentVariantIndex}
      formatModelName={formatModelName}
      isGenerating={isGenerating}
    />
  );
})}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </div>

<div className="border-t border-gray-800 p-4 bg-black/50">
    <div className="max-w-6xl mx-auto w-full">
        <FocusModeInputForm 
            onSubmit={handleSubmit} 
            isGenerating={isGenerating} 
            primaryModel={primaryModel} 
        />
    </div>
</div>
    </div>
  );
};

export default FocusModeOverlay;