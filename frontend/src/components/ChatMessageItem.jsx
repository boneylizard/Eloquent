import React from 'react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkSoftBreaks from '@/utils/remarkSoftBreaks';
import remarkDialogueQuotes from '@/utils/remarkDialogueQuotes';
import CodeBlock from './CodeBlock';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import { Button } from '@/components/ui/button';
import { Loader2, PlayCircle as PlayIcon, RotateCcw } from 'lucide-react';

const ChatMessageItem = React.memo(function ChatMessageItem({
  msg,
  isLastMessage, // We'll use this to control avatar visibility
  renderAvatar,
  renderUserAvatar,
  PRIMARY_API_URL,
  primaryCharacter,
  secondaryCharacter,
  editingMessageId,
  editingMessageContent,
  setEditingMessageContent,
  handleSaveEditedMessage,
  handleCancelEdit,
  handleEditUserMessage,
  handleRegenerateFromEditedPrompt,
  editingBotMessageId,
  editingBotMessageContent,
  setEditingBotMessageContent,
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
  getCurrentVariantContent,
  getVariantCount,
  navigateVariant,
  currentVariantIndex,
  formatModelName,
  isGenerating
}) {

  // Image Message Type
  if (msg.type === 'image') {
    return (
      <div className={cn("my-6 flex items-start gap-4", msg.role === 'user' ? 'justify-end' : '')}>
        {msg.role !== 'user' && (
          <div className="flex-shrink-0"> {/* REMOVED fixed size */}
            {isLastMessage && (
              <div className="animate-[pulse_0.4s_ease-in-out_1]"> {/* RESTORED animation */}
                {renderAvatar(msg, PRIMARY_API_URL, msg.modelId === 'primary' ? primaryCharacter : secondaryCharacter)}
              </div>
            )}
          </div>
        )}
        <div className="flex-1">
          <SimpleChatImageMessage message={msg} onRegenerate={handleRegenerateImage} regenerationQueue={regenerationQueue} />
        </div>
        {msg.role === 'user' && <div className="flex-shrink-0">{renderUserAvatar(msg)}</div>}
      </div>
    );
  }

  // Regular Text/System Message Type
  return (
    <div className={cn("my-6 flex items-start gap-4", msg.role === 'user' ? 'justify-end' : '', msg.role === 'system' ? 'justify-center' : '')}>
      {msg.role !== 'user' && msg.role !== 'system' && (
        <div className="flex-shrink-0"> {/* REMOVED fixed size */}
          {isLastMessage && (
            <div className="animate-[pulse_0.4s_ease-in-out_1]"> {/* RESTORED animation */}
              {renderAvatar(msg, PRIMARY_API_URL, msg.modelId === 'primary' ? primaryCharacter : secondaryCharacter)}
            </div>
          )}
        </div>
      )}

      <div className={cn("flex-1 max-w-[70ch]", msg.role === 'user' ? 'order-first' : '')}>
        {msg.role === 'user' ? (
          // User Message Content
          <div className="bg-blue-900/20 p-4 rounded-lg border border-blue-500/30">
            {editingMessageId === msg.id ? (
              <div className="space-y-3">
                <textarea
                  value={editingMessageContent}
                  onChange={(e) => setEditingMessageContent(e.target.value)}
                  className="w-full min-h-[100px] bg-black/40 border border-blue-500/50 rounded p-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  autoFocus
                />
                <div className="flex justify-end gap-2">
                  <Button variant="ghost" size="sm" onClick={handleCancelEdit}>
                    Cancel
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => handleSaveEditedMessage(msg.id, editingMessageContent)}>
                    Save
                  </Button>
                  <Button variant="default" size="sm" onClick={() => {
                    handleSaveEditedMessage(msg.id, editingMessageContent);
                    setTimeout(() => handleRegenerateFromEditedPrompt(msg.id, editingMessageContent), 100);
                  }}>
                    Save & Regenerate
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-blue-300 font-medium">You</span>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleEditUserMessage(msg.id, msg.content)}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                    </Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleRegenerateFromEditedPrompt(msg.id)} disabled={isGenerating}>
                      <RotateCcw size={12} />
                    </Button>
                  </div>
                </div>
                <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm prose-invert max-w-none text-white chat-prose">{msg.content}</ReactMarkdown>
              </>
            )}
          </div>
        ) : msg.role === 'system' ? (
          // System Message Content
          <div className="bg-yellow-900/20 p-3 rounded-lg border border-yellow-500/30 text-center">
            <p className="text-yellow-200 text-sm">{msg.content}</p>
          </div>
        ) : (
          // Bot Message Content
          <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-600">
            <div className="text-xs text-gray-400 mb-2 font-medium flex items-center justify-between">
              <span>{msg.characterName || (msg.modelName ? formatModelName(msg.modelName) : "Assistant")}</span>
              <div className="flex items-center gap-1">
                {ttsEnabled && (
                  <Button variant={isPlayingAudio === msg.id ? "destructive" : "ghost"} size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleSpeakerClick(msg.id, getCurrentVariantContent(msg.id, msg.content))} disabled={isGenerating || (isPlayingAudio && isPlayingAudio !== msg.id)}>
                    {isPlayingAudio === msg.id ? <Loader2 className="animate-spin" size={12} /> : <PlayIcon size={12} />}
                  </Button>
                )}
                <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleEditBotMessage(msg.id)} disabled={isGenerating}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
                </Button>
                <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleGenerateVariant(msg.id)} disabled={isGenerating}>
                  <RotateCcw size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={() => handleContinueGeneration(msg.id)} disabled={isGenerating}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="9,18 15,12 9,6" /></svg>
                </Button>
              </div>
            </div>

            {msg.role === 'bot' && getVariantCount(msg.id) > 1 && (
              <div className="flex items-center justify-between mb-2 text-xs text-gray-400 bg-gray-800/50 px-2 py-1 rounded">
                <button onClick={() => navigateVariant(msg.id, 'prev')} className="hover:text-white">← Previous</button>
                <span>{(currentVariantIndex[msg.id] || 0) + 1} of {getVariantCount(msg.id)}</span>
                <button onClick={() => navigateVariant(msg.id, 'next')} className="hover:text-white">Next →</button>
              </div>
            )}
            <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm prose-invert max-w-none text-white chat-prose">
              {getCurrentVariantContent(msg.id, msg.content)}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {msg.role === 'user' && <div className="flex-shrink-0">{renderUserAvatar(msg)}</div>}
    </div>
  );
});

export default ChatMessageItem;
