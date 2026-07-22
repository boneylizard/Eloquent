import React, { useMemo } from 'react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkSoftBreaks from '@/utils/remarkSoftBreaks';
import remarkDialogueQuotes from '@/utils/remarkDialogueQuotes';
import CodeBlock from './CodeBlock';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import MessageEditField from './MessageEditField';
import { Button } from '@/components/ui/button';
import { Loader2, PlayCircle as PlayIcon, RotateCcw } from 'lucide-react';
import ThinkingBlock from './ThinkingBlock';
import { resolveMessageThinkDisplay } from '../utils/thinkStreamParser';
import { splitStreamingContent, nextStreamingTokenKey } from '../utils/streamingText';

const ChatMessageItem = React.memo(function ChatMessageItem({
  msg,
  isLastMessage, // We'll use this to control avatar visibility
  renderAvatar,
  renderUserAvatar,
  PRIMARY_API_URL,
  primaryCharacter,
  secondaryCharacter,
  editingMessageId,
  handleSaveEditedMessage,
  handleCancelEdit,
  handleEditUserMessage,
  handleRegenerateFromEditedPrompt,
  editingBotMessageId,
  handleEditBotMessage,
  handleSaveBotMessage,
  handleCancelBotEdit,
  handleGenerateVariant,
  handleContinueGeneration,
  ttsEnabled,
  isPlayingAudio,
  handleSpeakerClick,
  handleRegenerateImage,
  onCancelRegenerations,
  isRegenerationRunning,
  regenerationQueue,
  getCurrentVariantContent,
  getVariantCount,
  navigateVariant,
  currentVariantIndex,
  formatModelName,
  isGenerating
}) {
  const variantContent = getCurrentVariantContent(msg.id, msg.content);
  const botThinkDisplay = useMemo(() => {
    if (msg.role !== 'bot') return null;
    return resolveMessageThinkDisplay(variantContent, msg?.reasoningText);
  }, [msg.role, variantContent, msg?.reasoningText]);
  const displayContent = botThinkDisplay ? botThinkDisplay.visibleContent : variantContent;
  const displayReasoning = msg?.reasoningText || (botThinkDisplay ? botThinkDisplay.reasoningText : '');

  // Image Message Type
  if (msg.type === 'image') {
    return (
      <div className={cn("my-3 md:my-6 flex items-start gap-2 md:gap-4", msg.role === 'user' ? 'justify-end' : '')}>
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
          <SimpleChatImageMessage message={msg} onRegenerate={handleRegenerateImage} regenerationQueue={regenerationQueue} onCancelRegenerations={onCancelRegenerations} isRegenerationRunning={isRegenerationRunning} />
        </div>
        {msg.role === 'user' && <div className="flex-shrink-0">{renderUserAvatar(msg)}</div>}
      </div>
    );
  }

  // Regular Text/System Message Type
  return (
    <div className={cn("my-3 md:my-6 flex items-start gap-2 md:gap-4", msg.role === 'user' ? 'justify-end' : '', msg.role === 'system' ? 'justify-center' : '')}>
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
          <div className="bg-[var(--chat-user-bg)] p-2.5 md:p-4 rounded-lg border border-[var(--chat-user-border)]">
            {editingMessageId === msg.id ? (
              <MessageEditField
                initialValue={msg.content}
                messageId={msg.id}
                onSave={handleSaveEditedMessage}
                onCancel={handleCancelEdit}
                onSaveAndRegenerate={handleRegenerateFromEditedPrompt}
                rows={4}
                saveLabel="Save"
                showSaveAndRegenerate
                disabledSaveAndRegenerate={isGenerating}
                textareaClassName="min-h-[100px] bg-[var(--chat-focus-surface)] border-[var(--chat-focus-border)] text-[var(--chat-focus-text-bright)] text-sm focus:ring-[var(--ring)]"
              />
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-[var(--primary)] font-medium">You</span>
                  <div className="flex gap-1">
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleEditUserMessage(msg.id)}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
                    </Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleRegenerateFromEditedPrompt(msg.id)} disabled={isGenerating}>
                      <RotateCcw size={12} />
                    </Button>
                  </div>
                </div>
                <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm prose-invert max-w-none text-[var(--chat-focus-text-bright)] chat-prose">{msg.content}</ReactMarkdown>
              </>
            )}
          </div>
        ) : msg.role === 'system' ? (
          // System Message Content
          <div className="bg-[var(--chat-system-bg)] p-2 md:p-3 rounded-lg border border-[var(--chat-system-border)] text-center">
            <p className="text-[var(--chat-system-text)] text-sm">{msg.content}</p>
          </div>
        ) : (
          // Bot Message Content
          <div className={cn("bg-[var(--chat-bot-bg)] p-2.5 md:p-4 rounded-lg border border-[var(--chat-bot-border)]", msg?.isStreaming && "streaming-active")}>
            <ThinkingBlock
              enabled={msg?.reasoningEnabled === true}
              reasoningText={displayReasoning || ''}
              streaming={msg?.reasoningStreaming === true}
              startedAtMs={msg?.reasoningStartedAtMs ?? null}
              finishedSeconds={typeof msg?.reasoningSeconds === 'number' ? msg.reasoningSeconds : null}
            />
            <div className="text-xs text-[var(--muted-foreground)] mb-2 font-medium flex items-center justify-between">
              <span>{msg.characterName || (msg.modelName ? formatModelName(msg.modelName) : "Assistant")}</span>
              <div className="flex items-center gap-1">
                {ttsEnabled && (
                  <Button variant={isPlayingAudio === msg.id ? "destructive" : "ghost"} size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleSpeakerClick(msg.id, displayContent)} disabled={isGenerating || (isPlayingAudio && isPlayingAudio !== msg.id)}>
                    {isPlayingAudio === msg.id ? <Loader2 className="animate-spin" size={12} /> : <PlayIcon size={12} />}
                  </Button>
                )}
                <Button variant="ghost" size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleEditBotMessage(msg.id)} disabled={isGenerating}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
                </Button>
                <Button variant="ghost" size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleGenerateVariant(msg.id)} disabled={isGenerating}>
                  <RotateCcw size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-6 w-6 text-[var(--muted-foreground)] hover:text-[var(--foreground)]" onClick={() => handleContinueGeneration(msg.id)} disabled={isGenerating}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9,18 15,12 9,6" /></svg>
                </Button>
              </div>
            </div>

            {msg.role === 'bot' && getVariantCount(msg.id) > 1 && (
              <div className="flex items-center justify-between mb-2 text-xs text-[var(--muted-foreground)] bg-[var(--chat-focus-surface)] px-2.5 py-1.5 rounded-full border border-[var(--chat-focus-border)]">
                <button onClick={() => navigateVariant(msg.id, 'prev')} className="hover:text-[var(--foreground)] transition-colors">← Previous</button>
                <span className="tabular-nums font-medium">{(currentVariantIndex[msg.id] || 0) + 1} of {getVariantCount(msg.id)}</span>
                <button onClick={() => navigateVariant(msg.id, 'next')} className="hover:text-[var(--foreground)] transition-colors">Next →</button>
              </div>
            )}
            {msg.isStreaming && displayContent ? (
              (() => {
                const { completed, streaming } = splitStreamingContent(displayContent, true);
                const tokenKey = nextStreamingTokenKey();
                return (
                  <div className="prose prose-sm prose-invert max-w-none text-[var(--chat-focus-text-bright)] chat-prose">
                    {completed && (
                      <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="max-w-none">
                        {completed}
                      </ReactMarkdown>
                    )}
                    <span key={tokenKey} className="streaming-token">{streaming}</span>
                    <span className="blinking-cursor" aria-hidden="true" />
                  </div>
                );
              })()
            ) : (
              <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm prose-invert max-w-none text-[var(--chat-focus-text-bright)] chat-prose">
                {displayContent}
              </ReactMarkdown>
            )}
          </div>
        )}
      </div>

      {msg.role === 'user' && <div className="flex-shrink-0">{renderUserAvatar(msg)}</div>}
    </div>
  );
});

export default ChatMessageItem;
