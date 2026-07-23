import React, { memo, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, X, PlayCircle as PlayIcon, FastForward, Pause, RotateCcw, Cpu, Pencil, ChevronRight, Trash2, Globe } from 'lucide-react';
import NanoGptModelSelectorPopover from './NanoGptModelSelectorPopover';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkSoftBreaks from '@/utils/remarkSoftBreaks';
import remarkDialogueQuotes from '@/utils/remarkDialogueQuotes';
import { cn } from '@/lib/utils';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import CodeBlock from './CodeBlock';
import MessageEditField from './MessageEditField';
import CharacterAvatarMedia from './CharacterAvatarMedia';
import ThinkingBlock from './ThinkingBlock';
import WebSearchProgress from './WebSearchProgress';
import { resolveMessageThinkDisplay } from '../utils/thinkStreamParser';
import { getBackendUrl } from '../config/api';
import { getActiveCharacterAvatar, resolveAvatarDisplayUrl } from '../utils/characterAvatars';
import { resolveBotMessageSpeaker, resolveEndpointDisplay } from '../utils/resolveEndpointDisplay';
import { splitStreamingContent, nextStreamingTokenKey } from '../utils/streamingText';

/** ~66–70 characters per line — comfortable eye-scan width without full-screen lines. */
const CHAT_READING_MEASURE = '70ch';
const CHAT_MESSAGE_GAP = '0.75rem';

function chatBubbleWidth(avatarPx) {
    return `min(100%, calc(${CHAT_READING_MEASURE} + ${avatarPx}px + ${CHAT_MESSAGE_GAP}))`;
}

function AvatarRing({ sizePx, url, alt, fallbackLabel, fallbackIcon, videoKey, className = '' }) {
    const ringStyle = { width: sizePx, height: sizePx };
    const fallbackContent = fallbackIcon || fallbackLabel;
  const fallbackClass = fallbackIcon
    ? 'text-lg leading-none'
    : 'text-sm font-semibold text-muted-foreground';
    return (
        <div
            className={cn(
                'shrink-0 overflow-hidden rounded-full border border-[var(--chat-avatar-border)] bg-muted',
                className
            )}
            style={ringStyle}
            title={alt}
        >
            {url ? (
                <CharacterAvatarMedia
                    url={url}
                    alt={alt}
                    fit="cover"
                    className="h-full w-full object-cover"
                    videoKey={videoKey || url}
                />
            ) : (
                <div className={cn('flex h-full w-full items-center justify-center', fallbackClass)}>
                    {fallbackContent}
                </div>
            )}
        </div>
    );
}

const ChatMessage = memo(({
    msg,
    content,
    isGenerating,
    isTranscribing,
    isPlayingAudio,
    ttsPlaybackState,
    isStreamingTtsPaused,
    editingMessageId,
    editingBotMessageId,
    primaryCharacter,
    secondaryCharacter,
    activeCharacter,
    characters,
    primaryModel,
    primaryIsAPI,
    settings,
    nanoGptCatalog,
    userProfile,
    userCharacter,
    isMultiRoleMode,
    characterAvatarSize,
    userAvatarSize,
    variantCount,
    variantIndex,
    PRIMARY_API_URL,
    regenerationQueue,
    isRegenerationRunning,
    ttsEnabled,

    onEditUserMessage,
    onCancelEdit,
    onSaveEditedMessage,
    onRegenerateFromEditedPrompt,
    onDeleteMessage,

    onEditBotMessage,
    onCancelBotEdit,
    onSaveBotMessage,
    onGenerateVariant,
    onGenerateVariantWithModel,
    onContinueGeneration,
    onNavigateVariant,
    onNavigateGreeting,
    onSpeakerClick,
    onChunkedSpeakerClick,
    onRegenerateImage,
    onCancelRegenerations,

    formatModelName,
}) => {
    const showDiagnostics = settings?.showReasoningDiagnostics === true;
    const [thinkingOpen, setThinkingOpen] = useState(false);

    const botThinkDisplay = useMemo(() => {
        if (msg.role !== 'bot') return null;
        return resolveMessageThinkDisplay(content, msg?.reasoningText);
    }, [msg.role, content, msg?.reasoningText]);

    const displayContent = botThinkDisplay ? botThinkDisplay.visibleContent : content;
    const displayReasoning = botThinkDisplay ? botThinkDisplay.reasoningText : '';
    const botSpeakerCtx = {
        characters,
        activeCharacter,
        primaryCharacter,
        secondaryCharacter,
        primaryModel,
        primaryIsAPI,
        settings,
        catalog: nanoGptCatalog,
        getActiveCharacterAvatar,
    };

    const resolveBotSpeaker = (message) =>
        resolveBotMessageSpeaker(message, botSpeakerCtx);

    const renderAvatar = (message, apiUrl) => {
        const speaker = resolveBotSpeaker(message);
        const displayUrl = resolveAvatarDisplayUrl(speaker.avatarUrl, apiUrl || getBackendUrl());
        const fallbackIcon = !displayUrl && speaker.icon ? speaker.icon : null;
        const fallbackLabel = !fallbackIcon && speaker.displayName
            ? speaker.displayName.charAt(0).toUpperCase()
            : '?';
        return (
            <AvatarRing
                sizePx={characterAvatarSize}
                url={displayUrl}
                alt={speaker.displayName}
                fallbackLabel={fallbackLabel}
                fallbackIcon={fallbackIcon}
                videoKey={`${message.id}-${displayUrl || speaker.icon || 'fallback'}`}
            />
        );
    };

    const renderUserAvatar = (message) => {
        const roleplayAvatar = isMultiRoleMode && message?.characterId
            ? message?.avatar || userCharacter?.avatar
            : null;
        const userAvatarSource = roleplayAvatar || userProfile?.avatar;
        const userName = isMultiRoleMode && message?.characterId
            ? (message?.characterName || userCharacter?.name || 'User')
            : (userProfile?.name || userProfile?.username || 'User');
        const userDisplayUrl = resolveAvatarDisplayUrl(userAvatarSource, PRIMARY_API_URL || getBackendUrl());
        return (
            <AvatarRing
                sizePx={userAvatarSize}
                url={userDisplayUrl}
                alt={userName}
                fallbackLabel={userName ? userName.charAt(0).toUpperCase() : 'U'}
                videoKey={`user-${message?.id || 'profile'}-${userDisplayUrl || 'fallback'}`}
                className="border-primary/50"
            />
        );
    };

    const botBubbleWidth = chatBubbleWidth(characterAvatarSize);
    const userBubbleWidth = chatBubbleWidth(userAvatarSize);
    const textColClass = 'min-w-0 flex-1 basis-0';
    const textColStyle = { maxWidth: CHAT_READING_MEASURE, width: '100%' };
    const rowLayoutClass = 'flex items-start gap-1.5 md:gap-3 max-w-full';
    const greetingOptions = Array.isArray(msg?.characterGreeting?.options)
        ? msg.characterGreeting.options
        : [];
    const greetingIndex = Number.isInteger(msg?.characterGreeting?.index)
        ? msg.characterGreeting.index
        : 0;
    const canCycleGreeting =
        msg.role === 'bot'
        && greetingOptions.length > 1
        && variantCount <= 1
        && typeof onNavigateGreeting === 'function';

    if (msg.type === 'image' || msg.type === 'video') {
        return (
            <div
                className={cn(
                    'my-1.5 rounded-lg p-1.5 shadow-sm md:p-3 max-w-full',
                    rowLayoutClass,
                    msg.role === 'user' ? 'ml-auto bg-primary/10' : 'bg-secondary'
                )}
                style={{ width: msg.role === 'user' ? userBubbleWidth : botBubbleWidth }}
            >
                {msg.role !== 'user' && renderAvatar(msg, PRIMARY_API_URL)}
                <div className={textColClass} style={textColStyle}>
                    <SimpleChatImageMessage
                        message={msg}
                        onRegenerate={onRegenerateImage}
                        regenerationQueue={regenerationQueue}
                        onCancelRegenerations={onCancelRegenerations}
                        isRegenerationRunning={isRegenerationRunning}
                    />
                </div>
                {msg.role === 'user' && renderUserAvatar(msg)}
            </div>
        );
    }

    const isSystem = msg.role === 'system';

    const renderReasoningDiagnostics = (message) => {
        if (!showDiagnostics || message.role !== 'bot') return null;

        const capOn =
            message?.reasoningEnabled === true ||
            message?.reasoningCapabilitySource === 'inline';
        let capSource = 'none';
        if (message?.reasoningCapabilitySource === 'inline') {
            capSource = 'inline';
        } else if (primaryIsAPI && primaryModel) {
            try {
                const resolved = resolveEndpointDisplay(primaryModel, settings, nanoGptCatalog);
                if (resolved?.capabilitySource) {
                    capSource = resolved.capabilitySource;
                }
            } catch {
                // ignore resolution errors in debug UI
            }
        }

        const hasContent = typeof displayContent === 'string' && displayContent.length > 0;
        const hasReasoningText =
            typeof displayReasoning === 'string' && displayReasoning.trim().length > 0;

        let sseStatus = 'none';
        if (hasReasoningText) sseStatus = 'reasoning';
        else if (hasContent) sseStatus = 'content';

        const uiRendered = message?.reasoningEnabled === true &&
            (message?.reasoningStreaming || hasReasoningText);
        const uiStatus = uiRendered ? (thinkingOpen ? 'open' : 'closed') : 'hidden';

        return (
            <div className="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] font-mono text-muted-foreground/80">
                <span className="uppercase tracking-[0.16em] text-muted-foreground/90">
                    Reasoning status
                </span>
                <span>
                    CAP: {capOn ? 'on' : 'off'}
                    {capOn && capSource !== 'none' ? ` (${capSource})` : ''}
                </span>
                <span>SSE: {sseStatus}</span>
                <span>UI: {uiStatus}</span>
            </div>
        );
    };

  const messageBody = (
    <div className={cn('relative w-full', textColClass)} style={textColStyle}>
      {msg.role === 'user' ? (
        editingMessageId === msg.id ? (
          <MessageEditField
            initialValue={msg.content}
            messageId={msg.id}
            onSave={onSaveEditedMessage}
            onCancel={onCancelEdit}
            onSaveAndRegenerate={onRegenerateFromEditedPrompt}
            rows={3}
            saveLabel="Save"
            showSaveAndRegenerate
            disabledSaveAndRegenerate={isGenerating}
          />
        ) : (
          <div className="group relative w-full">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">You</span>
              <div className="relative z-10 flex gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-150">
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onEditUserMessage(msg.id)} title="Edit message">
                  <Pencil size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onRegenerateFromEditedPrompt(msg.id)} disabled={isGenerating} title="Regenerate from this message">
                  <RotateCcw size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6 text-muted-foreground hover:bg-[var(--chat-action-hover-danger)] hover:text-destructive" onClick={() => onDeleteMessage(msg.id)} disabled={isGenerating} title="Delete message">
                  <Trash2 size={12} />
                </Button>
              </div>
            </div>
            <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm dark:prose-invert chat-prose max-w-none w-full break-words">
              {msg.content}
            </ReactMarkdown>
          </div>
        )
      ) : (
        <>
          {msg.role === 'bot' && (
            <>
              <ThinkingBlock
                enabled={msg?.reasoningEnabled === true}
                reasoningText={displayReasoning || ''}
                streaming={msg?.reasoningStreaming === true}
                startedAtMs={msg?.reasoningStartedAtMs ?? null}
                finishedSeconds={typeof msg?.reasoningSeconds === 'number' ? msg.reasoningSeconds : null}
                onOpenChange={setThinkingOpen}
              />
              {renderReasoningDiagnostics(msg)}
              {(msg.webSearchProgress || (msg.webSearchSources?.length > 0 && !msg.isStreaming)) && (
                <WebSearchProgress
                  progress={msg.webSearchProgress}
                  sources={msg.webSearchSources || []}
                  isComplete={!msg.webSearchProgress}
                />
              )}
            </>
          )}
          {msg.role === 'bot' && (
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs font-medium text-muted-foreground">
              <span>{resolveBotSpeaker(msg).displayName}</span>
              <div className={cn(
                'relative z-10 flex items-center gap-1 opacity-100 transition-opacity duration-150',
                ttsPlaybackState?.messageId === msg.id ? 'md:opacity-100' : 'md:opacity-0 md:group-hover:opacity-100',
              )}>
                {ttsEnabled && (
                  <>
                    {ttsPlaybackState?.messageId === msg.id && (
                      <span className="mr-1 text-[11px] font-normal text-muted-foreground" role="status" aria-live="polite">
                        {ttsPlaybackState.phase === 'synthesising' ? 'Preparing voice…' : 'Playing voice'}
                      </span>
                    )}
                    <Button
                      variant={isPlayingAudio === msg.id ? 'destructive' : 'ghost'}
                      size="icon"
                      className="h-9 w-9 transition-transform active:scale-90 md:h-6 md:w-6"
                      onClick={() => onSpeakerClick(msg.id, displayContent)}
                      disabled={isGenerating || isTranscribing || (isPlayingAudio && isPlayingAudio !== msg.id)}
                      title={ttsPlaybackState?.messageId === msg.id ? 'Stop voice playback' : 'Read message aloud'}
                      aria-label={ttsPlaybackState?.messageId === msg.id ? 'Stop voice playback' : 'Read message aloud'}
                    >
                      {isPlayingAudio === msg.id ? <Loader2 className="animate-spin" size={12} /> : <PlayIcon size={12} />}
                    </Button>
                    <Button variant={isPlayingAudio === msg.id ? (isStreamingTtsPaused ? 'secondary' : 'destructive') : 'ghost'} size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onChunkedSpeakerClick(msg.id, displayContent)} disabled={isGenerating || isTranscribing || (isPlayingAudio && isPlayingAudio !== msg.id)} title="Play chunked TTS">
                      {isPlayingAudio === msg.id ? (isStreamingTtsPaused ? <PlayIcon size={12} /> : <Pause size={12} />) : <FastForward size={12} />}
                    </Button>
                  </>
                )}
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onEditBotMessage(msg.id)} disabled={isGenerating || editingBotMessageId === msg.id} title="Edit AI response">
                  <Pencil size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onGenerateVariant(msg.id)} disabled={isGenerating || isTranscribing} title="Regenerate">
                  <RotateCcw size={16} />
                </Button>
                {primaryIsAPI && typeof onGenerateVariantWithModel === 'function' && (
                  <NanoGptModelSelectorPopover
                    className="inline-flex"
                    compact
                    showAutoRoutingToggle={false}
                    updatePrimaryOnSelect={false}
                    currentModelId={primaryModel}
                    primaryApiUrl={PRIMARY_API_URL}
                    onSelectModelId={(endpointId) => onGenerateVariantWithModel(msg.id, endpointId)}
                    trigger={({ setOpen, open }) => (
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-9 w-9 md:h-6 md:w-6"
                        disabled={isGenerating || isTranscribing}
                        title="Regenerate with model…"
                        onClick={() => setOpen(!open)}
                        aria-expanded={open}
                      >
                        <Cpu size={14} />
                      </Button>
                    )}
                  />
                )}
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6" onClick={() => onContinueGeneration(msg.id)} disabled={isGenerating || isTranscribing} title="Continue response">
                  <ChevronRight size={12} />
                </Button>
                <Button variant="ghost" size="icon" className="h-9 w-9 md:h-6 md:w-6 text-muted-foreground hover:bg-[var(--chat-action-hover-danger)] hover:text-destructive" onClick={() => onDeleteMessage(msg.id)} disabled={isGenerating} title="Delete message">
                  <Trash2 size={12} />
                </Button>
              </div>
            </div>
          )}
          <div className="group relative w-full">
            {editingBotMessageId === msg.id ? (
              <MessageEditField initialValue={content} messageId={msg.id} onSave={onSaveBotMessage} onCancel={onCancelBotEdit} rows={6} saveLabel="Save Edit" className="mb-2" textareaClassName="min-h-[120px]" />
            ) : (
              <>
                {canCycleGreeting && (
                  <div
                    className="mb-2 flex items-center justify-between rounded-full border border-border/50 bg-muted/40 px-2.5 py-1 text-[11px] text-muted-foreground"
                    role="group"
                    aria-label="Opening message choices"
                    title="This character card includes alternate opening messages."
                  >
                    <button
                      type="button"
                      onClick={() => onNavigateGreeting(msg.id, 'previous')}
                      disabled={isGenerating}
                      className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
                      aria-label="Previous opening message"
                    >
                      <ChevronRight size={12} className="rotate-180" />
                      <span className="hidden sm:inline">Previous</span>
                    </button>
                    <span className="tabular-nums font-medium">
                      Greeting {greetingIndex + 1} of {greetingOptions.length}
                    </span>
                    <button
                      type="button"
                      onClick={() => onNavigateGreeting(msg.id, 'next')}
                      disabled={isGenerating}
                      className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
                      aria-label="Next opening message"
                    >
                      <span className="hidden sm:inline">Next</span>
                      <ChevronRight size={12} />
                    </button>
                  </div>
                )}
                {msg.role === 'bot' && variantCount > 1 && (
                  <div className="mb-2 flex items-center justify-between rounded-full bg-muted/60 px-3 py-1.5 text-xs text-muted-foreground border border-border/50">
                    <button type="button" onClick={() => onNavigateVariant(msg.id, 'prev')} className="hover:text-foreground transition-colors flex items-center gap-1"><ChevronRight size={12} className="rotate-180" /> Previous</button>
                    <span className="tabular-nums font-medium">{variantIndex + 1} of {variantCount}</span>
                    <button type="button" onClick={() => onNavigateVariant(msg.id, 'next')} className="hover:text-foreground transition-colors flex items-center gap-1">Next <ChevronRight size={12} /></button>
                  </div>
                )}
                {msg.role === 'bot' && Array.isArray(msg.webSearchSources) && msg.webSearchSources.length > 0 && (
                  <div className="mb-2 flex flex-wrap gap-1.5" role="list" aria-label="Web search sources">
                    {msg.webSearchSources.map((s, i) => (
                      <a
                        key={s.url || `src-${i}`}
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        role="listitem"
                        className="inline-flex items-center gap-1 max-w-[16rem] truncate rounded-full border border-border/60 bg-background/80 px-2.5 py-0.5 text-[11px] text-blue-600 dark:text-blue-400 hover:bg-accent hover:underline transition-colors"
                        title={s.url}
                      >
                        <Globe size={10} className="shrink-0 opacity-60" />
                        <span className="truncate">[{i + 1}] {s.title || 'Source'}</span>
                      </a>
                    ))}
                  </div>
                )}
                {msg.error ? (
                  <div className="group relative rounded border border-red-200 bg-red-100 p-3 pr-8 dark:border-red-900 dark:bg-red-900/30">
                    <div className="break-words text-sm font-medium text-red-600 dark:text-red-400 whitespace-pre-wrap">{msg.content}</div>
                    <button type="button" onClick={() => onDeleteMessage(msg.id)} className="absolute right-2 top-2 rounded-full p-1 text-red-500 opacity-100 transition-colors hover:bg-red-200 md:opacity-0 group-hover:opacity-100 dark:hover:bg-red-900/50" title="Dismiss error">
                      <X size={14} />
                    </button>
                  </div>
                ) : msg.isStreaming && displayContent ? (
                  (() => {
                    const { completed, streaming } = splitStreamingContent(displayContent, true);
                    const tokenKey = nextStreamingTokenKey();
                    return (
                      <div className="prose prose-sm dark:prose-invert chat-prose max-w-none w-full break-words">
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
                  <ReactMarkdown components={{ code: CodeBlock }} remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]} className="prose prose-sm dark:prose-invert chat-prose max-w-none w-full break-words">
                    {displayContent}
                  </ReactMarkdown>
                )}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );

    if (isSystem) {
        return (
            <div className="message-bubble mx-auto my-1.5 max-w-[98%] rounded-lg p-1.5 text-center shadow-sm md:my-3 md:max-w-[80%] md:p-3">
                {messageBody}
            </div>
        );
    }

    return (
        <div
            className={cn(
                'message-bubble group my-1.5 p-1.5 shadow-sm transition-all duration-200 md:my-3 md:p-3',
                rowLayoutClass,
                msg.role === 'user'
                    ? 'ml-auto border border-[var(--chat-user-border)] bg-[var(--chat-user-bg)] text-secondary-foreground'
                    : 'border border-[var(--chat-bot-border)] bg-[var(--chat-bot-bg)] text-muted-foreground'
            )}
            style={{ borderRadius: 'var(--radius)', width: msg.role === 'user' ? userBubbleWidth : botBubbleWidth }}
        >
            {msg.role !== 'user' && renderAvatar(msg, PRIMARY_API_URL)}
            {messageBody}
            {msg.role === 'user' && renderUserAvatar(msg)}
        </div>
    );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage;
