import React, { memo } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, X, PlayCircle as PlayIcon, RotateCcw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import CodeBlock from './CodeBlock';
import { getBackendUrl } from '../config/api';

const ChatMessage = memo(({
    msg,
    content,
    isGenerating,
    isTranscribing,
    isPlayingAudio,
    editingMessageId,
    editingMessageContent,
    editingBotMessageId,
    editingBotMessageContent,
    primaryCharacter,
    secondaryCharacter,
    userProfile,
    characterAvatarSize,
    userAvatarSize,
    variantCount,
    variantIndex,
    PRIMARY_API_URL,
    regenerationQueue,
    ttsEnabled,

    // Handlers
    onEditUserMessage,
    onCancelEdit,
    onChangeEditingMessageContent,
    onSaveEditedMessage,
    onRegenerateFromEditedPrompt,
    onDeleteMessage,

    onEditBotMessage,
    onCancelBotEdit,
    onChangeEditingBotMessageContent,
    onSaveBotMessage,
    onGenerateVariant,
    onContinueGeneration,
    onNavigateVariant,
    onSpeakerClick,
    onRegenerateImage,

    formatModelName,
}) => {

    // --- Avatar Rendering Logic ---
    const renderAvatar = (message, apiUrl, activeCharacter) => {
        const avatarSource = message.avatar || (message.role === 'bot' && activeCharacter?.avatar);
        const characterName = message.characterName
            || (message.role === 'bot' && activeCharacter?.name)
            || 'activeCharacter';

        // Responsive avatar size: smaller on mobile unless overridden
        const sizeStyle = {
            width: `${characterAvatarSize}px`,
            height: `${characterAvatarSize}px`
        };

        let displayUrl = null;
        if (avatarSource) {
            if (avatarSource.startsWith('/')) {
                displayUrl = `${apiUrl || getBackendUrl()}${avatarSource}`;
            } else {
                displayUrl = avatarSource;
            }
        }

        if (displayUrl) {
            return (
                <img
                    src={displayUrl}
                    alt={`${characterName || '?'}`}
                    onError={(e) => { e.target.style.display = 'none'; }}
                    className="rounded-full object-cover border border-gray-300 dark:border-gray-600 flex-shrink-0"
                    style={sizeStyle}
                />
            );
        } else {
            return (
                <div
                    title={characterName || '?'}
                    className="rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-sm font-semibold text-gray-600 dark:text-gray-300 border border-gray-300 dark:border-gray-600 flex-shrink-0"
                    style={sizeStyle}
                >
                    {characterName ? characterName.charAt(0).toUpperCase() : '?'}
                </div>
            );
        }
    };

    const renderUserAvatar = () => {
        const userAvatarSource = userProfile?.avatar;
        const userName = userProfile?.name || 'User';
        let userDisplayUrl = null;

        // Responsive avatar size logic similar to character avatar
        const userSizeStyle = {
            width: `${userAvatarSize}px`,
            height: `${userAvatarSize}px`
        };

        if (userAvatarSource) {
            userDisplayUrl = userAvatarSource.startsWith('/')
                ? `${PRIMARY_API_URL || getBackendUrl()}${userAvatarSource}`
                : userAvatarSource;
        }

        if (userDisplayUrl) {
            return (
                <img
                    src={userDisplayUrl}
                    alt={`${userName}'s avatar`}
                    title={userName}
                    onError={(e) => { e.target.style.display = 'none'; }}
                    className="rounded-full object-cover border border-gray-300 dark:border-gray-600 flex-shrink-0"
                    style={userSizeStyle}
                />
            );
        } else {
            return (
                <div
                    title={userName}
                    className="rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold flex-shrink-0 border border-primary/50"
                    style={userSizeStyle}
                >
                    {userName ? userName.charAt(0).toUpperCase() : 'U'}
                </div>
            );
        }
    };

    // --- Image Message Rendering ---
    if (msg.type === 'image') {
        return (
            <div
                className={cn(
                    "my-3 p-2 md:p-3 rounded-lg flex items-start gap-2 md:gap-3 shadow-sm",
                    msg.role === 'user' ? 'bg-primary/10 justify-end ml-2 md:ml-10' : 'bg-secondary mr-2 md:mr-10'
                )}
            >
                {msg.role !== 'user' && renderAvatar(msg, PRIMARY_API_URL, msg.modelId === 'primary' ? primaryCharacter : secondaryCharacter)}

                <div className={cn("flex-1 min-w-0", msg.role === 'user' ? 'order-first' : '')}>
                    <SimpleChatImageMessage message={msg} onRegenerate={onRegenerateImage} regenerationQueue={regenerationQueue} />
                </div>

                {msg.role === 'user' && renderUserAvatar()}
            </div >
        );
    }

    // --- Regular Message Rendering ---
    return (
        <div
            className={cn(
                "my-2 md:my-3 p-2 md:p-3 flex items-start gap-2 md:gap-3 shadow-sm group transition-all duration-200 message-bubble",
                msg.role === 'user'
                    ? 'bg-secondary text-secondary-foreground justify-end ml-2 md:ml-10 border border-transparent'
                    : msg.role === 'system'
                        ? 'bg-yellow-100 dark:bg-yellow-900/20 text-center mx-auto max-w-[95%] md:max-w-[80%] rounded-lg'
                        : 'bg-muted text-muted-foreground mr-2 md:mr-10 border border-border'
            )}
            style={msg.role !== 'system' ? { borderRadius: 'var(--radius)' } : {}}
        >
            {msg.role !== 'user' && renderAvatar(msg, PRIMARY_API_URL, msg.modelId === 'primary' ? primaryCharacter : secondaryCharacter)}

            {/* Added min-w-0 here to fix flexbox overflow on mobile */}
            <div className={cn("flex-1 relative min-w-0", msg.role === 'user' ? 'order-first' : '')}>
                {/* USER MESSAGE EDIT FUNCTIONALITY */}
                {msg.role === 'user' ? (
                    editingMessageId === msg.id ? (
                        // Edit mode for user messages
                        <div className="space-y-2">
                            <Textarea
                                value={editingMessageContent}
                                onChange={(e) => onChangeEditingMessageContent(e.target.value)}
                                className="w-full resize-none bg-background border-input"
                                rows={3}
                                autoFocus
                            />
                            <div className="flex gap-2 justify-end">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={onCancelEdit}
                                >
                                    Cancel
                                </Button>
                                <Button
                                    variant="default"
                                    size="sm"
                                    onClick={() => onSaveEditedMessage(msg.id, editingMessageContent)}
                                    disabled={!editingMessageContent.trim()}
                                >
                                    Save
                                </Button>
                                <Button
                                    variant="secondary"
                                    size="sm"
                                    onClick={() => {
                                        onSaveEditedMessage(msg.id, editingMessageContent);
                                        setTimeout(() => onRegenerateFromEditedPrompt(msg.id, editingMessageContent), 100);
                                    }}
                                    disabled={!editingMessageContent.trim() || isGenerating}
                                >
                                    Save & Regenerate
                                </Button>
                            </div>
                        </div>
                    ) : (
                        // Display mode for user messages with edit button
                        <div className="relative group">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-muted-foreground font-medium">You</span>
                                <div className="flex gap-1 opacity-100 md:opacity-0 group-hover:opacity-100 transition-opacity z-10 relative">
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6"
                                        onClick={() => onEditUserMessage(msg.id, msg.content)}
                                        title="Edit message"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                            <path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                        </svg>
                                    </Button>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6"
                                        onClick={() => onRegenerateFromEditedPrompt(msg.id)}
                                        disabled={isGenerating}
                                        title="Regenerate from this message"
                                    >
                                        <RotateCcw size={12} />
                                    </Button>
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6 text-muted-foreground hover:text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30"
                                        onClick={() => onDeleteMessage(msg.id)}
                                        disabled={isGenerating}
                                        title="Delete message"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <line x1="18" y1="6" x2="6" y2="18" />
                                            <line x1="6" y1="6" x2="18" y2="18" />
                                        </svg>
                                    </Button>
                                </div>
                            </div>
                            <ReactMarkdown
                                components={{ code: CodeBlock }}
                                remarkPlugins={[remarkGfm]}
                                className="prose prose-sm dark:prose-invert max-w-none break-words"
                            >
                                {msg.content}
                            </ReactMarkdown>
                        </div>
                    )
                ) : (
                    /* BOT/SYSTEM MESSAGE RENDERING */
                    <>
                        {msg.role === 'bot' && (
                            <div className="text-xs text-muted-foreground mb-1 font-medium flex items-center justify-between flex-wrap gap-2">
                                <span>{msg.characterName || (msg.modelName ? formatModelName(msg.modelName) : "Assistant")}</span>

                                <div className="flex items-center gap-1 opacity-100 md:opacity-0 group-hover:opacity-100 transition-opacity z-10 relative">
                                    {/* Per-message TTS button for non-user messages */}
                                    {ttsEnabled && msg.role !== 'user' && msg.role !== 'system' && (
                                        <Button
                                            variant={isPlayingAudio === msg.id ? "destructive" : "ghost"}
                                            size="icon"
                                            className="h-9 w-9 md:h-6 md:w-6"
                                            onClick={() => onSpeakerClick(msg.id, content)}
                                            disabled={isGenerating || isTranscribing || (isPlayingAudio && isPlayingAudio !== msg.id)}
                                        >
                                            {isPlayingAudio === msg.id ? (
                                                <Loader2 className="animate-spin" size={12} />
                                            ) : (
                                                <PlayIcon size={12} />
                                            )}
                                        </Button>
                                    )}

                                    {/* Edit button for bot messages */}
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6"
                                        onClick={() => onEditBotMessage(msg.id)}
                                        disabled={isGenerating || editingBotMessageId === msg.id}
                                        title="Edit AI response"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                            <path d="m18.5 2.5 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                        </svg>
                                    </Button>

                                    {/* Regenerate button */}
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6"
                                        onClick={() => onGenerateVariant(msg.id)}
                                        disabled={isGenerating || isTranscribing}
                                        title="Generate variant"
                                    >
                                        <RotateCcw size={16} />
                                    </Button>

                                    {/* Continue button */}
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6"
                                        onClick={() => onContinueGeneration(msg.id)}
                                        disabled={isGenerating || isTranscribing}
                                        title="Continue response"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <polyline points="9,18 15,12 9,6" />
                                        </svg>
                                    </Button>

                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-9 w-9 md:h-6 md:w-6 text-muted-foreground hover:text-red-500 hover:bg-red-100 dark:hover:bg-red-900/30"
                                        onClick={() => onDeleteMessage(msg.id)}
                                        disabled={isGenerating}
                                        title="Delete message"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <line x1="18" y1="6" x2="6" y2="18" />
                                            <line x1="6" y1="6" x2="18" y2="18" />
                                        </svg>
                                    </Button>
                                </div>
                            </div>
                        )}

                        <div className="relative group">
                            {/* Show edit mode if this message is being edited */}
                            {editingBotMessageId === msg.id ? (
                                <div className="space-y-2 mb-2">
                                    <Textarea
                                        value={editingBotMessageContent}
                                        onChange={(e) => onChangeEditingBotMessageContent(e.target.value)}
                                        className="w-full resize-none bg-background border-input min-h-[120px]"
                                        rows={6}
                                        autoFocus
                                    />
                                    <div className="flex gap-2 justify-end">
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={onCancelBotEdit}
                                        >
                                            Cancel
                                        </Button>
                                        <Button
                                            variant="default"
                                            size="sm"
                                            onClick={() => onSaveBotMessage(msg.id, editingBotMessageContent)}
                                            disabled={!editingBotMessageContent.trim()}
                                        >
                                            Save Edit
                                        </Button>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    {/* Variant navigation - only show if there are multiple variants */}
                                    {msg.role === 'bot' && variantCount > 1 && (
                                        <div className="flex items-center justify-between mb-2 text-xs text-muted-foreground bg-muted/50 px-2 py-1 rounded">
                                            <button
                                                onClick={() => onNavigateVariant(msg.id, 'prev')}
                                                className="hover:text-foreground"
                                            >
                                                ← Previous
                                            </button>
                                            <span>
                                                {variantIndex + 1} of {variantCount}
                                            </span>
                                            <button
                                                onClick={() => onNavigateVariant(msg.id, 'next')}
                                                className="hover:text-foreground"
                                            >
                                                Next →
                                            </button>
                                        </div>
                                    )}

                                    {msg.error ? (
                                        <div className="bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-900 rounded p-3 relative pr-8 group">
                                            <div className="text-sm text-red-600 dark:text-red-400 font-medium whitespace-pre-wrap break-words">
                                                {msg.content}
                                            </div>
                                            <button
                                                onClick={() => onDeleteMessage(msg.id)}
                                                className="absolute top-2 right-2 p-1 text-red-500 hover:bg-red-200 dark:hover:bg-red-900/50 rounded-full transition-colors opacity-100 md:opacity-0 group-hover:opacity-100"
                                                title="Dismiss error"
                                            >
                                                <X size={14} />
                                            </button>
                                        </div>
                                    ) : (
                                        <ReactMarkdown
                                            components={{ code: CodeBlock }}
                                            remarkPlugins={[remarkGfm]}
                                            className="prose prose-sm dark:prose-invert max-w-none break-words"
                                        >
                                            {content}
                                        </ReactMarkdown>
                                    )}
                                </>
                            )}
                        </div>
                    </>
                )}
            </div>
            {!msg.error && msg.role === 'system' && (
                <button
                    onClick={() => onDeleteMessage(msg.id)}
                    className="absolute -right-6 top-1/2 transform -translate-y-1/2 p-1 text-muted-foreground hover:text-red-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Delete system message"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>
            )}

            {msg.role === 'user' && renderUserAvatar()}
        </div>
    );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage;