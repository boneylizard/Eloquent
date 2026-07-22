import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
    Loader2,
    Plus,
    Cpu,
    Eye,
    FastForward,
    Mic,
    MicOff,
    X,
    PlayCircle as PlayIcon,
    Focus,
    Save,
    Users,
    BookOpen,
    Phone,
    PhoneOff,
    ChevronRight,
    ChevronLeft,
    Settings2,
    MoreVertical
} from 'lucide-react';
import { cn } from '@/lib/utils';

const ControlPanel = ({
    // State
    messages,
    isGenerating,
    isRecording,
    isTranscribing,
    isPlayingAudio,
    sttEnabled,
    ttsEnabled,
    showModelSelector,
    isSummarizing,
    isGeneratingCharacter,
    isAnalyzingCharacter,
    showAuthorNote,
    isCallModeActive,
    // Handlers
    setShowModelSelector,
    createNewConversation,
    handleVisualizeScene,
    handleAiContinue,
    handleMicClick,
    handleStopGeneration,
    handleSpeakerClick,
    stopTTS,
    handleAutoPlayToggle,
    isFocusModeActive,
    handleFocusModeToggle,
    handleCallModeToggle,
    handleCreateSummary,
    availableSummaries = [],
    handleAppendToSummary,
    handleGenerateCharacter,
    setShowAuthorNote,
    getCharacterButtonState,
    // New props for identifying active audio
    skippedMessageIds,
    setSkippedMessageIds
}) => {
    const [isOpen, setIsOpen] = useState(true);

    // Helper to determine active audio for the last message
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
    const isLastMessagePlaying = lastMessage && isPlayingAudio === lastMessage.id;

    // Character button state
    const charButtonState = getCharacterButtonState ? getCharacterButtonState() : {
        disabled: false, variant: "outline", className: "", title: "Generate character"
    };

    const togglePanel = () => setIsOpen(!isOpen);

    return (
        <div
            className={cn(
                "fixed z-50 flex flex-col transition-all duration-300 ease-in-out",
                // Below app navbar (--app-navbar-offset); centered on large screens
                "top-[calc(var(--app-navbar-offset,3rem)+0.5rem)] lg:top-1/2 lg:transform lg:-translate-y-1/2",
                // Horizontal positioning
                isOpen ? "right-2 lg:right-4" : "right-0 translate-x-[calc(100%-40px)]"
            )}
        >
            <div className={cn(
                "bg-background/95 backdrop-blur-md border border-border shadow-lg rounded-xl flex flex-col transition-all duration-300",
                // Scroll handling
                "max-h-[calc(100vh-6rem)] overflow-y-auto [&::-webkit-scrollbar]:hidden",
                isOpen ? "w-[60px] p-2 gap-2" : "w-[40px] p-1 gap-1 opacity-80 hover:opacity-100"
            )}>

                {/* Collapse Toggle */}
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-full mb-1 hover:bg-muted"
                    onClick={togglePanel}
                    title={isOpen ? "Collapse Controls" : "Expand Controls"}
                >
                    {isOpen ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                </Button>

                {isOpen && (
                    <>
                        {/* --- Session Group --- */}
                        <div className="flex flex-col gap-2 pb-2 border-b border-border/50">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={createNewConversation}
                                title="New Chat"
                            >
                                <Plus size={20} />
                            </Button>
                        </div>

                        {/* --- Generation Group --- */}
                        <div className="flex flex-col gap-2 py-2 border-b border-border/50">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={handleVisualizeScene}
                                disabled={isGenerating || messages.length === 0}
                                title="Visualize Scene"
                            >
                                <Eye size={20} />
                            </Button>

                            {isGenerating && (
                                <Button
                                    variant="destructive"
                                    size="icon"
                                    className="h-10 w-full animate-pulse"
                                    onClick={handleStopGeneration}
                                    title="Stop Generation"
                                >
                                    <X size={20} />
                                </Button>
                            )}
                        </div>

                        {/* --- Audio Group --- */}
                        {(sttEnabled || ttsEnabled) && (
                            <div className="flex flex-col gap-2 py-2 border-b border-border/50">
                                {sttEnabled && (
                                    <Button
                                        variant={isRecording ? "destructive" : "ghost"}
                                        size="icon"
                                        className={cn(
                                            "h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors",
                                            isRecording && "animate-pulse ring-2 ring-destructive/40"
                                        )}
                                        onClick={handleMicClick}
                                        disabled={isTranscribing}
                                        title={isRecording ? "Stop Recording" : "Start Voice Input"}
                                    >
                                        {isTranscribing ? <Loader2 className="animate-spin" size={20} /> : isRecording ? <MicOff size={20} /> : <Mic size={20} />}
                                    </Button>
                                )}

                                {lastMessage && lastMessage.role !== "user" && ttsEnabled && (
                                    <Button
                                        variant={isLastMessagePlaying ? "destructive" : "ghost"}
                                        size="icon"
                                        className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                        onClick={() => handleSpeakerClick(lastMessage.id, lastMessage.content)}
                                        disabled={isGenerating || isTranscribing || (isPlayingAudio && !isLastMessagePlaying)}
                                        title={isLastMessagePlaying ? "Stop Audio" : "Play Response"}
                                    >
                                        {isLastMessagePlaying ? <Loader2 className="animate-spin" size={20} /> : <PlayIcon size={20} />}
                                    </Button>
                                )}

                                {/* Global Stop Audio (if something is playing) */}
                                {ttsEnabled && isPlayingAudio && (
                                    <Button
                                        variant="destructive"
                                        size="icon"
                                        className="h-10 w-full"
                                        onClick={() => {
                                            if (isPlayingAudio) setSkippedMessageIds(prev => new Set(prev).add(isPlayingAudio));
                                            stopTTS();
                                        }}
                                        title="Stop All Audio"
                                    >
                                        <X size={20} />
                                    </Button>
                                )}
                            </div>
                        )}




                        {/* --- Tools Group --- */}
                        <div className="flex flex-col gap-2 pt-2">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={handleCreateSummary}
                                disabled={isSummarizing || isGenerating || messages.length < 2}
                                title="Summarize current conversation"
                            >
                                {isSummarizing ? <Loader2 className="animate-spin" size={20} /> : <Save size={20} />}
                            </Button>
                            {availableSummaries.length > 0 && messages.length >= 2 && (
                                <select
                                    className="h-9 w-full px-2 text-xs border rounded bg-background text-muted-foreground"
                                    value=""
                                    onChange={(e) => {
                                        const id = e.target.value;
                                        e.target.value = '';
                                        if (!id) return;
                                        const summary = availableSummaries.find(s => s.id === id);
                                        if (summary) handleAppendToSummary(summary);
                                    }}
                                    disabled={isGenerating}
                                    title="Append current chat to an existing summary"
                                >
                                    <option value="">Append to...</option>
                                    {availableSummaries.map(s => (
                                        <option key={s.id} value={s.id}>{s.title.length > 20 ? s.title.slice(0, 20) + '…' : s.title}</option>
                                    ))}
                                </select>
                            )}

                            <Button
                                variant={charButtonState.variant === "outline" ? "ghost" : charButtonState.variant} // Normalize to ghost
                                size="icon"
                                className={cn("h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors", charButtonState.className)}
                                onClick={handleGenerateCharacter}
                                disabled={charButtonState.disabled || isGeneratingCharacter || isAnalyzingCharacter}
                                title={charButtonState.title}
                            >
                                {isGeneratingCharacter ? <Loader2 className="animate-spin" size={20} /> : isAnalyzingCharacter ? <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" /> : <Users size={20} />}
                            </Button>

                            <div className="flex flex-col gap-1 border-t border-border/30 pt-2 mt-2">
                                <Button
                                    type="button"
                                    variant="ghost"
                                    className={cn(
                                        "h-auto min-h-10 w-full flex-col gap-0.5 px-1 py-1.5 transition-colors",
                                        isFocusModeActive
                                            ? "bg-purple-500/15 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200"
                                            : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                                    )}
                                    onClick={handleFocusModeToggle}
                                    aria-pressed={isFocusModeActive}
                                    title={isFocusModeActive ? "Exit Focus Mode" : "Enter Focus Mode"}
                                >
                                    {isFocusModeActive ? <X size={15} /> : <Focus size={15} />}
                                    <span className="text-[9px] font-medium">Focus</span>
                                </Button>

                                <Button
                                    type="button"
                                    variant="ghost"
                                    className={cn(
                                        "h-auto min-h-10 w-full flex-col gap-0.5 px-1 py-1.5 transition-colors",
                                        isCallModeActive
                                            ? "bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/20 hover:text-cyan-200"
                                            : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
                                    )}
                                    onClick={handleCallModeToggle}
                                    aria-pressed={isCallModeActive}
                                    title={isCallModeActive ? "Exit Call Mode" : "Start Call Mode"}
                                >
                                    {isCallModeActive ? <PhoneOff size={15} /> : <Phone size={15} />}
                                    <span className="text-[9px] font-medium">Call</span>
                                </Button>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default ControlPanel;
