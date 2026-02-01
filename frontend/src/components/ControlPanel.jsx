import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
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
    Copy,
    Layers,
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
    settings,
    showModelSelector,
    isSummarizing,
    isGeneratingCharacter,
    isAnalyzingCharacter,
    showAuthorNote,
    showStoryTracker,
    showChoiceGenerator,
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
    setIsFocusModeActive,
    updateSettings,
    handleCreateSummary,
    handleGenerateCharacter,
    setShowAuthorNote,
    setShowStoryTracker,
    setShowChoiceGenerator,
    handleCallModeToggle,
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
                "fixed right-0 top-1/2 transform -translate-y-1/2 z-50 flex flex-col transition-all duration-300 ease-in-out",
                isOpen ? "right-4" : "right-0 translate-x-[calc(100%-40px)]" // Peek out when closed
            )}
        >
            <div className={cn(
                "bg-background/95 backdrop-blur-md border border-border shadow-lg rounded-xl flex flex-col overflow-hidden transition-all duration-300",
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

                            <Button
                                variant={showModelSelector ? "secondary" : "ghost"}
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={() => setShowModelSelector(!showModelSelector)}
                                title={showModelSelector ? "Hide Models" : "Show Models"}
                            >
                                <Cpu size={20} />
                            </Button>

                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={() => setIsFocusModeActive(true)}
                                title="Enter Focus Mode"
                            >
                                <Focus size={20} />
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

                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={handleAiContinue}
                                disabled={isGenerating || isTranscribing || !messages.some(m => m.role === 'bot')}
                                title="Continue AI response"
                            >
                                <FastForward size={20} />
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

                                {/* AutoTTS Toggle */}
                                {ttsEnabled && (
                                    <div className="flex flex-col items-center justify-center pt-1" title="Toggle Auto-TTS">
                                        <Switch
                                            id="panel-autotts"
                                            checked={settings?.ttsAutoPlay || false}
                                            onCheckedChange={handleAutoPlayToggle}
                                            className="scale-75 data-[state=checked]:bg-primary"
                                        />
                                        <span className="text-[9px] font-medium text-muted-foreground mt-0.5">TTS</span>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* --- Call Mode --- */}
                        {sttEnabled && ttsEnabled && (
                            <div className="flex flex-col gap-2 py-2 border-b border-border/50">
                                <Button
                                    variant={isCallModeActive ? "destructive" : "ghost"}
                                    size="icon"
                                    onClick={handleCallModeToggle}
                                    title={isCallModeActive ? "Exit Call Mode" : "Enter Call Mode"}
                                    className={cn(
                                        "h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors",
                                        isCallModeActive && "ring-2 ring-destructive/40"
                                    )}
                                >
                                    {isCallModeActive ? <PhoneOff size={20} /> : <Phone size={20} />}
                                </Button>
                            </div>
                        )}


                        {/* --- Tools Group --- */}
                        <div className="flex flex-col gap-2 pt-2">
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={handleCreateSummary}
                                disabled={isSummarizing || messages.length < 2}
                                title="Summarize current conversation"
                            >
                                {isSummarizing ? <Loader2 className="animate-spin" size={20} /> : <Save size={20} />}
                            </Button>

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

                            <Button
                                variant={showAuthorNote ? "secondary" : "ghost"}
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={() => setShowAuthorNote(!showAuthorNote)}
                                title="Author's Note"
                            >
                                <BookOpen size={20} />
                            </Button>

                            <Button
                                variant={showStoryTracker ? "secondary" : "ghost"}
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={() => setShowStoryTracker(!showStoryTracker)}
                                title="Story Tracker"
                            >
                                <Copy size={20} />
                            </Button>

                            <Button
                                variant={showChoiceGenerator ? "secondary" : "ghost"}
                                size="icon"
                                className="h-10 w-full hover:bg-primary/10 hover:text-primary transition-colors"
                                onClick={() => setShowChoiceGenerator(!showChoiceGenerator)}
                                title="Choice Generator"
                            >
                                <Layers size={20} />
                            </Button>

                            {/* Roles Toggle */}
                            <div className="flex flex-col items-center justify-center pt-1" title="Toggle Roles">
                                <Switch
                                    id="panel-roles"
                                    checked={settings.multiRoleMode || false}
                                    onCheckedChange={(checked) => {
                                        updateSettings({
                                            multiRoleMode: checked,
                                            autoSelectSpeaker: checked ? settings.autoSelectSpeaker : false
                                        });
                                    }}
                                    className="scale-75 data-[state=checked]:bg-primary"
                                />
                                <span className="text-[9px] font-medium text-muted-foreground mt-0.5">Roles</span>
                            </div>
                            {settings.multiRoleMode && (
                                <div className="flex flex-col items-center justify-center pt-1" title="Toggle Auto Speaker">
                                    <Switch
                                        id="panel-autospeak"
                                        checked={settings.autoSelectSpeaker || false}
                                        onCheckedChange={(checked) => updateSettings({ autoSelectSpeaker: checked })}
                                        className="scale-75 data-[state=checked]:bg-primary"
                                    />
                                    <span className="text-[9px] font-medium text-muted-foreground mt-0.5">Auto</span>
                                </div>
                            )}

                            <div className="flex flex-col items-center justify-center pt-1" title="Performance Mode (reduce typing lag)">
                                <Switch
                                    id="panel-perf"
                                    checked={settings.performanceMode || false}
                                    onCheckedChange={(checked) => updateSettings({ performanceMode: checked })}
                                    className="scale-75 data-[state=checked]:bg-primary"
                                />
                                <span className="text-[9px] font-medium text-muted-foreground mt-0.5">Perf</span>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default ControlPanel;
