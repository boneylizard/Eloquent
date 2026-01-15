import React, { useState, useEffect, useRef, useCallback, useMemo, inputRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Loader2, Send, Layers, Users, Mic, MicOff, Copy, Check, PlayCircle as PlayIcon, X, Cpu, RotateCcw, Globe, Phone, PhoneOff, Focus, Code, ArrowLeft, Eye } from 'lucide-react';
import { cn } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import CharacterSelector from './CharacterSelector';
import ModelSelector from './ModelSelector';
import SimpleChatImageButton from './SimpleChatImageButton';
import SimpleChatImageMessage from './SimpleChatImageMessage';
import RAGIndicator from './RAGIndicator';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import ChatImageUploadButton from './ChatImageUploadButton';
import FocusModeOverlay from './FocusModeOverlay';
import CallModeOverlay from './CallModeOverlay';
import CodeBlock from './CodeBlock';
import ChatInputForm from './ChatInputForm';
import ChatMessage from './ChatMessage';
import CodeEditorOverlay from './CodeEditorOverlay';
import ForensicLinguistics from './ForensicLinguistics';
import StoryTracker, { getStoryTrackerContext } from './StoryTracker';
import ChoiceGenerator from './ChoiceGenerator';
import { getBackendUrl } from '../config/api';

// CORRECT PLACEMENT: Component defined at the top level, accepting props.
const WebSearchControl = ({ webSearchEnabled, setWebSearchEnabled, isGenerating, isRecording, isTranscribing }) => (
  <div className="flex items-center gap-2 px-2 py-1 bg-muted/50 rounded-md border">
    <Switch
      id="web-search"
      checked={webSearchEnabled}
      onCheckedChange={setWebSearchEnabled}
      disabled={isGenerating || isRecording || isTranscribing}
    />
    <Label htmlFor="web-search" className="text-xs flex items-center gap-1">
      <Globe size={14} className={webSearchEnabled ? 'text-blue-500' : 'text-muted-foreground'} />
      Web Search
    </Label>
  </div>
);

// Main Chat Component
const Chat = ({ layoutMode }) => {
  // Get state and functions from useApp context
  const {
    // Model/Chat state
    activeModel, primaryModel, secondaryModel, dualModeEnabled, setDualModeEnabled, buildSystemPrompt, formatPrompt, cleanModelOutput, botMsg, abortController, setAbortController,
    messages, setMessages, sendMessage, sendDualMessage, isGenerating, isModelLoading,
    createNewConversation, startAgentConversation, agentConversationActive, PRIMARY_API_URL, generateReply, fetchMemoriesFromAgent, fetchTriggeredLore, isStreamingStopped, handleStopGeneration,
    // Character info
    activeCharacter, primaryCharacter, secondaryCharacter,
    getGenerationSystemPrompt,
    // Audio / STT / TTS flags & functions
    sttEnabled, ttsEnabled, isRecording, isTranscribing, primaryIsAPI, secondaryIsAPI,
    isPlayingAudio, playTTS, stopTTS, audioError, setAudioError, generateUniqueId, saveCharacter, generateImage, SECONDARY_API_URL, startStreamingTTS, stopStreamingTTS, addStreamingText, endStreamingTTS,
    startRecording, stopRecording, MEMORY_API_URL, ttsClient, setAudioQueue, setIsAutoplaying,
    // Avatar sizes
    userAvatarSize, characterAvatarSize, speechDetected, audioQueue, isAutoplaying, callModeRecording,
    // User profile
    userProfile,
    // Settings
    settings, updateSettings, setIsGenerating, activeConversation, isCallModeActive, startCallMode, stopCallMode, setIsCallModeActive,
    backgroundImage, // Add backgroundImage from context
  } = useApp();

  // Local state for the input field
  const [messageVariants, setMessageVariants] = useState({}); // Store variants by message ID
  const [currentVariantIndex, setCurrentVariantIndex] = useState({}); // Track which variant is showing
  const [agentTopic, setAgentTopic] = useState('');
  const [agentTurns, setAgentTurns] = useState(3);
  const audioPlaybackRef = useRef({ context: null, source: null });
  const [editingMessageId, setEditingMessageId] = useState(null);
  const [editingMessageContent, setEditingMessageContent] = useState('');
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [showFloatingControls, setShowFloatingControls] = useState(true);
  const [regeneratingMessageData, setRegeneratingMessageData] = useState(null);
  const [manuallyStoppedAudio, setManuallyStoppedAudio] = useState(false);
  const [autoplayPaused, setAutoplayPaused] = useState(false);
  const messagesEndRef = useRef(null);
  const [inputValue, setInputValue] = useState('');
  const [isFocusModeActive, setIsFocusModeActive] = useState(false);
  const [regeneratingMessageId, setRegeneratingMessageId] = useState(null);
  const prevMessageCount = useRef(messages.length);
  const [skippedMessageIds, setSkippedMessageIds] = useState(new Set());
  const [editingBotMessageId, setEditingBotMessageId] = useState(null);
  const [editingBotMessageContent, setEditingBotMessageContent] = useState('');
  const [setIsStreamingStopped] = useState(false);
  const [codeEditorEnabled, setCodeEditorEnabled] = useState(false);
  const [showForensicLinguistics, setShowForensicLinguistics] = useState(false);
  const [characterReadiness, setCharacterReadiness] = useState({
    score: 0,
    detected_elements: [],
    suggested_names: [],
    status: 'idle'
  });
  const [isAnalyzingCharacter, setIsAnalyzingCharacter] = useState(false);
  const [showCharacterPreview, setShowCharacterPreview] = useState(false);
  const [generatedCharacter, setGeneratedCharacter] = useState(null);
  const [isGeneratingCharacter, setIsGeneratingCharacter] = useState(false);
  const [characterFeedback, setCharacterFeedback] = useState('');
  const [isGeneratingCharacterImage, setIsGeneratingCharacterImage] = useState(false);
  const [characterImageUrl, setCharacterImageUrl] = useState(null);
  const [characterImagePrompt, setCharacterImagePrompt] = useState('');
  const [customImagePrompt, setCustomImagePrompt] = useState('');
  const [regenerationQueue, setRegenerationQueue] = useState(0);
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const streamingTtsMessageIdRef = useRef(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [characterImageSettings, setCharacterImageSettings] = useState({
    width: 512,
    height: 512,
    steps: 25,
    guidance_scale: 7.5,
    sampler: 'Euler a',
    seed: -1,
    model: ''
  });
  const [availableModels, setAvailableModels] = useState([]);
  const [autoAnalyzeImages, setAutoAnalyzeImages] = useState(false);
  // Author's Note state - persist to localStorage
  const [authorNoteEnabled, setAuthorNoteEnabled] = useState(false);
  const [authorNote, setAuthorNote] = useState(() => {
    return localStorage.getItem('eloquent-author-note') || '';
  });
  const [showAuthorNote, setShowAuthorNote] = useState(false);

  // Story Tracker and Choice Generator state
  const [showStoryTracker, setShowStoryTracker] = useState(false);
  const [showChoiceGenerator, setShowChoiceGenerator] = useState(false);
  const [isAnalyzingStory, setIsAnalyzingStory] = useState(false);

  const autoEnhanceEnabled = localStorage.getItem('adetailer-auto-enhance') === 'true';
  const adetailerSettings = JSON.parse(localStorage.getItem('adetailer-settings') || '{}');
  const selectedAdetailerModel = localStorage.getItem('adetailer-selected-model') || 'face_yolov8n.pt';



  const handleVisualizeScene = useCallback(async () => {
    if (messages.length === 0 || isGenerating) return;

    // Optional: Add a system message saying "Visualizing scene..."
    const tempId = generateUniqueId();
    setMessages(prev => [...prev, {
      id: tempId,
      role: 'system',
      content: '🎨 Visualizing current scene...'
    }]);

    try {
      const response = await fetch(`${PRIMARY_API_URL}/sd-local/visualize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages,
          model_name: primaryModel,
          gpu_id: 0
        })
      });

      if (!response.ok) throw new Error('Visualization failed');

      const data = await response.json();

      // Remove temp message and add image message
      setMessages(prev => {
        const filtered = prev.filter(m => m.id !== tempId);
        return [...filtered, {
          id: generateUniqueId(),
          role: 'bot',
          type: 'image', // Ensure it uses the image renderer
          content: data.generated_prompt, // Use prompt as content
          imagePath: data.image_url,
          prompt: data.generated_prompt,
          model: 'SD-Local',
          timestamp: new Date().toISOString()
        }];
      });

    } catch (error) {
      console.error('Visualization error:', error);
      setMessages(prev => prev.map(m => m.id === tempId ? { ...m, content: `❌ Visualization failed: ${error.message}`, error: true } : m));
    }
  }, [messages, isGenerating, primaryModel, PRIMARY_API_URL, setMessages, generateUniqueId]);

  // Author's Note helper functions
  const countTokens = (text) => {
    // Simple token estimation: roughly 4 characters per token
    return Math.ceil(text.length / 4);
  };

  const handleAuthorNoteChange = (value) => {
    const tokenCount = countTokens(value);
    if (tokenCount <= 150) {
      setAuthorNote(value);
      localStorage.setItem('eloquent-author-note', value);
    }
  };

  const clearAuthorNote = () => {
    setAuthorNote('');
    localStorage.removeItem('eloquent-author-note');
  };

  const getAuthorNoteTokenCount = () => countTokens(authorNote);

  // Story Tracker auto-detect handler
  const handleAnalyzeStory = async () => {
    if (messages.length === 0) return;

    setIsAnalyzingStory(true);
    try {
      const recentMessages = messages.slice(-15);
      const context = recentMessages
        .map(m => `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`)
        .join('\n');

      const prompt = `You are a story state tracker for a roleplay conversation.
Your job is to extract the *current state* from the conversation.

Carefully read the conversation and decide:
- characters: distinct named people/beings involved so far
- inventory: items the protagonist currently has, found, or clearly owns
- locations: important places that matter to the story
- plotPoints: key events that have already happened
- sceneSummary: a one-sentence summary of the current scene and situation

You may think through the conversation internally, but your *final* output must be a single JSON object on the last line, with **no extra text before or after it**.
Do NOT include backticks, code fences, or explanations around the JSON.

The JSON must have exactly this shape:
{
  "characters": ["Name A", "Name B"],
  "inventory": ["item A", "item B"],
  "locations": ["place A", "place B"],
  "plotPoints": ["event A", "event B"],
  "sceneSummary": "Current situation description..."
}

If you are unsure about a field, return an empty array for that field (or an empty string for sceneSummary).

CONVERSATION:
${context}

Now output ONLY the final JSON object on the last line, with no commentary:`;

      if (!primaryModel) {
        console.error('Story analysis error: No model loaded');
        return;
      }

      const response = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          model_name: primaryModel, // Add required model_name field
          max_tokens: 500,
          temperature: 0.3,
          stop: ['\n\n'],
          stream: true, // Enable streaming
          gpu_id: 0,
          request_purpose: 'story_analysis' // Add purpose for tracking
        })
      });

      if (response.ok) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let sseBuffer = ''; // Buffer for incomplete SSE events

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          sseBuffer += chunk;

          // Process complete SSE events (each ends with \n\n)
          const events = sseBuffer.split('\n\n');
          sseBuffer = events.pop() || '';

          for (const line of events) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              // Handle both text and token formats
              if (parsed.text) {
                fullText += parsed.text;
              } else if (parsed.token) {
                fullText += parsed.token;
              }
            } catch (e) {
              // Ignore malformed chunks
            }
          }
        }

        // Debug log to help inspect provider output if parsing fails
        console.debug('[StoryTracker] Raw model output:', fullText);

        // Parse and merge with existing tracker data
        // Extract the JSON object from the first '{' to the last '}'
        const firstBrace = fullText.indexOf('{');
        const lastBrace = fullText.lastIndexOf('}');

        if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
          const jsonStr = fullText.slice(firstBrace, lastBrace + 1);
          let detected;
          try {
            detected = JSON.parse(jsonStr);
          } catch (e) {
            console.warn('[StoryTracker] Failed to parse JSON from model output:', e, jsonStr);
            detected = null;
          }

          if (detected && typeof detected === 'object') {
            const existing = JSON.parse(localStorage.getItem('eloquent-story-tracker') || '{}');

            const mergeUnique = (arr1 = [], arr2 = []) => {
              const existingValues = new Set(arr1.map(i => i.value?.toLowerCase()));
              const newItems = arr2
                .filter(val => val && !existingValues.has(val.toLowerCase()))
                .map(val => ({ id: Date.now() + Math.random(), value: val, notes: '' }));
              return [...arr1, ...newItems];
            };

            const merged = {
              characters: mergeUnique(existing.characters, detected.characters),
              inventory: mergeUnique(existing.inventory, detected.inventory),
              locations: mergeUnique(existing.locations, detected.locations),
              plotPoints: mergeUnique(existing.plotPoints, detected.plotPoints),
              sceneSummary: detected.sceneSummary || existing.sceneSummary || '',
              storyNotes: existing.storyNotes || '',
              currentObjective: existing.currentObjective || '',
              customFields: existing.customFields || []
            };

            localStorage.setItem('eloquent-story-tracker', JSON.stringify(merged));
            // Force re-render of tracker if open
            setShowStoryTracker(false);
            setTimeout(() => setShowStoryTracker(true), 50);
          } else {
            console.warn('[StoryTracker] No valid JSON object detected in model output.');
          }
        }
      }
    } catch (err) {
      console.error('Story analysis error:', err);
    } finally {
      setIsAnalyzingStory(false);
    }
  };

  // Choice selection handler - sends the action to the AI for a response
  const handleChoiceSelect = async (choice, description = '', type = 'prose') => {
    if (!choice || isGenerating) return;

    let messageToSend = choice;

    if (type === 'director') {
      // Format as a director's/OOC note to steer the narrative
      messageToSend = `(Director: ${choice})`;
    } else if (type === 'direct') {
      // Format as a simple action if it wasn't expanded into prose
      messageToSend = `*${choice}*`;
    }
    // If type is 'prose', the choice is already the full expanded text

    // Send the message directly to get an AI response
    await sendMessage(messageToSend);
  };

  // Mic click handler
  const handleMicClick = () => {
    if (audioError) setAudioError(null); // Clear error on interaction

    if (isRecording) {
      // Pass setInputValue as the callback to stopRecording
      stopRecording(setInputValue);
    } else {
      startRecording();
    }
  };

  // Handle "back" - remove ONLY the last message (User or Bot)
  const handleBack = useCallback(() => {
    if (messages.length === 0) return;

    // Remove only the very last message
    // This allows undoing a specific generation or user input without wiping the whole turn
    setMessages(prev => {
      const lastMsg = prev[prev.length - 1];
      const newMessages = prev.slice(0, -1);

      // Clean up variants if we removed a message
      if (lastMsg) {
        setMessageVariants(v => {
          const newV = { ...v };
          delete newV[lastMsg.id];
          return newV;
        });
      }
      return newMessages;
    });

    // Stop any ongoing audio playback
    if (audioPlaybackRef.current.source) {
      audioPlaybackRef.current.source.stop();
    }
    if (audioPlaybackRef.current.context) {
      audioPlaybackRef.current.context.close();
      audioPlaybackRef.current = { context: null, source: null };
    }
    stopTTS();
  }, [messages, setMessages, setMessageVariants, stopTTS]);

  const handleSaveEditedMessage = useCallback(async (messageId, newContent) => {
    if (!newContent.trim()) return;

    // Update the message in the messages array
    setMessages(prev => prev.map(msg =>
      msg.id === messageId
        ? { ...msg, content: newContent.trim() }
        : msg
    ));

    // Clear editing state
    setEditingMessageId(null);
    setEditingMessageContent('');
  }, [setMessages]);

  const handleCancelEdit = useCallback(() => {
    setEditingMessageId(null);
    setEditingMessageContent('');
  }, []);
  // Add these functions before your return statement:
  const handleDeleteMessage = useCallback((id) => {
    setMessages(prev => prev.filter(m => m.id !== id));
  }, [setMessages]);

  const handleEditUserMessage = useCallback((messageId, currentContent) => {
    setEditingMessageId(messageId);
    setEditingMessageContent(currentContent);
  }, []);
  // Add this regeneration function that uses the edited prompt
  const handleRegenerateFromEditedPrompt = useCallback(
    async (userMessageId, overrideContent = null) => {
      if (isGenerating) return;

      // 1) Locate the user message
      const userMsgIndex = messages.findIndex((m) => m.id === userMessageId);
      if (userMsgIndex < 0) return;

      const userMsg = messages[userMsgIndex];
      if (userMsg.role !== "user") return;

      // 2) Use overrideContent (from "Save & Regenerate") to avoid React state race
      const editedPromptText = (overrideContent ?? userMsg.content ?? "").trim();
      if (!editedPromptText) return;

      // 3) Build a local history slice that definitely contains the edited content
      const regenHistory = messages
        .slice(0, userMsgIndex + 1)
        .map((m) => (m.id === userMessageId ? { ...m, content: editedPromptText } : m));

      // 4) Find the FIRST bot message after this user message
      let botMsgIndex = -1;
      for (let i = userMsgIndex + 1; i < messages.length; i++) {
        if (messages[i].role === "bot") {
          botMsgIndex = i;
          break;
        }
      }

      // --- CASE A: No bot message to regenerate; create a new bot reply ---
      if (botMsgIndex === -1) {
        setIsGenerating(true);
        const botId = generateUniqueId();

        try {
          if (settings.ttsAutoPlay && settings.ttsEnabled) startStreamingTTS(botId);

          // Placeholder for streaming
          const placeholderBotMessage = {
            id: botId,
            role: "bot",
            content: "",
            modelId: "primary",
            characterName: activeCharacter?.name,
            avatar: activeCharacter?.avatar,
            isStreaming: true,
          };
          setMessages((prev) => [...prev, placeholderBotMessage]);

          let lastSentTtsContent = "";
          await generateReply(
            editedPromptText,
            regenHistory,
            (token, fullContent) => {
              // Update UI
              setMessages((prev) => prev.map((m) => (m.id === botId ? { ...m, content: fullContent } : m)));

              // Update TTS
              if (settings.ttsAutoPlay && settings.ttsEnabled) {
                const newTextChunk = fullContent.slice(lastSentTtsContent.length);
                if (newTextChunk) {
                  addStreamingText(newTextChunk);
                  lastSentTtsContent = fullContent;
                }
              }
            },
            { webSearchEnabled, authorNote }
          );

          // Finalize
          setMessages((prev) => prev.map((m) => (m.id === botId ? { ...m, isStreaming: false } : m)));
          if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();
        } catch (err) {
          console.error("Regeneration error:", err);
          // Optional: mark placeholder as failed
          setMessages((prev) =>
            prev.map((m) => (m.id === botId ? { ...m, isStreaming: false } : m))
          );
          if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();
        } finally {
          setIsGenerating(false);
        }
        return;
      }

      // --- CASE B: Regenerate as a variant on the first bot message after the edited user message ---
      const botMsg = messages[botMsgIndex];
      const botMsgId = botMsg.id;

      // Add variant + compute new index in a way that never relies on stale `messageVariants`
      let newVariantIndex = 0;
      setMessageVariants((prev) => {
        const existingVariants = prev[botMsgId] || [];
        const currentContent = botMsg.content ?? "";

        // Ensure the current content is captured at least once (if variants list doesn't already include it)
        const nextVariants = existingVariants.includes(currentContent)
          ? [...existingVariants, ""]
          : [...existingVariants, currentContent, ""];

        newVariantIndex = nextVariants.length - 1;

        return {
          ...prev,
          [botMsgId]: nextVariants,
        };
      });

      // Point UI to the new (last) variant using the same computed index
      setCurrentVariantIndex((prev) => ({
        ...prev,
        [botMsgId]: newVariantIndex,
      }));

      setIsGenerating(true);
      try {
        if (settings.ttsAutoPlay && settings.ttsEnabled) startStreamingTTS(botMsgId);

        let lastSentTtsContent = "";
        const botContent = await generateReply(
          editedPromptText,
          regenHistory,
          (token, fullContent) => {
            // Update messages for streaming
            setMessages((prev) =>
              prev.map((m) => (m.id === botMsgId ? { ...m, content: fullContent, isStreaming: true } : m))
            );

            // Update variants for streaming (always update the last slot)
            setMessageVariants((prev) => {
              const arr = (prev[botMsgId] || []).slice();
              if (arr.length === 0) arr.push(fullContent);
              else arr[arr.length - 1] = fullContent;
              return { ...prev, [botMsgId]: arr };
            });

            // Update TTS
            if (settings.ttsAutoPlay && settings.ttsEnabled) {
              const newTextChunk = fullContent.slice(lastSentTtsContent.length);
              if (newTextChunk) {
                addStreamingText(newTextChunk);
                lastSentTtsContent = fullContent;
              }
            }
          },
          { webSearchEnabled, authorNote }
        );

        // Finalize message state
        setMessages((prev) =>
          prev.map((m) => (m.id === botMsgId ? { ...m, content: botContent, isStreaming: false } : m))
        );

        if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();
      } catch (err) {
        console.error("Regeneration error:", err);
        setMessages((prev) =>
          prev.map((m) => (m.id === botMsgId ? { ...m, isStreaming: false } : m))
        );
        if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();
      } finally {
        setIsGenerating(false);
      }
    },
    [
      isGenerating,
      messages,
      setMessages,
      setMessageVariants,
      setCurrentVariantIndex,
      setIsGenerating,
      activeCharacter,
      generateReply,
      settings,
      webSearchEnabled,
      authorNote,
    ]
  );



  // Replace the existing generateCharacterImagePrompt function in Chat.jsx with this much better version:

  const generateCharacterImagePrompt = useCallback((character) => {
    if (!character) return '';

    const name = character.name || 'character';
    const description = character.description || '';
    const scenario = character.scenario || '';
    const modelInstructions = character.model_instructions || '';
    const loreEntries = character.loreEntries || [];

    let promptParts = [];

    // Start with basic character type
    promptParts.push(`portrait of ${name}`);

    // Extract profession/role from description and scenario
    const professionTerms = [
      'researcher', 'scientist', 'professor', 'doctor', 'engineer', 'artist', 'warrior', 'mage', 'knight',
      'merchant', 'scholar', 'student', 'teacher', 'investigator', 'detective', 'journalist', 'writer',
      'programmer', 'developer', 'analyst', 'consultant', 'therapist', 'psychologist', 'physician',
      'librarian', 'archivist', 'historian', 'archaeologist', 'anthropologist', 'sociologist',
      'philosopher', 'theologian', 'monk', 'priest', 'nun', 'shaman', 'witch', 'sorcerer',
      'admiral', 'captain', 'general', 'soldier', 'guard', 'spy', 'assassin', 'thief',
      'chef', 'cook', 'baker', 'farmer', 'blacksmith', 'craftsman', 'artisan', 'musician', 'dancer',
      'actor', 'singer', 'poet', 'composer', 'director', 'producer', 'photographer', 'videographer',
      'designer', 'architect', 'engineer', 'mechanic', 'pilot', 'astronaut', 'explorer', 'adventurer',
      'navigator', 'cartographer', 'geologist', 'biologist', 'chemist', 'physicist', 'mathematician',
      'statistician', 'economist', 'politician', 'diplomat', 'activist', 'lobbyist', 'lawyer', 'judge',
      'barrister', 'solicitor', 'paralegal', 'notary', 'clerk', 'secretary', 'administrator', 'manager',
      'executive', 'entrepreneur', 'investor', 'banker', 'accountant', 'auditor', 'financial analyst',
      'insurance agent', 'real estate agent', 'salesperson', 'marketer', 'public relations specialist',
      'human resources manager', 'recruiter', 'trainer', 'coach', 'consultant', 'strategist', 'analyst',
      'researcher', 'developer', 'programmer', 'data scientist', 'data analyst', 'web developer', 'software engineer',
      'mobile developer', 'game developer', 'system administrator', 'network engineer', 'security analyst',
      'cloud architect', 'AI specialist', 'machine learning engineer', 'data engineer', 'DevOps engineer', 'UI/UX designer',
      'graphic designer', 'visual artist', 'illustrator', 'animator', '3D modeler', 'game designer', 'sound designer',
      'audio engineer', 'video editor', 'content creator', 'blogger', 'vlogger', 'streamer', 'influencer',
      'social media manager', 'community manager', 'customer support representative', 'technical support specialist',
      'help desk technician', 'IT support', 'field technician', 'maintenance worker', 'janitor', 'cleaner', 'security guard',
      'bouncer', 'doorman', 'concierge', 'receptionist', 'waiter', 'waitress', 'bartender', 'barista', 'hostess', 'cashier',
      'sales associate', 'retail worker', 'store manager', 'warehouse worker', 'logistics coordinator', 'supply chain manager',
      'transportation specialist', 'delivery driver', 'truck driver', 'bus driver', 'taxi driver', 'chauffeur', 'pilot',
      'flight attendant', 'air traffic controller', 'ship captain', 'fisherman', 'farmer', 'rancher', 'gardener', 'landscaper',
      'horticulturist', 'florist', 'veterinarian', 'zookeeper', 'animal trainer', 'wildlife biologist', 'conservationist',
      'environmentalist', 'ecologist', 'geographer', 'cartographer', 'meteorologist', 'climatologist', 'oceanographer', 'astronomer',
      'astrophysicist', 'cosmologist', 'geophysicist', 'geologist', 'paleontologist', 'archaeologist', 'anthropologist', 'sociologist',
      'psychologist', 'psychiatrist', 'counselor', 'therapist', 'social worker', 'nurse', 'doctor', 'surgeon', 'dentist',
      'pharmacist', 'optometrist', 'chiropractor', 'physiotherapist', 'occupational therapist', 'speech therapist',
      'nutritionist', 'dietitian', 'health coach', 'fitness trainer', 'yoga instructor', 'pilates instructor', 'martial arts instructor',
      'lifestyle coach', 'life coach', 'business coach', 'career coach', 'executive coach', 'leadership coach', 'team coach',
      'relationship coach', 'parenting coach', 'financial coach', 'wealth manager', 'investment advisor', 'tax advisor',
      'estate planner', 'retirement planner', 'insurance broker', 'real estate broker', 'mortgage broker', 'loan officer',
      'financial planner', 'business consultant', 'management consultant', 'strategy consultant', 'operations consultant',
      'marketing consultant', 'sales consultant', 'HR consultant', 'IT consultant', 'legal consultant', 'compliance consultant',
      'risk management consultant', 'project manager', 'program manager', 'portfolio manager', 'product manager',
      'brand manager', 'content manager', 'social media manager', 'community manager', 'customer success manager',
      'technical account manager', 'client success manager', 'partner manager', 'channel manager', 'business development manager',
      'sales manager', 'marketing manager', 'operations manager', 'finance manager', 'HR manager', 'IT manager',
      'legal manager', 'compliance manager', 'risk manager', 'project coordinator', 'program coordinator'
    ];

    const settingTerms = [
      'academic', 'university', 'laboratory', 'research', 'conference', 'office', 'corporate',
      'medieval', 'fantasy', 'modern', 'futuristic', 'cyberpunk', 'steampunk', 'victorian',
      'ancient', 'classical', 'renaissance', 'baroque', 'gothic', 'art deco', 'industrial', 'post-apocalyptic',
      'urban', 'rural', 'suburban', 'countryside', 'coastal', 'desert', 'jungle', 'arctic',
      'space', 'sci-fi', 'dystopian', 'utopian', 'mythical', 'supernatural', 'horror', 'thriller',
      'noir', 'western', 'historical', 'contemporary', 'futuristic', 'retro', 'vintage', 'modernist',
      'minimalist', 'maximalist', 'abstract', 'surreal', 'fantastical', 'whimsical', 'magical', 'enchanted',
      'steampunk', 'cybernetic', 'biopunk', 'dieselpunk', 'atompunk', 'clockpunk', 'solarpunk', 'lunarpunk',
      'postmodern', 'avant-garde', 'experimental', 'conceptual', 'performance', 'installation', 'street art',
      'graffiti', 'mural', 'digital art', 'photography', 'illustration', 'comics', 'graphic novel',
      'anime', 'manga', 'cartoon', '3D art', 'pixel art', 'low poly', 'high poly', 'realistic',
      'impressionist', 'expressionist', 'cubist', 'futurist', 'constructivist', 'dadaist', 'surrealist',
      'abstract expressionist', 'color field', 'op art', 'kinetic art', 'light art', 'sound art', 'video art',
      'installation art', 'performance art', 'conceptual art', 'social practice', 'community art', 'public art',
      'environmental art', 'eco-art', 'land art', 'earthworks', 'site-specific', 'contextual', 'interactive',
      'immersive', 'augmented reality', 'virtual reality', 'mixed media', 'collage', 'assemblage', 'found object'
    ];

    const personalityTerms = [
      'thoughtful', 'measured', 'serious', 'professional', 'confident', 'wise', 'intelligent',
      'charismatic', 'charming', 'mysterious', 'brooding', 'cheerful', 'optimistic', 'cynical',
      'stern', 'gentle', 'kind', 'harsh', 'cold', 'warm', 'passionate', 'calm', 'intense', 'focused', 'distracted',
      'humorous', 'witty', 'sarcastic', 'dry', 'playful', 'serious-minded', 'analytical', 'logical',
      'creative', 'imaginative', 'innovative', 'practical', 'down-to-earth', 'idealistic', 'realistic',
      'empathetic', 'sympathetic', 'compassionate', 'understanding', 'patient', 'impatient', 'forgiving',
      'judgmental', 'critical', 'supportive', 'encouraging', 'motivational', 'inspirational', 'aspirational',
      'ambitious', 'driven', 'goal-oriented', 'results-focused', 'detail-oriented', 'big-picture thinker',
      'perfectionist', 'procrastinator', 'organized', 'disorganized', 'methodical', 'spontaneous',
      'systematic', 'intuitive', 'pragmatic', 'theoretical', 'philosophical', 'scientific', 'spiritual',
      'religious', 'secular', 'moral', 'ethical', 'principled', 'unprincipled', 'honest', 'dishonest',
      'trustworthy', 'untrustworthy', 'loyal', 'disloyal', 'faithful', 'unfaithful', 'dependable',
      'unreliable', 'punctual', 'tardy', 'responsible', 'irresponsible', 'accountable', 'unaccountable',
      'self-disciplined', 'undisciplined', 'self-motivated', 'unmotivated', 'proactive', 'reactive',
      'assertive', 'passive', 'aggressive', 'submissive', 'dominant', 'submissive', 'cooperative',
      'uncooperative', 'collaborative', 'competitive', 'team player', 'independent', 'self-sufficient',
      'interdependent', 'social', 'antisocial', 'introverted', 'extroverted', 'ambiverted', 'gregarious',
      'reserved', 'outgoing', 'shy', 'confident', 'self-assured', 'self-doubting', 'insecure', 'self-critical',
      'self-reflective', 'self-aware', 'self-conscious', 'self-accepting', 'self-loving', 'self-hating',
      'self-improving', 'self-destructive', 'self-sabotaging', 'self-empowered', 'self-enlightened'
    ];

    // Extract professions
    const foundProfessions = professionTerms.filter(term =>
      description.toLowerCase().includes(term) ||
      scenario.toLowerCase().includes(term) ||
      loreEntries.some(entry => entry.content?.toLowerCase().includes(term))
    );

    // Extract settings
    const foundSettings = settingTerms.filter(term =>
      description.toLowerCase().includes(term) ||
      scenario.toLowerCase().includes(term) ||
      loreEntries.some(entry => entry.content?.toLowerCase().includes(term))
    );

    // Extract personality traits
    const foundPersonality = personalityTerms.filter(term =>
      description.toLowerCase().includes(term) ||
      modelInstructions.toLowerCase().includes(term)
    );

    // Detect gender from pronouns and terms
    const allText = `${description} ${modelInstructions} ${scenario} ${loreEntries.map(e => e.content).join(' ')}`.toLowerCase();

    const maleTerms = ['he ', 'him ', 'his ', 'himself', 'man', 'male', 'guy', 'boy', 'gentleman', 'sir', 'mister', 'mr.'];
    const femaleTerms = ['she ', 'her ', 'hers', 'herself', 'woman', 'female', 'girl', 'lady', 'madam', 'miss', 'mrs.', 'ms.'];

    const maleCount = maleTerms.reduce((count, term) => count + (allText.split(term).length - 1), 0);
    const femaleCount = femaleTerms.reduce((count, term) => count + (allText.split(term).length - 1), 0);

    let genderTerm = '';
    if (maleCount > femaleCount && maleCount > 0) {
      genderTerm = 'male';
    } else if (femaleCount > maleCount && femaleCount > 0) {
      genderTerm = 'female';
    }

    console.log(`🎯 Gender detection: male=${maleCount}, female=${femaleCount}, determined=${genderTerm || 'unknown'}`);

    // Build character descriptor
    let characterDescriptor = [];

    // Add gender if detected
    if (genderTerm) {
      characterDescriptor.push(genderTerm);
    }

    // Add profession if found
    if (foundProfessions.length > 0) {
      characterDescriptor.push(foundProfessions[0]);
    }

    // Add key personality traits
    if (foundPersonality.length > 0) {
      characterDescriptor.push(foundPersonality.slice(0, 2).join(' and '));
    }

    // Look for physical descriptions in the persona
    const physicalDescriptors = [];
    const physicalTerms = [
      'tall', 'short', 'young', 'old', 'elderly', 'middle-aged', 'mature',
      'beard', 'mustache', 'glasses', 'spectacles', 'bald', 'long hair', 'short hair',
      'blonde', 'brunette', 'red hair', 'black hair', 'white hair', 'gray hair', 'grey hair',
      'blue eyes', 'brown eyes', 'green eyes', 'hazel eyes', 'dark eyes', 'bright eyes',
      'pale', 'tan', 'dark skin', 'light skin', 'fair skin', 'olive skin',
      'thin', 'slender', 'athletic', 'muscular', 'stocky', 'heavy-set', 'chubby', 'plump', 'fit', 'toned',
      'scarred', 'tattooed', 'pierced', 'freckles', 'wrinkles', 'blemished', 'smooth skin', 'rosy cheeks', 'sun-kissed',
      'handsome', 'beautiful', 'pretty', 'ugly', 'attractive', 'unattractive', 'charming', 'dashing', 'elegant', 'graceful', 'rugged', 'dapper', 'stylish', 'fashionable',
      'disheveled', 'neat', 'well-groomed', 'unkempt', 'scruffy', 'polished', 'refined', 'rustic', 'vintage', 'retro',
      'futuristic', 'cybernetic', 'mechanical', 'steampunk', 'medieval', 'fantasy', 'mythical', 'supernatural', 'ethereal', 'otherworldly', 'alien', 'robotic', 'android',
      'cyborg', 'post-apocalyptic', 'dystopian', 'utopian', 'noir', 'gothic', 'baroque', 'art deco', 'modernist', 'minimalist', 'abstract', 'surreal', 'whimsical', 'magical', 'enchanted',
      'mythical', 'legendary', 'heroic', 'villainous', 'anti-hero', 'protagonist', 'antagonist', 'sidekick', 'mentor', 'apprentice', 'rival', 'friend', 'foe', 'ally', 'enemy'
    ];

    physicalTerms.forEach(term => {
      if (description.toLowerCase().includes(term)) {
        physicalDescriptors.push(term);
      }
    });

    // Look for clothing/style descriptions
    const clothingTerms = [
      'suit', 'formal', 'casual', 'robes', 'dress', 'uniform', 'lab coat', 'jacket',
      'shirt', 'tie', 'bow tie', 'vest', 'sweater', 'cloak', 'armor', 'leather',
      'elegant', 'professional', 'business', 'academic', 'scholarly', 'futuristic', 'cyberpunk', 'steampunk', 'medieval', 'fantasy', 'vintage', 'retro',
      'modern', 'contemporary', 'traditional', 'cultural', 'ethnic', 'bohemian', 'gothic', 'punk', 'grunge', 'hippie', 'rocker', 'biker', 'athletic',
      'sportswear', 'casual wear', 'streetwear', 'workwear', 'loungewear', 'activewear', 'formal wear', 'evening wear', 'wedding attire', 'costume', 'theatrical', 'historical', 'period costume', 'fantasy costume', 'sci-fi costume',
      'military uniform', 'naval uniform', 'air force uniform', 'police uniform', 'firefighter uniform', 'paramedic uniform', 'security uniform', 'construction worker uniform', 'chef uniform', 'doctor scrubs', 'nurse scrubs', 'scientist lab coat', 'researcher attire', 'academic gown',
      'business casual', 'smart casual', 'business formal', 'cocktail attire', 'black tie', 'white tie', 'tuxedo', 'evening gown', 'ball gown', 'prom dress', 'wedding dress', 'bridesmaid dress', 'groom suit', 'bridal suit', 'groomsmen suit',
      'party wear', 'festive attire', 'holiday outfit', 'seasonal clothing', 'summer wear', 'winter wear', 'spring wear', 'autumn wear', 'beachwear', 'swimwear', 'activewear', 'yoga pants', 'gym clothes',
      'workout gear', 'athleisure', 'loungewear', 'pajamas', 'sleepwear', 'underwear', 'lingerie', 'nightgown', 'robe', 'slippers',
      'accessories', 'jewelry', 'watch', 'hat', 'scarf', 'gloves', 'belt', 'sunglasses', 'bag', 'backpack', 'briefcase', 'purse', 'wallet', 'phone case',
      'shoes', 'boots', 'sneakers', 'heels', 'flats', 'loafers', 'sandals', 'flip flops', 'work boots', 'hiking boots', 'dress shoes', 'casual shoes',
      'athletic shoes', 'running shoes', 'cross trainers', 'basketball shoes', 'soccer cleats', 'football cleats', 'tennis shoes', 'golf shoes'
    ];

    const foundClothing = clothingTerms.filter(term =>
      description.toLowerCase().includes(term) ||
      scenario.toLowerCase().includes(term)
    );

    // Combine descriptors
    if (characterDescriptor.length > 0) {
      promptParts.push(characterDescriptor.join(', '));
    }

    if (physicalDescriptors.length > 0) {
      promptParts.push(physicalDescriptors.slice(0, 3).join(', '));
    }

    if (foundClothing.length > 0) {
      promptParts.push(`wearing ${foundClothing[0]}`);
    }

    // Add setting context
    let settingContext = '';
    if (scenario.toLowerCase().includes('academic') || scenario.toLowerCase().includes('university') || scenario.toLowerCase().includes('research')) {
      settingContext = 'academic setting, university office';
    } else if (scenario.toLowerCase().includes('laboratory') || scenario.toLowerCase().includes('lab')) {
      settingContext = 'laboratory setting, scientific environment';
    } else if (scenario.toLowerCase().includes('conference')) {
      settingContext = 'professional conference setting';
    } else if (scenario.toLowerCase().includes('medieval') || foundSettings.includes('medieval')) {
      settingContext = 'medieval fantasy setting';
    } else if (scenario.toLowerCase().includes('modern') || scenario.toLowerCase().includes('contemporary')) {
      settingContext = 'modern setting';
    } else if (foundSettings.length > 0) {
      settingContext = `${foundSettings[0]} setting`;
    }

    if (settingContext) {
      promptParts.push(settingContext);
    }

    // Determine art style based on setting
    let artStyle = 'realistic portrait';
    if (foundSettings.includes('fantasy') || foundSettings.includes('medieval')) {
      artStyle = 'fantasy character art, detailed digital painting';
    } else if (foundSettings.includes('cyberpunk') || foundSettings.includes('futuristic')) {
      artStyle = 'cyberpunk character art, digital illustration';
    } else if (foundSettings.includes('steampunk')) {
      artStyle = 'steampunk character portrait, vintage industrial';
    } else if (scenario.toLowerCase().includes('academic') || foundProfessions.includes('researcher')) {
      artStyle = 'professional headshot, academic portrait';
    }

    // Add quality modifiers
    const qualityTags = [
      artStyle,
      'high quality',
      'detailed',
      'professional artwork',
      'sharp focus',
      'good lighting'
    ];

    // Combine everything
    const finalPrompt = [...promptParts, ...qualityTags].join(', ');

    console.log('🎨 Generated detailed prompt from character data:', finalPrompt);
    return finalPrompt;
  }, []);

  // Add this function to generate character image
  const handleGenerateCharacterImage = useCallback(async (useCustomPrompt = false) => {
    if (!generatedCharacter || isGeneratingCharacterImage) return;

    setIsGeneratingCharacterImage(true);
    try {
      const prompt = useCustomPrompt && customImagePrompt.trim()
        ? customImagePrompt.trim()
        : characterImagePrompt || generateCharacterImagePrompt(generatedCharacter);

      console.log('🎨 Generating character image with prompt:', prompt);
      console.log('🎨 Using settings:', characterImageSettings);

      const response = await generateImage(prompt, {
        width: characterImageSettings.width,
        height: characterImageSettings.height,
        steps: characterImageSettings.steps,
        guidance_scale: characterImageSettings.guidance_scale,
        sampler: characterImageSettings.sampler,
        seed: characterImageSettings.seed,
        model: characterImageSettings.model,
        negative_prompt: 'blurry, low quality, distorted, malformed, ugly, bad anatomy, bad proportions, pixelated, watermark, text, logo, signature, username, artist name, copyright'
      });

      if (response && response.image_urls && response.image_urls.length > 0) {
        setCharacterImageUrl(response.image_urls[0]);
        console.log('✅ Character image generated successfully');

        // Update seed with the actual used seed if it was random
        if (response.parameters && response.parameters.seed !== undefined) {
          setCharacterImageSettings(prev => ({ ...prev, seed: response.parameters.seed }));
        }
      } else {
        throw new Error('No image URLs returned from generation');
      }

    } catch (error) {
      console.error('❌ Error generating character image:', error);
      alert(`Failed to generate character image: ${error.message}`);
    } finally {
      setIsGeneratingCharacterImage(false);
    }
  }, [generatedCharacter, characterImagePrompt, customImagePrompt, generateCharacterImagePrompt, generateImage, isGeneratingCharacterImage, characterImageSettings]);
  // TTS click handler
  // TTS click handler
  const handleSpeakerClick = useCallback((messageId, text) => {
    if (audioError) setAudioError(null);

    if (isPlayingAudio === messageId) {
      // Stop playing AND add this message to skipped list
      stopTTS();
      setSkippedMessageIds(prev => {
        const newSet = new Set(prev);
        newSet.add(messageId);
        return newSet;
      });
    } else if (!isPlayingAudio) {
      playTTS(messageId, text);
    }
  }, [audioError, isPlayingAudio, stopTTS, playTTS]);

  const handleAutoPlayToggle = (value) => {
    console.log(`🔊 [TTS] Autoplay turned ${value ? 'ON' : 'OFF'}`);
    if (audioError) setAudioError(null); // Clear any audio errors
    updateSettings({ ttsAutoPlay: value });
  };

  // Format model name
  const formatModelName = useCallback((name) => {
    if (!name) return 'None';

    // Handle API models
    if (name === 'openai-api-8000') return 'OpenAI API (8000)';
    if (name === 'openai-api-8001') return 'OpenAI API (8001)';

    // Handle regular models
    let displayName = name.split('/').pop().split('\\').pop();
    if (displayName.endsWith('.bin') || displayName.endsWith('.gguf')) {
      displayName = displayName.substring(0, displayName.lastIndexOf('.'));
    }
    return displayName;
  }, []);

  useEffect(() => {
  }, [messages]); // REMOVED messageVariants and isGenerating

  // Focus input effect



  useEffect(() => {
    if (messages && messages.length > 0 && autoAnalyzeImages) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.type === 'image' && !lastMessage.autoAnalyzed) {
        // Mark as analyzed to prevent re-analysis
        setMessages(prev => prev.map(msg =>
          msg.id === lastMessage.id ? { ...msg, autoAnalyzed: true } : msg
        ));

        // Trigger analysis after a short delay
        setTimeout(() => {
          handleAutoAnalyzeImage(lastMessage);
        }, 500);
      }
    }
  }, [messages, autoAnalyzeImages]);
  // 1) Load from localStorage whenever the conversation changes or on first mount
  useEffect(() => {
    if (!activeConversation) {
      // no conversation → clear out
      setMessageVariants({});
      setCurrentVariantIndex({});
      return;
    }

    try {
      const key = `LiangLocal-variants-${activeConversation}`;
      const stored = localStorage.getItem(key);

      if (stored) {
        const { messageVariants: loadedVariants, currentVariantIndex: loadedIndex } = JSON.parse(stored);
        setMessageVariants(loadedVariants || {});
        setCurrentVariantIndex(loadedIndex || {});
      } else {
        // first time → empty
        setMessageVariants({});
        setCurrentVariantIndex({});
      }
    } catch (err) {
      console.error('❌ Error loading variants:', err);
      setMessageVariants({});
      setCurrentVariantIndex({});
    }
  }, [activeConversation]);

  // 2) Save to localStorage any time your variants or indices change
  useEffect(() => {
    if (!activeConversation) return;

    try {
      const key = `LiangLocal-variants-${activeConversation}`;
      const payload = JSON.stringify({
        messageVariants,
        currentVariantIndex
      });
      localStorage.setItem(key, payload);
    } catch (err) {
      console.error('❌ Error saving variants:', err);
    }
  }, [activeConversation, messageVariants, currentVariantIndex]);
  // Modify the handleSubmit function to include web search and author's note
  const handleSubmit = async (text) => {
    if (text && !isGenerating) {
      const shouldUseDual = dualModeEnabled && primaryModel && secondaryModel;
      if (shouldUseDual) {
        await sendDualMessage(text, webSearchEnabled);
      } else {
        // Pass author's note directly to sendMessage (3rd parameter)
        const noteToSend = authorNote.trim() || null;
        if (noteToSend) {
          console.log("📝 [Author's Note] Sending with message:", noteToSend.substring(0, 50) + "...");
        }
        await sendMessage(text, webSearchEnabled, noteToSend);
      }
    }
  };


  const handleGenerateCharacter = useCallback(async () => {
    if (isGeneratingCharacter) {
      return;
    }

    try {
      setIsGeneratingCharacter(true);
      console.log('🎨 Starting character generation...');

      // Get API endpoint info if using external API
      const apiEndpoint = primaryIsAPI && settings.customApiEndpoints?.find(e => e.enabled);

      const response = await fetch(`${PRIMARY_API_URL}/character/generate-from-conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.slice(-30),
          analysis: characterReadiness,
          model_name: primaryModel,
          use_api: primaryIsAPI,
          api_endpoint: apiEndpoint ? {
            url: apiEndpoint.url,
            api_key: apiEndpoint.apiKey,
            model: primaryModel
          } : null
        }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success') {
          setGeneratedCharacter(result.character_json);
          setShowCharacterPreview(true);
          console.log('✅ Character generated:', result.character_json.name);
        } else {
          console.error('Character generation failed:', result.error);
          // Better error display - show retry option
          const retry = confirm(`Character generation failed: ${result.error}\n\nWould you like to try again?`);
          if (retry) {
            // Retry immediately
            setTimeout(() => handleGenerateCharacter(), 100);
            return; // Don't set isGeneratingCharacter to false yet
          }
        }
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error generating character:', error);
      const retry = confirm(`Error generating character: ${error.message}\n\nWould you like to try again?`);
      if (retry) {
        setTimeout(() => handleGenerateCharacter(), 100);
        return; // Don't set isGeneratingCharacter to false yet
      }
    } finally {
      setIsGeneratingCharacter(false); // Always reset the generating state
    }
  }, [characterReadiness, messages, isGeneratingCharacter, PRIMARY_API_URL, primaryModel]);

  const handleCallModeToggle = useCallback(async () => {
    if (isCallModeActive) {
      await stopCallMode();
    } else {
      await startCallMode();
    }
  }, [isCallModeActive, startCallMode, stopCallMode]);
  // Add this function to handle character refinement

  const handleRefineCharacter = useCallback(async () => {
    if (!generatedCharacter || !characterFeedback.trim() || isGeneratingCharacter) {
      return;
    }

    try {
      setIsGeneratingCharacter(true);
      console.log('🔄 Refining character with feedback:', characterFeedback);

      const response = await fetch(`${PRIMARY_API_URL}/character/refine-generated`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_json: generatedCharacter,
          feedback: characterFeedback,
          original_messages: messages.slice(-30),
          model_name: primaryModel  // ADD THIS LINE
        }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success') {
          setGeneratedCharacter(result.character_json);
          setCharacterFeedback(''); // Clear feedback after successful refinement
          console.log('✅ Character refined:', result.character_json.name);
        } else {
          console.error('Character refinement failed:', result.error);
          alert(`Character refinement failed: ${result.error}`);
        }
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error refining character:', error);
      alert(`Error refining character: ${error.message}`);
    } finally {
      setIsGeneratingCharacter(false);
    }
  }, [generatedCharacter, characterFeedback, messages, isGeneratingCharacter, PRIMARY_API_URL]);
  const handleAutoAnalyzeImage = useCallback(async (imageMessage) => {
    if (!imageMessage.imagePath || !primaryModel) return;

    try {
      // Convert image URL to base64
      const response = await fetch(imageMessage.imagePath);
      const blob = await response.blob();

      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
      });

      // Call vision API for auto-analysis
      const analysisResponse = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: "Analyze this image in detail. Describe what you see, including objects, people, settings, colors, mood, and any text visible in the image.",
          model_name: primaryModel,
          image_base64: base64,
          image_type: 'image/png',
          temperature: 0.7,
          max_tokens: 1024,
          userProfile: { id: userProfile?.id ?? 'anonymous' }
        })
      });

      if (analysisResponse.ok) {
        const result = await analysisResponse.json();
        const analysisMsg = {
          id: generateUniqueId(),
          role: 'bot',
          content: `🔍 **Auto-Analysis:** ${result.text || 'No analysis available'}`,
          modelId: 'primary'
        };
        setMessages(prev => [...prev, analysisMsg]);
      }

    } catch (error) {
      console.error('Auto vision analysis error:', error);
    }
  }, [primaryModel, PRIMARY_API_URL, userProfile, generateUniqueId]);
  // 3. Then add this regenerate function inside your Chat component (after other functions):
  const handleRegenerateImage = useCallback(async (imageParams) => {
    console.log('Starting image regeneration with params:', imageParams);

    // Check auto-enhance conditions
    const autoEnhanceEnabled = localStorage.getItem('adetailer-auto-enhance') === 'true';
    const adetailerSettings = JSON.parse(localStorage.getItem('adetailer-settings') || '{}');
    const selectedAdetailerModel = localStorage.getItem('adetailer-selected-model') || 'face_yolov8n.pt';

    try {
      setRegenerationQueue(prev => prev + 1);

      // Use existing generateImage function instead of direct API call
      const responseData = await generateImage(imageParams.prompt, {
        negative_prompt: imageParams.negative_prompt,
        width: imageParams.width,
        height: imageParams.height,
        steps: imageParams.steps,
        guidance_scale: imageParams.guidance_scale,
        sampler: imageParams.sampler,
        seed: -1, // Random seed for variation
        model: imageParams.model
      });

      if (responseData && Array.isArray(responseData.image_urls) && responseData.image_urls.length > 0) {
        responseData.image_urls.forEach(imageUrl => {
          // Generate message ID first
          const messageId = `${Date.now()}-${Math.random().toString(36).substr(2, 7)}-regen`;

          const imageMessage = {
            id: messageId,
            role: 'bot',
            type: 'image',
            content: imageParams.prompt,
            imagePath: imageUrl,
            prompt: imageParams.prompt,
            negative_prompt: imageParams.negative_prompt || '',
            width: responseData.parameters?.width || imageParams.width,
            height: responseData.parameters?.height || imageParams.height,
            steps: responseData.parameters?.steps || imageParams.steps,
            guidance_scale: responseData.parameters?.cfg_scale || imageParams.guidance_scale,
            model: responseData.parameters?.sd_model_checkpoint || imageParams.model,
            sampler: responseData.parameters?.sampler_name || imageParams.sampler,
            seed: responseData.parameters?.seed !== undefined ? responseData.parameters.seed : -1,
            timestamp: new Date().toISOString()
          };

          setMessages(prev => [...prev, imageMessage]);

          // NEW: Apply auto-enhancement if enabled
          if (autoEnhanceEnabled) {
            autoEnhanceRegeneratedImage(imageUrl, imageParams.prompt, messageId, adetailerSettings, selectedAdetailerModel);
          }
        });
      } else {
        setMessages(prev => [...prev, {
          id: `${Date.now()}-regen-error`,
          role: 'system',
          content: 'Image regeneration completed, but no images were returned.',
          error: true
        }]);
      }
    } catch (error) {
      console.error('Regeneration error:', error);
      setMessages(prev => [...prev, {
        id: `${Date.now()}-regen-error`,
        role: 'system',
        content: `Regeneration failed: ${error.message}`,
        error: true
      }]);
    } finally {
      setRegenerationQueue(prev => Math.max(0, prev - 1));
    }
  }, [generateImage, setMessages, setRegenerationQueue]);

  // NEW: Auto-enhancement function for regenerated images
  const autoEnhanceRegeneratedImage = useCallback(async (imageUrl, originalPrompt, messageId, settings, modelName) => {
    try {
      // Check if ADetailer is available
      const statusResponse = await fetch(`${MEMORY_API_URL}/sd-local/adetailer-status`);
      if (!statusResponse.ok) return;

      const statusData = await statusResponse.json();
      if (!statusData.available) return;

      const response = await fetch(`${MEMORY_API_URL}/sd-local/enhance-adetailer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          original_prompt: originalPrompt,
          face_prompt: settings.facePrompt || 'detailed face, high quality, sharp focus',
          strength: settings.strength || 0.35,
          confidence: settings.confidence || 0.3,
          model_name: modelName
        })
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success' && result.enhanced_image_url) {
          // Update the regenerated message with enhancement
          setTimeout(() => {
            setMessages(prev => prev.map(msg =>
              msg.id === messageId
                ? {
                  ...msg,
                  imagePath: result.enhanced_image_url,
                  enhancement_history: [imageUrl, result.enhanced_image_url],
                  current_enhancement_level: 1,
                  enhanced: true,
                  enhancement_settings: { ...settings, model_name: modelName }
                }
                : msg
            ));
          }, 1000); // Small delay like in the original auto-enhance
        }
      }
    } catch (error) {
      console.error('Auto-enhancement of regenerated image failed:', error);
    }
  }, [MEMORY_API_URL, setMessages]);
  // Add this function to save the character to your library

  const handleSaveCharacter = useCallback(() => {
    if (!generatedCharacter) return;

    try {
      // Format to match your CharacterEditor structure exactly
      const characterToSave = {
        id: null, // New character
        name: generatedCharacter.name || 'Generated Character',
        description: generatedCharacter.description || '', // This is your "persona" field
        model_instructions: generatedCharacter.model_instructions || '',
        scenario: generatedCharacter.scenario || '',
        first_message: generatedCharacter.first_message || '',
        example_dialogue: Array.isArray(generatedCharacter.example_dialogue)
          ? generatedCharacter.example_dialogue
          : [{ role: 'user', content: '' }, { role: 'character', content: '' }],
        loreEntries: Array.isArray(generatedCharacter.loreEntries)
          ? generatedCharacter.loreEntries
          : [],
        avatar: characterImageUrl || null, // FIXED: Use generated image if available
        created_at: new Date().toISOString().split('T')[0]
      };

      saveCharacter(characterToSave);

      // Close modal and reset
      setShowCharacterPreview(false);
      setGeneratedCharacter(null);
      setCharacterFeedback('');
      setCharacterImageUrl(null); // Clear the generated image state

      alert(`Character "${characterToSave.name}" saved to your library!`);
    } catch (error) {
      console.error('Error saving character:', error);
      alert(`Error saving character: ${error.message}`);
    }
  }, [generatedCharacter, saveCharacter, characterImageUrl]); // Add characterImageUrl to dependencies
  // Helper function to get button state and styling

  const getCharacterButtonState = () => {
    // Always clickable - let the backend handle validation
    return {
      disabled: false,
      variant: "outline",
      className: "flex-shrink-0 h-10 w-10 hover:bg-purple-500/20 hover:border-purple-500",
      title: "Generate character from conversation"
    };
  };



  const bothModelsLoaded = primaryModel && secondaryModel;

  // Agent conversation handler
  const handleStartAgentConversation = () => {
    if (agentTopic.trim() && bothModelsLoaded) {
      startAgentConversation(agentTopic, agentTurns);
      setAgentTopic('');
    }
  };
  const handleGenerateVariant = useCallback(async (messageId) => {
    if (isGenerating) return;

    // 1) find the original bot message and its user prompt
    const msgIndex = messages.findIndex(m => m.id === messageId);
    if (msgIndex < 0) return;
    const botMsg = messages[msgIndex];
    if (botMsg.role !== 'bot') return;

    let userIdx = -1;
    for (let i = msgIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        userIdx = i;
        break;
      }
    }
    if (userIdx < 0) return;
    const userText = messages[userIdx].content;

    // 2) seed variants with [ original ] → [ original, '' ]
    setMessageVariants(prev => {
      const base = prev[messageId]?.slice() || [botMsg.content];
      const newArr = [...base, ''];
      setCurrentVariantIndex(ci => ({
        ...ci,
        [messageId]: newArr.length - 1
      }));
      return { ...prev, [messageId]: newArr };
    });

    setIsGenerating(true);
    try {
      // 3) Use shared helper for full system prompt (includes Lore, Memories, User Profile with tag substitution)
      const systemMsg = await getGenerationSystemPrompt(
        userText,
        activeCharacter,
        authorNote // Pass author note directly
      );

      // Include story tracker context if available
      let fullSystemMsg = systemMsg;
      const storyContext = getStoryTrackerContext();
      if (storyContext) {
        fullSystemMsg += `\n\nSTORY STATE:\n${storyContext}`;
      }

      // --- DYNAMIC CONTEXT PRUNING (8k Limit) ---
      const MAX_CONTEXT_TOKENS = 7500;
      const estimateTokens = (str) => Math.ceil((str || '').length / 4);
      const systemTokens = estimateTokens(fullSystemMsg);
      let availableTokens = MAX_CONTEXT_TOKENS - systemTokens;

      const historyUpToUser = messages.slice(0, userIdx + 1);
      const reversedHistory = [...historyUpToUser].reverse();
      let slicedMessages = [];
      let currentTokens = 0;
      const MIN_CONTINUITY = 5;

      for (let i = 0; i < reversedHistory.length; i++) {
        const msg = reversedHistory[i];
        const msgTokens = estimateTokens(msg.content) + 10;
        if (i >= MIN_CONTINUITY && (currentTokens + msgTokens) > availableTokens) break;
        slicedMessages.unshift(msg);
        currentTokens += msgTokens;
      }

      const prompt = formatPrompt(slicedMessages, primaryModel, fullSystemMsg);

      // 5) streaming payload (same as before)
      const {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        frequencyPenalty = 0, // Ensure these defaults match generateReply
        presencePenalty = 0,
        use_rag,
        selectedDocuments = []
      } = settings;

      const payload = {
        prompt,
        model_name: primaryModel,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        use_rag,
        rag_docs: selectedDocuments,
        gpu_id: 0,
        userProfile: { id: userProfile?.id ?? 'anonymous' },
        authorNote: authorNote.trim() || undefined,
        directProfileInjection: settings.directProfileInjection, // Ensure this is passed if backend needs it, though frontend injection handled it
        memoryEnabled: true,
        stream: true
      };

      // Create abort controller for this generation
      const controller = new AbortController();
      setAbortController(controller);

      const res = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal // Add this line
      });
      if (!res.ok) throw new Error(`Stream error ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = '';
      let done = false;
      let sseBuffer = ''; // Buffer for incomplete SSE events

      // Start streaming TTS if enabled
      if (settings.ttsAutoPlay && settings.ttsEnabled) startStreamingTTS(messageId);
      let lastSentTtsContent = '';

      try {
        while (!done) {
          const { done: rDone, value } = await reader.read();
          if (rDone) break;
          const chunk = decoder.decode(value, { stream: true });
          sseBuffer += chunk;
          const events = sseBuffer.split('\n\n');
          sseBuffer = events.pop() || '';
          for (const line of events) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') {
              done = true;
              break;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                acc += parsed.text;
                const partial = cleanModelOutput(acc);

                // Send to TTS
                const newTextChunk = partial.slice(lastSentTtsContent.length);
                if (newTextChunk) {
                  addStreamingText(newTextChunk);
                  lastSentTtsContent = partial;
                }

                setMessageVariants(prev => {
                  const arr = prev[messageId]?.slice() || [];
                  arr[arr.length - 1] = partial;
                  return { ...prev, [messageId]: arr };
                });
              }
            } catch { }
          }
        }

        if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log('🛑 Generation stopped by user');
          // Don't throw, just finish with whatever we have
        } else {
          throw error;
        }
      } finally {
        setAbortController(null);
      }

      // 7) final cleanup
      const finalText = cleanModelOutput(acc);
      setMessageVariants(prev => {
        const arr = prev[messageId]?.slice() || [];
        arr[arr.length - 1] = finalText;
        return { ...prev, [messageId]: arr };
      });

    } catch (err) {
      console.error('Variant streaming error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [
    isGenerating,
    messages,
    settings,
    primaryModel,
    activeCharacter,
    userProfile?.id,
    authorNote,
    PRIMARY_API_URL,
    formatPrompt,
    buildSystemPrompt,
    cleanModelOutput,
    fetchMemoriesFromAgent,
    fetchTriggeredLore
  ]);
  const getCurrentVariantContent = useCallback((messageId, originalContent) => {
    const variants = messageVariants[messageId];
    if (!variants || variants.length === 0) {
      return originalContent;
    }
    const index = currentVariantIndex[messageId] || 0;
    return variants[index] || originalContent;
  }, [messageVariants, currentVariantIndex]);

  const getVariantCount = useCallback((messageId) => {
    const variants = messageVariants[messageId];
    return variants ? variants.length : 0;
  }, [messageVariants]);

  const navigateVariant = useCallback((messageId, direction) => {
    const variants = messageVariants[messageId];
    if (!variants || variants.length <= 1) return;

    const currentIndex = currentVariantIndex[messageId] || 0;
    let newIndex;

    if (direction === 'next') {
      newIndex = (currentIndex + 1) % variants.length;
    } else {
      newIndex = currentIndex === 0 ? variants.length - 1 : currentIndex - 1;
    }

    setCurrentVariantIndex(prev => ({
      ...prev,
      [messageId]: newIndex
    }));
  }, [messageVariants, currentVariantIndex]);

  // Add this useEffect to fetch available SD models:
  useEffect(() => {
    const fetchSDModels = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/sd/status`);
        if (response.ok) {
          const data = await response.json();
          if (data.automatic1111 && data.models) {
            setAvailableModels(data.models);
            if (data.models.length > 0 && !characterImageSettings.model) {
              const firstModel = data.models[0];
              const modelName = firstModel.model_name || firstModel.title || firstModel.name;
              setCharacterImageSettings(prev => ({ ...prev, model: modelName }));
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch SD models:', error);
      }
    };

    if (showCharacterPreview) {
      fetchSDModels();
    }
  }, [showCharacterPreview, SECONDARY_API_URL]);
  // Add this useEffect to auto-generate prompt when character changes
  useEffect(() => {
    if (generatedCharacter && showCharacterPreview) {
      const autoPrompt = generateCharacterImagePrompt(generatedCharacter);
      setCharacterImagePrompt(autoPrompt);
      setCustomImagePrompt(autoPrompt);
      setCharacterImageUrl(null); // Clear previous image
    }
  }, [generatedCharacter, showCharacterPreview, generateCharacterImagePrompt]);
  useEffect(() => {
    // This ref will hold our AudioContext and the currently playing source


    const playNextInQueue = async () => {
      if (!audioQueue) return; // <-- ADD THIS LINE
      if (!isAutoplaying || audioQueue.length === 0) {
        if (audioQueue.length === 0 && isAutoplaying) {
          // If the queue is empty but we thought we were playing, stop.
          setIsAutoplaying(false);
        }
        return;
      }

      // 1. Get the next audio chunk (ArrayBuffer) from the queue
      const audioBuffer = audioQueue[0];

      try {
        // 2. Create a new AudioContext for this chunk
        const context = new (window.AudioContext || window.webkitAudioContext)();
        audioPlaybackRef.current.context = context;

        // 3. Decode the audio data
        const decodedBuffer = await context.decodeAudioData(audioBuffer);
        const source = context.createBufferSource();
        source.buffer = decodedBuffer;

        // 4. Apply settings
        source.playbackRate.value = settings.ttsSpeed || 1.0;

        // 5. Connect to speakers and store the source so we can stop it
        source.connect(context.destination);
        audioPlaybackRef.current.source = source;

        // 6. Define what happens when this chunk is finished
        source.onended = () => {
          // Clean up the context and source
          context.close();
          audioPlaybackRef.current = { context: null, source: null };
          // IMPORTANT: Remove the chunk we just played from the queue's state.
          // This will trigger the useEffect to run again and play the next chunk.
          setAudioQueue(prevQueue => prevQueue.slice(1));
        };

        // 7. Play the sound
        source.start(0);

      } catch (error) {
        console.error("❌ Error decoding or playing audio data:", error);
        // If a chunk is corrupt or fails, skip to the next one
        setAudioQueue(prevQueue => prevQueue.slice(1));
      }
    };

    playNextInQueue();

    // This is the cleanup function for the useEffect hook
    return () => {
      // If the component unmounts or this effect re-runs, stop any playing audio
      if (audioPlaybackRef.current.source) {
        try {
          audioPlaybackRef.current.source.stop();
        } catch (e) { /* Ignore errors if already stopped */ }
      }
      if (audioPlaybackRef.current.context) {
        try {
          audioPlaybackRef.current.context.close();
        } catch (e) { /* Ignore errors if already closed */ }
      }
    };
  }, [audioQueue, isAutoplaying, settings.ttsSpeed, setAudioQueue, setIsAutoplaying]);

  const handleEditBotMessage = useCallback((messageId) => {
    // Get the current variant content to edit
    const currentContent = getCurrentVariantContent(messageId,
      messages.find(m => m.id === messageId)?.content || ''
    );
    setEditingBotMessageId(messageId);
    setEditingBotMessageContent(currentContent);
  }, [messages, getCurrentVariantContent]);

  const handleSaveBotMessage = useCallback((messageId, newContent) => {
    if (!newContent.trim()) return;

    // Update the current variant with the edited content
    setMessageVariants(prev => {
      const variants = prev[messageId] || [];
      const currentIndex = currentVariantIndex[messageId] || 0;

      if (variants.length === 0) {
        // No variants yet, create first one
        return { ...prev, [messageId]: [newContent.trim()] };
      } else {
        // Update current variant
        const updatedVariants = [...variants];
        updatedVariants[currentIndex] = newContent.trim();
        return { ...prev, [messageId]: updatedVariants };
      }
    });

    // Clear editing state
    setEditingBotMessageId(null);
    setEditingBotMessageContent('');
  }, [currentVariantIndex]);

  const handleCancelBotEdit = useCallback(() => {
    setEditingBotMessageId(null);
    setEditingBotMessageContent('');
  }, []);
  const handleContinueGeneration = useCallback(async (messageId) => {
    if (isGenerating) return;

    // Find the bot message and its preceding user message
    const botMsgIndex = messages.findIndex(m => m.id === messageId);
    if (botMsgIndex < 0) return;

    const botMsg = messages[botMsgIndex];
    if (botMsg.role !== 'bot') return;

    // Find the user message that prompted this bot response
    let userIdx = -1;
    for (let i = botMsgIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        userIdx = i;
        break;
      }
    }
    if (userIdx < 0) return;
    const userText = messages[userIdx].content;

    // Get the CURRENTLY DISPLAYED variant content to continue from
    const currentContent = getCurrentVariantContent(messageId, botMsg.content);

    // Get current variant index - we'll update THIS variant, not create a new one
    const currentIndex = currentVariantIndex[messageId] || 0;

    // If there are no variants yet, create the first one with current content
    if (!messageVariants[messageId] || messageVariants[messageId].length === 0) {
      setMessageVariants(prev => ({
        ...prev,
        [messageId]: [currentContent]
      }));
    }

    setIsGenerating(true);
    try {
      // Get memory and lore context (same as generation)
      const agentMem = await fetchMemoriesFromAgent(userText);
      const lore = await fetchTriggeredLore(userText, activeCharacter);
      let memoryContext = '';

      if (agentMem.length) {
        memoryContext = agentMem.map((m, i) => `[${i + 1}] ${m.content}`).join('\n');
      }

      if (lore.length) {
        const loreContext = lore.map(l => {
          if (typeof l === 'string') {
            return `• ${l}`;
          } else if (typeof l === 'object' && l.content) {
            return `• ${l.content}`;
          } else {
            return `• ${JSON.stringify(l)}`;
          }
        }).join('\n');

        if (memoryContext) {
          memoryContext += `\n\n[WORLD KNOWLEDGE - Essential lore guidance for this response]\n${loreContext}`;
        } else {
          memoryContext = `[WORLD KNOWLEDGE - Essential lore guidance for this response]\n${loreContext}`;
        }
      }

      // Build context with current response included
      const historyUpToUser = messages.slice(0, userIdx + 1);
      const recent = historyUpToUser.slice(-5);
      let systemMsg = activeCharacter
        ? buildSystemPrompt(activeCharacter)
        : 'You are a helpful assistant...';

      if (memoryContext) {
        systemMsg += `\n\nUSER CONTEXT:\n${memoryContext}`;
      }

      // Include story tracker context if available
      const storyContext = getStoryTrackerContext();
      if (storyContext) {
        systemMsg += `\n\nSTORY STATE:\n${storyContext}`;
      }

      // Add the current response to context and ask to continue
      const contextWithResponse = [
        ...recent,
        { role: 'assistant', content: currentContent }
      ];

      const continuePrompt = formatPrompt(contextWithResponse, primaryModel, systemMsg) +
        '\n\nContinue your response from where you left off. Do not repeat what you already said, just continue naturally:';

      const {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        use_rag,
        selectedDocuments = []
      } = settings;

      const payload = {
        prompt: continuePrompt,
        model_name: primaryModel,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        use_rag,
        rag_docs: selectedDocuments,
        gpu_id: 0,
        userProfile: { id: userProfile?.id ?? 'anonymous' },
        authorNote: authorNote.trim() || undefined,
        memoryEnabled: true,
        stream: true
      };

      // Create abort controller
      const controller = new AbortController();
      setAbortController(controller);

      const res = await fetch(`${PRIMARY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      if (!res.ok) throw new Error(`Stream error ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let continuationText = '';
      let done = false;
      let sseBuffer = ''; // Buffer for incomplete SSE events

      let lastSentTtsContent = '';
      if (settings.ttsAutoPlay && settings.ttsEnabled) startStreamingTTS(messageId);

      try {
        while (!done) {
          const { done: rDone, value } = await reader.read();
          if (rDone) break;
          const chunk = decoder.decode(value, { stream: true });
          sseBuffer += chunk;
          const events = sseBuffer.split('\n\n');
          sseBuffer = events.pop() || '';
          for (const line of events) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') {
              done = true;
              break;
            }
            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                continuationText += parsed.text;
                const partial = cleanModelOutput(continuationText);

                // Send to TTS
                const newTextChunk = partial.slice(lastSentTtsContent.length);
                if (newTextChunk) {
                  addStreamingText(newTextChunk);
                  lastSentTtsContent = partial;
                }

                const fullContent = currentContent + ' ' + partial;

                // Update the CURRENT variant in place (don't create new variant)
                setMessageVariants(prev => {
                  const arr = prev[messageId]?.slice() || [currentContent];
                  arr[currentIndex] = fullContent; // Update the variant we're currently viewing
                  return { ...prev, [messageId]: arr };
                });
              }
            } catch { }
          }
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          console.log('🛑 Continue generation stopped by user');
        } else {
          throw error;
        }
      } finally {
        setAbortController(null);
      }

      if (settings.ttsAutoPlay && settings.ttsEnabled) endStreamingTTS();

      // Final cleanup - update the current variant with final content
      const finalContinuation = cleanModelOutput(continuationText);
      const finalFullContent = currentContent + ' ' + finalContinuation;

      setMessageVariants(prev => {
        const arr = prev[messageId]?.slice() || [currentContent];
        arr[currentIndex] = finalFullContent; // Update current variant, not create new one
        return { ...prev, [messageId]: arr };
      });

    } catch (err) {
      console.error('Continue generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [
    isGenerating,
    messages,
    messageVariants,
    currentVariantIndex,
    setMessageVariants,
    setAbortController,
    setIsGenerating,
    getCurrentVariantContent,
    fetchMemoriesFromAgent,
    fetchTriggeredLore,
    activeCharacter,
    buildSystemPrompt,
    formatPrompt,
    cleanModelOutput,
    settings,
    primaryModel,
    userProfile,
    authorNote,
    PRIMARY_API_URL
  ]);
  // Avatar rendering function
  const sizeStyle = { width: `${characterAvatarSize}px`, height: `${characterAvatarSize}px` };
  const renderAvatar = (message, apiUrl, activeCharacter) => {
    const avatarSource = message.avatar || (message.role === 'bot' && activeCharacter?.avatar);
    const characterName = message.characterName
      || (message.role === 'bot' && activeCharacter?.name)
      || 'activeCharacter';

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

  // User avatar rendering helper
  const renderUserAvatar = () => {
    const userAvatarSource = userProfile?.avatar;
    const userName = userProfile?.name || 'User';
    let userDisplayUrl = null;

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
          className="w-10 h-10 rounded-full object-cover border border-gray-300 dark:border-gray-600 flex-shrink-0"
        />
      );
    } else {
      return (
        <div
          title={userName}
          className="w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold flex-shrink-0 border border-primary/50"
        >
          {userName ? userName.charAt(0).toUpperCase() : 'U'}
        </div>
      );
    }
  };

  // --- Component Render ---
  return (
    <div
      className="flex flex-col h-full bg-background text-foreground transition-all duration-500"
      style={{
        backgroundImage: backgroundImage ? `url("${backgroundImage}")` : undefined,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundBlendMode: 'overlay',
        backgroundColor: backgroundImage ? 'rgba(0,0,0,0.85)' : undefined // Dark overlay
      }}
    >
      {/* Header Area */}
      <div className="border-b border-border p-3 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold">Chat</h2>
            <CharacterSelector layoutMode={layoutMode} />
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost" size="sm"
              onClick={() => setShowModelSelector(!showModelSelector)}
            >
              {showModelSelector ? "Hide Models" : "Models"}
            </Button>

            <RAGIndicator className="ml-2" />

            {bothModelsLoaded && (
              <Button
                variant={dualModeEnabled ? "secondary" : "outline"} size="sm"
                onClick={() => setDualModeEnabled(!dualModeEnabled)}
                title="Toggle dual-model mode"
              >
                <Layers size={16} />
                <span className="ml-1">{dualModeEnabled ? "Dual Mode" : "Single Mode"}</span>
              </Button>
            )}

            {/* Toggle for floating controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowFloatingControls(!showFloatingControls)}
              title={showFloatingControls ? "Hide Floating Controls" : "Show Floating Controls"}
            >
              {showFloatingControls ? "Hide Controls" : "Show Controls"}
            </Button>

            {/* New Chat Button */}
            <Button variant="ghost" size="sm" onClick={createNewConversation}>
              New Chat
            </Button>
          </div>
        </div>

        {/* Model selector */}
        {showModelSelector && <ModelSelector />}

        {/* Display Audio Error */}
        {audioError && (
          <div className="text-red-500 text-sm mt-2 p-2 bg-red-100 dark:bg-red-900/30 rounded border border-red-500/50">
            Error: {audioError}
            <Button variant="ghost" size="sm" onClick={() => setAudioError(null)} className="ml-2 text-red-500">Dismiss</Button>
          </div>
        )}

        {/* Current model info with API indicators */}
        <div className="flex flex-wrap gap-2 text-sm">
          <div className={`px-2 py-1 rounded flex items-center gap-1 ${primaryModel ? 'bg-blue-100 dark:bg-blue-900/30' : 'bg-muted'}`}>
            {primaryIsAPI ? <Globe className="w-3 h-3 text-blue-500" /> : <Cpu className="w-3 h-3 text-green-500" />}
            <span className="font-medium">Primary:</span>
            <span>{formatModelName(primaryModel)}</span>
            {primaryIsAPI && <span className="text-xs text-blue-600 dark:text-blue-400">(API)</span>}
          </div>
          <div className={`px-2 py-1 rounded flex items-center gap-1 ${secondaryModel ? 'bg-purple-100 dark:bg-purple-900/30' : 'bg-muted'}`}>
            {secondaryIsAPI ? <Globe className="w-3 h-3 text-blue-500" /> : <Cpu className="w-3 h-3 text-green-500" />}
            <span className="font-medium">Secondary:</span>
            <span>{formatModelName(secondaryModel)}</span>
            {secondaryIsAPI && <span className="text-xs text-blue-600 dark:text-blue-400">(API)</span>}
          </div>
        </div>

        {/* Agent conversation controls */}
        {bothModelsLoaded && (
          <div className="flex items-center gap-2 mt-2">
            <div className="flex-1">
              <Input
                value={agentTopic} onChange={(e) => setAgentTopic(e.target.value)}
                placeholder="Enter topic for models to discuss..."
                disabled={agentConversationActive || isGenerating}
                className="bg-background border-input"
              />
            </div>
            <div className="w-16">
              <Input
                type="number" min="1" max="10" value={agentTurns}
                onChange={(e) => setAgentTurns(parseInt(e.target.value) || 3)}
                disabled={agentConversationActive || isGenerating}
                title="Number of conversation turns"
                className="bg-background border-input"
              />
            </div>
            <Button
              onClick={handleStartAgentConversation}
              disabled={!agentTopic.trim() || agentConversationActive || isGenerating}
              size="sm"
            >
              <Users size={16} /><span className="ml-1">Start</span>
            </Button>
          </div>
        )}
      </div>

      {/* Message Display Area with Floating Controls */}
      <div className="relative flex-1">
        <ScrollArea className={`h-full p-4 ${backgroundImage ? 'bg-transparent' : 'bg-background'}`}>
          <div className="max-w-4xl mx-auto p-4">
            {/* Floating controls - positioned fixed relative to viewport */}
            {showFloatingControls && (
              <div className="fixed right-6 top-1/2 transform -translate-y-1/2 z-50 flex flex-col gap-2 bg-background/80 backdrop-blur-sm p-2 rounded-md border border-border shadow-md">
                {/* Models Button */}
                <Button
                  variant={showModelSelector ? "secondary" : "outline"}
                  size="icon"
                  onClick={() => setShowModelSelector(!showModelSelector)}
                  title={showModelSelector ? "Hide Models" : "Show Models"}
                  className="flex-shrink-0 h-10 w-10"
                >
                  <Cpu size={18} />
                </Button>

                {/* Visualize Scene Button */}
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleVisualizeScene}
                  disabled={isGenerating || messages.length === 0}
                  title="Visualize Scene"
                  className="flex-shrink-0 h-10 w-10"
                >
                  <Eye size={18} />
                </Button>

                {/* Microphone Button */}
                {sttEnabled && (
                  <Button
                    variant={isRecording ? "destructive" : "outline"}
                    size="icon"
                    className="flex-shrink-0 h-10 w-10"
                    onClick={handleMicClick}
                    disabled={isTranscribing}
                    title={
                      isRecording
                        ? "Stop Recording"
                        : isTranscribing
                          ? "Processing..."
                          : "Start Recording"
                    }
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
                {isGenerating && (
                  <Button
                    variant="destructive"
                    size="icon"
                    className="flex-shrink-0 h-10 w-10"
                    onClick={handleStopGeneration}
                    title="Stop Generation"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <rect x="6" y="6" width="12" height="12" />
                    </svg>
                  </Button>
                )}

                {/* Speaker Button for Last Message */}
                {messages.length > 0 &&
                  messages[messages.length - 1].role !== "user" &&
                  ttsEnabled && (
                    <Button
                      variant={
                        isPlayingAudio === messages[messages.length - 1].id
                          ? "destructive"
                          : "outline"
                      }
                      size="icon"
                      className="flex-shrink-0 h-10 w-10"
                      onClick={() =>
                        handleSpeakerClick(
                          messages[messages.length - 1].id,
                          messages[messages.length - 1].content
                        )
                      }
                      disabled={
                        isGenerating ||
                        isTranscribing ||
                        (isPlayingAudio &&
                          isPlayingAudio !== messages[messages.length - 1].id)
                      }
                      title={
                        isPlayingAudio === messages[messages.length - 1].id
                          ? "Stop TTS"
                          : "Play Last Message"
                      }
                    >
                      {isPlayingAudio === messages[messages.length - 1].id ? (
                        <Loader2 className="animate-spin" size={18} />
                      ) : (
                        <PlayIcon size={18} />
                      )}
                    </Button>
                  )}

                {/* NEW Dedicated Stop TTS Button - Always visible when ANY audio is playing */}
                {ttsEnabled && isPlayingAudio && (
                  <Button
                    variant="destructive"
                    size="icon"
                    className="flex-shrink-0 h-10 w-10"
                    onClick={() => {
                      if (isPlayingAudio) {
                        // Add currently playing message to skipped list before stopping
                        setSkippedMessageIds(prev => {
                          const newSet = new Set(prev);
                          newSet.add(isPlayingAudio);
                          return newSet;
                        });
                      }
                      stopTTS();
                    }}
                    title="Stop All Audio Playback"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <rect x="6" y="6" width="12" height="12" />
                    </svg>
                  </Button>
                )}

                {/* Auto-Play Toggle */}
                {ttsEnabled && (
                  <div className="flex flex-col items-center gap-1 bg-card/90 p-1 rounded">
                    <Switch
                      id="floating-autoplay"
                      checked={settings?.ttsAutoPlay || false}
                      onCheckedChange={handleAutoPlayToggle}
                    />
                    <Label htmlFor="floating-autoplay" className="text-xs">
                      Auto
                    </Label>
                    {/* Auto-Analyze Images Toggle */}
                    <div className="flex flex-col items-center gap-1 bg-card/90 p-1 rounded">
                      <Switch
                        id="floating-auto-analyze"
                        checked={autoAnalyzeImages}
                        onCheckedChange={setAutoAnalyzeImages}
                      />
                      <Label htmlFor="floating-auto-analyze" className="text-xs">
                        Auto
                      </Label>
                      <Label htmlFor="floating-auto-analyze" className="text-[10px] text-muted-foreground">
                        Analyze
                      </Label>
                    </div>
                  </div>

                )}
                {/* Focus Mode Button */}
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setIsFocusModeActive(true)}
                  title="Enter Focus Mode"
                  className="flex-shrink-0 h-10 w-10"
                >
                  <Focus size={18} />
                </Button>

                {/* Character Auto-Generation Button */}
                {(() => {
                  const buttonState = getCharacterButtonState();
                  return (
                    <Button
                      variant={buttonState.variant}
                      size="icon"
                      className={buttonState.className}
                      onClick={handleGenerateCharacter}
                      disabled={buttonState.disabled || isGeneratingCharacter || isAnalyzingCharacter}
                      title={buttonState.title}
                    >
                      {isGeneratingCharacter ? (
                        <Loader2 className="animate-spin" size={18} />
                      ) : isAnalyzingCharacter ? (
                        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                          <circle cx="12" cy="7" r="4" />
                          <path d="M12 14l2 2 4-4" />
                        </svg>
                      )}
                    </Button>
                  );
                })()}
                {/* Author's Note Button */}
                <Button
                  variant={showAuthorNote ? "secondary" : "outline"}
                  size="icon"
                  onClick={() => setShowAuthorNote(!showAuthorNote)}
                  title="Toggle Author's Note"
                  className="flex-shrink-0 h-10 w-10"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14,2 14,8 20,8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10,9 9,9 8,9" />
                  </svg>
                </Button>

                {/* Story Tracker Button */}
                <Button
                  variant={showStoryTracker ? "secondary" : "outline"}
                  size="icon"
                  onClick={() => setShowStoryTracker(!showStoryTracker)}
                  title="Story Tracker - Track characters, inventory, locations"
                  className="flex-shrink-0 h-10 w-10"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
                    <line x1="4" y1="22" x2="4" y2="15" />
                  </svg>
                </Button>

                {/* Choice Generator Button */}
                <Button
                  variant={showChoiceGenerator ? "secondary" : "outline"}
                  size="icon"
                  onClick={() => setShowChoiceGenerator(!showChoiceGenerator)}
                  title="Generate action choices"
                  className="flex-shrink-0 h-10 w-10"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="2" width="8" height="8" rx="1" />
                    <rect x="14" y="2" width="8" height="8" rx="1" />
                    <rect x="2" y="14" width="8" height="8" rx="1" />
                    <rect x="14" y="14" width="8" height="8" rx="1" />
                  </svg>
                </Button>

                {/* Author's Note indicator */}
                {authorNote && (
                  <div className="text-xs text-center bg-orange-100 dark:bg-orange-900/30 p-1 rounded max-w-[60px] border border-orange-300">
                    <div className="font-medium text-orange-700 dark:text-orange-300">Note</div>
                    <div className="text-[10px] text-orange-600 dark:text-orange-400">{getAuthorNoteTokenCount()}/150</div>
                  </div>
                )}

                {sttEnabled && ttsEnabled && (
                  <Button
                    variant={isCallModeActive ? "destructive" : "outline"}
                    size="icon"
                    onClick={handleCallModeToggle}
                    title={isCallModeActive ? "Exit Call Mode" : "Enter Call Mode"}
                    className="h-9 w-9"
                  >
                    {isCallModeActive ? (
                      <PhoneOff size={18} />
                    ) : (
                      <Phone size={18} />
                    )}
                  </Button>
                )}
                {/* New Chat Button */}
                <Button
                  variant="outline"
                  size="icon"
                  className="flex-shrink-0 h-10 w-10"
                  onClick={createNewConversation}
                  title="New Chat"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 5v14M5 12h14" />
                  </svg>
                </Button>
              </div>
            )}

            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <h3 className="text-lg font-medium mb-2">No messages yet</h3>
                <p className="max-w-md mb-4">
                  {!primaryModel ? "Load a model to start chatting" : "Send a message or use the microphone"}
                </p>
                {!primaryModel && (
                  <Button variant="outline" onClick={() => setShowModelSelector(true)}>Load Model</Button>
                )}
              </div>
            ) : (
              messages.map((msg) => {
                const content = getCurrentVariantContent(msg.id, msg.content);
                const variantCount = getVariantCount(msg.id);
                const variantIndex = currentVariantIndex[msg.id] || 0;

                return (
                  <ChatMessage
                    key={msg.id}
                    msg={msg}
                    content={content}
                    isGenerating={isGenerating}
                    isTranscribing={isTranscribing}
                    isPlayingAudio={isPlayingAudio}
                    editingMessageId={editingMessageId}
                    editingMessageContent={editingMessageContent}
                    editingBotMessageId={editingBotMessageId}
                    editingBotMessageContent={editingBotMessageContent}
                    primaryCharacter={primaryCharacter}
                    secondaryCharacter={secondaryCharacter}
                    userProfile={userProfile}
                    characterAvatarSize={characterAvatarSize}
                    userAvatarSize={userAvatarSize}
                    variantCount={variantCount}
                    variantIndex={variantIndex}
                    PRIMARY_API_URL={PRIMARY_API_URL}
                    regenerationQueue={regenerationQueue}
                    ttsEnabled={ttsEnabled}

                    onEditUserMessage={handleEditUserMessage}
                    onCancelEdit={handleCancelEdit}
                    onChangeEditingMessageContent={setEditingMessageContent}
                    onSaveEditedMessage={handleSaveEditedMessage}
                    onRegenerateFromEditedPrompt={handleRegenerateFromEditedPrompt}
                    onDeleteMessage={handleDeleteMessage}

                    onEditBotMessage={handleEditBotMessage}
                    onCancelBotEdit={handleCancelBotEdit}
                    onChangeEditingBotMessageContent={setEditingBotMessageContent}
                    onSaveBotMessage={handleSaveBotMessage}
                    onGenerateVariant={handleGenerateVariant}
                    onContinueGeneration={handleContinueGeneration}
                    onNavigateVariant={navigateVariant}
                    onSpeakerClick={handleSpeakerClick}
                    onRegenerateImage={handleRegenerateImage}

                    formatModelName={formatModelName}
                  />
                );
              })
            )}
          </div>
        </ScrollArea>
      </div>
      {/* Author's Note Panel */}
      {showAuthorNote && (
        <div className="border-t border-border bg-orange-50 dark:bg-orange-950/20">
          <div className="max-w-4xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-orange-600">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14,2 14,8 20,8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10,9 9,9 8,9" />
                </svg>
                <Label className="font-medium text-orange-700 dark:text-orange-300">Author's Note</Label>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs ${getAuthorNoteTokenCount() > 140 ? 'text-red-500' : 'text-orange-600 dark:text-orange-400'}`}>
                  {getAuthorNoteTokenCount()}/150 tokens
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowAuthorNote(false)}
                  className="h-6 w-6 p-0"
                >
                  <X size={14} />
                </Button>
              </div>
            </div>
            <Textarea
              value={authorNote}
              onChange={(e) => handleAuthorNoteChange(e.target.value)}
              placeholder="Add custom instructions for this session (e.g., 'Focus on emotional responses' or 'Use a more casual tone')"
              className="w-full resize-none bg-background border-orange-200 dark:border-orange-800 text-sm"
              rows={3}
            />
            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-orange-600 dark:text-orange-400">
                {authorNote ? (
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    Active - included with every message ({getAuthorNoteTokenCount()}/150 tokens)
                  </span>
                ) : (
                  "This note will be included with all messages"
                )}
              </div>
              {authorNote && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAuthorNote}
                  className="h-6 text-xs text-red-500 hover:text-red-600 hover:bg-red-50"
                >
                  Clear Note
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Add this right before your existing <form className="border-t border-border p-4..."> */}
      <div className="border-t border-border bg-muted/5">
        <div className="max-w-4xl mx-auto px-4 py-2 flex items-center justify-between">
          <WebSearchControl
            webSearchEnabled={webSearchEnabled}
            setWebSearchEnabled={setWebSearchEnabled}
            isGenerating={isGenerating}
            isRecording={isRecording}
            isTranscribing={isTranscribing}
          />
          {webSearchEnabled && (
            <div className="text-xs text-muted-foreground">
              Search enabled for next message
            </div>
          )}
        </div>
      </div>
      <div className="border-t border-border">
        <div className="max-w-4xl mx-auto">
          {/* Input Form with Image Button */}
          <ChatInputForm
            onSubmit={handleSubmit}
            isGenerating={isGenerating}
            isModelLoading={isModelLoading}
            isRecording={isRecording}
            isTranscribing={isTranscribing}
            agentConversationActive={agentConversationActive}
            primaryModel={primaryModel}
            webSearchEnabled={webSearchEnabled}
            inputValue={inputValue}
            setInputValue={setInputValue}
            onBack={handleBack}
            canGoBack={messages.length > 0 && messages.some(m => m.role === 'user')}
          />
        </div>
      </div>
      {/* Character Preview Modal */}
      {showCharacterPreview && generatedCharacter && (
        <Dialog open={showCharacterPreview} onOpenChange={setShowCharacterPreview}>
          {/*
        FIX: The previous Tailwind classes were likely overridden by the component's
        default styles. Applying the width directly via an inline `style` attribute
        ensures our settings take precedence.
      */}
          <DialogContent
            className="max-h-[90vh] overflow-y-auto fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
            style={{ width: '90vw', maxWidth: '1600px' }}
          >
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                  <circle cx="12" cy="7" r="4" />
                </svg>
                Generated Character: {generatedCharacter.name}
              </DialogTitle>
            </DialogHeader>

            <div className="space-y-6">
              {/* Character Overview */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold mb-2">Name</h3>
                  <p className="text-sm bg-muted p-2 rounded">{generatedCharacter.name}</p>
                </div>

                {generatedCharacter.avatar && (
                  <div>
                    <h3 className="font-semibold mb-2">Avatar</h3>
                    <img
                      src={generatedCharacter.avatar}
                      alt={generatedCharacter.name}
                      className="w-16 h-16 rounded-full object-cover"
                    />
                  </div>
                )}
              </div>

              {/* Character Description */}
              <div>
                <h3 className="font-semibold mb-2">Character Persona</h3>
                <p className="text-sm bg-muted p-3 rounded whitespace-pre-wrap">
                  {generatedCharacter.description}
                </p>
              </div>

              {/* Model Instructions */}
              {generatedCharacter.model_instructions && (
                <div>
                  <h3 className="font-semibold mb-2">Model Instructions</h3>
                  <p className="text-sm bg-muted p-3 rounded whitespace-pre-wrap">
                    {generatedCharacter.model_instructions}
                  </p>
                </div>
              )}

              {/* Scenario */}
              {generatedCharacter.scenario && (
                <div>
                  <h3 className="font-semibold mb-2">Scenario</h3>
                  <p className="text-sm bg-muted p-3 rounded whitespace-pre-wrap">
                    {generatedCharacter.scenario}
                  </p>
                </div>
              )}

              {/* First Message */}
              {generatedCharacter.first_message && (
                <div>
                  <h3 className="font-semibold mb-2">Greeting Message</h3>
                  <p className="text-sm bg-muted p-3 rounded whitespace-pre-wrap">
                    {generatedCharacter.first_message}
                  </p>
                </div>
              )}

              {/* Example Dialogue */}
              {generatedCharacter.example_dialogue && generatedCharacter.example_dialogue.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Example Dialogue</h3>
                  <div className="bg-muted p-3 rounded space-y-2">
                    {generatedCharacter.example_dialogue.map((turn, index) => (
                      <div key={index} className="text-sm">
                        <strong>{turn.role === 'user' ? 'User' : generatedCharacter.name}:</strong> {turn.content}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Lore Entries */}
              {generatedCharacter.loreEntries && generatedCharacter.loreEntries.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Lore Entries</h3>
                  <div className="space-y-2">
                    {generatedCharacter.loreEntries.map((entry, index) => (
                      <div key={index} className="bg-muted p-3 rounded">
                        <p className="text-sm mb-1">{entry.content}</p>
                        <div className="text-xs text-muted-foreground">
                          Keywords: {Array.isArray(entry.keywords) ? entry.keywords.join(', ') : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {/* Character Image Generation Section */}
              <div className="border-t pt-4">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="9" cy="9" r="2" />
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                  </svg>
                  Character Portrait
                </h3>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Left Column - Image Display */}
                  <div className="space-y-3">
                    <div className="aspect-square bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                      {characterImageUrl ? (
                        <img
                          src={characterImageUrl}
                          alt={`${generatedCharacter.name} portrait`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="text-center text-muted-foreground p-4">
                          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <circle cx="9" cy="9" r="2" />
                            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                          </svg>
                          <p className="text-sm">No image generated yet</p>
                          <p className="text-xs text-muted-foreground">Configure settings and click generate</p>
                        </div>
                      )}
                    </div>

                    {/* Main Generation Controls */}
                    <div className="flex gap-2">
                      <Button
                        onClick={() => handleGenerateCharacterImage(false)}
                        disabled={isGeneratingCharacterImage}
                        variant="default"
                        size="sm"
                        className="flex-1"
                      >
                        {isGeneratingCharacterImage ? (
                          <>
                            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                              <path d="M12 2v10l3-3m-6 0l3 3" />
                              <path d="M12 22v-10" />
                              <path d="M4 14h16" />
                            </svg>
                            {characterImageUrl ? 'Regenerate' : 'Generate Portrait'}
                          </>
                        )}
                      </Button>

                      <Button
                        onClick={() => setCharacterImageSettings(prev => ({ ...prev, seed: -1 }))}
                        disabled={isGeneratingCharacterImage}
                        variant="outline"
                        size="sm"
                        title="Randomize seed"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                          <path d="M3 3v5h5" />
                          <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                          <path d="M21 21v-5h-5" />
                        </svg>
                      </Button>
                    </div>
                  </div>

                  {/* Right Column - Controls */}
                  <div className="space-y-4">
                    {/* Model Selection */}
                    {availableModels.length > 0 && (
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Model</Label>
                        <select
                          value={characterImageSettings.model || ''}
                          onChange={e => setCharacterImageSettings(prev => ({ ...prev, model: e.target.value }))}
                          disabled={isGeneratingCharacterImage}
                          className="w-full rounded border border-input bg-background text-foreground px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                        >
                          {availableModels.map((m, i) => {
                            const name = m.model_name || m.title || m.name;
                            return <option key={i} value={name} className="bg-background text-foreground">{name}</option>;
                          })}
                        </select>
                      </div>
                    )}

                    {/* Image Dimensions */}
                    <div className="space-y-3">
                      <Label className="text-sm font-medium">Dimensions</Label>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant={characterImageSettings.width === 512 && characterImageSettings.height === 512 ? "secondary" : "outline"}
                          onClick={() => setCharacterImageSettings(prev => ({ ...prev, width: 512, height: 512 }))}
                          disabled={isGeneratingCharacterImage}
                        >
                          1:1
                        </Button>
                        <Button
                          size="sm"
                          variant={characterImageSettings.width === 768 && characterImageSettings.height === 512 ? "secondary" : "outline"}
                          onClick={() => setCharacterImageSettings(prev => ({ ...prev, width: 768, height: 512 }))}
                          disabled={isGeneratingCharacterImage}
                        >
                          3:2
                        </Button>
                        <Button
                          size="sm"
                          variant={characterImageSettings.width === 512 && characterImageSettings.height === 768 ? "secondary" : "outline"}
                          onClick={() => setCharacterImageSettings(prev => ({ ...prev, width: 512, height: 768 }))}
                          disabled={isGeneratingCharacterImage}
                        >
                          2:3
                        </Button>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {characterImageSettings.width} × {characterImageSettings.height}
                      </div>
                    </div>

                    {/* Steps */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Steps: {characterImageSettings.steps}</Label>
                      <Slider
                        min={10}
                        max={50}
                        step={1}
                        value={[characterImageSettings.steps]}
                        onValueChange={([val]) => setCharacterImageSettings(prev => ({ ...prev, steps: val }))}
                        disabled={isGeneratingCharacterImage}
                      />
                    </div>

                    {/* Guidance Scale */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Guidance Scale: {characterImageSettings.guidance_scale.toFixed(1)}</Label>
                      <Slider
                        min={1}
                        max={15}
                        step={0.1}
                        value={[characterImageSettings.guidance_scale]}
                        onValueChange={([val]) => setCharacterImageSettings(prev => ({ ...prev, guidance_scale: val }))}
                        disabled={isGeneratingCharacterImage}
                      />
                    </div>

                    {/* Sampler */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Sampler</Label>
                      <select
                        value={characterImageSettings.sampler}
                        onChange={e => setCharacterImageSettings(prev => ({ ...prev, sampler: e.target.value }))}
                        disabled={isGeneratingCharacterImage}
                        className="w-full rounded border border-input bg-background text-foreground px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                      >
                        <option value="Euler a">Euler a</option>
                        <option value="Euler">Euler</option>
                        <option value="LMS">LMS</option>
                        <option value="Heun">Heun</option>
                        <option value="DPM2">DPM2</option>
                        <option value="DPM2 a">DPM2 a</option>
                        <option value="DPM++ 2S a">DPM++ 2S a</option>
                        <option value="DPM++ 2M">DPM++ 2M</option>
                        <option value="DPM++ SDE">DPM++ SDE</option>
                        <option value="DDIM">DDIM</option>
                        <option value="DPM++ 2M Karras">DPM++ 2M Karras</option>
                        <option value="DPM++ 2S a Karras">DPM++ 2S a Karras</option>
                        <option value="DPM++ SDE Karras">DPM++ SDE Karras</option>
                      </select>
                    </div>

                    {/* Seed */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Seed</Label>
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          value={characterImageSettings.seed}
                          onChange={e => setCharacterImageSettings(prev => ({ ...prev, seed: Number(e.target.value) }))}
                          placeholder="-1 for random"
                          className="flex-1 rounded border border-input bg-transparent px-3 py-2 text-sm"
                          disabled={isGeneratingCharacterImage}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setCharacterImageSettings(prev => ({ ...prev, seed: -1 }))}
                          disabled={isGeneratingCharacterImage}
                        >
                          Random
                        </Button>
                      </div>
                    </div>

                    {/* Prompt Controls */}
                    <div className="border-t pt-4 space-y-3">
                      <Label className="text-sm font-medium">Image Prompt</Label>
                      <div className="space-y-2">
                        <Button
                          variant={!showCustomPrompt ? "secondary" : "outline"}
                          size="sm"
                          onClick={() => setShowCustomPrompt(false)}
                          className="mr-2"
                        >
                          Auto-Generated
                        </Button>
                        <Button
                          variant={showCustomPrompt ? "secondary" : "outline"}
                          size="sm"
                          onClick={() => setShowCustomPrompt(true)}
                        >
                          Custom Prompt
                        </Button>
                      </div>

                      {showCustomPrompt ? (
                        <div>
                          <Textarea
                            value={customImagePrompt}
                            onChange={(e) => setCustomImagePrompt(e.target.value)}
                            placeholder="Enter custom image prompt..."
                            className="min-h-[80px] text-sm"
                          />
                          <Button
                            onClick={() => handleGenerateCharacterImage(true)}
                            disabled={isGeneratingCharacterImage || !customImagePrompt.trim()}
                            size="sm"
                            className="mt-2 w-full"
                          >
                            Generate with Custom Prompt
                          </Button>
                        </div>
                      ) : (
                        <div className="bg-muted p-3 rounded text-sm min-h-[80px]">
                          <p className="font-medium mb-1">Auto-generated prompt:</p>
                          <p className="text-muted-foreground text-xs">{characterImagePrompt}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              {/* Feedback Section */}
              <div className="border-t pt-4">
                <h3 className="font-semibold mb-2">Refine Character</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Provide feedback to improve the character. Be specific about what you'd like changed.
                </p>

                <Textarea
                  value={characterFeedback}
                  onChange={(e) => setCharacterFeedback(e.target.value)}
                  placeholder="e.g., 'Make them more sarcastic and add a mysterious past' or 'Change the setting to modern day'"
                  className="mb-3"
                  rows={3}
                />

                <div className="flex gap-2">
                  <Button
                    onClick={handleRefineCharacter}
                    disabled={!characterFeedback.trim() || isGeneratingCharacter}
                    variant="outline"
                    size="sm"
                  >
                    {isGeneratingCharacter ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Refining...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                          <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                          <path d="M3 3v5h5" />
                          <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                          <path d="M21 21v-5h-5" />
                        </svg>
                        Refine Character
                      </>
                    )}
                  </Button>

                  {characterFeedback && (
                    <Button
                      onClick={() => setCharacterFeedback('')}
                      variant="ghost"
                      size="sm"
                    >
                      Clear Feedback
                    </Button>
                  )}
                </div>
              </div>
            </div>

            <DialogFooter className="flex justify-between">
              <Button
                variant="outline"
                onClick={() => {
                  setShowCharacterPreview(false);
                  setGeneratedCharacter(null);
                  setCharacterFeedback('');
                }}
              >
                Cancel
              </Button>

              <Button
                onClick={handleSaveCharacter}
                disabled={isGeneratingCharacter}
                className="bg-green-600 hover:bg-green-700"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                  <polyline points="17,21 17,13 7,13 7,21" />
                  <polyline points="7,3 7,8 15,8" />
                </svg>
                Save to Library
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
      {isFocusModeActive && (
        <FocusModeOverlay
          isActive={isFocusModeActive}
          onExit={() => setIsFocusModeActive(false)}
          messages={messages}
          handleSubmit={handleSubmit}
          isGenerating={isGenerating}
          primaryModel={primaryModel}
          renderAvatar={renderAvatar}
          renderUserAvatar={renderUserAvatar}
          PRIMARY_API_URL={PRIMARY_API_URL}
          activeCharacter={activeCharacter}
          primaryCharacter={primaryCharacter}
          secondaryCharacter={secondaryCharacter}
          getCurrentVariantContent={getCurrentVariantContent}
          getVariantCount={getVariantCount}
          navigateVariant={navigateVariant}
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
          currentVariantIndex={currentVariantIndex}
          formatModelName={formatModelName}
          stopTTS={stopTTS}
        />
      )}
      {/* Code Editor Overlay */}
      {codeEditorEnabled && (
        <CodeEditorOverlay
          isOpen={codeEditorEnabled}
          onClose={() => setCodeEditorEnabled(false)}
        />
      )}
      {/* Call Mode Overlay - CONDITIONALLY MOUNTED */}
      {isCallModeActive && (
        <CallModeOverlay
          isActive={isCallModeActive}
          onExit={stopCallMode}
          activeCharacter={activeCharacter}
          isRecording={isRecording}
          isTranscribing={isTranscribing}
          onStartRecording={startRecording}
          onStopRecording={() => stopRecording(setInputValue)}
          PRIMARY_API_URL={PRIMARY_API_URL}
          onOpenStoryTracker={() => setShowStoryTracker(true)}
          onOpenChoiceGenerator={() => setShowChoiceGenerator(true)}
          messages={messages}
          onRegenerate={handleGenerateVariant}
        />
      )}
      {/* Forensic Linguistics Overlay */}
      {showForensicLinguistics && (
        <ForensicLinguistics
          isOpen={showForensicLinguistics}
          onClose={() => setShowForensicLinguistics(false)}
        />
      )}

      {/* Story Tracker Panel */}
      <StoryTracker
        isOpen={showStoryTracker}
        onClose={() => setShowStoryTracker(false)}
        messages={messages}
        onAnalyze={handleAnalyzeStory}
        isAnalyzing={isAnalyzingStory}
        onInjectContext={(context) => setInputValue(prev => prev + (prev ? '\n\n' : '') + context)}
        activeCharacter={activeCharacter}
      />

      {/* Choice Generator Panel */}
      <ChoiceGenerator
        isOpen={showChoiceGenerator}
        onClose={() => setShowChoiceGenerator(false)}
        messages={messages}
        onSelectChoice={handleChoiceSelect}
        apiUrl={PRIMARY_API_URL}
        isGenerating={isGenerating}
        primaryModel={primaryModel}
        activeCharacter={activeCharacter}
        userProfile={userProfile}
      />

    </div>
  );
};

export default Chat;