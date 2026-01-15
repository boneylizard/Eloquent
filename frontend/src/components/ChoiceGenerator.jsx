import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { X, Loader2, Sparkles, ArrowRight, RefreshCw, Dice6, Edit3, Wand2, Settings2, Plus, Trash2, Send, RotateCcw, ChevronDown, ChevronUp } from 'lucide-react';

const ChoiceGenerator = ({
  isOpen,
  onClose,
  messages,
  onSelectChoice,
  apiUrl,
  isGenerating: parentIsGenerating,
  primaryModel,
  activeCharacter,
  userProfile // Add user profile for context
}) => {
  const [choices, setChoices] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [regeneratingIndex, setRegeneratingIndex] = useState(null);
  const [expandingChoice, setExpandingChoice] = useState(null); // Track when expanding a choice to full prompt
  const [error, setError] = useState(null);
  const [editingIndex, setEditingIndex] = useState(null);
  const [editAction, setEditAction] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [customPrompt, setCustomPrompt] = useState('');
  const [showSettings, setShowSettings] = useState(true); // Show settings by default
  const [behaviorMode, setBehaviorMode] = useState('balanced');
  const [isDirectorMode, setIsDirectorMode] = useState(false); // Toggle between Character and Director

  // Get story tracker data for context
  const getStoryContext = () => {
    try {
      const saved = localStorage.getItem('eloquent-story-tracker');
      if (saved) {
        const data = JSON.parse(saved);
        const parts = [];
        if (data.currentObjective) {
          parts.push(`Current objective: ${data.currentObjective}`);
        }
        if (data.characters?.length > 0) {
          parts.push(`Characters: ${data.characters.map(c => c.value + (c.notes ? ` (${c.notes})` : '')).join(', ')}`);
        }
        if (data.locations?.length > 0) {
          parts.push(`Locations: ${data.locations.map(l => l.value).join(', ')}`);
        }
        if (data.inventory?.length > 0) {
          parts.push(`Inventory: ${data.inventory.map(i => i.value).join(', ')}`);
        }
        if (data.plotPoints?.length > 0) {
          parts.push(`Recent events: ${data.plotPoints.slice(-3).map(p => p.value).join('; ')}`);
        }
        if (data.storyNotes) {
          parts.push(`Notes: ${data.storyNotes}`);
        }
        return parts.join('\n');
      }
    } catch (e) {
      console.error('Failed to load story tracker:', e);
    }
    return '';
  };

  // Get user profile context
  const getUserContext = () => {
    if (!userProfile) return '';
    const parts = [];
    if (userProfile.name) parts.push(`User's name: ${userProfile.name}`);
    if (userProfile.personality) parts.push(`User's personality: ${userProfile.personality}`);
    if (userProfile.background) parts.push(`User's background: ${userProfile.background}`);
    return parts.join('\n');
  };

  const getBehaviorInstructions = () => {
    switch (behaviorMode) {
      case 'dramatic':
        return 'STYLE: Make choices dramatic and high-stakes. Include bold, risky options that could change the story significantly. Think climactic moments.';
      case 'subtle':
        return 'STYLE: Make choices subtle and nuanced. Focus on dialogue, observation, careful approaches, and emotional undertones.';
      case 'chaotic':
        return 'STYLE: Make choices wild and unpredictable. Include absurd, funny, fourth-wall-breaking, or completely unexpected options.';
      case 'romantic':
        return 'STYLE: Focus on romantic, intimate, or emotionally charged options. Include flirty, tender, vulnerable, or passionate choices.';
      case 'action':
        return 'STYLE: Focus on physical action, combat, or adventurous options. Include fighting, fleeing, daring maneuvers, or athletic feats.';
      default:
        return 'STYLE: Balance choices between safe, moderate, and risky options. Include a variety of approaches.';
    }
  };

  const callAPI = async (prompt, temperature = 0.85) => {
    const response = await fetch(`${apiUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        model_name: primaryModel || 'default',
        max_tokens: 800,
        temperature,
        stop: ['```'],
        stream: true,
        gpu_id: 0,
        request_purpose: 'choice_generation'
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

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
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            if (parsed.text) {
              fullText += parsed.text;
            } else if (parsed.token) {
              fullText += parsed.token;
            }
          } catch (e) {
            // Skip unparseable chunks
          }
        }
      }
    }

    return fullText;
  };

  const buildChoicesPrompt = (numChoices = 4, continueFrom = null, existingChoices = []) => {
    const recentMessages = messages.slice(-10);
    const context = recentMessages
      .map(m => `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`)
      .join('\n');

    const characterContext = activeCharacter
      ? `CHARACTER: ${activeCharacter.name}\nPersonality: ${activeCharacter.personality || activeCharacter.description || 'Not specified'}\n`
      : '';

    const storyContext = getStoryContext();
    const storySection = storyContext ? `\nSTORY STATE:\n${storyContext}\n` : '';

    const userContext = getUserContext();
    const userSection = userContext ? `\nUSER PLAYING AS:\n${userContext}\n` : '';

    const behaviorInst = getBehaviorInstructions();

    // Custom direction - make it prominent
    const customDirection = customPrompt.trim()
      ? `\n⚠️ IMPORTANT USER DIRECTION - YOU MUST FOLLOW THIS:\n${customPrompt.trim()}\n`
      : '';

    const continueSection = continueFrom
      ? `\nBASE ACTION: "${continueFrom}"\nGenerate ${numChoices} variations or ways to elaborate on this action.\n`
      : '';

    const avoidSection = existingChoices.length > 0
      ? `\nDo NOT repeat these existing choices:\n${existingChoices.map(c => `- ${c.action}`).join('\n')}\n`
      : '';

    if (isDirectorMode) {
      return `You are a creative "Director" agent for a roleplay.
${characterContext}${userSection}${storySection}
${behaviorInst}
${customDirection}
RECENT CONVERSATION:
${context}

${continueSection}${avoidSection}
Generate ${numChoices} distinct "Narrative Beats" or "Plot Events" that should happen next.
These should NOT be actions for the user character, but things that happen in the world, NPC actions, or environmental shifts.

Each choice needs:
- "action": A short, punchy title for the event (3-6 words)
- "description": One sentence explaining the narrative beat
- "emoji": A single relevant emoji for this event

Output ONLY a JSON array:
[{"action": "...", "description": "...", "emoji": "..."}]`;
    }

    return `You are a creative roleplay choice generator.
${characterContext}${userSection}${storySection}
${behaviorInst}
${customDirection}
RECENT CONVERSATION:
${context}

${continueSection}${avoidSection}
Generate exactly ${numChoices} distinct action choice${numChoices > 1 ? 's' : ''} the user could take next.
Each choice needs:
- "action": A short, punchy action phrase (3-6 words)
- "description": One sentence explaining what this choice means
- "emoji": A single relevant emoji for this action

Output ONLY a JSON array, nothing else:
[{"action": "...", "description": "...", "emoji": "..."}]`;
  };

  // Generate a full user prompt from a choice
  const expandChoiceToFullPrompt = async (choice) => {
    const recentMessages = messages.slice(-6);
    const context = recentMessages
      .map(m => `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`)
      .join('\n');

    const characterContext = activeCharacter
      ? `You are writing a message TO ${activeCharacter.name}.\nTheir personality: ${activeCharacter.personality || activeCharacter.description || 'Not specified'}\n`
      : '';

    const storyContext = getStoryContext();
    const userContext = getUserContext();

    const prompt = `You are a roleplay prompt writer. Write a first-person user message/action.
${characterContext}
${userContext ? `THE USER'S CHARACTER:\n${userContext}\n` : ''}
${storyContext ? `STORY CONTEXT:\n${storyContext}\n` : ''}

RECENT CONVERSATION:
${context}

THE USER WANTS TO: ${choice.action}
(${choice.description})

Write a 2-4 sentence first-person roleplay message where the user takes this action. 
- Write in first person (I, me, my)
- Include physical actions in *asterisks*
- Include any dialogue in "quotes"
- Make it feel natural and in-character
- Match the tone and style of the conversation

Write ONLY the user's message, nothing else:`;

    const response = await callAPI(prompt, 0.9);

    // Clean up the response - remove any quotes or prefixes
    let cleaned = response.trim();
    // Remove "User:" or similar prefixes
    cleaned = cleaned.replace(/^(User|Me|I):\s*/i, '');
    // Remove surrounding quotes if present
    if ((cleaned.startsWith('"') && cleaned.endsWith('"')) ||
      (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
      cleaned = cleaned.slice(1, -1);
    }

    return cleaned;
  };

  // Generate all 4 choices
  const generateAllChoices = async (continueFrom = null) => {
    if (messages.length === 0) {
      setError('No conversation to generate choices from');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const prompt = buildChoicesPrompt(4, continueFrom);
      console.log('[ChoiceGenerator] Prompt:', prompt); // Debug log

      const fullText = await callAPI(prompt);
      console.log('[ChoiceGenerator] Response:', fullText); // Debug log

      // Parse JSON from response
      const firstBracket = fullText.indexOf('[');
      const lastBracket = fullText.lastIndexOf(']');

      if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
        const jsonStr = fullText.slice(firstBracket, lastBracket + 1);
        const parsedChoices = JSON.parse(jsonStr);

        if (Array.isArray(parsedChoices) && parsedChoices.length > 0) {
          setChoices(parsedChoices.slice(0, 4));
        } else {
          throw new Error('Invalid choice format');
        }
      } else {
        throw new Error('Could not parse choices from response');
      }
    } catch (err) {
      console.error('Choice generation error:', err);
      setError(err.message || 'Failed to generate choices');
      setChoices([
        { action: 'Continue the conversation', description: 'Keep talking and see where it leads.' },
        { action: 'Take a bold action', description: 'Do something decisive and unexpected.' },
        { action: 'Observe and wait', description: 'Hold back and assess the situation carefully.' },
        { action: 'Change the subject', description: 'Steer things in a completely new direction.' }
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Regenerate a single choice
  const regenerateSingleChoice = async (index, continueFrom = null) => {
    setRegeneratingIndex(index);
    setError(null);

    try {
      const otherChoices = choices.filter((_, i) => i !== index);
      const prompt = buildChoicesPrompt(1, continueFrom, otherChoices);
      const fullText = await callAPI(prompt);

      const firstBracket = fullText.indexOf('[');
      const lastBracket = fullText.lastIndexOf(']');

      if (firstBracket !== -1 && lastBracket !== -1) {
        const jsonStr = fullText.slice(firstBracket, lastBracket + 1);
        const parsedChoices = JSON.parse(jsonStr);

        if (Array.isArray(parsedChoices) && parsedChoices.length > 0) {
          const newChoices = [...choices];
          newChoices[index] = parsedChoices[0];
          setChoices(newChoices);
        }
      }
    } catch (err) {
      console.error('Single choice regeneration error:', err);
      setError(`Failed to regenerate choice ${index + 1}`);
    } finally {
      setRegeneratingIndex(null);
    }
  };

  // Expand a choice to variations
  const expandFromChoice = async (index) => {
    const choice = choices[index];
    setRegeneratingIndex(index);
    setError(null);

    try {
      const prompt = buildChoicesPrompt(1, choice.action, choices.filter((_, i) => i !== index));
      const fullText = await callAPI(prompt);

      const firstBracket = fullText.indexOf('[');
      const lastBracket = fullText.lastIndexOf(']');

      if (firstBracket !== -1 && lastBracket !== -1) {
        const jsonStr = fullText.slice(firstBracket, lastBracket + 1);
        const parsedChoices = JSON.parse(jsonStr);

        if (Array.isArray(parsedChoices) && parsedChoices.length > 0) {
          const newChoices = [...choices];
          newChoices[index] = parsedChoices[0];
          setChoices(newChoices);
        }
      }
    } catch (err) {
      console.error('Expand choice error:', err);
      setError(`Failed to expand choice ${index + 1}`);
    } finally {
      setRegeneratingIndex(null);
    }
  };

  // Handle selecting a choice - expand to full prompt first
  const handleSelectChoice = async (choice) => {
    setExpandingChoice(choice);
    setError(null);

    try {
      if (isDirectorMode) {
        // In Director Mode, we send it as a director note
        onSelectChoice(choice.action, choice.description, 'director');
        onClose();
        return;
      }

      // Generate full prompt from the choice
      const fullPrompt = await expandChoiceToFullPrompt(choice);

      // Send the expanded prompt
      onSelectChoice(fullPrompt, choice.description, 'prose');
      onClose();
    } catch (err) {
      console.error('Failed to expand choice:', err);
      // Fallback to just the action text
      onSelectChoice(`*${choice.action}*`, choice.description, 'direct');
      onClose();
    } finally {
      setExpandingChoice(null);
    }
  };

  const handleStartEdit = (index) => {
    setEditingIndex(index);
    setEditAction(choices[index].action);
    setEditDescription(choices[index].description);
  };

  const handleSaveEdit = (index) => {
    if (editAction.trim()) {
      const newChoices = [...choices];
      newChoices[index] = {
        action: editAction.trim(),
        description: editDescription.trim() || choices[index].description
      };
      setChoices(newChoices);
    }
    setEditingIndex(null);
    setEditAction('');
    setEditDescription('');
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
    setEditAction('');
    setEditDescription('');
  };

  const handleDeleteChoice = (index) => {
    setChoices(choices.filter((_, i) => i !== index));
  };

  const handleAddChoice = () => {
    setChoices([...choices, { action: 'New action', description: 'Describe what happens...' }]);
    setEditingIndex(choices.length);
    setEditAction('New action');
    setEditDescription('Describe what happens...');
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

      {/* Expanding overlay */}
      {expandingChoice && (
        <div className="absolute inset-0 bg-black/70 z-10 flex items-center justify-center">
          <div className="text-center text-white">
            <Loader2 className="w-12 h-12 mx-auto animate-spin mb-4" />
            <p className="text-lg font-medium">Writing your action...</p>
            <p className="text-sm text-white/70 mt-1">"{expandingChoice.action}"</p>
          </div>
        </div>
      )}

      {/* Panel */}
      <div className="relative w-full max-w-xl bg-background border rounded-xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className={`flex items-center justify-between p-4 border-b transition-colors duration-500 ${isDirectorMode
            ? 'bg-gradient-to-r from-purple-500/10 to-indigo-500/10'
            : 'bg-gradient-to-r from-amber-500/10 to-orange-500/10'
          }`}>
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg transition-colors duration-500 ${isDirectorMode ? 'bg-purple-500/20' : 'bg-amber-500/20'
              }`}>
              {isDirectorMode ? (
                <Sparkles className="w-5 h-5 text-purple-500" />
              ) : (
                <Dice6 className="w-5 h-5 text-amber-500" />
              )}
            </div>
            <div>
              <h2 className="text-lg font-bold">
                {isDirectorMode ? 'Director Mode' : 'What will you do?'}
              </h2>
              <p className="text-xs text-muted-foreground">
                {isDirectorMode
                  ? 'Drive the story with narrative beats'
                  : (activeCharacter ? `${activeCharacter.name} will respond` : 'Choose your action')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex bg-muted rounded-lg p-0.5 mr-2">
              <button
                onClick={() => {
                  setIsDirectorMode(false);
                  setChoices([]);
                }}
                className={`px-3 py-1 text-xs rounded-md transition-all ${!isDirectorMode ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
              >
                Character
              </button>
              <button
                onClick={() => {
                  setIsDirectorMode(true);
                  setChoices([]);
                }}
                className={`px-3 py-1 text-xs rounded-md transition-all ${isDirectorMode ? 'bg-purple-500 text-white shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
              >
                Director
              </button>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Settings Panel - Collapsible but prominent */}
        <div className="border-b">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="w-full p-3 flex items-center justify-between hover:bg-muted/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Settings2 className="w-4 h-4 text-amber-500" />
              <span className="font-medium text-sm">Generation Settings</span>
              {customPrompt && (
                <span className="text-xs bg-amber-500/20 text-amber-600 px-2 py-0.5 rounded">
                  Direction set
                </span>
              )}
            </div>
            {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showSettings && (
            <div className="p-4 pt-0 space-y-4 bg-muted/20">
              {/* Behavior Mode */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">Tone & Style</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { value: 'balanced', label: 'Balanced', emoji: '⚖️' },
                    { value: 'dramatic', label: 'Dramatic', emoji: '🎭' },
                    { value: 'subtle', label: 'Subtle', emoji: '🤫' },
                    { value: 'chaotic', label: 'Chaotic', emoji: '🎲' },
                    { value: 'romantic', label: 'Romantic', emoji: '💕' },
                    { value: 'action', label: 'Action', emoji: '⚔️' }
                  ].map(mode => (
                    <Button
                      key={mode.value}
                      variant={behaviorMode === mode.value ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setBehaviorMode(mode.value)}
                      className={`text-xs justify-start ${behaviorMode === mode.value ? 'bg-amber-500 hover:bg-amber-600' : ''}`}
                    >
                      {mode.emoji} {mode.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Custom Direction */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">
                  🎯 Your Direction (AI will follow this)
                </label>
                <Textarea
                  placeholder="Tell the AI what kind of choices you want...&#10;&#10;Examples:&#10;• 'Include a choice where I confess my feelings'&#10;• 'One option should involve using magic'&#10;• 'Make the choices more aggressive'&#10;• 'Add a funny/silly option'"
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  className="h-24 text-sm resize-none"
                />
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto flex-1">
          {choices.length === 0 && !isGenerating && !error && (
            <div className="text-center py-6">
              <Sparkles className="w-10 h-10 mx-auto text-amber-500/50 mb-3" />
              <p className="text-muted-foreground mb-2">
                Generate choices based on your story
              </p>
              <p className="text-xs text-muted-foreground/70 mb-4">
                Clicking a choice will write a full action and send it
              </p>
              <Button
                onClick={() => generateAllChoices()}
                disabled={messages.length === 0 || parentIsGenerating}
                className={`transition-all duration-500 ${isDirectorMode
                    ? 'bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600'
                    : 'bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600'
                  }`}
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Generate {isDirectorMode ? 'Beats' : 'Choices'}
              </Button>
            </div>
          )}

          {isGenerating && (
            <div className="text-center py-6">
              <Loader2 className="w-10 h-10 mx-auto text-amber-500 animate-spin mb-3" />
              <p className="text-muted-foreground">Generating choices...</p>
              {behaviorMode !== 'balanced' && (
                <p className="text-xs text-amber-500 mt-1">Mode: {behaviorMode}</p>
              )}
              {customPrompt && (
                <p className="text-xs text-muted-foreground/70 mt-1 max-w-xs mx-auto truncate">
                  Direction: "{customPrompt}"
                </p>
              )}
            </div>
          )}

          {error && (
            <div className="text-center py-3 text-red-500 text-sm mb-3 bg-red-500/10 rounded-lg">
              {error}
            </div>
          )}

          {choices.length > 0 && !isGenerating && (
            <div className="space-y-3">
              {choices.map((choice, index) => (
                <div
                  key={index}
                  className={`w-full text-left p-3 rounded-lg border transition-all
                    ${regeneratingIndex === index
                      ? (isDirectorMode ? 'bg-purple-500/10 border-purple-500/30' : 'bg-amber-500/10 border-amber-500/30')
                      : `bg-card hover:bg-accent/50 ${isDirectorMode ? 'hover:border-purple-500/30' : 'hover:border-amber-500/30'}`
                    }
                    ${editingIndex === index ? (isDirectorMode ? 'ring-2 ring-purple-500/50' : 'ring-2 ring-amber-500/50') : ''}
                  `}
                >
                  {regeneratingIndex === index ? (
                    <div className="flex items-center justify-center py-3">
                      <Loader2 className="w-5 h-5 text-amber-500 animate-spin mr-2" />
                      <span className="text-sm text-muted-foreground">Regenerating...</span>
                    </div>
                  ) : editingIndex === index ? (
                    <div className="space-y-3">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1 block">Action</label>
                        <Input
                          value={editAction}
                          onChange={(e) => setEditAction(e.target.value)}
                          placeholder="Short action (3-6 words)"
                          className="font-semibold"
                          autoFocus
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1 block">Description</label>
                        <Textarea
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          placeholder="What does this action mean?"
                          className="h-16 text-sm resize-none"
                        />
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        <Button size="sm" onClick={() => handleSaveEdit(index)}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={handleCancelEdit}>Cancel</Button>
                        <Button
                          size="sm"
                          variant="default"
                          className="bg-green-600 hover:bg-green-700"
                          onClick={() => {
                            handleSaveEdit(index);
                            setTimeout(() => handleSelectChoice({ action: editAction, description: editDescription }), 100);
                          }}
                        >
                          <Send className="w-3 h-3 mr-1" />
                          Save & Do It
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="group">
                      {/* Choice content - clickable to send */}
                      <button
                        className="w-full text-left"
                        onClick={() => handleSelectChoice(choice)}
                        disabled={parentIsGenerating || expandingChoice}
                      >
                        <div className="flex items-start gap-2">
                          <span className={`w-8 h-8 rounded-full flex items-center justify-center text-lg flex-shrink-0 mt-0.5 ${isDirectorMode ? 'bg-purple-500/20 shadow-inner' : 'bg-amber-500/20'
                            }`}>
                            {choice.emoji || (isDirectorMode ? '🎬' : index + 1)}
                          </span>
                          <div className="flex-1 min-w-0">
                            <span className="font-semibold block">{choice.action}</span>
                            <p className="text-sm text-muted-foreground mt-0.5">
                              {choice.description}
                            </p>
                          </div>
                          <ArrowRight className="w-5 h-5 text-amber-500/30 group-hover:text-amber-500 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1" />
                        </div>
                      </button>

                      {/* Action buttons */}
                      <div className="flex items-center gap-1 mt-2 ml-8 opacity-50 group-hover:opacity-100 transition-opacity">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-xs px-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartEdit(index);
                          }}
                        >
                          <Edit3 className="w-3 h-3 mr-1" />
                          Edit
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-xs px-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            regenerateSingleChoice(index);
                          }}
                        >
                          <RotateCcw className="w-3 h-3 mr-1" />
                          Regen
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-xs px-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            expandFromChoice(index);
                          }}
                        >
                          <Wand2 className="w-3 h-3 mr-1" />
                          Vary
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 text-xs px-2 text-red-500 hover:text-red-600"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteChoice(index);
                          }}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Add custom choice */}
              <button
                onClick={handleAddChoice}
                className="w-full p-2 rounded-lg border border-dashed border-muted-foreground/30 hover:border-amber-500/50 
                           text-muted-foreground hover:text-amber-500 transition-all flex items-center justify-center gap-2 text-sm"
              >
                <Plus className="w-4 h-4" />
                Add custom choice
              </button>

              {/* Regenerate all */}
              <div className="flex justify-center pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => generateAllChoices()}
                  disabled={isGenerating || parentIsGenerating || regeneratingIndex !== null}
                >
                  <RefreshCw className="w-3 h-3 mr-2" />
                  Regenerate All
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};

export default ChoiceGenerator;
