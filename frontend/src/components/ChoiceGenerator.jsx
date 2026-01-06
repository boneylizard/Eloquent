import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './ui/button';
import { X, Loader2, Sparkles, ArrowRight, RefreshCw, Dice6 } from 'lucide-react';

const ChoiceGenerator = ({ 
  isOpen, 
  onClose, 
  messages, 
  onSelectChoice, 
  apiUrl,
  isGenerating: parentIsGenerating 
}) => {
  const [choices, setChoices] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);

  const generateChoices = async () => {
    if (messages.length === 0) {
      setError('No conversation to generate choices from');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      // Build context from recent messages
      const recentMessages = messages.slice(-10);
      const context = recentMessages
        .map(m => `${m.role === 'user' ? 'User' : 'Character'}: ${m.content}`)
        .join('\n');

      const prompt = `Based on this roleplay/story conversation, generate exactly 4 distinct action choices the user could take next. Make them varied - include safe, risky, creative, and unexpected options.

CONVERSATION:
${context}

Respond ONLY with a JSON array of exactly 4 choices. Each choice should be an object with "action" (short action name, 3-5 words) and "description" (one sentence elaboration). Example format:
[
  {"action": "Draw your weapon", "description": "Ready yourself for combat and take an aggressive stance."},
  {"action": "Attempt diplomacy", "description": "Try to negotiate and find a peaceful resolution."},
  {"action": "Search the room", "description": "Look around for clues, items, or alternative exits."},
  {"action": "Do something unexpected", "description": "Surprise everyone with an unconventional approach."}
]

JSON array only, no other text:`;

      const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens: 500,
          temperature: 0.9,
          stop: ['\n\n', '```']
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                fullText += parsed.token;
              }
            } catch (e) {
              // Skip unparseable chunks
            }
          }
        }
      }

      // Parse the choices from the response
      const jsonMatch = fullText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const parsedChoices = JSON.parse(jsonMatch[0]);
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
      // Provide fallback choices
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

  const handleSelectChoice = (choice) => {
    onSelectChoice(choice.action);
    onClose();
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      {/* Panel */}
      <div className="relative w-full max-w-lg bg-background border rounded-xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-amber-500/10 to-orange-500/10">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-amber-500/20">
              <Dice6 className="w-5 h-5 text-amber-500" />
            </div>
            <div>
              <h2 className="text-lg font-bold">What will you do?</h2>
              <p className="text-xs text-muted-foreground">Choose your next action</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="p-4">
          {choices.length === 0 && !isGenerating && !error && (
            <div className="text-center py-8">
              <Sparkles className="w-12 h-12 mx-auto text-amber-500/50 mb-4" />
              <p className="text-muted-foreground mb-4">
                Generate action choices based on your current story
              </p>
              <Button 
                onClick={generateChoices} 
                disabled={messages.length === 0 || parentIsGenerating}
                className="bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                Generate Choices
              </Button>
            </div>
          )}

          {isGenerating && (
            <div className="text-center py-8">
              <Loader2 className="w-12 h-12 mx-auto text-amber-500 animate-spin mb-4" />
              <p className="text-muted-foreground">Thinking of possibilities...</p>
            </div>
          )}

          {error && (
            <div className="text-center py-4 text-red-500 text-sm">
              {error}
            </div>
          )}

          {choices.length > 0 && !isGenerating && (
            <div className="space-y-3">
              {choices.map((choice, index) => (
                <button
                  key={index}
                  className="w-full text-left p-4 rounded-lg border bg-card hover:bg-accent hover:border-amber-500/50 transition-all group"
                  onClick={() => handleSelectChoice(choice)}
                  disabled={parentIsGenerating}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-500 flex items-center justify-center text-sm font-bold">
                          {index + 1}
                        </span>
                        <span className="font-semibold">{choice.action}</span>
                      </div>
                      <p className="text-sm text-muted-foreground pl-8">
                        {choice.description}
                      </p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-amber-500 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1" />
                  </div>
                </button>
              ))}

              {/* Regenerate button */}
              <div className="flex justify-center pt-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={generateChoices}
                  disabled={isGenerating || parentIsGenerating}
                >
                  <RefreshCw className="w-3 h-3 mr-2" />
                  Different Choices
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

