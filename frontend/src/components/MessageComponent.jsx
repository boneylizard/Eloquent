
// --- MessageComponent.jsx - Updated with Save Memory button ---

import React, { useState } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { Button } from './ui/button';
import { AlertCircle, Save, Check } from 'lucide-react';
import { Tooltip } from './ui/tooltip';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { processAndAddMemory } from '../utils/memoryUtils'; // <-- Check for this import

const MEMORY_CATEGORIES = [
  { value: 'personal_info', label: 'Personal Info' },
  { value: 'preferences', label: 'Preferences' },
  { value: 'interests', label: 'Interests' },
  { value: 'facts', label: 'Facts' },
  { value: 'skills', label: 'Skills' },
  { value: 'opinions', label: 'Opinions' },
  { value: 'experiences', label: 'Experiences' },
  { value: 'other', label: 'Other' }
];

const MessageComponent = ({ message, onReply }) => {
  const { detectMemoryIntent, createExplicitMemory } = useMemory();
  const [isMemoryDialogOpen, setIsMemoryDialogOpen] = useState(false);
  const [memorySaved, setMemorySaved] = useState(false);
  const [memoryContent, setMemoryContent] = useState('');
  const [memoryCategory, setMemoryCategory] = useState('personal_info');
  const [memoryImportance, setMemoryImportance] = useState(0.8);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Detect if message has memory intent
  const hasMemoryIntent = message.role === 'user' && detectMemoryIntent(message.content);
  
  const handleSaveMemory = () => {
    // Extract content after memory indicator if possible
    if (hasMemoryIntent && message.role === 'user') {
      const content = message.content;
      const memoryIndicators = [
        "remember that", "please remember", "note that i", 
        "for future reference", "remember this", "save this"
      ];
      
      // Find the indicator used
      let indicatorUsed = null;
      let startIndex = -1;
      
      for (const indicator of memoryIndicators) {
        const index = content.toLowerCase().indexOf(indicator);
        if (index !== -1 && (startIndex === -1 || index < startIndex)) {
          startIndex = index;
          indicatorUsed = indicator;
        }
      }
      
      if (startIndex !== -1) {
        // Extract content after the indicator
        const extractedContent = content.substring(startIndex + indicatorUsed.length).trim();
        
        // Remove leading punctuation
        const cleanedContent = extractedContent.replace(/^[:\s]+/, '');
        
        setMemoryContent(cleanedContent);
        
        // Try to detect category
        if (content.includes("prefer") || content.includes("like") || content.includes("want")) {
          setMemoryCategory('preferences');
        } else if (content.includes("am a") || content.includes("my name")) {
          setMemoryCategory('personal_info');
        } else if (content.includes("skilled") || content.includes("good at")) {
          setMemoryCategory('skills');
        }
      } else {
        setMemoryContent(content);
      }
    } else {
      setMemoryContent(message.content);
    }
    
    setIsMemoryDialogOpen(true);
  };
  
  const handleSaveMemoryConfirm = async () => {
    setIsProcessing(true);
    try {
      // --- Calls the new utility function with only the content ---
      const result = await processAndAddMemory(memoryContent);
      // --- End Call ---

      // Handles success/error based on backend response
      if (result.status === 'success') {
        setMemorySaved(true);
        setTimeout(() => {
          setIsMemoryDialogOpen(false);
          setTimeout(() => setMemorySaved(false), 300);
        }, 1500);
      } else {
        console.error("Failed to save memory:", result.error || "Unknown error");
        alert(`Failed to save memory: ${result.error || 'Please check backend logs.'}`);
      }
    } catch (error) {
      console.error("Error saving memory:", error);
      alert(`Error saving memory: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
  return (
    <div className={`message ${message.role}`}>
      <div className="message-header">
        <div className="message-avatar">
          {/* Avatar implementation */}
        </div>
        <div className="message-info">
          <span className="message-sender">{message.role === 'user' ? 'You' : (message.charName || 'Assistant')}</span>
          {message.modelName && <span className="model-tag">{message.modelName}</span>}
        </div>
        <div className="message-actions">
          {message.role === 'user' && (
            <Tooltip content={hasMemoryIntent ? "Contains memory intent" : "Save as memory"}>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={handleSaveMemory}
                className={hasMemoryIntent ? "memory-intent-indicator" : ""}
              >
                {hasMemoryIntent ? <AlertCircle className="h-4 w-4 text-yellow-500" /> : <Save className="h-4 w-4" />}
              </Button>
            </Tooltip>
          )}
          {/* Other message actions... */}
        </div>
      </div>
      
      <div className="message-content">
        {message.content}
      </div>
      
      {/* Memory Dialog */}
      <Dialog open={isMemoryDialogOpen} onOpenChange={setIsMemoryDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {memorySaved ? (
                <div className="flex items-center text-green-500">
                  <Check className="mr-2 h-4 w-4" /> Memory Saved
                </div>
              ) : (
                "Save as Memory"
              )}
            </DialogTitle>
          </DialogHeader>
          
          {!memorySaved && (
            <>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="memory-content">Memory Content</Label>
                  <Textarea
                    id="memory-content"
                    value={memoryContent}
                    onChange={(e) => setMemoryContent(e.target.value)}
                    placeholder="Enter information to remember"
                    rows={3}
                  />
                </div>
                <div className="message-actions">
      {/* --- Replace existing save button logic with this --- */}
      {(message.role === 'user' || message.role === 'assistant') && (
        <Tooltip content="Save message content as memory">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSaveMemory} // Opens the dialog
            className="opacity-50 hover:opacity-100 transition-opacity" // Example style
          >
            <Save className="h-4 w-4" /> {/* Always Save icon */}
          </Button>
        </Tooltip>
      )}
      {/* --- End Replacement --- */}
      {/* Other message actions... */}
    </div>
                
                <div className="space-y-2">
                  <Label htmlFor="memory-category">Category</Label>
                  <Select value={memoryCategory} onValueChange={setMemoryCategory}>
                    <SelectTrigger id="memory-category">
                      <SelectValue placeholder="Select a category" />
                    </SelectTrigger>
                    <SelectContent>
                      {MEMORY_CATEGORIES.map((category) => (
                        <SelectItem key={category.value} value={category.value}>
                          {category.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="memory-importance">Importance</Label>
                    <span className="text-sm text-muted-foreground">
                      {memoryImportance.toFixed(1)}
                    </span>
                  </div>
                  <Slider
                    id="memory-importance"
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    value={[memoryImportance]}
                    onValueChange={(values) => setMemoryImportance(values[0])}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Low</span>
                    <span>Medium</span>
                    <span>High</span>
                  </div>
                </div>
              </div>
              
              <DialogFooter>
                <Button 
                  variant="outline" 
                  onClick={() => setIsMemoryDialogOpen(false)}
                  disabled={isProcessing}
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handleSaveMemoryConfirm}
                  disabled={!memoryContent.trim() || isProcessing}
                >
                  {isProcessing ? "Saving..." : "Save Memory"}
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MessageComponent;