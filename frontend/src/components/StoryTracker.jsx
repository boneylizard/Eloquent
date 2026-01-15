import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { X, Plus, Trash2, Users, Package, MapPin, Flag, Sparkles, ChevronDown, ChevronRight, Pin, Copy, MessageSquare, Star, Edit3, Check, BookOpen } from 'lucide-react';

const StoryTracker = ({
  isOpen,
  onClose,
  messages,
  onAnalyze,
  isAnalyzing,
  onInjectContext, // New: callback to inject text into chat
  activeCharacter // New: current character for context
}) => {
  const [trackerData, setTrackerData] = useState({
    characters: [],
    inventory: [],
    locations: [],
    plotPoints: [],
    customFields: [],
    storyNotes: '', // general story notes
    currentObjective: '', // current goal/objective
    sceneSummary: '' // New: concise description of current scene
  });

  const [expandedSections, setExpandedSections] = useState({
    characters: true,
    inventory: true,
    locations: true,
    plotPoints: true,
    customFields: false
  });

  const [newItem, setNewItem] = useState({ section: '', value: '' });
  const [editingItem, setEditingItem] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [showContextPanel, setShowContextPanel] = useState(false);

  // Load tracker data from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('eloquent-story-tracker');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setTrackerData(prev => ({ ...prev, ...parsed }));
      } catch (e) {
        console.error('Failed to load tracker data:', e);
      }
    }
  }, [isOpen]); // Reload when opened

  // Save tracker data to localStorage
  useEffect(() => {
    localStorage.setItem('eloquent-story-tracker', JSON.stringify(trackerData));
  }, [trackerData]);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const addItem = (section, value) => {
    if (!value.trim()) return;
    setTrackerData(prev => ({
      ...prev,
      [section]: [...prev[section], {
        id: Date.now(),
        value: value.trim(),
        notes: '',
        pinned: false,
        important: false
      }]
    }));
    setNewItem({ section: '', value: '' });
  };

  const removeItem = (section, id) => {
    setTrackerData(prev => ({
      ...prev,
      [section]: prev[section].filter(item => item.id !== id)
    }));
  };

  const updateItem = (section, id, updates) => {
    setTrackerData(prev => ({
      ...prev,
      [section]: prev[section].map(item =>
        item.id === id ? { ...item, ...updates } : item
      )
    }));
  };

  const togglePin = (section, id) => {
    updateItem(section, id, { pinned: !trackerData[section].find(i => i.id === id)?.pinned });
  };

  const toggleImportant = (section, id) => {
    updateItem(section, id, { important: !trackerData[section].find(i => i.id === id)?.important });
  };

  const clearAll = () => {
    if (window.confirm('Clear all tracked items? This cannot be undone.')) {
      setTrackerData({
        characters: [],
        inventory: [],
        locations: [],
        plotPoints: [],
        customFields: [],
        storyNotes: '',
        currentObjective: '',
        sceneSummary: ''
      });
    }
  };
  // Generate context summary for AI
  const generateContextSummary = () => {
    const sections = [];

    if (trackerData.currentObjective) {
      sections.push(`[CURRENT OBJECTIVE]: ${trackerData.currentObjective}`);
    }

    if (trackerData.characters?.length > 0) {
      const chars = trackerData.characters.map(c => c.notes ? `${c.value} (${c.notes})` : c.value).join(', ');
      sections.push(`[CHARACTERS]: ${chars}`);
    }

    if (trackerData.inventory?.length > 0) {
      const items = trackerData.inventory.map(i => i.notes ? `${i.value} (${i.notes})` : i.value).join(', ');
      sections.push(`[INVENTORY]: ${items}`);
    }

    if (trackerData.locations?.length > 0) {
      const locs = trackerData.locations.map(l => l.notes ? `${l.value} (${l.notes})` : l.value).join(', ');
      sections.push(`[LOCATIONS]: ${locs}`);
    }

    if (trackerData.plotPoints?.length > 0) {
      const plots = trackerData.plotPoints.map(p => `• ${p.notes ? `${p.value}: ${p.notes}` : p.value}`).join('\n');
      sections.push(`[KEY EVENTS / RECENT HISTORY]:\n${plots}`);
    }

    if (trackerData.storyNotes) {
      sections.push(`[STORY NOTES / WORLD BACKGROUND]:\n${trackerData.storyNotes}`);
    }

    if (trackerData.sceneSummary) {
      sections.push(`[CURRENT SCENE]: ${trackerData.sceneSummary}`);
    }

    if (trackerData.customFields?.length > 0) {
      const custom = trackerData.customFields.map(c => c.notes ? `${c.value}: ${c.notes}` : c.value).join(', ');
      sections.push(`[ADDITIONAL DETAILS]: ${custom}`);
    }

    if (sections.length === 0) return '';

    return `\n\n[STORY TRACKER - Essential continuity guidance for this response]\n${sections.join('\n\n')}`;
  };

  const copyContextToClipboard = () => {
    const context = generateContextSummary();
    navigator.clipboard.writeText(context);
  };

  const injectContextToChat = () => {
    if (onInjectContext) {
      const context = generateContextSummary();
      onInjectContext(context);
    }
  };

  const sections = [
    { key: 'characters', label: 'Characters', icon: Users, color: 'text-blue-500', bg: 'bg-blue-500/10', borderColor: 'border-blue-500/30' },
    { key: 'inventory', label: 'Inventory', icon: Package, color: 'text-amber-500', bg: 'bg-amber-500/10', borderColor: 'border-amber-500/30' },
    { key: 'locations', label: 'Locations', icon: MapPin, color: 'text-green-500', bg: 'bg-green-500/10', borderColor: 'border-green-500/30' },
    { key: 'plotPoints', label: 'Plot Points', icon: Flag, color: 'text-purple-500', bg: 'bg-purple-500/10', borderColor: 'border-purple-500/30' },
    { key: 'customFields', label: 'Custom', icon: Sparkles, color: 'text-pink-500', bg: 'bg-pink-500/10', borderColor: 'border-pink-500/30' }
  ];

  // Sort items: pinned first, then important, then regular
  const sortItems = (items) => {
    return [...items].sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      if (a.important && !b.important) return -1;
      if (!a.important && b.important) return 1;
      return 0;
    });
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-md h-[85vh] bg-background border rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-indigo-500/10 to-purple-500/10">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-500/20">
              <BookOpen className="w-5 h-5 text-indigo-500" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Story Tracker</h2>
              <p className="text-xs text-muted-foreground">
                {activeCharacter ? `Story with ${activeCharacter.name}` : 'Track your adventure'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={onAnalyze}
              disabled={isAnalyzing || messages.length === 0}
              className="text-xs"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin mr-1" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3 mr-1" />
                  Auto-Detect
                </>
              )}
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Current Objective */}
        <div className="p-3 border-b bg-gradient-to-r from-amber-500/5 to-orange-500/5">
          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1 mb-1">
            🎯 Current Objective
          </label>
          <Input
            placeholder="What are you trying to accomplish?"
            value={trackerData.currentObjective}
            onChange={(e) => setTrackerData(prev => ({ ...prev, currentObjective: e.target.value }))}
            className="h-8 text-sm bg-background/50"
          />
        </div>

        {/* Scene Summary */}
        <div className="p-3 border-b bg-gradient-to-r from-blue-500/5 to-indigo-500/5">
          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1 mb-1">
            🎬 Scene Summary
          </label>
          <Textarea
            placeholder="Describe the current scene, mood, and immediate surroundings..."
            value={trackerData.sceneSummary || ''}
            onChange={(e) => setTrackerData(prev => ({ ...prev, sceneSummary: e.target.value }))}
            className="min-h-[40px] text-sm bg-background/50 resize-none py-1"
          />
        </div>

        {/* Content */}
        <ScrollArea className="flex-1 p-3">
          <div className="space-y-3">
            {sections.map(({ key, label, icon: Icon, color, bg, borderColor }) => (
              <div key={key} className={`rounded-lg border ${borderColor} ${bg}`}>
                {/* Section Header */}
                <button
                  className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors"
                  onClick={() => toggleSection(key)}
                >
                  <div className="flex items-center gap-2">
                    <Icon className={`w-4 h-4 ${color}`} />
                    <span className="font-medium">{label}</span>
                    <span className="text-xs text-muted-foreground bg-background/50 px-1.5 py-0.5 rounded">
                      {trackerData[key]?.length || 0}
                    </span>
                  </div>
                  {expandedSections[key] ? (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  )}
                </button>

                {/* Section Content */}
                {expandedSections[key] && (
                  <div className="px-3 pb-3 space-y-2">
                    {/* Items */}
                    {sortItems(trackerData[key] || []).map(item => (
                      <div
                        key={item.id}
                        className={`flex items-start gap-2 p-2 rounded bg-background/50 border transition-all group
                          ${item.pinned ? 'border-amber-500/50 bg-amber-500/5' : 'border-border/50'}
                          ${item.important ? 'ring-1 ring-yellow-500/30' : ''}
                        `}
                      >
                        <div className="flex-1 min-w-0">
                          {editingItem === item.id ? (
                            <div className="space-y-1">
                              <Input
                                value={editValue}
                                onChange={(e) => setEditValue(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') {
                                    updateItem(key, item.id, { value: editValue });
                                    setEditingItem(null);
                                  }
                                  if (e.key === 'Escape') setEditingItem(null);
                                }}
                                className="h-7 text-sm"
                                autoFocus
                              />
                              <div className="flex gap-1">
                                <Button
                                  size="sm"
                                  className="h-6 text-xs"
                                  onClick={() => {
                                    updateItem(key, item.id, { value: editValue });
                                    setEditingItem(null);
                                  }}
                                >
                                  <Check className="w-3 h-3" />
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <>
                              <div className="font-medium text-sm flex items-center gap-1">
                                {item.important && <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />}
                                {item.pinned && <Pin className="w-3 h-3 text-amber-500" />}
                                <span className="truncate">{item.value}</span>
                              </div>
                              <Input
                                placeholder="Add notes..."
                                value={item.notes || ''}
                                onChange={(e) => updateItem(key, item.id, { notes: e.target.value })}
                                className="mt-1 h-6 text-xs bg-transparent border-dashed border-border/30 focus:border-border"
                              />
                            </>
                          )}
                        </div>

                        {/* Item actions */}
                        <div className="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-5 w-5"
                            onClick={() => toggleImportant(key, item.id)}
                            title="Mark as important"
                          >
                            <Star className={`w-3 h-3 ${item.important ? 'text-yellow-500 fill-yellow-500' : ''}`} />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-5 w-5"
                            onClick={() => togglePin(key, item.id)}
                            title="Pin to top"
                          >
                            <Pin className={`w-3 h-3 ${item.pinned ? 'text-amber-500' : ''}`} />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-5 w-5"
                            onClick={() => {
                              setEditingItem(item.id);
                              setEditValue(item.value);
                            }}
                            title="Edit"
                          >
                            <Edit3 className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-5 w-5 text-red-500 hover:text-red-600 hover:bg-red-500/10"
                            onClick={() => removeItem(key, item.id)}
                            title="Delete"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    ))}

                    {/* Add New Item */}
                    <div className="flex gap-2">
                      <Input
                        placeholder={`Add ${label.toLowerCase()}...`}
                        value={newItem.section === key ? newItem.value : ''}
                        onChange={(e) => setNewItem({ section: key, value: e.target.value })}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && newItem.section === key) {
                            addItem(key, newItem.value);
                          }
                        }}
                        className="h-7 text-sm"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => {
                          if (newItem.section === key) {
                            addItem(key, newItem.value);
                          } else {
                            setNewItem({ section: key, value: '' });
                          }
                        }}
                      >
                        <Plus className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Story Notes */}
            <div className="rounded-lg border border-slate-500/30 bg-slate-500/10 p-3">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1 mb-2">
                📝 Story Notes
              </label>
              <Textarea
                placeholder="General notes about the story, character relationships, secrets discovered..."
                value={trackerData.storyNotes || ''}
                onChange={(e) => setTrackerData(prev => ({ ...prev, storyNotes: e.target.value }))}
                className="min-h-[60px] text-sm bg-background/50 resize-none"
              />
            </div>
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="p-3 border-t bg-muted/30 space-y-2">
          {/* Context Actions */}
          <div className="flex gap-2">
            <Button
              variant="default"
              size="sm"
              className="flex-1 text-xs bg-indigo-600 hover:bg-indigo-700 text-white font-bold"
              onClick={() => {
                localStorage.setItem('eloquent-story-tracker', JSON.stringify(trackerData));
                alert('Story Tracker state saved! This will be used in all subsequent messages.');
              }}
            >
              <Check className="w-3 h-3 mr-1" />
              Save Changes
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex-1 text-xs"
              onClick={copyContextToClipboard}
            >
              <Copy className="w-3 h-3 mr-1" />
              Copy
            </Button>
            {onInjectContext && (
              <Button
                variant="outline"
                size="sm"
                className="flex-1 text-xs"
                onClick={injectContextToChat}
              >
                <MessageSquare className="w-3 h-3 mr-1" />
                Inject
              </Button>
            )}
          </div>

          {/* Stats and Clear */}
          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">
              {Object.entries(trackerData)
                .filter(([k]) => Array.isArray(trackerData[k]))
                .reduce((sum, [, arr]) => sum + arr.length, 0)} items tracked
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearAll}
              className="text-xs text-red-500 hover:text-red-600"
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Clear All
            </Button>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
};

// Export helper to get tracker context for AI prompts
export const getStoryTrackerContext = () => {
  try {
    const saved = localStorage.getItem('eloquent-story-tracker');
    if (!saved) return '';

    const tracker = JSON.parse(saved);
    const sections = [];

    if (tracker.currentObjective) {
      sections.push(`[CURRENT OBJECTIVE]: ${tracker.currentObjective}`);
    }

    if (tracker.characters?.length > 0) {
      const chars = tracker.characters.map(c => c.notes ? `${c.value} (${c.notes})` : c.value).join(', ');
      sections.push(`[CHARACTERS]: ${chars}`);
    }

    if (tracker.inventory?.length > 0) {
      const items = tracker.inventory.map(i => i.notes ? `${i.value} (${i.notes})` : i.value).join(', ');
      sections.push(`[INVENTORY]: ${items}`);
    }

    if (tracker.locations?.length > 0) {
      const locs = tracker.locations.map(l => l.notes ? `${l.value} (${l.notes})` : l.value).join(', ');
      sections.push(`[LOCATIONS]: ${locs}`);
    }

    if (tracker.plotPoints?.length > 0) {
      const plots = tracker.plotPoints.map(p => `• ${p.notes ? `${p.value}: ${p.notes}` : p.value}`).join('\n');
      sections.push(`[KEY EVENTS / RECENT HISTORY]:\n${plots}`);
    }

    if (tracker.storyNotes) {
      sections.push(`[STORY NOTES / WORLD BACKGROUND]:\n${tracker.storyNotes}`);
    }

    if (tracker.sceneSummary) {
      sections.push(`[CURRENT SCENE]: ${tracker.sceneSummary}`);
    }

    if (tracker.customFields?.length > 0) {
      const custom = tracker.customFields.map(c => c.notes ? `${c.value}: ${c.notes}` : c.value).join(', ');
      sections.push(`[ADDITIONAL DETAILS]: ${custom}`);
    }

    if (sections.length === 0) return '';

    return `\n\n[STORY TRACKER - Essential continuity guidance for this response]\n${sections.join('\n\n')}`;
  } catch (e) {
    console.error('Failed to get story tracker context:', e);
  }
  return '';
};

export default StoryTracker;
