import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Input } from './ui/input';
import { X, Plus, Trash2, Users, Package, MapPin, Flag, Sparkles, ChevronDown, ChevronRight } from 'lucide-react';

const StoryTracker = ({ isOpen, onClose, messages, onAnalyze, isAnalyzing }) => {
  const [trackerData, setTrackerData] = useState({
    characters: [],
    inventory: [],
    locations: [],
    plotPoints: [],
    customFields: []
  });
  
  const [expandedSections, setExpandedSections] = useState({
    characters: true,
    inventory: true,
    locations: true,
    plotPoints: true,
    customFields: false
  });
  
  const [newItem, setNewItem] = useState({ section: '', value: '' });

  // Load tracker data from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('eloquent-story-tracker');
    if (saved) {
      try {
        setTrackerData(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load tracker data:', e);
      }
    }
  }, []);

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
      [section]: [...prev[section], { id: Date.now(), value: value.trim(), notes: '' }]
    }));
    setNewItem({ section: '', value: '' });
  };

  const removeItem = (section, id) => {
    setTrackerData(prev => ({
      ...prev,
      [section]: prev[section].filter(item => item.id !== id)
    }));
  };

  const updateItemNotes = (section, id, notes) => {
    setTrackerData(prev => ({
      ...prev,
      [section]: prev[section].map(item => 
        item.id === id ? { ...item, notes } : item
      )
    }));
  };

  const clearAll = () => {
    setTrackerData({
      characters: [],
      inventory: [],
      locations: [],
      plotPoints: [],
      customFields: []
    });
  };

  const sections = [
    { key: 'characters', label: 'Characters', icon: Users, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    { key: 'inventory', label: 'Inventory', icon: Package, color: 'text-amber-500', bg: 'bg-amber-500/10' },
    { key: 'locations', label: 'Locations', icon: MapPin, color: 'text-green-500', bg: 'bg-green-500/10' },
    { key: 'plotPoints', label: 'Plot Points', icon: Flag, color: 'text-purple-500', bg: 'bg-purple-500/10' },
    { key: 'customFields', label: 'Custom', icon: Sparkles, color: 'text-pink-500', bg: 'bg-pink-500/10' }
  ];

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      
      {/* Panel */}
      <div className="relative w-full max-w-md h-[80vh] bg-background border rounded-xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-indigo-500/10 to-purple-500/10">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-500/20">
              <Flag className="w-5 h-5 text-indigo-500" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Story Tracker</h2>
              <p className="text-xs text-muted-foreground">Track your adventure state</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
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

        {/* Content */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-3">
            {sections.map(({ key, label, icon: Icon, color, bg }) => (
              <div key={key} className={`rounded-lg border ${bg}`}>
                {/* Section Header */}
                <button
                  className="w-full flex items-center justify-between p-3 hover:bg-white/5 transition-colors"
                  onClick={() => toggleSection(key)}
                >
                  <div className="flex items-center gap-2">
                    <Icon className={`w-4 h-4 ${color}`} />
                    <span className="font-medium">{label}</span>
                    <span className="text-xs text-muted-foreground">
                      ({trackerData[key].length})
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
                    {trackerData[key].map(item => (
                      <div
                        key={item.id}
                        className="flex items-start gap-2 p-2 rounded bg-background/50 border border-border/50 group"
                      >
                        <div className="flex-1">
                          <div className="font-medium text-sm">{item.value}</div>
                          <Input
                            placeholder="Add notes..."
                            value={item.notes}
                            onChange={(e) => updateItemNotes(key, item.id, e.target.value)}
                            className="mt-1 h-7 text-xs bg-transparent border-dashed"
                          />
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity text-red-500 hover:text-red-600 hover:bg-red-500/10"
                          onClick={() => removeItem(key, item.id)}
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
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
                        className="h-8 text-sm"
                      />
                      <Button
                        variant="outline"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => {
                          if (newItem.section === key) {
                            addItem(key, newItem.value);
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
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="p-3 border-t bg-muted/30 flex justify-between items-center">
          <span className="text-xs text-muted-foreground">
            {Object.values(trackerData).flat().length} items tracked
          </span>
          <Button variant="ghost" size="sm" onClick={clearAll} className="text-xs text-red-500 hover:text-red-600">
            <Trash2 className="w-3 h-3 mr-1" />
            Clear All
          </Button>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default StoryTracker;

