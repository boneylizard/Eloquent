// SimpleUserProfileEditor.jsx - Fixed version with correct API URLs
import React, { useState, useEffect } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Save, Loader2, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { useApp } from '../contexts/AppContext';

const SimpleUserProfileEditor = () => {
  const { activeProfileId } = useMemory();
  const { SECONDARY_API_URL } = useApp();
  const [profileText, setProfileText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [existingMemories, setExistingMemories] = useState([]);

  // Load existing profile memories - copied from working MemoryEditorTab
  const loadExistingMemories = async () => {
    if (!activeProfileId) {
      setExistingMemories([]);
      return;
    }
    setIsLoading(true);
    try {
      const res = await fetch(`${SECONDARY_API_URL}/memory/get_all?user_id=${activeProfileId}`);
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      if (data.status === 'success' && Array.isArray(data.memories)) {
        const sorted = data.memories.sort((a, b) => new Date(b.created) - new Date(a.created));
        setExistingMemories(sorted);
        // Convert existing memories to readable text
        const existingText = sorted
          .map(memory => memory.content)
          .join('\n\n');
        setProfileText(existingText);
      } else {
        throw new Error(data.error || 'Unexpected response');
      }
    } catch (error) {
      console.error('Error loading memories:', error);
      setExistingMemories([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadExistingMemories();
  }, [activeProfileId]);

  // Parse text into memory entries
  const parseTextToMemories = (text) => {
    if (!text.trim()) return [];

    const sections = text.includes('\n\n') 
      ? text.split('\n\n').filter(s => s.trim())
      : text.split('\n').filter(s => s.trim());

    return sections.map(content => {
      const trimmed = content.trim();
      
      // Simple categorization
      let category = 'personal_info';
      let importance = 0.7;

      if (trimmed.toLowerCase().includes('expert') || 
          trimmed.toLowerCase().includes('professional') ||
          trimmed.toLowerCase().includes('specializ')) {
        category = 'expertise';
        importance = 0.9;
      } else if (trimmed.toLowerCase().includes('hobby') || 
                 trimmed.toLowerCase().includes('interest') ||
                 trimmed.toLowerCase().includes('enjoy') ||
                 trimmed.toLowerCase().includes('love')) {
        category = 'personal_interest';
        importance = 0.7;
      } else if (trimmed.toLowerCase().includes('prefer') || 
                 trimmed.toLowerCase().includes('like') ||
                 trimmed.toLowerCase().includes('dislike')) {
        category = 'preferences';
        importance = 0.6;
      }

      return {
        content: trimmed,
        category: category,
        importance: importance,
        type: 'manual',
        tags: [],
        created: new Date().toISOString(),
        accessed: 0,
        user_id: activeProfileId,  // Fixed: backend expects user_id, not user
        last_accessed: new Date().toISOString()
      };
    });
  };

  // Save only new/changed content
  const handleSave = async () => {
    if (!activeProfileId) {
      alert('No active profile selected');
      return;
    }

    if (!profileText.trim()) {
      alert('Please enter some profile information');
      return;
    }

    setIsSaving(true);
    try {
      const newMemories = parseTextToMemories(profileText);
      const existingContents = existingMemories.map(m => m.content.trim());
      
      // Only save memories that don't already exist
      const memoriesToAdd = newMemories.filter(memory => 
        !existingContents.includes(memory.content.trim())
      );

      if (memoriesToAdd.length === 0) {
        alert('No new content to save');
        setIsSaving(false);
        return;
      }

      let successCount = 0;
      for (const memory of memoriesToAdd) {
        try {
          const saveResponse = await fetch(`${SECONDARY_API_URL}/memory/memory/create`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(memory)
          });

          if (saveResponse.ok) {
            successCount++;
          } else {
            console.error(`Failed to save: ${memory.content.substring(0, 50)}...`);
          }
        } catch (error) {
          console.error(`Error saving memory: ${error.message}`);
        }
      }

      if (successCount > 0) {
        alert(`Successfully added ${successCount} new memory entries!`);
        // Reload to show updated memories
        await loadExistingMemories();
      } else {
        alert('Failed to save any new memories');
      }
      
    } catch (error) {
      console.error('Error saving profile:', error);
      alert(`Error saving profile: ${error.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="animate-spin mr-2 h-4 w-4" />
          Loading profile...
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Profile Information
            <Button 
              variant="outline" 
              size="sm" 
              onClick={loadExistingMemories}
              disabled={isLoading}
            >
              <RefreshCw className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </CardTitle>
          <CardDescription>
            Add information about yourself. Only new content will be saved as memory entries.
          </CardDescription>
          {activeProfileId && (
            <div className="text-sm text-muted-foreground">
              Active Profile: {activeProfileId}
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="profile-text">Profile Content</Label>
            <Textarea
              id="profile-text"
              value={profileText}
              onChange={(e) => setProfileText(e.target.value)}
              placeholder={`Add information about yourself:

I am an expert in UX design and AI alignment theory.

My favorite hobby is building experimental AI tools.

I prefer concise, technical responses.

I work as a software engineer at a tech startup.`}
              className="min-h-[500px] max-h-[70vh] resize-y font-mono text-sm leading-relaxed"
              style={{ 
                fieldSizing: 'content',
                overflowY: 'auto'
              }}
            />
            <div className="text-xs text-muted-foreground">
              <p>• Separate different topics with blank lines</p>
              <p>• System automatically categorizes as expertise, interests, preferences, etc.</p>
              <p>• Only new content will be added when you save</p>
              <p>• Current memories: {existingMemories.length}</p>
            </div>
          </div>

          <div className="flex justify-end space-x-2">
            <Button
              onClick={handleSave}
              disabled={isSaving || !profileText.trim() || !activeProfileId}
            >
              {isSaving ? (
                <>
                  <Loader2 className="animate-spin mr-2 h-4 w-4" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save New Content
                </>
              )}
            </Button>
          </div>

          {!activeProfileId && (
            <div className="text-sm text-amber-600 bg-amber-50 dark:bg-amber-950/20 p-3 rounded">
              No active profile selected. Please select or create a profile above.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default SimpleUserProfileEditor;