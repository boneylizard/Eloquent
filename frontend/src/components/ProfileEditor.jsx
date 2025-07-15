// ProfileEditor.jsx
import React, { useState, useEffect } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Textarea } from './ui/textarea';
import { Separator } from './ui/separator';
import { Save, Upload, X } from 'lucide-react';

const ProfileEditor = () => {
  const { userProfile, updateUserProfile } = useMemory();
  const [profile, setProfile] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const [changes, setChanges] = useState(false);
  
  // Initialize profile data when userProfile changes
  useEffect(() => {
    if (userProfile) {
      setProfile({
        ...userProfile,
        // Ensure all required sections exist
        preferences: userProfile.preferences || { 
          responseStyle: 'balanced',
          topics: []
        },
        personality: userProfile.personality || {
          traits: [],
          interests: [],
          values: []
        },
        personal_info: userProfile.personal_info || {}
      });
      setChanges(false);
    }
  }, [userProfile]);
  
  const handleSave = () => {
    // Convert string inputs to arrays before saving
    const profileToSave = {
      ...profile,
      personality: {
        ...profile.personality,
        traits: Array.isArray(profile.personality?.traits) 
          ? profile.personality.traits 
          : (profile.personality?.traits || '').split(',').map(item => item.trim()).filter(item => item.length > 0),
        interests: Array.isArray(profile.personality?.interests) 
          ? profile.personality.interests 
          : (profile.personality?.interests || '').split(',').map(item => item.trim()).filter(item => item.length > 0),
        values: Array.isArray(profile.personality?.values) 
          ? profile.personality.values 
          : (profile.personality?.values || '').split(',').map(item => item.trim()).filter(item => item.length > 0)
      },
      preferences: {
        ...profile.preferences,
        topics: Array.isArray(profile.preferences?.topics) 
          ? profile.preferences.topics 
          : (profile.preferences?.topics || '').split(',').map(item => item.trim()).filter(item => item.length > 0)
      }
    };
    
    updateUserProfile(profileToSave);
    setChanges(false);
  };
  
  const handleChange = (section, field, value) => {
    setProfile(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
    setChanges(true);
  };
  
  const handleStringChange = (section, field, value) => {
    // Just store the raw string, don't convert to array yet
    setProfile(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
    setChanges(true);
  };
  
  const handleAvatarUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    setIsUploading(true);
    
    try {
      const response = await fetch('http://localhost:8000/upload_avatar', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const result = await response.json();
      if (result.file_url) {
        setProfile(prev => ({
          ...prev,
          avatar: result.file_url
        }));
        setChanges(true);
      }
    } catch (error) {
      console.error('Avatar upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };
  
  // Helper function to get display value (convert array to string for display)
  const getDisplayValue = (section, field) => {
    const value = profile[section]?.[field];
    if (Array.isArray(value)) {
      return value.join(', ');
    }
    return value || '';
  };
  
  if (!userProfile) {
    return <div>No profile selected</div>;
  }
  
  return (
    <div className="space-y-6">
      <Tabs defaultValue="basic">
        <TabsList className="w-full">
          <TabsTrigger value="basic">Basic Info</TabsTrigger>
          <TabsTrigger value="personality">Personality</TabsTrigger>
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>
        
        {/* Basic Info Tab */}
        <TabsContent value="basic" className="space-y-4 pt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="col-span-1 md:col-span-2">
              <Label htmlFor="profile-name">Display Name</Label>
              <Input
                id="profile-name"
                value={profile.name || ''}
                onChange={(e) => setProfile(prev => ({ ...prev, name: e.target.value }))}
                className="mt-1"
              />
            </div>
            
            <div>
              <Label htmlFor="avatar">Profile Avatar</Label>
              <div className="flex items-center gap-3 mt-1">
                {profile.avatar && (
                  <div className="w-16 h-16 rounded-full overflow-hidden border">
                    <img 
                      src={profile.avatar} 
                      alt="Avatar" 
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXVzZXIiPjxwYXRoIGQ9Ik0xOSAyMXYtMmE0IDQgMCAwIDAtNC00SDlhNCA0IDAgMCAwLTQgNHYyIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSI3IiByPSI0Ii8+PC9zdmc+';
                      }}
                    />
                  </div>
                )}
                
                <div className="flex-1">
                  <Input
                    id="avatar"
                    type="file"
                    accept="image/*"
                    onChange={handleAvatarUpload}
                    disabled={isUploading}
                  />
                  {isUploading && <p className="text-xs text-muted-foreground mt-1">Uploading...</p>}
                </div>
              </div>
            </div>
            
            <div>
              <Label htmlFor="location">Location</Label>
              <Input
                id="location"
                value={profile.personal_info?.location || ''}
                onChange={(e) => handleChange('personal_info', 'location', e.target.value)}
                placeholder="City, Country"
                className="mt-1"
              />
            </div>
          </div>
          
          <div>
            <Label htmlFor="bio">Bio/Description</Label>
            <Textarea
              id="bio"
              value={profile.personal_info?.bio || ''}
              onChange={(e) => handleChange('personal_info', 'bio', e.target.value)}
              placeholder="A brief description about you"
              className="mt-1"
              rows={3}
            />
          </div>
        </TabsContent>
        
        {/* Personality Tab */}
        <TabsContent value="personality" className="space-y-4 pt-4">
          <div>
            <Label htmlFor="traits">Personality Traits</Label>
            <Textarea
              id="traits"
              value={getDisplayValue('personality', 'traits')}
              onChange={(e) => handleStringChange('personality', 'traits', e.target.value)}
              placeholder="curious, analytical, creative, etc. (comma-separated)"
              className="mt-1"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Describe your personality traits to help the AI understand you better.
            </p>
          </div>
          
          <div>
            <Label htmlFor="interests">Interests & Hobbies</Label>
            <Textarea
              id="interests"
              value={getDisplayValue('personality', 'interests')}
              onChange={(e) => handleStringChange('personality', 'interests', e.target.value)}
              placeholder="technology, photography, hiking, etc. (comma-separated)"
              className="mt-1"
            />
          </div>
          
          <div>
            <Label htmlFor="values">Values & Beliefs</Label>
            <Textarea
              id="values"
              value={getDisplayValue('personality', 'values')}
              onChange={(e) => handleStringChange('personality', 'values', e.target.value)}
              placeholder="Environmental conservation, intellectual honesty, etc. (comma-separated)"
              className="mt-1"
            />
          </div>
        </TabsContent>
        
        {/* Preferences Tab */}
        <TabsContent value="preferences" className="space-y-4 pt-4">
          <div>
            <Label htmlFor="responseStyle">Preferred Response Style</Label>
            <Select
              value={profile.preferences?.responseStyle || 'balanced'}
              onValueChange={(value) => handleChange('preferences', 'responseStyle', value)}
            >
              <SelectTrigger id="responseStyle" className="mt-1">
                <SelectValue placeholder="Select style" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="concise">Concise</SelectItem>
                <SelectItem value="balanced">Balanced</SelectItem>
                <SelectItem value="detailed">Detailed</SelectItem>
                <SelectItem value="analytical">Analytical</SelectItem>
                <SelectItem value="creative">Creative</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              How you prefer AI responses to be formatted.
            </p>
          </div>
          
          <div>
            <Label htmlFor="topics">Favorite Topics</Label>
            <Textarea
              id="topics"
              value={getDisplayValue('preferences', 'topics')}
              onChange={(e) => handleStringChange('preferences', 'topics', e.target.value)}
              placeholder="AI, science, art, history, etc. (comma-separated)"
              className="mt-1"
            />
          </div>
          
          <Separator className="my-4" />
          
          <div>
            <Label htmlFor="interaction_style">Interaction Style</Label>
            <Select
              value={profile.preferences?.interaction_style || 'friendly'}
              onValueChange={(value) => handleChange('preferences', 'interaction_style', value)}
            >
              <SelectTrigger id="interaction_style" className="mt-1">
                <SelectValue placeholder="Select style" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="friendly">Friendly & Casual</SelectItem>
                <SelectItem value="professional">Professional & Formal</SelectItem>
                <SelectItem value="educational">Educational & Informative</SelectItem>
                <SelectItem value="creative">Creative & Imaginative</SelectItem>
                <SelectItem value="empathetic">Empathetic & Supportive</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </TabsContent>
        
        {/* System Tab */}
        <TabsContent value="system" className="space-y-4 pt-4">
          <div>
            <Label htmlFor="format">Profile Format</Label>
            <Select
              value={profile.system_settings?.preferred_format || '.json'}
              onValueChange={(value) => handleChange('system_settings', 'preferred_format', value)}
            >
              <SelectTrigger id="format" className="mt-1">
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value=".json">JSON</SelectItem>
                <SelectItem value=".yaml">YAML</SelectItem>
                <SelectItem value=".toml">TOML</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              Format for saving this profile. Changes apply on next save.
            </p>
          </div>
          
          <div>
            <Label className="flex items-center justify-between">
              Memory Settings
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setProfile(prev => ({
                    ...prev,
                    system_settings: {
                      ...prev.system_settings,
                      memory_enabled: !prev.system_settings?.memory_enabled
                    }
                  }));
                  setChanges(true);
                }}
              >
                {profile.system_settings?.memory_enabled ? 'Enabled' : 'Disabled'}
              </Button>
            </Label>
            <p className="text-xs text-muted-foreground mt-1">
              Control whether the AI can create and use memories for this profile.
            </p>
          </div>
          
          <div>
            <Label htmlFor="explicit_only">Memory Creation Mode</Label>
            <Select
              value={profile.system_settings?.explicit_memory_only ? 'explicit' : 'auto'}
              onValueChange={(value) => handleChange(
                'system_settings', 
                'explicit_memory_only', 
                value === 'explicit'
              )}
            >
              <SelectTrigger id="explicit_only" className="mt-1">
                <SelectValue placeholder="Select mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="explicit">Explicit Only (On Request)</SelectItem>
                <SelectItem value="auto">Automatic (AI Decides)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              Control when new memories are created from conversations.
            </p>
          </div>
        </TabsContent>
      </Tabs>
      
      <div className="flex justify-end">
        <Button 
          onClick={handleSave} 
          disabled={!changes}
          className="px-4"
        >
          <Save className="mr-2 h-4 w-4" />
          Save Profile
        </Button>
      </div>
    </div>
  );
};

export default ProfileEditor;