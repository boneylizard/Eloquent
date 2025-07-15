// ProfileSelector.jsx - Fixed to fetch memory counts from backend
import React, { useState, useEffect } from 'react';
import { useMemory } from '../contexts/MemoryContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Edit, Trash2, Plus, Save, X, RefreshCw } from 'lucide-react';
import { useApp } from '../contexts/AppContext';

const ProfileSelector = () => {
  const { 
    profiles, 
    activeProfileId, 
    switchProfile, 
    addProfile, 
    renameProfile, 
    deleteProfile,
    isLoading
  } = useMemory();
  const { SECONDARY_API_URL } = useApp(); // Use SECONDARY_API_URL for memory operations
  const [newProfileName, setNewProfileName] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [memoryCounts, setMemoryCounts] = useState({}); // Store memory counts by profile ID
  
  // Fetch memory count for a specific profile
  const fetchMemoryCount = async (profileId) => {
    try {
      const response = await fetch(`${SECONDARY_API_URL}/memory/get_all?user_id=${profileId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success' && Array.isArray(data.memories)) {
          return data.memories.length;
        }
      }
      return 0;
    } catch (error) {
      console.error(`Error fetching memory count for profile ${profileId}:`, error);
      return 0;
    }
  };

  // Fetch memory counts for all profiles
  const fetchAllMemoryCounts = async () => {
    const counts = {};
    for (const profile of profiles) {
      counts[profile.id] = await fetchMemoryCount(profile.id);
    }
    setMemoryCounts(counts);
  };

  // Fetch memory counts when profiles change
  useEffect(() => {
    if (profiles.length > 0) {
      fetchAllMemoryCounts();
    }
  }, [profiles, SECONDARY_API_URL]);

  // Handle backend profile sync
  const syncWithBackend = async () => {
    if (!activeProfileId) return;
    
    try {
      const response = await fetch(`${SECONDARY_API_URL}/user/profile/set-active/${activeProfileId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        console.log("✅ [Profile] Profile synced with backend");
      } else {
        console.error("❌ [Profile] Failed to sync profile with backend");
      }
    } catch (error) {
      console.error("❌ [Profile] Error syncing profile:", error);
    }
  };

  // Call sync when active profile changes
  useEffect(() => {
    if (activeProfileId) {
      console.log("🔄 [Profile] Active profile changed to:", activeProfileId);
      syncWithBackend();
    }
  }, [activeProfileId]);
  
  const handleCreateProfile = () => {
    if (newProfileName.trim()) {
      const newProfileId = addProfile(newProfileName.trim());
      setMemoryCounts(prev => ({ ...prev, [newProfileId]: 0 })); // New profiles start with 0 memories
      setNewProfileName('');
      setIsCreating(false);
    }
  };
  
  const handleRenameProfile = (id) => {
    if (editName.trim()) {
      renameProfile(id, editName.trim());
      setEditingId(null);
      setEditName('');
    }
  };
  
  const handleDeleteProfile = (id) => {
    if (window.confirm('Are you sure you want to delete this profile? All associated memories will be lost.')) {
      deleteProfile(id);
      // Remove memory count for deleted profile
      setMemoryCounts(prev => {
        const updated = { ...prev };
        delete updated[id];
        return updated;
      });
    }
  };

  // Refresh memory counts manually
  const handleRefreshCounts = () => {
    fetchAllMemoryCounts();
  };
  
  if (isLoading) {
    return <div className="flex items-center justify-center p-4">
      <RefreshCw className="h-5 w-5 animate-spin mr-2" />
      <span>Loading profiles...</span>
    </div>;
  }
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Your Profiles</h3>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handleRefreshCounts}
            title="Refresh memory counts"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setIsCreating(true)}
            disabled={isCreating}
          >
            <Plus className="h-4 w-4 mr-1" /> New Profile
          </Button>
        </div>
      </div>
      
      {/* Profile creation form */}
      {isCreating && (
        <div className="border rounded-md p-3 mb-4 flex items-center gap-2">
          <Input
            placeholder="Profile name"
            value={newProfileName}
            onChange={(e) => setNewProfileName(e.target.value)}
            className="flex-1"
            autoFocus
          />
          <Button size="sm" onClick={handleCreateProfile} disabled={!newProfileName.trim()}>
            <Save className="h-4 w-4 mr-1" /> Create
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setIsCreating(false)}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}
      
      {/* Profile cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {profiles.map((profile) => (
          <Card
            key={profile.id}
            className={`p-3 border cursor-pointer transition-colors ${
              profile.id === activeProfileId ? 'border-primary bg-primary/5' : ''
            }`}
            onClick={() => profile.id !== activeProfileId && switchProfile(profile.id)}
          >
            {editingId === profile.id ? (
              <div className="flex items-center gap-2">
                <Input
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="flex-1"
                  autoFocus
                />
                <Button size="sm" onClick={() => handleRenameProfile(profile.id)}>
                  <Save className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ) : (
              <div className="flex justify-between items-center">
                <div>
                  <div className="font-medium flex items-center">
                    {profile.name}
                    {profile.id === activeProfileId && (
                      <Badge variant="outline" className="ml-2 text-xs">Active</Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {memoryCounts[profile.id] || 0} memories
                  </div>
                </div>
                
                <div className="flex gap-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditingId(profile.id);
                      setEditName(profile.name);
                    }}
                  >
                    <Edit className="h-4 w-4" />
                  </Button>
                  
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteProfile(profile.id);
                    }}
                    disabled={profiles.length <= 1}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </Card>
        ))}
      </div>
      
      <div className="text-xs text-muted-foreground mt-2">
        <p>Click a profile to switch. The active profile's memories and preferences will be used in conversations.</p>
      </div>
    </div>
  );
};

export default ProfileSelector;