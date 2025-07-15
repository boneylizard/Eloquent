import React, { createContext, useState, useCallback, useContext, useEffect } from 'react';

const MemoryContext = createContext(null);

export const useMemory = () => useContext(MemoryContext);

// Helper to generate IDs
const generateId = () => `profile_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

// Initialize with default structure for a profile - NO MEMORIES ARRAY
const createDefaultProfile = (name = "Default User") => ({
  id: generateId(),
  name,
  avatar: null,
  preferences: {
    topics: [],
    responseStyle: "balanced"
  }
  // Removed memories array - all memories now backend-only
});

// Initial state - only profile metadata, no memories
const defaultMemoryState = {
  profiles: [createDefaultProfile()],
  activeProfileId: null
};

export const MemoryProvider = ({ children }) => {
  const [memoryState, setMemoryState] = useState(defaultMemoryState);
  const [isLoading, setIsLoading] = useState(true);

  // Load profiles from localStorage (metadata only)
  useEffect(() => {
    const loadProfiles = () => {
      setIsLoading(true);
      try {
        const savedProfiles = localStorage.getItem('user-profiles');
        
        if (savedProfiles) {
          const parsedState = JSON.parse(savedProfiles);
          
          // Clean any existing memories from localStorage profiles
          if (parsedState.profiles) {
            parsedState.profiles = parsedState.profiles.map(profile => ({
              id: profile.id,
              name: profile.name,
              avatar: profile.avatar,
              preferences: profile.preferences || { topics: [], responseStyle: "balanced" }
              // Explicitly remove memories, conversations arrays
            }));
          }
          
          setMemoryState(parsedState);
        } else {
          // Check for legacy format and migrate metadata only
          const legacyMemories = localStorage.getItem('user-memories');
          if (legacyMemories) {
            const legacyData = JSON.parse(legacyMemories);
            
            const migratedState = {
              profiles: [{
                id: generateId(),
                name: legacyData.userProfile?.name || "Migrated User",
                avatar: legacyData.userProfile?.avatar || null,
                preferences: legacyData.userProfile?.preferences || { topics: [], responseStyle: "balanced" }
                // No memories - they should be migrated via API if needed
              }],
              activeProfileId: null
            };
            
            setMemoryState(migratedState);
          }
        }
      } catch (error) {
        console.error("Error loading profiles:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadProfiles();
  }, []);

  // Set active profile if not already set
  useEffect(() => {
    if (!isLoading && (!memoryState.activeProfileId || !memoryState.profiles.find(p => p.id === memoryState.activeProfileId))) {
      setMemoryState(prev => ({
        ...prev,
        activeProfileId: prev.profiles[0]?.id
      }));
    }
  }, [isLoading, memoryState.activeProfileId, memoryState.profiles]);

  // Save profiles to localStorage (metadata only, no memories)
  useEffect(() => {
    if (!isLoading) {
      const profilesMetadataOnly = {
        profiles: memoryState.profiles.map(profile => ({
          id: profile.id,
          name: profile.name,
          avatar: profile.avatar,
          preferences: profile.preferences
          // Exclude memories - they live in backend only
        })),
        activeProfileId: memoryState.activeProfileId
      };
      localStorage.setItem('user-profiles', JSON.stringify(profilesMetadataOnly));
    }
  }, [memoryState, isLoading]);

  // Get the active profile (metadata only)
  const activeProfile = memoryState.profiles.find(p => p.id === memoryState.activeProfileId) || memoryState.profiles[0];

  // Switch to a different profile
  const switchProfile = useCallback((profileId) => {
    if (memoryState.profiles.some(p => p.id === profileId)) {
      setMemoryState(prev => ({
        ...prev,
        activeProfileId: profileId
      }));
    }
  }, [memoryState.profiles]);

  // Add a new profile
  const addProfile = useCallback((name = "New User") => {
    const newProfile = createDefaultProfile(name);
    
    setMemoryState(prev => ({
      ...prev,
      profiles: [...prev.profiles, newProfile],
      activeProfileId: newProfile.id
    }));
    
    return newProfile.id;
  }, []);

  // Rename a profile
  const renameProfile = useCallback((profileId, newName) => {
    setMemoryState(prev => ({
      ...prev,
      profiles: prev.profiles.map(profile => 
        profile.id === profileId 
          ? { ...profile, name: newName } 
          : profile
      )
    }));
  }, []);

  // Delete a profile
  const deleteProfile = useCallback((profileId) => {
    if (memoryState.profiles.length <= 1) {
      alert("Cannot delete the only profile.");
      return;
    }
    
    setMemoryState(prev => {
      const updatedProfiles = prev.profiles.filter(profile => profile.id !== profileId);
      const newActiveId = prev.activeProfileId === profileId 
        ? updatedProfiles[0].id 
        : prev.activeProfileId;
        
      return {
        ...prev,
        profiles: updatedProfiles,
        activeProfileId: newActiveId
      };
    });
  }, [memoryState.profiles]);

  // Update user profile (metadata only)
  const updateUserProfile = useCallback((updates) => {
    setMemoryState(prev => ({
      ...prev,
      profiles: prev.profiles.map(profile => 
        profile.id === prev.activeProfileId 
          ? {
              ...profile,
              ...updates
            }
          : profile
      )
    }));
  }, []);

  // REMOVED: All memory management functions (addMemory, editMemory, deleteMemory, etc.)
  // These should now only be done via backend API calls

  // Backend-only memory functions - these just pass through to API
  const getRelevantMemories = useCallback(async (query, limit = 5) => {
    // This should call the backend API directly
    console.warn("getRelevantMemories should call backend API directly");
    return [];
  }, []);

  const resetMemories = useCallback(async () => {
    if (window.confirm('Are you sure you want to reset all memories for this profile? This cannot be undone.')) {
      // This should call the backend API to clear memories
      console.warn("resetMemories should call backend API directly");
    }
  }, []);

  return (
    <MemoryContext.Provider value={{
      // Profile metadata only
      userProfile: activeProfile,
      
      // Multi-profile management (metadata only)
      profiles: memoryState.profiles,
      activeProfileId: memoryState.activeProfileId,
      switchProfile,
      addProfile,
      renameProfile,
      deleteProfile,
      
      // Status
      isLoading,
      
      // Profile functions (metadata only)
      updateUserProfile,
      
      // Memory functions - these should be implemented via direct API calls in components
      getRelevantMemories,
      resetMemories,
      
      // REMOVED: Local memory state and management
      // Components should use backend APIs directly via fetch/axios
    }}>
      {children}
    </MemoryContext.Provider>
  );
};

export default MemoryContext;