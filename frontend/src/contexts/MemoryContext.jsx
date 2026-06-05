import React, { createContext, useState, useCallback, useContext, useEffect } from 'react';
import * as indexedDbStorage from '../utils/indexedDbStorage';
import {
  parseProfilesBlob,
  pickBestProfilesSource,
  shouldPersistUserProfiles,
} from '../utils/userProfilesStorage';

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

const cleanProfilesState = (parsedState) => {
  if (!parsedState?.profiles) return parsedState;
  return {
    ...parsedState,
    profiles: parsedState.profiles.map((profile) => ({
      id: profile.id,
      name: profile.name,
      avatar: profile.avatar,
      preferences: profile.preferences || { topics: [], responseStyle: "balanced" }
    }))
  };
};

export const MemoryProvider = ({ children }) => {
  const [memoryState, setMemoryState] = useState(defaultMemoryState);
  const [isLoading, setIsLoading] = useState(true);
  const [profilesLoadStatus, setProfilesLoadStatus] = useState('pending');

  // Load profiles: localStorage first (sync), then optional IndexedDB if richer
  useEffect(() => {
    let cancelled = false;

    const applyFromRaw = (raw) => {
      const parsed = parseProfilesBlob(raw);
      if (!parsed || parsed.count === 0) return false;
      setMemoryState(cleanProfilesState(typeof raw === 'string' ? JSON.parse(raw) : raw));
      setProfilesLoadStatus('loaded');
      return true;
    };

    const loadProfiles = async () => {
      setIsLoading(true);
      setProfilesLoadStatus('pending');
      let loaded = false;
      let loadedCount = 0;

      try {
        const savedProfiles = localStorage.getItem('user-profiles');
        if (savedProfiles && applyFromRaw(savedProfiles)) {
          loaded = true;
          loadedCount = parseProfilesBlob(savedProfiles)?.count ?? 0;
        } else {
          const legacyMemories = localStorage.getItem('user-memories');
          if (legacyMemories) {
            const legacyData = JSON.parse(legacyMemories);
            setMemoryState({
              profiles: [{
                id: generateId(),
                name: legacyData.userProfile?.name || "Migrated User",
                avatar: legacyData.userProfile?.avatar || null,
                preferences: legacyData.userProfile?.preferences || { topics: [], responseStyle: "balanced" }
              }],
              activeProfileId: null
            });
            setProfilesLoadStatus('loaded');
            loaded = true;
          }
        }

        if (!cancelled) {
          try {
            const lsRaw = localStorage.getItem('user-profiles');
            const idbRaw = await indexedDbStorage.getItem('user-profiles', { skipMigration: true });
            if (!cancelled) {
              const best = pickBestProfilesSource(idbRaw, lsRaw);
              if (best.raw) {
                const bestCount = parseProfilesBlob(best.raw)?.count ?? 0;
                if (!loaded || bestCount > loadedCount) {
                  applyFromRaw(best.raw);
                  loaded = true;
                  loadedCount = bestCount;
                  if (best.restoreIdb) {
                    const str = typeof best.raw === 'string' ? best.raw : JSON.stringify(parseProfilesBlob(best.raw));
                    void indexedDbStorage.setItem('user-profiles', str);
                  }
                }
              }
            }
          } catch (idbErr) {
            console.warn('[MemoryContext] IndexedDB profile read skipped:', idbErr);
          }
        }

        if (!loaded && !cancelled) {
          setProfilesLoadStatus('empty');
        }
      } catch (error) {
        console.error("Error loading profiles:", error);
        if (!cancelled) setProfilesLoadStatus('error');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    void loadProfiles();
    return () => { cancelled = true; };
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

  // Save profiles to localStorage (+ IndexedDB mirror); never clobber richer on-disk data
  useEffect(() => {
    if (isLoading || !shouldPersistUserProfiles(memoryState, profilesLoadStatus)) {
      return;
    }
    const profilesMetadataOnly = {
      profiles: memoryState.profiles.map(profile => ({
        id: profile.id,
        name: profile.name,
        avatar: profile.avatar,
        preferences: profile.preferences
      })),
      activeProfileId: memoryState.activeProfileId
    };
    const str = JSON.stringify(profilesMetadataOnly);
    try {
      localStorage.setItem('user-profiles', str);
    } catch (e) {
      console.warn('[MemoryContext] localStorage setItem failed:', e);
    }
    void indexedDbStorage.setItem('user-profiles', str);
  }, [memoryState, isLoading, profilesLoadStatus]);

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
