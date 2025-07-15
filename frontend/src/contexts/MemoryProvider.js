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

const addMemory = useCallback((memory) => {
  setMemoryState(prev => ({
    ...prev,
    profiles: prev.profiles.map(profile => 
      profile.id === prev.activeProfileId 
        ? {
            ...profile,
            memories: [...profile.memories, memory]
          }
        : profile
    )
  }));

  // Toggle memory opt-in setting
  const toggleMemoryOptIn = useCallback((value) => {
    setMemoryState(prev => ({
      ...prev,
      memoryOptIn: value
    }));
    
    setMemorySettings(prev => ({
      ...prev,
      autoMemoryEnabled: value
    }));
  }, []);

  // Toggle explicit memory only setting
  const toggleExplicitMemoryOnly = useCallback((value) => {
    setMemoryState(prev => ({
      ...prev,
      explicitMemoryOnly: value
    }));
    
    setMemorySettings(prev => ({
      ...prev,
      explicitMemoryOnly: value
    }));
  }, []);

  return (
    <MemoryContext.Provider value={{
      userProfile: activeProfile,
      memories: activeProfile?.memories || [],
      conversationSummaries: activeProfile?.conversations?.summaries || [],
      
      // Memory settings
      memorySettings,
      updateMemorySettings: setMemorySettings,
      memoryOptIn: memoryState.memoryOptIn,
      toggleMemoryOptIn,
      explicitMemoryOnly: memoryState.explicitMemoryOnly,
      toggleExplicitMemoryOnly,
      
      // Multi-profile management
      profiles: memoryState.profiles,
      activeProfileId: memoryState.activeProfileId,
      
      // Status
      isLoading,
      
      // Existing functions
      updateUserProfile,
      addMemory,
      editMemory,
      deleteMemory,
      
      // New functions
      createExplicitMemory,
      getRelevantMemories,
      detectMemoryIntent,
      resetMemories
    }}>
      {children}
    </MemoryContext.Provider>
  );
};

export default MemoryProvider;