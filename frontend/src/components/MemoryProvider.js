// MemoryProvider.js
import React, { createContext, useContext, useState } from 'react';

const MemoryContext = createContext();

export const MemoryProvider = ({ children }) => {
  const [memorySettings, setMemorySettings] = useState({
    autoMemoryEnabled: false,
    explicitMemoryOnly: true,
    memoryPreference: 'opt-in',
    memoryImportanceThreshold: 0.7,
  });

  const updateMemorySettings = (newSettings) => {
    setMemorySettings(prev => ({ ...prev, ...newSettings }));
  };

  const toggleMemoryOptIn = (enabled) => {
    setMemorySettings(prev => ({ ...prev, autoMemoryEnabled: enabled }));
  };

  const toggleExplicitMemoryOnly = (enabled) => {
    setMemorySettings(prev => ({ ...prev, explicitMemoryOnly: enabled }));
  };

  const value = {
    memorySettings,
    updateMemorySettings,
    toggleMemoryOptIn,
    toggleExplicitMemoryOnly,
    resetMemories: () => console.log("Resetting memories..."), // Stub
    curate: () => console.log("Curating duplicates..."),      // Stub
    memories: [],  // Inject your real memory array here
    isLoading: false, // Optional loading state
  };

  return (
    <MemoryContext.Provider value={value}>
      {children}
    </MemoryContext.Provider>
  );
};

export const useMemory = () => useContext(MemoryContext);
