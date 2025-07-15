// Save this file as: src/components/LoreDebugger.jsx

import React, { useState } from 'react';
import { useApp } from '../contexts/AppContext'; // Make sure this path is correct

const LoreDebugger = () => {
  const [testText, setTestText] = useState('');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Get needed values from App context - removed unnecessary useContext
  const { activeCharacter, fetchTriggeredLore } = useApp();

  const handleTest = async () => {
    if (!testText) {
      setError('Please enter some text to test');
      return;
    }

    if (!activeCharacter) {
      setError('No active character selected');
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const loreResults = await fetchTriggeredLore(testText, activeCharacter);
      setResults(loreResults);
      console.log('🌍 [LORE TEST] Results:', loreResults);
    } catch (err) {
      setError(err.message || 'An error occurred');
      console.error('🌍 [LORE TEST] Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg shadow-md">
      <h2 className="text-lg font-bold mb-4">Lore System Debugger</h2>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600 mb-2">
          Test if keywords in your text trigger character lore entries
        </p>
        
        <div className="flex mb-2">
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter text with potential keywords..."
            className="flex-1 p-2 border rounded-md mr-2 text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600"
          />
          <button
            onClick={handleTest}
            disabled={isLoading || !activeCharacter}
            className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:bg-gray-400"
          >
            {isLoading ? 'Testing...' : 'Test Lore'}
          </button>
        </div>
        
        {activeCharacter ? (
          <div className="text-sm text-gray-600">
            Testing with character: <span className="font-medium">{activeCharacter.name}</span>
            {activeCharacter.lore_entries && (
              <span> ({activeCharacter.lore_entries.length} lore entries)</span>
            )}
          </div>
        ) : (
          <div className="text-sm text-red-500">No character selected</div>
        )}
      </div>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      {results && (
        <div className="mt-4">
          <h3 className="font-semibold mb-2">
            {results.length} Lore {results.length === 1 ? 'Entry' : 'Entries'} Triggered
          </h3>
          
          {results.length === 0 ? (
            <p className="text-gray-500 italic">No lore entries were triggered</p>
          ) : (
            <div className="space-y-3">
              {results.map((lore, index) => (
                <div key={index} className="p-3 bg-white border border-gray-200 rounded-md">
                  <div className="font-medium text-blue-700">
                    Keyword: {lore.keyword || 'Unknown'}
                  </div>
                  <div className="mt-1 text-gray-700">{lore.content}</div>
                  {lore.importance && (
                    <div className="mt-1 text-sm text-gray-500">
                      Importance: {lore.importance}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LoreDebugger;