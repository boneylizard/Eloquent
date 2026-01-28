// Save this file as: src/components/LoreDebugger.jsx

import React, { useState } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Input } from './ui/input';

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
    <div className="p-4 bg-card text-card-foreground rounded-lg border border-border shadow-sm">
      <h2 className="text-lg font-bold mb-4">Lore System Debugger</h2>
      
      <div className="mb-4">
        <p className="text-sm text-muted-foreground mb-2">
          Test if keywords in your text trigger character lore entries
        </p>
        
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-3 mb-2">
          <Input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            placeholder="Enter text with potential keywords..."
            className="flex-1"
          />
          <Button
            onClick={handleTest}
            disabled={isLoading || !activeCharacter}
            className="w-full md:w-auto"
          >
            {isLoading ? 'Testing...' : 'Test Lore'}
          </Button>
        </div>
        
        {activeCharacter ? (
          <div className="text-sm text-muted-foreground">
            Testing with character: <span className="font-medium">{activeCharacter.name}</span>
            {activeCharacter.lore_entries && (
              <span> ({activeCharacter.lore_entries.length} lore entries)</span>
            )}
          </div>
        ) : (
          <div className="text-sm text-destructive">No character selected</div>
        )}
      </div>
      
      {error && (
        <div className="mb-4 p-3 border border-destructive/40 bg-destructive/10 text-destructive rounded-md">
          {error}
        </div>
      )}
      
      {results && (
        <div className="mt-4">
          <h3 className="font-semibold mb-2">
            {results.length} Lore {results.length === 1 ? 'Entry' : 'Entries'} Triggered
          </h3>
          
          {results.length === 0 ? (
            <p className="text-muted-foreground italic">No lore entries were triggered</p>
          ) : (
            <div className="space-y-3">
              {results.map((lore, index) => (
                <div key={index} className="p-3 bg-muted/40 border border-border rounded-md">
                  <div className="font-medium text-foreground">
                    Keyword: {lore.keyword || 'Unknown'}
                  </div>
                  <div className="mt-1 text-muted-foreground">{lore.content}</div>
                  {lore.importance && (
                    <div className="mt-1 text-sm text-muted-foreground">
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
