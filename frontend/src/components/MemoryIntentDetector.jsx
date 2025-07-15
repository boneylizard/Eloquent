// MemoryIntentDetector.jsx - Enhanced version with text visibility fix
// This component provides visual feedback and explicit memory creation

import React, { useState, useEffect, useCallback } from 'react';
import { useMemory } from '../contexts/MemoryContext';

/**
 * Enhanced component that detects memory intent in input text
 * and provides a UI for creating explicit memories
 */
const MemoryIntentDetector = ({ 
  text, 
  onDetected,
  allowExplicitCreation = true
}) => {
  const memoryContext = useMemory();
  const [memoryIntent, setMemoryIntent] = useState(null);
  const [showMemoryForm, setShowMemoryForm] = useState(false);
  const [memoryData, setMemoryData] = useState({
    content: '',
    category: 'preferences',
    importance: 0.7
  });
  
  // Analyze text for memory intent when it changes
  useEffect(() => {
    if (!text || text.length < 10) {
      setMemoryIntent(null);
      return;
    }
    
    // Check for explicit memory intent with regex patterns
    const memoryPatterns = [
      /remember (?:that|this)(.*?)(?:\.|$)/i,
      /please remember(.*?)(?:\.|$)/i,
      /(?:save|store|keep) (?:this|that) (?:information|fact|preference)(.*?)(?:\.|$)/i,
      /(?:make a note|note down|take note)(.*?)(?:\.|$)/i,
      /(?:for|as a) (?:future|later) reference(.*?)(?:\.|$)/i,
      /for your (?:records|memory|information)(.*?)(?:\.|$)/i,
      /add (?:this|that) to (?:your|my) memory(.*?)(?:\.|$)/i,
      /don't forget (?:that|this)(.*?)(?:\.|$)/i,
    ];
    
    // Check for memory intent in the text
    let detectedIntent = null;
    for (const pattern of memoryPatterns) {
      const match = text.match(pattern);
      if (match) {
        // Extract the content from the match
        const extractedContent = match[1]?.trim();
        if (extractedContent && extractedContent.length > 5) {
          detectedIntent = {
            pattern: pattern.toString(),
            content: extractedContent,
            fullMatch: match[0]
          };
          break;
        }
      }
    }
    
    // Update state if we found intent
    if (detectedIntent) {
      setMemoryIntent(detectedIntent);
      setMemoryData(prev => ({
        ...prev,
        content: detectedIntent.content
      }));
      
      // Call the callback if provided
      if (onDetected) {
        onDetected(detectedIntent);
      }
    } else {
      setMemoryIntent(null);
    }
  }, [text, onDetected]);
  
  // Handle creation of explicit memory
  const createMemory = useCallback(() => {
    if (!memoryContext || !memoryData.content) return;
    
    try {
      // Add the memory using the context
      memoryContext.addMemory({
        content: memoryData.content,
        category: memoryData.category,
        importance: memoryData.importance,
        type: 'explicit', // Mark as explicitly created
        created: new Date().toISOString()
      });
      
      // Reset the form
      setShowMemoryForm(false);
      setMemoryIntent(null);
      
      // Show success message or feedback
      console.log('🧠 Memory created successfully');
      
      // Optional: Call API to sync with backend
      if (memoryContext.syncWithBackend) {
        memoryContext.syncWithBackend();
      }
    } catch (error) {
      console.error('Error creating memory:', error);
    }
  }, [memoryContext, memoryData]);
  
  // If no memory intent detected, don't render anything
  if (!memoryIntent) return null;
  
  return (
    <div className="memory-intent-container p-2 my-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-md border border-yellow-300 dark:border-yellow-700">
      {/* Memory Intent Indicator */}
      <div className="memory-intent-indicator flex items-center mb-2">
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          viewBox="0 0 20 20" 
          fill="currentColor" 
          className="w-5 h-5 mr-2 text-yellow-600 dark:text-yellow-500"
        >
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5.5a.75.75 0 001.5 0V5zm0 8a.75.75 0 00-1.5 0v.01a.75.75 0 001.5 0V13z" clipRule="evenodd" />
        </svg>
        <div>
          <p className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
            Memory intent detected
          </p>
          <p className="text-xs text-yellow-700 dark:text-yellow-400">
            Would you like to save this information to memory?
          </p>
        </div>
      </div>
      
      {/* Memory Intent Actions */}
      {!showMemoryForm ? (
        <div className="flex justify-between items-center">
          <p className="text-sm text-yellow-800 dark:text-yellow-200 truncate mr-4">
            "{memoryIntent.content}"
          </p>
          <div className="space-x-2">
            <button
              onClick={() => setShowMemoryForm(true)}
              className="px-3 py-1 text-xs bg-yellow-200 hover:bg-yellow-300 dark:bg-yellow-800 dark:hover:bg-yellow-700 text-yellow-900 dark:text-yellow-100 rounded"
            >
              Edit & Save
            </button>
            
            <button
              onClick={createMemory}
              className="px-3 py-1 text-xs bg-green-200 hover:bg-green-300 dark:bg-green-800 dark:hover:bg-green-700 text-green-900 dark:text-green-100 rounded"
            >
              Save
            </button>
            
            <button
              onClick={() => setMemoryIntent(null)}
              className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 rounded"
            >
              Ignore
            </button>
          </div>
        </div>
      ) : (
        <div className="memory-form space-y-2">
          {/* Memory Content */}
          <div>
            <label className="block text-xs font-medium text-yellow-800 dark:text-yellow-300 mb-1">
              Memory Content
            </label>
            <textarea
              value={memoryData.content}
              onChange={(e) => setMemoryData(prev => ({ ...prev, content: e.target.value }))}
              className="w-full px-2 py-1 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border border-yellow-300 dark:border-yellow-700 rounded focus:ring-2 focus:ring-yellow-500 dark:focus:ring-yellow-600"
              rows={2}
            />
          </div>
          
          {/* Memory Category */}
          <div className="flex space-x-2">
            <div className="flex-1">
              <label className="block text-xs font-medium text-yellow-800 dark:text-yellow-300 mb-1">
                Category
              </label>
              <select
                value={memoryData.category}
                onChange={(e) => setMemoryData(prev => ({ ...prev, category: e.target.value }))}
                className="w-full px-2 py-1 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border border-yellow-300 dark:border-yellow-700 rounded focus:ring-2 focus:ring-yellow-500 dark:focus:ring-yellow-600"
              >
                <option value="preferences">Preference</option>
                <option value="personal_info">Personal Info</option>
                <option value="facts">Fact</option>
                <option value="skills">Skill</option>
                <option value="experiences">Experience</option>
                <option value="projects">Project</option>
                <option value="other">Other</option>
              </select>
            </div>
            
            {/* Memory Importance */}
            <div className="flex-1">
              <label className="block text-xs font-medium text-yellow-800 dark:text-yellow-300 mb-1">
                Importance ({memoryData.importance.toFixed(1)})
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={memoryData.importance}
                onChange={(e) => setMemoryData(prev => ({ ...prev, importance: parseFloat(e.target.value) }))}
                className="w-full h-6 bg-yellow-100 dark:bg-yellow-900 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
          
          {/* Form Action Buttons */}
          <div className="flex justify-end space-x-2">
            <button
              onClick={() => setShowMemoryForm(false)}
              className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 rounded"
            >
              Cancel
            </button>
            <button
              onClick={createMemory}
              className="px-3 py-1 text-xs bg-green-200 hover:bg-green-300 dark:bg-green-800 dark:hover:bg-green-700 text-green-900 dark:text-gray-100 rounded"
            >
              Save Memory
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default MemoryIntentDetector;