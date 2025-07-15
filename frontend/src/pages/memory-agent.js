// memory-agent.js
function getBackendUrl() {
  try {
    const settings = JSON.parse(localStorage.getItem('LiangLocal-settings') || '{}');
    const isSingleGpuMode = settings.singleGpuMode === true;
    return isSingleGpuMode ? 'http://localhost:8000' : 'http://localhost:8001';
  } catch (error) {
    console.warn('Could not read GPU mode from settings, defaulting to dual GPU mode');
    return 'http://localhost:8001';
  }
}
// This function will be called before sending prompts to GPU0
async function enhancePromptWithMemory(userPrompt) {
  try {
    // Call GPU1 endpoint to get relevant memories
    const response = await fetch(`${getBackendUrl()}/api/memory/relevant`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt: userPrompt }),
    });
    
    const data = await response.json();
    
    // If we have relevant memories, enhance the prompt
    if (data.memories && data.memories.length > 0) {
      const memoryContext = formatMemories(data.memories);
      return `${memoryContext}\n\nUser: ${userPrompt}`;
    }
    
    // Otherwise return original prompt
    return userPrompt;
  } catch (error) {
    console.error('Memory enhancement failed:', error);
    // Fall back to original prompt if anything goes wrong
    return userPrompt;
  }
}

// This function will be called after receiving response from GPU0
async function observeConversation(userPrompt, aiResponse) {
  try {
    // Call GPU1 endpoint to analyze conversation for memory creation
    await fetch(`${getBackendUrl()}/api/memory/observe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        prompt: userPrompt,
        response: aiResponse 
      }),
    });
    
    // No need to wait for or process the response
  } catch (error) {
    console.error('Memory observation failed:', error);
    // Fail silently - doesn't affect the main conversation
  }
}

// Helper function to format memories into a string
function formatMemories(memories) {
  const formattedMemories = memories.map(memory => 
    `- ${memory.content}`
  ).join('\n');
  
  return `<relevant_context>\n${formattedMemories}\n</relevant_context>`;
}

module.exports = {
  enhancePromptWithMemory,
  observeConversation
};