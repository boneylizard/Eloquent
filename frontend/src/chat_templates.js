// src/utils/chat_templates.js

const TEMPLATES = {
  // Ultra simple format - minimal markers
  "minimal": {
    system_start: "### System:\n",
    system_end: "\n\n",
    user_start: "### User:\n",
    user_end: "\n",
    assistant_start: "### Assistant:\n",
    assistant_end: "\n\n",
    default_system: "You are a helpful assistant chatting with the user. Answer only what the user has actually asked."
  }
};

/**
 * Gets the appropriate chat template for a given model
 */
export function getTemplateForModel(modelName) {
  // Always use minimal format
  return TEMPLATES.minimal;
}

/**
 * Formats a complete prompt using the appropriate template
 */
export function formatPrompt(messages, modelName, systemMessage = null) {
  console.log("🔍 [DEBUG] Raw messages being formatted:", JSON.stringify(messages));
  const template = getTemplateForModel(modelName);
  let prompt = '';

  // Add system message
  const sysMsg = systemMessage || template.default_system;
  prompt += template.system_start + sysMsg + template.system_end;

  // Add just the last few messages to avoid confusion
  const recentMessages = messages.slice(-4); // Only use the most recent 4 messages
  
  for (const message of recentMessages) {
    // Skip system messages
    if (message.role === 'system') continue;
    
    if (message.role === 'user') {
      prompt += template.user_start + message.content + template.user_end;
    } else {
      prompt += template.assistant_start + message.content + template.assistant_end;
    }
  }

  // Add assistant prefix for the next response
  prompt += template.assistant_start;

  return prompt;
}

export default TEMPLATES;