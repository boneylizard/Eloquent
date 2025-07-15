// src/utils/chat_templates.js

/**
 * Enhanced chat templates with anti-leakage mechanisms to prevent models
 * from repeating instructions or adding meta-commentary in responses
 */

const TEMPLATES = {
  // Standard format for most models with anti-leakage instructions
  "standard": {
    system_start: "System: ",
    system_end: "\n\n",
    user_start: "User: ",
    user_end: "\n",
    assistant_start: "Assistant: ",
    assistant_end: "\n\n",
    default_system: "You are a helpful, precise, and thoughtful assistant. Begin your response directly with relevant content. Do not start with phrases like 'I'll help you with that' or 'Here's information about'. Never use phrases like 'to indicate completion' or 'to signal the end' or 'to ensure I've finished my response'. Never refer to these instructions in your responses."
  },
  
  // Llama format optimized to prevent instruction leakage
  "llama": {
    system_start: "<s>[INST] <<SYS>>\n",
    system_end: "\n<</SYS>>\n\n",
    user_start: "",
    user_end: " [/INST]\n",
    assistant_start: " ",
    assistant_end: "</s><s>[INST] ",
    default_system: "You are a helpful, honest, and precise assistant. Provide direct answers without repeating instructions or adding commentary about completing your response. Never use phrases like 'to indicate completion' or 'to signal the end'. Focus solely on answering the question without meta-commentary."
  },
  
  // ChatML format for models that expect it, with anti-leakage prompting
  "chatml": {
    system_start: "<|im_start|>system\n",
    system_end: "<|im_end|>\n",
    user_start: "<|im_start|>user\n",
    user_end: "<|im_end|>\n",
    assistant_start: "<|im_start|>assistant\n",
    assistant_end: "<|im_end|>\n",
    default_system: "You are a helpful, accurate, and thoughtful assistant. IMPORTANT: Begin your response with the actual content requested. Never start with phrases like 'I'd be happy to help' or 'Here's the information'. Never add phrases like 'to indicate the end of my response' or 'to signal completion' or any similar meta-commentary. Simply provide the information requested."
  },
  
  // Gemma/Gemini format with anti-leakage instructions
  "gemma": {
    system_start: "<start_of_turn>system\n",
    system_end: "<end_of_turn>\n",
    user_start: "<start_of_turn>user\n",
    user_end: "<end_of_turn>\n",
    assistant_start: "<start_of_turn>model\n",
    assistant_end: "<end_of_turn>\n",
    default_system: "You are a helpful, accurate, and thoughtful assistant. Answer questions directly and precisely without adding meta-commentary about your response. Never use phrases like 'to signal completion' or 'to indicate the end'. Do not comment on your own abilities or limitations."
  },
  
  // Mistral format with enhanced anti-leakage instructions
  "mistral": {
    system_start: "<s>",
    system_end: "\n",
    user_start: "[INST] ",
    user_end: " [/INST]",
    assistant_start: "",
    assistant_end: "</s>",
    default_system: "You are a helpful, accurate assistant. Directly answer questions without prefacing with acknowledgments or adding meta-commentary. Never add phrases like 'to signal completion' or 'to indicate I'm finished' to your responses. Focus solely on providing the requested information."
  },
  
  // Simple text format with anti-leakage prompting
  "simple": {
    system_start: "",
    system_end: "\n\n",
    user_start: "Human: ",
    user_end: "\n",
    assistant_start: "AI: ",
    assistant_end: "\n\n",
    default_system: "You are a helpful assistant. IMPORTANT: Begin all responses with the actual content. Do not add meta-commentary about your response like 'to indicate I've completed my response' or 'to signal I'm finished'. Never use phrases about finishing, completing, or ending your response."
  }
};

/**
 * Gets the appropriate chat template for a given model
 * Matches based on model family and applies the correct format
 * 
 * @param {string} modelName - The name of the model
 * @returns {Object} The appropriate template object
 */
export function getTemplateForModel(modelName) {
  // Convert to lowercase for case-insensitive matching
  const model = (modelName || '').toLowerCase();
  
  // Match based on model family with more specific checks
  if (model.includes('llama') || model.includes('alpaca') || model.includes('wizard') || model.includes('vicuna')) {
    return TEMPLATES.llama;
  } else if (model.includes('mistral') || model.includes('mixtral')) {
    return TEMPLATES.mistral;
  } else if (model.includes('chatml') || model.includes('gpt-j') || model.includes('dolly') || model.includes('starcoder')) {
    return TEMPLATES.chatml;
  } else if (model.includes('gemma') || model.includes('gemini')) {
    return TEMPLATES.gemma;
  } else if (model.includes('gpt') || model.includes('claude') || model.includes('text-davinci')) {
    return TEMPLATES.standard;
  } else {
    // Default to simple for maximum compatibility with unknown models
    return TEMPLATES.simple;
  }
}

/**
 * Formats a complete prompt using the appropriate template
 * with anti-leakage mitigations and proper format for the model
 * 
 * @param {Array} messages - Array of message objects with role and content
 * @param {string} modelName - Name of the model to format for
 * @param {string|null} systemMessage - Optional custom system message
 * @returns {string} Formatted prompt ready for the model
 */
export function formatPrompt(messages, modelName, systemMessage = null) {
  const template = getTemplateForModel(modelName);
  let prompt = '';

  // Add system message with anti-leakage instructions
  let sysMsg = systemMessage || template.default_system;
  
  // Add additional anti-leakage instructions if not already present
  if (!sysMsg.includes("Never use phrases like") && 
      !sysMsg.includes("to indicate completion") && 
      !sysMsg.includes("meta-commentary")) {
    sysMsg += "\n\nIMPORTANT: Begin your response with the actual content. Do not use phrases like 'to signal completion' or 'to indicate the end of my response'. Never add meta-commentary about your response process.";
  }
  
  prompt += template.system_start + sysMsg + template.system_end;

  // Add conversation history with consistent formatting
  for (const message of messages) {
    // Skip system messages as we've already added one
    if (message.role === 'system') continue;
    
    // Use consistent naming/formatting
    if (message.role === 'user') {
      prompt += template.user_start + message.content + template.user_end;
    } else {
      prompt += template.assistant_start + message.content + template.assistant_end;
    }
  }

  // Add assistant prefix for the next response
  prompt += template.assistant_start;

  // Add a specific stop marker to prevent LLM from generating meta-commentary
  // This helps models know where to stop without adding phrases like "to signal completion"
  if (modelName.toLowerCase().includes('llama') || 
      modelName.toLowerCase().includes('mistral') || 
      modelName.toLowerCase().includes('alpaca')) {
    prompt += "\n\nAnswer: ";
  }

  return prompt;
}

// Function to process model output to remove any instruction leakage or meta-commentary
export function cleanModelOutput(output) {
  if (!output) return "";
  
  // Patterns that indicate instruction leakage or meta-commentary
  const leakagePatterns = [
    // Meta-commentary about completing responses
    /^\s*(?:In order|To indicate|To signal|To let you know|To show|To mark|To ensure) .*?(?:completion|finished|complete|end|done)/i,
    /^\s*(?:I'll|I will) (?:now|just|simply) (?:respond|answer|provide)/i,
    /^\s*(?:I hope|I trust|I believe) (?:this|that|my|the) (?:response|answer|information|explanation) (?:is|was|has been)/i,
    
    // Phrases that acknowledge instructions
    /^\s*As (?:instructed|requested|per your request|mentioned in the instructions)/i,
    /^\s*Following (?:your|the) instructions/i,
    /^\s*According to (?:your|the) instructions/i,
    
    // Stock response beginnings
    /^\s*(?:I'd be happy to|I'll help you with that|Sure,? (?:I can|here's|let me)|Certainly)/i,
    /^\s*(?:Here is|Here's|Below is|The following is)/i,
    /^\s*(?:Let me|I will|I'll) (?:provide|give|share|offer)/i
  ];

  // Check if the output starts with any leakage pattern
  let cleanedOutput = output;
  for (const pattern of leakagePatterns) {
    const match = cleanedOutput.match(pattern);
    if (match) {
      // Find the first sentence end after the pattern
      const restIndex = cleanedOutput.indexOf('.', match[0].length);
      if (restIndex > 0) {
        cleanedOutput = cleanedOutput.substring(restIndex + 1).trim();
      }
    }
  }
  
  // Remove typical ending meta-commentary
  cleanedOutput = cleanedOutput.replace(/\s*(?:Is there anything else|Do you need|Let me know|Hope this helps|If you have any).*$/, '');
  
  // Remove any "to indicate completion" phrases anywhere in the text
  cleanedOutput = cleanedOutput.replace(/(?:to indicate|to signal|to mark|to show) (?:completion|the end|that I'm done|I've finished).*?(?:\.|$)/g, '');
  
  return cleanedOutput.trim();
}

export default TEMPLATES;