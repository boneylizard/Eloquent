// utils/openaiApi.js
// OpenAI API compatibility functions for Eloquent frontend

/**
 * Convert Eloquent messages to OpenAI chat format
 */
export function convertToOpenAIMessages(messages, systemPrompt = null) {
  const openaiMessages = [];

  // Add system message if provided
  if (systemPrompt) {
    openaiMessages.push({
      role: "system",
      content: systemPrompt
    });
  }

  // Convert existing messages
  for (const msg of messages) {
    if (msg.role === 'user') {
      openaiMessages.push({
        role: "user",
        content: msg.content
      });
    } else if (msg.role === 'bot' || msg.role === 'assistant') {
      const assistantMessage = {
        role: "assistant",
        content: msg.content || ""  // Ensure content is never null
      };

      // Preserve tool calls if they exist
      if (msg.tool_calls) {
        assistantMessage.tool_calls = msg.tool_calls;
      }

      openaiMessages.push(assistantMessage);
    }
    // Skip system messages as they're handled separately
  }

  return openaiMessages;
}

/**
 * Generate reply using OpenAI-compatible API
 */
export async function generateReplyOpenAI({
  messages,
  systemPrompt,
  model,      // Changed from modelName to model for consistency with OpenAI API
  modelName,  // Fallback support
  settings,
  apiUrl,
  apiKey = null,
  stream = true,
  targetGpuId = 0  // NEW: Add explicit GPU ID parameter
}) {
  const modelToUse = model || modelName;
  const openaiMessages = convertToOpenAIMessages(messages, systemPrompt);

  // FIXED: Use explicit targetGpuId instead of settings.gpu_id
  const validMaxTokens = (settings.max_tokens && settings.max_tokens > 0)
    ? settings.max_tokens
    : 4096; // Use a sensible default if the value is -1 or invalid.

  const payload = {
    model: modelToUse,
    messages: openaiMessages,
    temperature: settings.temperature || 0.7,
    top_p: settings.top_p || 0.9,
    top_k: settings.top_k || 40,
    repetition_penalty: settings.repetition_penalty || 1.1,
    max_tokens: validMaxTokens, // Use the corrected, valid value
    stream: stream,
    gpu_id: targetGpuId
  };

  // Add stop sequences if configured
  if (settings.stop_sequences && settings.stop_sequences.length > 0) {
    payload.stop = settings.stop_sequences;
  }

  // Build headers
  const headers = {
    'Content-Type': 'application/json'
  };

  // Add authorization header if API key is provided
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  console.log("🌐 [OpenAI API] Sending request:");
  console.log("🌐 [OpenAI API] URL:", `${apiUrl}/v1/chat/completions`);
  console.log("🌐 [OpenAI API] Model:", modelToUse);
  console.log("🌐 [OpenAI API] GPU ID:", targetGpuId);
  console.log("🌐 [OpenAI API] Payload GPU ID:", payload.gpu_id);

  const response = await fetch(`${apiUrl}/v1/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status}`);
  }

  return response;
}

/**
 * Process OpenAI streaming response
 */
export async function processOpenAIStream(response, onToken, onComplete, onError) {
  try {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();

          if (data === '[DONE]') {
            onComplete(accumulatedText);
            return accumulatedText;
          }

          try {
            const parsed = JSON.parse(data);

            // Check for error response from backend
            if (parsed.error) {
              const errorMsg = parsed.error.message || 'Unknown API error';
              console.error('[OpenAI Stream] API Error:', parsed.error);
              onError(new Error(errorMsg));
              return accumulatedText;
            }

            const content = parsed.choices?.[0]?.delta?.content;

            if (content) {
              accumulatedText += content;
              onToken(content, accumulatedText);
            }

            // Check for finish_reason
            const finishReason = parsed.choices?.[0]?.finish_reason;
            if (finishReason === 'stop') {
              onComplete(accumulatedText);
              return accumulatedText;
            }
          } catch (e) {
            // Skip invalid JSON chunks
            continue;
          }
        }
      }
    }

    onComplete(accumulatedText);
    return accumulatedText;

  } catch (error) {
    onError(error);
    throw error;
  }
}

/**
 * Non-streaming OpenAI API call
 */
export async function generateReplyOpenAINonStreaming({
  messages,
  systemPrompt,
  model,      // Changed from modelName to model
  modelName,  // Fallback support
  settings,
  apiUrl,
  apiKey = null,
  targetGpuId = 0  // NEW: Add explicit GPU ID parameter
}) {
  const modelToUse = model || modelName;
  const response = await generateReplyOpenAI({
    messages,
    systemPrompt,
    model: modelToUse,
    settings,
    apiUrl,
    apiKey,
    stream: false,
    targetGpuId: targetGpuId  // FIXED: Pass through the GPU ID
  });

  const result = await response.json();
  return result.choices?.[0]?.message?.content || "[No response]";
}

/**
 * Get available models from OpenAI endpoint
 */
export async function getOpenAIModels(apiUrl, apiKey = null) {
  try {
    const headers = {};
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const response = await fetch(`${apiUrl}/v1/models`, { headers });
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }

    const data = await response.json();
    return data.data || [];
  } catch (error) {
    console.error('Error fetching OpenAI models:', error);
    return [];
  }
}

/**
 * Test OpenAI API connection
 */
export async function testOpenAIConnection(apiUrl, apiKey = null) {
  try {
    const headers = {};
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const response = await fetch(`${apiUrl}/v1/health`, { headers });
    return response.ok;
  } catch (error) {
    console.error('OpenAI API connection test failed:', error);
    return false;
  }
}