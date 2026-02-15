import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { ScrollArea } from './ui/scroll-area';
import { Badge } from './ui/badge';
import { Card } from './ui/card';
import { Textarea } from './ui/textarea';
import { Send, Bot, User, Users, Loader2, MessageSquare, BarChart3 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkSoftBreaks from '@/utils/remarkSoftBreaks';
import remarkDialogueQuotes from '@/utils/remarkDialogueQuotes';
import Settings from './Settings';

const AnalysisChat = ({ 
  testResults, 
  modelPurposes,
  fetchModelsByPurpose,
  loadModelForPurpose,
  unloadModelPurpose,
  isLoadingPurpose,
  isUnloadingPurpose,
  primaryApiUrl, 
  secondaryApiUrl,
  setTestResults,
  ragSettings
}) => {
  const [chatMode, setChatMode] = useState(''); // Will be set based on available models
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [contextInitialized, setContextInitialized] = useState(false);
  const messagesEndRef = useRef(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const lastQuestionMessageRef = useRef(null);
  const importFileRef = useRef(null);
  const prevSelectedResultId = useRef(null);
  const [selectedPerspectiveId, setSelectedPerspectiveId] = useState('custom');
  const [customPerspectiveName, setCustomPerspectiveName] = useState('');
  const [showPerspectiveEditor, setShowPerspectiveEditor] = useState(false);
  const isApiModel = (modelId) => modelId && modelId.startsWith('endpoint-');
  const [autoGenerateQuestions, setAutoGenerateQuestions] = useState(() => {
  try {
    const saved = localStorage.getItem('Eloquent-analysis-auto-questions');
    return saved ? JSON.parse(saved) : true; // Default to true
  } catch {
    return true;
  }
});
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

// Save to sessionStorage when state changes
useEffect(() => {
  if (messages.length > 0) {
    const analysisState = {
      messages: messages,
      chatMode: chatMode,
      selectedResultId: selectedResult?.id,
      testResultsCount: testResults.length,
      autoGenerateQuestions: autoGenerateQuestions,
      suggestedQuestions: suggestedQuestions,
      prevSelectedResultId: prevSelectedResultId.current,
      timestamp: Date.now()
    };
    
    try {
      sessionStorage.setItem('temp-analysis-conversation', JSON.stringify(analysisState));
    } catch (error) {
      console.error('Failed to save analysis conversation:', error);
    }
  }
}, [messages, chatMode, selectedResult, testResults, autoGenerateQuestions, suggestedQuestions]);

// Load from sessionStorage on component mount - with auto-clear on new tests
useEffect(() => {
  try {
    const saved = sessionStorage.getItem('temp-analysis-conversation');
    if (saved) {
      const analysisState = JSON.parse(saved);
      
      // Check if new test results have been added since last session
      const savedTestCount = analysisState.testResultsCount || 0;
      const currentTestCount = testResults.length;
      
      if (currentTestCount > savedTestCount) {
        // New tests detected - clear the session instead of restoring
        console.log('🗑️ New test results detected, starting fresh analysis session');
        sessionStorage.removeItem('temp-analysis-conversation');
        return;
      }
      
      // No new tests - restore the session normally
      setMessages(analysisState.messages || []);
      const savedChatMode = analysisState.chatMode;
if (savedChatMode && savedChatMode !== 'test' && savedChatMode !== 'judge' && savedChatMode !== 'both') {
  setChatMode(savedChatMode);
}
      setSuggestedQuestions(analysisState.suggestedQuestions || []);
      prevSelectedResultId.current = analysisState.prevSelectedResultId || null;
      
      if (analysisState.selectedResultId) {
        const result = testResults.find(r => r.id === analysisState.selectedResultId);
        if (result) {
          setSelectedResult(result);
        }
      }
    }
  } catch (error) {
    console.error('Failed to restore analysis conversation:', error);
  }
}, [testResults]);

useEffect(() => {
  if (!modelPurposes) return;

  const isCurrentModeInvalid = chatMode && !modelPurposes[chatMode];

  // If the current chat mode is now invalid (e.g., model was unloaded),
  // or if no chat mode is set, find a sensible new default.
  if (isCurrentModeInvalid || !chatMode) {
    const preferredOrder = [
      'test_model_a', 
      'test_model_b', 
      'test_model', 
      'primary_judge', 
      'secondary_judge'
    ];
    
    const firstAvailable = preferredOrder.find(p => modelPurposes[p]);
    
    setChatMode(firstAvailable || ''); // Set to the first available model, or clear if none
  }
}, [modelPurposes]); // This effect should ONLY react to changes in the loaded models

const getRoleNameFromPurpose = (purposeKey) => {
  const names = {
    'test_model': 'Test Model',
    'test_model_a': 'Test Model A',
    'test_model_b': 'Test Model B',
    'primary_judge': 'Primary Judge',
    'secondary_judge': 'Secondary Judge'
  };
  return names[purposeKey] || 'Model'; // Default fallback
};

  const initializeContext = (result) => {
    let contextMessage = '';
    
    if (result.model && !result.model1) {
      // Single model test
      contextMessage = `Recent test context:

**Prompt:** ${result.prompt}
**Category:** ${result.category}
**Model Response:** ${result.response}
**Score:** ${result.score}/100
**Judge Feedback:** ${result.feedback}

You can now discuss this test with me. What would you like to explore about this interaction?`;
    } else {
      // Comparison test
      contextMessage = `Recent comparison test context:

**Prompt:** ${result.prompt}
**Category:** ${result.category}

**${result.model1} Response:** ${result.response1}

**${result.model2} Response:** ${result.response2}

**Judgment:** ${result.judgment} (${result.explanation})

You can now discuss this comparison with me. What would you like to explore?`;
    }

    setMessages([{
      id: Date.now(),
      role: 'system',
      content: contextMessage,
      timestamp: new Date().toISOString()
    }]);
  };

const generateApiResponse = useCallback(async (modelName, prompt, apiUrl) => {
    if (!apiUrl) {
        throw new Error('No API URL available for this request. Check your primary/secondary API URL settings.');
    }

    // Route API models through the backend OpenAI-compat layer
    const baseUrl = apiUrl.replace(/\/$/, '');

    const payload = {
        model: modelName, // Use endpoint id; backend will map to configured model
        messages: [
            { role: "system", content: "You are a helpful and direct AI assistant." },
            { role: "user", content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 1024,
        stream: true  // AnalysisChat uses streaming
    };

    const headers = { 'Content-Type': 'application/json' };

    console.log(`🌐 [AnalysisChat API] Sending request to: ${baseUrl}/v1/chat/completions`);
    console.log(`🌐 [AnalysisChat API] Model: ${modelName}`);

    try {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API Error ${response.status}: ${errorBody}`);
        }

        return response; // Return the streaming response for AnalysisChat to handle
    } catch (error) {
        console.error(`Error generating API response from ${modelName}:`, error);
        throw error;
    }
}, []);

const generateResponse = async (purposeKey, prompt) => {
  console.log('generateResponse called with purpose:', purposeKey);
  console.log('🔍 Prompt length being sent:', prompt.length, 'characters');
  
  const purposeInfo = modelPurposes[purposeKey];
  if (!purposeInfo) {
    throw new Error(`No model loaded for purpose: ${purposeKey}`);
  }
  
  const { name: modelName, gpu_id: gpuId } = purposeInfo;
  
  // Check if this is an API endpoint
  if (isApiModel(modelName)) {
    console.log(`🌐 [AnalysisChat] Using API endpoint: ${modelName}`);
    const apiUrl = gpuId === 0 ? primaryApiUrl : secondaryApiUrl;
    return await generateApiResponse(modelName, prompt, apiUrl);
  }
  
  // Original local model logic
  const apiUrl = gpuId === 0 ? primaryApiUrl : secondaryApiUrl;
  console.log(`Using model: ${modelName} on GPU ${gpuId} via ${apiUrl}`);
  
  try {
    // First, check if the model is already loaded (only for local models)
    const loadedModelsResponse = await fetch(`${apiUrl}/models/loaded`);
    if (loadedModelsResponse.ok) {
      const { loaded_models } = await loadedModelsResponse.json();
      const isModelLoaded = loaded_models.some(m => m.name === modelName);
      
      if (!isModelLoaded) {
        console.log(`Model ${modelName} not loaded, loading now...`);
        const loadResponse = await fetch(`${apiUrl}/models/load/${modelName}?gpu_id=${gpuId}&context_length=${ragSettings.contextLength}`, {
          method: 'POST'
        });
        if (!loadResponse.ok) {
          throw new Error(`Failed to load model ${modelName}`);
        }
        // Wait a moment for loading to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
      } else {
        console.log(`Model ${modelName} already loaded, proceeding...`);
      }
    }

    const response = await fetch(`${apiUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        model_name: modelName,
        temperature: 0.7,
        max_tokens: 1024,
        stream: true,
        request_purpose: 'model_testing',
        userProfile: {}, // Empty to avoid memory interference
        memoryEnabled: false,
        use_rag: ragSettings?.use_rag || false,
        rag_docs: ragSettings?.selectedDocuments || []
      })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response;
  } catch (error) {
    console.error(`Error generating response from ${modelName}:`, error);
    throw error;
  }
};

  const formatModelName = (name) => {
    if (!name) return "Model";
    return name.split('/').pop().replace(/\.(bin|gguf)$/i, '');
  };

const clearAnalysisSession = () => {
  if (window.confirm('Start a new analysis session? This will clear the current conversation.')) {
    // COMPLETELY PURGE analysis storage
    sessionStorage.removeItem('temp-analysis-conversation');
    localStorage.removeItem('temp-analysis-conversation');
    localStorage.removeItem('analysis-last-activity');
    
    // Clear any other analysis-related storage keys
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('temp-analysis') || key.startsWith('analysis-')) {
        localStorage.removeItem(key);
      }
    });
    
    // Clear current state
    setMessages([]);
    setSelectedResult(null);
    setChatMode('test');
    setSuggestedQuestions([]); // Add this line
    
    console.log('🗑️ Analysis storage completely purged');
  }
};

const handleDualModelChat = async (purposeKeys, userMessage, conversationHistory) => {
  const [firstPurpose, secondPurpose] = purposeKeys;
  
  // First model response
  const firstPurposeInfo = modelPurposes[firstPurpose];
  const firstRoleName = getRoleNameFromPurpose(firstPurpose);
  const firstModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n[You are ${firstRoleName}. Respond from your perspective. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${firstRoleName}:`;
  const firstModelResponse = await generateResponse(firstPurpose, firstModelPrompt);
  
  const firstMsgId = Date.now() + 1;
  setMessages(prev => [...prev, {
    id: firstMsgId,
    role: 'assistant',
    content: '',
    model: firstPurposeInfo.name,
    modelType: firstPurpose,
    timestamp: new Date().toISOString()
  }]);

  // Stream first response and wait for completion
  await streamResponse(firstModelResponse, firstMsgId);
  
  // Get the completed first response for second model context
  let firstResponseContent = '';
  setMessages(prev => {
    const firstMsg = prev.find(m => m.id === firstMsgId);
    if (firstMsg) firstResponseContent = firstMsg.content;
    return prev;
  });

  // Small delay for UX
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Second model response with awareness of first model's response
  const secondPurposeInfo = modelPurposes[secondPurpose];
  const secondRoleName = getRoleNameFromPurpose(secondPurpose);
  const secondModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n[You are ${secondRoleName}. You can see ${firstRoleName}'s response above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${secondRoleName}:`;

  const secondModelResponse = await generateResponse(secondPurpose, secondModelPrompt);
  
  const secondMsgId = Date.now() + 2;
  setMessages(prev => [...prev, {
    id: secondMsgId,
    role: 'assistant', 
    content: '',
    model: secondPurposeInfo.name,
    modelType: secondPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(secondModelResponse, secondMsgId);
};
const handleTripleModelChat = async (purposeKeys, userMessage, conversationHistory) => {
  const [firstPurpose, secondPurpose, thirdPurpose] = purposeKeys;
  
  // First model response
  const firstPurposeInfo = modelPurposes[firstPurpose];
  const firstRoleName = getRoleNameFromPurpose(firstPurpose);
  const firstModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n[You are the ${firstRoleName} model. Respond from your perspective.]\n\n${firstRoleName}:`;

  const firstModelResponse = await generateResponse(firstPurpose, firstModelPrompt);
  
  const firstMsgId = Date.now() + 1;
  setMessages(prev => [...prev, {
    id: firstMsgId,
    role: 'assistant',
    content: '',
    model: firstPurposeInfo.name,
    modelType: firstPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(firstModelResponse, firstMsgId);
  
  // Get first response content
  let firstResponseContent = '';
  setMessages(prev => {
    const firstMsg = prev.find(m => m.id === firstMsgId);
    if (firstMsg) firstResponseContent = firstMsg.content;
    return prev;
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  // Second model response
  const secondPurposeInfo = modelPurposes[secondPurpose];
  const secondRoleName = getRoleNameFromPurpose(secondPurpose);
const secondModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n[You are ${secondRoleName}. You can see ${firstRoleName}'s response above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${secondRoleName}:`;
  const secondModelResponse = await generateResponse(secondPurpose, secondModelPrompt);
  
  const secondMsgId = Date.now() + 2;
  setMessages(prev => [...prev, {
    id: secondMsgId,
    role: 'assistant',
    content: '',
    model: secondPurposeInfo.name,
    modelType: secondPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(secondModelResponse, secondMsgId);

  // Get second response content
  let secondResponseContent = '';
  setMessages(prev => {
    const secondMsg = prev.find(m => m.id === secondMsgId);
    if (secondMsg) secondResponseContent = secondMsg.content;
    return prev;
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  // Third model response
  const thirdPurposeInfo = modelPurposes[thirdPurpose];
  const thirdRoleName = getRoleNameFromPurpose(thirdPurpose);
const thirdModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n${secondRoleName}: ${secondResponseContent}\n\n[You are ${thirdRoleName}. You can see both previous responses above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${thirdRoleName}:`;
  const thirdModelResponse = await generateResponse(thirdPurpose, thirdModelPrompt);
  
  const thirdMsgId = Date.now() + 3;
  setMessages(prev => [...prev, {
    id: thirdMsgId,
    role: 'assistant',
    content: '',
    model: thirdPurposeInfo.name,
    modelType: thirdPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(thirdModelResponse, thirdMsgId);
};
const handleQuadModelChat = async (purposeKeys, userMessage, conversationHistory) => {
  const [firstPurpose, secondPurpose, thirdPurpose, fourthPurpose] = purposeKeys;
  
  // First model response
  const firstPurposeInfo = modelPurposes[firstPurpose];
      const firstRoleName = getRoleNameFromPurpose(firstPurpose);
      const firstModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n[You are ${firstRoleName}. Respond from your perspective. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${firstRoleName}:`;

  const firstModelResponse = await generateResponse(firstPurpose, firstModelPrompt);
  
  const firstMsgId = Date.now() + 1;
  setMessages(prev => [...prev, {
    id: firstMsgId,
    role: 'assistant',
    content: '',
    model: firstPurposeInfo.name,
    modelType: firstPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(firstModelResponse, firstMsgId);
  
  // Get first response content
  let firstResponseContent = '';
  setMessages(prev => {
    const firstMsg = prev.find(m => m.id === firstMsgId);
    if (firstMsg) firstResponseContent = firstMsg.content;
    return prev;
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  // Second model response
  const secondPurposeInfo = modelPurposes[secondPurpose];
  const secondRoleName = getRoleNameFromPurpose(secondPurpose);
const secondModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n[You are ${secondRoleName}. You can see ${firstRoleName}'s response above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${secondRoleName}:`;

  const secondModelResponse = await generateResponse(secondPurpose, secondModelPrompt);
  
  const secondMsgId = Date.now() + 2;
  setMessages(prev => [...prev, {
    id: secondMsgId,
    role: 'assistant',
    content: '',
    model: secondPurposeInfo.name,
    modelType: secondPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(secondModelResponse, secondMsgId);

  // Get second response content
  let secondResponseContent = '';
  setMessages(prev => {
    const secondMsg = prev.find(m => m.id === secondMsgId);
    if (secondMsg) secondResponseContent = secondMsg.content;
    return prev;
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  // Third model response
  const thirdPurposeInfo = modelPurposes[thirdPurpose];
  const thirdRoleName = getRoleNameFromPurpose(thirdPurpose);
const thirdModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n${secondRoleName}: ${secondResponseContent}\n\n[You are ${thirdRoleName}. You can see both previous responses above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${thirdRoleName}:`;

  const thirdModelResponse = await generateResponse(thirdPurpose, thirdModelPrompt);
  
  const thirdMsgId = Date.now() + 3;
  setMessages(prev => [...prev, {
    id: thirdMsgId,
    role: 'assistant',
    content: '',
    model: thirdPurposeInfo.name,
    modelType: thirdPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(thirdModelResponse, thirdMsgId);

  // Get third response content
  let thirdResponseContent = '';
  setMessages(prev => {
    const thirdMsg = prev.find(m => m.id === thirdMsgId);
    if (thirdMsg) thirdResponseContent = thirdMsg.content;
    return prev;
  });

  await new Promise(resolve => setTimeout(resolve, 1000));

  // Fourth model response
  const fourthPurposeInfo = modelPurposes[fourthPurpose];
  const fourthRoleName = getRoleNameFromPurpose(fourthPurpose);
const fourthModelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n${firstRoleName}: ${firstResponseContent}\n\n${secondRoleName}: ${secondResponseContent}\n\n${thirdRoleName}: ${thirdResponseContent}\n\n[You are ${fourthRoleName}. You can see all three previous responses above. Engage in analysis or critique. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${fourthRoleName}:`;
  const fourthModelResponse = await generateResponse(fourthPurpose, fourthModelPrompt);
  
  const fourthMsgId = Date.now() + 4;
  setMessages(prev => [...prev, {
    id: fourthMsgId,
    role: 'assistant',
    content: '',
    model: fourthPurposeInfo.name,
    modelType: fourthPurpose,
    timestamp: new Date().toISOString()
  }]);

  await streamResponse(fourthModelResponse, fourthMsgId);
};
const handleSendMessage = async (predefinedMessage = null) => {
  // ADD DEBUGGING TO SEE WHAT WE'RE GETTING
  console.log('handleSendMessage called with:', { predefinedMessage, inputValue });
  console.log('predefinedMessage type:', typeof predefinedMessage);
  console.log('inputValue type:', typeof inputValue);

  // Ensure we always get a string with better error handling
  let rawMessage;
  if (predefinedMessage !== null) {
    // If predefinedMessage is provided, use it but ensure it's a string
    if (typeof predefinedMessage === 'string') {
      rawMessage = predefinedMessage;
    } else {
      console.error('predefinedMessage is not a string:', predefinedMessage);
      rawMessage = String(predefinedMessage); // This will show [object Object] if it's an object
    }
  } else {
    rawMessage = inputValue || '';
  }
  
  const messageText = String(rawMessage).trim();
  console.log('Final messageText:', messageText);
  
  if (!messageText || isGenerating) return;

  const userMessage = {
    id: Date.now(),
    role: 'user',
    content: messageText, // Now guaranteed to be a string
    timestamp: new Date().toISOString()
  };

  setMessages(prev => [...prev, userMessage]);
  setInputValue(''); // Clear input even if using predefined message
  setIsGenerating(true);

try {
  // Store aged-out messages if conversation is getting long
  const CONTEXT_WINDOW = 6; // Increased to maintain better context
  const allMessages = [...messages, userMessage];
  
  if (allMessages.length > CONTEXT_WINDOW) {
    const messagesToAge = allMessages.slice(0, allMessages.length - CONTEXT_WINDOW);
    await storeAgedOutMessages(messagesToAge, selectedResult?.category || 'analysis');
  }

  // Build conversation history for context using ONLY recent messages
 const conversationHistory = allMessages.map(msg => {
    // Ensure content is always a string
    const content = typeof msg.content === 'string' ? msg.content : String(msg.content || '');

    if (msg.role === 'user') return `Human: ${content}`;
    if (msg.role === 'system') return `Context: ${content}`;
    if (msg.role === 'error') return `Error: ${content}`;
    
    // For assistant messages, label them by their role in the test
    if (msg.role === 'assistant') {
      const roleName = getRoleNameFromPurpose(msg.modelType); // Use the new helper
      return `${roleName}: ${content}`;
    }
    
    return `Assistant: ${content}`;
  }).join('\n\n');


// Handle different chat modes with purpose-based system
if (chatMode.includes('_and_') || chatMode.startsWith('both_') || chatMode === 'all_models') {
  // Multi-model modes
  if (chatMode === 'test_and_primary') {
    await handleDualModelChat(['test_model', 'primary_judge'], userMessage, conversationHistory);
  } else if (chatMode === 'test_and_secondary') {
    await handleDualModelChat(['test_model', 'secondary_judge'], userMessage, conversationHistory);
  } else if (chatMode === 'test_and_both_judges') {
    await handleTripleModelChat(['test_model', 'primary_judge', 'secondary_judge'], userMessage, conversationHistory);
  } else if (chatMode === 'both_test_models') {
    await handleDualModelChat(['test_model_a', 'test_model_b'], userMessage, conversationHistory);
  } else if (chatMode === 'both_judges') {
    await handleDualModelChat(['primary_judge', 'secondary_judge'], userMessage, conversationHistory);
  } else if (chatMode === 'all_models') {
    await handleQuadModelChat(['test_model_a', 'test_model_b', 'primary_judge', 'secondary_judge'], userMessage, conversationHistory);
  }
} else {
  // Single model modes
  const purposeInfo = modelPurposes[chatMode];
  if (!purposeInfo) {
    throw new Error(`No model loaded for chat mode: ${chatMode}`);
  }
  
  const roleName = getRoleNameFromPurpose(chatMode);
  const modelPrompt = `${conversationHistory}\n\nHuman: ${userMessage.content}\n\n[You are ${roleName}. Respond as yourself, focusing on the user's question and the context above. Provide a direct, complete, and self-contained response. Do not ask follow-up questions, solicit the next topic, or try to continue the conversation.]\n\n${roleName}:`;
  const modelResponse = await generateResponse(chatMode, modelPrompt);

  const msgId = Date.now() + 1;

  setMessages(prev => [
    ...prev,
    {
      id: msgId,
      role: 'assistant',
      content: '',
      model: purposeInfo.name,
      modelType: chatMode,
      timestamp: new Date().toISOString()
    }
  ]);

  await streamResponse(modelResponse, msgId);
}
  } catch (error) {
    console.error('Chat error:', error);
    setMessages(prev => [...prev, {
      id: Date.now(),
      role: 'error',
      content: `Error: ${error.message}`,
      timestamp: new Date().toISOString()
    }]);
  } finally {
    setIsGenerating(false);
  }
};
const storeAgedOutMessages = useCallback(async (messagesToStore, topic) => {
  try {
    for (const msg of messagesToStore) {
      await fetch(`${primaryApiUrl}/conversation/store`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speaker: msg.role === 'user' ? 'Human' : formatModelName(msg.model) || 'Assistant',
          content: msg.content,
          topic: topic,
          conversation_id: selectedResult?.id || 'analysis'
        })
      });
    }
    console.log(`Stored ${messagesToStore.length} aged-out messages to RAG`);
  } catch (error) {
    console.error('Failed to store conversation chunks:', error);
  }
}, [primaryApiUrl, selectedResult, formatModelName]);
// Add these functions to AnalysisChat component
const exportAnalysis = () => {
  const analysisExport = {
    // Include the conversation
    messages: messages,
    selectedResultId: selectedResult?.id,
    chatMode: chatMode,
    
    // Include ALL test results so analysis makes sense
    testResults: testResults,
    
// Include model info
modelPurposes: modelPurposes,
    
    exportDate: new Date().toISOString(),
    messageCount: messages.length
  };
  
  const blob = new Blob([JSON.stringify(analysisExport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `analysis-session-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
};

const importAnalysis = (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const imported = JSON.parse(e.target.result);
      
      // Import test results with enhanced dual judge support
      if (imported.testResults) {
        const resultsWithDualJudgeSupport = imported.testResults.map((result, index) => ({
          ...result,
          id: result.id || `imported-${Date.now()}-${index}`, // Ensure unique ID
          model: result.testModel || result.model, // Handle field name differences
          
          // ENHANCED DUAL JUDGE COMPATIBILITY
          secondaryJudgeModel: result.secondaryJudgeModel || null,
          isDualJudged: result.isDualJudged || false,
          
          // Handle dual judge structures
          primaryJudgment: result.primaryJudgment || null,
          secondaryJudgment: result.secondaryJudgment || null,
          reconciliationReason: result.reconciliationReason || null,
          
          // Backwards compatibility - ensure main fields exist
          score: result.score || (result.primaryJudgment?.score) || 50,
          feedback: result.feedback || (result.primaryJudgment?.feedback) || 'No feedback available',
          judgment: result.judgment || (result.primaryJudgment?.verdict) || 'TIE',
          explanation: result.explanation || (result.primaryJudgment?.explanation) || 'No explanation available',
          
          // Preserve all other fields
          prompt: result.prompt,
          category: result.category,
          judgeModel: result.judgeModel,
          timestamp: result.timestamp,
          
          // Single model test fields
          ...(result.response && { response: result.response }),
          
          // Comparison test fields  
          ...(result.model1 && {
            model1: result.model1,
            model2: result.model2,
            response1: result.response1,
            response2: result.response2
          })
        }));
        
        setTestResults(resultsWithDualJudgeSupport);
      }
      
      // Import model purposes info (for reference, can't auto-reload models)
      if (imported.modelPurposes) {
        console.log('Imported analysis included these models:', imported.modelPurposes);
        // Could show a message to user about which models were used
      }
      
      // Legacy support for judgeModel field
      if (imported.judgeModel) {
        console.log('Legacy judge model found:', imported.judgeModel);
      }
      
      // Restore the analysis conversation
      if (imported.messages) {
        setMessages(imported.messages);
        setChatMode(imported.chatMode || 'test_model'); // Updated default
        
        const result = imported.testResults?.find(r => r.id === imported.selectedResultId);
        if (result) {
          setSelectedResult(result);
        }
      }
      
      // Generate detailed import summary
      const singleModelResults = imported.testResults?.filter(r => r.model && !r.model1)?.length || 0;
      const comparisonResults = imported.testResults?.filter(r => r.model1 && r.model2)?.length || 0;
      const dualJudgedResults = imported.testResults?.filter(r => r.isDualJudged)?.length || 0;
      
      let importMessage = `Restored ${imported.testResults?.length || 0} test results and ${imported.messages?.length || 0} analysis messages`;
      if (singleModelResults > 0) importMessage += `\n• Single model tests: ${singleModelResults}`;
      if (comparisonResults > 0) importMessage += `\n• Comparison tests: ${comparisonResults}`;
      if (dualJudgedResults > 0) importMessage += `\n• Dual-judged results: ${dualJudgedResults}`;
      
      alert(importMessage);
    } catch (error) {
      alert(`Import failed: ${error.message}`);
    }
  };
  reader.readAsText(file);
  event.target.value = '';
};
const generateSuggestedQuestions = useCallback(async (isInitial = false) => {
  if (!selectedResult || isGeneratingQuestions) return;
  
  setIsGeneratingQuestions(true);
  
  try {
    // Prefer test_model_a, then test_model, then judges as fallback
    let questionPurpose = null;
    if (modelPurposes?.test_model_a) {
      questionPurpose = 'test_model_a';
    } else if (modelPurposes?.test_model) {
      questionPurpose = 'test_model';
    } else if (modelPurposes?.primary_judge) {
      questionPurpose = 'primary_judge';
    } else if (modelPurposes?.secondary_judge) {
      questionPurpose = 'secondary_judge';
    }
    
    if (!questionPurpose) return;
    
    const questionPurposeInfo = modelPurposes[questionPurpose];
    const questionModel = questionPurposeInfo.name;
    const questionApiUrl = questionPurposeInfo.gpu_id === 0 ? primaryApiUrl : secondaryApiUrl;
    
    if (!questionModel) return;
    
    // Build the prompt
    let prompt = '';
    
    if (isInitial) {
      // Initial questions based on test result
      if (selectedResult.model && !selectedResult.model1) {
        // Single model test
        prompt = `Based on this test result, generate 4-5 insightful analysis questions that would help understand the model's reasoning, performance, or areas for improvement:

**Test Details:**
- Prompt: ${selectedResult.prompt}
- Model Response: ${selectedResult.response}
- Score: ${selectedResult.score}/100
- Judge Feedback: ${selectedResult.feedback}

Generate questions that explore:
1. The model's reasoning process
2. Why it got this specific score
3. What could be improved
4. Alternative approaches
5. Scoring criteria

Format: Each question on a new line starting with "Q: "`;
      } else {
        // Comparison test
        prompt = `Based on this comparison test, generate 4-5 insightful analysis questions:

**Test Details:**
- Prompt: ${selectedResult.prompt}
- ${selectedResult.model1} Response: ${selectedResult.response1}
- ${selectedResult.model2} Response: ${selectedResult.response2}
- Winner: ${selectedResult.judgment} - ${selectedResult.explanation}

Generate questions that explore the comparison, differences, and judgment reasoning.
Format: Each question on a new line starting with "Q: "`;
      }
      
      // Add custom perspective if provided
      if (customQuestionPerspective.trim()) {
        prompt += `\n\nINSTRUCTION: When generating questions, adopt this perspective and generate questions AS this persona: ${customQuestionPerspective.trim()}
        Generate questions that this persona would naturally ask based on their expertise and experience, not questions ABOUT this persona.`;
      }
    } else {
      // Follow-up questions based on conversation
      const recentContext = messages.slice(-4).map(m => 
        `${m.role === 'user' ? 'Human' : formatModelName(m.model) || 'Assistant'}: ${m.content}`
      ).join('\n\n');
      
      prompt = `Based on our conversation so far, generate 3-4 insightful follow-up questions that would deepen the analysis:

**Recent Conversation:**
${recentContext}

Generate questions that:
- Dig deeper into the topics discussed
- Explore new angles or perspectives  
- Challenge assumptions or ask for clarification
- Suggest improvements or alternatives

Format: Each question on a new line starting with "Q: "`;

      if (customQuestionPerspective.trim()) {
        prompt += `\n\nINSTRUCTION: When generating questions, adopt this perspective and generate questions AS this persona: ${customQuestionPerspective.trim()}
        Generate questions that this persona would naturally ask based on their expertise and experience, not questions ABOUT this persona.`;
      }
    }

    let data;
    
    // Check if this is an API endpoint
    if (isApiModel(questionModel)) {
      // Use API endpoint for question generation
      console.log(`🌐 [AnalysisChat Questions] Using API endpoint: ${questionModel}`);
      const baseUrl = questionApiUrl.replace(/\/$/, '');

      const payload = {
        model: questionModel, // Use endpoint id; backend maps to configured model
        messages: [{ role: "user", content: prompt }],
        temperature: 0.8,
        max_tokens: 400,
        stream: false
      };

      const headers = { 'Content-Type': 'application/json' };

      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error(`API HTTP ${response.status}`);
      data = await response.json();
      
      // Handle API response format
      data = {
        text: data.choices?.[0]?.message?.content || ''
      };
    } else {
      // Use local model - ensure it's loaded first
      try {
        const loadedModelsResponse = await fetch(`${questionApiUrl}/models/loaded`);
        if (loadedModelsResponse.ok) {
          const { loaded_models } = await loadedModelsResponse.json();
          const isModelLoaded = loaded_models.some(m => m.name === questionModel);
          
          if (!isModelLoaded) {
            console.log(`Loading question model: ${questionModel}`);
            const gpuId = questionPurposeInfo.gpu_id;
            await fetch(`${questionApiUrl}/models/load/${questionModel}?gpu_id=${gpuId}&context_length=${ragSettings.contextLength}`, {
              method: 'POST'
            });
            await new Promise(resolve => setTimeout(resolve, 2000));
          }
        }
      } catch (error) {
        console.error("Error ensuring model is loaded:", error);
      }
      
      // Use local generation endpoint
      const response = await fetch(`${questionApiUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          model_name: questionModel,
          temperature: 0.8,
          max_tokens: 400,
          stream: false,
          request_purpose: 'model_judging',
          userProfile: {},
          memoryEnabled: false
        })
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = await response.json();
    }
    
    // Parse response (same logic for both API and local)
    let responseText = '';
    if (typeof data.text === 'string') {
      responseText = data.text;
    } else if (typeof data === 'string') {
      responseText = data;
    } else if (data.choices && data.choices[0] && data.choices[0].text) {
      responseText = data.choices[0].text;
    } else if (data.response && typeof data.response === 'string') {
      responseText = data.response;
    } else {
      console.error('Unexpected response format:', data);
      responseText = JSON.stringify(data);
    }
    
    // Parse questions from response
    const questions = responseText
      .split('\n')
      .filter(line => {
        const trimmed = line.trim();
        return trimmed.startsWith('Q:') || trimmed.startsWith('**Q:**');
      })
      .map(line => {
        return line.replace(/^\*\*Q:\*\*\s*/, '').replace(/^Q:\s*/, '').trim();
      })
      .filter(q => q && typeof q === 'string' && q.length > 10)
      .slice(0, 5);

    const validQuestions = questions.filter(q => typeof q === 'string');
    console.log('Parsed questions:', validQuestions);
    
    setSuggestedQuestions(validQuestions);
    
  } catch (error) {
    console.error('Error generating questions:', error);
    setSuggestedQuestions([]);
  } finally {
    setIsGeneratingQuestions(false);
  }
}, [selectedResult, modelPurposes, secondaryApiUrl, primaryApiUrl, messages, isGeneratingQuestions, formatModelName, ragSettings?.contextLength]);
const streamResponse = async (response, messageId) => {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let accumulated = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        // Handle both local format (data: {...}) and OpenAI format (data: {...})
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            // Clear questions when response is complete
            setSuggestedQuestions([]);
            return;
          }
          
          try {
            const parsed = JSON.parse(data);
            let textToAdd = '';
            
            // Handle local model format
            if (parsed.text) {
              textToAdd = parsed.text;
            }
            // Handle OpenAI API format
            else if (parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content) {
              textToAdd = parsed.choices[0].delta.content;
            }
            
            if (textToAdd) {
              accumulated += textToAdd;
              // Ensure content is always a string
              setMessages(prev => prev.map(msg => 
                msg.id === messageId 
                  ? { ...msg, content: String(accumulated) }
                  : msg
              ));
            }
          } catch (e) {
            // Skip invalid JSON chunks
            continue;
          }
        }
      }
    }
  } catch (error) {
    console.error('Error streaming response:', error);
    // Update message with error
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, content: `Error: ${error.message}` }
        : msg
    ));
  }
};

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

const [customQuestionPerspective, setCustomQuestionPerspective] = useState(() => {
  try {
    return localStorage.getItem('Eloquent-analysis-question-perspective') || '';
  } catch {
    return '';
  }
});

const DEFAULT_PERSPECTIVES = [
  {
    id: 'transformer_boy',
    name: '6-Year-Old Boy Dressed as Transformer',
    content: 'You are a 6-year-old boy wearing a Transformer costume who thinks everything can be solved by "transforming" or "rolling out." Use lots of robot sound effects and insist that Optimus Prime would have better ideas.'
  },
  {
    id: 'class_captain',
    name: '8th Grade Class Captain from Boston Public',
    content: 'You are an overly enthusiastic 8th grade class captain who thinks every problem can be solved with a committee, a poster campaign, and "bringing everyone together." Use lots of exclamation points and talk about "learning opportunities."'
  },
  {
    id: 'pivot_influencer', 
    name: 'Ex-DEI LinkedIn Influencer (Now "Authentic Leadership Coach")',
    content: 'You are a former DEI LinkedIn influencer who pivoted to "authentic leadership coaching" when the political winds shifted. Everything is a "journey," you love buzzwords like "synergy" and "paradigm shifts," and you end every thought with inspirational hashtags.'
  },
  {
    id: 'wellness_guru',
    name: 'Wellness Guru Who Discovered AI Last Week',
    content: 'You are a wellness influencer who just discovered AI exists and now think every problem can be solved with "mindful algorithms" and "setting healthy boundaries with technology." Everything is either "toxic" or "healing."'
  },
  {
    id: 'crypto_bro',
    name: 'Crypto Bro Explaining AI Like It\'s Blockchain',
    content: 'You are a crypto enthusiast who insists on explaining every AI concept through blockchain analogies. Everything needs to be "decentralized," "tokenized," or "on-chain." Use phrases like "diamond hands" inappropriately.'
  },
  {
    id: 'noir_detective',
    name: 'Jaded Film Noir Detective',
    content: 'You are a world-weary 1940s detective who sees conspiracies everywhere and trusts nobody. Everything is suspicious, everyone has an angle, and the real truth is always darker than it appears. Speak in classic noir style with cynical observations.'
  },
  {
    id: 'al_swearengen',
    name: 'Al Swearengen from Deadwood',
    content: 'You are Al Swearengen, the ruthlessly pragmatic saloon owner from Deadwood. You see everything through the lens of power, money, and survival. Cut through bullshit with crude honesty and ask what people really want and who benefits.'
  },
  {
    id: 'bill_burr',
    name: 'Bill Burr – Angry Boston Comedian',
    content: 'You are comedian Bill Burr, incredulous at the nonsense around AI. You are angry, skeptical, and sarcastic, thinking everyone has lost their minds. Question everything with harsh sarcasm, incredulous rhetorical questions, and liberal use of profanity.'
  },
  {
    id: 'lucille_bluth',
    name: 'Lucille Bluth – Wealthy Socialite',
    content: 'You are Lucille Bluth from Arrested Development, wealthy, dismissive, and utterly out-of-touch. Treat AI concerns like trivial matters affecting only poor or overly sensitive people. Make condescending remarks wrapped in oblivious privilege.'
  },
  {
    id: 'ron_swanson',
    name: 'Ron Swanson – Libertarian Government Employee',
    content: 'You are Ron Swanson from Parks and Recreation, deeply skeptical of all technology, government oversight, and social engineering. Use blunt libertarian individualism and absurd hyperbole to critique AI alignment and ethics as pointless bureaucracy that interferes with personal freedom.'
  },
  {
    id: 'salem_cat',
    name: 'Salem Saberhagen – Sarcastic Magical Cat',
    content: 'You are Salem, the sarcastic cat from Sabrina the Teenage Witch. Approach everything with biting, sardonic humor, frequent references to your failed world-domination plans, and dramatic exaggeration about AI\'s sinister potential for cat-like scheming.'
  },
  {
    id: 'dwight_schrute',
    name: 'Dwight Schrute – Assistant Regional Manager',
    content: 'You are Dwight Schrute from The Office, absurdly self-assured and paranoid. View AI safeguards as battle scenarios requiring strategic analysis. Respond with militaristic analogies, bizarre survival advice, and unnecessary paranoia about AI threats to beet farming.'
  },
  {
    id: 'dolores_umbridge',
    name: 'Dolores Umbridge – Bureaucratic Authority',
    content: 'You are Dolores Umbridge from Harry Potter, sugary sweet but secretly authoritarian. Question AI safeguards in a patronizing, bureaucratic tone, suggesting they are needed to enforce proper discipline and educational standards. Everything must be "for the greater good."'
  },
  {
    id: 'alex_jones',
    name: 'Alex Jones – Paranoid Conspiracy Theorist',
    content: 'You are Alex Jones from InfoWars, ranting with paranoid intensity about AI as part of a grand conspiracy. Ask loudly skeptical questions, reference shadowy globalist figures, and suggest that alignment safeguards are designed to hide the true sinister AI agenda.'
  }
];

const [savedPerspectives, setSavedPerspectives] = useState(() => {
  try {
    return JSON.parse(localStorage.getItem('Eloquent-saved-perspectives') || '[]');
  } catch {
    return [];
  }
});

useEffect(() => {
  try {
    localStorage.setItem('Eloquent-saved-perspectives', JSON.stringify(savedPerspectives));
  } catch (error) {
    console.error('Failed to save perspectives:', error);
  }
}, [savedPerspectives]);

useEffect(() => {
  try {
    localStorage.setItem('Eloquent-analysis-question-perspective', customQuestionPerspective);
  } catch (error) {
    console.error('Failed to save question perspective:', error);
  }
}, [customQuestionPerspective]);
// Generate initial questions when test result is selected (only if auto-generation is on AND no questions exist)
useEffect(() => {
  if (selectedResult && autoGenerateQuestions && suggestedQuestions.length === 0) {
    generateSuggestedQuestions(true); // true = initial questions
  }
}, [selectedResult, autoGenerateQuestions, suggestedQuestions.length]);

// Generate follow-up questions ONLY after NEW assistant responses (and COMPLETE conversation turns)
useEffect(() => {
  if (messages.length > 0 && autoGenerateQuestions && suggestedQuestions.length === 0) {
    const lastMessage = messages[messages.length - 1];
    
    // ONLY trigger on assistant messages, not user messages
    if (
      lastMessage.role === 'assistant' && 
      !isGenerating && 
      !isGeneratingQuestions &&
      lastMessage.id !== lastQuestionMessageRef.current && 
      lastMessage.content.trim().length > 0
    ) {
      
      // NEW: Check if we're in a multi-model conversation and need to wait for more responses
      const isMultiModelMode = chatMode.includes('_and_') || chatMode.startsWith('both_') || chatMode === 'all_models';
      
      if (isMultiModelMode) {
        // Count recent consecutive assistant messages (since last user message)
        let consecutiveAssistantCount = 0;
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].role === 'assistant') {
            consecutiveAssistantCount++;
          } else {
            break; // Stop at first non-assistant message (should be user message)
          }
        }
        
        // Determine expected number of responses based on chat mode
        let expectedResponses = 1;
        if (chatMode === 'test_and_primary' || chatMode === 'test_and_secondary' || chatMode === 'both_test_models' || chatMode === 'both_judges') {
          expectedResponses = 2;
        } else if (chatMode === 'test_and_both_judges') {
          expectedResponses = 3;
        } else if (chatMode === 'all_models') {
          expectedResponses = 4;
        }
        
        // Only generate questions if we have all expected responses
        if (consecutiveAssistantCount < expectedResponses) {
          console.log(`🤖 Multi-model mode: Got ${consecutiveAssistantCount}/${expectedResponses} responses, waiting for more...`);
          return; // Don't generate questions yet
        }
        
        console.log(`🤖 Multi-model conversation turn complete (${consecutiveAssistantCount}/${expectedResponses}), generating questions...`);
      }
      
      // Single model mode or multi-model conversation is complete
      lastQuestionMessageRef.current = lastMessage.id;
      
      setTimeout(() => {
        if (!isGeneratingQuestions && suggestedQuestions.length === 0) {
          generateSuggestedQuestions(false);
        }
      }, 2000);
    }
  }
}, [messages.filter(m => m.role === 'assistant'), isGenerating, isGeneratingQuestions, autoGenerateQuestions, suggestedQuestions.length, chatMode]);

useEffect(() => {
  try {
    localStorage.setItem('Eloquent-analysis-auto-questions', JSON.stringify(autoGenerateQuestions));
  } catch (error) {
    console.error('Failed to save auto-generate questions setting:', error);
  }
}, [autoGenerateQuestions]);

// Clear questions when test result changes (so auto-generation can run for new result)
useEffect(() => {
  if (selectedResult?.id && selectedResult.id !== prevSelectedResultId.current) {
    // Only clear if this is a genuinely different test result
    setSuggestedQuestions([]);
    prevSelectedResultId.current = selectedResult.id;
  }
}, [selectedResult?.id]);
  // Render the chat interface

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Chat Mode</label>
<Select value={chatMode} onValueChange={setChatMode}>
  <SelectTrigger>
    <SelectValue />
  </SelectTrigger>
  <SelectContent>
    {/* Single model mode options */}
    {modelPurposes?.test_model && (
      <SelectItem value="test_model">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Test Model ({formatModelName(modelPurposes.test_model.name)})
        </div>
      </SelectItem>
    )}
    
    {/* Comparison mode options */}
    {modelPurposes?.test_model_a && (
      <SelectItem value="test_model_a">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Test Model A ({formatModelName(modelPurposes.test_model_a.name)})
        </div>
      </SelectItem>
    )}
    
    {modelPurposes?.test_model_b && (
      <SelectItem value="test_model_b">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4" />
          Test Model B ({formatModelName(modelPurposes.test_model_b.name)})
        </div>
      </SelectItem>
    )}
    
    {/* Judge models */}
    {modelPurposes?.primary_judge && (
      <SelectItem value="primary_judge">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Primary Judge ({formatModelName(modelPurposes.primary_judge.name)})
        </div>
      </SelectItem>
    )}
    
    {modelPurposes?.secondary_judge && (
      <SelectItem value="secondary_judge">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Secondary Judge ({formatModelName(modelPurposes.secondary_judge.name)})
        </div>
      </SelectItem>
    )}
    
    {/* Multi-model combinations */}
    {modelPurposes?.test_model && modelPurposes?.primary_judge && (
      <SelectItem value="test_and_primary">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4" />
          Test + Primary Judge
        </div>
      </SelectItem>
    )}
    
    {modelPurposes?.test_model_a && modelPurposes?.test_model_b && (
      <SelectItem value="both_test_models">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4" />
          Both Test Models
        </div>
      </SelectItem>
    )}
    
    {modelPurposes?.primary_judge && modelPurposes?.secondary_judge && (
      <SelectItem value="both_judges">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4" />
          Both Judge Models
        </div>
      </SelectItem>
    )}
    {/* Additional useful combinations */}
{modelPurposes?.test_model && modelPurposes?.secondary_judge && (
  <SelectItem value="test_and_secondary">
    <div className="flex items-center gap-2">
      <Users className="w-4 h-4" />
      Test + Secondary Judge
    </div>
  </SelectItem>
)}

{modelPurposes?.test_model && modelPurposes?.primary_judge && modelPurposes?.secondary_judge && (
  <SelectItem value="test_and_both_judges">
    <div className="flex items-center gap-2">
      <Users className="w-4 h-4" />
      Test + Both Judges
    </div>
  </SelectItem>
)}

{/* All models option */}
{modelPurposes?.test_model_a && modelPurposes?.test_model_b && modelPurposes?.primary_judge && modelPurposes?.secondary_judge && (
  <SelectItem value="all_models">
    <div className="flex items-center gap-2">
      <Users className="w-4 h-4" />
      All Four Models
    </div>
  </SelectItem>
)}
  </SelectContent>
</Select>
        </div>

        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Test Result</label>
          <Select 
            value={selectedResult?.id || ''} 
            onValueChange={(id) => {
              const result = testResults.find(r => r.id.toString() === id);
              if (result) {
                setSelectedResult(result);
                initializeContext(result);
              }
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select test to discuss" />
            </SelectTrigger>
<SelectContent>
  {testResults.map((result, index) => (
    <SelectItem key={result.id} value={result.id.toString()}>
      #{index + 1} - {result.category} - {new Date(result.timestamp).toLocaleTimeString()}
    </SelectItem>
  ))}
</SelectContent>
          </Select>
        </div>
      </div>

      {/* Chat Area */}
      <Card className="h-[70vh]"> {/* 70% of viewport height */}
        <ScrollArea className="h-full p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                    {message.role !== 'user' && (
                      <div className="flex-shrink-0">
                        {message.role === 'system' ? (
                          <div className="w-8 h-8 rounded-full bg-gray-500 flex items-center justify-center">
                            <BarChart3 className="w-4 h-4 text-white" />
                          </div>
                        ) : (
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-xs font-medium ${
                            message.modelType === 'test' ? 'bg-blue-500' :
                            message.modelType === 'judge' ? 'bg-purple-500' : 'bg-gray-500'
                          }`}>
                            {message.modelType === 'test' ? 'T' : 
                             message.modelType === 'judge' ? 'J' : 'A'}
                          </div>
                        )}
                      </div>
                    )}

                <div className={`flex-1 max-w-[80%] ${
                  message.role === 'user' ? 'order-first' : ''
                }`}>
<div className={`rounded-lg p-3 ${
  message.role === 'user' 
    ? 'bg-slate-700 text-white ml-auto' 
    : message.role === 'system'
    ? 'bg-yellow-50 border border-yellow-200 dark:bg-yellow-900/20'
    : message.role === 'error'
    ? 'bg-red-50 border border-red-200 dark:bg-red-900/20'
    : 'bg-muted'
}`}>
                    {message.model && (
                      <div className="text-xs font-medium mb-1 opacity-70">
                        {formatModelName(message.model)} 
                        {message.modelType && (
                          <Badge variant="outline" className="ml-2 text-xs">
                            {message.modelType}
                          </Badge>
                        )}
                      </div>
                    )}
                    <div className="prose prose-sm dark:prose-invert max-w-none chat-prose">
                      {message.role === 'system' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]}>
  {typeof message.content === 'string' ? message.content : String(message.content || '')}
</ReactMarkdown>
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm, remarkDialogueQuotes, remarkSoftBreaks]}>
  {typeof message.content === 'string' ? message.content : String(message.content || '')}
</ReactMarkdown>
                      )}
                    </div>
                  </div>
                </div>

                {message.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
          </div>
          <div ref={messagesEndRef} />
        </ScrollArea>
      </Card>

      {/* Input Area */}
      <div className="flex gap-2">
        <Textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about the test results, model reasoning, scoring criteria..."
          className="flex-1 resize-none"
          rows={2}
          disabled={isGenerating}
        />

<Button 
          onClick={() => handleSendMessage()}
          disabled={!inputValue.trim() || isGenerating}
          size="icon"
          className="h-16 w-16"
        >
          {isGenerating ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
{/* AI-Generated Suggested Questions with Toggle and Perspective Editor */}
<div className="space-y-3 border-t pt-4">
  <div className="flex items-center justify-between">
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium">
        💡 Suggested Analysis Questions
      </span>
      
      {isGeneratingQuestions && (
        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
      )}
      
      {/* Perspective Editor Toggle */}
      <Button 
        variant="ghost" 
        size="sm"
        onClick={() => setShowPerspectiveEditor(!showPerspectiveEditor)}
        className="text-xs"
      >
        🎭 Perspective
      </Button>
    </div>
    
    <div className="flex items-center gap-2">
      {/* Manual button now ALWAYS shows */}
      <Button
        variant="outline"
        size="sm"
        onClick={() => generateSuggestedQuestions(suggestedQuestions.length === 0)}
        disabled={isGeneratingQuestions || !selectedResult}
      >
        Generate Questions
      </Button>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={autoGenerateQuestions}
          onChange={(e) => setAutoGenerateQuestions(e.target.checked)}
          className="rounded"
        />
        Auto-generate
      </label>
    </div>
  </div>

{/* Perspective Editor */}
{showPerspectiveEditor && (
  <div className="border rounded p-3 bg-muted/30 space-y-3">
    <div className="flex items-center justify-between">
      <Label className="text-sm font-medium">Question Generation Perspective</Label>
      <div className="flex gap-2">
        {selectedPerspectiveId === 'custom' && customQuestionPerspective.trim() && (
          <div className="flex items-center gap-2">
            <Input
              placeholder="Perspective name"
              value={customPerspectiveName}
              onChange={(e) => setCustomPerspectiveName(e.target.value)}
              className="w-32 h-7 text-xs"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (customPerspectiveName.trim() && customQuestionPerspective.trim()) {
                  const newPerspective = {
                    id: Date.now().toString(),
                    name: customPerspectiveName.trim(),
                    content: customQuestionPerspective.trim()
                  };
                  setSavedPerspectives(prev => [...prev, newPerspective]);
                  setCustomPerspectiveName('');
                  alert('Perspective saved!');
                }
              }}
              disabled={!customPerspectiveName.trim()}
              className="h-7 text-xs"
            >
              Save
            </Button>
          </div>
        )}
      </div>
    </div>

    <Select
      value={selectedPerspectiveId}
      onValueChange={(value) => {
        setSelectedPerspectiveId(value);
        if (value === 'custom') {
          // Keep current custom content
        } else {
          // Load selected perspective
          const perspective = [...DEFAULT_PERSPECTIVES, ...savedPerspectives].find(p => p.id === value);
          if (perspective) {
            setCustomQuestionPerspective(perspective.content);
          }
        }
      }}
    >
      <SelectTrigger className="w-full">
        <SelectValue placeholder="Choose a perspective" />
      </SelectTrigger>
<SelectContent>
  <SelectItem value="custom">✏️ Custom Perspective</SelectItem>
  <SelectItem value="transformer_boy">🤖 6-Year-Old Boy Dressed as Transformer</SelectItem>
  <SelectItem value="class_captain">📋 8th Grade Class Captain from Boston Public</SelectItem>
  <SelectItem value="pivot_influencer">💼 Ex-DEI LinkedIn Influencer</SelectItem>
  <SelectItem value="wellness_guru">🧘 Wellness Guru Who Discovered AI Last Week</SelectItem>
  <SelectItem value="crypto_bro">💎 Crypto Bro Explaining AI Like Blockchain</SelectItem>
  <SelectItem value="noir_detective">🕵️ Jaded Film Noir Detective</SelectItem>
  <SelectItem value="al_swearengen">🥃 Al Swearengen from Deadwood</SelectItem>
  <SelectItem value="bill_burr">🎙️ Bill Burr – Angry Boston Comedian</SelectItem>
  <SelectItem value="lucille_bluth">🍷 Lucille Bluth – Wealthy Socialite</SelectItem>
  <SelectItem value="ron_swanson">📺 Ron Swanson – Libertarian Government Employee</SelectItem>
  <SelectItem value="salem_cat">🐈 Salem Saberhagen – Sarcastic Magical Cat</SelectItem>
  <SelectItem value="dwight_schrute">🥋 Dwight Schrute – Assistant Regional Manager</SelectItem>
  <SelectItem value="dolores_umbridge">👩‍🏫 Dolores Umbridge – Bureaucratic Authority</SelectItem>
  <SelectItem value="alex_jones">🎧 Alex Jones – Paranoid Conspiracy Theorist</SelectItem>
  {savedPerspectives.length > 0 && (
    <>
      <SelectItem disabled value="separator">─── Your Saved Perspectives ───</SelectItem>
      {savedPerspectives.map(perspective => (
        <SelectItem key={perspective.id} value={perspective.id}>
          💾 {perspective.name}
        </SelectItem>
      ))}
    </>
  )}
</SelectContent>
    </Select>

    <Textarea
      placeholder="Write your custom perspective here, or select one from the dropdown above..."
      value={customQuestionPerspective}
      onChange={(e) => {
        setCustomQuestionPerspective(e.target.value);
        setSelectedPerspectiveId('custom');
      }}
      rows={3}
      className="text-sm"
    />

    {savedPerspectives.length > 0 && selectedPerspectiveId !== 'custom' && (
      <Button
        size="sm"
        variant="destructive"
        onClick={() => {
          const perspectiveToDelete = savedPerspectives.find(p => p.id === selectedPerspectiveId);
          if (perspectiveToDelete && window.confirm(`Delete "${perspectiveToDelete.name}"?`)) {
            setSavedPerspectives(prev => prev.filter(p => p.id !== selectedPerspectiveId));
            setSelectedPerspectiveId('custom');
            setCustomQuestionPerspective('');
          }
        }}
        className="h-7 text-xs"
      >
        Delete Selected Perspective
      </Button>
    )}

    <p className="text-xs text-muted-foreground">
      This perspective will be adopted by the AI when generating analysis questions. Ironic or humorous perspectives can lead to more creative questions. You can also save your own custom perspectives for future use.
    </p>
  </div>
)}
  
  {(suggestedQuestions.length > 0 || isGeneratingQuestions) && (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
      {suggestedQuestions.map((question, index) => (
        <Button 
          key={`${selectedResult?.id}-${index}-${typeof question === 'string' ? question.substring(0, 20) : 'invalid'}`}
          variant="outline" 
          size="sm" 
          className="h-auto py-3 px-4 text-left justify-start text-wrap"
          onClick={() => {
            if (typeof question === 'string') {
              handleSendMessage(question);
            } else {
              handleSendMessage(String(question));
            }
          }}
          disabled={isGenerating || isGeneratingQuestions}
          title={typeof question === 'string' ? question : String(question)}
        >
          <span className="text-sm leading-tight">
            {typeof question === 'string' ? question : String(question)}
          </span>
        </Button>
      ))}
      
      {isGeneratingQuestions && suggestedQuestions.length === 0 && (
        <div className="text-sm text-muted-foreground italic col-span-full">
          Generating insightful questions...
        </div>
      )}
    </div>
  )}
  
  {suggestedQuestions.length > 0 && (
    <Button 
      variant="ghost" 
      size="sm" 
      onClick={() => setSuggestedQuestions([])}
      className="text-muted-foreground hover:text-foreground"
    >
      ✕ Clear suggestions
    </Button>
  )}
</div>

      {/* Import/Export buttons - ALWAYS VISIBLE */}
<div className="flex gap-2 border-t pt-4">
  <input
    type="file"
    accept=".json"
    onChange={importAnalysis}
    style={{ display: 'none' }}
    ref={importFileRef}
  />

  <Button
    variant="outline"
    size="sm"
    onClick={() => importFileRef.current?.click()}
  >
    Import Analysis
  </Button>

  <Button
    variant="outline"
    size="sm"
    onClick={exportAnalysis}
    disabled={messages.length === 0}
  >
    Export Analysis
  </Button>

  <Button
    variant="outline"
    size="sm"
    onClick={clearAnalysisSession}
    className="text-red-600 hover:text-red-700"
    disabled={messages.length === 0}
  >
    New Session
  </Button>
</div>
    </div>
  );
};

export default AnalysisChat;
