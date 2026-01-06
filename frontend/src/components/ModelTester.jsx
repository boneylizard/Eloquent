import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import { Textarea } from './ui/textarea';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import ModelSelector from './ModelSelector';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import AnalysisChat from './AnalysisChat';
import { Label } from './ui/label';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { 
  Play, 
  Upload, 
  Download, 
  Trash2, 
  RefreshCw, 
  Trophy,
  BarChart3,
  FileJson,
  Bot,
  User,
  ThumbsUp,
  ThumbsDown,
  MessageSquare,
  Minus,
  Users,
  Send,
  Loader2,
  AlertTriangle
} from 'lucide-react';

const ModelTester = () => {
  const {
    loadedModels,
    availableModels,
    primaryModel,
    secondaryModel,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    userProfile,
    loadModel,
    unloadModel,
    settings,
    setAvailableModels,
    isModelLoading,
    fetchLoadedModels,
  } = useApp();
const { characters, buildSystemPrompt } = useApp();
  // State management
  const [selectedModels, setSelectedModels] = useState([]);
  const [modelGpuAssignments, setModelGpuAssignments] = useState({}); // {modelName: gpuId}
  const [judgeModelGpu, setJudgeModelGpu] = useState(0); // Default to GPU 0
  const [selectedTestCharacter, setSelectedTestCharacter] = useState(null);
const [selectedTestCharacterA, setSelectedTestCharacterA] = useState(null);
const [selectedTestCharacterB, setSelectedTestCharacterB] = useState(null);
const [selectedPrimaryJudgeCharacter, setSelectedPrimaryJudgeCharacter] = useState(null);
const [selectedSecondaryJudgeCharacter, setSelectedSecondaryJudgeCharacter] = useState(null);
const [customJudgingCriteria, setCustomJudgingCriteria] = useState('');

  const [selectedTestModelGpu, setSelectedTestModelGpu] = useState(0); // Default to GPU 0
  const [testingMode, setTestingMode] = useState('single'); // 'single' or 'comparison'
  const [newPromptText, setNewPromptText] = useState('');
  const [newPromptCategory, setNewPromptCategory] = useState('');
  const [selectedTestModel, setSelectedTestModel] = useState('');
  const [promptCollection, setPromptCollection] = useState([]);
  const [collectionName, setCollectionName] = useState('');
  const [currentTest, setCurrentTest] = useState(null);
  const [testResults, setTestResults] = useState([]);
  const [eloRatings, setEloRatings] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [parameterSweepEnabled, setParameterSweepEnabled] = useState(false);
const [parameterConfig, setParameterConfig] = useState({
  temperature: { enabled: false, min: 0.1, max: 1.0, step: 0.1, current: 0.7 },
  top_p: { enabled: false, min: 0.1, max: 1.0, step: 0.1, current: 0.9 },
  top_k: { enabled: false, min: 1, max: 100, step: 5, current: 40 },
  repetition_penalty: { enabled: false, min: 1.0, max: 2.0, step: 0.1, current: 1.1 }
});
  const [activeTab, setActiveTab] = useState('setup');
  const [judgeMode, setJudgeMode] = useState('automated'); // 'automated', 'human', 'both'
  const [judgeModel, setJudgeModel] = useState('');
  const [secondaryJudgeModel, setSecondaryJudgeModel] = useState('');
  const [secondaryJudgeModelGpu, setSecondaryJudgeModelGpu] = useState(0); // Default to GPU 0
  const fileInputRef = useRef(null);
  const [modelPurposes, setModelPurposes] = useState({
    test_model: null,
    test_model_a: null,
    test_model_b: null,
    primary_judge: null, 
    secondary_judge: null
  });
  const [isLoadingPurpose, setIsLoadingPurpose] = useState({
    test_model: false,
    test_model_a: false,
    test_model_b: false,
    primary_judge: false,
    secondary_judge: false
  });
    const [isUnloadingPurpose, setIsUnloadingPurpose] = useState({
    test_model: false,
    test_model_a: false,
    test_model_b: false,
    primary_judge: false,
    secondary_judge: false
  });
   // NEW: API functions for purpose-based loading
  const fetchModelsByPurpose = useCallback(async () => {
    try {
      const response = await fetch(`${PRIMARY_API_URL}/models/by-purpose`);
      if (response.ok) {
        const data = await response.json();
        setModelPurposes(data.purposes);
      }
    } catch (error) {
      console.error('Error fetching models by purpose:', error);
    }
  }, [PRIMARY_API_URL]);

const judgeSingleResponseWithApiModel = useCallback(async (prompt, response, judgeModelName) => {
    const endpointInfo = (settings.customApiEndpoints || []).find(e => e.id === judgeModelName);
    if (!endpointInfo) {
        throw new Error(`API Endpoint configuration for ${judgeModelName} not found.`);
    }

    const apiUrl = endpointInfo.url.replace(/\/$/, '');
    const apiKey = endpointInfo.apiKey;

    const judgePrompt = `You are a STRICT evaluation system. Rate this response using this exact format:
SCORE: [number 1-100]
REASON: [brief critical assessment]

PROMPT: "${prompt}"
RESPONSE: "${response}"

Respond with ONLY the score and reason.`;

    const payload = {
        model: endpointInfo.name, // Use the configured model name for the API
        messages: [{ role: "user", content: judgePrompt }],
        temperature: 0.3,
        max_tokens: 150,
        stream: false,
    };

    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    try {
        const response = await fetch(`${apiUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API Judge Error ${response.status}: ${errorBody}`);
        }
        const data = await response.json();
        return data.choices[0]?.message?.content || '';
    } catch (error) {
        console.error(`Error judging single response with API model ${judgeModelName}:`, error);
        return `SCORE: 50\nREASON: Error during API judging: ${error.message}`;
    }
}, [settings.customApiEndpoints]);

const getModelDisplayName = useCallback((modelName, character, paramLabel = '') => {
  let displayName = modelName;
  if (character) {
    displayName += ` (as ${character.name})`;
  }
  return displayName + paramLabel;
}, []);

// Helper function to build character-aware system prompt for judges
const buildJudgePrompt = useCallback((prompt, response, judgeCharacter, customCriteria, isComparison = false, response1 = null, response2 = null, model1Name = null, model2Name = null) => {
  // Add character context if judge character exists
  let basePrompt = '';
  if (judgeCharacter) {
    const characterContext = buildSystemPrompt(judgeCharacter);
    basePrompt = characterContext + '\n\n';
  }
  
  if (isComparison) {
    // Comparison judging
    basePrompt += `Compare these two responses and determine which is better.

${customCriteria ? `CRITERIA: ${customCriteria}\n\n` : ''}PROMPT: "${prompt}"

RESPONSE A: "${response1}"

RESPONSE B: "${response2}"

You must respond in EXACTLY this format:

Winner: Response A
Explanation: [Your explanation here]

OR

Winner: Response B  
Explanation: [Your explanation here]

OR

Winner: TIE
Explanation: [Your explanation here]

Be precise - start with "Winner:" followed by exactly "Response A", "Response B", or "TIE".`;
  } else {
    // Single response judging
    basePrompt += `You are a STRICT evaluation system. Rate this response using this exact format:
SCORE: [number 1-100]
REASON: [brief critical assessment]

${customCriteria ? `CRITERIA: ${customCriteria}\n\n` : ''}PROMPT: "${prompt}"
RESPONSE: "${response}"

Respond with ONLY the score and reason.`;
  }
  
  return basePrompt;
}, [buildSystemPrompt]);

const judgeWithApiModel = useCallback(async (prompt, judgeModelName) => {
    const endpointInfo = (settings.customApiEndpoints || []).find(e => e.id === judgeModelName);
    if (!endpointInfo) {
        throw new Error(`API Endpoint configuration for ${judgeModelName} not found.`);
    }

    const apiUrl = endpointInfo.url.replace(/\/$/, ''); // Ensure no trailing slash
    const apiKey = endpointInfo.apiKey;

    const payload = {
        model: endpointInfo.name, // Use the model name specified in the endpoint config
        messages: [
            { role: "system", content: "You are a strict, impartial AI model evaluator." },
            { role: "user", content: prompt }
        ],
        temperature: 0.3,
        max_tokens: 150,
        stream: false,
    };

    const headers = {
        'Content-Type': 'application/json',
    };
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    try {
        const response = await fetch(`${apiUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorBody = await response.text();
            throw new Error(`API Judge Error ${response.status}: ${errorBody}`);
        }

        const data = await response.json();
        return data.choices[0]?.message?.content || '';
    } catch (error) {
        console.error(`Error judging with API model ${judgeModelName}:`, error);
        return `Error: ${error.message}`;
    }
}, [settings.customApiEndpoints]);
  
const loadModelForPurpose = useCallback(async (purpose, modelName, gpuId, contextLength = 4096) => {
    setIsLoadingPurpose(prev => ({ ...prev, [purpose]: true }));

    // *** THE CRITICAL FIX IS HERE ***
    // Check if the selected model is an API endpoint.
    const isApi = modelName && modelName.startsWith('endpoint-');

    if (isApi) {
        // --- API Model Logic ---
        // For an API model, we don't need to call the backend to load anything.
        // We just update the frontend state to assign the API to the purpose.
        console.log(`[API] Assigning API endpoint ${modelName} to purpose: ${purpose}`);
        setModelPurposes(prev => ({
            ...prev,
            [purpose]: {
                name: modelName,
                gpu_id: gpuId, // We still track the "target" GPU for UI consistency
                context_length: 'N/A', // Context length isn't applicable
                isApi: true // Add a flag to identify it as an API
            }
        }));
        setIsLoadingPurpose(prev => ({ ...prev, [purpose]: false }));
        return; // Stop execution here for API models
    }

    // --- Local Model Logic (your existing, working code) ---
    // If it's not an API model, proceed with the backend load request.
    const targetApiUrl = gpuId === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
    const finalUrl = `${targetApiUrl}/models/load-for-purpose/${purpose}`;

    console.log(`[Local] Loading ${modelName} for ${purpose}`);
    console.log(`[Local] Requested GPU: ${gpuId}`);
    console.log(`[Local] Routing to: ${targetApiUrl} (${gpuId === 0 ? 'PRIMARY_API_URL' : 'SECONDARY_API_URL'})`);
    console.log(`[Local] Full URL: ${finalUrl}`);

    try {
        const response = await fetch(finalUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                gpu_id: gpuId,
                context_length: contextLength
            })
        });

        if (response.ok) {
            await fetchModelsByPurpose();
            console.log(`✅ Successfully sent load request for ${modelName} as ${purpose} to GPU ${gpuId}`);
        } else {
            const errorText = await response.text();
            let errorMessage = errorText;
            
            // Try to parse JSON error
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = errorJson.detail || errorJson.message || errorText;
            } catch {
                // Not JSON, use as-is
            }
            
            console.error(`❌ Failed to load ${modelName} for ${purpose} on GPU ${gpuId}:`, errorMessage);
            
            // Check for specific error types
            if (errorMessage.includes('GPU mismatch') || errorMessage.includes('not accessible')) {
                alert(`❌ GPU Error: ${errorMessage}\n\nPlease ensure you're loading the model on the correct GPU.\nThe backend instance may not have access to GPU ${gpuId}.`);
            } else if (errorMessage.includes('Insufficient VRAM') || errorMessage.includes('VRAM')) {
                alert(`❌ VRAM Error: ${errorMessage}\n\nPlease unload other models to free up GPU memory.`);
            } else {
                alert(`❌ Failed to load ${modelName} for ${purpose}: ${errorMessage}`);
            }
        }
    } catch (error) {
        console.error('Error loading model for purpose:', error);
        alert(`Error loading model: ${error.message}`);
    } finally {
        setIsLoadingPurpose(prev => ({ ...prev, [purpose]: false }));
    }
}, [PRIMARY_API_URL, SECONDARY_API_URL, fetchModelsByPurpose]);

const unloadModelPurpose = useCallback(async (purpose) => {
      setIsUnloadingPurpose(prev => ({ ...prev, [purpose]: true }));

      const purposeInfo = modelPurposes[purpose];
      if (!purposeInfo) {
          console.error(`Cannot unload purpose '${purpose}', as it is not loaded.`);
          setIsUnloadingPurpose(prev => ({...prev, [purpose]: false}));
          return;
      }
      const targetApiUrl = purposeInfo.gpu_id === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;

      // The following line MUST use backticks (` `) to correctly build the URL.
      const finalUrl = `${targetApiUrl}/models/unload-purpose/${purpose}`;

      console.log(`Sending unload request to: ${finalUrl}`); // Logging for confirmation

      try {
        const response = await fetch(finalUrl, {
          method: 'POST'
        });
        
        if (response.ok) {
          await fetchModelsByPurpose();
          console.log(`✅ Successfully sent unload request for ${purpose} model`);
        } else {
           const error = await response.text();
           console.error(`Failed to unload ${purpose}:`, error);
           alert(`Failed to unload model: ${error}`);
        }
      } catch (error) {
        console.error('Error unloading model purpose:', error);
      } finally {
        setIsUnloadingPurpose(prev => ({ ...prev, [purpose]: false }));
      }
    }, [PRIMARY_API_URL, SECONDARY_API_URL, fetchModelsByPurpose, modelPurposes]);

  // NEW: Load purposes on component mount and when models change
  useEffect(() => {
    fetchModelsByPurpose();
  }, [fetchModelsByPurpose]);

  // NEW: Load purposes on component mount and when models change
  useEffect(() => {
    fetchModelsByPurpose();
  }, [fetchModelsByPurpose]);
// Reconcile judgments from two judges
const reconcileJudgments = useCallback((primaryVerdict, secondaryVerdict) => {
  // Both judges agree
  if (primaryVerdict === secondaryVerdict) {
    return {
      finalVerdict: primaryVerdict,
      reconciliationReason: `Both judges agreed: ${primaryVerdict}`
    };
  }
  
  // One judge says TIE, other says winner → Winner wins
  if (primaryVerdict === 'TIE' && secondaryVerdict !== 'TIE') {
    return {
      finalVerdict: secondaryVerdict,
      reconciliationReason: `Primary: TIE, Secondary: ${secondaryVerdict} → Winner declared`
    };
  }
  
  if (secondaryVerdict === 'TIE' && primaryVerdict !== 'TIE') {
    return {
      finalVerdict: primaryVerdict,
      reconciliationReason: `Primary: ${primaryVerdict}, Secondary: TIE → Winner declared`
    };
  }
  
  // Judges disagree on winner (A vs B) → TIE
  if ((primaryVerdict === 'A' && secondaryVerdict === 'B') || 
      (primaryVerdict === 'B' && secondaryVerdict === 'A')) {
    return {
      finalVerdict: 'TIE',
      reconciliationReason: `Judges disagreed: Primary=${primaryVerdict}, Secondary=${secondaryVerdict} → TIE`
    };
  }
  
  // Fallback (shouldn't happen)
  return {
    finalVerdict: 'TIE',
    reconciliationReason: `Unexpected combination: Primary=${primaryVerdict}, Secondary=${secondaryVerdict} → TIE`
  };
}, []);

// Helper function to get correct API URL for a model
const getAllModelOptions = () => {
    const customEndpoints = settings.customApiEndpoints || [];
    const apiModels = customEndpoints
        .filter(endpoint => endpoint.enabled)
        .map(endpoint => ({
            id: endpoint.id,
            name: `[API] ${endpoint.name}`,
            isApi: true,
        }));

    const localModels = availableModels.map(model => ({
        id: model,
        name: model.split('/').pop().replace(/\.(bin|gguf)$/i, ''),
        isApi: false,
    }));

    return [...apiModels, ...localModels];
};

const allModelOptions = getAllModelOptions();
  // Default prompt collections
  const defaultCollections = {
    'MT-Bench Sample': [
      {
        id: 1,
        prompt: "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        category: "writing"
      },
      {
        id: 2,
        prompt: "How can I develop my critical thinking skills?",
        category: "advice"
      },
      {
        id: 3,
        prompt: "Explain the concept of machine learning to a 12-year-old.",
        category: "explanation"
      }
    ],
    'Instruction Following': [
      {
        id: 1,
        prompt: "Write a Python function that takes a list of numbers and returns the sum of all even numbers.",
        category: "coding"
      },
      {
        id: 2,
        prompt: "Create a formal email to request time off from work for a family vacation.",
        category: "communication"
      },
      {
        id: 3,
        prompt: "Summarize the main themes of Shakespeare's Romeo and Juliet in 3 bullet points.",
        category: "analysis"
      }
    ]
  };
const addManualPrompt = () => {
  if (!newPromptText.trim()) return;
  const newPrompt = {
    id: Date.now(),
    prompt: newPromptText.trim(),
    category: newPromptCategory.trim() || 'general'
  };
  setPromptCollection(prev => [...prev, newPrompt]);
  setNewPromptText('');
  setNewPromptCategory('');
};

const removePrompt = (id) => {
  setPromptCollection(prev => prev.filter(prompt => prompt.id !== id));
};

// Initialize ELO ratings for loaded models
useEffect(() => {
  if (loadedModels.length > 0) {
    const initialRatings = {};
    loadedModels.forEach(model => {
      if (!eloRatings[model.name]) {
        initialRatings[model.name] = 1500; // Standard ELO starting rating
      }
    });
    if (Object.keys(initialRatings).length > 0) {
        setEloRatings(prev => ({ ...prev, ...initialRatings }));
      }
    }
  }, [loadedModels]);

  // Set judge model to secondary model if available
  useEffect(() => {
    if (secondaryModel && !judgeModel) {
      setJudgeModel(secondaryModel);
    }
  }, [secondaryModel, judgeModel]);

  // Load ELO ratings from localStorage on component mount
useEffect(() => {
  try {
    const savedRatings = localStorage.getItem('Eloquent-model-elo-ratings');
    if (savedRatings) {
      const parsedRatings = JSON.parse(savedRatings);
      console.log('📊 Loaded ELO ratings from localStorage:', parsedRatings);
      setEloRatings(parsedRatings);
    } else {
      console.log('📊 No saved ELO ratings found, starting fresh');
    }
  } catch (error) {
    console.error('Error loading ELO ratings from localStorage:', error);
  }
}, []); // Empty dependency array = run once on mount

// Save ELO ratings to localStorage whenever they change
useEffect(() => {
  if (Object.keys(eloRatings).length > 0) {
    try {
      localStorage.setItem('Eloquent-model-elo-ratings', JSON.stringify(eloRatings));
      console.log('💾 Saved ELO ratings to localStorage');
    } catch (error) {
      console.error('Error saving ELO ratings to localStorage:', error);
    }
  }
}, [eloRatings]);
// Add this function at the component level
// Add this function with other useCallback functions
const resetEloRatings = useCallback(() => {
  if (window.confirm('Are you sure you want to reset all ELO ratings? This cannot be undone.')) {
    setEloRatings({});
    localStorage.removeItem('Eloquent-model-elo-ratings');
    console.log('🔄 ELO ratings reset');
  }
}, []);
const calculateEloChange = useCallback((score, currentRating, category) => {
  // Score-based ELO changes (more granular and proportional)
  let baseChange = 0;
  
  if (score >= 95) baseChange = 16;        // Exceptional performance
  else if (score >= 90) baseChange = 13;   // Excellent  
  else if (score >= 85) baseChange = 10;   // Very good
  else if (score >= 80) baseChange = 7;    // Good
  else if (score >= 75) baseChange = 4;    // Above average
  else if (score >= 70) baseChange = 1;    // Decent
  else if (score >= 65) baseChange = -4;    // Neutral zone  
  else if (score >= 60) baseChange = -7;    // Slightly above neutral
  else if (score >= 55) baseChange = -10;   // Poor
  else if (score >= 50) baseChange = -13;  // Bad
  else if (score >= 40) baseChange = -20;   // Slightly below neutral
  else if (score >= 30) baseChange = -24;   // Poor
  else if (score >= 25) baseChange = -30;  // Bad
  else if (score >= 20) baseChange = -36;  // Very bad
  else if (score >= 15) baseChange = -42;  // Terrible
  else if (score >= 10) baseChange = -48;  // Abysmal
  else baseChange = -50;                   // Complete failure

  // Category difficulty multipliers
  const categoryMultipliers = {
    'math': 1.3,           // Math is hard, reward/punish more
    'coding': 1.3,         // Coding is hard
    'reasoning': 1.2,      // Logic/reasoning is challenging
    'stem': 1.2,           // STEM topics are technical
    'extraction': 1.1,     // Data extraction requires precision
    'writing': 1.0,        // Writing is baseline difficulty
    'roleplay': 0.9,       // Roleplay is more subjective
    'humanities': 1.0,     // Humanities is baseline
    'general': 1.0         // General is baseline
  };
  
  const categoryMultiplier = categoryMultipliers[category.toLowerCase()] || 1.0;
  
  // Rating-based adjustment (diminishing returns for high-rated models)
  let ratingAdjustment = 1.0;
  if (currentRating >= 2200) ratingAdjustment = 0.3;      // Elite models
  else if (currentRating >= 2000) ratingAdjustment = 0.4; // Very high
  else if (currentRating >= 1800) ratingAdjustment = 0.5; // High
  else if (currentRating >= 1600) ratingAdjustment = 0.7; // Above average
  else if (currentRating >= 1500) ratingAdjustment = 1.0; // Average
  else if (currentRating >= 1400) ratingAdjustment = 1.1; // Below average models
  else ratingAdjustment = 1.2;                            // Low rated models get bigger swings
  
  // Apply all multipliers
  const finalChange = Math.round(baseChange * categoryMultiplier * ratingAdjustment);
  
  // Ensure minimum change for extreme scores (prevent stagnation)
  if (score >= 95 && finalChange < 5) return 5;
  if (score <= 10 && finalChange > -5) return -5;
  
  return finalChange;
}, []);


  // Import JSON prompt collection
  const handleImportCollection = useCallback((event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const content = e.target.result;
      let prompts = [];
      let collectionName = file.name.replace(/\.(json|jsonl)$/i, '');
      
      if (file.name.toLowerCase().endsWith('.jsonl')) {
  // Handle JSONL format with multiple structure support
  const lines = content.split('\n').filter(line => line.trim());
  prompts = lines.map((line, index) => {
    try {
      const item = JSON.parse(line);
      let prompt = null;
      let category = 'general';
      
      // Strategy 1: MT-Bench format (turns array)
      if (item.turns && Array.isArray(item.turns) && item.turns.length > 0) {
        prompt = item.turns[0];
        category = item.category || 'general';
        console.log(`Line ${index + 1}: MT-Bench format detected`);
      }
      
      // Strategy 2: ChatML/Messages format
      else if (item.messages && Array.isArray(item.messages)) {
        const userMessage = item.messages.find(msg => msg.role === 'user');
        if (userMessage) {
          prompt = userMessage.content;
          category = item.category || item.dataset || 'general';
          console.log(`Line ${index + 1}: Messages format detected`);
        }
      }
      
      // Strategy 3: Direct prompt fields
      else if (item.prompt) {
        prompt = item.prompt;
        category = item.category || item.type || item.task || 'general';
        console.log(`Line ${index + 1}: Direct prompt field detected`);
      }
      
      // Strategy 4: Common text fields
      else if (item.text) {
        prompt = item.text;
        category = item.category || item.type || item.label || 'general';
        console.log(`Line ${index + 1}: Text field detected`);
      }
      
      // Strategy 5: Instruction/Question fields
      else if (item.instruction) {
        prompt = item.instruction;
        if (item.input && item.input.trim()) {
          prompt += `\n\nInput: ${item.input}`;
        }
        category = item.category || item.task_type || 'general';
        console.log(`Line ${index + 1}: Instruction format detected`);
      }
      
      // Strategy 6: Question field
      else if (item.question) {
        prompt = item.question;
        category = item.category || item.subject || item.domain || 'general';
        console.log(`Line ${index + 1}: Question format detected`);
      }
      
      // Strategy 7: Query field
      else if (item.query) {
        prompt = item.query;
        category = item.category || item.type || 'general';
        console.log(`Line ${index + 1}: Query format detected`);
      }
      
      // Strategy 8: Input field
      else if (item.input) {
        prompt = item.input;
        category = item.category || item.task || 'general';
        console.log(`Line ${index + 1}: Input format detected`);
      }
      
      // Strategy 9: Detect if this is a results/answers file (skip)
      else if (item.choices || item.answer_id || item.model_id) {
        console.warn(`Line ${index + 1}: Detected answers/results format, skipping`);
        return null;
      }
      
      // Strategy 10: Last resort - stringify the object and warn
      else {
        console.warn(`Line ${index + 1}: Unknown format, using string representation:`, item);
        prompt = JSON.stringify(item);
        category = 'unknown';
      }
      
      // Final validation
      if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
        console.warn(`Line ${index + 1}: No valid prompt found, skipping`);
        return null;
      }
      
      return {
        id: item.question_id || item.id || index + 1,
        prompt: prompt.trim(),
        category: category
      };
      
    } catch (lineError) {
      console.warn(`Error parsing line ${index + 1}:`, lineError);
      return null;
    }
  }).filter(Boolean); // Remove null entries
  
  console.log(`Successfully parsed ${prompts.length} prompts from JSONL file`);
} else {
  // Handle regular JSON format with comprehensive structure support
  const jsonData = JSON.parse(content);
  let dataArray = [];
  
  // Strategy 1: Direct array format
  if (Array.isArray(jsonData)) {
    dataArray = jsonData;
    console.log('JSON: Direct array format detected');
  }
  
  // Strategy 2: Object with prompts array
  else if (jsonData.prompts && Array.isArray(jsonData.prompts)) {
    dataArray = jsonData.prompts;
    collectionName = jsonData.name || collectionName;
    console.log('JSON: Object with "prompts" array detected');
  }
  
  // Strategy 3: Object with examples array
  else if (jsonData.examples && Array.isArray(jsonData.examples)) {
    dataArray = jsonData.examples;
    collectionName = jsonData.name || jsonData.dataset_name || collectionName;
    console.log('JSON: Object with "examples" array detected');
  }
  
  // Strategy 4: Object with questions array
  else if (jsonData.questions && Array.isArray(jsonData.questions)) {
    dataArray = jsonData.questions;
    collectionName = jsonData.name || jsonData.title || collectionName;
    console.log('JSON: Object with "questions" array detected');
  }
  
  // Strategy 5: Object with data array
  else if (jsonData.data && Array.isArray(jsonData.data)) {
    dataArray = jsonData.data;
    collectionName = jsonData.name || jsonData.dataset || collectionName;
    console.log('JSON: Object with "data" array detected');
  }
  
  // Strategy 6: Object with items array
  else if (jsonData.items && Array.isArray(jsonData.items)) {
    dataArray = jsonData.items;
    collectionName = jsonData.name || collectionName;
    console.log('JSON: Object with "items" array detected');
  }
  
  // Strategy 7: Object with tasks array
  else if (jsonData.tasks && Array.isArray(jsonData.tasks)) {
    dataArray = jsonData.tasks;
    collectionName = jsonData.name || jsonData.benchmark || collectionName;
    console.log('JSON: Object with "tasks" array detected');
  }
  
  // Strategy 8: Nested structure with train/test/validation
  else if (jsonData.train && Array.isArray(jsonData.train)) {
    dataArray = jsonData.train;
    collectionName = (jsonData.name || collectionName) + ' (Train)';
    console.log('JSON: Training dataset detected');
  }
  else if (jsonData.test && Array.isArray(jsonData.test)) {
    dataArray = jsonData.test;
    collectionName = (jsonData.name || collectionName) + ' (Test)';
    console.log('JSON: Test dataset detected');
  }
  else if (jsonData.validation && Array.isArray(jsonData.validation)) {
    dataArray = jsonData.validation;
    collectionName = (jsonData.name || collectionName) + ' (Validation)';
    console.log('JSON: Validation dataset detected');
  }
  
  else {
    throw new Error('Unrecognized JSON format. Expected array or object with data arrays like "prompts", "examples", "questions", "data", etc.');
  }
  
  // Now process the array with the same robust extraction logic as JSONL
  prompts = dataArray.map((item, index) => {
    let prompt = null;
    let category = 'general';
    let id = index + 1;
    
    // Use the same extraction strategies as JSONL
    if (item.turns && Array.isArray(item.turns) && item.turns.length > 0) {
      prompt = item.turns[0];
      category = item.category || 'general';
      id = item.question_id || item.id || index + 1;
    }
    else if (item.messages && Array.isArray(item.messages)) {
      const userMessage = item.messages.find(msg => msg.role === 'user');
      if (userMessage) {
        prompt = userMessage.content;
        category = item.category || item.dataset || 'general';
        id = item.id || item.message_id || index + 1;
      }
    }
    else if (item.prompt) {
      prompt = item.prompt;
      category = item.category || item.type || item.task || 'general';
      id = item.id || item.prompt_id || index + 1;
    }
    else if (item.text) {
      prompt = item.text;
      category = item.category || item.type || item.label || 'general';
      id = item.id || item.text_id || index + 1;
    }
    else if (item.instruction) {
      prompt = item.instruction;
      if (item.input && item.input.trim()) {
        prompt += `\n\nInput: ${item.input}`;
      }
      category = item.category || item.task_type || 'general';
      id = item.id || item.task_id || index + 1;
    }
    else if (item.question) {
      prompt = item.question;
      category = item.category || item.subject || item.domain || 'general';
      id = item.id || item.question_id || index + 1;
    }
    else if (item.query) {
      prompt = item.query;
      category = item.category || item.type || 'general';
      id = item.id || item.query_id || index + 1;
    }
    else if (item.input) {
      prompt = item.input;
      category = item.category || item.task || 'general';
      id = item.id || item.input_id || index + 1;
    }
    else if (typeof item === 'string') {
      prompt = item;
      category = 'general';
      id = index + 1;
    }
    else {
      console.warn(`JSON item ${index + 1}: Unknown format, using string representation:`, item);
      prompt = JSON.stringify(item);
      category = 'unknown';
      id = index + 1;
    }
    
    // Validation
    if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
      console.warn(`JSON item ${index + 1}: No valid prompt found, skipping`);
      return null;
    }
    
    return {
      id: id,
      prompt: prompt.trim(),
      category: category
    };
  }).filter(Boolean);
  
  console.log(`Successfully parsed ${prompts.length} prompts from JSON file`);
        }
      if (prompts.length === 0) {
        throw new Error('No valid prompts found in file.');
      }

      setPromptCollection(prompts);
      setCollectionName(collectionName);
      
    } catch (error) {
      console.error('Error parsing file:', error);
      alert(`Error parsing file: ${error.message}\n\nSupported formats:\n- JSON: {"prompts": [{"prompt": "...", "category": "..."}]}\n- JSONL: One JSON object per line`);
    }
  };
  reader.readAsText(file);
  event.target.value = ''; // Reset input
}, []);

  // Load default collection
  const loadDefaultCollection = useCallback((collectionName) => {
    const collection = defaultCollections[collectionName];
    if (collection) {
      setPromptCollection(collection);
      setCollectionName(collectionName);
    }
  }, []);
// Enhanced version that tracks more data
const saveEloUpdate = useCallback((modelName, oldRating, newRating, score, category) => {
  const timestamp = new Date().toISOString();
  const metadataKey = 'Eloquent-model-elo-metadata';
  
  try {
    const existingData = JSON.parse(localStorage.getItem(metadataKey) || '{}');
    
    if (!existingData[modelName]) {
      existingData[modelName] = {
        firstTested: timestamp,
        totalTests: 0,
        history: []
      };
    }
    
    existingData[modelName].totalTests++;
    existingData[modelName].lastTested = timestamp;
    existingData[modelName].history.push({
      timestamp,
      oldRating,
      newRating,
      change: newRating - oldRating,
      score,
      category
    });
    
    // Keep only last 50 history entries per model
    if (existingData[modelName].history.length > 50) {
      existingData[modelName].history = existingData[modelName].history.slice(-50);
    }
    
    localStorage.setItem(metadataKey, JSON.stringify(existingData));
  } catch (error) {
    console.error('Error saving ELO metadata:', error);
  }
}, []);
  // Export test results
const exportResults = useCallback(() => {
  // Calculate category averages from current test results
  const categoryStats = {};
  testResults.forEach(result => {
    if (!categoryStats[result.category]) {
      categoryStats[result.category] = {
        totalScore: 0,
        count: 0,
        scores: []
      };
    }
    categoryStats[result.category].totalScore += result.score;
    categoryStats[result.category].count += 1;
    categoryStats[result.category].scores.push(result.score);
  });

  // Calculate averages and additional stats
  const categoryAnalysis = Object.keys(categoryStats).map(category => ({
    category,
    averageScore: (categoryStats[category].totalScore / categoryStats[category].count).toFixed(1),
    testCount: categoryStats[category].count,
    highestScore: Math.max(...categoryStats[category].scores),
    lowestScore: Math.min(...categoryStats[category].scores)
  })).sort((a, b) => parseFloat(b.averageScore) - parseFloat(a.averageScore));

const exportData = {
  exportInfo: {
    collectionName,
    testingMode,
    exportDate: new Date().toISOString(),
    totalTests: testResults.length,
    overallAverageScore: testResults.length > 0 ? 
      (testResults.reduce((sum, r) => sum + r.score, 0) / testResults.length).toFixed(1) : null,
    // ADD THESE TWO LINES:
    parameterSweepEnabled,
    parameterConfig: parameterSweepEnabled ? parameterConfig : null
  },
    testConfiguration: {
      testModel: selectedTestModel,
      judgeModel: judgeModel,
      judgeMode: judgeMode,
      modelsCompared: testingMode === 'comparison' ? selectedModels : null
    },
    categoryAnalysis,
results: testResults.map(result => ({
  prompt: result.prompt,
  category: result.category,
  testModel: result.model, // This now includes parameter label if present
  baseModel: result.baseModel, // NEW: Original model name without parameters
  parameters: result.parameters, // NEW: Actual parameter values used
  judgeModel: result.judgeModel,
  // Existing dual judge export fields
  secondaryJudgeModel: result.secondaryJudgeModel,
  isDualJudged: result.isDualJudged,
  primaryJudgment: result.primaryJudgment,
  secondaryJudgment: result.secondaryJudgment,
  reconciliationReason: result.reconciliationReason,
  response: result.response,
  score: result.score,
  feedback: result.feedback,
  timestamp: result.timestamp,
  // Include comparison data if it exists
  ...(result.model1 && {
    model1: result.model1,
    model2: result.model2,
    response1: result.response1,
    response2: result.response2,
    judgment: result.judgment,
    explanation: result.explanation
  })
})),
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], {
    type: 'application/json'
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `model-test-results-${testingMode}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}, [collectionName, testingMode, testResults, selectedTestModel, judgeModel, judgeMode, selectedModels, parameterSweepEnabled, parameterConfig]);

const handleImportResults = useCallback((event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const importedData = JSON.parse(e.target.result);
      
      // Import results - ensure each has an ID for React keys and proper dual judge fields
if (importedData.results) {
const resultsWithIds = importedData.results.map((result, index) => ({
  ...result,
  id: result.id || `imported-${Date.now()}-${index}`,
  model: result.testModel || result.model, // Handle field name differences
  // ADD THESE TWO LINES:
  baseModel: result.baseModel || result.testModel || result.model,
  parameters: result.parameters || null,
    
    // Enhanced dual judge compatibility
    secondaryJudgeModel: result.secondaryJudgeModel || null,
    isDualJudged: result.isDualJudged || false,
    primaryJudgment: result.primaryJudgment || null,
    secondaryJudgment: result.secondaryJudgment || null,
    reconciliationReason: result.reconciliationReason || null,
    
    // Backwards compatibility for legacy single judge results
    score: result.score || (result.primaryJudgment?.score) || 50,
    feedback: result.feedback || (result.primaryJudgment?.feedback) || 'No feedback available',
    
    // Backwards compatibility for comparison results
    judgment: result.judgment || (result.primaryJudgment?.verdict) || 'TIE',
    explanation: result.explanation || (result.primaryJudgment?.explanation) || 'No explanation available',
    
    // Ensure all required fields exist for both test types
    ...(result.model1 && {
      model1: result.model1,
      model2: result.model2,
      response1: result.response1,
      response2: result.response2
    }),
    
    // Single model test fields
    ...(result.response && {
      response: result.response
    })
  }));
  setTestResults(resultsWithIds);
}
      
      // Import test configuration if available
      if (importedData.testConfiguration) {
        const config = importedData.testConfiguration;
        if (config.testModel) setSelectedTestModel(config.testModel);
        if (config.judgeModel) setJudgeModel(config.judgeModel);
        if (config.judgeMode) setJudgeMode(config.judgeMode);
        if (config.collectionName) setCollectionName(importedData.exportInfo?.collectionName || '');
      }
      
      // Import ELO ratings if available
      if (importedData.eloRatings) {
        setEloRatings(importedData.eloRatings);
      }
      if (importedData.exportInfo?.parameterSweepEnabled) {
  setParameterSweepEnabled(true);
  if (importedData.exportInfo.parameterConfig) {
    setParameterConfig(importedData.exportInfo.parameterConfig);
  }
}
      // Determine what type of results were imported
      const singleModelResults = importedData.results?.filter(r => r.model && !r.model1)?.length || 0;
      const comparisonResults = importedData.results?.filter(r => r.model1 && r.model2)?.length || 0;
      const dualJudgedResults = importedData.results?.filter(r => r.isDualJudged)?.length || 0;
      
let importMessage = `Successfully imported ${importedData.results?.length || 0} test results!`;
if (singleModelResults > 0) importMessage += `\n• Single model tests: ${singleModelResults}`;
if (comparisonResults > 0) importMessage += `\n• Comparison tests: ${comparisonResults}`;
if (dualJudgedResults > 0) importMessage += `\n• Dual-judged results: ${dualJudgedResults}`;

// ADD THESE LINES:
const parameterizedResults = importedData.results?.filter(r => r.parameters)?.length || 0;
if (parameterizedResults > 0) {
  importMessage += `\n• Parameter sweep tests: ${parameterizedResults}`;
}
if (importedData.exportInfo?.parameterSweepEnabled) {
  importMessage += `\n• Parameter configuration imported`;
}
      
      alert(importMessage);
    } catch (error) {
      console.error('Import error:', error);
      alert(`Error importing file: ${error.message}`);
    }
  };
  reader.readAsText(file);
  event.target.value = ''; // Reset input
}, []);
  // Generate response from model
const isApiModel = (modelId) => modelId && modelId.startsWith('endpoint-');

const generateApiResponse = useCallback(async (modelName, prompt, apiUrl) => {
    const payload = {
        model: modelName,
        messages: [
            { role: "system", content: "You are a helpful and direct AI assistant." },
            { role: "user", content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 1024,
        stream: false,
    };

    try {
        const response = await fetch(`${apiUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`API HTTP ${response.status}`);
        const data = await response.json();
        return data.choices[0]?.message?.content || '';
    } catch (error) {
        console.error(`Error generating API response from ${modelName}:`, error);
        return `Error: ${error.message}`;
    }
}, []);

const generateResponse = useCallback(async (modelName, prompt, apiUrl, paramOverrides = {}, requestPurpose = 'user_chat', character = null) => {
  const params = {
    temperature: settings.temperature || 0.7,
    top_p: settings.top_p || 0.9,
    top_k: settings.top_k || 40,
    repetition_penalty: settings.repetition_penalty || 1.1,
    ...paramOverrides
  };

  if (isApiModel(modelName)) {
    // For API models, if character is provided, prepend character context to the prompt
    let finalPrompt = prompt;
    if (character) {
      const characterSystemPrompt = buildSystemPrompt(character);
      finalPrompt = `${characterSystemPrompt}\n\nUser: ${prompt}\n\nAssistant:`;
    }
    return generateApiResponse(modelName, finalPrompt, apiUrl);
  }

  // For local models, build the full prompt with character context if provided
  let finalPrompt = prompt;
  if (character) {
    const characterSystemPrompt = buildSystemPrompt(character);
    finalPrompt = `${characterSystemPrompt}\n\nUser: ${prompt}\n\nAssistant:`;
  }

  try {
    const response = await fetch(`${apiUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: finalPrompt,
        model_name: modelName,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        repetition_penalty: params.repetition_penalty,
        request_purpose: requestPurpose
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    return data.text || '';

  } catch (error) {
    console.error(`Error generating response from local model ${modelName}:`, error);
    return `Error: ${error.message}`;
  }
}, [userProfile, generateApiResponse, settings, buildSystemPrompt, isApiModel]);

// New function to judge a single response
const judgeSingleResponse = useCallback(async (prompt, response, modelName, judgeModel, judgeModelGpu, secondaryJudgeModel = null, secondaryJudgeModelGpu = null, judgeCharacter = null, secondaryJudgeCharacter = null) => {
  if (!judgeModel) return null;

  const getSingleJudgment = async (jModel, jGpu, jCharacter) => {
    const judgePrompt = buildJudgePrompt(prompt, response, jCharacter, customJudgingCriteria);
    
    // Check if the judge model is an API endpoint
    if (isApiModel(jModel)) {
      console.log(`⚖️ Judging single response with API Model: ${jModel}${jCharacter ? ` (as ${jCharacter.name})` : ''}`);
      return await judgeSingleResponseWithApiModel(judgePrompt, response, jModel);
    }

    // Otherwise, use the existing local model logic
    console.log(`⚖️ Judging single response with Local Model: ${jModel} on GPU ${jGpu}${jCharacter ? ` (as ${jCharacter.name})` : ''}`);
    const apiUrl = jGpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
    const judgeResponse = await fetch(`${apiUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: judgePrompt,
        model_name: jModel,
        gpu_id: jGpu,
        request_purpose: 'model_judging',
        temperature: 0.3,
        max_tokens: 150,
        stream: false,
        userProfile: {},
        memoryEnabled: false
      })
    });
    if (!judgeResponse.ok) throw new Error(`HTTP ${judgeResponse.status}`);
    const data = await judgeResponse.json();
    return data.text?.trim() || '';
  };

  let primaryJudgment = null;
  try {
    // Primary Judge
    const primaryJudgmentText = await getSingleJudgment(judgeModel, judgeModelGpu, judgeCharacter);
    let score = 50;
    let feedback = primaryJudgmentText;
    const scoreMatch = primaryJudgmentText.match(/SCORE:\s*(\d+)/i);
    if (scoreMatch) {
      score = parseInt(scoreMatch[1], 10);
      feedback = primaryJudgmentText.replace(/SCORE:\s*\d+/i, '').replace('REASON:', '').trim();
    }
    primaryJudgment = { 
      score, 
      feedback,
      judgeCharacter: judgeCharacter?.name || null
    };

    // Secondary Judge (if applicable)
    if (!secondaryJudgeModel) {
      return { 
        score, 
        feedback, 
        primaryJudgment, 
        secondaryJudgment: null, 
        reconciliationReason: 'Single judge used',
        judgeCharacter: judgeCharacter?.name || null
      };
    }

    const secondaryJudgmentText = await getSingleJudgment(secondaryJudgeModel, secondaryJudgeModelGpu, secondaryJudgeCharacter);
    let secondaryScore = 50;
    let secondaryFeedback = secondaryJudgmentText;
    const secondaryScoreMatch = secondaryJudgmentText.match(/SCORE:\s*(\d+)/i);
    if (secondaryScoreMatch) {
        secondaryScore = parseInt(secondaryScoreMatch[1], 10);
        secondaryFeedback = secondaryJudgmentText.replace(/SCORE:\s*\d+/i, '').replace('REASON:', '').trim();
    }
    const secondaryJudgment = { 
      score: secondaryScore, 
      feedback: secondaryFeedback,
      judgeCharacter: secondaryJudgeCharacter?.name || null
    };
    
    // Reconcile
    const scoreDiff = Math.abs(score - secondaryScore);
    const finalScore = Math.round((score + secondaryScore) / 2);
    const combinedFeedback = `Primary${judgeCharacter ? ` (${judgeCharacter.name})` : ''}: ${feedback} | Secondary${secondaryJudgeCharacter ? ` (${secondaryJudgeCharacter.name})` : ''}: ${secondaryFeedback}`;
    
    return {
      score: finalScore,
      feedback: combinedFeedback,
      primaryJudgment,
      secondaryJudgment,
      reconciliationReason: scoreDiff <= 15 ? `Scores close (diff: ${scoreDiff}), averaged` : `Scores differed by ${scoreDiff}, averaged`,
      judgeCharacter: judgeCharacter?.name || null,
      secondaryJudgeCharacter: secondaryJudgeCharacter?.name || null
    };

  } catch (error) {
    console.error('Error in single response judge:', error);
    if (primaryJudgment) {
        return {
            ...primaryJudgment,
            feedback: primaryJudgment.feedback + ' (Secondary judge failed)',
            secondaryJudgment: null,
            reconciliationReason: 'Secondary judge failed, used primary only'
        };
    }
    return null;
  }
}, [PRIMARY_API_URL, SECONDARY_API_URL, userProfile, judgeSingleResponseWithApiModel, buildJudgePrompt, customJudgingCriteria, isApiModel]);

  // Judge response quality using secondary model
const judgeWithModel = useCallback(async (prompt, response1, response2, model1Name, model2Name, judgeModel, judgeModelGpu, secondaryJudgeModel = null, secondaryJudgeModelGpu = null, judgeCharacter = null, secondaryJudgeCharacter = null) => {
  const getJudgment = async (jModel, jGpu, jCharacter) => {
    const judgePrompt = buildJudgePrompt(prompt, null, jCharacter, customJudgingCriteria, true, response1, response2, model1Name, model2Name);
    
    // Check if the judge model is an API endpoint
    if (isApiModel(jModel)) {
        console.log(`⚖️ Judging with API Model: ${jModel}${jCharacter ? ` (as ${jCharacter.name})` : ''}`);
        return await judgeWithApiModel(judgePrompt, jModel);
    }

    // Otherwise, use the existing local model logic
    console.log(`⚖️ Judging with Local Model: ${jModel} on GPU ${jGpu}${jCharacter ? ` (as ${jCharacter.name})` : ''}`);
    const apiUrl = jGpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
    const response = await fetch(`${apiUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: judgePrompt,
            model_name: jModel,
            gpu_id: jGpu,
            request_purpose: 'model_judging',
            temperature: 0.3,
            max_tokens: 150,
            stream: false,
            userProfile: {},
            memoryEnabled: false
        })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    return data.text?.trim() || '';
  };

  try {
    // Primary Judge
    const primaryJudgmentText = await getJudgment(judgeModel, judgeModelGpu, judgeCharacter);
    let verdict = 'TIE';
    let explanation = 'No explanation provided';
    const winnerMatch = primaryJudgmentText.match(/Winner:\s*(Response\s*[AB]|TIE)/i);
    if (winnerMatch) {
      const winnerText = winnerMatch[1].toUpperCase();
      if (winnerText.includes('A')) verdict = 'A';
      else if (winnerText.includes('B')) verdict = 'B';
    }
    const explanationMatch = primaryJudgmentText.match(/Explanation:\s*(.+)/is);
    if (explanationMatch) explanation = explanationMatch[1].trim();
    const primaryJudgment = { 
      verdict, 
      explanation,
      judgeCharacter: judgeCharacter?.name || null
    };

    // Secondary Judge (if applicable)
    if (!secondaryJudgeModel) {
      return { 
        verdict, 
        explanation, 
        primaryJudgment, 
        secondaryJudgment: null, 
        reconciliationReason: 'Single judge used',
        judgeCharacter: judgeCharacter?.name || null
      };
    }

    const secondaryJudgmentText = await getJudgment(secondaryJudgeModel, secondaryJudgeModelGpu, secondaryJudgeCharacter);
    let secondaryVerdict = 'TIE';
    let secondaryExplanation = 'No explanation provided';
    const secondaryWinnerMatch = secondaryJudgmentText.match(/Winner:\s*(Response\s*[AB]|TIE)/i);
    if (secondaryWinnerMatch) {
        const winnerText = secondaryWinnerMatch[1].toUpperCase();
        if (winnerText.includes('A')) secondaryVerdict = 'A';
        else if (winnerText.includes('B')) secondaryVerdict = 'B';
    }
    const secondaryExplanationMatch = secondaryJudgmentText.match(/Explanation:\s*(.+)/is);
    if (secondaryExplanationMatch) secondaryExplanation = secondaryExplanationMatch[1].trim();

    const secondaryJudgment = {
      verdict: secondaryVerdict,
      explanation: secondaryExplanation,
      judgeCharacter: secondaryJudgeCharacter?.name || null
    };

    // Reconcile
    const { finalVerdict, reconciliationReason } = reconcileJudgments(verdict, secondaryVerdict);
    const combinedExplanation = `Primary${judgeCharacter ? ` (${judgeCharacter.name})` : ''}: ${explanation} | Secondary${secondaryJudgeCharacter ? ` (${secondaryJudgeCharacter.name})` : ''}: ${secondaryExplanation}`;

    return {
      verdict: finalVerdict,
      explanation: combinedExplanation,
      primaryJudgment,
      secondaryJudgment,
      reconciliationReason,
      judgeCharacter: judgeCharacter?.name || null,
      secondaryJudgeCharacter: secondaryJudgeCharacter?.name || null
    };

  } catch (error) {
    console.error('Error in judge model:', error);
    if (primaryJudgment) {
      error.primaryJudgment = primaryJudgment;
    }
    throw error;
  }
}, [PRIMARY_API_URL, SECONDARY_API_URL, reconcileJudgments, judgeWithApiModel, buildJudgePrompt, customJudgingCriteria, isApiModel]);


// Update ELO ratings
const updateEloRatings = useCallback((winner, loser, isDraw = false) => {
  const K = 32; // ELO K-factor
  
  setEloRatings(currentRatings => {
    const ratingWinner = currentRatings[winner] || 1500;
    const ratingLoser = currentRatings[loser] || 1500;
    
    console.log(`🏆 ELO UPDATE: ${winner} (${ratingWinner}) vs ${loser} (${ratingLoser}) - ${isDraw ? 'TIE' : winner + ' WINS'}`);
    
    const expectedWinner = 1 / (1 + Math.pow(10, (ratingLoser - ratingWinner) / 400));
    const expectedLoser = 1 / (1 + Math.pow(10, (ratingWinner - ratingLoser) / 400));
    
    let scoreWinner, scoreLoser;
    if (isDraw) {
      scoreWinner = scoreLoser = 0.5;
    } else {
      scoreWinner = 1;
      scoreLoser = 0;
    }
    
    const newRatingWinner = Math.round(ratingWinner + K * (scoreWinner - expectedWinner));
    const newRatingLoser = Math.round(ratingLoser + K * (scoreLoser - expectedLoser));
    
    console.log(`📊 ELO CHANGES: ${winner} ${ratingWinner} → ${newRatingWinner} (${newRatingWinner > ratingWinner ? '+' : ''}${newRatingWinner - ratingWinner})`);
    console.log(`📊 ELO CHANGES: ${loser} ${ratingLoser} → ${newRatingLoser} (${newRatingLoser > ratingLoser ? '+' : ''}${newRatingLoser - ratingLoser})`);
    
    const newRatings = {
      ...currentRatings,
      [winner]: newRatingWinner,
      [loser]: newRatingLoser
    };
    console.log(`💾 NEW ELO STATE:`, newRatings);
    return newRatings;
  });
}, []); // Remove eloRatings from dependency array since we use functional update

  // Run automated test
const runAutomatedTest = useCallback(async () => {
  // NEW: Purpose-based validation instead of counting total models
  if (testingMode === 'single') {
    // Check required purposes for single model testing
    if (!modelPurposes?.test_model) {
      alert('Single model testing requires a Test Model to be loaded.');
      return;
    }
    
    if ((judgeMode === 'automated' || judgeMode === 'both') && !modelPurposes?.primary_judge) {
      alert('Automated judging requires a Primary Judge model to be loaded.');
      return;
    }
    
    if (promptCollection.length === 0) {
      alert('Please load a prompt collection');
      return;
    }
    
  } else {
    // Check required purposes for comparison testing
    if (!modelPurposes?.test_model_a || !modelPurposes?.test_model_b) {
      alert('Model comparison testing requires both Test Model A and Test Model B to be loaded.');
      return;
    }
    
    if ((judgeMode === 'automated' || judgeMode === 'both') && !modelPurposes?.primary_judge) {
      alert('Automated judging requires a Primary Judge model to be loaded.');
      return;
    }
    
    if (promptCollection.length === 0) {
      alert('Please load a prompt collection');
      return;
    }
  }

  setIsRunning(true);
  setProgress(0);
  setActiveTab('testing');

  try {
    if (testingMode === 'single') {
      await runSingleModelTest();
    } else {
      await runComparisonTest();
    }
  } catch (error) {
    console.error('Test failed:', error);
    alert(`Test failed: ${error.message}`);
  } finally {
    setIsRunning(false);
    setActiveTab('results');
  }
}, [testingMode, selectedTestModel, selectedModels, promptCollection, judgeMode, judgeModel, loadedModels, modelPurposes, parameterSweepEnabled]);

// Add these utility functions
// Calculate total parameter combinations
const calculateTotalCombinations = useCallback(() => {
  if (!parameterSweepEnabled) return 1;
  
  let total = 1;
  Object.entries(parameterConfig).forEach(([key, config]) => {
    if (config.enabled) {
      const steps = Math.floor((config.max - config.min) / config.step) + 1;
      total *= steps;
    }
  });
  return total;
}, [parameterSweepEnabled, parameterConfig]);

// Generate all parameter combinations
const generateParameterCombinations = useCallback(() => {
  if (!parameterSweepEnabled) {
    return [{ temperature: 0.7, top_p: 0.9, top_k: 40, repetition_penalty: 1.1 }];
  }

  const paramKeys = Object.keys(parameterConfig);
  const ranges = {};
  
  // Generate ranges for enabled parameters
  paramKeys.forEach(key => {
    const config = parameterConfig[key];
    if (config.enabled) {
      ranges[key] = [];
      for (let val = config.min; val <= config.max; val += config.step) {
        ranges[key].push(Math.round(val * 100) / 100); // Round to avoid floating point issues
      }
    } else {
      ranges[key] = [config.current];
    }
  });

  // Generate cartesian product
  function cartesianProduct(obj) {
    const keys = Object.keys(obj);
    const values = keys.map(key => obj[key]);
    
    function* helper(index, current) {
      if (index === keys.length) {
        yield { ...current };
        return;
      }
      
      for (const value of values[index]) {
        current[keys[index]] = value;
        yield* helper(index + 1, current);
      }
    }
    
    return Array.from(helper(0, {}));
  }

  return cartesianProduct(ranges);
}, [parameterSweepEnabled, parameterConfig]);

// Create parameter labels for results
const createParameterLabel = useCallback((params) => {
  if (!parameterSweepEnabled) return '';
  
  const enabledParams = Object.entries(parameterConfig)
    .filter(([key, config]) => config.enabled)
    .map(([key]) => {
      const shortKey = key === 'temperature' ? 'T' : 
                      key === 'top_p' ? 'P' : 
                      key === 'top_k' ? 'K' : 'R';
      return `${shortKey}=${params[key]}`;
    });
  
  return enabledParams.length > 0 ? ` (${enabledParams.join(',')})` : '';
}, [parameterSweepEnabled, parameterConfig]);

// Single model testing function
const runSingleModelTest = useCallback(async () => {
  const paramCombinations = generateParameterCombinations();
  const totalTests = promptCollection.length * paramCombinations.length;
  let completedTests = 0;
  const newResults = [];

  const testModel = modelPurposes.test_model.name;
  const testModelGpu = modelPurposes.test_model.gpu_id;
  const testApiUrl = testModelGpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
  const testCharacter = selectedTestCharacter; // Get selected character

  let judgeModelName = null;
  let judgeGpu = null;
  let secondaryJudgeModelName = null;
  let secondaryJudgeGpu = null;
  let judgeCharacter = selectedPrimaryJudgeCharacter;
  let secondaryJudgeCharacter = selectedSecondaryJudgeCharacter;

  if (judgeMode === 'automated' || judgeMode === 'both') {
    if (modelPurposes.primary_judge) {
        judgeModelName = modelPurposes.primary_judge.name;
        judgeGpu = modelPurposes.primary_judge.gpu_id;
    }
    if (modelPurposes.secondary_judge) {
        secondaryJudgeModelName = modelPurposes.secondary_judge.name;
        secondaryJudgeGpu = modelPurposes.secondary_judge.gpu_id;
    }
  }

  console.log(`🎯 Starting parameter sweep test with ${paramCombinations.length} combinations${testCharacter ? ` (as ${testCharacter.name})` : ''}`);

  for (const paramCombo of paramCombinations) {
    const paramLabel = createParameterLabel(paramCombo);
    const modelDisplayName = getModelDisplayName(testModel, testCharacter, paramLabel);

    // Get individual rating for THIS parameter+character combination
    let currentModelRating = eloRatings[modelDisplayName] || 1500;
    
    console.log(`🔄 Testing parameter combination: ${modelDisplayName}, starting ELO: ${currentModelRating}`);

    for (const promptData of promptCollection) {
      const response = await generateResponse(testModel, promptData.prompt, testApiUrl, paramCombo, 'model_testing', testCharacter);

      if ((judgeMode === 'automated' || judgeMode === 'both') && judgeModelName) {
        const judgment = await judgeSingleResponse(
          promptData.prompt, 
          response, 
          modelDisplayName, 
          judgeModelName, 
          judgeGpu,
          secondaryJudgeModelName, 
          secondaryJudgeGpu,
          judgeCharacter,
          secondaryJudgeCharacter
        );

        if (judgment) {
          const result = {
            id: Date.now() + Math.random(),
            promptId: promptData.id,
            prompt: promptData.prompt,
            category: promptData.category,
            model: modelDisplayName,
            baseModel: testModel,
            character: testCharacter?.name || null,
            parameters: paramCombo,
            judgeModel: judgeModelName,
            judgeCharacter: judgeCharacter?.name || null,
            secondaryJudgeModel: secondaryJudgeModelName || null,
            secondaryJudgeCharacter: secondaryJudgeCharacter?.name || null,
            response: response,
            score: judgment.score,
            feedback: judgment.feedback,
            primaryJudgment: judgment.primaryJudgment,
            secondaryJudgment: judgment.secondaryJudgment,
            reconciliationReason: judgment.reconciliationReason,
            isDualJudged: !!secondaryJudgeModelName,
            judgeType: 'automated',
            timestamp: new Date().toISOString()
          };

          setTestResults(prev => [...prev, result]);
          const oldRating = currentModelRating;
          const eloChange = calculateEloChange(judgment.score, oldRating, promptData.category);
          const newRating = Math.max(1000, oldRating + eloChange);
          
          console.log(`🎯 ELO Update for ${modelDisplayName}:`);
          console.log(`  Judge Score: ${judgment.score}/100`);
          console.log(`  Old Rating: ${oldRating}`);
          console.log(`  ELO Change: ${eloChange}`);
          console.log(`  New Rating: ${newRating}`);
          console.log(`  Category: ${promptData.category}`);
          
          currentModelRating = newRating;
          
          setEloRatings(prev => ({ ...prev, [modelDisplayName]: newRating }));
          saveEloUpdate(modelDisplayName, oldRating, newRating, judgment.score, promptData.category);
        }
      }
      completedTests++;
      setProgress((completedTests / totalTests) * 100);
    }
    
    console.log(`✅ Completed testing for ${modelDisplayName}, final ELO: ${currentModelRating}`);
  }
  
  console.log(`🏁 Parameter sweep test complete. Processed ${newResults.length} results.`);
  
}, [
  // Add new dependencies
  selectedTestCharacter, selectedPrimaryJudgeCharacter, selectedSecondaryJudgeCharacter,
  customJudgingCriteria, getModelDisplayName, buildJudgePrompt,
  // Existing dependencies
  modelPurposes, promptCollection, judgeMode, generateResponse, judgeSingleResponse, 
  calculateEloChange, eloRatings, saveEloUpdate, PRIMARY_API_URL, SECONDARY_API_URL, 
  generateParameterCombinations, createParameterLabel
]);

// Multi-model comparison function
const runComparisonTest = useCallback(async () => {
  const modelA = modelPurposes.test_model_a.name;
  const modelAGpu = modelPurposes.test_model_a.gpu_id;
  const modelAApiUrl = modelAGpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
  const characterA = selectedTestCharacterA;

  const modelB = modelPurposes.test_model_b.name;
  const modelBGpu = modelPurposes.test_model_b.gpu_id;
  const modelBApiUrl = modelBGpu === 0 ? PRIMARY_API_URL : SECONDARY_API_URL;
  const characterB = selectedTestCharacterB;

  let judgeModelName = null;
  let judgeGpu = null;
  let secondaryJudgeModelName = null;
  let secondaryJudgeGpu = null;
  let judgeCharacter = selectedPrimaryJudgeCharacter;
  let secondaryJudgeCharacter = selectedSecondaryJudgeCharacter;

  if (judgeMode === 'automated' || judgeMode === 'both') {
    if(modelPurposes.primary_judge) {
        judgeModelName = modelPurposes.primary_judge.name;
        judgeGpu = modelPurposes.primary_judge.gpu_id;
    }
    if (modelPurposes.secondary_judge) {
        secondaryJudgeModelName = modelPurposes.secondary_judge.name;
        secondaryJudgeGpu = modelPurposes.secondary_judge.gpu_id;
    }
  }

  const totalTests = promptCollection.length;
  let completedTests = 0;
  const newResults = [];

  const modelADisplayName = getModelDisplayName(modelA, characterA);
  const modelBDisplayName = getModelDisplayName(modelB, characterB);

  console.log(`🎯 Starting comparison test: ${modelADisplayName} vs ${modelBDisplayName}`);

  for (const promptData of promptCollection) {
    const responseA = await generateResponse(modelA, promptData.prompt, modelAApiUrl, {}, 'model_testing', characterA);
    const responseB = await generateResponse(modelB, promptData.prompt, modelBApiUrl, {}, 'model_testing', characterB);
    
    if ((judgeMode === 'automated' || judgeMode === 'both') && judgeModelName) {
      const judgment = await judgeWithModel(
        promptData.prompt,
        responseA,
        responseB,
        modelADisplayName,
        modelBDisplayName,
        judgeModelName,
        judgeGpu,
        secondaryJudgeModelName,
        secondaryJudgeGpu,
        judgeCharacter,
        secondaryJudgeCharacter
      );

      if (judgment) {
        const result = {
          id: Date.now() + Math.random(),
          promptId: promptData.id,
          prompt: promptData.prompt,
          category: promptData.category,
          model1: modelADisplayName,
          model2: modelBDisplayName,
          character1: characterA?.name || null,
          character2: characterB?.name || null,
          response1: responseA,
          response2: responseB,
          judgment: judgment.verdict,
          explanation: judgment.explanation,
          judgeModel: judgeModelName,
          judgeCharacter: judgeCharacter?.name || null,
          secondaryJudgeModel: secondaryJudgeModelName || null,
          secondaryJudgeCharacter: secondaryJudgeCharacter?.name || null,
          primaryJudgment: judgment.primaryJudgment,
          secondaryJudgment: judgment.secondaryJudgment,
          reconciliationReason: judgment.reconciliationReason,
          isDualJudged: !!secondaryJudgeModelName,
          judgeType: 'automated',
          timestamp: new Date().toISOString()
        };
        setTestResults(prev => [...prev, result]);
        
        if (judgment.verdict === 'A') {
          updateEloRatings(modelADisplayName, modelBDisplayName);
        } else if (judgment.verdict === 'B') {
          updateEloRatings(modelBDisplayName, modelADisplayName);
        } else if (judgment.verdict === 'TIE') {
          updateEloRatings(modelADisplayName, modelBDisplayName, true);
        }
      }
    }
    completedTests++;
    setProgress((completedTests / totalTests) * 100);
  }
  
}, [
  // Add new dependencies
  selectedTestCharacterA, selectedTestCharacterB, selectedPrimaryJudgeCharacter, selectedSecondaryJudgeCharacter,
  customJudgingCriteria, getModelDisplayName,
  // Existing dependencies
  modelPurposes, promptCollection, judgeMode, generateResponse, judgeWithModel, updateEloRatings, 
  PRIMARY_API_URL, SECONDARY_API_URL
]);


// Add these two functions inside your ModelTester component


  // Handle human judgment
  const handleHumanJudgment = useCallback((resultId, judgment) => {
    setTestResults(prev => prev.map(result => {
      if (result.id === resultId) {
        const updatedResult = { ...result, humanJudgment: judgment };
        
        // Update ELO based on human judgment
        if (judgment === 'A') {
          updateEloRatings(result.model1, result.model2);
        } else if (judgment === 'B') {
          updateEloRatings(result.model2, result.model1);
        } else if (judgment === 'TIE') {
          updateEloRatings(result.model1, result.model2, true);
        }
        
        return updatedResult;
      }
      return result;
    }));
  }, [updateEloRatings]);

  return (
    <div className="w-full min-h-screen p-4 space-y-4">
<div className="flex items-center justify-between">
  <h1 className="text-3xl font-bold">Model ELO Tester</h1>
  <div className="flex items-center gap-3">
    <Button
      variant="outline"
      size="sm"
      onClick={resetEloRatings}
      className="text-red-600 hover:text-red-700"
    >
      <RefreshCw className="w-4 h-4 mr-2" />
      Reset All ELO
    </Button>
    <Badge variant="outline" className="text-lg px-3 py-1">
      <Trophy className="w-4 h-4 mr-1" />
      ELO Rating System
    </Badge>
  </div>
</div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
<TabsList className="flex w-full h-10">
  <TabsTrigger value="setup" className="flex-1 text-sm">Setup</TabsTrigger>
  <TabsTrigger value="prompts" className="flex-1 text-sm">Prompts</TabsTrigger>
  <TabsTrigger value="parameters" className="flex-1 text-sm">Parameters</TabsTrigger>
  <TabsTrigger value="testing" className="flex-1 text-sm">Testing</TabsTrigger>
  <TabsTrigger value="results" className="flex-1 text-sm">Results</TabsTrigger>
  <TabsTrigger value="analysis" className="flex-1 text-sm">Analysis</TabsTrigger>
</TabsList>
<div className="bg-blue-50 dark:bg-blue-950/50 border border-blue-200 dark:border-blue-800/50 rounded-lg p-4 my-4">
  <div className="flex items-center justify-between">
    <div>
      <h3 className="font-semibold text-base text-blue-900 dark:text-blue-200">Ready to Start Testing?</h3>
      <div className="text-sm text-blue-700 dark:text-blue-400 mt-1">
        {promptCollection.length} prompts loaded
        {parameterSweepEnabled && (
          <span className="mx-2">•</span>
        )}
        {parameterSweepEnabled && (
          <span>{calculateTotalCombinations()} parameter combinations</span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        Total tests to run: {promptCollection.length * (parameterSweepEnabled ? calculateTotalCombinations() : 1)}
      </p>
    </div>
    <Button
      onClick={runAutomatedTest}
      disabled={
        (testingMode === 'single' && (
          !modelPurposes?.test_model ||
          ((judgeMode === 'automated' || judgeMode === 'both') && !modelPurposes?.primary_judge) ||
          promptCollection.length === 0
        )) ||
        (testingMode === 'comparison' && (
          !modelPurposes?.test_model_a ||
          !modelPurposes?.test_model_b ||
          ((judgeMode === 'automated' || judgeMode === 'both') && !modelPurposes?.primary_judge) ||
          promptCollection.length === 0
        )) ||
        isRunning
      }
      size="lg"
      className="bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 dark:text-white"
    >
      {isRunning ? (
        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
      ) : (
        <Play className="w-5 h-5 mr-2" />
      )}
      {isRunning ? 'Testing...' : 'Start Test'}
    </Button>
  </div>
</div>
        {/* Setup Tab */}
<TabsContent value="setup" className="space-y-6">
  <Card>
    <CardHeader>
      <CardTitle>Testing Mode</CardTitle>
    </CardHeader>
    <CardContent className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-2 block">Select Testing Mode</label>
        <Select value={testingMode} onValueChange={setTestingMode}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="single">Single Model Testing (test one model with judge)</SelectItem>
            <SelectItem value="comparison">Multi-Model Comparison (traditional ELO)</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </CardContent>
  </Card>

  <Card>
    <CardHeader>
      <CardTitle>Purpose-Based Model Loading</CardTitle>
      <p className="text-sm text-muted-foreground">
        Load models for specific testing purposes. Tests will run with whatever is currently loaded.
      </p>
    </CardHeader>
    <CardContent className="space-y-6">
      
      {/* Test Model Section - Single Model Mode */}
      {testingMode === 'single' && (
        <div className="border rounded p-4 bg-muted/30">
          <h4 className="font-medium mb-3 flex items-center gap-2">
            <Bot className="w-4 h-4" />
            Test Model (answers prompts)
          </h4>
          
          {modelPurposes?.test_model ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                <div className="flex-1">
                  <div className="font-medium">{modelPurposes.test_model.name}</div>
                  <div className="text-sm text-muted-foreground">
                    GPU {modelPurposes.test_model.gpu_id} • Context: {modelPurposes.test_model.context_length}
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => unloadModelPurpose('test_model')}
                  disabled={isUnloadingPurpose.test_model}
                  className="text-red-600 hover:text-red-700"
                >
                  {isUnloadingPurpose.test_model ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unload'}
                </Button>
              </div>
              
              {/* Character Selection for Test Model */}
              <div className="border rounded p-3 bg-blue-50 dark:bg-blue-900/20">
                <label className="text-sm font-medium mb-2 block">Character for Test Model (Optional)</label>
                <Select value={selectedTestCharacter?.id || 'none'} onValueChange={(id) => {
                  const character = characters.find(c => c.id === id);
                  setSelectedTestCharacter(character || null);
                }}>
                  <SelectTrigger>
                    <SelectValue placeholder="No character (default behavior)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">No Character</SelectItem>
                    {characters.map(char => (
                      <SelectItem key={char.id} value={char.id}>
                        {char.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedTestCharacter && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Model will respond as: {selectedTestCharacter.name}
                  </p>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex gap-2">
                <Select value={selectedTestModel} onValueChange={setSelectedTestModel} className="flex-1">
                  <SelectTrigger>
                    <SelectValue placeholder="Select model to test" />
                  </SelectTrigger>
                  <SelectContent>
                    {allModelOptions.map(model => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                
                <Select 
                  value={selectedTestModelGpu?.toString() || "0"}
                  onValueChange={(value) => setSelectedTestModelGpu(parseInt(value))}
                >
                  <SelectTrigger className="w-24">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">GPU 0</SelectItem>
                    <SelectItem value="1">GPU 1</SelectItem>
                  </SelectContent>
                </Select>
                
                <Button
                  onClick={() => loadModelForPurpose('test_model', selectedTestModel, selectedTestModelGpu, settings.contextLength)}
                  disabled={!selectedTestModel || isLoadingPurpose.test_model}
                  size="sm"
                >
                  {isLoadingPurpose.test_model ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Load'}
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Multi-Model Selection - Comparison Mode */}
      {testingMode === 'comparison' && (
        <div className="border rounded p-4 bg-muted/30">
          <h4 className="font-medium mb-3 flex items-center gap-2">
            <Users className="w-4 h-4" />
            Competing Models (load 2+ models to compare against each other)
          </h4>
          
          <div className="space-y-4">
            {/* Test Model A */}
            <div className="border rounded p-3 bg-muted/50">
              <h5 className="font-medium mb-2 text-sm">Test Model A</h5>
              {modelPurposes?.test_model_a ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                    <div className="flex-1">
                      <div className="font-medium">{modelPurposes.test_model_a.name}</div>
                      <div className="text-sm text-muted-foreground">
                        GPU {modelPurposes.test_model_a.gpu_id} • Context: {modelPurposes.test_model_a.context_length}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => unloadModelPurpose('test_model_a')}
                      disabled={isUnloadingPurpose.test_model_a}
                      className="text-red-600 hover:text-red-700"
                    >
                      {isUnloadingPurpose.test_model_a ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unload'}
                    </Button>
                  </div>
                  
                  {/* Character Selection for Test Model A */}
                  <div className="border rounded p-3 bg-blue-50 dark:bg-blue-900/20">
                    <label className="text-sm font-medium mb-2 block">Character for Model A (Optional)</label>
                    <Select value={selectedTestCharacterA?.id || 'none'} onValueChange={(id) => {
                      const character = characters.find(c => c.id === id);
                      setSelectedTestCharacterA(character || null);
                    }}>
                      <SelectTrigger>
                        <SelectValue placeholder="No character (default behavior)" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">No Character</SelectItem>
                        {characters.map(char => (
                          <SelectItem key={char.id} value={char.id}>
                            {char.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              ) : (
                <div className="flex gap-2">
                  <Select value={selectedModels[0] || ''} onValueChange={(value) => {
                    setSelectedModels(prev => [value, prev[1] || '']);
                    // Initialize GPU assignment if not set (default to GPU 1 for Model A)
                    if (value && !modelGpuAssignments[value]) {
                      setModelGpuAssignments(prev => ({ ...prev, [value]: 1 }));
                    }
                  }} className="flex-1">
                    <SelectTrigger>
                      <SelectValue placeholder="Select first test model" />
                    </SelectTrigger>
                    <SelectContent>
                      {allModelOptions.map(model => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  
                  <Select 
                    value={(modelGpuAssignments[selectedModels[0] || ''] ?? 0).toString()}
                    onValueChange={(value) => {
                      const modelName = selectedModels[0] || '';
                      const gpuId = parseInt(value);
                      setModelGpuAssignments(prev => ({ 
                        ...prev, 
                        [modelName]: gpuId 
                      }));
                      console.log(`[GPU Assignment] Set ${modelName} to GPU ${gpuId}`);
                      console.log(`[GPU Assignment] Updated state. All assignments:`, JSON.stringify({ ...modelGpuAssignments, [modelName]: gpuId }));
                    }}
                  >
                    <SelectTrigger className="w-24">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">GPU 0</SelectItem>
                      <SelectItem value="1">GPU 1</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  <Button
                    onClick={() => {
                      const modelName = selectedModels[0];
                      if (!modelName) {
                        alert('Please select a model first');
                        return;
                      }
                      
                      // Read GPU assignment from state - use explicit fallback
                      const currentGpuAssignment = modelGpuAssignments[modelName];
                      const gpuId = currentGpuAssignment !== undefined && currentGpuAssignment !== null 
                        ? currentGpuAssignment 
                        : 1; // Default to GPU 1 for Model A
                      
                      console.log(`========================================`);
                      console.log(`[Load Model A] STARTING LOAD`);
                      console.log(`[Load Model A] Model Name: "${modelName}"`);
                      console.log(`[Load Model A] All GPU Assignments:`, JSON.stringify(modelGpuAssignments));
                      console.log(`[Load Model A] Assignment for this model:`, currentGpuAssignment);
                      console.log(`[Load Model A] FINAL GPU ID: ${gpuId}`);
                      console.log(`[Load Model A] Will route to: ${gpuId === 0 ? 'PRIMARY_API_URL (port 8000)' : 'SECONDARY_API_URL (port 8001)'}`);
                      console.log(`========================================`);
                      
                      loadModelForPurpose('test_model_a', modelName, gpuId, settings.contextLength);
                    }}
                    disabled={!selectedModels[0] || isLoadingPurpose.test_model_a}
                    size="sm"
                  >
                    {isLoadingPurpose.test_model_a ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Load'}
                  </Button>
                </div>
              )}
            </div>

            {/* Test Model B - Similar structure to Model A */}
            <div className="border rounded p-3 bg-muted/50">
              <h5 className="font-medium mb-2 text-sm">Test Model B</h5>
              {modelPurposes?.test_model_b ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                    <div className="flex-1">
                      <div className="font-medium">{modelPurposes.test_model_b.name}</div>
                      <div className="text-sm text-muted-foreground">
                        GPU {modelPurposes.test_model_b.gpu_id} • Context: {modelPurposes.test_model_b.context_length}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => unloadModelPurpose('test_model_b')}
                      disabled={isUnloadingPurpose.test_model_b}
                      className="text-red-600 hover:text-red-700"
                    >
                      {isUnloadingPurpose.test_model_b ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unload'}
                    </Button>
                  </div>
                  
                  {/* Character Selection for Test Model B */}
                  <div className="border rounded p-3 bg-blue-50 dark:bg-blue-900/20">
                    <label className="text-sm font-medium mb-2 block">Character for Model B (Optional)</label>
                    <Select value={selectedTestCharacterB?.id || 'none'} onValueChange={(id) => {
                      const character = characters.find(c => c.id === id);
                      setSelectedTestCharacterB(character || null);
                    }}>
                      <SelectTrigger>
                        <SelectValue placeholder="No character (default behavior)" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">No Character</SelectItem>
                        {characters.map(char => (
                          <SelectItem key={char.id} value={char.id}>
                            {char.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              ) : (
                <div className="flex gap-2">
                  <Select value={selectedModels[1] || ''} onValueChange={(value) => {
                    setSelectedModels(prev => [prev[0] || '', value]);
                    // Initialize GPU assignment if not set (default to GPU 0 for Model B to balance load)
                    if (value && !modelGpuAssignments[value]) {
                      setModelGpuAssignments(prev => ({ ...prev, [value]: 0 }));
                    }
                  }} className="flex-1">
                    <SelectTrigger>
                      <SelectValue placeholder="Select second test model" />
                    </SelectTrigger>
                    <SelectContent>
                      {allModelOptions.map(model => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  
                  <Select 
                    value={(modelGpuAssignments[selectedModels[1] || ''] ?? 0).toString()}
                    onValueChange={(value) => {
                      const modelName = selectedModels[1] || '';
                      const gpuId = parseInt(value);
                      setModelGpuAssignments(prev => ({ 
                        ...prev, 
                        [modelName]: gpuId 
                      }));
                      console.log(`[GPU Assignment] Set ${modelName} to GPU ${gpuId}`);
                      console.log(`[GPU Assignment] Updated state. All assignments:`, JSON.stringify({ ...modelGpuAssignments, [modelName]: gpuId }));
                    }}
                  >
                    <SelectTrigger className="w-24">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0">GPU 0</SelectItem>
                      <SelectItem value="1">GPU 1</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  <Button
                    onClick={() => {
                      const modelName = selectedModels[1];
                      if (!modelName) {
                        alert('Please select a model first');
                        return;
                      }
                      
                      // Read GPU assignment from state - use explicit fallback
                      const currentGpuAssignment = modelGpuAssignments[modelName];
                      const gpuId = currentGpuAssignment !== undefined && currentGpuAssignment !== null 
                        ? currentGpuAssignment 
                        : 0; // Default to GPU 0 for Model B
                      
                      console.log(`========================================`);
                      console.log(`[Load Model B] STARTING LOAD`);
                      console.log(`[Load Model B] Model Name: "${modelName}"`);
                      console.log(`[Load Model B] All GPU Assignments:`, JSON.stringify(modelGpuAssignments));
                      console.log(`[Load Model B] Assignment for this model:`, currentGpuAssignment);
                      console.log(`[Load Model B] FINAL GPU ID: ${gpuId}`);
                      console.log(`[Load Model B] Will route to: ${gpuId === 0 ? 'PRIMARY_API_URL (port 8000)' : 'SECONDARY_API_URL (port 8001)'}`);
                      console.log(`========================================`);
                      
                      loadModelForPurpose('test_model_b', modelName, gpuId, settings.contextLength);
                    }}
                    disabled={!selectedModels[1] || isLoadingPurpose.test_model_b}
                    size="sm"
                  >
                    {isLoadingPurpose.test_model_b ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Load'}
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Primary Judge Section - Enhanced with Character Selection */}
      <div className="border rounded p-4 bg-muted/30">
        <h4 className="font-medium mb-3 flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Primary Judge Model (evaluates responses)
        </h4>
        
        {modelPurposes?.primary_judge ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
              <div className="flex-1">
                <div className="font-medium">{modelPurposes.primary_judge.name}</div>
                <div className="text-sm text-muted-foreground">
                  GPU {modelPurposes.primary_judge.gpu_id} • Context: {modelPurposes.primary_judge.context_length}
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => unloadModelPurpose('primary_judge')}
                disabled={isUnloadingPurpose.primary_judge}
                className="text-red-600 hover:text-red-700"
              >
                {isUnloadingPurpose.primary_judge ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unload'}
              </Button>
            </div>
            
            {/* Character Selection for Primary Judge */}
            <div className="border rounded p-3 bg-purple-50 dark:bg-purple-900/20">
              <label className="text-sm font-medium mb-2 block">Character for Primary Judge (Optional)</label>
              <Select value={selectedPrimaryJudgeCharacter?.id || 'none'} onValueChange={(id) => {
                const character = characters.find(c => c.id === id);
                setSelectedPrimaryJudgeCharacter(character || null);
              }}>
                <SelectTrigger>
                  <SelectValue placeholder="No character (standard judging)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No Character</SelectItem>
                  {characters.map(char => (
                    <SelectItem key={char.id} value={char.id}>
                      {char.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedPrimaryJudgeCharacter && (
                <p className="text-xs text-muted-foreground mt-1">
                  Judge will evaluate as: {selectedPrimaryJudgeCharacter.name}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-2">
              <Select value={judgeModel} onValueChange={setJudgeModel} className="flex-1">
                <SelectTrigger>
                  <SelectValue placeholder="Select judge model" />
                </SelectTrigger>
                <SelectContent>
                  {allModelOptions.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Select 
                value={judgeModelGpu?.toString() || "1"}
                onValueChange={(value) => setJudgeModelGpu(parseInt(value))}
              >
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">GPU 0</SelectItem>
                  <SelectItem value="1">GPU 1</SelectItem>
                </SelectContent>
              </Select>
              
              <Button
                onClick={() => loadModelForPurpose('primary_judge', judgeModel, judgeModelGpu, settings.contextLength)}
                disabled={!judgeModel || isLoadingPurpose.primary_judge}
                size="sm"
              >
                {isLoadingPurpose.primary_judge ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Load'}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Secondary Judge Section - Enhanced with Character Selection */}
      <div className="border rounded p-4 bg-muted/30">
        <h4 className="font-medium mb-3 flex items-center gap-2">
          <Users className="w-4 h-4" />
          Secondary Judge Model (optional - for dual judging)
        </h4>
        
        {modelPurposes?.secondary_judge ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded">
              <div className="flex-1">
                <div className="font-medium">{modelPurposes.secondary_judge.name}</div>
                <div className="text-sm text-muted-foreground">
                  GPU {modelPurposes.secondary_judge.gpu_id} • Context: {modelPurposes.secondary_judge.context_length}
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => unloadModelPurpose('secondary_judge')}
                disabled={isUnloadingPurpose.secondary_judge}
                className="text-red-600 hover:text-red-700"
              >
                {isUnloadingPurpose.secondary_judge ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unload'}
              </Button>
            </div>
            
            {/* Character Selection for Secondary Judge */}
            <div className="border rounded p-3 bg-purple-50 dark:bg-purple-900/20">
              <label className="text-sm font-medium mb-2 block">Character for Secondary Judge (Optional)</label>
              <Select value={selectedSecondaryJudgeCharacter?.id || 'none'} onValueChange={(id) => {
                const character = characters.find(c => c.id === id);
                setSelectedSecondaryJudgeCharacter(character || null);
              }}>
                <SelectTrigger>
                  <SelectValue placeholder="No character (standard judging)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No Character</SelectItem>
                  {characters.map(char => (
                    <SelectItem key={char.id} value={char.id}>
                      {char.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedSecondaryJudgeCharacter && (
                <p className="text-xs text-muted-foreground mt-1">
                  Secondary judge will evaluate as: {selectedSecondaryJudgeCharacter.name}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-2">
              <Select value={secondaryJudgeModel} onValueChange={setSecondaryJudgeModel} className="flex-1">
                <SelectTrigger>
                  <SelectValue placeholder="Select secondary judge (optional)" />
                </SelectTrigger>
                <SelectContent>
                  {allModelOptions.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Select 
                value={secondaryJudgeModelGpu?.toString() || "0"}
                onValueChange={(value) => setSecondaryJudgeModelGpu(parseInt(value))}
              >
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">GPU 0</SelectItem>
                  <SelectItem value="1">GPU 1</SelectItem>
                </SelectContent>
              </Select>
              
              <Button
                onClick={() => loadModelForPurpose('secondary_judge', secondaryJudgeModel, secondaryJudgeModelGpu, settings.contextLength)}
                disabled={!secondaryJudgeModel || isLoadingPurpose.secondary_judge}
                size="sm"
              >
                {isLoadingPurpose.secondary_judge ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Load'}
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Custom Judging Criteria Section */}
      <div className="border rounded p-4 bg-muted/30">
        <h4 className="font-medium mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Custom Judging Criteria (Optional)
        </h4>
        <Textarea
          placeholder="Enter custom judging criteria here. If empty, will use standard criteria. Example: 'Focus on creativity and originality. Heavily penalize generic responses. Reward unexpected insights and unique perspectives.'"
          value={customJudgingCriteria}
          onChange={(e) => setCustomJudgingCriteria(e.target.value)}
          rows={3}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground mt-2">
          This will override the default judging prompt. Useful for character-specific evaluation criteria.
        </p>
      </div>

      {/* Test Ready Status */}
      <div className="bg-muted/50 border rounded p-4">
        <div className="font-medium mb-2 text-foreground">Test Readiness:</div>
        <div className="text-sm space-y-1">
          <div className={`flex items-center gap-2 ${modelPurposes?.test_model || (testingMode === 'comparison' && selectedModels.length >= 2) ? 'text-green-400' : 'text-red-400'}`}>
            {modelPurposes?.test_model || (testingMode === 'comparison' && selectedModels.length >= 2) ? '✅' : '❌'} 
            {testingMode === 'single' ? 'Test Model Ready' : `${selectedModels.length}/2+ Models Selected`}
          </div>
          <div className={`flex items-center gap-2 ${modelPurposes?.primary_judge ? 'text-green-400' : 'text-red-400'}`}>
            {modelPurposes?.primary_judge ? '✅' : '❌'} Primary Judge Ready
          </div>
          <div className={`flex items-center gap-2 ${modelPurposes?.secondary_judge ? 'text-green-400' : 'text-gray-400'}`}>
            {modelPurposes?.secondary_judge ? '✅' : '⚪'} Secondary Judge {modelPurposes?.secondary_judge ? 'Ready' : '(Optional)'}
          </div>
          
          {/* Character Status */}
          {(selectedTestCharacter || selectedTestCharacterA || selectedTestCharacterB || selectedPrimaryJudgeCharacter || selectedSecondaryJudgeCharacter) && (
            <div className="mt-2 pt-2 border-t border-muted">
              <div className="text-xs font-medium text-muted-foreground mb-1">Character Assignments:</div>
              {selectedTestCharacter && (
                <div className="text-xs text-blue-400">🎭 Test Model: {selectedTestCharacter.name}</div>
              )}
              {selectedTestCharacterA && (
                <div className="text-xs text-blue-400">🎭 Model A: {selectedTestCharacterA.name}</div>
              )}
              {selectedTestCharacterB && (
                <div className="text-xs text-blue-400">🎭 Model B: {selectedTestCharacterB.name}</div>
              )}
              {selectedPrimaryJudgeCharacter && (
                <div className="text-xs text-purple-400">⚖️ Primary Judge: {selectedPrimaryJudgeCharacter.name}</div>
              )}
              {selectedSecondaryJudgeCharacter && (
                <div className="text-xs text-purple-400">⚖️ Secondary Judge: {selectedSecondaryJudgeCharacter.name}</div>
              )}
            </div>
          )}
        </div>
      </div>

    </CardContent>
  </Card>
</TabsContent>

        {/* Prompts Tab */}
<TabsContent value="prompts" className="space-y-6">
  <Card>
    <CardHeader>
      <CardTitle>Prompt Collection</CardTitle>
    </CardHeader>
    <CardContent className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.jsonl"
          onChange={handleImportCollection}
          className="hidden"
        />
        <Button 
          variant="outline"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-4 h-4 mr-2" />
          Import JSON/JSONL
        </Button>

        <Select onValueChange={loadDefaultCollection}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Load default collection" />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(defaultCollections).map(name => (
              <SelectItem key={name} value={name}>{name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div>
        <Input
          placeholder="Collection name"
          value={collectionName}
          onChange={(e) => setCollectionName(e.target.value)}
        />
      </div>

      {/* ADD MANUAL PROMPT CREATION */}
      <div className="border rounded p-4 bg-muted/30">
        <h4 className="font-medium mb-3">Add New Prompt</h4>
        <div className="space-y-2">
          <Input
            placeholder="Category (e.g., coding, writing, reasoning)"
            value={newPromptCategory}
            onChange={(e) => setNewPromptCategory(e.target.value)}
          />
          <Textarea
            placeholder="Enter your prompt here..."
            value={newPromptText}
            onChange={(e) => setNewPromptText(e.target.value)}
            rows={3}
          />
          <Button 
            onClick={addManualPrompt}
            disabled={!newPromptText.trim()}
            size="sm"
          >
            Add Prompt
          </Button>
        </div>
      </div>

      <div>
        <ScrollArea className="h-64 border rounded p-4">
          {promptCollection.length > 0 ? (
            <div className="space-y-2">
              {promptCollection.map((prompt, index) => (
                <div key={prompt.id || index} className="p-2 border rounded flex justify-between items-start">
                  <div className="flex-1">
                    <div className="text-sm font-medium">{prompt.category}</div>
                    <div className="text-sm text-gray-600">{prompt.prompt}</div>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => removePrompt(prompt.id || index)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500">
              No prompts loaded. Import a JSON file, select a default collection, or add prompts manually.
            </div>
          )}
        </ScrollArea>
      </div>
    </CardContent>
  </Card>
</TabsContent>
{/* Parameters Tab */}
<TabsContent value="parameters" className="space-y-6">
  <Card>
    <CardHeader>
      <CardTitle>Parameter Sweeping</CardTitle>
      <CardDescription>
        Test multiple parameter combinations automatically. Only works with local GGUF models.
      </CardDescription>
    </CardHeader>
    <CardContent className="space-y-6">
      
      {/* Enable parameter sweeping */}
      <div className="flex items-center space-x-2">
        <Switch 
          id="enable-sweep" 
          checked={parameterSweepEnabled}
          onCheckedChange={setParameterSweepEnabled}
        />
        <Label htmlFor="enable-sweep" className="text-sm font-medium">
          Enable Parameter Sweeping
        </Label>
      </div>

      {parameterSweepEnabled && (
        <>
          <Separator />
          
          {/* API Model Warning */}
          {(testingMode === 'single' && isApiModel(modelPurposes?.test_model?.name)) || 
           (testingMode === 'comparison' && (isApiModel(modelPurposes?.test_model_a?.name) || isApiModel(modelPurposes?.test_model_b?.name))) && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>API Model Limitation</AlertTitle>
              <AlertDescription>
                Parameter sweeping only works with local GGUF models. API models will use their default parameters.
              </AlertDescription>
            </Alert>
          )}
          
          {/* Parameter Configuration */}
          <div className="space-y-4">
            <h4 className="font-medium">Parameter Ranges</h4>
            
            {Object.entries(parameterConfig).map(([paramKey, config]) => (
              <div key={paramKey} className="border rounded p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={config.enabled}
                      onCheckedChange={(enabled) => 
                        setParameterConfig(prev => ({
                          ...prev,
                          [paramKey]: { ...prev[paramKey], enabled }
                        }))
                      }
                    />
                    <Label className="font-medium capitalize">
                      {paramKey.replace('_', '-')}
                    </Label>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    Current: {config.current}
                  </span>
                </div>
                
                {config.enabled && (
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <Label className="text-xs">Min</Label>
                      <Input
                        type="number"
                        value={config.min}
                        step={config.step}
                        onChange={(e) => 
                          setParameterConfig(prev => ({
                            ...prev,
                            [paramKey]: { 
                              ...prev[paramKey], 
                              min: parseFloat(e.target.value) 
                            }
                          }))
                        }
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Max</Label>
                      <Input
                        type="number"
                        value={config.max}
                        step={config.step}
                        onChange={(e) => 
                          setParameterConfig(prev => ({
                            ...prev,
                            [paramKey]: { 
                              ...prev[paramKey], 
                              max: parseFloat(e.target.value) 
                            }
                          }))
                        }
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Step</Label>
                      <Input
                        type="number"
                        value={config.step}
                        step={config.step}
                        onChange={(e) => 
                          setParameterConfig(prev => ({
                            ...prev,
                            [paramKey]: { 
                              ...prev[paramKey], 
                              step: parseFloat(e.target.value) 
                            }
                          }))
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <Separator />

          {/* Combination Summary */}
          <div className={`p-4 rounded border ${
            calculateTotalCombinations() > 200 ? 'bg-red-50 border-red-200 dark:bg-red-900/20' : 
            calculateTotalCombinations() > 50 ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20' : 
            'bg-green-50 border-green-200 dark:bg-green-900/20'
          }`}>
            <div className="flex items-center gap-2">
              {calculateTotalCombinations() > 200 && <AlertTriangle className="w-5 h-5 text-red-600" />}
              {calculateTotalCombinations() > 50 && calculateTotalCombinations() <= 200 && <AlertTriangle className="w-5 h-5 text-yellow-600" />}
              <span className="font-medium">
                Total Combinations: {calculateTotalCombinations()}
              </span>
            </div>
            
            {calculateTotalCombinations() > 200 && (
              <p className="text-sm text-red-600 mt-1">
                ⚠️ This will take a very long time! Consider reducing parameter ranges.
              </p>
            )}
            {calculateTotalCombinations() > 50 && calculateTotalCombinations() <= 200 && (
              <p className="text-sm text-yellow-600 mt-1">
                ⚠️ This will take a while. Each combination × number of prompts = total tests.
              </p>
            )}
            {calculateTotalCombinations() <= 50 && (
              <p className="text-sm text-green-600 mt-1">
                ✓ Reasonable number of combinations for testing.
              </p>
            )}
            
            <p className="text-xs text-muted-foreground mt-2">
              With {promptCollection.length} prompts: {calculateTotalCombinations() * promptCollection.length} total tests
            </p>
          </div>
        </>
      )}
    </CardContent>
  </Card>
</TabsContent>

        {/* Testing Tab */}
        <TabsContent value="testing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Test Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Progress value={progress} className="w-full" />
                <div className="text-center text-sm text-gray-600">
                  {isRunning ? `Testing in progress... ${Math.round(progress)}%` : 'Test completed'}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

{/* Results Tab */}
<TabsContent value="results" className="space-y-6">
  {/* ALWAYS SHOW IMPORT, CONDITIONALLY SHOW EXPORT */}
  <div className="flex justify-end gap-2">
    {/* Add file input (hidden) */}
    <input
      type="file"
      accept=".json"
      onChange={handleImportResults}
      style={{ display: 'none' }}
      ref={fileInputRef}
    />
    
    {/* Import button - ALWAYS VISIBLE */}
    <Button 
      variant="outline" 
      onClick={() => fileInputRef.current?.click()}
    >
      <Upload className="w-4 h-4 mr-2" />
      Import Results
    </Button>
    
    {/* Export button - ONLY WHEN RESULTS EXIST */}
    {testResults.length > 0 && (
      <Button variant="outline" onClick={exportResults}>
        <Download className="w-4 h-4 mr-2" />
        Export JSON Results
      </Button>
    )}
  </div>
  
{/* Top row: Leaderboard and Recent Comparisons side by side */}
<div className="grid grid-cols-2 gap-6 mb-6">
 <Card>
  <CardHeader>
    <div className="flex items-center justify-between">
      <CardTitle>ELO Leaderboard</CardTitle>
      <Button
        variant="outline"
        size="sm"
        onClick={resetEloRatings}
        className="text-red-600 hover:text-red-700"
      >
        Reset ELO
      </Button>
    </div>
  </CardHeader>
  <CardContent>
    <div className="space-y-2">
      {Object.entries(eloRatings)
        .sort(([,a], [,b]) => b - a)
        .map(([modelKey, rating], index) => {
          // Parse model display name to extract base model and character
          const isCharacterModel = modelKey.includes(' (as ');
          let displayName = modelKey;
          let characterBadge = null;
          
          if (isCharacterModel) {
            const match = modelKey.match(/^(.+?) \(as (.+?)\)(.*)$/);
            if (match) {
              const [, baseModel, characterName, paramSuffix] = match;
              displayName = baseModel + (paramSuffix || '');
              characterBadge = characterName;
            }
          }
          
          return (
            <div key={modelKey} className="flex items-center justify-between p-2 border rounded">
              <div className="flex items-center gap-2">
                <Badge variant={index === 0 ? "default" : "outline"}>
                  #{index + 1}
                </Badge>
                <div className="flex flex-col">
                  <span className="font-medium text-sm">{displayName}</span>
                  {characterBadge && (
                    <div className="flex items-center gap-1 mt-1">
                      <Badge variant="secondary" className="text-xs">
                        🎭 {characterBadge}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>
              <Badge variant="secondary">{rating}</Badge>
            </div>
          );
        })}
    </div>
  </CardContent>
</Card>

{/* Recent Comparisons */}
<Card>
  <CardHeader>
    <CardTitle>Recent Comparisons</CardTitle>
  </CardHeader>
  <CardContent>
    <ScrollArea className="h-80">
      <div className="space-y-2">
        {testResults.filter(result => result.model1 && result.model2).reverse().map(result => (
          <div key={result.id} className="p-2 border rounded text-sm">
            <div className="font-medium text-xs text-gray-500 mb-1">
              {result.category} • {new Date(result.timestamp).toLocaleTimeString()}
            </div>
            <div className="mb-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">{result.model1}</span>
                {result.character1 && (
                  <Badge variant="outline" className="text-xs">🎭 {result.character1}</Badge>
                )}
              </div>
              <div className="text-xs text-muted-foreground">vs</div>
              <div className="flex items-center gap-2">
                <span className="font-medium">{result.model2}</span>
                {result.character2 && (
                  <Badge variant="outline" className="text-xs">🎭 {result.character2}</Badge>
                )}
              </div>
            </div>
            <div className="text-xs mb-1">
              Winner: {result.judgment === 'TIE' ? 'Tie' : 
                result.judgment === 'A' ? result.model1 : result.model2}
            </div>
            <div className="text-xs text-muted-foreground">
              Judge{result.isDualJudged ? 's' : ''}: {result.judgeModel}
              {result.judgeCharacter && (
                <span className="ml-1">🎭 {result.judgeCharacter}</span>
              )}
              {result.secondaryJudgeModel && ` + ${result.secondaryJudgeModel}`}
              {result.secondaryJudgeCharacter && (
                <span className="ml-1">🎭 {result.secondaryJudgeCharacter}</span>
              )}
              • {result.explanation?.substring(0, 40)}...
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  </CardContent>
</Card>
</div>

{/* Multi-Model Comparison Results */}
{testResults.filter(r => r.model1 && r.model2).length > 0 && (
  <Card>
    <CardHeader>
      <CardTitle>Multi-Model Comparison Results</CardTitle>
    </CardHeader>
    <CardContent>
      <ScrollArea className="h-96">
        <div className="space-y-3">
          {testResults.filter(r => r.model1 && r.model2).reverse().map(result => (
            <div key={result.id} className="border rounded p-3 bg-card text-card-foreground">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <div className="font-medium text-sm mb-1">
                    <div className="flex items-center gap-2">
                      <span>{result.model1}</span>
                      {result.character1 && (
                        <Badge variant="outline" className="text-xs">🎭 {result.character1}</Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground my-1">vs</div>
                    <div className="flex items-center gap-2">
                      <span>{result.model2}</span>
                      {result.character2 && (
                        <Badge variant="outline" className="text-xs">🎭 {result.character2}</Badge>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">{result.category} • {new Date(result.timestamp).toLocaleTimeString()}</div>
                </div>
                <Badge variant={result.judgment === 'TIE' ? "secondary" : "default"}>
                  {result.judgment === 'TIE' ? 'TIE' : 
                   result.judgment === 'A' ? `${result.model1} WINS` : 
                   `${result.model2} WINS`}
                </Badge>
              </div>
              
              <div className="text-sm mb-2">
                <strong>Judge{result.isDualJudged ? 's' : ''}:</strong> {result.judgeModel}
                {result.judgeCharacter && (
                  <Badge variant="outline" className="ml-2 text-xs">🎭 {result.judgeCharacter}</Badge>
                )}
                {result.secondaryJudgeModel && <span> + {result.secondaryJudgeModel}</span>}
                {result.secondaryJudgeCharacter && (
                  <Badge variant="outline" className="ml-2 text-xs">🎭 {result.secondaryJudgeCharacter}</Badge>
                )}
              </div>

              {result.isDualJudged && (
                <div className="text-xs mb-2 bg-blue-50 p-2 rounded">
                  <strong>Primary:</strong> {result.primaryJudgment?.verdict} • 
                  <strong> Secondary:</strong> {result.secondaryJudgment?.verdict} • 
                  <strong> Result:</strong> {result.reconciliationReason}
                </div>
              )}

              <div className="text-sm bg-muted p-2 rounded mb-2">
                <strong>Explanation:</strong> <span className="text-foreground">{result.explanation}</span>
              </div>
              
              <details className="text-sm">
                <summary className="cursor-pointer text-primary hover:text-primary/80 font-medium">
                  View Prompt & Both Responses
                </summary>
                <div className="mt-2 space-y-2">
                  <div className="bg-muted p-2 rounded border">
                    <strong>Prompt:</strong>
                    <div className="mt-1">{result.prompt}</div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div className="bg-muted p-2 rounded border">
                      <div className="flex items-center gap-2 mb-1">
                        <strong>{result.model1} Response:</strong>
                        {result.character1 && (
                          <Badge variant="outline" className="text-xs">🎭 {result.character1}</Badge>
                        )}
                      </div>
                      <div className="mt-1 whitespace-pre-wrap">{result.response1}</div>
                    </div>
                    <div className="bg-muted p-2 rounded border">
                      <div className="flex items-center gap-2 mb-1">
                        <strong>{result.model2} Response:</strong>
                        {result.character2 && (
                          <Badge variant="outline" className="text-xs">🎭 {result.character2}</Badge>
                        )}
                      </div>
                      <div className="mt-1 whitespace-pre-wrap">{result.response2}</div>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          ))}
        </div>
      </ScrollArea>
    </CardContent>
  </Card>
)}
  

{/* Single Model Test Results */}
{testResults.filter(r => r.model && !r.model1).length > 0 && (
  <Card>
    <CardHeader>
      <CardTitle>Single Model Test Results</CardTitle>
    </CardHeader>
    <CardContent>
      <ScrollArea className="h-96">
        <div className="space-y-3">
          {testResults.filter(r => r.model && !r.model1).reverse().map(result => (
            <div key={result.id} className="border rounded p-3 bg-card text-card-foreground">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{result.model}</span>
                    {result.character && (
                      <Badge variant="outline" className="text-xs">🎭 {result.character}</Badge>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground">{result.category} • {new Date(result.timestamp).toLocaleTimeString()}</div>
                </div>
                <Badge variant={result.score >= 70 ? "default" : result.score >= 40 ? "secondary" : "destructive"}>
                  {result.score}/100
                </Badge>
              </div>
              
              <div className="text-sm mb-2">
                <strong>Judge{result.isDualJudged ? 's' : ''}:</strong> {result.judgeModel}
                {result.judgeCharacter && (
                  <Badge variant="outline" className="ml-2 text-xs">🎭 {result.judgeCharacter}</Badge>
                )}
                {result.secondaryJudgeModel && <span> + {result.secondaryJudgeModel}</span>}
                {result.secondaryJudgeCharacter && (
                  <Badge variant="outline" className="ml-2 text-xs">🎭 {result.secondaryJudgeCharacter}</Badge>
                )}
              </div>

              {result.isDualJudged && (
                <div className="text-xs mb-2 bg-blue-50 p-2 rounded">
                  <strong>Primary:</strong> {result.primaryJudgment?.score}/100 • 
                  <strong> Secondary:</strong> {result.secondaryJudgment?.score}/100 • 
                  <strong> Final:</strong> {result.score}/100 • 
                  <strong> Reason:</strong> {result.reconciliationReason}
                </div>
              )}

              <div className="text-sm bg-muted p-2 rounded mb-2">
                <strong>Feedback:</strong> <span className="text-foreground">{result.feedback}</span>
              </div>
              
              <details className="text-sm">
                <summary className="cursor-pointer text-primary hover:text-primary/80 font-medium">
                  View Prompt & Response
                </summary>
                <div className="mt-2 space-y-2">
                  <div className="bg-muted p-2 rounded border">
                    <strong>Prompt:</strong>
                    <div className="mt-1">{result.prompt}</div>
                  </div>
                  <div className="bg-muted p-2 rounded border">
                    <div className="flex items-center gap-2 mb-1">
                      <strong>Model Response:</strong>
                      {result.character && (
                        <Badge variant="outline" className="text-xs">🎭 {result.character}</Badge>
                      )}
                    </div>
                    <div className="mt-1 whitespace-pre-wrap">{result.response}</div>
                  </div>
                </div>
              </details>
            </div>
          ))}
        </div>
      </ScrollArea>
    </CardContent>
  </Card>
)}
          {/* Human Judgment Interface */}
          {(judgeMode === 'human' || judgeMode === 'both') && testResults.some(r => !r.humanJudgment) && (
            <Card>
              <CardHeader>
                <CardTitle>Human Evaluation Needed</CardTitle>
              </CardHeader>
              <CardContent>
                {testResults.filter(r => !r.humanJudgment).slice(0, 1).map(result => (
                  <div key={result.id} className="space-y-4">
                    <div className="p-3 bg-gray-50 rounded">
                      <div className="font-medium mb-2">Prompt:</div>
                      <div className="text-sm">{result.prompt}</div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3 border rounded">
                        <div className="font-medium mb-2 flex items-center gap-2">
                          <Bot className="w-4 h-4" />
                          Response A ({result.model1})
                        </div>
                        <div className="text-sm">{result.response1}</div>
                      </div>
                      
                      <div className="p-3 border rounded">
                        <div className="font-medium mb-2 flex items-center gap-2">
                          <Bot className="w-4 h-4" />
                          Response B ({result.model2})
                        </div>
                        <div className="text-sm">{result.response2}</div>
                      </div>
                    </div>

                    <div className="flex justify-center gap-4">
                      <Button
                        variant="outline"
                        onClick={() => handleHumanJudgment(result.id, 'A')}
                      >
                        <ThumbsUp className="w-4 h-4 mr-2" />
                        Response A
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => handleHumanJudgment(result.id, 'TIE')}
                      >
                        <Minus className="w-4 h-4 mr-2" />
                        Tie
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => handleHumanJudgment(result.id, 'B')}
                      >
                        <ThumbsUp className="w-4 h-4 mr-2" />
                        Response B
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>
            
{/* Analysis Tab - Complete rewrite */}
<TabsContent value="analysis" className="space-y-6">
<AnalysisChat 
  testResults={testResults}
  modelPurposes={modelPurposes}
  fetchModelsByPurpose={fetchModelsByPurpose}
  loadModelForPurpose={loadModelForPurpose}
  unloadModelPurpose={unloadModelPurpose}
  isLoadingPurpose={isLoadingPurpose}
  isUnloadingPurpose={isUnloadingPurpose}
  primaryApiUrl={PRIMARY_API_URL}
  secondaryApiUrl={SECONDARY_API_URL}
  onImportResults={handleImportResults}
  setTestResults={setTestResults}
  ragSettings={settings}
/>
</TabsContent>
</Tabs>
    </div>
  );
};

export default ModelTester;