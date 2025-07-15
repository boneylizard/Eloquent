// Enhanced ModelSelector.jsx with OpenAI API integration
import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Globe, Cpu } from 'lucide-react';
import { getContextLength, saveContextLength } from '../utils/apiCall';

const ModelSelector = () => {
  const {
    availableModels,
    loadedModels,
    primaryModel,
    secondaryModel,
    loadModel,
    unloadModel,
    isModelLoading,
    // Add settings to get custom endpoints
    settings,
    // Add new state for API models
    primaryIsAPI,
    secondaryIsAPI,
    setPrimaryIsAPI,
    setSecondaryIsAPI,
    setPrimaryModel,
    setSecondaryModel
  } = useApp();

  const [selectedModel, setSelectedModel] = useState('');
  const [selectedGpu, setSelectedGpu] = useState('0');
  const [contextLength, setContextLength] = useState(getContextLength());

  // Get API model options from settings
  const getAPIModels = () => {
    const customEndpoints = settings.customApiEndpoints || [];
    return customEndpoints
      .filter(endpoint => endpoint.enabled)
      .map(endpoint => ({
        id: endpoint.id,
        name: endpoint.name,
        url: endpoint.url,
        apiKey: endpoint.apiKey,
        icon: Globe
      }));
  };

  const API_MODELS = getAPIModels();

  // Context length presets
  const contextPresets = [
    { label: '4K', value: 4096 },
    { label: '8K', value: 8192 },
    { label: '16K', value: 16384 },
    { label: '32K', value: 32768 }
  ];

  // Check if a model ID is an API model
const isAPIModel = (modelId) => {
  // Check if it's a configured custom API endpoint
  const isCustomEndpoint = API_MODELS.some(api => api.id === modelId);
  
  // Check if it follows the endpoint pattern (starts with "endpoint-")
  const isEndpointPattern = modelId && modelId.startsWith('endpoint-');
  
  return isCustomEndpoint || isEndpointPattern;
};

// Get the API info for a model
const getAPIInfo = (modelId) => {
  // First check if it's a configured custom endpoint
  const customEndpoint = API_MODELS.find(api => api.id === modelId);
  if (customEndpoint) {
    return customEndpoint;
  }
  
  // If it's an endpoint pattern model, create a mock API info
  if (modelId && modelId.startsWith('endpoint-')) {
    return {
      id: modelId,
      name: `API Endpoint (${modelId})`,
      url: 'auto-detected',
      apiKey: '',
      icon: Globe
    };
  }
  
  return null;
};

  // Format model name for display
  const formatModelName = (name) => {
    if (isAPIModel(name)) {
      const apiInfo = getAPIInfo(name);
      return apiInfo ? apiInfo.name : name;
    }
    
    // Remove file extension and path for regular models
    let displayName = name.split('/').pop().split('\\').pop();
    if (displayName.endsWith('.bin') || displayName.endsWith('.gguf')) {
      displayName = displayName.substring(0, displayName.lastIndexOf('.'));
    }
    return displayName;
  };

  // Get model info for a specific GPU
  const getModelForGpu = (gpuId) => {
    if (gpuId === 0) {
      return primaryIsAPI ? primaryModel : loadedModels.find(m => m.gpu_id === 0)?.name || null;
    } else {
      return secondaryIsAPI ? secondaryModel : loadedModels.find(m => m.gpu_id === 1)?.name || null;
    }
  };

  // Check if a GPU has an API model active
  const getIsAPIForGpu = (gpuId) => {
    return gpuId === 0 ? primaryIsAPI : secondaryIsAPI;
  };

  // Format the context length for display
  const formatContextLength = (length) => {
    return `${(length / 1024).toFixed(0)}K`;
  };

  // Handle context length slider change
  const handleContextLengthChange = (value) => {
    const newLength = Array.isArray(value) ? value[0] : value;
    setContextLength(newLength);
    saveContextLength(newLength);
  };

  // Handle context length preset button click
  const handlePresetClick = (value) => {
    setContextLength(value);
    saveContextLength(value);
  };

  // Enhanced model loading/API selection
const handleLoadModel = async () => {
  if (!selectedModel) return;

  // DEBUG: Log the detection results
  console.log("🔍 [ModelSelector] Selected model:", selectedModel);
  console.log("🔍 [ModelSelector] isAPIModel check:", isAPIModel(selectedModel));
  console.log("🔍 [ModelSelector] API_MODELS:", API_MODELS);
  console.log("🔍 [ModelSelector] Starts with endpoint-:", selectedModel.startsWith('endpoint-'));

  const gpuId = parseInt(selectedGpu);
  
  if (isAPIModel(selectedModel)) {
    console.log("🌐 [DEBUG] Setting API model:", selectedModel, "for GPU", gpuId);
    const apiInfo = getAPIInfo(selectedModel);
    console.log(`🌐 Selecting API model: ${apiInfo.name} for GPU ${gpuId}`);
    
    // Update the appropriate API state
    if (gpuId === 0) {
      console.log("🌐 [DEBUG] About to call setPrimaryIsAPI(true)");  
      setPrimaryIsAPI(true);
      console.log("🌐 [DEBUG] About to call setPrimaryModel with:", selectedModel);
      setPrimaryModel(selectedModel);
      console.log("🌐 [DEBUG] Primary API setup complete");
    } else {
      setSecondaryIsAPI(true);
      setSecondaryModel(selectedModel);
    }
  } else {
    // Handle regular model loading
    console.log(`🔧 Loading model: ${selectedModel} on GPU ${gpuId}`);
    
    // Clear API state for this GPU
    if (gpuId === 0) {
      setPrimaryIsAPI(false);
    } else {
      setSecondaryIsAPI(false);
    }
    
    // Load the model normally
    await loadModel(selectedModel, gpuId, contextLength);
  }
};

  // Enhanced model unloading
  const handleUnloadModel = async () => {
    const gpuId = parseInt(selectedGpu);
    
    if (getIsAPIForGpu(gpuId)) {
      // Clear API model
      console.log(`🌐 Clearing API model for GPU ${gpuId}`);
      if (gpuId === 0) {
        setPrimaryIsAPI(false);
        setPrimaryModel(null);
      } else {
        setSecondaryIsAPI(false);
        setSecondaryModel(null);
      }
    } else {
      // Unload regular model
      const modelToUnload = gpuId === 0 ? primaryModel : secondaryModel;
      if (modelToUnload) {
        await unloadModel(modelToUnload);
      }
    }
  };

  // Get all available options (regular models + API models)
  const getAllModelOptions = () => {
    const regularModels = availableModels.map(model => ({
      id: model,
      name: formatModelName(model),
      type: 'local',
      icon: Cpu
    }));
    
    const apiModels = API_MODELS.map(api => ({
      id: api.id,
      name: api.name,
      type: 'api',
      icon: api.icon
    }));
    
    return [...apiModels, ...regularModels];
  };

  const allOptions = getAllModelOptions();

  return (
    <div className="rounded-md border p-4 mb-4">
      <h3 className="text-lg font-medium mb-3">Model Selection</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* GPU 0 (Primary) Status */}
        <div className="rounded-md bg-muted p-3">
          <div className="flex items-center justify-between mb-1">
            <h4 className="font-medium">Primary GPU (0)</h4>
            {primaryIsAPI && <Badge variant="secondary" className="text-xs"><Globe className="w-3 h-3 mr-1" />API</Badge>}
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${getModelForGpu(0) ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm truncate">
              {getModelForGpu(0) ? formatModelName(getModelForGpu(0)) : 'No model loaded'}
            </span>
          </div>
        </div>

        {/* GPU 1 (Secondary) Status */}
        <div className="rounded-md bg-muted p-3">
          <div className="flex items-center justify-between mb-1">
            <h4 className="font-medium">Secondary GPU (1)</h4>
            {secondaryIsAPI && <Badge variant="secondary" className="text-xs"><Globe className="w-3 h-3 mr-1" />API</Badge>}
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${getModelForGpu(1) ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm truncate">
              {getModelForGpu(1) ? formatModelName(getModelForGpu(1)) : 'No model loaded'}
            </span>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {/* Model Selection */}
        <div className="flex flex-col md:flex-row gap-3">
          <div className="flex-1">
            <label className="text-sm font-medium mb-1 block">Select Model or API</label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a model or API endpoint" />
              </SelectTrigger>
              <SelectContent>
                {API_MODELS.length > 0 ? (
                  <>
                    <div className="px-2 py-1 text-xs font-medium text-muted-foreground">API Endpoints</div>
                    {API_MODELS.map((api) => (
                      <SelectItem key={api.id} value={api.id}>
                        <div className="flex items-center gap-2">
                          <Globe className="w-4 h-4 text-blue-500" />
                          {api.name}
                        </div>
                      </SelectItem>
                    ))}
                    <div className="px-2 py-1 text-xs font-medium text-muted-foreground border-t mt-1 pt-2">Local Models</div>
                  </>
                ) : (
                  <div className="px-2 py-2 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2 mb-1">
                      <Globe className="w-4 h-4" />
                      No API endpoints configured
                    </div>
                    <div className="text-[10px]">Add custom endpoints in Settings → LLM Settings</div>
                    <div className="border-t mt-2 pt-2 text-xs font-medium">Local Models</div>
                  </div>
                )}
                {availableModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-green-500" />
                      {formatModelName(model)}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="w-24">
            <label className="text-sm font-medium mb-1 block">Target</label>
            <Select value={selectedGpu} onValueChange={setSelectedGpu}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">GPU 0</SelectItem>
                <SelectItem value="1">GPU 1</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Context Length Adjuster - only show for local models */}
        {selectedModel && !isAPIModel(selectedModel) && (
          <div className="border rounded-md p-3 bg-muted/30">
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">Context Length</label>
              <span className="text-sm font-semibold">{formatContextLength(contextLength)}</span>
            </div>
            
            <Slider 
              value={[contextLength]} 
              min={2048} 
              max={32768} 
              step={1024} 
              onValueChange={handleContextLengthChange}
              className="mb-3"
            />
            
            <div className="flex justify-between gap-2">
              {contextPresets.map(preset => (
                <Button 
                  key={preset.value}
                  variant={contextLength === preset.value ? "default" : "outline"} 
                  size="sm"
                  onClick={() => handlePresetClick(preset.value)}
                  className="flex-1"
                >
                  {preset.label}
                </Button>
              ))}
            </div>
            
            <div className="mt-2 text-xs text-muted-foreground">
              <p>Larger context lengths allow longer outputs but use more GPU memory. Use 32K for coding tasks.</p>
            </div>
          </div>
        )}

        {/* API Info - show when API model is selected */}
        {selectedModel && isAPIModel(selectedModel) && (
          <div className="border rounded-md p-3 bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300 mb-2">
              <Globe className="w-4 h-4" />
              <span className="font-medium text-sm">OpenAI API Mode</span>
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400">
              This will use the OpenAI-compatible endpoint at /v1/chat/completions. 
              Make sure you have models loaded on the backend for the API to work.
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button 
            className="flex-1" 
            onClick={handleLoadModel} 
            disabled={!selectedModel || isModelLoading}
          >
            {isModelLoading ? 'Loading...' : isAPIModel(selectedModel) ? 'Select API' : 'Load Model'}
          </Button>
          <Button 
            className="flex-1" 
            variant="outline" 
            onClick={handleUnloadModel}
            disabled={!getModelForGpu(parseInt(selectedGpu)) || isModelLoading}
          >
            {getIsAPIForGpu(parseInt(selectedGpu)) ? 'Clear API' : 'Unload'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;