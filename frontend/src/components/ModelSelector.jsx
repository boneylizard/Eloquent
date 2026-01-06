// Enhanced ModelSelector.jsx with OpenAI API integration
import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
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
    setSecondaryModel,
    PRIMARY_API_URL
  } = useApp();

  const [selectedApiModel, setSelectedApiModel] = useState('');
  const [selectedLocalModel, setSelectedLocalModel] = useState('');
  const [selectedGpu, setSelectedGpu] = useState('0');
  const [contextLength, setContextLength] = useState(getContextLength());
  const [gpuCount, setGpuCount] = useState(2); // Default to 2, will be updated from backend

  // Fetch GPU count from backend
  useEffect(() => {
    const fetchGpuCount = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/system/gpu_info`);
        if (response.ok) {
          const data = await response.json();
          if (data.gpu_count) {
            setGpuCount(data.gpu_count);
            // If selected GPU is beyond available GPUs, reset to 0
            if (parseInt(selectedGpu) >= data.gpu_count) {
              setSelectedGpu('0');
            }
          }
        }
      } catch (error) {
        console.error("Error fetching GPU count:", error);
      }
    };
    fetchGpuCount();
  }, [PRIMARY_API_URL, selectedGpu]);

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

  // Get model info for a specific GPU (supports any number of GPUs)
  const getModelForGpu = (gpuId) => {
    // Check loadedModels first (works for any GPU)
    const loadedModel = loadedModels.find(m => m.gpu_id === gpuId);
    if (loadedModel) {
      return loadedModel.name;
    }
    
    // Fallback to primary/secondary for backward compatibility (GPU 0/1 only)
    if (gpuId === 0 && primaryIsAPI) {
      return primaryModel;
    } else if (gpuId === 1 && secondaryIsAPI) {
      return secondaryModel;
    }
    
    return null;
  };

  // Check if a GPU has an API model active
  const getIsAPIForGpu = (gpuId) => {
    // For now, only GPU 0/1 can have API models (backward compatibility)
    // This could be expanded later to support API models on any GPU
    return gpuId === 0 ? primaryIsAPI : (gpuId === 1 ? secondaryIsAPI : false);
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

  // Handle API model selection (independent of GPUs)
  const handleSelectApi = async () => {
    if (!selectedApiModel) return;
    
    console.log(`🌐 Selecting API model: ${selectedApiModel}`);
    const apiInfo = getAPIInfo(selectedApiModel);
    
    // Set as primary API (for chat) - doesn't use a GPU slot
    setPrimaryIsAPI(true);
    setPrimaryModel(selectedApiModel);
    setSelectedApiModel(''); // Clear selection
  };

  // Handle local model loading on GPU
  const handleLoadLocalModel = async () => {
    if (!selectedLocalModel) return;
    
    const gpuId = parseInt(selectedGpu);
    console.log(`🔧 Loading model: ${selectedLocalModel} on GPU ${gpuId}`);
    
    // Clear API state for this GPU (if any) - but only if it was set for this specific GPU
    // Note: API is now independent, so we don't need to clear it here
    
    // Load the model normally
    await loadModel(selectedLocalModel, gpuId, contextLength);
    setSelectedLocalModel(''); // Clear selection
  };

  // Handle clearing API (separate from GPU unloading)
  const handleClearApi = () => {
    setPrimaryIsAPI(false);
    setPrimaryModel(null);
  };

  // Enhanced model unloading (for local GPU models only)
  const handleUnloadModel = async () => {
    const gpuId = parseInt(selectedGpu);
    const model = getModelForGpu(gpuId);
    
    if (model && !getIsAPIForGpu(gpuId)) {
      // Only unload if it's a local model (not API)
      await unloadModel(model, gpuId);
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
      
      {/* API Status (separate from GPUs) */}
      {primaryIsAPI && (
        <div className="rounded-md bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 p-3 mb-4">
          <div className="flex items-center justify-between mb-1">
            <h4 className="font-medium flex items-center gap-2">
              <Globe className="w-4 h-4 text-blue-500" />
              API Model (Chat)
            </h4>
            <Badge variant="secondary" className="text-xs"><Globe className="w-3 h-3 mr-1" />Active</Badge>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-sm truncate">
              {formatModelName(primaryModel)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">Using external API - GPUs are free for local models</p>
        </div>
      )}

      <div className={`grid grid-cols-1 ${gpuCount <= 2 ? 'md:grid-cols-2' : gpuCount <= 4 ? 'md:grid-cols-2 lg:grid-cols-4' : 'md:grid-cols-3 lg:grid-cols-4'} gap-4 mb-4`}>
        {/* Dynamically render GPU status for all available GPUs */}
        {Array.from({ length: gpuCount }, (_, i) => {
          const model = getModelForGpu(i);
          const isApi = getIsAPIForGpu(i);
          // Only show local models on GPUs (API doesn't use GPU slots anymore)
          const displayModel = isApi ? null : model;
          
          return (
            <div key={i} className="rounded-md bg-muted p-3">
              <div className="flex items-center justify-between mb-1">
                <h4 className="font-medium">GPU {i}</h4>
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${displayModel ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm truncate">
                  {displayModel ? formatModelName(displayModel) : 'No model loaded'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="space-y-4">
        {/* API Model Selection (Independent of GPUs) */}
        <div className="border rounded-md p-3 bg-blue-50/50 dark:bg-blue-950/10">
          <label className="text-sm font-medium mb-2 block flex items-center gap-2">
            <Globe className="w-4 h-4 text-blue-500" />
            API Model (for Chat) - Doesn't use GPU
          </label>
          <div className="flex gap-2">
            <Select value={selectedApiModel} onValueChange={setSelectedApiModel}>
              <SelectTrigger className="flex-1">
                <SelectValue placeholder={API_MODELS.length > 0 ? "Select API endpoint" : "No API endpoints configured"} />
              </SelectTrigger>
              <SelectContent>
                {API_MODELS.length > 0 ? (
                  API_MODELS.map((api) => (
                    <SelectItem key={api.id} value={api.id}>
                      <div className="flex items-center gap-2">
                        <Globe className="w-4 h-4 text-blue-500" />
                        {api.name}
                      </div>
                    </SelectItem>
                  ))
                ) : (
                  <div className="px-2 py-2 text-xs text-muted-foreground">
                    <div className="text-[10px]">Add custom endpoints in Settings → LLM Settings</div>
                  </div>
                )}
              </SelectContent>
            </Select>
            <Button 
              onClick={handleSelectApi}
              disabled={!selectedApiModel || isModelLoading}
              variant="outline"
            >
              {primaryIsAPI ? 'Update API' : 'Select API'}
            </Button>
            {primaryIsAPI && (
              <Button 
                onClick={handleClearApi}
                variant="outline"
                className="text-red-600 hover:text-red-700"
              >
                Clear API
              </Button>
            )}
          </div>
          {primaryIsAPI && (
            <p className="text-xs text-muted-foreground mt-2">
              ✓ Using API: {formatModelName(primaryModel)} - Your GPUs are free for other models
            </p>
          )}
        </div>

        <Separator />

        {/* Local GPU Model Selection */}
        <div>
          <label className="text-sm font-medium mb-2 block">Local GPU Models</label>
          <div className="flex flex-col md:flex-row gap-3">
            <div className="flex-1">
              <Select value={selectedLocalModel} onValueChange={setSelectedLocalModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a local model" />
                </SelectTrigger>
                <SelectContent>
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
              <label className="text-sm font-medium mb-1 block">GPU</label>
              <Select value={selectedGpu} onValueChange={setSelectedGpu}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: gpuCount }, (_, i) => (
                    <SelectItem key={i} value={i.toString()}>GPU {i}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Context Length Adjuster - only show for local models */}
        {selectedLocalModel && (
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

        {/* Action Buttons for Local Models */}
        <div className="flex gap-2">
          <Button 
            className="flex-1" 
            onClick={handleLoadLocalModel} 
            disabled={!selectedLocalModel || isModelLoading}
          >
            {isModelLoading ? 'Loading...' : 'Load Model'}
          </Button>
          <Button 
            className="flex-1" 
            variant="outline" 
            onClick={handleUnloadModel}
            disabled={!getModelForGpu(parseInt(selectedGpu)) || getIsAPIForGpu(parseInt(selectedGpu)) || isModelLoading}
          >
            Unload
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;