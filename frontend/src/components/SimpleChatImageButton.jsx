import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { AlertTriangle, Image, Loader2, X, Sparkles, Info } from 'lucide-react';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './ui/select';
import * as Path from 'path-browserify';

const SimpleChatImageButton = () => {
    const {
        sdStatus,
        checkSdStatus,
        generateImage,
        isImageGenerating,
        apiError,
        clearError,
        setMessages,
        settings,
        MEMORY_API_URL,
        generateUniqueId,
    } = useApp();

    // Existing state
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [prompt, setPrompt] = useState('');
    const [negativePrompt, setNegativePrompt] = useState('');
    const [width, setWidth] = useState(512);
    const [height, setHeight] = useState(512);
    const [steps, setSteps] = useState(20);
    const [guidanceScale, setGuidanceScale] = useState(7.0);
    const [sampler, setSampler] = useState('Euler a');
    const [seed, setSeed] = useState(-1);
    const [selectedSampler, setSelectedSampler] = useState("dpmpp2m");
    
    // AUTOMATIC1111 state
    const [selectedModel, setSelectedModel] = useState('');
    const [availableModels, setAvailableModels] = useState([]);
    
    // Local SD state
    const [localSdStatus, setLocalSdStatus] = useState({ 
        available: false, 
        model_loaded: false, 
        current_model: null 
    });
    const [localModels, setLocalModels] = useState([]);
    const [isLocalModelLoading, setIsLocalModelLoading] = useState(false);

    // NEW: ADetailer state - persistent auto-enhance toggle
    const [autoEnhanceEnabled, setAutoEnhanceEnabled] = useState(() => {
        return localStorage.getItem('adetailer-auto-enhance') === 'true';
    });
    const [adetailerAvailable, setAdetailerAvailable] = useState(false);
    
    // NEW: ADetailer models state
    const [availableAdetailerModels, setAvailableAdetailerModels] = useState([]);
    const [selectedAdetailerModel, setSelectedAdetailerModel] = useState('face_yolov8n.pt');
 const availableSamplers = [
  "dpmpp2m",
  "dpmpp2s_a",
  "euler_a",
  "euler",
  "heun",
  "dpm2",
  "ipndm",
  "ipndm_v",
  "lcm",
  "ddim_trailing",
  "tcd"
];   
const [adetailerSettings, setAdetailerSettings] = useState(() => {
    const saved = localStorage.getItem('adetailer-settings');
    return saved ? JSON.parse(saved) : {
        strength: 0.35,  // FIXED: Reduced from 0.4 (compatible with stable-diffusion.cpp)
        confidence: 0.3, // FIXED: This was already good 
        facePrompt: 'detailed face, high quality, sharp focus', // FIXED: Simplified prompt
        modelName: 'face_yolov8n.pt'
    };
});

    // Determine which engine and availability
    const imageEngine = settings?.imageEngine || 'auto1111';
    const isAvailable = imageEngine === 'EloDiffusion' 
        ? localSdStatus?.available 
        : sdStatus?.automatic1111;

    // Existing functions...
    const checkLocalSdStatus = useCallback(async () => {
        try {
            const res = await fetch(`${MEMORY_API_URL}/sd-local/status`);
            if (res.ok) {
                const data = await res.json();
                setLocalSdStatus(data);
            }
        } catch (err) {
            console.error('Local SD status check failed:', err);
        }
    }, [MEMORY_API_URL]);

    const fetchLocalModels = useCallback(async () => {
        try {
            const res = await fetch(`${MEMORY_API_URL}/sd-local/list-models`);
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'success') {
                    setLocalModels(data.models || []);
                }
            }
        } catch (err) {
            console.error('Failed to fetch local SD models:', err);
        }
    }, [MEMORY_API_URL]);

    // NEW: Fetch ADetailer models
    const fetchAdetailerModels = useCallback(async () => {
        try {
            const res = await fetch(`${MEMORY_API_URL}/sd-local/adetailer-models`);
            if (res.ok) {
                const data = await res.json();
                if (data.available && data.models) {
                    setAvailableAdetailerModels(data.models);
                    // Set default model if none selected
                    if (!selectedAdetailerModel && data.models.length > 0) {
                        setSelectedAdetailerModel(data.models[0]);
                    }
                }
            }
        } catch (err) {
            console.error('Failed to fetch ADetailer models:', err);
        }
    }, [MEMORY_API_URL, selectedAdetailerModel]);

    // NEW: Check ADetailer availability
    const checkAdetailerStatus = useCallback(async () => {
        try {
            const res = await fetch(`${MEMORY_API_URL}/sd-local/adetailer-status`);
            if (res.ok) {
                const data = await res.json();
                setAdetailerAvailable(data.available);
            }
        } catch (err) {
            console.error('ADetailer status check failed:', err);
            setAdetailerAvailable(false);
        }
    }, [MEMORY_API_URL]);

    const handleLoadLocalModel = useCallback(async (modelFilename) => {
        if (!modelFilename || typeof modelFilename !== 'string') {
            console.error("Invalid filename:", modelFilename);
            return;
        }

        setIsLocalModelLoading(true);
        try {
            const response = await fetch(`${MEMORY_API_URL}/sd-local/load-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_filename: modelFilename })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server responded with ${response.status}: ${errorText}`);
            }

            await checkLocalSdStatus();
        } catch (err) {
            alert(`Failed to load model: ${err.message}`);
            console.error('Model loading failed:', err);
        } finally {
            setIsLocalModelLoading(false);
        }
    }, [MEMORY_API_URL, checkLocalSdStatus]);

    // Persist auto-enhance state
    useEffect(() => {
        localStorage.setItem('adetailer-auto-enhance', autoEnhanceEnabled.toString());
    }, [autoEnhanceEnabled]);

    // Persist settings
    useEffect(() => {
        localStorage.setItem('adetailer-settings', JSON.stringify(adetailerSettings));
    }, [adetailerSettings]);

    // FIXED: Auto-enhance function - now updates in place with history tracking
    const autoEnhanceImage = useCallback(async (imageUrl, originalPrompt, messageId) => {
        if (!autoEnhanceEnabled || !adetailerAvailable || !messageId) return;
        
        try {
            const response = await fetch(`${MEMORY_API_URL}/sd-local/enhance-adetailer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: imageUrl,
                    original_prompt: originalPrompt,
                    face_prompt: adetailerSettings.facePrompt,
                    strength: adetailerSettings.strength,
                    confidence: adetailerSettings.confidence,
                    model_name: selectedAdetailerModel,
                    sampler: selectedSampler
                })
            });

            if (response.ok) {
                const result = await response.json();
                if (result.status === 'success' && result.enhanced_image_url) {
                    // FIXED: Update existing message with enhancement history tracking
                    setTimeout(() => {
                        setMessages(prev => prev.map(msg => 
                            msg.id === messageId 
                                ? { 
                                    ...msg, 
                                    imagePath: result.enhanced_image_url,
                                    // Initialize enhancement history with original and enhanced
                                    enhancement_history: [imageUrl, result.enhanced_image_url],
                                    current_enhancement_level: 1,
                                    enhanced: true,
                                    enhancement_settings: { ...adetailerSettings, model_name: selectedAdetailerModel }
                                  }
                                : msg
                        ));
                    }, 1000);
                }
            }
        } catch (error) {
            console.error('Auto-enhancement failed:', error);
        }
    }, [autoEnhanceEnabled, adetailerAvailable, adetailerSettings, selectedAdetailerModel, selectedSampler, MEMORY_API_URL, setMessages]);

    // Effects
    useEffect(() => {
        if (isDialogOpen) {
            checkSdStatus();
            checkLocalSdStatus();
            fetchLocalModels();
            checkAdetailerStatus();
            fetchAdetailerModels(); // NEW: Fetch ADetailer models
        }
    }, [isDialogOpen, checkSdStatus, checkLocalSdStatus, fetchLocalModels, checkAdetailerStatus, fetchAdetailerModels]);

    useEffect(() => {
        if (sdStatus?.models?.length) {
            setAvailableModels(sdStatus.models);
            if (!selectedModel) {
                const first = sdStatus.models[0];
                setSelectedModel(first.model_name || first.title || first.name);
            }
        }
    }, [sdStatus, selectedModel]);

    const handleGenerateImage = async () => {
        if (!prompt.trim() || !isAvailable || isImageGenerating) {
            return;
        }

        clearError();
        try {
            const responseData = await generateImage(prompt, {
                negative_prompt: negativePrompt,
                width,
                height,
                steps,
                guidance_scale: guidanceScale,
                sampler: 'Euler a',
                seed: seed,
                model: selectedModel
            });

            if (responseData && Array.isArray(responseData.image_urls) && responseData.image_urls.length > 0) {
                responseData.image_urls.forEach(imageUrl => {
                    // Generate message ID first
                    const messageId = `${Date.now()}-${Math.random().toString(36).substr(2, 7)}-img`;
                    
                    const imageMessage = {
                        id: messageId, // Use the generated ID
                        role: 'bot',
                        type: 'image',
                        content: prompt,
                        imagePath: imageUrl,
                        prompt: prompt,
                        negative_prompt: negativePrompt,
                        width: responseData.parameters?.width || width,
                        height: responseData.parameters?.height || height,
                        steps: responseData.parameters?.steps || steps,
                        guidance_scale: responseData.parameters?.cfg_scale || guidanceScale,
                        model: responseData.parameters?.sd_model_checkpoint || selectedModel,
                        sampler: responseData.parameters?.sampler_name || 'Euler a',
                        seed: responseData.parameters?.seed !== undefined ? responseData.parameters.seed : -1,
                        timestamp: new Date().toISOString()
                    };
                    
                    setMessages(prev => [...prev, imageMessage]);
                    
                    // FIXED: Auto-enhance if enabled - now passes message ID
                    if (autoEnhanceEnabled && adetailerAvailable) {
                        autoEnhanceImage(imageUrl, prompt, messageId);
                    }
                });
            } else {
                setMessages(prev => [
                    ...prev, {
                        id: `${Date.now()}-error`,
                        role: 'system',
                        content: `Image generation completed, but no images were returned.`,
                        error: true
                    }
                ]);
            }
            setPrompt('');
            setNegativePrompt('');
            setIsDialogOpen(false);
        } catch (err) {
            console.error('Error during image generation process:', err);
            setMessages(prev => [
                ...prev, {
                    id: `${Date.now()}-catch`,
                    role: 'system',
                    content: `Error generating image: ${err.message}.`,
                    error: true
                }
            ]);
        }
    };

    return (
        <>
            <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full p-0"
                title="Generate Image"
                onClick={() => setIsDialogOpen(true)}
            >
                <Image className="h-4 w-4" />
            </Button>

            {isDialogOpen && createPortal(
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div
                        className="relative w-full max-w-lg bg-background rounded-lg p-6 shadow-xl max-h-[90vh] flex flex-col overflow-hidden"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-start justify-between border-b pb-3 mb-4">
                            <div>
                                <h3 className="text-lg font-semibold">EloDiffusion Generator</h3>
                                <p className="text-sm text-muted-foreground">Configure and generate an image.</p>
                            </div>
                            <Button variant="ghost" size="icon" className="-mt-1 -mr-2" onClick={() => setIsDialogOpen(false)}>
                                <X className="h-4 w-4" />
                            </Button>
                        </div>

                        <div className="space-y-4 overflow-y-auto">
                            {apiError && (
                                <div className="text-xs bg-red-100 dark:bg-red-900/30 p-2 rounded flex items-start gap-2">
                                    <AlertTriangle className="h-4 w-4 text-red-700 dark:text-red-500" />
                                    <span>{apiError}</span>
                                </div>
                            )}

                            {/* FIXED AUTO-ENHANCE SECTION WITH MODEL SELECTION */}
                            {adetailerAvailable && (
                                <div className={`p-3 rounded-lg border-2 transition-all ${
                                    autoEnhanceEnabled 
                                        ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30' 
                                        : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50'
                                }`}>
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <Sparkles className={`h-5 w-5 ${autoEnhanceEnabled ? 'text-purple-500' : 'text-gray-400'}`} />
                                            <div>
                                                <Label className="text-sm font-medium">Auto-Enhance</Label>
                                                <p className="text-xs text-muted-foreground">
                                                    {autoEnhanceEnabled ? 'All images will be automatically enhanced' : 'Click to enable automatic face enhancement'}
                                                </p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <input
                                                type="checkbox"
                                                id="auto-enhance-main"
                                                checked={autoEnhanceEnabled}
                                                onChange={(e) => setAutoEnhanceEnabled(e.target.checked)}
                                                className="w-5 h-5 rounded border-2 border-purple-500 text-purple-500 focus:ring-purple-500"
                                            />
                                        </div>
                                    </div>
                                    
{autoEnhanceEnabled && (
    <div className="mt-3 pt-3 border-t border-purple-200 dark:border-purple-700">
        {/* Model Selection Dropdown - unchanged, this works fine */}
        <div className="mb-3">
            <Label className="text-xs">Enhancement Model</Label>
            <Select
                value={selectedAdetailerModel}
                onValueChange={(value) => {
                    setSelectedAdetailerModel(value);
                    setAdetailerSettings(prev => ({...prev, modelName: value}));
                }}
            >
                <SelectTrigger className="w-full mt-1 text-xs">
                    <SelectValue placeholder="Select enhancement model" />
                </SelectTrigger>
                <SelectContent>
                    {availableAdetailerModels.map((model) => (
                        <SelectItem key={model} value={model} className="text-xs">
                            {model}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs">
            {/* FIXED: Better strength range - compatible with stable-diffusion.cpp */}
            <div>
                <Label>Strength: {adetailerSettings.strength.toFixed(2)}</Label>
                <input
                    type="range"
                    min="0.20"       // FIXED: Minimum raised to prevent no effect
                    max="0.50"       // FIXED: Maximum lowered to prevent over-processing  
                    step="0.05"      // FIXED: Finer control
                    value={adetailerSettings.strength}
                    onChange={(e) => setAdetailerSettings(prev => ({...prev, strength: parseFloat(e.target.value)}))}
                    className="w-full mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                    Lower = subtle changes, Higher = dramatic changes
                </p>
            </div>
            
            {/* FIXED: Better confidence range - this is for YOLO detection, fully compatible */}
            <div>
                <Label>Confidence: {adetailerSettings.confidence.toFixed(2)}</Label>
                <input
                    type="range"
                    min="0.20"       // FIXED: Better minimum for face detection
                    max="0.60"       // FIXED: Research shows 0.8+ is too selective
                    step="0.05"      // FIXED: Finer control
                    value={adetailerSettings.confidence}
                    onChange={(e) => setAdetailerSettings(prev => ({...prev, confidence: parseFloat(e.target.value)}))}
                    className="w-full mt-1"
                />
                <p className="text-xs text-muted-foreground mt-1">
                    Lower = detect more faces, Higher = only clear faces
                </p>
            </div>
        </div>
        
        {/* FIXED: Better face prompt guidance - fully compatible */}
        <div className="mt-2">
            <Label className="text-xs">Face Enhancement Prompt</Label>
            <input
                type="text"
                value={adetailerSettings.facePrompt}
                onChange={(e) => setAdetailerSettings(prev => ({...prev, facePrompt: e.target.value}))}
                placeholder="detailed face, high quality, sharp focus"
                className="w-full mt-1 text-xs p-2 border rounded bg-background text-foreground"
            />
            <p className="text-xs text-muted-foreground mt-1">
                Keep simple. Avoid duplicating your main prompt terms.
            </p>
        </div>
        
        {/* FIXED: Add helpful tips specific to stable-diffusion.cpp limitations */}
        <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs">
            <strong>💡 Tips for stable-diffusion.cpp:</strong><br/>
            • If faces look over-processed, lower Strength to 0.25-0.30<br/>
            • If enhancement is too subtle, try face_yolov8s.pt model first<br/>
            • Keep face prompts simple - complex prompts can cause conflicts
        </div>
        
        {/* FIXED: Add warning about current issues */}
        <div className="mt-2 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs">
            <strong>⚠️ Note:</strong> If auto-enhance is making faces worse, this is a known issue with mask processing. Update your backend settings to fix it.
        </div>
    </div>
)}
                                </div>
                            )}

                            {!adetailerAvailable && (
                                <div className="p-3 border rounded-md bg-yellow-50 dark:bg-yellow-900/20">
                                    <div className="flex items-start gap-2 text-xs text-yellow-800 dark:text-yellow-200">
                                        <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                                        <span>
                                            ADetailer not available. Install with: <code className="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">pip install ultralytics</code>
                                        </span>
                                    </div>
                                </div>
                            )}
                            
                            {/* Existing model selection */}
                            {sdStatus?.automatic1111 && availableModels.length > 0 && (
                                <div className="space-y-2">
                                    <Label htmlFor="sd-model" className="text-xs">Model</Label>
                                    <select 
                                        id="sd-model" 
                                        value={selectedModel} 
                                        onChange={e => setSelectedModel(e.target.value)} 
                                        disabled={isImageGenerating} 
                                        className="w-full rounded border border-input bg-background text-foreground px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                                    >
                                        {availableModels.map((m, i) => {
                                            const name = m.model_name || m.title || m.name;
                                            return <option key={i} value={name} className="bg-background text-foreground">{name}</option>;
                                        })}
                                    </select>
                                </div>
                            )}
                            
                            {/* LOCAL SD MODEL SELECTION - RESTORED */}
                            {imageEngine === 'EloDiffusion' && localModels.length > 0 && (
                                <div className="space-y-2">
                                    <Label htmlFor="local-model" className="text-xs">Local SD Model</Label>
                                    <div className="flex gap-2">
                                        <select 
                                            id="local-model" 
                                            value={localSdStatus.current_model || ''} 
                                            onChange={e => {
                                                if (e.target.value) {
                                                    handleLoadLocalModel(e.target.value);
                                                }
                                            }}
                                            disabled={isImageGenerating || isLocalModelLoading} 
                                            className="flex-1 rounded border border-input bg-background text-foreground px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                                        >
                                            <option value="">Select a model...</option>
                                            {localModels.map((model, i) => (
                                                <option key={i} value={model} className="bg-background text-foreground">
                                                    {model}
                                                </option>
                                            ))}
                                        </select>
                                        {isLocalModelLoading && (
                                            <div className="flex items-center">
                                                <Loader2 className="h-4 w-4 animate-spin" />
                                            </div>
                                        )}
                                    </div>
                                    {localSdStatus.current_model && (
                                        <p className="text-xs text-green-600 dark:text-green-400">
                                            ✓ Loaded: {localSdStatus.current_model}
                                        </p>
                                    )}
                                    {!localSdStatus.model_loaded && (
                                        <p className="text-xs text-yellow-600 dark:text-yellow-400">
                                            ⚠ No model loaded. Select a model above to generate images.
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* Prompt inputs */}
                            <div className="space-y-2">
                                <Label htmlFor="prompt" className="text-xs">Prompt</Label>
                                <Textarea 
                                    id="prompt" 
                                    placeholder="Describe the image..." 
                                    value={prompt} 
                                    onChange={e => setPrompt(e.target.value)} 
                                    rows={2} 
                                    disabled={isImageGenerating || !isAvailable} 
                                />
                            </div>
                            
                            <div className="space-y-2">
                                <Label htmlFor="negativePrompt" className="text-xs">Negative Prompt</Label>
                                <Textarea 
                                    id="negativePrompt" 
                                    placeholder="Elements to avoid..." 
                                    value={negativePrompt} 
                                    onChange={e => setNegativePrompt(e.target.value)} 
                                    rows={1} 
                                    disabled={isImageGenerating || !isAvailable} 
                                />
                            </div>
<div className="mb-2">
  <label className="block text-sm font-medium text-foreground">
    Sampler
  </label>
  <select
    className="mt-1 block w-full rounded-md border border-border bg-background text-foreground shadow-sm focus:border-ring focus:ring-ring sm:text-sm"
    value={selectedSampler}
    onChange={(e) => setSelectedSampler(e.target.value)}
  >
    {availableSamplers.map((sampler) => (
      <option key={sampler} value={sampler}>
        {sampler}
      </option>
    ))}
  </select>
</div>
                            {/* Size Controls - RESTORED FULL VERSION */}
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs items-center">
                                    <Label>Size</Label>
                                    <span>{width} × {height}</span>
                                </div>
                                <div className="flex gap-2">
                                    <Button size="sm" variant={width === 512 && height === 512 ? "secondary" : "outline"} onClick={() => { setWidth(512); setHeight(512); }} disabled={isImageGenerating}>1:1</Button>
                                    <Button size="sm" variant={width === 768 && height === 512 ? "secondary" : "outline"} onClick={() => { setWidth(768); setHeight(512); }} disabled={isImageGenerating}>3:2</Button>
                                    <Button size="sm" variant={width === 512 && height === 768 ? "secondary" : "outline"} onClick={() => { setWidth(512); setHeight(768); }} disabled={isImageGenerating}>2:3</Button>
                                </div>
                                
                                {/* Width Slider */}
                                <div className="space-y-1">
                                    <Label className="text-xs">Width: {width}px</Label>
                                    <Slider
                                        min={256}
                                        max={1024}
                                        step={64}
                                        value={[width]}
                                        onValueChange={([v]) => setWidth(v)}
                                        disabled={isImageGenerating}
                                    />
                                </div>
                                
                                {/* Height Slider */}
                                <div className="space-y-1">
                                    <Label className="text-xs">Height: {height}px</Label>
                                    <Slider
                                        min={256}
                                        max={1024}
                                        step={64}
                                        value={[height]}
                                        onValueChange={([v]) => setHeight(v)}
                                        disabled={isImageGenerating}
                                    />
                                </div>
                            </div>
                            
                            <div className="space-y-2">
                                <Label className="text-xs">Steps: {steps}</Label>
                                <Slider min={4} max={50} step={1} value={[steps]} onValueChange={val => setSteps(val[0])} />
                            </div>
                            
                            <div className="space-y-2">
                                <Label className="text-xs">Guidance Scale: {guidanceScale.toFixed(1)}</Label>
                                <Slider min={1} max={15} step={0.1} value={[guidanceScale]} onValueChange={val => setGuidanceScale(val[0])} disabled={isImageGenerating} />
                            </div>

                            <Button 
                                className={`w-full ${autoEnhanceEnabled && adetailerAvailable ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700' : ''}`}
                                onClick={handleGenerateImage} 
                                disabled={isImageGenerating || !prompt.trim()}
                            >
                                {isImageGenerating ? 
                                    <>
                                        <Loader2 className="animate-spin mr-2 h-4 w-4" />
                                        Generating...
                                    </> : 
                                    <>
                                        <Image className="mr-2 h-4 w-4" />
                                        Generate Image
                                        {autoEnhanceEnabled && adetailerAvailable && (
                                            <>
                                                <Sparkles className="ml-2 h-4 w-4" />
                                                <span className="ml-1 text-xs">+ Auto-Enhance</span>
                                            </>
                                        )}
                                    </>
                                }
                            </Button>
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </>
    );
};

export default SimpleChatImageButton;