// SimpleChatImageMessage.jsx - Enhanced with in-place enhancement replacement

import React, { useState, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Loader2, Download, Copy, ZoomIn, Check, X, RotateCcw, Sparkles, Undo, ArrowUpCircle, Image as ImageIcon } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { useApp } from '../contexts/AppContext';

const SimpleChatImageMessage = ({ message, onRegenerate, regenerationQueue }) => {
  const { primaryModel, MEMORY_API_URL, PRIMARY_API_URL, setMessages, generateUniqueId, userProfile, setBackgroundImage } = useApp();

  // Existing state
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [isViewerOpen, setIsViewerOpen] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showImageQuery, setShowImageQuery] = useState(false);
  const [imageQuery, setImageQuery] = useState('');
  // Add this state near your other useState declarations
  const [selectedGpuId, setSelectedGpuId] = useState(0);
  // Manual enhancement state
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [scaleFactor, setScaleFactor] = useState("2");
  const [showEnhanceSettings, setShowEnhanceSettings] = useState(false);

  const [adetailerSettings, setAdetailerSettings] = useState(() => {
    const defaultModel = (() => {
      const savedModel = localStorage.getItem('adetailer-selected-model');
      if (savedModel) return savedModel;
      const savedSettings = localStorage.getItem('adetailer-settings');
      if (savedSettings) {
        try {
          const parsed = JSON.parse(savedSettings);
          if (parsed?.modelName) return parsed.modelName;
        } catch (error) {
          // Ignore malformed settings
        }
      }
      return 'face_yolov8n.pt';
    })();

    const defaults = {
      strength: 0.35,
      steps: 45,
      confidence: 0.3,
      facePrompt: 'detailed face, high quality, sharp focus',
      modelName: defaultModel,
      sampler: 'euler_a',
      useOriginalPrompt: true,
      useOriginalNegativePrompt: false
    };

    const saved = localStorage.getItem('adetailer-settings');
    if (!saved) return defaults;

    try {
      const parsed = JSON.parse(saved);
      return {
        ...defaults,
        ...parsed,
        strength: Number.isFinite(Number(parsed?.strength)) ? Number(parsed.strength) : defaults.strength,
        steps: Number.isFinite(Number(parsed?.steps)) ? Number(parsed.steps) : defaults.steps,
        confidence: Number.isFinite(Number(parsed?.confidence)) ? Number(parsed.confidence) : defaults.confidence,
        modelName: parsed?.modelName || defaultModel,
        sampler: parsed?.sampler || defaults.sampler,
        useOriginalPrompt: typeof parsed?.useOriginalPrompt === 'boolean' ? parsed.useOriginalPrompt : defaults.useOriginalPrompt,
        useOriginalNegativePrompt: typeof parsed?.useOriginalNegativePrompt === 'boolean' ? parsed.useOriginalNegativePrompt : defaults.useOriginalNegativePrompt
      };
    } catch (error) {
      return defaults;
    }
  });

  const [availableAdetailerModels, setAvailableAdetailerModels] = useState([]);
  const [adetailerAvailable, setAdetailerAvailable] = useState(false);

  // Existing functions
  const getImageUrl = () => {
    if (message.imagePath && typeof message.imagePath === 'string') {
      return message.imagePath;
    }
    console.warn('SimpleChatImageMessage: message.imagePath is missing or invalid!', message);
    return '';
  };

  const imageUrl = getImageUrl();
  const adetailerModelOptions = (() => {
    const models = Array.isArray(availableAdetailerModels) ? [...availableAdetailerModels] : [];
    const current = adetailerSettings.modelName;
    if (current && !models.includes(current)) {
      models.unshift(current);
    }
    return models;
  })();
  const adetailerSamplerOptions = [
    'default',
    'euler',
    'euler_a',
    'heun',
    'dpm2',
    'dpm++2s_a',
    'dpm++2m',
    'dpm++2mv2',
    'ipndm',
    'ipndm_v',
    'lcm',
    'ddim_trailing',
    'tcd'
  ];

  useEffect(() => {
    localStorage.setItem('adetailer-settings', JSON.stringify(adetailerSettings));
    if (adetailerSettings.modelName) {
      localStorage.setItem('adetailer-selected-model', adetailerSettings.modelName);
    }
  }, [adetailerSettings]);

  useEffect(() => {
    let isMounted = true;

    const safeFetchJson = async (url) => {
      if (!url) return null;
      try {
        const response = await fetch(url);
        if (!response.ok) return null;
        return await response.json();
      } catch (error) {
        return null;
      }
    };

    const fetchAdetailerInfo = async () => {
      const [primaryStatus, primaryModels, memoryStatus, memoryModels] = await Promise.all([
        safeFetchJson(PRIMARY_API_URL ? `${PRIMARY_API_URL}/sd-local/adetailer-status` : null),
        safeFetchJson(PRIMARY_API_URL ? `${PRIMARY_API_URL}/sd-local/adetailer-models` : null),
        safeFetchJson(MEMORY_API_URL ? `${MEMORY_API_URL}/sd-local/adetailer-status` : null),
        safeFetchJson(MEMORY_API_URL ? `${MEMORY_API_URL}/sd-local/adetailer-models` : null)
      ]);

      if (!isMounted) return;

      const status = primaryStatus || memoryStatus;
      if (status && typeof status.available === 'boolean') {
        setAdetailerAvailable(status.available);
      }

      const modelsData = primaryModels || memoryModels;
      if (modelsData && Array.isArray(modelsData.models)) {
        setAvailableAdetailerModels(modelsData.models);
      }
    };

    fetchAdetailerInfo();
    return () => {
      isMounted = false;
    };
  }, [PRIMARY_API_URL, MEMORY_API_URL]);

  // Enhanced enhancement function with history tracking
  const handleManualEnhance = useCallback(async () => {
    if (!imageUrl || isEnhancing) return;

    const settings = {
      strength: Number.isFinite(adetailerSettings.strength) ? adetailerSettings.strength : 0.35,
      steps: Number.isFinite(adetailerSettings.steps) ? Math.max(1, Math.round(adetailerSettings.steps)) : 45,
      confidence: Number.isFinite(adetailerSettings.confidence) ? adetailerSettings.confidence : 0.3,
      facePrompt: adetailerSettings.facePrompt || 'detailed face, high quality, sharp focus',
      modelName: adetailerSettings.modelName || 'face_yolov8n.pt',
      sampler: adetailerSettings.sampler || 'euler_a',
      useOriginalPrompt: adetailerSettings.useOriginalPrompt !== false,
      useOriginalNegativePrompt: adetailerSettings.useOriginalNegativePrompt === true
    };
    const originalPrompt = settings.useOriginalPrompt ? (message.prompt || '') : '';
    const negativePrompt = settings.useOriginalNegativePrompt ? (message.negative_prompt || '') : '';

    setIsEnhancing(true);

    try {
      const response = await fetch(`${PRIMARY_API_URL}/sd-local/enhance-adetailer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          original_prompt: originalPrompt,
          face_prompt: settings.facePrompt,
          negative_prompt: negativePrompt,
          strength: settings.strength,
          steps: settings.steps,
          confidence: settings.confidence,
          sampler: settings.sampler,
          model_name: settings.modelName,
          gpu_id: message.gpuId || 0 // Use the GPU ID from the message or default to 0
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Enhancement failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success' && result.enhanced_image_url) {
        const appliedSettings = result.parameters
          ? {
            strength: result.parameters.strength ?? settings.strength,
            steps: result.parameters.steps ?? settings.steps,
            effective_steps: result.parameters.effective_steps,
            confidence: result.parameters.confidence ?? settings.confidence,
            facePrompt: result.parameters.face_prompt ?? settings.facePrompt,
            model_name: result.parameters.model_name ?? settings.modelName,
            sampler: result.parameters.sampler ?? settings.sampler,
            negative_prompt: result.parameters.negative_prompt ?? negativePrompt
          }
          : { ...settings, model_name: settings.modelName };

        // Update message with enhancement history tracking
        setMessages(prev => prev.map(msg =>
          msg.id === message.id
            ? {
              ...msg,
              imagePath: result.enhanced_image_url,
              // Initialize or update enhancement history
              enhancement_history: msg.enhancement_history
                ? [...msg.enhancement_history, result.enhanced_image_url]
                : [msg.imagePath, result.enhanced_image_url],
              current_enhancement_level: (msg.current_enhancement_level || 0) + 1,
              enhanced: true,
              enhancement_settings: appliedSettings
            }
            : msg
        ));

      } else {
        throw new Error('No enhanced image returned');
      }

    } catch (error) {
      console.error('Manual enhancement error:', error);
      const errorMsg = {
        id: generateUniqueId(),
        role: 'system',
        content: `❌ Enhancement failed: ${error.message}`,
        error: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsEnhancing(false);
    }
  }, [imageUrl, isEnhancing, PRIMARY_API_URL, setMessages, message, generateUniqueId, adetailerSettings]);

  // Add this state near your other useState declarations
  const [upscalerModels, setUpscalerModels] = useState([]);
  const [selectedUpscaler, setSelectedUpscaler] = useState('');

  // Fetch available upscalers on mount
  useEffect(() => {
    const fetchUpscalers = async () => {
      try {
        const response = await fetch(`${PRIMARY_API_URL}/sd-local/upscalers`);
        if (response.ok) {
          const data = await response.json();
          if (data.models && data.models.length > 0) {
            setUpscalerModels(data.models);
            // Default to localStorage preference or first available
            const savedUpscaler = localStorage.getItem('local-upscaler-model');
            if (savedUpscaler && data.models.includes(savedUpscaler)) {
              setSelectedUpscaler(savedUpscaler);
            } else {
              setSelectedUpscaler(data.models[0]);
            }
          }
        }
      } catch (error) {
        console.error("Error fetching upscalers:", error);
      }
    };
    fetchUpscalers();
  }, [PRIMARY_API_URL]);

  const handleUpscale = useCallback(async () => {
    if (!imageUrl || isUpscaling) return;
    setIsUpscaling(true);

    try {
      const response = await fetch(`${PRIMARY_API_URL}/sd-local/upscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_url: imageUrl,
          scale_factor: parseFloat(scaleFactor),
          strength: 0.2,
          prompt: message.prompt || "",
          gpu_id: message.gpuId || 0,
          model_name: selectedUpscaler // Pass selected model
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Upscale failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success' && result.image_url) {
        // Update message
        setMessages(prev => prev.map(msg =>
          msg.id === message.id
            ? {
              ...msg,
              imagePath: result.image_url,
              width: (msg.width || 512) * parseFloat(scaleFactor),
              height: (msg.height || 512) * parseFloat(scaleFactor),
              // Add to history
              enhancement_history: msg.enhancement_history
                ? [...msg.enhancement_history, result.image_url]
                : [msg.imagePath, result.image_url],
              current_enhancement_level: (msg.current_enhancement_level || 0) + 1,
              enhanced: true,
              upscaled: true
            }
            : msg
        ));
      }
    } catch (error) {
      console.error('Upscale error:', error);
      // Optional: show toast or error message
    } finally {
      setIsUpscaling(false);
    }
  }, [imageUrl, isUpscaling, PRIMARY_API_URL, setMessages, message, scaleFactor, selectedUpscaler]);

  const handleSetBackground = () => {
    if (imageUrl) {
      setBackgroundImage(imageUrl);
    }
  };

  // Reset to last enhancement level
  const handleResetToLast = useCallback(() => {
    const currentLevel = message.current_enhancement_level || 0;
    if (currentLevel <= 0 || !message.enhancement_history) return;

    const newLevel = currentLevel - 1;
    const previousImageUrl = message.enhancement_history[newLevel];

    setMessages(prev => prev.map(msg =>
      msg.id === message.id
        ? {
          ...msg,
          imagePath: previousImageUrl,
          current_enhancement_level: newLevel,
          enhanced: newLevel > 0,
          enhancement_settings: newLevel > 0 ? msg.enhancement_settings : undefined
        }
        : msg
    ));
  }, [message.id, message.current_enhancement_level, message.enhancement_history, setMessages]);

  // Existing vision analysis function
  const handleAnalyzeImage = useCallback(async (customQuery = null) => {
    const question = customQuery || "Analyze this image in detail. Describe what you see, including objects, people, settings, colors, mood, and any text visible in the image.";
    const systemPrompt = 'System: You are a helpful AI assistant.';
    const fullPrompt = `${systemPrompt}\n\nHuman: ${question}`;

    if (!imageUrl || isAnalyzing) return;

    setIsAnalyzing(true);
    setShowImageQuery(false);
    setImageQuery('');

    try {
      // Convert image URL to base64
      const response = await fetch(imageUrl);
      const blob = await response.blob();

      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
      });

      // Call vision API
      const analysisResponse = await fetch(`${MEMORY_API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: fullPrompt,
          model_name: primaryModel,
          image_base64: base64,
          image_type: 'image/png',
          temperature: 0.7,
          max_tokens: 1024,
          userProfile: { id: userProfile?.id ?? 'anonymous' }
        })
      });

      if (analysisResponse.ok) {
        const result = await analysisResponse.json();
        const analysisMsg = {
          id: generateUniqueId(),
          role: 'bot',
          content: `**Question:** ${question}\n\n**Answer:** ${result.text || 'No analysis available'}`,
          modelId: 'primary'
        };
        setMessages(prev => [...prev, analysisMsg]);
      } else {
        throw new Error(`Analysis failed: ${analysisResponse.status}`);
      }

    } catch (error) {
      console.error('Vision analysis error:', error);
      const errorMsg = {
        id: generateUniqueId(),
        role: 'bot',
        content: `Image analysis failed: ${error.message}`,
        modelId: 'primary'
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsAnalyzing(false);
    }
  }, [imageUrl, isAnalyzing, primaryModel, MEMORY_API_URL, setMessages, generateUniqueId, userProfile]);

  // Existing functions
  const handleCopyPrompt = () => {
    navigator.clipboard.writeText(message.prompt || '');
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!imageUrl) {
      console.warn('Cannot download: No valid image URL available.');
      return;
    }

    const link = document.createElement('a');
    link.href = imageUrl;
    const filename = message.prompt ?
      `sd-image-${message.prompt.substring(0, 50).replace(/[^a-z0-9]/gi, '_')}.png` :
      `sd-image-${Date.now()}.png`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!imageUrl) {
    return (
      <div className='bg-red-100 dark:bg-red-900/30 p-3 rounded-lg text-sm text-red-800 dark:text-red-200'>
        <p>Unable to load image. Image URL is missing or invalid.</p>
        {message.prompt && <p className='mt-1 text-xs italic'>Prompt: {message.prompt.substring(0, 100)}...</p>}
      </div>
    );
  }

  return (
    <div className='w-full'>
      <div className='bg-gray-50 dark:bg-gray-800 rounded-lg p-2 my-2 relative'>
        {/* Show enhanced badge if this is an enhanced image */}
        {message.enhanced && (
          <div className="absolute top-2 left-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 z-10">
            <Sparkles className="h-3 w-3" />
            Enhanced {message.current_enhancement_level > 1 ? `${message.current_enhancement_level}x` : ''}
          </div>
        )}

        {!isImageLoaded && (
          <div className='w-full h-64 flex items-center justify-center bg-gray-100 dark:bg-gray-700 rounded-md'>
            <Loader2 className='h-8 w-8 animate-spin text-gray-400' />
          </div>
        )}

        <img
          src={imageUrl}
          alt={message.prompt || 'Generated image'}
          className='w-full max-w-2xl mx-auto rounded-md object-contain cursor-pointer'
          style={{
            display: isImageLoaded ? 'block' : 'none',
            maxHeight: '400px',
            minHeight: '200px'
          }}
          onLoad={() => setIsImageLoaded(true)}
          onClick={() => setIsViewerOpen(true)}
          onError={(e) => {
            console.error('Failed to load image:', imageUrl, e);
            e.target.style.display = 'none';
          }}
        />

        <div className='flex flex-col gap-1 mt-2 text-sm'>
          {message.prompt && (
            <p className='font-medium' title={message.prompt}>
              {message.prompt.length > 100
                ? `${message.prompt.substring(0, 100)}...`
                : message.prompt}
            </p>
          )}

          {/* Enhanced image info */}
          {message.enhanced && message.enhancement_settings && (
            <div className='text-xs text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/20 p-2 rounded'>
              Enhanced {message.current_enhancement_level}x with ADetailer | Model: {message.enhancement_settings.model_name || message.enhancement_settings.modelName} | Sampler: {message.enhancement_settings.sampler || 'n/a'} | Strength: {message.enhancement_settings.strength} | Steps: {message.enhancement_settings.steps ?? 'n/a'}{message.enhancement_settings.effective_steps ? ` (${message.enhancement_settings.effective_steps} effective)` : ''} | Confidence: {message.enhancement_settings.confidence}
            </div>
          )}

          <div className='flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400'>
            {message.model && (
              <div><span className='font-medium'>Model:</span> {message.model}</div>
            )}
            {message.width && message.height && (
              <div><span className='font-medium'>Size:</span> {message.width}×{message.height}</div>
            )}
            {message.steps && (
              <div><span className='font-medium'>Steps:</span> {message.steps}</div>
            )}
            {message.guidance_scale && (
              <div><span className='font-medium'>CFG:</span> {message.guidance_scale}</div>
            )}
            {message.sampler && (
              <div><span className='font-medium'>Sampler:</span> {message.sampler}</div>
            )}
            {message.seed !== undefined && message.seed !== -1 && (
              <div><span className='font-medium'>Seed:</span> {message.seed}</div>
            )}
          </div>

          <div className='flex justify-end gap-2 mt-1 flex-wrap'>
            {/* Regenerate button */}
            {onRegenerate && (
              <Button
                size='sm'
                variant='ghost'
                onClick={() => onRegenerate({
                  prompt: message.original_prompt || message.prompt || '',
                  negative_prompt: message.original_negative_prompt || message.negative_prompt || '',
                  width: message.original_width || message.width || 512,
                  height: message.original_height || message.height || 512,
                  steps: message.original_steps || message.steps || 20,
                  guidance_scale: message.original_guidance_scale || message.guidance_scale || 7.0,
                  sampler: message.original_sampler || message.sampler || 'Euler a',
                  seed: -1,
                  model: message.original_model || message.model || '',
                  gpu_id: message.gpuId ?? 0
                })}
                className='h-8 px-2 text-xs'
                title='Regenerate with same parameters'
              >
                <RotateCcw className='h-3 w-3 mr-1' />
                Regenerate {regenerationQueue > 0 && `(${regenerationQueue})`}
              </Button>
            )}

            {/* Enhancement controls */}
            <div className="flex items-center gap-1">
              <Button
                size='sm'
                variant='secondary'
                onClick={handleManualEnhance}
                disabled={isEnhancing}
                className='h-8 px-2 text-xs bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white border-0'
                title={`${message.enhanced ? 'Enhance further' : 'Enhance faces and details'} with ADetailer`}
              >
                {isEnhancing ? (
                  <Loader2 className='h-3 w-3 mr-1 animate-spin' />
                ) : (
                  <Sparkles className='h-3 w-3 mr-1' />
                )}
                {isEnhancing
                  ? 'Enhancing...'
                  : message.enhanced
                    ? `Enhance Again (${(message.current_enhancement_level || 0) + 1}x)`
                    : 'Enhance'
                }
              </Button>

              <Button
                size='sm'
                variant='outline'
                onClick={() => setShowEnhanceSettings(prev => !prev)}
                className='h-8 px-2 text-xs'
                title='Configure ADetailer settings'
              >
                {showEnhanceSettings ? 'Hide ADetailer' : 'ADetailer'}
              </Button>
            </div>



            {/* Upscale Controls */}
            <div className="flex items-center gap-1">
              {upscalerModels.length > 0 && (
                <Select
                  value={selectedUpscaler}
                  onValueChange={(val) => {
                    setSelectedUpscaler(val);
                    localStorage.setItem('local-upscaler-model', val);
                  }}
                >
                  <SelectTrigger className="h-8 w-[140px] text-xs px-2 bg-transparent border-input/50 truncate">
                    <SelectValue placeholder="Model" />
                  </SelectTrigger>
                  <SelectContent>
                    {upscalerModels.map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              <Select value={scaleFactor} onValueChange={setScaleFactor}>
                <SelectTrigger className="h-8 w-[60px] text-xs px-2 bg-transparent border-input/50">
                  <SelectValue placeholder="2x" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2">2x</SelectItem>
                  <SelectItem value="3">3x</SelectItem>
                  <SelectItem value="4">4x</SelectItem>
                </SelectContent>
              </Select>

              <Button
                size='sm'
                variant='outline'
                onClick={handleUpscale}
                disabled={isUpscaling || isEnhancing}
                className='h-8 px-2 text-xs border bg-primary/5 text-primary hover:bg-primary/10 transition-colors'
                title={`Upscale ${scaleFactor}x`}
              >
                {isUpscaling ? (
                  <Loader2 className='h-3 w-3 mr-1 animate-spin' />
                ) : (
                  <ArrowUpCircle className='h-3 w-3 mr-1' />
                )}
                {isUpscaling ? '...' : 'Upscale'}
              </Button>
            </div>

            {/* Set Background Button */}
            <Button
              size='sm'
              variant='outline'
              onClick={handleSetBackground}
              className='h-8 px-2 text-xs border bg-primary/5 text-primary hover:bg-primary/10 transition-colors'
              title='Set as chat background'
            >
              <ImageIcon className='h-3 w-3 mr-1' />
              Set BG
            </Button>

            {/* Reset to last button (shown when enhanced) */}
            {message.enhanced && message.current_enhancement_level > 0 && (
              <Button
                size='sm'
                variant='outline'
                onClick={handleResetToLast}
                className='h-8 px-2 text-xs'
                title='Go back to previous enhancement level'
              >
                <Undo className='h-3 w-3 mr-1' />
                Reset to Last
              </Button>
            )}

            {/* Ask about image button */}
            <Button
              size='sm'
              variant='secondary'
              onClick={() => setShowImageQuery(true)}
              className='h-8 px-2 text-xs'
              disabled={isAnalyzing}
              title='Ask a question about this image'
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10-7-3-7-10-7Z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              Ask
            </Button>

            {/* Existing buttons */}
            {message.prompt && (
              <Button
                size='sm'
                variant='ghost'
                onClick={handleCopyPrompt}
                className='h-8 px-2 text-xs'
                title='Copy prompt to clipboard'
              >
                {isCopied ? <Check className='h-3 w-3 mr-1' /> : <Copy className='h-3 w-3 mr-1' />}
                {isCopied ? 'Copied' : 'Copy prompt'}
              </Button>
            )}

            <Button
              size='sm'
              variant='ghost'
              onClick={handleDownload}
              className='h-8 px-2 text-xs'
              title='Download image file'
            >
              <Download className='h-3 w-3 mr-1' />
              Download
            </Button>

            <Button
              size='sm'
              variant='ghost'
              onClick={() => setIsViewerOpen(true)}
              className='h-8 px-2 text-xs'
              title='View image in full size'
            >
              View
            </Button>

            <Button
              size='sm'
              variant='ghost'
              onClick={() => setMessages(prev => prev.filter(m => m.id !== message.id))}
              className='h-8 px-2 text-xs text-muted-foreground hover:text-foreground hover:bg-white/10'
              title='Delete image message'
            >
              <X className='h-3 w-3 mr-1' />
              Delete
            </Button>
          </div>

          {showEnhanceSettings && (
            <div className="mt-2 p-2 rounded-md border bg-gray-50 dark:bg-gray-900/20 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-xs">ADetailer Model</Label>
                  {adetailerModelOptions.length > 0 ? (
                    <Select
                      value={adetailerSettings.modelName}
                      onValueChange={(value) => setAdetailerSettings(prev => ({ ...prev, modelName: value }))}
                    >
                      <SelectTrigger className="w-full text-xs">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {adetailerModelOptions.map((model) => (
                          <SelectItem key={model} value={model} className="text-xs">
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <input
                      type="text"
                      value={adetailerSettings.modelName}
                      onChange={(e) => setAdetailerSettings(prev => ({ ...prev, modelName: e.target.value }))}
                      placeholder="face_yolov8n.pt"
                      className="w-full text-xs p-2 border rounded bg-background text-foreground"
                    />
                  )}
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Sampler</Label>
                  <Select
                    value={adetailerSettings.sampler}
                    onValueChange={(value) => setAdetailerSettings(prev => ({ ...prev, sampler: value }))}
                  >
                    <SelectTrigger className="w-full text-xs">
                      <SelectValue placeholder="Select sampler" />
                    </SelectTrigger>
                    <SelectContent>
                      {adetailerSamplerOptions.map((sampler) => (
                        <SelectItem key={sampler} value={sampler} className="text-xs">
                          {sampler}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Steps</Label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    step="1"
                    value={adetailerSettings.steps}
                    onChange={(e) => {
                      const next = Number.parseInt(e.target.value, 10);
                      if (Number.isFinite(next)) {
                        const clamped = Math.min(100, Math.max(1, next));
                        setAdetailerSettings(prev => ({ ...prev, steps: clamped }));
                      }
                    }}
                    className="w-full text-xs p-2 border rounded bg-background text-foreground"
                  />
                  <div className="text-[11px] text-muted-foreground">
                    Effective steps: {Math.max(1, Math.round(adetailerSettings.steps * adetailerSettings.strength))}
                  </div>
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Strength: {adetailerSettings.strength.toFixed(2)}</Label>
                  <input
                    type="range"
                    min="0.05"
                    max="1.00"
                    step="0.01"
                    value={adetailerSettings.strength}
                    onChange={(e) => setAdetailerSettings(prev => ({ ...prev, strength: parseFloat(e.target.value) }))}
                    className="w-full mt-1"
                  />
                </div>

                <div className="space-y-1">
                  <Label className="text-xs">Confidence: {adetailerSettings.confidence.toFixed(2)}</Label>
                  <input
                    type="range"
                    min="0.05"
                    max="0.95"
                    step="0.01"
                    value={adetailerSettings.confidence}
                    onChange={(e) => setAdetailerSettings(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                    className="w-full mt-1"
                  />
                </div>
              </div>

              <div className="mt-2 space-y-2">
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={adetailerSettings.useOriginalPrompt}
                    onChange={(e) => setAdetailerSettings(prev => ({ ...prev, useOriginalPrompt: e.target.checked }))}
                  />
                  Reuse original prompt
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={adetailerSettings.useOriginalNegativePrompt}
                    onChange={(e) => setAdetailerSettings(prev => ({ ...prev, useOriginalNegativePrompt: e.target.checked }))}
                    disabled={!message.negative_prompt}
                  />
                  Reuse original negative prompt
                </label>
                {!message.negative_prompt && (
                  <div className="text-[11px] text-muted-foreground">
                    No negative prompt stored for this image.
                  </div>
                )}
              </div>

              <div className="mt-2">
                <Label className="text-xs">Face Prompt</Label>
                <input
                  type="text"
                  value={adetailerSettings.facePrompt}
                  onChange={(e) => setAdetailerSettings(prev => ({ ...prev, facePrompt: e.target.value }))}
                  placeholder="detailed face, high quality, sharp focus"
                  className="w-full mt-1 text-xs p-2 border rounded bg-background text-foreground"
                />
              </div>

              {!adetailerAvailable && (
                <div className="mt-2 text-xs text-yellow-700 dark:text-yellow-300">
                  ADetailer not available. Install ultralytics and check the model directory.
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Image viewer dialog - keep existing implementation */}
      <Dialog open={isViewerOpen} onOpenChange={setIsViewerOpen}>
        <DialogContent className='fixed inset-0 flex items-center justify-center p-4'>
          <div className='relative w-full max-w-[90vw] sm:max-w-4xl bg-background rounded-lg p-6'>
            <DialogHeader>
              <DialogTitle>{message.prompt || 'Generated Image'}</DialogTitle>
              <Button
                variant='ghost'
                size='icon'
                className='absolute right-4 top-4'
                onClick={() => setIsViewerOpen(false)}
                aria-label='Close viewer'
              >
                <X className='h-4 w-4' />
              </Button>
            </DialogHeader>

            <div className='flex flex-col items-center mt-4'>
              <img
                src={imageUrl}
                alt={message.prompt || 'Generated image'}
                className='max-h-[70vh] max-w-full object-contain rounded-md'
              />

              <div className='w-full flex flex-col sm:flex-row justify-between items-start mt-4 gap-4'>
                <div className='flex-1 max-w-full sm:max-w-md'>
                  {message.prompt && (
                    <>
                      <p className='font-medium text-sm mb-1'>Prompt:</p>
                      <p className='text-sm mb-3'>{message.prompt}</p>
                    </>
                  )}

                  {message.negative_prompt && (
                    <>
                      <p className='font-medium text-sm mb-1'>Negative prompt:</p>
                      <p className='text-sm mb-3'>{message.negative_prompt}</p>
                    </>
                  )}

                  <div className='grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-600 dark:text-gray-400'>
                    {message.model && (
                      <div><span className='font-medium'>Model:</span> {message.model}</div>
                    )}
                    {message.width && message.height && (
                      <div><span className='font-medium'>Size:</span> {message.width}×{message.height}</div>
                    )}
                    {message.steps && (
                      <div><span className='font-medium'>Steps:</span> {message.steps}</div>
                    )}
                    {message.guidance_scale && (
                      <div><span className='font-medium'>Guidance Scale:</span> {message.guidance_scale}</div>
                    )}
                    {message.sampler && (
                      <div><span className='font-medium'>Sampler:</span> {message.sampler}</div>
                    )}
                    {message.seed !== undefined && message.seed !== -1 && (
                      <div><span className='font-medium'>Seed:</span> {message.seed}</div>
                    )}
                  </div>
                </div>

                <div className='flex gap-2 flex-shrink-0'>
                  {/* Regenerate in viewer */}
                  {onRegenerate && (
                    <Button
                      size='sm'
                      variant='ghost'
                      onClick={() => onRegenerate({
                        prompt: message.original_prompt || message.prompt || '',
                        negative_prompt: message.original_negative_prompt || message.negative_prompt || '',
                        width: message.original_width || message.width || 512,
                        height: message.original_height || message.height || 512,
                        steps: message.original_steps || message.steps || 20,
                        guidance_scale: message.original_guidance_scale || message.guidance_scale || 7.0,
                        sampler: message.original_sampler || message.sampler || 'Euler a',
                        seed: -1,
                        model: message.original_model || message.model || '',
                        gpu_id: message.gpuId ?? 0
                      })}
                      className='h-8 px-2 text-xs'
                      title='Regenerate with same parameters'
                    >
                      <RotateCcw className='h-3 w-3 mr-1' />
                      Regenerate {regenerationQueue > 0 && `(${regenerationQueue})`}
                    </Button>
                  )}

                  {message.prompt && (
                    <Button onClick={handleCopyPrompt} variant='outline' size='sm'>
                      {isCopied ? <Check className='h-4 w-4 mr-1' /> : <Copy className='h-4 w-4 mr-1' />}
                      {isCopied ? 'Copied' : 'Copy prompt'}
                    </Button>
                  )}
                  <Button onClick={handleDownload} size='sm'>
                    <Download className='h-4 w-4 mr-1' />
                    Download
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Image query dialog */}
      {
        showImageQuery && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="relative w-full max-w-md bg-background rounded-lg p-4 shadow-xl">
              <h3 className="text-lg font-semibold mb-3">Ask about this image</h3>
              <textarea
                value={imageQuery}
                onChange={(e) => setImageQuery(e.target.value)}
                placeholder="What would you like to know about this image? e.g. 'How many people are in this image?' or 'Describe the mood of this scene'"
                className="w-full p-2 border rounded resize-none bg-background text-foreground"
                rows={3}
                autoFocus
              />
              <div className="flex gap-2 mt-3">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowImageQuery(false);
                    setImageQuery('');
                  }}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => handleAnalyzeImage(imageQuery.trim())}
                  disabled={!imageQuery.trim() || isAnalyzing}
                >
                  {isAnalyzing ? 'Asking...' : 'Ask'}
                </Button>
              </div>
            </div>
          </div>
        )
      }
    </div >
  );
};

export default SimpleChatImageMessage;
