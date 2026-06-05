import React, { useState, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Upload, X, Send, Sparkles, ChevronDown, ChevronRight, Crop } from 'lucide-react';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import ImageCropEditor from './ImageCropEditor';
import { CROP_PRESETS, cropImageToBlob } from '../utils/imageCrop';
import { createRouteTraceId, logRouteTrace, resolveUnifiedRequestRoute } from '../utils/requestRouting';

const ChatImageUploadButton = () => {
  const {
    sendMessage,
    isGenerating,
    generateUniqueId,
    setMessages,
    userProfile,
    primaryModel,
    primaryIsAPI,
    PRIMARY_API_URL,
    activeCharacter,
    settings,
    userCharacter,
  } = useApp();

  // Component state: selectedImages is always an array (1 = single, many = batch)
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedImages, setSelectedImages] = useState([]); // array of { file, name, size, type, base64, previewUrl? }
  const [messageText, setMessageText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchUpscaleQueueProgress, setBatchUpscaleQueueProgress] = useState(null); // { current: number, total: number } when running

  // Crop before upload
  const [cropEnabled, setCropEnabled] = useState(true);
  const [cropWidth, setCropWidth] = useState(512);
  const [cropHeight, setCropHeight] = useState(512);
  const [cropEditorIndex, setCropEditorIndex] = useState(null);
  const [isCropping, setIsCropping] = useState(false);

  // Enhancement State
  const [showEnhancementOptions, setShowEnhancementOptions] = useState(false);
  const [enableAdetailer, setEnableAdetailer] = useState(false);
  const [enableUpscale, setEnableUpscale] = useState(false);

  const [adetailerSettings, setAdetailerSettings] = useState({
    modelName: 'face_yolov8n.pt',
    strength: 0.35,
    steps: 20,
    confidence: 0.3,
    sampler: 'euler_a'
  });

  const [upscaleSettings, setUpscaleSettings] = useState({
    scale_factor: "2",
    model_name: ''
  });

  // New batch mode: Upscale + Rembg + Auto-crop
  const [batchUpscaleRembgMode, setBatchUpscaleRembgMode] = useState(false);
  const [rembgPadding, setRembgPadding] = useState(10);

  const [availableAdetailerModels, setAvailableAdetailerModels] = useState([]);
  const [upscalerModels, setUpscalerModels] = useState([]);

  // Fetch models when dialog opens
  React.useEffect(() => {
    if (isDialogOpen && PRIMARY_API_URL) {
      const fetchModels = async () => {
        try {
          const [adetailerRes, upscaleRes] = await Promise.all([
            fetch(`${PRIMARY_API_URL}/sd-local/adetailer-models`),
            fetch(`${PRIMARY_API_URL}/sd-local/upscalers`)
          ]);

          if (adetailerRes.ok) {
            const data = await adetailerRes.json();
            if (data.models) setAvailableAdetailerModels(data.models);
          }

          if (upscaleRes.ok) {
            const data = await upscaleRes.json();
            if (data.models) {
              setUpscalerModels(data.models);
              if (data.models.length > 0 && !upscaleSettings.model_name) {
                setUpscaleSettings(prev => ({ ...prev, model_name: data.models[0] }));
              }
            }
          }
        } catch (error) {
          console.error('Error fetching enhancement models:', error);
        }
      };
      fetchModels();
    }
  }, [isDialogOpen, PRIMARY_API_URL]);

  // File input ref
  const fileInputRef = useRef(null);

  // Allowed image types
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
  const maxFileSize = 10 * 1024 * 1024; // 10MB limit

  // Convert file to base64
  const fileToBase64 = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }, []);

  const replaceImageItem = useCallback(async (item, blob, cropRegion = null) => {
    if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
    const type = blob.type || 'image/jpeg';
    const ext = type === 'image/png' ? 'png' : 'jpg';
    const baseName = item.name.replace(/\.[^.]+$/, '') || 'image';
    const file = new File([blob], `${baseName}_cropped.${ext}`, { type });
    const base64Data = await fileToBase64(file);
    return {
      ...item,
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      base64: base64Data,
      previewUrl: URL.createObjectURL(file),
      cropRegion: cropRegion || item.cropRegion || null,
      cropped: true,
    };
  }, [fileToBase64]);

  const applyCropToItemAt = useCallback(async (index, cropRegion = null) => {
    const item = selectedImages[index];
    if (!item) return;
    const w = Math.max(1, Math.round(cropWidth));
    const h = Math.max(1, Math.round(cropHeight));
    const src = item.previewUrl || item.base64;
    const blob = await cropImageToBlob(src, w, h, cropRegion || item.cropRegion);
    const next = await replaceImageItem(item, blob, cropRegion || item.cropRegion);
    setSelectedImages((prev) => prev.map((img, i) => (i === index ? next : img)));
  }, [selectedImages, cropWidth, cropHeight, replaceImageItem]);

  const applyCropToAll = useCallback(async () => {
    if (!selectedImages.length) return;
    setIsCropping(true);
    try {
      const w = Math.max(1, Math.round(cropWidth));
      const h = Math.max(1, Math.round(cropHeight));
      const next = [];
      for (const item of selectedImages) {
        const src = item.previewUrl || item.base64;
        const blob = await cropImageToBlob(src, w, h, item.cropRegion);
        next.push(await replaceImageItem(item, blob, item.cropRegion));
      }
      setSelectedImages(next);
    } catch (error) {
      console.error('Batch crop failed:', error);
      alert(error?.message || 'Failed to crop images.');
    } finally {
      setIsCropping(false);
    }
  }, [selectedImages, cropWidth, cropHeight, replaceImageItem]);

  // Process a single file into { file, name, size, type, base64, previewUrl }
  const processFile = useCallback(async (file) => {
    return {
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      cropRegion: null,
      cropped: false,
    };
  }, []);

  // Handle file selection (single or multiple files)
  const handleFilesSelect = useCallback(async (files) => {
    const fileList = Array.from(files || []);
    if (fileList.length === 0) return;

    const valid = [];
    for (const file of fileList) {
      if (!allowedTypes.includes(file.type)) {
        alert(`Skipping ${file.name}: invalid type. Use JPEG, PNG, WebP, or GIF.`);
        continue;
      }
      if (file.size > maxFileSize) {
        alert(`Skipping ${file.name}: file must be under 10MB.`);
        continue;
      }
      valid.push(file);
    }
    if (valid.length === 0) return;

    setIsProcessing(true);
    try {
      const processed = await Promise.all(valid.map((f) => processFile(f)));
      setSelectedImages(processed);
      setIsDialogOpen(true);
    } catch (error) {
      console.error('Error processing images:', error);
      alert('Error processing images. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  }, [processFile]);

  // Handle button click - open file picker
  const handleButtonClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Handle file input change (supports multiple)
  const handleFileInputChange = useCallback((e) => {
    const files = e.target.files;
    if (files?.length) {
      handleFilesSelect(files);
    }
    e.target.value = '';
  }, [handleFilesSelect]);

  // Remove one image from batch
  const removeImageAt = useCallback((index) => {
    setSelectedImages((prev) => {
      const next = prev.filter((_, i) => i !== index);
      prev[index]?.previewUrl && URL.revokeObjectURL(prev[index].previewUrl);
      return next;
    });
  }, []);

  const handleSendWithImage = useCallback(async () => {
    if (!selectedImages?.length || isGenerating || isCropping) return;

    // NEW: Batch Upscale + Rembg Mode
    if (batchUpscaleRembgMode) {
      setIsProcessing(true);
      setBatchUpscaleQueueProgress({ current: 0, total: selectedImages.length });

      try {
        for (let i = 0; i < selectedImages.length; i++) {
          setBatchUpscaleQueueProgress({ current: i + 1, total: selectedImages.length });

          const item = selectedImages[i];

          // Step 1: Upload original image
          const formData = new FormData();
          formData.append('file', item.file);
          const uploadResponse = await fetch(`${PRIMARY_API_URL}/upload_avatar`, {
            method: 'POST',
            body: formData
          });
          if (!uploadResponse.ok) throw new Error(`Failed to upload image ${i + 1}`);
          const uploadData = await uploadResponse.json();
          let imageUrl = uploadData.file_url;

          // Step 2: Upscale
          const upscaleResponse = await fetch(`${PRIMARY_API_URL}/sd-local/upscale`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_url: imageUrl,
              scale_factor: parseFloat(upscaleSettings.scale_factor),
              model_name: upscaleSettings.model_name,
            })
          });
          if (!upscaleResponse.ok) throw new Error(`Failed to upscale image ${i + 1}`);
          const upscaleData = await upscaleResponse.json();
          imageUrl = upscaleData.image_url;

          // Step 3: Remove background + get bounding box
          const rembgResponse = await fetch(`${PRIMARY_API_URL}/rembg/remove-background`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_url: imageUrl,
              padding: rembgPadding
            })
          });
          if (!rembgResponse.ok) throw new Error(`Failed to remove background ${i + 1}`);
          const rembgData = await rembgResponse.json();
          const bbox = rembgData.bounding_box;
          const rembgImageUrl = rembgData.image_url;

          // Step 4: Auto-crop to bounding box
          let finalImageUrl = rembgImageUrl;
          if (bbox) {
            const cropBlob = await cropImageToBlob(
              rembgImageUrl,
              bbox.x2 - bbox.x1,
              bbox.y2 - bbox.y1,
              {
                x: bbox.x1,
                y: bbox.y1,
                width: bbox.x2 - bbox.x1,
                height: bbox.y2 - bbox.y1
              }
            );

            const cropFormData = new FormData();
            cropFormData.append('file', cropBlob, `cropped_${item.name}`);
            const cropUpload = await fetch(`${PRIMARY_API_URL}/upload_avatar`, {
              method: 'POST',
              body: cropFormData
            });
            if (!cropUpload.ok) throw new Error(`Failed to upload cropped image ${i + 1}`);
            const cropData = await cropUpload.json();
            finalImageUrl = cropData.file_url;
          }

          // Download final processed image
          const downloadResponse = await fetch(finalImageUrl);
          const blob = await downloadResponse.blob();
          const downloadUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = downloadUrl;
          a.download = `processed_${item.name}`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(downloadUrl);
        }
      } finally {
        setIsProcessing(false);
        setBatchUpscaleQueueProgress(null);
        handleCloseDialog();
        return;
      }
    }

    let imagesToSend = selectedImages;
    if (cropEnabled) {
      setIsCropping(true);
      try {
        const w = Math.max(1, Math.round(cropWidth));
        const h = Math.max(1, Math.round(cropHeight));
        const cropped = [];
        for (const item of selectedImages) {
          if (item.cropped) {
            cropped.push(item);
            continue;
          }
          const src = item.previewUrl || item.base64;
          const blob = await cropImageToBlob(src, w, h, item.cropRegion);
          cropped.push(await replaceImageItem(item, blob, item.cropRegion));
        }
        imagesToSend = cropped;
        setSelectedImages(cropped);
      } catch (error) {
        console.error('Crop before send failed:', error);
        alert(error?.message || 'Failed to crop images before upload.');
        setIsCropping(false);
        return;
      }
      setIsCropping(false);
    }

    const total = imagesToSend.length;
    const runEnhancements = enableAdetailer || enableUpscale;
    if (runEnhancements) setBatchUpscaleQueueProgress({ current: 0, total });

    try {
      // 1. Add user message with text (if any)
      if (messageText.trim()) {
        const userMsg = {
          id: generateUniqueId(),
          role: 'user',
          content: messageText.trim()
        };
        if (settings.multiRoleMode && userCharacter) {
          userMsg.characterId = userCharacter.id;
          userMsg.characterName = userCharacter.name;
          userMsg.avatar = userCharacter.avatar;
        }
        setMessages(prev => [...prev, userMsg]);
      }

      // 2. Upload each image and add image message (same format as SimpleChatImageButton)
      const uploaded = []; // { messageId, imageUrl }
      for (let i = 0; i < imagesToSend.length; i++) {
        const item = imagesToSend[i];
        const formData = new FormData();
        formData.append('file', item.file);
        const uploadResponse = await fetch(`${PRIMARY_API_URL}/upload_avatar`, {
          method: 'POST',
          body: formData
        });
        if (!uploadResponse.ok) throw new Error(`Failed to upload image ${i + 1}`);
        const uploadData = await uploadResponse.json();
        const imageUrl = uploadData.file_url;
        const imageMessage = {
          id: generateUniqueId(),
          role: 'bot',
          type: 'image',
          content: total > 1 ? `Uploaded image ${i + 1}/${total}` : 'Uploaded image',
          imagePath: imageUrl,
          prompt: 'Uploaded image',
          width: cropEnabled ? Math.round(cropWidth) : undefined,
          height: cropEnabled ? Math.round(cropHeight) : undefined,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, imageMessage]);
        uploaded.push({ messageId: imageMessage.id, imageUrl });
      }

      handleCloseDialog();
      if (!runEnhancements) setBatchUpscaleQueueProgress(null);

      // 3. Run enhancements in queue (one image after another)
      if (runEnhancements && uploaded.length > 0) {
        (async () => {
          for (let i = 0; i < uploaded.length; i++) {
            setBatchUpscaleQueueProgress({ current: i + 1, total: uploaded.length });
            let currentImageUrl = uploaded[i].imageUrl;
            const msgId = uploaded[i].messageId;
            try {
              if (enableAdetailer) {
                const enhanceResponse = await fetch(`${PRIMARY_API_URL}/sd-local/enhance-adetailer`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    image_url: currentImageUrl,
                    model_name: adetailerSettings.modelName,
                    strength: adetailerSettings.strength,
                    steps: adetailerSettings.steps,
                    confidence: adetailerSettings.confidence,
                    sampler: adetailerSettings.sampler,
                    prompt: messageText,
                  })
                });
                if (enhanceResponse.ok) {
                  const result = await enhanceResponse.json();
                  if (result.status === 'success' && result.enhanced_image_url) {
                    currentImageUrl = result.enhanced_image_url;
                    setMessages(prev => prev.map(msg =>
                      msg.id === msgId ? {
                        ...msg,
                        imagePath: currentImageUrl,
                        enhanced: true,
                        current_enhancement_level: (msg.current_enhancement_level || 0) + 1,
                        enhancement_history: msg.enhancement_history ? [...msg.enhancement_history, currentImageUrl] : [uploaded[i].imageUrl, currentImageUrl],
                        enhancement_settings: { ...adetailerSettings }
                      } : msg
                    ));
                  }
                }
              }
              if (enableUpscale) {
                const upscaleResponse = await fetch(`${PRIMARY_API_URL}/sd-local/upscale`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    image_url: currentImageUrl,
                    scale_factor: parseFloat(upscaleSettings.scale_factor),
                    model_name: upscaleSettings.model_name,
                    prompt: messageText
                  })
                });
                if (upscaleResponse.ok) {
                  const result = await upscaleResponse.json();
                  if (result.status === 'success' && result.image_url) {
                    currentImageUrl = result.image_url;
                    setMessages(prev => prev.map(msg =>
                      msg.id === msgId ? {
                        ...msg,
                        imagePath: currentImageUrl,
                        enhanced: true,
                        upscaled: true,
                        width: (msg.width || 512) * parseFloat(upscaleSettings.scale_factor),
                        height: (msg.height || 512) * parseFloat(upscaleSettings.scale_factor),
                        current_enhancement_level: (msg.current_enhancement_level || 0) + 1,
                        enhancement_history: msg.enhancement_history ? [...msg.enhancement_history, currentImageUrl] : [uploaded[i].imageUrl, currentImageUrl]
                      } : msg
                    ));
                  }
                }
              }
            } catch (err) {
              console.error(`Enhancement failed for image ${i + 1}:`, err);
            }
          }
          setBatchUpscaleQueueProgress(null);
        })();
      }

      // 4. Vision API only for first image when there's message text
      if (messageText.trim() && imagesToSend.length > 0) {
        const route = resolveUnifiedRequestRoute({
          primaryModel,
          primaryIsAPI,
          settings,
          requestPurpose: 'chat_image_analysis',
        });
        const traceId = createRouteTraceId();
        logRouteTrace({
          action: 'chat_image_analysis',
          route,
          requestPurpose: 'chat_image_analysis',
          traceId,
        });
        const firstImage = imagesToSend[0];
        const systemPrompt = activeCharacter
          ? `System: You are ${activeCharacter.name}. ${activeCharacter.description}\n\n${activeCharacter.model_instructions}`
          : 'System: You are a helpful AI assistant.';
        const fullPrompt = `${systemPrompt}\n\nHuman: ${messageText.trim()}`;
        const response = await fetch(`${PRIMARY_API_URL}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Router-Trace-Id': traceId },
          body: JSON.stringify({
            prompt: fullPrompt,
            model_name: route.effectiveModel || primaryModel,
            image_base64: firstImage.base64.split(',')[1],
            image_type: firstImage.type,
            temperature: 0.7,
            max_tokens: 1024,
            userProfile: { id: userProfile?.id ?? 'anonymous' },
            request_purpose: 'chat_image_analysis',
            selected_model: route.selectedModel || undefined,
            round_robin_enabled: route.autoEnabled,
          })
        });
        if (response.ok) {
          const result = await response.json();
          const botMsg = {
            id: generateUniqueId(),
            role: 'bot',
            content: result.text || 'No response from vision model',
            modelId: 'primary'
          };
          setMessages(prev => [...prev, botMsg]);
        }
      }
    } catch (error) {
      console.error('Error processing images:', error);
      alert(`Error: ${error.message}`);
      setBatchUpscaleQueueProgress(null);
    }
  }, [
    selectedImages,
    messageText,
    isGenerating,
    isCropping,
    cropEnabled,
    cropWidth,
    cropHeight,
    replaceImageItem,
    generateUniqueId,
    setMessages,
    primaryModel,
    primaryIsAPI,
    userProfile,
    PRIMARY_API_URL,
    settings,
    userCharacter,
    activeCharacter,
    enableAdetailer,
    enableUpscale,
    adetailerSettings,
    upscaleSettings,
  ]);

  // Handle closing the dialog and cleanup
  const handleCloseDialog = useCallback(() => {
    selectedImages.forEach((item) => item.previewUrl && URL.revokeObjectURL(item.previewUrl));
    setIsDialogOpen(false);
    setSelectedImages([]);
    setMessageText('');
  }, [selectedImages]);

  // Format file size for display
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <>
      {/* Hidden file input - multiple for batch upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept={allowedTypes.join(',')}
        multiple
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      {/* Upload button - supports single or batch */}
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 rounded-full p-0"
        title="Upload image(s) — select multiple for batch"
        onClick={handleButtonClick}
        disabled={isProcessing || isGenerating}
      >
        {isProcessing ? (
          <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : (
          <Upload className="h-4 w-4" />
        )}
      </Button>
      {/* Batch upscale queue progress (e.g. after dialog closed) */}
      {batchUpscaleQueueProgress && (
        <span className="text-xs text-muted-foreground whitespace-nowrap" title="Batch upscale in progress">
          Upscale {batchUpscaleQueueProgress.current}/{batchUpscaleQueueProgress.total}
        </span>
      )}

      {/* Image preview and message dialog */}
      {isDialogOpen && createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div
            className="relative w-full max-w-2xl bg-background rounded-lg p-6 shadow-xl max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between border-b pb-3 mb-4">
              <div>
                <h3 className="text-lg font-semibold">
                  {selectedImages.length > 1 ? `Upload ${selectedImages.length} images (batch)` : 'Upload Image'}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {selectedImages.length > 1 ? 'Add images to chat; optional batch upscale below.' : 'Add an image to your message'}
                </p>
              </div>
              <Button variant="ghost" size="icon" className="-mt-1 -mr-2" onClick={handleCloseDialog}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-4 overflow-y-auto flex-1">
              {/* Image preview(s) - grid for batch */}
              {selectedImages.length === 1 && !batchUpscaleRembgMode && (
                <div className="space-y-2">
                  <Label className="text-sm font-medium">
                    Image Preview
                  </Label>
                  <div className='border rounded-lg p-4 bg-muted/50'>
                    {selectedImages.map((item, index) => (
                      <div key={index} className="relative group">
                        <img
                          src={item.previewUrl}
                          alt={item.name}
                          className='max-w-full max-h-64 mx-auto rounded object-contain'
                        />
                        <div className="absolute top-1 right-1 flex gap-0.5">
                          {cropEnabled && (
                            <Button
                              type="button"
                              variant="secondary"
                              size="icon"
                              className="h-6 w-6 rounded-full opacity-90 hover:opacity-100"
                              onClick={() => setCropEditorIndex(index)}
                              aria-label="Crop image"
                              title={`Crop to ${cropWidth}×${cropHeight}`}
                            >
                              <Crop className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                        {item.cropped && (
                          <span className="absolute bottom-6 left-1 rounded bg-emerald-600/90 px-1 py-0.5 text-[10px] text-white">
                            cropped
                          </span>
                        )}
                        <div className='mt-2 text-xs text-muted-foreground text-center'>
                          {item.name} • {formatFileSize(item.size)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Show simple count in batch mode */}
              {selectedImages.length > 1 && (
                <div className="text-center py-4 bg-muted/20 rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    {selectedImages.length} image{selectedImages.length !== 1 ? 's' : ''} ready to process
                  </p>
                </div>
              )}

              {/* Batch Upscale + Rembg + Auto-crop */}
              <div className="border rounded-lg p-3 space-y-3 bg-orange-50/20 dark:bg-orange-950/20 border-orange-200 dark:border-orange-900">
                <div className="flex items-center justify-between gap-2">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-medium flex items-center gap-1.5">
                      <Sparkles className="h-4 w-4" />
                      Batch Upscale & Remove Background
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Upscale, remove background, and auto-crop to content
                    </p>
                  </div>
                  <Switch
                    checked={batchUpscaleRembgMode}
                    onCheckedChange={setBatchUpscaleRembgMode}
                  />
                </div>

                {batchUpscaleRembgMode && (
                  <div className="pt-2 border-t space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <Label className="text-xs">Scale Factor</Label>
                        <Select
                          value={upscaleSettings.scale_factor}
                          onValueChange={(val) => setUpscaleSettings(p => ({ ...p, scale_factor: val }))}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1.5">1.5x</SelectItem>
                            <SelectItem value="2">2x</SelectItem>
                            <SelectItem value="3">3x</SelectItem>
                            <SelectItem value="4">4x</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Upscaler</Label>
                        <Select
                          value={upscaleSettings.model_name}
                          onValueChange={(val) => setUpscaleSettings(p => ({ ...p, model_name: val }))}
                        >
                          <SelectTrigger className="h-8 text-xs">
                            <SelectValue placeholder="Auto" />
                          </SelectTrigger>
                          <SelectContent>
                            {upscalerModels.map(m => (
                              <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                     <div className="flex items-center justify-between">
                       <Label className="text-xs">Padding around crop: {rembgPadding}px</Label>
                       <input
                         type="range"
                         min="0"
                         max="100"
                         value={rembgPadding}
                         onChange={(e) => setRembgPadding(parseInt(e.target.value))}
                         className="w-24"
                       />
                     </div>

                     <div className="flex items-center justify-between">
                       <Label className="text-xs">Download after processing</Label>
                       <Switch
                         checked={true}
                         disabled={true}
                       />
                     </div>
                   </div>
                 )}
               </div>

              {/* Crop to exact dimensions */}
              <div className="border rounded-lg p-3 space-y-3 bg-muted/20">
                <div className="flex items-center justify-between gap-2">
                  <div className="space-y-0.5">
                    <Label className="text-sm font-medium flex items-center gap-1.5">
                      <Crop className="h-4 w-4" />
                      Crop to size
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Export every image at the same width and height (center crop by default).
                    </p>
                  </div>
                  <Switch checked={cropEnabled} onCheckedChange={setCropEnabled} />
                </div>

                {cropEnabled && (
                  <div className="space-y-3 pt-1 border-t">
                    <div className="flex flex-wrap gap-2">
                      {CROP_PRESETS.map((p) => (
                        <Button
                          key={p.id}
                          type="button"
                          size="sm"
                          variant={
                            cropWidth === p.width && cropHeight === p.height ? 'secondary' : 'outline'
                          }
                          className="text-xs h-7"
                          onClick={() => {
                            setCropWidth(p.width);
                            setCropHeight(p.height);
                          }}
                        >
                          {p.label}
                        </Button>
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <Label htmlFor="crop-w" className="text-xs">Width (px)</Label>
                        <input
                          id="crop-w"
                          type="number"
                          min={64}
                          max={4096}
                          step={1}
                          value={cropWidth}
                          onChange={(e) => setCropWidth(Number(e.target.value) || 512)}
                          className="w-full h-8 rounded-md border border-input bg-background px-2 text-sm"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="crop-h" className="text-xs">Height (px)</Label>
                        <input
                          id="crop-h"
                          type="number"
                          min={64}
                          max={4096}
                          step={1}
                          value={cropHeight}
                          onChange={(e) => setCropHeight(Number(e.target.value) || 512)}
                          className="w-full h-8 rounded-md border border-input bg-background px-2 text-sm"
                        />
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        disabled={isCropping || !selectedImages.length}
                        onClick={() => void applyCropToAll()}
                      >
                        {isCropping ? 'Cropping…' : `Apply ${cropWidth}×${cropHeight} to all`}
                      </Button>
                      <span className="text-xs text-muted-foreground self-center">
                        Or use <Crop className="inline h-3 w-3" /> on each thumbnail to adjust.
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Message text input */}
              <div className="space-y-2">
                <Label htmlFor="message-text" className="text-sm font-medium">
                  Message (optional)
                </Label>
                <Textarea
                  id="message-text"
                  placeholder="Add a message about this image..."
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  rows={3}
                  className="resize-none"
                />
              </div>

              {/* Enhancement Options (ADetailer, Upscale — batch = queue) */}
              <div className="border rounded-lg p-3 space-y-3">
                <button
                  className="flex items-center justify-between w-full text-sm font-medium"
                  onClick={() => setShowEnhancementOptions(!showEnhancementOptions)}
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-purple-500" />
                    Enhancement options {selectedImages.length > 1 ? '(batch: all in queue)' : ''}
                  </div>
                  {showEnhancementOptions ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </button>

                {showEnhancementOptions && (
                  <div className="pt-2 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                    <div className="border-b" />

                    {/* ADetailer Toggle */}
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label className="text-sm">ADetailer (Face/Object Enhancement)</Label>
                        <p className="text-xs text-muted-foreground">Automatically detect and refine details</p>
                      </div>
                      <Switch
                        checked={enableAdetailer}
                        onCheckedChange={setEnableAdetailer}
                      />
                    </div>

                    {enableAdetailer && (
                      <div className="pl-2 border-l-2 border-purple-200 dark:border-purple-900 space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <Label className="text-xs">Model</Label>
                            <Select
                              value={adetailerSettings.modelName}
                              onValueChange={(val) => setAdetailerSettings(p => ({ ...p, modelName: val }))}
                            >
                              <SelectTrigger className="h-8 text-xs">
                                <SelectValue placeholder="Model" />
                              </SelectTrigger>
                              <SelectContent>
                                {availableAdetailerModels.map(m => (
                                  <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-1">
                            <Label className="text-xs">Strength ({adetailerSettings.strength})</Label>
                            <input
                              type="range"
                              min="0.1" max="1.0" step="0.05"
                              value={adetailerSettings.strength}
                              onChange={(e) => setAdetailerSettings(p => ({ ...p, strength: parseFloat(e.target.value) }))}
                              className="w-full h-8 cursor-pointer"
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Upscale Toggle */}
                    <div className="flex items-center justify-between pt-2 border-t">
                      <div className="space-y-0.5">
                        <Label className="text-sm">Upscaling</Label>
                        <p className="text-xs text-muted-foreground">Increase resolution and sharpness</p>
                      </div>
                      <Switch
                        checked={enableUpscale}
                        onCheckedChange={setEnableUpscale}
                      />
                    </div>

                    {enableUpscale && (
                      <div className="pl-2 border-l-2 border-blue-200 dark:border-blue-900 space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="space-y-1">
                            <Label className="text-xs">Scale Factor</Label>
                            <Select
                              value={upscaleSettings.scale_factor}
                              onValueChange={(val) => setUpscaleSettings(p => ({ ...p, scale_factor: val }))}
                            >
                              <SelectTrigger className="h-8 text-xs">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="1.5">1.5x</SelectItem>
                                <SelectItem value="2">2x</SelectItem>
                                <SelectItem value="3">3x</SelectItem>
                                <SelectItem value="4">4x</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-1">
                            <Label className="text-xs">Upscaler</Label>
                            <Select
                              value={upscaleSettings.model_name}
                              onValueChange={(val) => setUpscaleSettings(p => ({ ...p, model_name: val }))}
                            >
                              <SelectTrigger className="h-8 text-xs">
                                <SelectValue placeholder="Auto" />
                              </SelectTrigger>
                              <SelectContent>
                                {upscalerModels.map(m => (
                                  <SelectItem key={m} value={m} className="text-xs">{m}</SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Info note */}
              <div className="text-xs text-muted-foreground bg-muted/50 p-3 rounded">
                <strong>Note:</strong> Vision support is currently in development.
                The image will be uploaded but may not be processed by the AI model yet.
              </div>
            </div>

            {/* Footer buttons */}
            <div className="flex justify-end gap-3 mt-4 pt-4 border-t">
              <Button variant="outline" onClick={handleCloseDialog}>
                Cancel
              </Button>
              <Button
                onClick={handleSendWithImage}
                disabled={!selectedImages?.length || isGenerating || isCropping}
                className="min-w-[100px]"
              >
                <Send className="w-4 h-4 mr-2" />
                {isCropping
                  ? 'Cropping…'
                  : selectedImages?.length > 1
                    ? `Send ${selectedImages.length} images`
                    : 'Send'}
              </Button>
            </div>
          </div>
        </div>,
        document.body
      )}

      <ImageCropEditor
        open={cropEditorIndex !== null && selectedImages[cropEditorIndex]}
        imageSrc={
          cropEditorIndex !== null
            ? selectedImages[cropEditorIndex]?.previewUrl || selectedImages[cropEditorIndex]?.base64
            : null
        }
        imageName={cropEditorIndex !== null ? selectedImages[cropEditorIndex]?.name : ''}
        targetWidth={cropWidth}
        targetHeight={cropHeight}
        initialRegion={
          cropEditorIndex !== null ? selectedImages[cropEditorIndex]?.cropRegion : null
        }
        onCancel={() => setCropEditorIndex(null)}
        onApply={async (region) => {
          const idx = cropEditorIndex;
          if (idx == null) return;
          setIsCropping(true);
          try {
            const item = selectedImages[idx];
            const w = Math.max(1, Math.round(cropWidth));
            const h = Math.max(1, Math.round(cropHeight));
            const src = item.previewUrl || item.base64;
            const blob = await cropImageToBlob(src, w, h, region);
            const next = await replaceImageItem(item, blob, region);
            setSelectedImages((prev) => prev.map((img, i) => (i === idx ? next : img)));
            setCropEditorIndex(null);
          } catch (error) {
            console.error(error);
            alert(error?.message || 'Crop failed.');
          } finally {
            setIsCropping(false);
          }
        }}
      />
    </>
  );
};

export default ChatImageUploadButton;
