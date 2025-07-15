import React, { useState } from 'react';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Textarea } from './ui/textarea';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { useApp } from '../contexts/AppContext';
import { Image, Loader2, X } from 'lucide-react';

/**
 * A button component that can be integrated into Chat.jsx to allow 
 * users to generate images directly within chat conversations
 */
const ChatImageButton = ({ onImageGenerated }) => {
  const { 
    sdStatus, 
    checkSdStatus, 
    generateImage, 
    isImageGenerating,
    apiError,
    clearError 
  } = useApp();
  
  const [isOpen, setIsOpen] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(20); // Reduced default for chat
  const [guidanceScale, setGuidanceScale] = useState(7.0);
  
  // Check SD status when popover opens
  const handleOpenChange = (open) => {
    if (open) {
      checkSdStatus();
    }
    setIsOpen(open);
  };
  
  // Handle image generation
  const handleGenerateImage = async () => {
    if (!prompt.trim()) return;
    
    try {
      await generateImage(prompt, {
        negative_prompt: negativePrompt,
        width,
        height,
        steps,
        guidance_scale: guidanceScale,
        sampler: 'Euler a', // Default sampler
        seed: -1 // Random seed
      });
      
      // Get the most recently generated image (first in the array)
      if (typeof onImageGenerated === 'function') {
        // Pass the generated image to the parent component
        onImageGenerated({ 
          type: 'image',
          prompt,
          negative_prompt: negativePrompt,
          width,
          height,
          steps,
          guidance_scale: guidanceScale
        });
      }
      
      // Reset form and close popover
      setPrompt('');
      setNegativePrompt('');
      setIsOpen(false);
    } catch (error) {
      console.error('Error generating image:', error);
    }
  };
  
  return (
    <Popover open={isOpen} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-8 w-8 rounded-full p-0"
          title="Generate Image"
        >
          <Image className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">Generate Image</h3>
            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="h-6 w-6">
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          {!sdStatus?.automatic1111 && (
            <div className="text-xs text-red-500">
              ⚠️ Stable Diffusion is not available. Make sure Automatic1111 is running.
            </div>
          )}
          
          {apiError && (
            <div className="text-xs text-red-500">
              Error: {apiError}
              <Button variant="link" size="sm" onClick={clearError} className="h-auto p-0 text-xs">
                Dismiss
              </Button>
            </div>
          )}
          
          <div className="space-y-2">
            <Label htmlFor="prompt" className="text-xs">Prompt</Label>
            <Textarea
              id="prompt"
              placeholder="Describe the image..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              className="resize-none text-sm"
              disabled={isImageGenerating || !sdStatus?.automatic1111}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="negativePrompt" className="text-xs">Negative Prompt</Label>
            <Textarea
              id="negativePrompt"
              placeholder="Elements to avoid..."
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              rows={1}
              className="resize-none text-sm"
              disabled={isImageGenerating || !sdStatus?.automatic1111}
            />
          </div>
          
          <Separator />
          
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <Label htmlFor="size">Size</Label>
              <span>{width}×{height}</span>
            </div>
            <div className="flex gap-2">
              <Button 
                size="sm" 
                variant={width === 512 && height === 512 ? "secondary" : "outline"} 
                className="text-xs h-7"
                onClick={() => { setWidth(512); setHeight(512); }}
                disabled={isImageGenerating}
              >
                1:1
              </Button>
              <Button 
                size="sm" 
                variant={width === 768 && height === 512 ? "secondary" : "outline"} 
                className="text-xs h-7"
                onClick={() => { setWidth(768); setHeight(512); }}
                disabled={isImageGenerating}
              >
                3:2
              </Button>
              <Button 
                size="sm" 
                variant={width === 512 && height === 768 ? "secondary" : "outline"} 
                className="text-xs h-7"
                onClick={() => { setWidth(512); setHeight(768); }}
                disabled={isImageGenerating}
              >
                2:3
              </Button>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <Label htmlFor="steps">Steps: {steps}</Label>
            </div>
            <Slider
              id="steps"
              min={10}
              max={50}
              step={1}
              value={[steps]}
              onValueChange={(value) => setSteps(value[0])}
              disabled={isImageGenerating}
            />
          </div>
          
<div className="space-y-2">
    <Label className="text-xs">Guidance Scale: {guidanceScale.toFixed(1)}</Label>
    <Slider 
        min={0.5}        // CHANGED: from min={1} to min={0.5} for FLUX compatibility
        max={15} 
        step={0.1} 
        value={[guidanceScale]} 
        onValueChange={val => setGuidanceScale(val[0])} 
        disabled={isImageGenerating} 
    />
</div>
          
          <Button
            className="w-full"
            onClick={handleGenerateImage}
            disabled={isImageGenerating || !prompt.trim() || !sdStatus?.automatic1111}
          >
            {isImageGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              "Generate Image"
            )}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default ChatImageButton;