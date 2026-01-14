import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Separator } from './ui/separator';
import { Slider } from './ui/slider';
import { Label } from './ui/label';
import { Card, CardContent } from './ui/card';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import {
  Settings,
  ImagePlus,
  Loader2,
  Download,
  Copy,
  Trash2,
  RefreshCw,
  AlertTriangle,
  Image as ImageIcon
} from 'lucide-react';

const ImageGen = () => {
  const {
    sdStatus,
    checkSdStatus,
    generateImage,
    generatedImages,
    isImageGenerating,
    apiError,
    clearError,
    PRIMARY_API_URL,
    SECONDARY_API_URL,
    setBackgroundImage // Add this
  } = useApp();

  // Image generation settings
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.0);
  const [sampler, setSampler] = useState('Euler a');
  const [seed, setSeed] = useState(-1);

  // UI State
  const [isCheckingStatus, setIsCheckingStatus] = useState(false);
  const [scaleFactor, setScaleFactor] = useState("2");
  const [selectedImage, setSelectedImage] = useState(null);
  const [samplers, setSamplers] = useState([
    'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a',
    'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DDIM'
  ]);

  // Determine the API URL to use for image paths
  const API_URL = SECONDARY_API_URL || PRIMARY_API_URL;

  // Check SD status on mount
  useEffect(() => {
    checkSdStatus();
  }, [checkSdStatus]);

  // Handle prompt submission
  const handleGenerateImage = async (e) => {
    e.preventDefault();

    if (!prompt.trim()) return;

    await generateImage(prompt, {
      negative_prompt: negativePrompt,
      width,
      height,
      steps,
      guidance_scale: guidanceScale,
      sampler,
      seed
    });
  };

  // Refresh SD status
  const handleRefreshStatus = async () => {
    setIsCheckingStatus(true);
    await checkSdStatus();
    setIsCheckingStatus(false);
  };

  // View image details
  const handleViewImage = (image) => {
    setSelectedImage(image);
  };

  // Download image
  const handleDownloadImage = (imagePath) => {
    const filename = imagePath.split('/').pop();

    // Create a link and trigger download
    const link = document.createElement('a');
    link.href = `${API_URL}/static/images/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Copy prompt to clipboard
  const handleCopyPrompt = (text) => {
    navigator.clipboard.writeText(text);
  };

  // Get image URL helper
  const getImageUrl = (imagePath) => {
    if (!imagePath) return '';

    // Handle different path formats
    const filename = imagePath.split('/').pop();
    return `${API_URL}/static/images/${filename}`;
  };

  return (
    <div className="container max-w-6xl mx-auto py-6">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Controls Column */}
        <div className="w-full md:w-1/3 space-y-6">
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-4">Image Generation</h2>

              {!sdStatus.automatic1111 && (
                <Alert className="mb-4">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Stable Diffusion Not Available</AlertTitle>
                  <AlertDescription>
                    Please make sure Automatic1111 WebUI is running on http://127.0.0.1:7860/
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-2"
                      onClick={handleRefreshStatus}
                      disabled={isCheckingStatus}
                    >
                      {isCheckingStatus ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Checking...
                        </>
                      ) : (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4" />
                          Check Again
                        </>
                      )}
                    </Button>
                  </AlertDescription>
                </Alert>
              )}

              {apiError && (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription className="flex justify-between items-center">
                    <span>{apiError}</span>
                    <Button variant="ghost" size="sm" onClick={clearError}>Dismiss</Button>
                  </AlertDescription>
                </Alert>
              )}

              <form onSubmit={handleGenerateImage} className="space-y-4">
                <div>
                  <Label htmlFor="prompt">Prompt</Label>
                  <Textarea
                    id="prompt"
                    placeholder="Describe the image you want to generate..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                    required
                    disabled={isImageGenerating || !sdStatus.automatic1111}
                  />
                </div>

                <div>
                  <Label htmlFor="negativePrompt">Negative Prompt</Label>
                  <Textarea
                    id="negativePrompt"
                    placeholder="Elements to avoid (e.g., blurry, low quality)..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    rows={2}
                    disabled={isImageGenerating || !sdStatus.automatic1111}
                  />
                </div>

                <Separator />

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="width">Width: {width}px</Label>
                    <Slider
                      id="width"
                      min={256}
                      max={1024}
                      step={64}
                      value={[width]}
                      onValueChange={(value) => setWidth(value[0])}
                      disabled={isImageGenerating || !sdStatus.automatic1111}
                    />
                  </div>

                  <div>
                    <Label htmlFor="height">Height: {height}px</Label>
                    <Slider
                      id="height"
                      min={256}
                      max={1024}
                      step={64}
                      value={[height]}
                      onValueChange={(value) => setHeight(value[0])}
                      disabled={isImageGenerating || !sdStatus.automatic1111}
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="steps">Steps: {steps}</Label>
                  <Slider
                    id="steps"
                    min={10}
                    max={50}
                    step={1}
                    value={[steps]}
                    onValueChange={(value) => setSteps(value[0])}
                    disabled={isImageGenerating || !sdStatus.automatic1111}
                  />
                </div>

                <div>
                  <Label htmlFor="guidanceScale">Guidance Scale: {guidanceScale}</Label>
                  <Slider
                    id="guidanceScale"
                    min={1}
                    max={20}
                    step={0.1}
                    value={[guidanceScale]}
                    onValueChange={(value) => setGuidanceScale(value[0])}
                    disabled={isImageGenerating || !sdStatus.automatic1111}
                  />
                </div>

                <div>
                  <Label htmlFor="sampler">Sampler</Label>
                  <select
                    id="sampler"
                    value={sampler}
                    onChange={(e) => setSampler(e.target.value)}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    disabled={isImageGenerating || !sdStatus.automatic1111}
                  >
                    {samplers.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <Label htmlFor="seed">Seed ({seed < 0 ? 'random' : seed})</Label>
                  <div className="flex gap-2">
                    <Input
                      id="seed"
                      type="number"
                      min="-1"
                      max="2147483647"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value))}
                      disabled={isImageGenerating || !sdStatus.automatic1111}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setSeed(-1)}
                      disabled={isImageGenerating || !sdStatus.automatic1111}
                    >
                      Random
                    </Button>
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  disabled={isImageGenerating || !prompt.trim() || !sdStatus.automatic1111}
                >
                  {isImageGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <ImagePlus className="mr-2 h-4 w-4" />
                      Generate Image
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Gallery Column */}
        <div className="w-full md:w-2/3">
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-4">Generated Images</h2>

              {generatedImages.length === 0 ? (
                <div className="text-center py-10 border border-dashed rounded-lg">
                  <ImageIcon className="mx-auto h-12 w-12 text-muted-foreground" />
                  <p className="mt-4 text-muted-foreground">No images generated yet</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {generatedImages.map((image) => (
                    <Card key={image.id} className="overflow-hidden">
                      <CardContent className="p-0">
                        <div
                          className="h-40 bg-cover bg-center cursor-pointer"
                          style={{ backgroundImage: `url(${getImageUrl(image.path)})` }}
                          onClick={() => handleViewImage(image)}
                        />
                        <div className="p-3">
                          <p className="text-sm line-clamp-2 h-10" title={image.prompt}>
                            {image.prompt}
                          </p>
                          <div className="flex justify-end mt-2 space-x-1">
                            <Button
                              size="icon"
                              variant="ghost"
                              onClick={() => handleCopyPrompt(image.prompt)}
                              title="Copy prompt"
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                            <Button
                              size="icon"
                              variant="ghost"
                              onClick={() => handleDownloadImage(image.path)}
                              title="Download image"
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Image Details Dialog */}
      <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Image Details</DialogTitle>
          </DialogHeader>
          {selectedImage && (
            <div className="space-y-4">
              <div className="relative aspect-square overflow-hidden rounded-lg">
                <img
                  src={getImageUrl(selectedImage.path)}
                  alt={selectedImage.prompt}
                  className="object-contain w-full h-full"
                />
              </div>

              <div className="space-y-2">
                <div>
                  <Label className="font-medium">Prompt</Label>
                  <div className="flex items-start mt-1">
                    <p className="text-sm flex-1">{selectedImage.prompt}</p>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => handleCopyPrompt(selectedImage.prompt)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {selectedImage.negative_prompt && (
                  <div>
                    <Label className="font-medium">Negative Prompt</Label>
                    <div className="flex items-start mt-1">
                      <p className="text-sm flex-1">{selectedImage.negative_prompt}</p>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => handleCopyPrompt(selectedImage.negative_prompt)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div>
                    <Label className="font-medium">Size</Label>
                    <p className="text-sm">{selectedImage.width}×{selectedImage.height}</p>
                  </div>
                  <div>
                    <Label className="font-medium">Steps</Label>
                    <p className="text-sm">{selectedImage.steps}</p>
                  </div>
                  <div>
                    <Label className="font-medium">Guidance Scale</Label>
                    <p className="text-sm">{selectedImage.guidance_scale}</p>
                  </div>
                  <div>
                    <Label className="font-medium">Sampler</Label>
                    <p className="text-sm">{selectedImage.sampler}</p>
                  </div>
                  <div>
                    <Label className="font-medium">Seed</Label>
                    <p className="text-sm">{selectedImage.seed}</p>
                  </div>
                </div>

                <div className="flex justify-end mt-4 gap-2">
                  <div className="flex items-center gap-1">
                    <Select value={scaleFactor} onValueChange={setScaleFactor}>
                      <SelectTrigger className="h-9 w-[70px] px-2 bg-transparent text-secondary-foreground border-input">
                        <SelectValue placeholder="2x" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2">2x</SelectItem>
                        <SelectItem value="3">3x</SelectItem>
                        <SelectItem value="4">4x</SelectItem>
                      </SelectContent>
                    </Select>

                    <Button
                      variant="outline"
                      onClick={async () => {
                        if (!selectedImage) return;
                        try {
                          const response = await fetch(`${PRIMARY_API_URL}/sd-local/upscale`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              image_url: getImageUrl(selectedImage.path),
                              scale_factor: parseFloat(scaleFactor),
                              strength: 0.2,
                              prompt: selectedImage.prompt,
                              // We don't have GPU ID in image metadata usually, default to 0
                              gpu_id: 0
                            })
                          });

                          if (!response.ok) throw new Error('Upscale failed');

                          const result = await response.json();
                          if (result.status === 'success') {
                            // Close dialog and refresh (or show success)
                            setSelectedImage(null);
                            handleRefreshStatus(); // To refresh gallery
                            // Ideally we'd show the new image immediately
                          }
                        } catch (e) {
                          console.error('Upscale error:', e);
                        }
                      }}
                    >
                      <ArrowUpCircle className="mr-2 h-4 w-4" />
                      Upscale
                    </Button>
                  </div>

                  <Button
                    variant="outline"
                    onClick={() => {
                      if (selectedImage) {
                        setBackgroundImage(getImageUrl(selectedImage.path));
                        // Optional: show toast
                      }
                    }}
                  >
                    <ImageIcon className="mr-2 h-4 w-4" />
                    Set BG
                  </Button>

                  <Button onClick={() => handleDownloadImage(selectedImage.path)}>
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </Button>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default ImageGen;