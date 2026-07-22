import React, { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Eye } from 'lucide-react';
import { cn } from '@/lib/utils';

const VISION_MODELS = [
  { id: 'LFM2.5-VL-450M-Extract', label: 'LFM2.5-VL-450M Extract', description: 'Recommended · very small, fast, structured image reading', mode: 'extract' },
  { id: 'LFM2.5-VL-450M', label: 'LFM2.5-VL-450M', description: 'Small general-purpose image description and Q&A', mode: 'chat' },
  { id: 'LFM2.5-VL-1.6B-Extract', label: 'LFM2.5-VL-1.6B Extract', description: 'Larger structured image reader for more difficult material', mode: 'extract' },
  { id: 'LFM2.5-VL-1.6B', label: 'LFM2.5-VL-1.6B', description: 'Larger general-purpose image description and Q&A', mode: 'chat' },
  { id: 'gemma-3-4b-it', label: 'Gemma 3 4B', description: 'Legacy compatibility · general vision and OCR', mode: 'chat' },
];

export function VisionModelSelector({ 
  value, 
  onChange, 
  disabled = false,
  className,
  showDescription = true 
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className={cn('w-full max-w-[300px]', className)}>
        <SelectValue placeholder="Select vision model..." />
      </SelectTrigger>
      <SelectContent position="popper" sideOffset={5}>
        {VISION_MODELS.map((model) => (
          <SelectItem key={model.id} value={model.id}>
            <div className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2">
                <Eye className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="font-medium">{model.label}</span>
                {model.mode && model.mode !== 'direct' && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                    {model.mode === 'extract' ? 'JSON Extract' : 'Vision Chat'}
                  </span>
                )}
              </div>
              {showDescription && model.description && (
                <span className="text-xs text-muted-foreground ml-5.5">{model.description}</span>
              )}
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export function VisionModelSettings({ 
  visionModel, 
  setVisionModel, 
  visionSchema, 
  setVisionSchema,
  className 
}) {
  const selectedModel = useMemo(() => 
    VISION_MODELS.find(m => m.id === visionModel), 
    [visionModel]
  );

  const isExtractMode = selectedModel?.mode === 'extract';
  const isChatMode = selectedModel?.mode === 'chat';

  return (
    <div className={cn('space-y-4 p-4 rounded-lg border border-border bg-background', className)}>
      <div className="flex items-center gap-2">
        <Eye className="h-5 w-5 text-primary" />
        <h3 className="font-semibold text-lg">Local image reader</h3>
      </div>
      
      <p className="text-sm text-muted-foreground">
        Mirid reads each attached image locally, then gives those observations to your chat model.
        The 450M Extract model is the recommended default: quick, compact, and purpose-built for reliable JSON.
      </p>

      <div className="space-y-2">
        <label className="text-sm font-medium">Vision Model</label>
        <VisionModelSelector 
          value={visionModel || ''} 
          onChange={setVisionModel}
          showDescription={true}
        />
      </div>

      {selectedModel && selectedModel.id && (
        <div className="space-y-2">
          {isExtractMode && (
            <>
              <label className="text-sm font-medium">What should Mirid notice? (YAML)</label>
              <textarea
                value={visionSchema || ''}
                onChange={(e) => setVisionSchema(e.target.value)}
                placeholder="description: A concise, factual account of the image&#10;objects: Important objects and where they appear&#10;scene_type: The kind of scene, document, screenshot, or interface&#10;text_content: Visible text, transcribed accurately&#10;colours: The dominant colours"
                className="w-full min-h-[120px] p-2 rounded border border-border bg-background text-sm font-mono resize-y"
                rows={8}
              />
              <p className="text-xs text-muted-foreground">
                These field names become the JSON keys. Leave this empty to use Mirid's balanced default.
              </p>
            </>
          )}
          {isChatMode && (
            <div className="p-3 rounded bg-muted/50 border border-border/50">
              <p className="text-sm text-muted-foreground">
                The image reader writes a natural description for the chat model. No schema is needed.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
