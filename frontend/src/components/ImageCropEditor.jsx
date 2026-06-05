import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { X, Check } from 'lucide-react';
import { clampCropRegion, defaultCropRegion, loadImageElement } from '../utils/imageCrop';

/**
 * Interactive crop editor: drag the crop box over the image.
 * Output region is in source-image pixel coordinates.
 */
export default function ImageCropEditor({
  open,
  imageSrc,
  imageName = 'image',
  targetWidth,
  targetHeight,
  initialRegion = null,
  onApply,
  onCancel,
}) {
  const containerRef = useRef(null);
  const dragRef = useRef(null);
  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [region, setRegion] = useState(null);
  const [displayBox, setDisplayBox] = useState({ w: 0, h: 0, offsetX: 0, offsetY: 0 });

  const tw = Math.max(1, Math.round(targetWidth || 512));
  const th = Math.max(1, Math.round(targetHeight || 512));

  useEffect(() => {
    if (!open || !imageSrc) return;
    let cancelled = false;
    (async () => {
      try {
        const img = await loadImageElement(imageSrc);
        if (cancelled) return;
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        setNatural({ w, h });
        setRegion(
          initialRegion
            ? clampCropRegion(initialRegion, w, h)
            : defaultCropRegion(w, h, tw, th)
        );
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, imageSrc, tw, th, initialRegion]);

  const updateDisplayBox = useCallback(() => {
    const el = containerRef.current;
    if (!el || !natural.w) return;
    const maxW = el.clientWidth;
    const maxH = Math.min(420, window.innerHeight * 0.5);
    const scale = Math.min(maxW / natural.w, maxH / natural.h, 1);
    const dw = natural.w * scale;
    const dh = natural.h * scale;
    setDisplayBox({
      w: dw,
      h: dh,
      offsetX: (maxW - dw) / 2,
      offsetY: 0,
      scale,
    });
  }, [natural]);

  useEffect(() => {
    if (!open) return;
    updateDisplayBox();
    window.addEventListener('resize', updateDisplayBox);
    return () => window.removeEventListener('resize', updateDisplayBox);
  }, [open, updateDisplayBox, natural]);

  const cropOverlay = useMemo(() => {
    if (!region || !displayBox.scale) return null;
    const s = displayBox.scale;
    return {
      width: region.width * s,
      height: region.height * s,
    };
  }, [region, displayBox.scale]);

  const pointerToSource = useCallback(
    (clientX, clientY, cropEl) => {
      if (!displayBox.scale || !cropEl) return null;
      const rect = cropEl.parentElement?.getBoundingClientRect();
      if (!rect) return null;
      const x = (clientX - rect.left) / displayBox.scale;
      const y = (clientY - rect.top) / displayBox.scale;
      return { x, y };
    },
    [displayBox.scale]
  );

  const onPointerDown = (e) => {
    if (!region) return;
    e.preventDefault();
    const start = pointerToSource(e.clientX, e.clientY);
    if (!start) return;
    dragRef.current = {
      startClient: { x: e.clientX, y: e.clientY },
      startRegion: { ...region },
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    const drag = dragRef.current;
    if (!drag || !displayBox.scale) return;
    const dx = (e.clientX - drag.startClient.x) / displayBox.scale;
    const dy = (e.clientY - drag.startClient.y) / displayBox.scale;
    setRegion(
      clampCropRegion(
        {
          x: drag.startRegion.x + dx,
          y: drag.startRegion.y + dy,
          width: drag.startRegion.width,
          height: drag.startRegion.height,
        },
        natural.w,
        natural.h
      )
    );
  };

  const onPointerUp = (e) => {
    dragRef.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* noop */
    }
  };

  const handleReset = () => {
    if (!natural.w) return;
    setRegion(defaultCropRegion(natural.w, natural.h, tw, th));
  };

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div
        className="relative w-full max-w-lg bg-background rounded-lg shadow-xl flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div>
            <h3 className="text-base font-semibold">Crop image</h3>
            <p className="text-xs text-muted-foreground truncate max-w-[280px]">
              {imageName} → {tw}×{th}px
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onCancel} aria-label="Close">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="p-4 space-y-3 overflow-y-auto flex-1">
          <p className="text-xs text-muted-foreground">
            Drag the box to choose what gets exported. Output is exactly {tw}×{th} pixels.
          </p>
          <div
            ref={containerRef}
            className="relative w-full bg-muted/40 rounded-lg overflow-hidden select-none touch-none flex justify-center"
            style={{ minHeight: 120 }}
          >
            {imageSrc && displayBox.w > 0 && (
              <div
                className="relative"
                style={{ width: displayBox.w, height: displayBox.h }}
              >
                <img
                  src={imageSrc}
                  alt=""
                  className="block w-full h-full pointer-events-none object-contain"
                  draggable={false}
                />
                {cropOverlay && (
                  <div
                    className="absolute border-2 border-white ring-1 ring-black/30 cursor-move"
                    style={{
                      left: region.x * displayBox.scale,
                      top: region.y * displayBox.scale,
                      width: cropOverlay.width,
                      height: cropOverlay.height,
                      boxShadow: '0 0 0 9999px rgba(0,0,0,0.5)',
                    }}
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerCancel={onPointerUp}
                  />
                )}
              </div>
            )}
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Selection:{' '}
              {region
                ? `${Math.round(region.width)}×${Math.round(region.height)} (source px)`
                : '…'}
            </span>
            <Button type="button" variant="link" className="h-auto p-0 text-xs" onClick={handleReset}>
              Reset to center
            </Button>
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t px-4 py-3">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            onClick={() => region && onApply?.(region)}
            disabled={!region}
          >
            <Check className="h-4 w-4 mr-1" />
            Apply crop
          </Button>
        </div>
      </div>
    </div>,
    document.body
  );
}
