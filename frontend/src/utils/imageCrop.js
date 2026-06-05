/**
 * Canvas-based image crop/resize utilities (no external crop library).
 */

export function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = src;
  });
}

/** Largest centered crop rect matching target aspect ratio (source pixel coords). */
export function defaultCropRegion(srcW, srcH, targetW, targetH) {
  const tw = Math.max(1, Math.round(targetW));
  const th = Math.max(1, Math.round(targetH));
  const targetAspect = tw / th;
  const srcAspect = srcW / srcH;
  let cropW;
  let cropH;
  if (srcAspect > targetAspect) {
    cropH = srcH;
    cropW = srcH * targetAspect;
  } else {
    cropW = srcW;
    cropH = srcW / targetAspect;
  }
  return {
    x: (srcW - cropW) / 2,
    y: (srcH - cropH) / 2,
    width: cropW,
    height: cropH,
  };
}

export function clampCropRegion(region, srcW, srcH) {
  const w = Math.min(Math.max(8, region.width), srcW);
  const h = Math.min(Math.max(8, region.height), srcH);
  const x = Math.min(Math.max(0, region.x), srcW - w);
  const y = Math.min(Math.max(0, region.y), srcH - h);
  return { x, y, width: w, height: h };
}

export async function cropImageToBlob(
  src,
  targetW,
  targetH,
  cropRegion = null,
  mimeType = 'image/jpeg',
  quality = 0.92
) {
  const img = await loadImageElement(src);
  const srcW = img.naturalWidth || img.width;
  const srcH = img.naturalHeight || img.height;
  const region = clampCropRegion(
    cropRegion || defaultCropRegion(srcW, srcH, targetW, targetH),
    srcW,
    srcH
  );

  const outW = Math.max(1, Math.round(targetW));
  const outH = Math.max(1, Math.round(targetH));
  const canvas = document.createElement('canvas');
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas not supported');

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, outW, outH);
  ctx.drawImage(
    img,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    outW,
    outH
  );

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Crop export failed'))),
      mimeType,
      quality
    );
  });
}

export async function cropFileToFile(file, targetW, targetH, cropRegion = null) {
  const src = URL.createObjectURL(file);
  try {
    const blob = await cropImageToBlob(src, targetW, targetH, cropRegion);
    const base = file.name.replace(/\.[^.]+$/, '') || 'image';
    const ext = blob.type === 'image/png' ? 'png' : 'jpg';
    return new File([blob], `${base}_cropped.${ext}`, { type: blob.type });
  } finally {
    URL.revokeObjectURL(src);
  }
}

export const CROP_PRESETS = [
  { id: '1x1', label: '1:1 (512×512)', width: 512, height: 512 },
  { id: '3x2', label: '3:2 (768×512)', width: 768, height: 512 },
  { id: '2x3', label: '2:3 (512×768)', width: 512, height: 768 },
  { id: '16x9', label: '16:9 (1024×576)', width: 1024, height: 576 },
  { id: '9x16', label: '9:16 (576×1024)', width: 576, height: 1024 },
];
