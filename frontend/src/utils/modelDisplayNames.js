const KNOWN_MODELS = [
  { key: 'deepseek-v4-pro-thinking', display: 'DeepSeek V4 Pro Thinking', short: 'DST', color: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' },
  { key: 'deepseek-v4-pro', display: 'DeepSeek V4 Pro', short: 'DS', color: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' },
  { key: 'glm-5.1-thinking', display: 'GLM 5.1 Thinking', short: 'G5T', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'glm-5.1', display: 'GLM 5.1', short: 'G5', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'glm-5-thinking', display: 'GLM 5 Thinking', short: 'G5T', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'glm-5', display: 'GLM 5', short: 'G5', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'glm-4.7-thinking', display: 'GLM 4.7 Thinking', short: 'G4T', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'glm-4.7', display: 'GLM 4.7', short: 'G4', color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  { key: 'mistral-large-3', display: 'Mistral Large 3', short: 'ML', color: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
];

function normalize(raw) {
  return String(raw || '')
    .replace(/^endpoint-/, '')
    .replace(/^api-/, '')
    .toLowerCase()
    .replace(/[^a-z0-9.-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/-$/, '')
    .trim();
}

export function matchModelName(raw) {
  if (!raw) return null;
  const normalized = normalize(raw);
  let best = null;
  let bestLen = 0;
  for (const model of KNOWN_MODELS) {
    if (normalized === model.key) return { ...model };
    if (normalized.includes(model.key) && model.key.length > bestLen) {
      best = { ...model };
      bestLen = model.key.length;
    }
  }
  if (best) return best;
  const cleaned = normalized.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  return { key: normalized, display: cleaned, short: cleaned.slice(0, 2).toUpperCase(), color: 'bg-muted text-muted-foreground border-border' };
}

export function getModelBadgeColor(modelInfo) {
  return modelInfo?.color || 'bg-muted text-muted-foreground border-border';
}
