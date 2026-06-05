/** Provider badge emoji for NanoGPT / API model display. */
export function providerIcon(provider) {
  const p = String(provider || '').toLowerCase();
  if (!p) return '⬜';
  if (p.includes('anthropic') || p.includes('claude')) return '🟠';
  if (p.includes('openai') || p === 'gpt') return '🟢';
  if (p.includes('gemini') || p.includes('google')) return '🔵';
  if (p.includes('deepseek')) return '🔷';
  if (p.includes('zhipu') || p.includes('glm')) return '🟣';
  if (p.includes('mistral')) return '🟡';
  if (p.includes('moonshot') || p.includes('kimi')) return '🌙';
  if (p.includes('nanogpt') || p === 'nano') return '⚡';
  if (p.includes('qwen')) return '🟠';
  if (p.includes('minimax')) return '⬛';
  if (p.includes('nvidia')) return '🟩';
  return '⬜';
}

export function inferProviderFromModelId(modelId) {
  const id = String(modelId || '').toLowerCase();
  if (!id) return '';
  if (id.includes('claude') || id.includes('anthropic')) return 'anthropic';
  if (id.includes('gpt') || id.includes('openai') || id.includes('o1') || id.includes('o3')) return 'openai';
  if (id.includes('gemini') || id.includes('google')) return 'google';
  if (id.includes('deepseek')) return 'deepseek';
  if (id.includes('glm') || id.includes('zhipu')) return 'zhipu';
  if (id.includes('mistral')) return 'mistral';
  if (id.includes('moonshot') || id.includes('kimi')) return 'moonshot';
  if (id.includes('qwen')) return 'qwen';
  if (id.includes('minimax')) return 'minimax';
  if (id.includes('nvidia')) return 'nvidia';
  return '';
}
