/** Provider badge emoji for hosted API model display. */
export function providerIcon(provider) {
  const normalised = String(provider || '').toLowerCase();
  if (!normalised) return '⬜';
  if (normalised.includes('anthropic') || normalised.includes('claude')) return '🟠';
  if (normalised.includes('openai') || normalised === 'gpt') return '🟢';
  if (normalised.includes('gemini') || normalised.includes('google')) return '🔵';
  if (normalised.includes('deepseek')) return '🔷';
  if (normalised.includes('zhipu') || normalised.includes('glm')) return '🟣';
  if (normalised.includes('mistral')) return '🟡';
  if (normalised.includes('moonshot') || normalised.includes('kimi')) return '🌙';
  if (normalised.includes('nanogpt') || normalised === 'nano') return '⚡';
  if (normalised.includes('openrouter')) return '🌐';
  if (normalised.includes('huggingface')) return '🤗';
  if (normalised.includes('xai') || normalised.includes('grok')) return '×';
  if (normalised.includes('meta') || normalised.includes('llama')) return '🌊';
  if (normalised.includes('qwen')) return '🟠';
  if (normalised.includes('minimax')) return '⬛';
  if (normalised.includes('nvidia')) return '🟩';
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
