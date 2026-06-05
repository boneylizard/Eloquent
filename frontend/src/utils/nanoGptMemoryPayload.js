/**
 * Merge NanoGPT Context Memory fields into /generate JSON bodies.
 * Backend only applies when the configured LLM endpoint URL contains nano-gpt.com.
 */

export function mergeNanoGptMemoryIntoPayload(payload, settings) {
  if (!settings?.nanoGptContextMemoryEnabled) {
    return payload;
  }
  const raw = parseInt(settings.nanoGptContextMemoryExpirationDays, 10);
  const days = Number.isFinite(raw) ? Math.min(365, Math.max(1, raw)) : 30;
  const mode = settings.nanoGptContextMemoryMode === 'suffix' ? 'suffix' : 'header';
  return {
    ...payload,
    nano_gpt_context_memory_enabled: true,
    nano_gpt_context_memory_mode: mode,
    nano_gpt_context_memory_expiration_days: days,
  };
}
