import { getBackendUrl } from '../config/api';

const API_BASE = getBackendUrl();

export async function logInteraction({
  characterId,
  characterName = '',
  surface = 'chat',
  userMessage = '',
  characterResponse = '',
  emotionalState = null,
}) {
  if (!characterId) {
    console.warn('[mirrorInteractionLog] No characterId, skipping log');
    return null;
  }

  try {
    const resp = await fetch(`${API_BASE}/lattice/interaction-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character_id: characterId,
        character_name: characterName,
        surface,
        user_message: String(userMessage || ''),
        character_response: String(characterResponse || ''),
        emotional_state: emotionalState,
      }),
    });

    if (!resp.ok) {
      console.warn(`[mirrorInteractionLog] POST failed (${resp.status})`);
      return null;
    }

    const data = await resp.json();
    return data?.result || null;
  } catch (err) {
    console.error('[mirrorInteractionLog] log error:', err);
    return null;
  }
}

export async function getInteractionContext(characterId, limit = 50) {
  if (!characterId) {
    console.warn('[mirrorInteractionLog] No characterId for context');
    return null;
  }

  try {
    const resp = await fetch(
      `${API_BASE}/lattice/interaction-log/${encodeURIComponent(characterId)}/context?limit=${limit}`,
      { method: 'GET' },
    );

    if (!resp.ok) {
      console.warn(`[mirrorInteractionLog] GET context failed (${resp.status})`);
      return null;
    }

    const data = await resp.json();
    return data?.context || null;
  } catch (err) {
    console.error('[mirrorInteractionLog] get context error:', err);
    return null;
  }
}

export function formatInteractionContext(context) {
  if (!context) return '';

  const { character_name, formatted_text, raw_count } = context;
  if (!formatted_text) return '';

  return `=== ${character_name || 'Character'} Interaction History (${raw_count} past exchanges) ===\n${formatted_text}\n`;
}

export function buildInteractionMemoryBlock(context) {
  if (!context) return [];

  const { formatted_text, character_name, raw_count } = context;
  if (!formatted_text) return [];

  const block = `[Interaction History for ${character_name || 'character'}]\nYou have had ${raw_count} past exchanges with the user across various surfaces. Here is your history:\n\n${formatted_text}`;

  return [
    {
      id: `interaction_log_${Date.now()}`,
      role: 'system',
      content: block,
    },
  ];
}
