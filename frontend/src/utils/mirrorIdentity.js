import { getBackendUrl, getMemoryUrl } from '../config/api';

export function resolveCanonicalCharacterId(character) {
  if (!character) return null;
  return character.id || character.character_id || null;
}

export function resolveCanonicalCharacterName(character) {
  if (!character) return 'Unknown';
  return character.name || character.character_name || 'Unknown';
}

export function resolveCanonicalUserId(userProfile, memoryContext) {
  return memoryContext?.activeProfileId || userProfile?.id || null;
}

export function buildRelationshipScope(userId, characterId) {
  if (!userId || !characterId) return null;
  return `relationship:${userId}:${characterId}`;
}

export function buildScopeKey(userId, characterId) {
  if (!userId || !characterId) return null;
  return `${userId}:${characterId}`;
}

const MEMORY_API_URL = getMemoryUrl();

export async function writeInteractionMemory({
  userId,
  characterId,
  characterName,
  userMessage,
  aiResponse,
  source,
  sourceThreadId,
  characterProfile,
  apiUrl,
}) {
  const baseUrl = apiUrl || MEMORY_API_URL;
  if (!userId || !characterId) return false;
  try {
    const resp = await fetch(`${baseUrl}/memory/agentic/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        character_id: characterId,
        character_name: characterName || 'Unknown',
        character_profile: characterProfile || null,
        user_message: userMessage,
        ai_response: aiResponse || '[acknowledged]',
      }),
    });
    return resp.ok;
  } catch (e) {
    console.warn('[mirrorIdentity] writeInteractionMemory error:', e);
    return false;
  }
}

export async function retrieveInteractionMemories({
  userId,
  characterId,
  query,
  limit = 200,
  apiUrl,
}) {
  const baseUrl = apiUrl || MEMORY_API_URL;
  if (!userId || !characterId) return [];
  try {
    const params = new URLSearchParams({
      user_id: userId,
      character_id: characterId,
    });
    if (query) params.set('query', query);
    const resp = await fetch(`${baseUrl}/memory/agentic?${params.toString()}`);
    if (!resp.ok) return [];
    const data = await resp.json();
    const insights = data?.profile?.insights || data?.insights || [];
    return insights.map(i => ({
      content: typeof i === 'string' ? i : i?.content || i?.text || i?.insight || '',
      timestamp: i?.timestamp || i?.created_at || '',
    }));
  } catch (e) {
    return [];
  }
}
