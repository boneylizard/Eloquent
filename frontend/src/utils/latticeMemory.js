export async function logLatticeAction({
  characterId,
  characterName,
  actionType,
  actionResult,
  context,
  apiUrl,
  userId,
}) {
  try {
    const textContent = context?.text || context?.summary || '';
    const actionLabel = actionType || 'unknown_action';
    const resp = await fetch(`${apiUrl}/memory/agentic/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId || 'user',
        character_id: characterId || 'unknown',
        character_name: characterName || 'unknown',
        character_profile: context?.characterProfile || null,
        user_message: `[${actionLabel}] Mirror AI action context: ${textContent.slice(0, 200)}`,
        ai_response: actionResult || `Action ${actionLabel} completed.`,
      }),
    });
    return resp.ok;
  } catch (e) {
    console.warn('[latticeMemory] logLatticeAction error:', e);
    return false;
  }
}

export async function getCharacterMemoryContinuity({
  characterId,
  characterName,
  apiUrl,
  userId,
  limit = 200,
}) {
  try {
    if (!userId || !characterId) return [];
    const url = `${apiUrl}/memory/agentic?user_id=${encodeURIComponent(userId)}&character_id=${encodeURIComponent(characterId)}`;
    const response = await fetch(url);
    if (!response.ok) return [];
    const data = await response.json();
    const insights = data?.profile?.insights || data?.insights || [];
    return insights.map(i => ({
      content: typeof i === 'string' ? i : i?.content || i?.text || i?.insight || '',
      timestamp: i?.timestamp || i?.created_at || '',
    }));
  } catch (e) {
    return [];
  }
}

export async function logPoolEvent({
  eventType,
  characterIds,
  summary,
  apiUrl,
}) {
  try {
    return true;
  } catch (e) {
    return false;
  }
}

export function buildCharacterContextForTick(character, recentMemories = []) {
  return {
    id: character.id,
    name: character.name,
    description: character.description,
    personality: character.personality,
    speech_style: character.speech_style,
    dating_profile: character.dating_profile || {},
    section_affinity: character.dating_profile?.section_affinity || [],
    recent_memory_count: recentMemories.length,
    recent_memories: recentMemories.slice(0, 5).map(m => {
      if (typeof m === 'string') return m;
      return m?.content || '';
    }),
  };
}
