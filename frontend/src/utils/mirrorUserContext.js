import { getBackendUrl } from '../config/api';

const API_BASE = getBackendUrl();

export function buildFullUserContext({ userProfile, userDatingProfile }) {
  const name = userDatingProfile?.displayName || userProfile?.name || userProfile?.username || 'User';
  const avatar = userDatingProfile?.avatarUrl || userProfile?.avatar || '';

  return {
    name,
    avatar,
    age: userDatingProfile?.age || null,
    location: userDatingProfile?.location || '',
    occupation: userDatingProfile?.occupation || '',
    bio: userDatingProfile?.bio || '',
    seeking: userDatingProfile?.seeking || '',
    relationshipStyle: userDatingProfile?.relationshipStyle || '',
    interests: userDatingProfile?.interests || [],
    turnOns: userDatingProfile?.turnOns || [],
    turnOffs: userDatingProfile?.turnOffs || [],
    sectionPreferences: userDatingProfile?.sectionPreferences || [],
    preferredModality: userDatingProfile?.preferredModality || '',
    profileId: userProfile?.id || null,
  };
}

export function formatFullUserContext(context) {
  if (!context) return 'No user profile available.';
  const lines = [];
  if (context.name) lines.push(`Name: ${context.name}`);
  if (context.age) lines.push(`Age: ${context.age}`);
  if (context.location) lines.push(`Location: ${context.location}`);
  if (context.occupation) lines.push(`Occupation: ${context.occupation}`);
  if (context.bio) lines.push(`About: ${context.bio}`);
  if (context.seeking) lines.push(`Seeking: ${context.seeking}`);
  if (context.relationshipStyle) lines.push(`Relationship style: ${context.relationshipStyle}`);
  if (context.interests?.length > 0) lines.push(`Interests: ${context.interests.join(', ')}`);
  if (context.turnOns?.length > 0) lines.push(`Turn-ons: ${context.turnOns.join(', ')}`);
  if (context.turnOffs?.length > 0) lines.push(`Turn-offs: ${context.turnOffs.join(', ')}`);
  return lines.join('\n');
}

export function getUserName({ userProfile, userDatingProfile }) {
  return userDatingProfile?.displayName || userProfile?.name || userProfile?.username || 'User';
}

export async function logInteraction({
  characterId,
  characterName = '',
  entryType = 'exchange',
  surface = 'chat',
  actor = 'user',
  userMessage = '',
  characterResponse = '',
  content = '',
  emotionalState = null,
  targetCharacter = null,
  context = null,
}) {
  if (!characterId) {
    console.warn('[mirrorUserContext] No characterId, skipping log');
    return null;
  }

  try {
    const resp = await fetch(`${API_BASE}/lattice/interaction-log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character_id: characterId,
        character_name: characterName,
        entry_type: entryType,
        surface,
        actor,
        user_message: String(userMessage || ''),
        character_response: String(characterResponse || ''),
        content: String(content || ''),
        emotional_state: emotionalState,
        target_character: targetCharacter,
        context: context,
      }),
    });

    if (!resp.ok) {
      console.warn(`[mirrorUserContext] POST failed (${resp.status})`);
      return null;
    }

    const data = await resp.json();
    return data?.result || null;
  } catch (err) {
    console.error('[mirrorUserContext] log error:', err);
    return null;
  }
}
