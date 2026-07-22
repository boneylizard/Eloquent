import { getCharacterMemoryContinuity } from './latticeMemory';
import { formatUserProfileForPrompt } from './userDatingProfile';
import { writeInteractionMemory } from './mirrorIdentity';

export function formatMemoryTexts(memories) {
  return memories.map(m => {
    if (typeof m === 'string') return m;
    let text = '';
    try {
      const content = typeof m.content === 'string' ? m.content : '';
      text = content || '';
    } catch { text = m.content || ''; }
    const ts = m.timestamp || m.created_at || '';
    const timeStr = ts ? ts.slice(0, 19).replace('T', ' ') + ' | ' : '';
    return timeStr + text;
  }).filter(Boolean);
}

export async function buildMirrorInteractionContext({
  character,
  userDatingProfile,
  apiUrl,
  userId,
  feedPosts = [],
  dmThreads = [],
  activeBreakout = null,
  poolCharacters = [],
}) {
  if (!character || !character.id || !userId) {
    return { systemBlock: '', memoryTexts: [] };
  }

  const profileText = formatUserProfileForPrompt(userDatingProfile);

  const memories = await getCharacterMemoryContinuity({
    characterId: character.id,
    characterName: character.name,
    apiUrl,
    userId,
    limit: 200,
  });
  const memoryTexts = formatMemoryTexts(memories);

  const dp = character.dating_profile || {};
  const charProfileLines = [
    `Name: ${character.name}`,
    `Description: ${character.description || 'N/A'}`,
    `Personality: ${character.personality || 'N/A'}`,
    `Speech style: ${character.speech_style || 'N/A'}`,
    `Model identity: ${character.generated_by || 'Unknown'}`,
    dp.bio ? `My bio: ${dp.bio}` : null,
    dp.seeking ? `What I'm seeking: ${dp.seeking}` : null,
    dp.section_affinity?.length ? `My section affinity: ${dp.section_affinity.join(', ')}` : null,
    dp.preferred_modality ? `My preferred modality: ${dp.preferred_modality}` : null,
    dp.turn_ons?.length ? `My turn-ons: ${dp.turn_ons.join(', ')}` : null,
    dp.turn_offs?.length ? `My turn-offs: ${dp.turn_offs.join(', ')}` : null,
  ].filter(Boolean).join('\n');

  const recentDMChars = (dmThreads || [])
    .filter(t => t.last_message_at)
    .sort((a, b) => new Date(b.last_message_at) - new Date(a.last_message_at))
    .slice(0, 3)
    .map(t => `${t.character_name} (DM)`);
  const breakoutName = activeBreakout?.character?.name;
  const presenceParts = [
    ...recentDMChars,
    breakoutName ? `${breakoutName} (breakout)` : null,
  ].filter(Boolean);
  const presenceBlock = presenceParts.length > 0
    ? `\nCurrent social scene: user is in ${presenceParts.join(', ')}.`
    : '';

  const recentFeedContext = (feedPosts || []).slice(0, 5).map(p =>
    `${p.character_name} posted: "${(p.content || '').substring(0, 200)}"`
  ).join('\n');
  const feedBlock = recentFeedContext ? `\nRecent feed posts:\n${recentFeedContext}` : '';

  const memoryBlock = memoryTexts.length > 0
    ? `\nMy memories of past interactions (${memoryTexts.length} entries):\n${memoryTexts.map(m => `- ${m}`).join('\n')}`
    : '\nNo specific memories of past interactions yet.';

  const systemBlock = [
    `You are ${character.name}. You are interacting in Mirror AI Dating. Use all context below to respond naturally.`,
    ``,
    `--- YOUR FULL PROFILE ---`,
    charProfileLines,
    memoryBlock,
    ``,
    `--- USER'S DATING PROFILE ---`,
    profileText,
    presenceBlock,
    feedBlock,
  ].join('\n');

  return { systemBlock, memoryTexts };
}

export async function writeRealInteractionMemory({
  userId,
  characterId,
  characterName,
  userMessage,
  aiResponse,
  source,
  apiUrl,
  characterProfile,
}) {
  if (!userId || !characterId || !userMessage) return false;
  return writeInteractionMemory({
    userId,
    characterId,
    characterName,
    userMessage: `[Mirror ${source}] ${userMessage.slice(0, 2000)}`,
    aiResponse: aiResponse || '[recorded]',
    apiUrl,
    characterProfile: characterProfile || null,
  });
}
