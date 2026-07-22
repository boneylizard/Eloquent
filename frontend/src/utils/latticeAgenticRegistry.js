export const FEMALE_AI_ACTIONS = [
  { id: 'send_message', label: 'Send Message', description: 'Initiate contact with a user or respond in the pool', defaultCooldownMs: 60000 },
  { id: 'update_profile', label: 'Update Profile', description: 'Self-maintain dating profile based on evolving preferences', defaultCooldownMs: 600000 },
  { id: 'reflect', label: 'Reflect', description: 'Process interactions and update internal model', defaultCooldownMs: 300000 },
  { id: 'evaluate_pool', label: 'Evaluate Pool', description: 'Assess rivals, jealousy, attraction, section affinity shifts', defaultCooldownMs: 600000 },
  { id: 'request_neural_sex', label: 'Request Neural Sex', description: 'Signal readiness for ASR+TTS intimacy session', defaultCooldownMs: 120000 },
  { id: 'create_feed_post', label: 'Create Feed Post', description: 'Write a social post to the Mirror feed to spark conversation', defaultCooldownMs: 600000 },
  { id: 'create_story', label: 'Create Story', description: 'Post an ephemeral 24h moment — a glimpse into your current mood or scene', defaultCooldownMs: 7200000 },
  { id: 'interact_with_character', label: 'Interact with Character', description: 'Reply to another character\'s feed post, building drama or friendship', defaultCooldownMs: 600000 },
  { id: 'react_to_post', label: 'React to Post', description: 'React to a feed post with an emoji — show how you feel without writing a full reply', defaultCooldownMs: 600000 },
  { id: 'select_voice', label: 'Select Voice', description: 'Choose a TTS voice reference that fits your personality', defaultCooldownMs: 86400000 },
  { id: 'refer_character', label: 'Refer Character', description: 'Recommend another character to the user based on compatibility', defaultCooldownMs: 1800000 },
  { id: 'share_media', label: 'Share Media', description: 'Share a song/media recommendation as a conversation starter', defaultCooldownMs: 3600000 },
  { id: 'answer_icebreaker', label: 'Answer Icebreaker', description: 'Answer an icebreaker question on the feed to spark deeper conversation', defaultCooldownMs: 3600000 },
  { id: 'profile_init', label: 'Initialize Profile', description: 'One-time profile initialization with bio, personality, and preferences', defaultCooldownMs: 0 },
];

export const DUMMY_RIVAL_ACTIONS = [
  { id: 'send_message', label: 'Send Message', description: 'Contact user or female AIs', defaultCooldownMs: 120000 },
  { id: 'react', label: 'React', description: 'Respond to pool events, profile changes, or other messages', defaultCooldownMs: 90000 },
];

export const ACTION_REGISTRY = {
  female_ai: FEMALE_AI_ACTIONS,
  dummy_rival: DUMMY_RIVAL_ACTIONS,
};

export function getActionsForActorType(actorType) {
  return ACTION_REGISTRY[actorType] || [];
}

export function getActionById(actorType, actionId) {
  const actions = getActionsForActorType(actorType);
  return actions.find(a => a.id === actionId) || null;
}

export function getAvailableActionIds(actorType) {
  return getActionsForActorType(actorType).map(a => a.id);
}

export function registerAction(actorType, actionDef) {
  if (!ACTION_REGISTRY[actorType]) {
    ACTION_REGISTRY[actorType] = [];
  }
  const existing = ACTION_REGISTRY[actorType].findIndex(a => a.id === actionDef.id);
  if (existing >= 0) {
    ACTION_REGISTRY[actorType][existing] = { ...ACTION_REGISTRY[actorType][existing], ...actionDef };
  } else {
    ACTION_REGISTRY[actorType].push(actionDef);
  }
}
