import * as indexedDbStorage from './indexedDbStorage.js';

export const CHARACTER_GROUPS_STORAGE_KEY = 'llm-character-groups';

export function normaliseCharacterGroup(group, fallbackId = '') {
  const characterIds = [...new Set(
    (Array.isArray(group?.characterIds) ? group.characterIds : [])
      .map((id) => String(id || '').trim())
      .filter(Boolean)
  )];

  return {
    id: String(group?.id || fallbackId).trim(),
    name: String(group?.name || '').trim(),
    characterIds,
    context: String(group?.context || '').trim(),
    created_at: String(group?.created_at || new Date().toISOString()),
    updated_at: String(group?.updated_at || group?.created_at || new Date().toISOString()),
  };
}

export function parseCharacterGroups(raw) {
  if (!raw) return [];

  try {
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (!Array.isArray(parsed)) return [];

    return parsed
      .map((group, index) => normaliseCharacterGroup(group, `group-${index + 1}`))
      .filter((group) => group.id && group.name);
  } catch {
    return [];
  }
}

export async function loadCharacterGroups() {
  const raw = await indexedDbStorage.getItem(CHARACTER_GROUPS_STORAGE_KEY);
  return parseCharacterGroups(raw);
}

export async function saveCharacterGroups(groups) {
  const normalised = (Array.isArray(groups) ? groups : [])
    .map((group, index) => normaliseCharacterGroup(group, `group-${index + 1}`))
    .filter((group) => group.id && group.name);

  await indexedDbStorage.setItem(
    CHARACTER_GROUPS_STORAGE_KEY,
    JSON.stringify(normalised)
  );
  return normalised;
}

export function createCharacterGroupId() {
  if (globalThis.crypto?.randomUUID) {
    return `group-${globalThis.crypto.randomUUID()}`;
  }
  return `group-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}
