/**
 * Install the fabricated Call Mode showcase (demo user, character, chat, memories).
 * Merges into existing browser storage — does not wipe unrelated profiles or chats.
 */

import * as indexedDbStorage from './indexedDbStorage';
import {
  CONVERSATIONS_INDEX_KEY,
  CONVERSATION_SHARD_PREFIX,
} from './conversationStorage';

const DEMO_FLAG_KEY = 'LiangLocal-demo-showcase-installed';

function parseJson(raw, fallback = null) {
  if (raw == null) return fallback;
  if (typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function upsertById(list, item, idKey = 'id') {
  const id = item?.[idKey];
  if (!id) return [...(list || []), item];
  const next = [...(list || [])];
  const idx = next.findIndex((row) => row?.[idKey] === id);
  if (idx >= 0) next[idx] = { ...next[idx], ...item };
  else next.push(item);
  return next;
}

async function readJsonKey(key, fallback) {
  const raw = await indexedDbStorage.getItem(key);
  return parseJson(raw, fallback);
}

async function writeJsonKey(key, value) {
  const str = JSON.stringify(value);
  await indexedDbStorage.setItem(key, str);
  try {
    localStorage.setItem(key, str);
  } catch (_) {
    /* ignore */
  }
}

async function mergeProfiles(demoProfiles) {
  const root = demoProfiles || {};
  const demoList = Array.isArray(root.profiles) ? root.profiles : [];
  const activeProfileId = root.activeProfileId || demoList[0]?.id || null;

  const existing = await readJsonKey('user-profiles', { profiles: [], activeProfileId: null });
  let profiles = Array.isArray(existing.profiles) ? [...existing.profiles] : [];
  for (const p of demoList) {
    profiles = upsertById(profiles, p);
  }
  const payload = {
    profiles,
    activeProfileId: activeProfileId || existing.activeProfileId,
  };
  await writeJsonKey('user-profiles', payload);
  return payload;
}

async function mergeCharacters(demoCharacters) {
  const demoList = Array.isArray(demoCharacters) ? demoCharacters : [];
  const existingRaw = await indexedDbStorage.getItem('llm-characters');
  let existing = parseJson(existingRaw, []);
  if (!Array.isArray(existing)) existing = [];
  let merged = [...existing];
  for (const c of demoList) {
    merged = upsertById(merged, c);
  }
  const str = JSON.stringify(merged);
  await indexedDbStorage.setItem('llm-characters', str);
  try {
    localStorage.setItem('llm-characters', str);
  } catch (_) {
    /* ignore */
  }
  return merged;
}

async function mergeConversationIndex(demoCatalog, conversationId) {
  const demoList = Array.isArray(demoCatalog) ? demoCatalog : [];
  const existingRaw = await indexedDbStorage.getItem(CONVERSATIONS_INDEX_KEY);
  let catalog = parseJson(existingRaw, []);
  if (!Array.isArray(catalog)) catalog = [];
  for (const entry of demoList) {
    catalog = upsertById(catalog, entry);
  }
  await writeJsonKey(CONVERSATIONS_INDEX_KEY, catalog);
  if (conversationId) {
    await writeJsonKey('Eloquent-active-conversation', conversationId);
  }
  return catalog;
}

async function writeConversationShard(conversationId, messages) {
  if (!conversationId || !Array.isArray(messages)) return;
  const key = `${CONVERSATION_SHARD_PREFIX}${conversationId}`;
  await writeJsonKey(key, messages);
}

async function mergeStoryTracker(demoTracker) {
  if (!demoTracker || typeof demoTracker !== 'object') return;
  await writeJsonKey('eloquent-story-tracker', demoTracker);
}

export async function fetchDemoShowcaseStatus(apiUrl) {
  const res = await fetch(`${apiUrl}/demo/showcase/status`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Status ${res.status}`);
  }
  return res.json();
}

export async function installDemoShowcase({
  apiUrl,
  setActiveProfile = true,
  reload = true,
} = {}) {
  if (!apiUrl) throw new Error('apiUrl is required');

  const packRes = await fetch(`${apiUrl}/demo/showcase/pack`);
  if (!packRes.ok) {
    throw new Error(`Failed to load demo pack (${packRes.status})`);
  }
  const packBody = await packRes.json();
  const pack = packBody?.pack || packBody;
  const ids = pack?.ids || {};
  const frontend = pack?.frontend || {};
  const conversationId = ids.conversationId || 'conv_demo_mira';

  const installRes = await fetch(`${apiUrl}/demo/showcase/install`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ set_active: setActiveProfile }),
  });
  if (!installRes.ok) {
    const text = await installRes.text();
    throw new Error(text || `Backend install failed (${installRes.status})`);
  }
  const installResult = await installRes.json();

  await mergeProfiles(frontend['user-profiles']);
  await mergeCharacters(frontend['llm-characters']);
  await mergeConversationIndex(frontend[CONVERSATIONS_INDEX_KEY], frontend['Eloquent-active-conversation']);
  const shardKey = `${CONVERSATION_SHARD_PREFIX}${conversationId}`;
  await writeConversationShard(conversationId, frontend[shardKey]);
  await mergeStoryTracker(frontend['eloquent-story-tracker']);

  try {
    localStorage.setItem(DEMO_FLAG_KEY, new Date().toISOString());
    localStorage.setItem('LiangLocal-prefer-local-profiles', '1');
  } catch (_) {
    /* ignore */
  }

  if (reload) {
    window.location.reload();
    return { ...installResult, reloaded: true };
  }
  return { ...installResult, reloaded: false };
}

export function isDemoShowcaseInstalledLocally() {
  try {
    return !!localStorage.getItem(DEMO_FLAG_KEY);
  } catch {
    return false;
  }
}
