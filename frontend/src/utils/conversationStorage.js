/**
 * Chat persistence — index + shards. Deleted chats are BANNED via tiny localStorage
 * keys first (sync, survives crash/IDB quota), then IndexedDB cleanup.
 */

import * as indexedDbStorage from './indexedDbStorage';
import { sanitizeMessagesForStorage } from './messagePersistence';
import { hydrateMessagesThinkFields } from './thinkStreamParser';

export const CONVERSATIONS_INDEX_KEY = 'Eloquent-conversations-index';
export const CONVERSATIONS_LEGACY_KEY = 'Eloquent-conversations';
export const CONVERSATIONS_STORAGE_VERSION_KEY = 'Eloquent-conversations-storage-v';
export const CONVERSATIONS_STORAGE_VERSION = '9';
export const CONVERSATIONS_DELETED_IDS_KEY = 'Eloquent-conversations-deleted-ids';
export const CONVERSATION_SHARD_PREFIX = 'Eloquent-conversation-';
export const CONVERSATION_BAN_LS_PREFIX = 'Eloquent-ban-';
export const OUTREACH_CONVERSATION_ID_PREFIX = 'outreach-conv-';

const LS_GHOST_KEYS = [
  'conversations',
  CONVERSATIONS_LEGACY_KEY,
  CONVERSATIONS_INDEX_KEY,
];

let writeChain = Promise.resolve();

function enqueueWrite(task) {
  const run = writeChain.then(() => task(), () => task());
  writeChain = run.catch((err) => {
    console.error('[conversationStorage] queued write failed:', err);
  });
  return run;
}

function shardKey(conversationId) {
  return `${CONVERSATION_SHARD_PREFIX}${conversationId}`;
}

function banLsKey(conversationId) {
  return `${CONVERSATION_BAN_LS_PREFIX}${conversationId}`;
}

function parseJson(raw, fallback) {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

/** Sidebar index may be a bare array or legacy `{ conversations: [...] }`. */
function parseCatalogIndex(raw) {
  const parsed = parseJson(raw, []);
  if (Array.isArray(parsed)) return parsed.filter((c) => c?.id);
  if (parsed && typeof parsed === 'object') {
    if (Array.isArray(parsed.conversations)) return parsed.conversations.filter((c) => c?.id);
    if (Array.isArray(parsed.catalog)) return parsed.catalog.filter((c) => c?.id);
  }
  return [];
}

/** Shards may be a message array, `{ messages }`, or a single message object. */
function parseMessagesFromShard(raw) {
  const parsed = parseJson(raw, null);
  if (Array.isArray(parsed)) return parsed.length > 0 ? parsed : [];
  if (parsed && typeof parsed === 'object') {
    if (Array.isArray(parsed.messages) && parsed.messages.length > 0) return parsed.messages;
    if (parsed.role && (parsed.content != null || parsed.tool_calls)) return [parsed];
  }
  return [];
}

function getLocalStorageShardKeys() {
  const keys = [];
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (key?.startsWith(CONVERSATION_SHARD_PREFIX)) keys.push(key);
    }
  } catch (_) { /* noop */ }
  return keys;
}

/** IndexedDB + orphaned localStorage copies (migration skipped shards on boot). */
async function listAllConversationShardKeys() {
  const idbKeys = await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX);
  const lsKeys = getLocalStorageShardKeys();
  return [...new Set([...idbKeys, ...lsKeys])];
}

/** Key count only — does not read message bodies (safe for boot / settings). */
export async function countConversationShardKeys() {
  const keys = await listAllConversationShardKeys();
  return keys.length;
}

async function readShardRaw(conversationId) {
  const key = shardKey(conversationId);
  let raw = await indexedDbStorage.getItem(key);
  if (raw == null) {
    try {
      raw = localStorage.getItem(key);
    } catch (_) { /* noop */ }
  }
  return raw;
}

async function readShardMessages(conversationId) {
  return parseMessagesFromShard(await readShardRaw(conversationId));
}

/** Copy message shards still in localStorage into IndexedDB (skipped by migrateFromLocalStorage). */
async function migrateLocalStorageShardsToIndexedDb() {
  const lsKeys = getLocalStorageShardKeys();
  let migrated = 0;
  for (const key of lsKeys) {
    const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
    if (!id) continue;
    const messages = parseMessagesFromShard(localStorage.getItem(key));
    if (messages.length === 0) continue;
    const payload = JSON.stringify(sanitizeMessagesForStorage(messages));
    try {
      await indexedDbStorage.setItem(key, payload);
      migrated += 1;
      try { localStorage.removeItem(key); } catch (_) {}
    } catch (e) {
      console.warn('[conversationStorage] Could not migrate shard to IndexedDB:', key, e);
    }
  }
  if (migrated > 0) {
    console.info(`[conversationStorage] Migrated ${migrated} shard(s) from localStorage to IndexedDB`);
  }
  return migrated;
}

function catalogEntryFromMessages(id, messages) {
  let name = 'Recovered Chat';
  const firstUser = messages.find(
    (m) => m?.role === 'user' && typeof m.content === 'string' && m.content.trim()
  );
  if (firstUser?.content) {
    const t = firstUser.content.trim();
    name = t.slice(0, 48) + (t.length > 48 ? '…' : '');
  }
  return {
    id,
    name,
    created: new Date().toISOString(),
    messageCount: messages.length,
  };
}

function stripMessages(conv) {
  if (!conv || typeof conv !== 'object') return conv;
  const messageCount = Array.isArray(conv.messages) ? conv.messages.length : 0;
  const { messages, ...meta } = conv;
  return { ...meta, messageCount };
}

/** Collect banned conversation ids — localStorage bans are checked first (crash-safe). */
export function getBannedConversationIdsSync() {
  const ids = new Set();
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(CONVERSATION_BAN_LS_PREFIX)) {
        const id = key.slice(CONVERSATION_BAN_LS_PREFIX.length);
        if (id) ids.add(id);
      }
    }
  } catch (_) { /* noop */ }
  return ids;
}

/**
 * SYNCHRONOUS — call before any async work when user deletes a tab.
 * Survives browser crash even if IndexedDB is full or the write queue never runs.
 */
export function banConversationIdSync(conversationId) {
  if (!conversationId) return;
  try {
    localStorage.setItem(banLsKey(conversationId), String(Date.now()));
  } catch (e) {
    console.error('[conversationStorage] Could not write ban key:', conversationId, e);
  }
}

export function scrubConversationLocalStorageGhosts() {
  for (const key of LS_GHOST_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch (_) { /* noop */ }
  }
  try {
    localStorage.removeItem('Eloquent-active-conversation');
  } catch (_) { /* noop */ }
}

async function getDeletedIdSet(idbOpts = {}) {
  const ids = getBannedConversationIdsSync();

  const raw = await indexedDbStorage.getItem(CONVERSATIONS_DELETED_IDS_KEY, idbOpts);
  const arr = parseJson(raw, []);
  if (Array.isArray(arr)) arr.filter(Boolean).forEach((id) => ids.add(id));

  return ids;
}

async function persistDeletedIdSet(ids) {
  const payload = JSON.stringify([...ids]);
  try {
    await indexedDbStorage.setItem(CONVERSATIONS_DELETED_IDS_KEY, payload);
  } catch (e) {
    console.warn('[conversationStorage] IDB tombstone list failed (ban keys still active):', e);
  }
}

async function addTombstone(conversationId) {
  if (!conversationId) return;
  banConversationIdSync(conversationId);
  const ids = await getDeletedIdSet();
  if (ids.has(conversationId)) return;
  ids.add(conversationId);
  await persistDeletedIdSet(ids);
}

function filterTombstones(conversations, deletedIds) {
  return (conversations || []).filter((c) => c?.id && !deletedIds.has(c.id));
}

async function writeLegacyMarker(ids) {
  const marker = JSON.stringify({ v: 6, ids: ids || [], savedAt: new Date().toISOString() });
  try {
    await indexedDbStorage.setItem(CONVERSATIONS_LEGACY_KEY, marker);
  } catch (_) { /* noop */ }
}

async function writeCatalogIndex(catalog, { allowEmpty = false } = {}) {
  if (!allowEmpty && (!Array.isArray(catalog) || catalog.length === 0)) {
    console.warn('[conversationStorage] Refused to write empty catalog (would wipe tab list)');
    return false;
  }
  const payload = JSON.stringify(catalog);
  try {
    await indexedDbStorage.setItem(CONVERSATIONS_INDEX_KEY, payload);
    scrubConversationLocalStorageGhosts();
    const verify = await indexedDbStorage.getItem(CONVERSATIONS_INDEX_KEY);
    if (verify !== payload) {
      throw new Error('[conversationStorage] Catalog index write verification failed');
    }
    await writeLegacyMarker(catalog.map((c) => c.id));
    return true;
  } catch (e) {
    console.error('[conversationStorage] Catalog write failed:', e);
    throw e;
  }
}

async function writeMessageShard(conversationId, messages, { deleteIfEmpty = false } = {}) {
  if (!conversationId) return;
  const banned = await getDeletedIdSet();
  if (banned.has(conversationId)) {
    await indexedDbStorage.removeItem(shardKey(conversationId));
    return;
  }
  const key = shardKey(conversationId);
  if (!Array.isArray(messages) || messages.length === 0) {
    // Never wipe a shard on empty payload unless explicitly deleting the chat.
    if (deleteIfEmpty) {
      await indexedDbStorage.removeItem(key);
    }
    return;
  }
  const slim = sanitizeMessagesForStorage(messages);
  await indexedDbStorage.setItem(key, JSON.stringify(slim));
}

/** Sidebar index entry missing but messages exist — restore tab metadata. */
async function ensureConversationInCatalog(conversationId, meta, deletedIds) {
  if (!conversationId || deletedIds.has(conversationId)) return false;
  const { catalog } = await readCatalogFromDisk();
  if (catalog.some((c) => c.id === conversationId)) return false;
  const entry = stripMessages({
    id: conversationId,
    name: meta?.name || 'Recovered Chat',
    created: meta?.created || new Date().toISOString(),
    messageCount: meta?.messageCount ?? 0,
    ...meta,
  });
  const next = [...catalog, entry];
  await writeCatalogIndex(next);
  console.warn(
    `[conversationStorage] Catalog was missing tab "${conversationId}" — restored from message save / shard`
  );
  return true;
}

async function readCatalogFromDisk(idbOpts = {}) {
  const deletedIds = await getDeletedIdSet(idbOpts);
  const index = parseCatalogIndex(await indexedDbStorage.getItem(CONVERSATIONS_INDEX_KEY, idbOpts));

  const filtered = index
    .filter((c) => c?.id && !deletedIds.has(c.id))
    .map(stripMessages);

  if (filtered.length !== index.length) {
    if (filtered.length > 0) {
      try {
        await writeCatalogIndex(filtered);
      } catch (e) {
        console.warn('[conversationStorage] Could not rewrite filtered index:', e);
      }
    } else if (index.length > 0) {
      // Do not overwrite index with [] when every tab is tombstoned — may be stale ban keys.
      console.warn(
        `[conversationStorage] Index has ${index.length} tab(s) but all are banned; keeping index on disk`
      );
    }
  }
  return { catalog: filtered, deletedIds, rawIndexCount: index.length };
}

/** Add sidebar tabs for message shards that exist but are missing from the index. */
async function mergeCatalogWithShards(catalog, deletedIds) {
  const byId = new Map((catalog || []).filter((c) => c?.id).map((c) => [c.id, c]));
  const keys = await listAllConversationShardKeys();
  let added = 0;

  for (const key of keys) {
    const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
    if (!id || deletedIds.has(id) || byId.has(id)) continue;

    const messages = await readShardMessages(id);
    if (messages.length === 0) continue;

    byId.set(id, catalogEntryFromMessages(id, messages));
    added += 1;
  }

  const merged = [...byId.values()];
  if (added > 0) {
    try {
      await writeCatalogIndex(merged);
      console.info(`[conversationStorage] Merged ${added} tab(s) from message shards into index (${merged.length} total)`);
    } catch (e) {
      console.error('[conversationStorage] Could not save merged catalog:', e);
    }
  }
  return merged;
}

async function destroyLegacyMonolithicBlobs(catalogIds, { removeImportedArrays = false } = {}) {
  if (removeImportedArrays) {
    for (const key of [CONVERSATIONS_LEGACY_KEY, 'conversations']) {
      const raw = await indexedDbStorage.getItem(key);
      const parsed = parseJson(raw, null);
      if (Array.isArray(parsed) && parsed.length > 0) {
        await indexedDbStorage.removeItem(key);
      }
    }
  } else {
    for (const key of [CONVERSATIONS_LEGACY_KEY, 'conversations']) {
      const raw = await indexedDbStorage.getItem(key);
      const parsed = parseJson(raw, null);
      if (Array.isArray(parsed) && parsed.length > 0) {
        console.warn(
          `[conversationStorage] Legacy blob "${key}" still holds ${parsed.length} chat(s) — not deleting until imported`
        );
      }
    }
  }
  await writeLegacyMarker(catalogIds);
  try {
    localStorage.removeItem('conversations');
  } catch (_) { /* noop */ }
}

/** Import old monolithic Eloquent-conversations array into index + shards before any cleanup. */
async function importLegacyMonolithicConversations(deletedIds) {
  const byId = new Map();
  let changed = false;

  for (const key of [CONVERSATIONS_LEGACY_KEY, 'conversations']) {
    const raw = await indexedDbStorage.getItem(key);
    const parsed = parseJson(raw, null);
    if (!Array.isArray(parsed) || parsed.length === 0) continue;

    for (const conv of parsed) {
      if (!conv?.id || deletedIds.has(conv.id)) continue;
      if (Array.isArray(conv.messages) && conv.messages.length > 0) {
        await writeMessageShard(conv.id, conv.messages);
      }
      byId.set(conv.id, stripMessages(conv));
      changed = true;
    }
  }

  if (!changed) return [];

  const catalog = [...byId.values()];
  try {
    await writeCatalogIndex(catalog);
    console.info(`[conversationStorage] Imported ${catalog.length} chat(s) from legacy monolithic storage`);
    await destroyLegacyMonolithicBlobs(catalog.map((c) => c.id), { removeImportedArrays: true });
  } catch (e) {
    console.error('[conversationStorage] Legacy import failed:', e);
  }
  return catalog;
}

async function purgeBannedShardsOnly(deletedIds) {
  if (!deletedIds?.size) return;
  const keys = await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX);

  await Promise.all(
    keys.map(async (key) => {
      const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
      if (id && deletedIds.has(id)) {
        await indexedDbStorage.removeItem(key);
        await indexedDbStorage.removeItem(`LiangLocal-variants-${id}`);
      }
    })
  );
}

/** Only when user explicitly repairs — never run on normal boot (was deleting all chats after crash). */
async function purgeOrphanShardsNotInCatalog(deletedIds, catalog) {
  const catalogIds = new Set((catalog || []).map((c) => c.id));
  const keys = await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX);
  await Promise.all(
    keys.map(async (key) => {
      const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
      if (!id || deletedIds.has(id) || !catalogIds.has(id)) {
        await indexedDbStorage.removeItem(key);
        await indexedDbStorage.removeItem(`LiangLocal-variants-${id}`);
      }
    })
  );
}

/** Rebuild tab list from message shards when index was lost (e.g. browser crash). */
async function recoverCatalogFromShards(deletedIds) {
  const keys = await listAllConversationShardKeys();
  const catalog = [];
  let skippedBanned = 0;
  let skippedEmpty = 0;

  for (const key of keys) {
    const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
    if (!id || deletedIds.has(id)) {
      if (id && deletedIds.has(id)) skippedBanned += 1;
      continue;
    }
    const messages = await readShardMessages(id);
    if (messages.length === 0) {
      skippedEmpty += 1;
      continue;
    }
    catalog.push(catalogEntryFromMessages(id, messages));
  }

  console.info(
    `[conversationStorage] recoverCatalogFromShards: ${keys.length} key(s), ` +
    `${catalog.length} recoverable, ${skippedBanned} banned, ${skippedEmpty} empty/unparseable`
  );

  if (catalog.length > 0) {
    try {
      await writeCatalogIndex(catalog);
      console.info(`[conversationStorage] Recovered ${catalog.length} chat(s) from message shards`);
    } catch (e) {
      console.error('[conversationStorage] Could not save recovered catalog:', e);
      throw e;
    }
  }
  return catalog;
}

async function purgeAllKeysForConversation(conversationId) {
  if (!conversationId) return;
  banConversationIdSync(conversationId);
  await indexedDbStorage.removeItem(shardKey(conversationId));
  await indexedDbStorage.removeItem(`LiangLocal-variants-${conversationId}`);
}

async function repairConversationStorageOnBootInner() {
  scrubConversationLocalStorageGhosts();
  await migrateLocalStorageShardsToIndexedDb();
  const deletedIds = await getDeletedIdSet();

  let { catalog } = await readCatalogFromDisk();
  if (catalog.length === 0) {
    const legacy = await importLegacyMonolithicConversations(deletedIds);
    if (legacy.length > 0) catalog = legacy;
  }
  await purgeBannedShardsOnly(deletedIds);

  ({ catalog } = await readCatalogFromDisk());
  if (catalog.length === 0) {
    catalog = await mergeCatalogWithShards([], deletedIds);
  }
  if (catalog.length === 0) {
    catalog = await recoverCatalogFromShards(deletedIds);
  }

  await destroyLegacyMonolithicBlobs(catalog.map((c) => c.id));
  await indexedDbStorage.setItem(CONVERSATIONS_STORAGE_VERSION_KEY, CONVERSATIONS_STORAGE_VERSION);
  return catalog;
}

export function repairConversationStorageOnBoot() {
  return enqueueWrite(() => repairConversationStorageOnBootInner());
}

/**
 * @param {{ skipShardScan?: boolean }} [options]
 * skipShardScan: read catalog/index only; defer per-shard rebuild to emergency recover (fast boot / settings).
 */
export async function loadConversationsFromStorage(options = {}) {
  const { skipShardScan = false, idbOpts = {} } = options;
  return enqueueWrite(async () => {
    scrubConversationLocalStorageGhosts();
    if (!skipShardScan) {
      await migrateLocalStorageShardsToIndexedDb();
    }
    const deletedIds = await getDeletedIdSet(idbOpts);

    let { catalog } = await readCatalogFromDisk(idbOpts);

    if (catalog.length === 0) {
      const legacy = await importLegacyMonolithicConversations(deletedIds);
      if (legacy.length > 0) catalog = legacy;
    }

    if (!skipShardScan) {
      catalog = await mergeCatalogWithShards(catalog, deletedIds);
      if (catalog.length === 0) {
        catalog = await recoverCatalogFromShards(deletedIds);
      }
    }

    await destroyLegacyMonolithicBlobs(catalog.map((c) => c.id));
    await indexedDbStorage.setItem(CONVERSATIONS_STORAGE_VERSION_KEY, CONVERSATIONS_STORAGE_VERSION);

    if (catalog.length === 0) {
      if (skipShardScan) {
        console.info(
          '[Eloquent] skipping emergency recover (deferred) — catalog empty on fast boot; ' +
          'run await window.eloquentChatStorage.emergencyRecover() if chat tabs are missing'
        );
      } else {
        const shardKeys = await listAllConversationShardKeys();
        const banCount = deletedIds.size;
        if (shardKeys.length > 0) {
          console.error(
            `[conversationStorage] ${shardKeys.length} message shard(s) on disk but 0 tabs loaded` +
            (banCount ? ` (${banCount} ban/tombstone id(s) — try window.eloquentChatStorage.emergencyRecover())` : '')
          );
        }
      }
      return [];
    }

    return catalog
      .filter((meta) => meta?.id && !deletedIds.has(meta.id))
      .map((meta) => ({
        ...stripMessages(meta),
        messages: [],
        agenticMemoryEnabled: true,
      }));
  });
}

/** Load messages for one tab only (avoids loading every chat into RAM at once). */
export async function loadConversationMessages(conversationId) {
  if (!conversationId) return [];
  const deletedIds = await getDeletedIdSet();
  if (deletedIds.has(conversationId)) return [];
  const shard = await readShardMessages(conversationId);
  return hydrateMessagesThinkFields(shard);
}

/** Save only the active tab's messages; ensures catalog row exists when saving non-empty shards. */
export async function saveActiveConversationMessages(conversationId, messages, catalogMeta = null) {
  if (!conversationId) return false;
  if (getBannedConversationIdsSync().has(conversationId)) return false;
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    if (deletedIds.has(conversationId)) return false;
    if (Array.isArray(messages) && messages.length > 0) {
      await writeMessageShard(conversationId, messages);
      await ensureConversationInCatalog(
        conversationId,
        { ...catalogMeta, messageCount: messages.length },
        deletedIds
      );
    }
    return true;
  });
}

export function isOutreachConversationId(conversationId) {
  return typeof conversationId === 'string' && conversationId.startsWith(OUTREACH_CONVERSATION_ID_PREFIX);
}

export function isOutreachConversation(conv) {
  return isOutreachConversationId(conv?.id);
}

/** Remove outreach sidebar tabs/shards left from bulk import (does not tombstone). */
export async function purgeOutreachConversationsFromStorage() {
  return enqueueWrite(async () => {
    const { catalog } = await readCatalogFromDisk();
    const outreachIds = new Set(
      catalog.filter((c) => isOutreachConversationId(c.id)).map((c) => c.id)
    );

    const keys = await listAllConversationShardKeys();
    for (const key of keys) {
      const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
      if (isOutreachConversationId(id)) outreachIds.add(id);
    }

    if (outreachIds.size === 0) return { purged: 0 };

    const filtered = catalog.filter((c) => !isOutreachConversationId(c.id));
    if (filtered.length !== catalog.length) {
      await writeCatalogIndex(filtered, { allowEmpty: filtered.length === 0 });
    }

    await Promise.all(
      [...outreachIds].map(async (id) => {
        await indexedDbStorage.removeItem(shardKey(id));
        await indexedDbStorage.removeItem(`LiangLocal-variants-${id}`);
      })
    );

    return { purged: outreachIds.size };
  });
}

/** Persist outreach conversation from server (catalog row + message shard). */
export async function persistOutreachConversation(conv) {
  if (!conv?.id) return false;
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    if (deletedIds.has(conv.id)) return false;
    const messages = Array.isArray(conv.messages) ? conv.messages : [];
    const meta = stripMessages({ ...conv, messageCount: messages.length });
    let { catalog } = await readCatalogFromDisk();
    const idx = catalog.findIndex((c) => c.id === conv.id);
    if (idx >= 0) {
      catalog[idx] = { ...catalog[idx], ...meta };
    } else {
      catalog = [...catalog, meta];
    }
    await writeCatalogIndex(catalog);
    if (messages.length > 0) {
      await writeMessageShard(conv.id, messages);
    }
    return true;
  });
}

export async function saveConversationCatalog(conversations, activeId = null, opts = {}) {
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    const inputIds = (conversations || []).map((c) => c?.id).filter(Boolean);
    const catalog = filterTombstones(conversations, deletedIds).map(stripMessages);
    const catalogIdSet = new Set(catalog.map((c) => c.id));
    const dropped = inputIds.filter((id) => !catalogIdSet.has(id));
    if (dropped.length > 0) {
      console.warn('[conversationStorage] Tab(s) omitted from catalog save (ban/tombstone):', dropped);
    }
    if (activeId && !catalogIdSet.has(activeId) && !deletedIds.has(activeId)) {
      console.error(
        '[conversationStorage] Active tab missing from catalog save — sidebar may lose this chat on reload:',
        activeId
      );
    }
    const wrote = await writeCatalogIndex(catalog, { allowEmpty: opts.allowEmpty === true });
    if (!wrote) return false;
    await purgeBannedShardsOnly(deletedIds);
    if (activeId && !deletedIds.has(activeId)) {
      try {
        await indexedDbStorage.setItem('Eloquent-active-conversation', activeId);
      } catch (_) { /* noop */ }
    }
    return true;
  });
}

export async function persistChatState(conversations, activeId = null, activeMessages = null, opts = {}) {
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    const list = filterTombstones(conversations, deletedIds);
    const catalog = list.map(stripMessages);
    const wrote = await writeCatalogIndex(catalog);
    if (!wrote && catalog.length > 0) return false;

    const { flushConversationId, flushMessages } = opts;
    if (
      flushConversationId
      && !deletedIds.has(flushConversationId)
      && Array.isArray(flushMessages)
      && flushMessages.length > 0
    ) {
      await writeMessageShard(flushConversationId, flushMessages);
    }
    if (
      activeId
      && !deletedIds.has(activeId)
      && Array.isArray(activeMessages)
      && activeMessages.length > 0
    ) {
      await writeMessageShard(activeId, activeMessages);
    }
    if (activeId && !deletedIds.has(activeId)) {
      try {
        await indexedDbStorage.setItem('Eloquent-active-conversation', activeId);
      } catch (_) { /* noop */ }
    }
    await purgeBannedShardsOnly(deletedIds);
    return true;
  });
}

export async function deleteConversationFromStorage(conversationId) {
  if (!conversationId) return false;
  banConversationIdSync(conversationId);

  return enqueueWrite(async () => {
    await addTombstone(conversationId);
    const deletedIds = await getDeletedIdSet();
    const { catalog } = await readCatalogFromDisk();
    const filtered = catalog.filter((c) => c.id !== conversationId);
    try {
      await writeCatalogIndex(filtered, { allowEmpty: true });
    } catch (e) {
      console.warn('[conversationStorage] Index update on delete failed; ban keys still apply:', e);
    }
    await purgeAllKeysForConversation(conversationId);
    await destroyLegacyMonolithicBlobs(filtered.map((c) => c.id));
    await purgeBannedShardsOnly(deletedIds);
    return true;
  });
}

export async function loadTombstonedConversationIds() {
  return [...(await getDeletedIdSet())];
}

export async function deleteAllConversationsFromStorage() {
  return enqueueWrite(async () => {
    const shardKeys = await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX);
    await Promise.all(shardKeys.map((k) => indexedDbStorage.removeItem(k)));
    const variantKeys = await indexedDbStorage.getKeysByPrefix('LiangLocal-variants-');
    await Promise.all(variantKeys.map((k) => indexedDbStorage.removeItem(k)));
    await indexedDbStorage.removeItem(CONVERSATIONS_INDEX_KEY);
    await indexedDbStorage.removeItem(CONVERSATIONS_LEGACY_KEY);
    await indexedDbStorage.removeItem(CONVERSATIONS_DELETED_IDS_KEY);
    await indexedDbStorage.removeItem('Eloquent-active-conversation');
    await indexedDbStorage.setItem(CONVERSATIONS_STORAGE_VERSION_KEY, CONVERSATIONS_STORAGE_VERSION);
    scrubConversationLocalStorageGhosts();
    try {
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const key = localStorage.key(i);
        if (key?.startsWith(CONVERSATION_BAN_LS_PREFIX)) {
          localStorage.removeItem(key);
        }
      }
    } catch (_) { /* noop */ }
  });
}

/** Purge orphan shards only (manual). Does NOT delete chats in the index. */
export async function repairAndPurgeGhostChats() {
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    const { catalog } = await readCatalogFromDisk();
    if (!catalog.length) {
      console.warn(
        '[conversationStorage] Repair skipped: catalog is empty — run Emergency recover or Recover from shards first'
      );
      return { purged: 0, skipped: true, reason: 'empty-catalog' };
    }
    await purgeOrphanShardsNotInCatalog(deletedIds, catalog);
    await destroyLegacyMonolithicBlobs(catalog.map((c) => c.id));
    return { purged: true, skipped: false };
  });
}

/** Force-write sidebar index from every non-banned shard (manual / debug). */
export async function ensureCatalogIncludesAllShards() {
  return enqueueWrite(async () => {
    const migrated = await migrateLocalStorageShardsToIndexedDb();
    const deletedIds = await getDeletedIdSet();
    let { catalog } = await readCatalogFromDisk();
    const before = catalog.length;
    catalog = await mergeCatalogWithShards(catalog, deletedIds);
    if (catalog.length === 0) {
      catalog = await recoverCatalogFromShards(deletedIds);
    }
    const added = catalog.length - before;
    console.info(
      `[conversationStorage] ensureCatalogIncludesAllShards: ${catalog.length} tab(s)` +
      (added > 0 ? ` (+${added} from shards)` : '') +
      (migrated ? `, migrated ${migrated} localStorage shard(s)` : '')
    );
    return { catalogCount: catalog.length, added, migrated };
  });
}

/** Last-resort: clear ban keys, import legacy blob, rebuild index from shards. Never touches settings keys. */
export async function emergencyRecoverAllConversations() {
  return enqueueWrite(async () => {
    console.info('[conversationStorage] Emergency recover starting (conversation keys only)…');
    const clearedBans = clearAllConversationBanKeysSync();
    try {
      await indexedDbStorage.removeItem(CONVERSATIONS_DELETED_IDS_KEY);
    } catch (_) { /* noop */ }

    const migrated = await migrateLocalStorageShardsToIndexedDb();
    const deletedIds = await getDeletedIdSet();
    const shardKeyCount = (await listAllConversationShardKeys()).length;
    console.info(
      `[conversationStorage] Emergency recover: cleared ${clearedBans} ban key(s), ` +
      `${shardKeyCount} shard key(s), migrated ${migrated} from localStorage`
    );

    let catalog = await importLegacyMonolithicConversations(deletedIds);
    if (!catalog.length) {
      ({ catalog } = await readCatalogFromDisk());
      console.info(`[conversationStorage] Emergency recover: ${catalog.length} tab(s) from existing index`);
    }
    catalog = await mergeCatalogWithShards(catalog, deletedIds);
    if (!catalog.length) {
      catalog = await recoverCatalogFromShards(deletedIds);
    }

    let catalogWritten = false;
    if (catalog.length > 0) {
      catalogWritten = await writeCatalogIndex(catalog);
      if (!catalogWritten) {
        console.error('[conversationStorage] Emergency recover: catalog write returned false');
      }
    } else {
      console.warn('[conversationStorage] Emergency recover: no recoverable chats found');
    }

    const result = {
      clearedBans,
      migrated,
      shardKeyCount,
      recovered: catalog.length,
      catalogWritten,
    };
    console.info('[conversationStorage] Emergency recover done:', result);
    return result;
  });
}

/** Try to restore sidebar tabs from surviving message shards. */
export async function recoverChatsFromShards() {
  return enqueueWrite(async () => {
    const migrated = await migrateLocalStorageShardsToIndexedDb();
    const deletedIds = await getDeletedIdSet();
    const banCount = deletedIds.size;
    if (banCount > 0) {
      console.warn(
        `[conversationStorage] recoverChatsFromShards: ${banCount} ban/tombstone id(s) still active — ` +
        'use Emergency recover to clear bans'
      );
    }
    const catalog = await recoverCatalogFromShards(deletedIds);
    console.info(
      `[conversationStorage] recoverChatsFromShards: ${catalog.length} tab(s)` +
      (migrated ? ` (migrated ${migrated} localStorage shard(s))` : '')
    );
    return catalog.length;
  });
}

/** Restore one tab in the index when its message shard exists but catalog row is missing. */
export async function recoverConversationCatalogEntry(conversationId) {
  if (!conversationId) return false;
  return enqueueWrite(async () => {
    const deletedIds = await getDeletedIdSet();
    if (deletedIds.has(conversationId)) return false;
    const { catalog } = await readCatalogFromDisk();
    if (catalog.some((c) => c.id === conversationId)) return false;
    const messages = await readShardMessages(conversationId);
    if (!Array.isArray(messages) || messages.length === 0) return false;
    await ensureConversationInCatalog(
      conversationId,
      catalogEntryFromMessages(conversationId, messages),
      deletedIds
    );
    return true;
  });
}

export function clearAllConversationBanKeysSync() {
  let count = 0;
  try {
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key?.startsWith(CONVERSATION_BAN_LS_PREFIX)) {
        localStorage.removeItem(key);
        count += 1;
      }
    }
  } catch (_) { /* noop */ }
  return count;
}

/** Ban + purge one tab (safe to call from console if sidebar delete failed). */
export async function permanentlyForgetConversation(conversationId) {
  if (!conversationId) return false;
  banConversationIdSync(conversationId);
  return deleteConversationFromStorage(conversationId);
}

/** DevTools helper — attach on window during hydration. */
export async function getChatStorageDebugInfo() {
  const banned = [...getBannedConversationIdsSync()];
  const { catalog, rawIndexCount } = await readCatalogFromDisk();
  const shardKeys = await listAllConversationShardKeys();
  const idbShardCount = (await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX)).length;
  const lsShardCount = getLocalStorageShardKeys().length;
  const orphans = [];
  for (const key of shardKeys) {
    const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
    if (!id || catalog.some((c) => c.id === id) || banned.includes(id)) continue;
    const msgCount = (await readShardMessages(id)).length;
    if (msgCount > 0) orphans.push({ id, messageCount: msgCount });
  }
  return {
    storageVersion: CONVERSATIONS_STORAGE_VERSION,
    bannedIds: banned,
    catalogIds: catalog.map((c) => ({ id: c.id, name: c.name, messageCount: c.messageCount })),
    rawIndexCount,
    orphanShardIds: orphans.map((o) => o.id),
    orphansWithMessages: orphans,
    shardCount: shardKeys.length,
    idbShardCount,
    localStorageShardCount: lsShardCount,
  };
}

export function installChatStorageDebugHelpers() {
  if (typeof window === 'undefined') return;
  if (window.eloquentChatStorage?.version === CONVERSATIONS_STORAGE_VERSION) return;
  window.eloquentChatStorage = {
    version: CONVERSATIONS_STORAGE_VERSION,
    forget: (id) => permanentlyForgetConversation(id),
    recover: () => recoverChatsFromShards().then((n) => {
      console.info(`[eloquentChatStorage] Recovered ${n} chat(s). Reload the page.`);
      if (n > 0) window.location.reload();
      return n;
    }),
    ensureCatalog: () => ensureCatalogIncludesAllShards().then((r) => {
      console.info('[eloquentChatStorage] ensureCatalog:', r);
      if (r.catalogCount > 0) window.location.reload();
      return r;
    }),
    listBans: () => [...getBannedConversationIdsSync()],
    clearBans: () => {
      const n = clearAllConversationBanKeysSync();
      console.info(`[eloquentChatStorage] Cleared ${n} ban key(s). Reload the page.`);
      return n;
    },
    emergencyRecover: () => emergencyRecoverAllConversations().then(async (r) => {
      console.info('[eloquentChatStorage] Emergency recover:', r);
      if (r.recovered > 0 && r.catalogWritten) window.location.reload();
      else if (r.recovered > 0 && !r.catalogWritten) {
        console.error('[eloquentChatStorage] Tabs recovered in memory but catalog was not written — retry or check IndexedDB quota');
      }
      return r;
    }),
    debug: () => getChatStorageDebugInfo().then((d) => {
      console.table(d.catalogIds);
      if (d.orphansWithMessages?.length) console.table(d.orphansWithMessages);
      console.log('[eloquentChatStorage]', d);
      return d;
    }),
  };
}

if (typeof window !== 'undefined') {
  installChatStorageDebugHelpers();
}
