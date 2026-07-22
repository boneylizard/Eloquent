import {
  CONVERSATION_SHARD_PREFIX,
  loadConversationsFromStorage,
} from './conversationStorage';
import indexedDbStorage from './indexedDbStorage';

/**
 * Full-text search across all conversation shards in IndexedDB.
 * Returns matching conversations with message excerpts.
 */
export async function searchAllConversations(query, allConversations = []) {
  if (!query || typeof query !== 'string' || query.trim().length < 2) {
    return [];
  }

  const trimmed = query.trim().toLowerCase();

  // Get all shard keys from IndexedDB
  const shardKeys = await indexedDbStorage.getKeysByPrefix(CONVERSATION_SHARD_PREFIX);

  const results = [];
  const seen = new Set();

  for (const key of shardKeys) {
    const id = key.slice(CONVERSATION_SHARD_PREFIX.length);
    if (!id || seen.has(id)) continue;

    // Skip tombstoned/banned IDs
    const isBanned = typeof localStorage !== 'undefined' &&
      Object.keys(localStorage).some((k) => k.startsWith('Eloquent-ban-') && k.includes(id));
    if (isBanned) continue;

    seen.add(id);

    // Read the shard
    const raw = await indexedDbStorage.getItem(key);
    if (!raw) continue;

    let messages;
    try {
      messages = JSON.parse(raw);
    } catch {
      continue;
    }

    if (!Array.isArray(messages) || messages.length === 0) continue;

    // Search each message's content
    const excerpts = [];
    for (const msg of messages) {
      const content = typeof msg.content === 'string' ? msg.content : '';
      if (!content) continue;

      const lowerContent = content.toLowerCase();
      const matchIndex = lowerContent.indexOf(trimmed);
      if (matchIndex === -1) continue;

      // Extract a snippet around the match
      const snippetStart = Math.max(0, matchIndex - 40);
      const snippetEnd = Math.min(content.length, matchIndex + trimmed.length + 40);
      const snippet = (snippetStart > 0 ? '...' : '') +
        content.slice(snippetStart, snippetEnd) +
        (snippetEnd < content.length ? '...' : '');

      excerpts.push({
        messageId: msg.id,
        role: msg.role,
        snippet,
        matchIndex: matchIndex - snippetStart + (snippetStart > 0 ? 3 : 0),
      });
    }

    if (excerpts.length > 0) {
      // Find conversation name from the provided list or use fallback
      const convMeta = allConversations.find((c) => c.id === id);
      results.push({
        conversationId: id,
        conversationName: convMeta?.name || 'Untitled',
        created: convMeta?.created,
        matchCount: excerpts.length,
        excerpts,
      });
    }
  }

  // Sort by most matches first, then by most recent
  results.sort((a, b) => b.matchCount - a.matchCount);

  return results;
}
