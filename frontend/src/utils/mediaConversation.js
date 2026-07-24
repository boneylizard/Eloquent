/**
 * Return the conversation that should own a chat media request.
 *
 * Callback-driven generation is used by non-chat surfaces such as the Chess
 * historian avatar picker, so it must never create a chat as a side effect.
 */
export function ensureMediaConversation({
  activeConversation,
  createNewConversation,
  onImageGenerated,
}) {
  if (typeof onImageGenerated === 'function') {
    return activeConversation || null;
  }

  if (activeConversation) {
    return activeConversation;
  }

  const conversation = createNewConversation({ forceEmpty: true });
  return conversation?.id || null;
}

/**
 * Append messages without duplicating an async completion that is retried or
 * replayed. Message ids are the durable identity used by chat persistence.
 */
export function appendUniqueConversationMessages(currentMessages, additions) {
  const current = Array.isArray(currentMessages) ? currentMessages : [];
  const incoming = Array.isArray(additions) ? additions.filter(Boolean) : [];
  if (incoming.length === 0) return current;

  const knownIds = new Set(current.map((message) => message?.id).filter(Boolean));
  const unique = incoming.filter((message) => {
    if (!message?.id) return true;
    if (knownIds.has(message.id)) return false;
    knownIds.add(message.id);
    return true;
  });

  return unique.length > 0 ? [...current, ...unique] : current;
}

/**
 * Update one durable message in its owning conversation. Returning the
 * original array when the message is absent keeps retries and deleted-chat
 * completions harmless.
 */
export function updateConversationMessageById(currentMessages, messageId, update) {
  const current = Array.isArray(currentMessages) ? currentMessages : [];
  if (!messageId) return current;

  let changed = false;
  const next = current.map((message) => {
    if (message?.id !== messageId) return message;
    const updated = typeof update === 'function'
      ? update(message)
      : { ...message, ...(update || {}) };
    if (!updated || updated === message) return message;
    changed = true;
    return updated;
  });

  return changed ? next : current;
}

/**
 * Apply an async media replacement only while the message still points to the
 * exact source sent to the backend. This makes replay idempotent and prevents a
 * slower, stale enhancement from overwriting a newer one.
 */
export function updateMediaMessageIfSourceMatches(message, expectedImagePath, update) {
  if (!message || message.imagePath !== expectedImagePath) return message;

  const updated = typeof update === 'function'
    ? update(message)
    : { ...message, ...(update || {}) };

  return updated || message;
}
