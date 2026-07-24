import assert from 'node:assert/strict';
import test from 'node:test';

import {
  appendUniqueConversationMessages,
  ensureMediaConversation,
  updateConversationMessageById,
  updateMediaMessageIfSourceMatches,
} from './mediaConversation.js';


test('creates exactly one empty conversation for chat media without an active chat', () => {
  const calls = [];
  const conversationId = ensureMediaConversation({
    activeConversation: null,
    createNewConversation: (options) => {
      calls.push(options);
      return { id: 'created-chat' };
    },
  });

  assert.equal(conversationId, 'created-chat');
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], { forceEmpty: true });
});


test('reuses the active conversation without creating another one', () => {
  let createCalls = 0;
  const conversationId = ensureMediaConversation({
    activeConversation: 'active-chat',
    createNewConversation: () => {
      createCalls += 1;
      return { id: 'unexpected-chat' };
    },
  });

  assert.equal(conversationId, 'active-chat');
  assert.equal(createCalls, 0);
});


test('callback-only image generation never creates a conversation', () => {
  let createCalls = 0;
  const conversationId = ensureMediaConversation({
    activeConversation: null,
    createNewConversation: () => {
      createCalls += 1;
      return { id: 'unexpected-chat' };
    },
    onImageGenerated: () => {},
  });

  assert.equal(conversationId, null);
  assert.equal(createCalls, 0);
});


test('async media appends are idempotent by message id', () => {
  const existing = [{ id: 'before', content: 'before' }];
  const image = { id: 'image-1', type: 'image' };

  const once = appendUniqueConversationMessages(existing, [image]);
  const twice = appendUniqueConversationMessages(once, [image]);

  assert.deepEqual(once, [existing[0], image]);
  assert.equal(twice, once);
});


test('message updates affect only the requested durable id', () => {
  const existing = [
    { id: 'image-1', imagePath: '/before.png' },
    { id: 'other', imagePath: '/other.png' },
  ];

  const updated = updateConversationMessageById(existing, 'image-1', (message) => ({
    ...message,
    imagePath: '/after.png',
  }));

  assert.equal(updated[0].imagePath, '/after.png');
  assert.equal(updated[1], existing[1]);
  assert.equal(
    updateConversationMessageById(updated, 'missing', { imagePath: '/never.png' }),
    updated,
  );
});


test('media replacement is idempotent when a persisted mutation is replayed', () => {
  const original = { id: 'image-1', imagePath: '/original.png', enhancement_history: ['/original.png'] };
  const replace = (message) => ({
    ...message,
    imagePath: '/enhanced.png',
    enhancement_history: [...message.enhancement_history, '/enhanced.png'],
  });

  const updated = updateMediaMessageIfSourceMatches(original, '/original.png', replace);
  const replayed = updateMediaMessageIfSourceMatches(updated, '/original.png', replace);

  assert.equal(updated.imagePath, '/enhanced.png');
  assert.deepEqual(updated.enhancement_history, ['/original.png', '/enhanced.png']);
  assert.equal(replayed, updated);
});


test('a stale media response cannot overwrite a newer replacement', () => {
  const newer = {
    id: 'image-1',
    imagePath: '/manual-result.png',
    enhancement_history: ['/original.png', '/manual-result.png'],
  };

  const stale = updateMediaMessageIfSourceMatches(
    newer,
    '/original.png',
    (message) => ({
      ...message,
      imagePath: '/stale-auto-result.png',
      enhancement_history: [...message.enhancement_history, '/stale-auto-result.png'],
    }),
  );

  assert.equal(stale, newer);
  assert.equal(stale.imagePath, '/manual-result.png');
  assert.deepEqual(stale.enhancement_history, ['/original.png', '/manual-result.png']);
});
