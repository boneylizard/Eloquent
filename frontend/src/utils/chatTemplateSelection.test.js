import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MODEL_DEFAULT_CHAT_TEMPLATE_ID,
  buildChatTemplateMessages,
  getChatTemplateRequestFields,
  getConversationChatTemplateId,
  normaliseChatTemplateId,
} from './chatTemplateSelection.js';

test('normalises missing and unknown template ids to model default', () => {
  assert.equal(normaliseChatTemplateId(), MODEL_DEFAULT_CHAT_TEMPLATE_ID);
  assert.equal(normaliseChatTemplateId('unknown'), MODEL_DEFAULT_CHAT_TEMPLATE_ID);
  assert.equal(normaliseChatTemplateId('chatml'), 'chatml');
  assert.equal(normaliseChatTemplateId('custom:template-1'), 'custom:template-1');
});

test('reads a template override from the active conversation', () => {
  const conversations = [{ id: 'one' }, { id: 'two', chatTemplateId: 'generic' }];
  assert.equal(getConversationChatTemplateId(conversations, 'one'), MODEL_DEFAULT_CHAT_TEMPLATE_ID);
  assert.equal(getConversationChatTemplateId(conversations, 'two'), 'generic');
});

test('builds neutral structured messages for backend template rendering', () => {
  assert.deepEqual(
    buildChatTemplateMessages([
      { role: 'user', content: 'Hello' },
      { role: 'bot', content: 'Hi' },
      { role: 'assistant', content: 'Again' },
      { role: 'image', content: 'skip' },
    ], 'Stay in character.'),
    [
      { role: 'system', content: 'Stay in character.' },
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi' },
      { role: 'assistant', content: 'Again' },
    ],
  );
});

test('does not send local template controls to hosted APIs', () => {
  assert.deepEqual(getChatTemplateRequestFields({
    conversations: [{ id: 'one', chatTemplateId: 'chatml' }],
    conversationId: 'one',
    history: [{ role: 'user', content: 'Hello' }],
    systemMessage: 'System',
    isApi: true,
  }), {});
});

test('falls back when a conversation points to a deleted custom template', () => {
  const fields = getChatTemplateRequestFields({
    conversations: [{ id: 'one', chatTemplateId: 'custom:deleted' }],
    conversationId: 'one',
    history: [{ role: 'user', content: 'Hello' }],
    customTemplates: {},
  });
  assert.equal(fields.chat_template_id, MODEL_DEFAULT_CHAT_TEMPLATE_ID);
});
