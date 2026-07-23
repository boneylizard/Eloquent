import assert from 'node:assert/strict';
import test from 'node:test';

import { formatPrompt } from './chat_templates.js';

test('post-history instructions are placed after chat history and before the reply', () => {
  const prompt = formatPrompt(
    [
      { role: 'user', content: 'Open the archive.' },
      { role: 'bot', content: 'The lock gives way.' },
    ],
    'llama-3',
    'You are Mara.',
    'Never decide the user’s actions.',
  );

  const historyIndex = prompt.indexOf('The lock gives way.');
  const instructionsIndex = prompt.indexOf('[POST-HISTORY INSTRUCTIONS]');
  const replyIndex = prompt.lastIndexOf('Answer:');

  assert.ok(historyIndex >= 0);
  assert.ok(instructionsIndex > historyIndex);
  assert.ok(replyIndex > instructionsIndex);
});
