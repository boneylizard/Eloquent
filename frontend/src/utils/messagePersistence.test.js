import assert from 'node:assert/strict';
import test from 'node:test';

import { sanitizeMessageForStorage } from './messagePersistence.js';


test('persisted enhancement history keeps its current level valid', () => {
  const history = [
    '/static/generated_images/original.png',
    '/static/generated_images/enhanced-1.png',
    '/static/generated_images/enhanced-2.png',
    '/static/generated_images/enhanced-3.png',
    '/static/generated_images/enhanced-4.png',
  ];

  const stored = sanitizeMessageForStorage({
    id: 'image-1',
    imagePath: history[4],
    enhancement_history: history,
    current_enhancement_level: 4,
  });

  assert.deepEqual(stored.enhancement_history, history);
  assert.equal(stored.current_enhancement_level, 4);
  assert.equal(
    stored.enhancement_history[stored.current_enhancement_level - 1],
    history[3],
  );
});
