import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildCharacterGreetingOptions,
  createCharacterGreetingState,
  cycleCharacterGreetingMessage,
  resolveCharacterLoreEntries,
  resolveCharacterPostHistoryInstructions,
  resolveCharacterPromptOverride,
} from './characterCardRuntime.js';

test('character greetings include the primary and alternate openings with tag replacement', () => {
  assert.deepEqual(
    buildCharacterGreetingOptions({
      name: 'Mara',
      first_message: 'Hello, {{user}}.',
      alternate_greetings: ['{{char}} closes the archive.', 'Hello, {{user}}.'],
    }, 'Alex'),
    ['Hello, Alex.', 'Mara closes the archive.'],
  );
});

test('greeting messages cycle in both directions', () => {
  const state = createCharacterGreetingState(['First', 'Second', 'Third']);
  const message = { id: 'greeting', content: 'First', characterGreeting: state };
  const previous = cycleCharacterGreetingMessage(message, 'previous');
  assert.equal(previous.content, 'Third');
  assert.equal(previous.characterGreeting.index, 2);
  const next = cycleCharacterGreetingMessage(previous, 'next');
  assert.equal(next.content, 'First');
  assert.equal(next.characterGreeting.index, 0);
});

test('character prompt overrides expand the original prompt and card tags', () => {
  assert.equal(
    resolveCharacterPromptOverride(
      'Keep {{char}} terse.\n{{original}}\nAddress {{user}} directly.',
      'Stay in character.',
      'Mara',
      'Alex',
    ),
    'Keep Mara terse.\nStay in character.\nAddress Alex directly.',
  );
});

test('post-history instructions are resolved from the card', () => {
  assert.equal(
    resolveCharacterPostHistoryInstructions({
      name: 'Mara',
      post_history_instructions: 'Never decide {{user}}’s actions.',
    }, 'Alex'),
    'Never decide Alex’s actions.',
  );
});

test('character lore honours disabled, constant, selective and case-sensitive entries', () => {
  const character = {
    loreEntries: [
      {
        content: 'Always present.',
        keywords: [],
        tavern_entry: { constant: true, insertion_order: 20 },
      },
      {
        content: 'Disabled.',
        keywords: ['archive'],
        tavern_entry: { enabled: false },
      },
      {
        content: 'Selective match.',
        keywords: ['archive'],
        tavern_entry: {
          selective: true,
          secondary_keys: ['midnight'],
          insertion_order: 10,
        },
      },
      {
        content: 'Exact case only.',
        keywords: ['Mara'],
        tavern_entry: { case_sensitive: true },
      },
    ],
  };

  assert.deepEqual(
    resolveCharacterLoreEntries(character, 'Mara enters the archive at midnight.')
      .map((entry) => entry.content),
    ['Exact case only.', 'Selective match.', 'Always present.'],
  );
  assert.deepEqual(
    resolveCharacterLoreEntries(character, 'mara enters the archive.')
      .map((entry) => entry.content),
    ['Always present.'],
  );
});
