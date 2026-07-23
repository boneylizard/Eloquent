import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  convertGingerToTavern,
  convertTavernToGinger,
  importCharacterCard,
  importCharacterCardFiles,
  isSupportedCharacterCardFile,
} from './CharacterCardUtils.js';

describe('TavernAI and SillyTavern card compatibility', () => {
  it('preserves V2 fields, future data, extensions and lore metadata', () => {
    const source = {
      spec: 'chara_card_v2',
      spec_version: '2.0',
      source_url: 'https://example.test/card',
      data: {
        name: 'Mara',
        description: 'An archivist who guards impossible maps.',
        personality: 'Patient, exacting and quietly funny.',
        scenario: 'A sealed archive after midnight.',
        first_mes: 'You should not have found this room.',
        mes_example: '{{user}}: Is that map alive?\n{{char}}: Only when it is frightened.\n{{user}}: Should I be worried?\n{{char}}: Not yet.',
        creator_notes: 'Designed for mystery roleplay.',
        system_prompt: 'Write Mara with restraint.',
        post_history_instructions: 'Never decide the user\'s actions.',
        alternate_greetings: ['The archive door closes behind you.'],
        tags: ['mystery', 'slow burn'],
        creator: 'Card Author',
        character_version: '2.3',
        future_card_field: { enabled: true },
        extensions: {
          third_party: { untouched: true },
          mirid: { background: 'Former royal cartographer.', speech_style: 'Dry and precise.' },
        },
        character_book: {
          name: 'Archive lore',
          token_budget: 700,
          extensions: { book_plugin: { mode: 'strict' } },
          entries: [{
            id: 42,
            keys: ['map'],
            content: 'The maps remember every traveller.',
            enabled: false,
            insertion_order: 9,
            extensions: { entry_plugin: 'keep-me' },
          }],
        },
      },
    };

    const internal = convertTavernToGinger(source);
    assert.equal(internal.personality, source.data.personality);
    assert.equal(internal.example_dialogue.length, 4);
    assert.deepEqual(internal.alternate_greetings, source.data.alternate_greetings);

    const exported = convertGingerToTavern({ ...internal, description: 'Updated description.' });
    assert.equal(exported.data.description, 'Updated description.');
    assert.equal(exported.data.personality, source.data.personality);
    assert.equal(exported.data.future_card_field.enabled, true);
    assert.equal(exported.data.extensions.third_party.untouched, true);
    assert.equal(exported.data.character_book.token_budget, 700);
    assert.equal(exported.data.character_book.entries[0].id, 42);
    assert.equal(exported.data.character_book.entries[0].enabled, false);
    assert.equal(exported.data.character_book.entries[0].extensions.entry_plugin, 'keep-me');
    assert.equal(exported.source_url, source.source_url);
    assert.match(exported.data.mes_example, /Should I be worried\?/);
  });

  it('imports the six-field TavernAI V1 format without losing personality', () => {
    const internal = convertTavernToGinger({
      name: 'Ada',
      description: 'A field researcher.',
      personality: 'Curious and candid.',
      scenario: 'A remote station.',
      first_mes: 'The radio finally works.',
      mes_example: 'You: Can you hear me?\nAda: Clearly.',
    });

    const exported = convertGingerToTavern(internal);
    assert.equal(exported.data.name, 'Ada');
    assert.equal(exported.data.personality, 'Curious and candid.');
    assert.equal(exported.data.first_mes, 'The radio finally works.');
    assert.equal(internal.example_dialogue.length, 2);
  });

  it('accepts mixed JSON and PNG selections without imposing a batch limit', async () => {
    const files = Array.from({ length: 512 }, (_, index) => ({
      name: `card-${index}.${index % 2 === 0 ? 'json' : 'png'}`,
      webkitRelativePath: `collection/card-${index}.${index % 2 === 0 ? 'json' : 'png'}`,
    }));
    files.push({ name: 'notes.txt', webkitRelativePath: 'collection/notes.txt' });

    const progress = [];
    const result = await importCharacterCardFiles(files, 'http://unused.test', {
      importer: async (file) => ({ name: file.name }),
      onProgress: (entry) => progress.push(entry),
    });

    assert.equal(result.imported.length, 512);
    assert.equal(result.failed.length, 0);
    assert.deepEqual(result.skipped, ['collection/notes.txt']);
    assert.equal(progress.length, 512);
    assert.equal(progress.at(-1).current, 512);
  });

  it('keeps valid cards when another selected card fails', async () => {
    const files = [
      { name: 'valid.json' },
      { name: 'broken.png' },
      { name: 'also-valid.PNG' },
    ];
    const result = await importCharacterCardFiles(files, null, {
      importer: async (file) => {
        if (file.name === 'broken.png') throw new Error('No embedded card data');
        return { name: file.name };
      },
    });

    assert.deepEqual(result.imported.map((entry) => entry.fileName), ['valid.json', 'also-valid.PNG']);
    assert.deepEqual(result.failed, [{
      fileName: 'broken.png',
      message: 'No embedded card data',
    }]);
  });

  it('rejects JSON that is not a named character card', async () => {
    const file = {
      name: 'settings.json',
      type: 'application/json',
      text: async () => JSON.stringify({ theme: 'dark' }),
    };

    await assert.rejects(
      importCharacterCard(file),
      /does not contain a named TavernAI character card/,
    );
  });

  it('recognises supported card extensions case-insensitively', () => {
    assert.equal(isSupportedCharacterCardFile({ name: 'card.JSON' }), true);
    assert.equal(isSupportedCharacterCardFile({ name: 'card.PnG' }), true);
    assert.equal(isSupportedCharacterCardFile({ name: 'portrait.jpg' }), false);
  });
});
