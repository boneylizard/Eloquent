import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { autoMapCharacterColumns, characterFromDatasetRow, parseCharacterDatasetText } from './characterDatasetImport.js';

describe('character dataset import', () => {
  it('parses JSONL and maps common aliases', () => {
    const rows = parseCharacterDatasetText('{"character_name":"Ada","persona":"Curious cartographer"}\n', 'cards.jsonl');
    const mapping = autoMapCharacterColumns(Object.keys(rows[0]));
    const result = characterFromDatasetRow(rows[0], mapping);
    assert.equal(result.character.name, 'Ada');
    assert.equal(result.character.description, 'Curious cartographer');
    assert.equal(result.valid, true);
  });

  it('imports a complete V2 card column', () => {
    const row = { card: { spec: 'chara_card_v2', data: { name: 'Mara', description: 'A patient archivist', first_mes: 'You found it.' } } };
    const result = characterFromDatasetRow(row, { card: 'card' });
    assert.equal(result.character.name, 'Mara');
    assert.equal(result.character.first_message, 'You found it.');
  });

  it('recognises a name embedded in SPB persona text', () => {
    const row = { persona_text: '<character>\n**Basic Information**\n**Name:** Navya Patil\n\nA kinetic UX designer.' };
    const mapping = autoMapCharacterColumns(Object.keys(row));
    const result = characterFromDatasetRow(row, mapping);
    assert.equal(result.character.name, 'Navya Patil');
    assert.equal(result.valid, true);
  });
});
