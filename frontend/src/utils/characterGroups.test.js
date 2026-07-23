import test from 'node:test';
import assert from 'node:assert/strict';

import {
  normaliseCharacterGroup,
  parseCharacterGroups,
} from './characterGroups.js';

test('character groups keep reusable members and shared instructions', () => {
  const group = normaliseCharacterGroup({
    id: 'group-one',
    name: '  The cast  ',
    characterIds: ['a', 'b', 'a', '', null],
    context: '  Stay in the same scene.  ',
    created_at: '2026-07-23T00:00:00.000Z',
  });

  assert.equal(group.id, 'group-one');
  assert.equal(group.name, 'The cast');
  assert.deepEqual(group.characterIds, ['a', 'b']);
  assert.equal(group.context, 'Stay in the same scene.');
});

test('character group parsing ignores malformed and unnamed entries', () => {
  const parsed = parseCharacterGroups(JSON.stringify([
    { id: 'valid', name: 'Valid group', characterIds: ['a', 'b'] },
    { id: 'unnamed', name: '', characterIds: ['a'] },
    null,
  ]));

  assert.equal(parsed.length, 1);
  assert.equal(parsed[0].id, 'valid');
});
