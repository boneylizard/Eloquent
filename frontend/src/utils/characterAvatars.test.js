import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveAvatarDisplayUrl } from './characterAvatars.js';


test('backend-relative character avatars resolve against the local API', () => {
  assert.equal(
    resolveAvatarDisplayUrl('/static/avatars/mara.png', 'http://localhost:8000'),
    'http://localhost:8000/static/avatars/mara.png',
  );
  assert.equal(
    resolveAvatarDisplayUrl('mara.png', 'http://localhost:8000'),
    'http://localhost:8000/static/mara.png',
  );
});


test('browser-managed and absolute avatar sources remain unchanged', () => {
  const sources = [
    'https://images.example.test/mara.png',
    'data:image/png;base64,AAAA',
    'blob:https://app.example.test/avatar-id',
    'file:///C:/Mirid/mara.png',
    '//cdn.example.test/mara.png',
  ];

  for (const source of sources) {
    assert.equal(resolveAvatarDisplayUrl(source, 'http://localhost:8000'), source);
  }
});
