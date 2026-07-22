import test from 'node:test';
import assert from 'node:assert/strict';

import { isExternalHref } from './externalLinks.js';

test('recognises links that belong in the system browser', () => {
  assert.equal(isExternalHref('https://openrouter.ai/settings/keys'), true);
  assert.equal(isExternalHref('http://example.com'), true);
  assert.equal(isExternalHref('mailto:hello@mirid.ai'), true);
  assert.equal(isExternalHref('tel:+61000000000'), true);
});

test('leaves app navigation and unsafe protocols alone', () => {
  assert.equal(isExternalHref('/docs'), false);
  assert.equal(isExternalHref('#models'), false);
  assert.equal(isExternalHref('javascript:alert(1)'), false);
  assert.equal(isExternalHref(''), false);
});
