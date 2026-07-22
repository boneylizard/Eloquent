import test from 'node:test';
import assert from 'node:assert/strict';

import { getWebSearchResearchPayload, webSearchPathLabel } from './webSearchResearch.js';

test('web search always requests automatic routing', () => {
  assert.deepEqual(getWebSearchResearchPayload({ webSearchStrategy: 'off' }), {
    web_search_strategy: 'auto',
  });
});

test('web search status does not expose routing choices', () => {
  assert.equal(webSearchPathLabel({ path: 'provider_native', status: 'native_delegated' }), 'Searching…');
  assert.equal(webSearchPathLabel({ path: 'eloquent_prefetch', source_count: 3 }), '3 sources');
});
