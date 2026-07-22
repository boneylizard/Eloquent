import assert from 'node:assert/strict';
import test from 'node:test';

import { isModuleEnabled, modulePolicy } from './modules.js';

test('keeps the Lattice pool unavailable while its source is retained', () => {
  assert.equal(modulePolicy('pool').lockedOff, true);
  assert.equal(isModuleEnabled('pool'), false);
});

test('keeps chess retired and inaccessible while its source is retained', () => {
  assert.equal(modulePolicy('chess').retired, true);
  assert.equal(modulePolicy('chess').lockedOff, true);
  assert.equal(isModuleEnabled('chess'), false);
});
