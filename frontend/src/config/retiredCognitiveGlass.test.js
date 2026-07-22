import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const sourceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('cognitive glass has no active frontend entry points', () => {
  const activeSources = [
    'App.jsx',
    'components/Chat.jsx',
    'components/Settings.jsx',
    'contexts/AppContext.jsx',
  ].map((path) => readFileSync(resolve(sourceRoot, path), 'utf8'));

  const activeBundleText = activeSources.join('\n');
  for (const retiredReference of [
    'CognitiveGlass',
    'AgenticToggle',
    'AgenticProfileSettings',
    'agenticEnabled',
    'toggleAgenticMode',
    'runAgenticTurn',
  ]) {
    assert.equal(activeBundleText.includes(retiredReference), false, retiredReference);
  }
});
