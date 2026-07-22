import { readFile } from 'node:fs/promises';
import path from 'node:path';

const inventoryPath = path.resolve('dist/.vite/module-inventory.json');
const inventory = JSON.parse(await readFile(inventoryPath, 'utf8'));
const sourceFiles = new Set(inventory.modules);
const includedModules = new Set(
  String(process.env.VITE_MIRID_INCLUDED_MODULES || '')
    .split(',')
    .map((moduleId) => moduleId.trim())
    .filter(Boolean),
);

const optionalModules = {
  elections: {
    entry: 'src/components/ElectionTracker.jsx',
    sources: [
      'src/components/ElectionMap.jsx',
      'src/components/ElectionNews.jsx',
      'src/components/ElectionTracker.jsx',
      'src/components/PollTable.jsx',
      'src/components/PollTrends.jsx',
    ],
  },
};

const retiredSources = [
  'src/components/ChatlogCondenserAgentPanel.jsx',
  'src/components/ChatlogCondenserOrchestratorPanel.jsx',
  'src/components/ChatlogCondenserPanel.jsx',
  'src/components/ChatlogCondenserRagOptions.jsx',
  'src/components/CodeEditorOverlay.jsx',
  'src/components/ForensicLinguistics.jsx',
  'src/components/MarketSimTab.jsx',
  'src/components/WatchTab.jsx',
];

const failures = [];

for (const [moduleId, policy] of Object.entries(optionalModules)) {
  const shouldBeIncluded = includedModules.has(moduleId);
  const bundledSources = policy.sources.filter((sourceFile) => sourceFiles.has(sourceFile));
  if (shouldBeIncluded && !sourceFiles.has(policy.entry)) {
    failures.push(`${policy.entry} is missing, but ${moduleId} is included`);
  }
  if (!shouldBeIncluded && bundledSources.length) {
    failures.push(`${moduleId} is not included, but these sources were bundled: ${bundledSources.join(', ')}`);
  }
}

for (const sourceFile of retiredSources) {
  if (sourceFiles.has(sourceFile)) failures.push(`${sourceFile} must not ship in a release artefact`);
}

if (failures.length) {
  throw new Error(`Release artefact policy failed:\n- ${failures.join('\n- ')}`);
}

console.log(`Release artefact policy passed (${sourceFiles.size} bundled source modules).`);
