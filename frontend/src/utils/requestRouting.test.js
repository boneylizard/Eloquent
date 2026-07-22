import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveUnifiedRequestRoute } from './requestRouting.js';

const endpoint = (id, rotateEnabled) => ({
  id,
  name: id,
  model: `${id}-model`,
  enabled: true,
  rotate_enabled: rotateEnabled,
});

test('falls back to the selected API model when auto-routing has an empty pool', () => {
  const route = resolveUnifiedRequestRoute({
    primaryModel: 'endpoint-selected',
    primaryIsAPI: true,
    settings: {
      apiEndpointRoundRobinEnabled: true,
      customApiEndpoints: [endpoint('endpoint-selected', false)],
    },
  });

  assert.equal(route.autoConfigured, true);
  assert.equal(route.autoEnabled, false);
  assert.equal(route.rotationPoolSize, 0);
  assert.equal(route.selectedModel, 'endpoint-selected');
  assert.equal(route.effectiveModel, 'endpoint-selected');
  assert.equal(route.fallbackReason, 'empty_rotation_pool');
});

test('uses the rotation pool when at least one endpoint is included', () => {
  const route = resolveUnifiedRequestRoute({
    primaryModel: 'endpoint-selected',
    primaryIsAPI: true,
    settings: {
      apiEndpointRoundRobinEnabled: true,
      customApiEndpoints: [
        endpoint('endpoint-selected', false),
        endpoint('endpoint-rotating', true),
      ],
    },
  });

  assert.equal(route.autoEnabled, true);
  assert.equal(route.rotationPoolSize, 1);
  assert.equal(route.effectiveModel, 'endpoint-rotating');
});

test('explicit request overrides remain pinned while auto-routing is configured', () => {
  const route = resolveUnifiedRequestRoute({
    primaryModel: 'endpoint-selected',
    primaryIsAPI: true,
    overrideModel: 'endpoint-override',
    settings: {
      apiEndpointRoundRobinEnabled: true,
      customApiEndpoints: [endpoint('endpoint-rotating', true)],
    },
  });

  assert.equal(route.autoConfigured, true);
  assert.equal(route.autoEnabled, false);
  assert.equal(route.exceptionPinned, true);
  assert.equal(route.effectiveModel, 'endpoint-override');
});
