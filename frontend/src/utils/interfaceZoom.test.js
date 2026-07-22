import test from 'node:test';
import assert from 'node:assert/strict';
import {
  INTERFACE_ZOOM_DEFAULT,
  INTERFACE_ZOOM_MAX,
  INTERFACE_ZOOM_MIN,
  normaliseInterfaceZoom,
} from './interfaceZoom.js';

test('normaliseInterfaceZoom uses the readable default for invalid values', () => {
  assert.equal(normaliseInterfaceZoom(undefined), INTERFACE_ZOOM_DEFAULT);
  assert.equal(normaliseInterfaceZoom(null), INTERFACE_ZOOM_DEFAULT);
  assert.equal(normaliseInterfaceZoom('not-a-number'), INTERFACE_ZOOM_DEFAULT);
});

test('normaliseInterfaceZoom clamps and rounds values', () => {
  assert.equal(normaliseInterfaceZoom(0.2), INTERFACE_ZOOM_MIN);
  assert.equal(normaliseInterfaceZoom(3), INTERFACE_ZOOM_MAX);
  assert.equal(normaliseInterfaceZoom(1.123), 1.12);
});
