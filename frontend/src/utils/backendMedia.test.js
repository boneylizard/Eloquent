import assert from 'node:assert/strict';
import test from 'node:test';

import {
  isBackendOwnedMediaSource,
  resolveBackendMediaUrl,
  selectMediaBackendUrl,
  withBackendMediaRetryToken,
} from './backendMedia.js';

test('backend media paths resolve against the primary API without changing the stored path', () => {
  const storedPath = '/static/generated_images/example.png';
  const resolved = resolveBackendMediaUrl(storedPath, {
    primaryApiUrl: 'http://127.0.0.1:8000/',
    memoryApiUrl: 'http://127.0.0.1:8001',
    gpuId: 0,
  });

  assert.equal(resolved, 'http://127.0.0.1:8000/static/generated_images/example.png');
  assert.equal(storedPath, '/static/generated_images/example.png');
});

test('GPU 1 media resolves against the memory API', () => {
  assert.equal(
    resolveBackendMediaUrl('static/generated_images/secondary.png', {
      primaryApiUrl: 'http://127.0.0.1:8000',
      memoryApiUrl: 'http://127.0.0.1:8001/',
      gpuId: 1,
    }),
    'http://127.0.0.1:8001/static/generated_images/secondary.png',
  );
});

test('media resolution falls back to the available backend URL', () => {
  assert.equal(
    resolveBackendMediaUrl('/static/generated_images/fallback.png', {
      primaryApiUrl: 'http://127.0.0.1:8000',
      memoryApiUrl: '',
      gpuId: '1',
    }),
    'http://127.0.0.1:8000/static/generated_images/fallback.png',
  );
});

test('GPU-routed media operations use the backend that owns the model state', () => {
  const urls = {
    primaryApiUrl: 'http://127.0.0.1:8000/',
    memoryApiUrl: 'http://127.0.0.1:8001/',
  };

  assert.equal(selectMediaBackendUrl(0, urls), 'http://127.0.0.1:8000');
  assert.equal(selectMediaBackendUrl(1, urls), 'http://127.0.0.1:8001');
  assert.equal(
    selectMediaBackendUrl(1, { primaryApiUrl: urls.primaryApiUrl }),
    'http://127.0.0.1:8000',
  );
});

test('absolute, embedded and blob media URLs are preserved', () => {
  const urls = [
    'https://images.example.test/example.png',
    'http://127.0.0.1:8000/static/example.png',
    'data:image/png;base64,AAAA',
    'blob:https://app.example.test/asset-id',
    'file:///C:/Mirid/example.png',
    '//cdn.example.test/example.png',
  ];

  for (const url of urls) {
    assert.equal(
      resolveBackendMediaUrl(url, {
        primaryApiUrl: 'http://127.0.0.1:8000',
        memoryApiUrl: 'http://127.0.0.1:8001',
      }),
      url,
    );
  }
});

test('blank and non-string media values resolve to an empty source', () => {
  assert.equal(resolveBackendMediaUrl('   ', { primaryApiUrl: 'http://127.0.0.1:8000' }), '');
  assert.equal(resolveBackendMediaUrl(null, { primaryApiUrl: 'http://127.0.0.1:8000' }), '');
});

test('only relative or local-backend sources are safe to cache-bust', () => {
  const apiUrls = {
    primaryApiUrl: 'http://127.0.0.1:8000/',
    memoryApiUrl: 'http://127.0.0.1:8001',
  };

  assert.equal(isBackendOwnedMediaSource('/static/generated_images/local.png', apiUrls), true);
  assert.equal(
    isBackendOwnedMediaSource('http://127.0.0.1:8001/static/generated_images/local.png', apiUrls),
    true,
  );
  assert.equal(
    isBackendOwnedMediaSource('https://cdn.example.test/image.png?signature=keep-me', apiUrls),
    false,
  );
  assert.equal(isBackendOwnedMediaSource('data:image/png;base64,AAAA', apiUrls), false);
});

test('retry tokens preserve existing query strings and fragments', () => {
  assert.equal(
    withBackendMediaRetryToken('http://127.0.0.1:8000/image.png?size=full#preview', 2),
    'http://127.0.0.1:8000/image.png?size=full&mirid_retry=2#preview',
  );
  assert.equal(
    withBackendMediaRetryToken('data:image/png;base64,AAAA', 2),
    'data:image/png;base64,AAAA',
  );
  assert.equal(
    withBackendMediaRetryToken('http://127.0.0.1:8000/image.png', 0),
    'http://127.0.0.1:8000/image.png',
  );
});
