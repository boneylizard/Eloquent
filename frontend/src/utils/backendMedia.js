/**
 * Resolve a backend-owned media path only at the point where the browser needs
 * to load it. Chat messages and settings should keep the backend's canonical
 * relative path so they remain portable when the local API port changes.
 */
export function selectMediaBackendUrl(
  gpuId,
  {
    primaryApiUrl = '',
    memoryApiUrl = '',
  } = {},
) {
  const useMemoryBackend = Number(gpuId) === 1;
  const preferredApiUrl = useMemoryBackend ? memoryApiUrl : primaryApiUrl;
  const fallbackApiUrl = useMemoryBackend ? primaryApiUrl : memoryApiUrl;
  return (preferredApiUrl || fallbackApiUrl || '').trim().replace(/\/+$/, '');
}

export function resolveBackendMediaUrl(
  mediaPath,
  {
    gpuId = 0,
    primaryApiUrl = '',
    memoryApiUrl = '',
  } = {},
) {
  if (typeof mediaPath !== 'string') return '';

  const source = mediaPath.trim();
  if (!source) return '';

  // Preserve remote, embedded and browser-managed sources exactly as supplied.
  // The general scheme check also covers file:, asset: and tauri: URLs.
  if (/^[a-z][a-z\d+.-]*:/i.test(source) || source.startsWith('//')) {
    return source;
  }

  const apiUrl = selectMediaBackendUrl(gpuId, { primaryApiUrl, memoryApiUrl });

  if (!apiUrl) return source;

  const relativePath = source.replace(/^(?:\.\/|\/)+/, '');
  return `${apiUrl}/${relativePath}`;
}

/**
 * Whether a source belongs to Mirid's backend and is therefore safe to
 * cache-bust. Third-party HTTP URLs may be signed, so their query strings must
 * never be changed.
 */
export function isBackendOwnedMediaSource(
  mediaPath,
  {
    primaryApiUrl = '',
    memoryApiUrl = '',
  } = {},
) {
  if (typeof mediaPath !== 'string' || !mediaPath.trim()) return false;

  const source = mediaPath.trim();
  if (!/^[a-z][a-z\d+.-]*:/i.test(source) && !source.startsWith('//')) {
    return true;
  }

  return [primaryApiUrl, memoryApiUrl]
    .filter((apiUrl) => typeof apiUrl === 'string' && apiUrl.trim())
    .some((apiUrl) => {
      const base = apiUrl.trim().replace(/\/+$/, '');
      return source === base || source.startsWith(`${base}/`);
    });
}

/**
 * Force a fresh HTTP request after a user retries a failed media load.
 * Embedded/blob sources cannot be cache-busted without changing their data.
 */
export function withBackendMediaRetryToken(mediaUrl, retryAttempt) {
  if (
    typeof mediaUrl !== 'string'
    || !/^https?:\/\//i.test(mediaUrl)
    || !Number.isFinite(Number(retryAttempt))
    || Number(retryAttempt) <= 0
  ) {
    return mediaUrl;
  }

  const hashIndex = mediaUrl.indexOf('#');
  const beforeHash = hashIndex >= 0 ? mediaUrl.slice(0, hashIndex) : mediaUrl;
  const hash = hashIndex >= 0 ? mediaUrl.slice(hashIndex) : '';
  const separator = beforeHash.includes('?') ? '&' : '?';
  return `${beforeHash}${separator}mirid_retry=${Math.trunc(Number(retryAttempt))}${hash}`;
}
