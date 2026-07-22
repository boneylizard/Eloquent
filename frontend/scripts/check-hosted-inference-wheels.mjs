import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const releasePath = path.resolve(scriptDir, '..', '..', 'runtime', 'inference-wheels.release.json');
const release = JSON.parse(await readFile(releasePath, 'utf8'));
const baseUrl = new URL(release.publishBaseUrl);
const match = baseUrl.pathname.match(/^\/datasets\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/);
if (!match) throw new Error(`Unsupported inference wheel URL: ${release.publishBaseUrl}`);

const [, repository, revision, directory] = match;
const apiUrl = `https://huggingface.co/api/datasets/${repository}/tree/${revision}/${directory}?recursive=false&expand=true`;
const response = await fetch(apiUrl, { signal: AbortSignal.timeout(30_000) });
if (!response.ok) throw new Error(`Hugging Face returned HTTP ${response.status} for ${apiUrl}`);
const files = await response.json();
const failures = [];

for (const wheel of release.packages) {
  const hosted = files.find((file) => file.path === `${directory}/${wheel.filename}`);
  if (!hosted) {
    failures.push(`${wheel.name} is missing: ${wheel.filename}`);
    continue;
  }
  if (hosted.size !== wheel.size) failures.push(`${wheel.name} size is ${hosted.size}; expected ${wheel.size}`);
  const hostedHash = hosted.lfs?.oid?.replace(/^sha256:/, '');
  if (hostedHash !== wheel.sha256) failures.push(`${wheel.name} SHA-256 is ${hostedHash || 'unavailable'}; expected ${wheel.sha256}`);

  const download = await fetch(`${release.publishBaseUrl}/${wheel.filename}`, {
    headers: { Range: 'bytes=0-0' },
    signal: AbortSignal.timeout(30_000),
  });
  const contentRange = download.headers.get('content-range');
  if (download.status !== 206 || contentRange !== `bytes 0-0/${wheel.size}`) {
    failures.push(`${wheel.name} does not support a verified one-byte resume request`);
  }
  await download.arrayBuffer();
}

const hostedManifestResponse = await fetch(`${release.publishBaseUrl}/inference-wheels.manifest.json`, {
  signal: AbortSignal.timeout(30_000),
});
if (!hostedManifestResponse.ok) {
  failures.push(`hosted inference manifest returned HTTP ${hostedManifestResponse.status}`);
} else {
  const hostedManifest = await hostedManifestResponse.json();
  for (const wheel of release.packages) {
    const hostedWheel = hostedManifest.packages?.find((item) => item.name === wheel.name);
    if (hostedWheel?.sha256 !== wheel.sha256 || hostedWheel?.size !== wheel.size) {
      failures.push(`hosted manifest does not match the trusted ${wheel.name} release`);
    }
  }
}

if (failures.length) {
  throw new Error(`Hosted Mirid inference wheel verification failed:\n- ${failures.join('\n- ')}`);
}

console.log(`Hosted Mirid inference wheels verified: ${repository}@${revision}.`);
