import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rustPath = path.resolve(scriptDir, '..', '..', 'src-tauri', 'src', 'runtime_windows.rs');
const rustSource = await readFile(rustPath, 'utf8');

function capture(pattern, label) {
  const match = rustSource.match(pattern);
  if (!match) throw new Error(`Could not read ${label} from ${rustPath}`);
  return match[1];
}

function asset(namePattern, sizePattern, hashPattern, label) {
  return {
    label,
    name: capture(namePattern, `${label} name`),
    size: Number(capture(sizePattern, `${label} size`).replaceAll('_', '')),
    sha256: capture(hashPattern, `${label} SHA-256`),
  };
}

const baseUrl = capture(/const HF_BASE:\s*&str\s*=\s*"([^"]+)"/, 'runtime base URL');
const match = new URL(baseUrl).pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/([^/]+)$/);
if (!match) throw new Error(`Unsupported Hugging Face runtime URL: ${baseUrl}`);

const [, repository, revision] = match;
const assets = [
  asset(
    /const RUNTIME_ARCHIVE:\s*&str\s*=\s*"([^"]+)"/,
    /const RUNTIME_ARCHIVE_SIZE:\s*u64\s*=\s*([\d_]+)/,
    /const RUNTIME_ARCHIVE_SHA256:\s*&str\s*=\s*\n?\s*"([a-f0-9]+)"/,
    'runtime archive',
  ),
  asset(
    /const SIDECAR_EXE:\s*&str\s*=\s*"([^"]+)"/,
    /const SIDECAR_EXE_SIZE:\s*u64\s*=\s*([\d_]+)/,
    /const SIDECAR_EXE_SHA256:\s*&str\s*=\s*"([a-f0-9]+)"/,
    'sidecar executable',
  ),
];

const apiUrl = `https://huggingface.co/api/models/${repository}/tree/${revision}?recursive=false&expand=true`;
const response = await fetch(apiUrl, { signal: AbortSignal.timeout(30_000) });
if (!response.ok) throw new Error(`Hugging Face returned HTTP ${response.status} for ${apiUrl}`);
const files = await response.json();
const failures = [];

for (const asset of assets) {
  const hosted = files.find((file) => file.path === asset.name);
  if (!hosted) {
    failures.push(`${asset.label} is missing: ${asset.name}`);
    continue;
  }
  if (hosted.size !== asset.size) {
    failures.push(`${asset.label} size is ${hosted.size}; release manifest expects ${asset.size}`);
  }
  const hostedHash = hosted.lfs?.oid?.replace(/^sha256:/, '');
  if (hostedHash !== asset.sha256) {
    failures.push(`${asset.label} SHA-256 is ${hostedHash || 'unavailable'}; release manifest expects ${asset.sha256}`);
  }
}

if (failures.length) {
  throw new Error(`Hosted Mirid runtime verification failed:\n- ${failures.join('\n- ')}`);
}

console.log(`Hosted Mirid runtime verified: ${repository}@${revision}.`);
