import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(scriptDir, '..');
const rootDir = path.resolve(frontendDir, '..');
const failures = [];

async function readJson(filePath) {
  return JSON.parse(await readFile(filePath, 'utf8'));
}

async function requireNonEmptyFile(filePath, label) {
  try {
    const file = await stat(filePath);
    if (!file.isFile() || file.size === 0) failures.push(`${label} is empty: ${filePath}`);
  } catch {
    failures.push(`${label} is missing: ${filePath}`);
  }
}

function capture(source, pattern, label) {
  const match = source.match(pattern);
  if (!match) {
    failures.push(`Could not read ${label}`);
    return null;
  }
  return match[1];
}

const nodeMajor = Number(process.versions.node.split('.')[0]);
if (nodeMajor !== 24) {
  failures.push(`Node 24 is required; current runtime is ${process.version}. Run your Node version manager from the repository root.`);
}

const packageJsonPath = path.join(frontendDir, 'package.json');
const packageLockPath = path.join(frontendDir, 'package-lock.json');
const tauriConfigPath = path.join(rootDir, 'src-tauri', 'tauri.conf.json');
const cargoPath = path.join(rootDir, 'src-tauri', 'Cargo.toml');
const rustLibPath = path.join(rootDir, 'src-tauri', 'src', 'runtime_windows.rs');
const runtimeReleasePath = path.join(rootDir, 'runtime', 'runtime-release.json');
const inferenceLockPath = path.join(rootDir, 'runtime', 'inference-wheels.lock.json');
const inferenceReleasePath = path.join(rootDir, 'runtime', 'inference-wheels.release.json');

const [packageJson, packageLock, tauriConfig, cargoSource, rustSource, runtimeRelease, inferenceLock, inferenceRelease] = await Promise.all([
  readJson(packageJsonPath),
  readJson(packageLockPath),
  readJson(tauriConfigPath),
  readFile(cargoPath, 'utf8'),
  readFile(rustLibPath, 'utf8'),
  readJson(runtimeReleasePath),
  readJson(inferenceLockPath),
  readJson(inferenceReleasePath),
]);
const secretLogSources = await Promise.all([
  path.join(rootDir, 'backend', 'app', 'main.py'),
  path.join(rootDir, 'backend', 'app', 'user_utils.py'),
  path.join(rootDir, 'backend', 'app', 'tts_backend.py'),
  path.join(rootDir, 'backend', 'app', 'tts_service.py'),
].map(async (filePath) => ({ filePath, source: await readFile(filePath, 'utf8') })));

const lockRoot = packageLock.packages?.[''];
if (!lockRoot) {
  failures.push('package-lock.json has no root package entry');
} else {
  if (lockRoot.name !== packageJson.name) failures.push('package-lock.json package name does not match package.json');
  if (lockRoot.version !== packageJson.version) failures.push('package-lock.json version does not match package.json');
}

if (packageJson.name !== 'mirid-frontend') failures.push('Frontend package name must be mirid-frontend');
if (tauriConfig.productName !== 'Mirid') failures.push('Tauri productName must be Mirid');
if (tauriConfig.identifier !== 'ai.mirid.desktop') failures.push('Tauri identifier must be ai.mirid.desktop');
if (tauriConfig.build?.beforeBuildCommand?.script !== 'npm run release:build') {
  failures.push('Tauri beforeBuildCommand must run npm run release:build');
}
const csp = tauriConfig.app?.security?.csp || '';
const imageDirective = csp
  .split(';')
  .map((directive) => directive.trim())
  .find((directive) => directive.startsWith('img-src '));
for (const loopbackImageSource of ['http://127.0.0.1:*', 'http://localhost:*']) {
  if (!imageDirective?.split(/\s+/).includes(loopbackImageSource)) {
    failures.push(`Tauri img-src must allow Mirid's local avatar source ${loopbackImageSource}`);
  }
}

const cargoVersion = capture(cargoSource, /^version\s*=\s*"([^"]+)"/m, 'Cargo package version');
for (const [label, version] of [
  ['frontend', packageJson.version],
  ['Tauri', tauriConfig.version],
  ['Cargo', cargoVersion],
]) {
  if (version !== packageJson.version) failures.push(`${label} version ${version} does not match ${packageJson.version}`);
}

for (const iconPath of tauriConfig.bundle?.icon || []) {
  await requireNonEmptyFile(path.join(rootDir, 'src-tauri', iconPath), 'Tauri icon');
}

const runtimeVersion = capture(rustSource, /const RUNTIME_VERSION:\s*&str\s*=\s*"([^"]+)"/, 'runtime version');
const runtimeBase = capture(rustSource, /const HF_BASE:\s*&str\s*=\s*"([^"]+)"/, 'runtime base URL');
const runtimeArchive = capture(rustSource, /const RUNTIME_ARCHIVE:\s*&str\s*=\s*"([^"]+)"/, 'runtime archive name');
const sidecarExe = capture(rustSource, /const SIDECAR_EXE:\s*&str\s*=\s*"([^"]+)"/, 'sidecar executable name');
const runtimeArchiveSize = capture(rustSource, /const RUNTIME_ARCHIVE_SIZE:\s*u64\s*=\s*([\d_]+)/, 'runtime archive size');
const sidecarExeSize = capture(rustSource, /const SIDECAR_EXE_SIZE:\s*u64\s*=\s*([\d_]+)/, 'sidecar executable size');
const runtimeArchiveSha256 = capture(rustSource, /const RUNTIME_ARCHIVE_SHA256:\s*&str\s*=\s*\n?\s*"([a-f0-9]+)"/, 'runtime archive SHA-256');
const sidecarExeSha256 = capture(rustSource, /const SIDECAR_EXE_SHA256:\s*&str\s*=\s*"([a-f0-9]+)"/, 'sidecar executable SHA-256');

if (!runtimeVersion?.trim()) failures.push('Runtime version must not be empty');
try {
  const parsedRuntimeBase = new URL(runtimeBase);
  if (parsedRuntimeBase.protocol !== 'https:') failures.push('Runtime base URL must use HTTPS');
} catch {
  failures.push(`Runtime base URL is invalid: ${runtimeBase}`);
}
if (!runtimeArchive?.endsWith('.7z')) failures.push('Runtime archive must be a .7z file');
if (!sidecarExe?.endsWith('.exe')) failures.push('Sidecar executable must be a Windows .exe');
if (!runtimeBase?.includes('/mirid-runtime/')) failures.push('Runtime base URL must use the Mirid runtime repository');
if (!runtimeArchive?.startsWith('mirid-runtime-')) failures.push('Runtime archive must use the Mirid release name');
if (!sidecarExe?.startsWith('mirid-sidecar-')) failures.push('Sidecar executable must use the Mirid release name');
const lockedRuntime = runtimeRelease.assets?.runtimeArchive;
const lockedSidecar = runtimeRelease.assets?.sidecarExecutable;
for (const [label, actual, locked] of [
  ['runtime version', runtimeVersion, runtimeRelease.runtimeVersion],
  ['runtime base URL', runtimeBase, runtimeRelease.baseUrl],
  ['runtime archive name', runtimeArchive, lockedRuntime?.filename],
  ['runtime archive size', Number(runtimeArchiveSize?.replaceAll('_', '')), lockedRuntime?.size],
  ['runtime archive SHA-256', runtimeArchiveSha256, lockedRuntime?.sha256],
  ['sidecar name', sidecarExe, lockedSidecar?.filename],
  ['sidecar size', Number(sidecarExeSize?.replaceAll('_', '')), lockedSidecar?.size],
  ['sidecar SHA-256', sidecarExeSha256, lockedSidecar?.sha256],
]) {
  if (actual !== locked) failures.push(`${label} does not match runtime/runtime-release.json`);
}
for (const [label, size] of [
  ['runtime archive', runtimeArchiveSize],
  ['sidecar executable', sidecarExeSize],
]) {
  if (!size || Number(size.replaceAll('_', '')) <= 0) failures.push(`${label} size must be positive`);
}
for (const [label, hash] of [
  ['runtime archive', runtimeArchiveSha256],
  ['sidecar executable', sidecarExeSha256],
]) {
  if (!/^[a-f0-9]{64}$/.test(hash || '')) failures.push(`${label} must have a lowercase SHA-256 digest`);
}
if (!/sevenz-rust\s*=/.test(cargoSource)) failures.push('The embedded sevenz-rust extractor dependency is required');
if (!/sha2\s*=/.test(cargoSource)) failures.push('The SHA-256 verification dependency is required');
if (/SEVENZ_PATH|Program Files\\7-Zip|7z\.exe/.test(rustSource)) {
  failures.push('Runtime installation must not depend on an external 7-Zip installation');
}
for (const { filePath, source } of secretLogSources) {
  for (const forbidden of [
    'Settings file contents:',
    'Current settings:',
    'Received settings for new stream: {tts_settings}',
    'settings_data[:100]',
  ]) {
    if (source.includes(forbidden)) failures.push(`Secret-bearing log pattern '${forbidden}' remains in ${filePath}`);
  }
}

if (inferenceRelease.python !== '3.12') failures.push('Inference wheels must target Python 3.12');
if (inferenceRelease.platform !== 'win_amd64') failures.push('Inference wheels must target win_amd64');
if (!inferenceRelease.cudaArchitectures?.includes('86')) failures.push('Inference wheels must support the Mirid RTX 3090 validation target (sm_86)');
if (inferenceRelease.publishBaseUrl !== inferenceLock.publishBaseUrl) {
  failures.push('Inference wheel release URL does not match the build lock');
}
for (const lockedPackage of inferenceLock.packages || []) {
  const releasedPackage = inferenceRelease.packages?.find((item) => item.name === lockedPackage.name);
  if (!releasedPackage) {
    failures.push(`Inference wheel release is missing ${lockedPackage.name}`);
    continue;
  }
  for (const field of ['version', 'sourceUrl', 'sourceSha256']) {
    if (releasedPackage[field] !== lockedPackage[field]) {
      failures.push(`${lockedPackage.name} ${field} does not match the reviewed build lock`);
    }
  }
  if (!releasedPackage.filename?.endsWith('.whl')) failures.push(`${lockedPackage.name} release filename must be a wheel`);
  if (!Number.isSafeInteger(releasedPackage.size) || releasedPackage.size <= 0) failures.push(`${lockedPackage.name} release size must be positive`);
  if (!/^[a-f0-9]{64}$/.test(releasedPackage.sha256 || '')) failures.push(`${lockedPackage.name} release SHA-256 is invalid`);
}

if (failures.length) {
  throw new Error(`Mirid release preflight failed:\n- ${failures.join('\n- ')}`);
}

console.log(`Mirid release preflight passed for v${packageJson.version} on ${process.version}.`);
