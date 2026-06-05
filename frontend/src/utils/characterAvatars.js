/** Multi-avatar helpers for character cards (up to 10 manual uploads + optional folder cycle). */

export const MAX_CHARACTER_AVATARS = 10;
/** Folder import can supply many looks for call-mode cycling (uploaded to server). */
export const MAX_AVATAR_FOLDER_ITEMS = Infinity;
/** Zoom below 1 shrinks letterboxed media; default 1 = full frame visible (contain). */
export const CALL_MODE_FULLSCREEN_ZOOM_MIN = 1;
export const CALL_MODE_FULLSCREEN_ZOOM_MAX = 2.8;
export const CALL_MODE_FULLSCREEN_ZOOM_DEFAULT = 1;
/** One-time reset of persisted zoom/pan from pre–fit-to-screen defaults. */
export const CALL_MODE_FRAMING_MIGRATION_KEY = 'LiangLocal-call-framing-fit-v1';

export function clampCallModeFullscreenZoom(zoom) {
  const z = Number(zoom);
  if (!Number.isFinite(z)) return CALL_MODE_FULLSCREEN_ZOOM_DEFAULT;
  return Math.max(
    CALL_MODE_FULLSCREEN_ZOOM_MIN,
    Math.min(CALL_MODE_FULLSCREEN_ZOOM_MAX, z)
  );
}

/** Apply pan/zoom transform only when user has zoomed in or panned off center. */
export function shouldApplyCallModeFullscreenTransform(zoom, panX, panY) {
  const z = clampCallModeFullscreenZoom(zoom);
  const px = Number(panX) || 0;
  const py = Number(panY) || 0;
  return z > 1.001 || Math.abs(px) > 0.01 || Math.abs(py) > 0.01;
}

/** Experimental video avatars (call mode autoplay). */
export const AVATAR_VIDEO_EXTENSIONS = ['.mp4', '.webm', '.mov', '.m4v'];
/** Common avatar image extensions (folder picker trusts these; server may still reject). */
export const AVATAR_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff', '.heic', '.heif', '.avif', '.svg'];
/** MIME + extension rules for single-file upload validation. */
export const AVATAR_MEDIA = {
  imageExtensions: AVATAR_IMAGE_EXTENSIONS,
  videoExtensions: AVATAR_VIDEO_EXTENSIONS,
  imageMimePrefixes: ['image/'],
  videoMimePrefixes: ['video/'],
  /** Windows folder picks often use this instead of image/png. */
  opaqueMimeTypes: ['', 'application/octet-stream'],
};
export const AVATAR_IMAGE_MAX_MB = 5;
export const AVATAR_VIDEO_MAX_MB = 80;
/** Call-mode video avatar hotkeys (see CallModeOverlay). */
export const CALL_MODE_VIDEO_HOTKEYS = {
  prevAvatar: '[',
  nextAvatar: ']',
  togglePlay: 'v',
  stop: 's',
  restart: 'g',
};

export function isAvatarVideoUrl(url) {
  if (!url || typeof url !== 'string') return false;
  const path = url.split('?')[0].split('#')[0].toLowerCase();
  return AVATAR_VIDEO_EXTENSIONS.some((ext) => path.endsWith(ext));
}
/** Manual uploads only (legacy `avatar` + `avatars`, max 10). */
export function getManualCharacterAvatarList(character) {
  if (!character) return [];
  const seen = new Set();
  const list = [];
  const push = (url) => {
    if (!url || typeof url !== 'string') return;
    const trimmed = url.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    list.push(trimmed);
  };
  if (Array.isArray(character.avatars)) {
    character.avatars.forEach(push);
  }
  push(character.avatar);
  return list.slice(0, MAX_CHARACTER_AVATARS);
}

/** Server URLs from an imported avatar folder (call-mode cycling). */
export function getCharacterAvatarFolderUrls(character) {
  if (!character || !Array.isArray(character.avatarFolderUrls)) return [];
  return character.avatarFolderUrls
    .filter((url) => typeof url === 'string' && url.trim())
    .map((url) => url.trim())
    .slice(0, MAX_AVATAR_FOLDER_ITEMS);
}

/** Session-only blob URLs from a folder pick (no upload). Not persisted across reload. */
export function getLocalAvatarFolderBlobUrls(character) {
  if (!character || !Array.isArray(character.localAvatarFolderBlobUrls)) return [];
  return character.localAvatarFolderBlobUrls
    .filter((url) => typeof url === 'string' && url.trim())
    .map((url) => url.trim())
    .slice(0, MAX_AVATAR_FOLDER_ITEMS);
}

/** Call-mode + editor cycle list: local folder blobs, uploaded folder, then manual avatars. */
export function getCallModeAvatarCycleList(character) {
  if (!character) return [];
  const seen = new Set();
  const list = [];
  const push = (url) => {
    if (!url || typeof url !== 'string') return;
    const trimmed = url.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    list.push(trimmed);
  };
  getLocalAvatarFolderBlobUrls(character).forEach(push);
  getCharacterAvatarFolderUrls(character).forEach(push);
  getManualCharacterAvatarList(character).forEach(push);
  return list;
}

/** Folder + manual avatars for display and cycling (folder not capped at 10). */
export function getCharacterAvatarList(character) {
  return getCallModeAvatarCycleList(character);
}

/** Basenames to drop from webkitdirectory picks (OS metadata only). */
const AVATAR_FOLDER_JUNK_BASENAMES = new Set(['thumbs.db', 'desktop.ini', '.ds_store', 'ds_store']);

export function getAvatarFolderFilePath(file) {
  return (file?.webkitRelativePath || file?.name || '').trim();
}

/** PNG: image/png, .png, .PNG, or empty type with .png extension (common on Windows). */
export function isPngAvatarFile(file) {
  if (!file) return false;
  const type = (file.type || '').toLowerCase();
  if (type === 'image/png') return true;
  return getAvatarFolderFilePath(file).toLowerCase().endsWith('.png');
}

/** Drop dotfiles and OS junk only — never reject media by MIME. */
export function isAvatarFolderJunk(file) {
  const raw = getAvatarFolderFilePath(file);
  if (!raw) return false;
  const segments = raw.split(/[/\\]/).filter(Boolean);
  for (const seg of segments) {
    const lower = seg.toLowerCase();
    if (AVATAR_FOLDER_JUNK_BASENAMES.has(lower)) return true;
    if (seg.startsWith('.') && seg.length > 1) return true;
  }
  return false;
}

function compareAvatarFolderFiles(a, b) {
  return (a.webkitRelativePath || a.name || '').localeCompare(b.webkitRelativePath || b.name || '', undefined, {
    numeric: true,
    sensitivity: 'base',
  });
}

/** Sort & filter File[] from a directory input — trust the picker; drop only obvious junk. */
export function filterAvatarFolderFiles(fileList) {
  const all = toAvatarFolderFileArray(fileList);
  if (!all.length) return [];
  return all.filter((file) => !isAvatarFolderJunk(file)).sort(compareAvatarFolderFiles);
}

/** @param {FileList|File[]} fileList */
export function toAvatarFolderFileArray(fileList) {
  if (Array.isArray(fileList)) return fileList;
  if (!fileList?.length) return [];
  return Array.from(fileList);
}

/** Local no-upload folder: only drop explicit OS junk basenames (no MIME/extension/dotfile rules). */
export function isLocalAvatarFolderJunkFile(file) {
  const raw = getAvatarFolderFilePath(file);
  if (!raw) return false;
  const base = raw.split(/[/\\]/).filter(Boolean).pop()?.toLowerCase() || '';
  return AVATAR_FOLDER_JUNK_BASENAMES.has(base);
}

/**
 * Files for local call-mode blob cycle — every picker file except explicit OS junk.
 * @param {FileList|File[]} pickedFiles snapshot from folder input (before clearing input value)
 */
export function prepareLocalAvatarFolderFiles(pickedFiles, maxItems = MAX_AVATAR_FOLDER_ITEMS) {
  const all = toAvatarFolderFileArray(pickedFiles);
  const sorted = all.filter((file) => !isLocalAvatarFolderJunkFile(file)).sort(compareAvatarFolderFiles);
  const truncatedCount = Math.max(0, sorted.length - maxItems);
  const files = sorted.slice(0, maxItems);
  return {
    files,
    folderLabel: getAvatarFolderLabelFromFiles(files) || getAvatarFolderLabelFromFiles(all),
    truncatedCount,
    junkSkipped: all.length - sorted.length,
  };
}

/** Folder name from webkitdirectory paths, or empty if unknown. */
export function getAvatarFolderLabelFromFiles(files) {
  const first = files?.[0];
  const rel = typeof first?.webkitRelativePath === 'string' ? first.webkitRelativePath : '';
  if (rel.includes('/')) return rel.split('/')[0];
  return '';
}

/** Sample paths for debugging when a folder pick looks empty. */
export function avatarFolderPickDebugSample(fileList, limit = 8) {
  const all = toAvatarFolderFileArray(fileList);
  return {
    count: all.length,
    sample: all.slice(0, limit).map((f) => f.webkitRelativePath || f.name || '(unnamed)'),
  };
}

/**
 * Prepare a webkitdirectory pick for upload (minimal filtering; no MIME/size probes).
 * @returns {{ folderLabel: string, files: File[], totalPicked: number, junkCount: number, truncatedCount: number, pickSample: string[] }}
 */
export function analyzeAvatarFolderPick(fileList, maxItems = MAX_AVATAR_FOLDER_ITEMS) {
  const all = toAvatarFolderFileArray(fileList);
  const folderLabel = getAvatarFolderLabelFromFiles(all);
  const junkCount = all.filter((f) => isAvatarFolderJunk(f)).length;
  let validSorted = filterAvatarFolderFiles(fileList);
  if (all.length > 0 && validSorted.length === 0) {
    validSorted = [...all].sort(compareAvatarFolderFiles);
  }
  const truncatedCount = Math.max(0, validSorted.length - maxItems);
  const files = validSorted.slice(0, maxItems);
  const { sample: pickSample } = avatarFolderPickDebugSample(fileList);
  const label =
    folderLabel ||
    (files.length ? `${files.length} file${files.length === 1 ? '' : 's'}` : 'folder');
  return {
    folderLabel: label,
    files,
    totalPicked: all.length,
    junkCount,
    truncatedCount,
    pickSample,
  };
}

function revokeBlobUrlList(urls) {
  (urls || []).forEach((url) => {
    if (typeof url === 'string' && url.startsWith('blob:')) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* ignore */
      }
    }
  });
}

export function revokeAvatarFolderBlobUrls(character) {
  revokeBlobUrlList(getCharacterAvatarFolderUrls(character));
  revokeBlobUrlList(getLocalAvatarFolderBlobUrls(character));
}

/** Strip session blob URLs before writing character to storage. */
export function omitPersistedLocalAvatarFolder(character) {
  if (!character || typeof character !== 'object') return character;
  const { localAvatarFolderBlobUrls: _local, ...rest } = character;
  return rest;
}

/** Re-attach in-memory local folder blobs when reading a character from storage. */
export function mergeSessionLocalAvatarFolder(storedCharacter, sessionCharacter) {
  if (!storedCharacter) return sessionCharacter || storedCharacter;
  if (!sessionCharacter || sessionCharacter.id !== storedCharacter.id) return storedCharacter;
  const localBlobs = getLocalAvatarFolderBlobUrls(sessionCharacter);
  if (!localBlobs.length) {
    return normalizeCharacterAvatars({
      ...storedCharacter,
      activeAvatarIndex:
        sessionCharacter.activeAvatarIndex ?? storedCharacter.activeAvatarIndex,
    });
  }
  return normalizeCharacterAvatars({
    ...storedCharacter,
    localAvatarFolderBlobUrls: localBlobs,
    avatarFolderLabel:
      sessionCharacter.avatarFolderLabel?.trim() || storedCharacter.avatarFolderLabel,
    activeAvatarIndex: sessionCharacter.activeAvatarIndex ?? storedCharacter.activeAvatarIndex,
  });
}

export function setCharacterAvatarFolder(character, urls, label = '') {
  if (!character || typeof character !== 'object') return character;
  const trimmed = (Array.isArray(urls) ? urls : [])
    .filter((u) => typeof u === 'string' && u.trim())
    .map((u) => u.trim())
    .slice(0, MAX_AVATAR_FOLDER_ITEMS);
  return normalizeCharacterAvatars({
    ...character,
    avatarFolderUrls: trimmed,
    avatarFolderLabel: typeof label === 'string' ? label.trim() : '',
    activeAvatarIndex: character?.activeAvatarIndex ?? 0,
  });
}

/** Session-only local folder blobs for call-mode (no server upload). */
export function setLocalAvatarFolderBlobs(character, blobUrls, label = '') {
  if (!character || typeof character !== 'object') return character;
  revokeBlobUrlList(getLocalAvatarFolderBlobUrls(character));
  const trimmed = (Array.isArray(blobUrls) ? blobUrls : [])
    .filter((u) => typeof u === 'string' && u.trim())
    .map((u) => u.trim())
    .slice(0, MAX_AVATAR_FOLDER_ITEMS);
  const folderLabel = typeof label === 'string' ? label.trim() : '';
  return normalizeCharacterAvatars({
    ...character,
    localAvatarFolderBlobUrls: trimmed,
    avatarFolderLabel: folderLabel || character.avatarFolderLabel || '',
    activeAvatarIndex: character?.activeAvatarIndex ?? 0,
  });
}

export function clearCharacterAvatarFolder(character) {
  if (!character || typeof character !== 'object') return character;
  revokeAvatarFolderBlobUrls(character);
  const next = {
    ...character,
    avatarFolderUrls: [],
    localAvatarFolderBlobUrls: [],
    avatarFolderLabel: '',
  };
  return normalizeCharacterAvatars(next);
}

export function clampAvatarIndex(index, listLength) {
  if (listLength <= 0) return 0;
  const n = Number(index);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(Math.floor(n), listLength - 1));
}

/** Active avatar URL for display and new messages. */
export function getActiveCharacterAvatar(character) {
  const list = getCharacterAvatarList(character);
  if (!list.length) return null;
  const idx = clampAvatarIndex(character?.activeAvatarIndex, list.length);
  return list[idx] ?? list[0];
}

/** Normalize stored character: sync avatars[], avatar, activeAvatarIndex. */
export function normalizeCharacterAvatars(character) {
  if (!character || typeof character !== 'object') return character;
  const list = getCharacterAvatarList(character);
  const manual = getManualCharacterAvatarList(character);
  const folder = getCharacterAvatarFolderUrls(character);
  const localFolder = getLocalAvatarFolderBlobUrls(character);
  const idx = clampAvatarIndex(character.activeAvatarIndex, list.length);
  const primary = list[idx] ?? list[0] ?? null;
  return {
    ...character,
    avatars: manual,
    avatarFolderUrls: folder,
    localAvatarFolderBlobUrls: localFolder,
    avatarFolderLabel:
      typeof character.avatarFolderLabel === 'string' ? character.avatarFolderLabel.trim() : '',
    avatar: primary,
    activeAvatarIndex: list.length ? idx : 0,
  };
}

export function cycleAvatarIndex(character, delta = 1) {
  const list = getCharacterAvatarList(character);
  if (list.length <= 1) return character;
  const cur = clampAvatarIndex(character?.activeAvatarIndex, list.length);
  const next = (cur + delta + list.length * 10) % list.length;
  return normalizeCharacterAvatars({ ...character, activeAvatarIndex: next });
}

export function setAvatarIndexOnCharacter(character, index) {
  const list = getCharacterAvatarList(character);
  if (!list.length) return normalizeCharacterAvatars(character);
  return normalizeCharacterAvatars({ ...character, activeAvatarIndex: clampAvatarIndex(index, list.length) });
}

export function addAvatarToCharacter(character, url) {
  if (!url) return normalizeCharacterAvatars(character);
  const manual = getManualCharacterAvatarList(character);
  if (manual.includes(url)) return normalizeCharacterAvatars(character);
  if (manual.length >= MAX_CHARACTER_AVATARS) return normalizeCharacterAvatars(character);
  const nextManual = [...manual, url];
  return normalizeCharacterAvatars({
    ...character,
    avatars: nextManual,
    activeAvatarIndex: getCharacterAvatarList(character).length === 0 ? 0 : character?.activeAvatarIndex ?? 0,
  });
}

export function removeAvatarAtIndex(character, index) {
  const list = getCharacterAvatarList(character);
  if (!list.length) return normalizeCharacterAvatars(character);
  const idx = clampAvatarIndex(index, list.length);
  const removedUrl = list[idx];
  const folder = getCharacterAvatarFolderUrls(character);
  const localFolder = getLocalAvatarFolderBlobUrls(character);
  const manual = getManualCharacterAvatarList(character);
  const nextLocal = localFolder.filter((u) => u !== removedUrl);
  const nextFolder = folder.filter((u) => u !== removedUrl);
  const nextManual = manual.filter((u) => u !== removedUrl);
  const nextList = [
    ...nextLocal,
    ...nextFolder,
    ...nextManual.filter((u) => !nextLocal.includes(u) && !nextFolder.includes(u)),
  ];
  let nextActive = clampAvatarIndex(character?.activeAvatarIndex, list.length);
  if (nextActive >= nextList.length) nextActive = Math.max(0, nextList.length - 1);
  return normalizeCharacterAvatars({
    ...character,
    avatars: nextManual,
    avatarFolderUrls: nextFolder,
    localAvatarFolderBlobUrls: nextLocal,
    avatar: nextList[nextActive] ?? null,
    activeAvatarIndex: nextActive,
  });
}

export function resolveAvatarDisplayUrl(avatarSource, apiUrl) {
  if (!avatarSource) return null;
  if (avatarSource.startsWith('http') || avatarSource.startsWith('data:')) return avatarSource;
  if (avatarSource.startsWith('/')) return `${apiUrl || ''}${avatarSource}`;
  if (apiUrl) return `${apiUrl}/static/${avatarSource}`;
  return avatarSource;
}

function hasKnownAvatarExtension(name) {
  const n = (name || '').toLowerCase();
  if (AVATAR_VIDEO_EXTENSIONS.some((ext) => n.endsWith(ext))) return true;
  return AVATAR_IMAGE_EXTENSIONS.some((ext) => n.endsWith(ext));
}

/** MIME / extension check for single-file avatar upload (not folder picker). */
export function isAllowedAvatarUpload(file) {
  if (!file) return false;
  if (isPngAvatarFile(file)) return true;
  const path = getAvatarFolderFilePath(file) || (file.name || '');
  const type = (file.type || '').toLowerCase();
  if (AVATAR_MEDIA.imageMimePrefixes.some((p) => type.startsWith(p))) return true;
  if (AVATAR_MEDIA.videoMimePrefixes.some((p) => type.startsWith(p))) return true;
  if (AVATAR_MEDIA.opaqueMimeTypes.includes(type)) {
    return hasKnownAvatarExtension(path);
  }
  return hasKnownAvatarExtension(path);
}

export function avatarUploadMaxBytes(file) {
  const isVideo = (file?.type || '').startsWith('video/') || isAvatarVideoUrl(file?.name || '');
  return isVideo
    ? AVATAR_VIDEO_MAX_MB * 1024 * 1024
    : AVATAR_IMAGE_MAX_MB * 1024 * 1024;
}
