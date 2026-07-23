from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .runtime_paths import data_path


AVATAR_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".webm",
    ".mov",
    ".m4v",
}


def avatar_storage_directory() -> Path:
    return data_path("avatars")


def is_managed_avatar_filename(filename: str) -> bool:
    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix.lower() not in AVATAR_EXTENSIONS:
        return False
    try:
        return str(uuid.UUID(candidate.stem)) == candidate.stem.lower()
    except (ValueError, AttributeError):
        return False


def migrate_legacy_avatar_files(
    legacy_static_directory: Path,
    destination: Path | None = None,
) -> list[Path]:
    destination = destination or avatar_storage_directory()
    destination.mkdir(parents=True, exist_ok=True)
    migrated: list[Path] = []
    if not legacy_static_directory.is_dir():
        return migrated

    for source in legacy_static_directory.iterdir():
        if source.is_symlink() or not source.is_file():
            continue
        if not is_managed_avatar_filename(source.name):
            continue
        target = destination / source.name
        if not target.exists():
            shutil.copy2(source, target)
            migrated.append(target)
    return migrated


def persistent_avatar_path(filename: str) -> Path | None:
    if not is_managed_avatar_filename(filename):
        return None
    return avatar_storage_directory() / filename


def contained_regular_file(candidate: Path, root: Path) -> Path | None:
    """Resolve a real file without following a file symlink beyond its root."""
    if candidate.is_symlink():
        return None
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve(strict=True)
        candidate_resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    return candidate_resolved if candidate_resolved.is_file() else None


def resolve_stored_avatar_file(
    avatar_source: str,
    legacy_static_directory: Path,
) -> Path | None:
    source = str(avatar_source or "").strip()
    if not source:
        return None

    parsed = urlparse(source)
    path = parsed.path if parsed.scheme or parsed.netloc else source
    if path.startswith("/avatars/"):
        filename = path.removeprefix("/avatars/")
        if Path(filename).name != filename:
            return None
        candidates = [persistent_avatar_path(filename)]
    elif path.startswith("/static/"):
        filename = path.removeprefix("/static/")
        if Path(filename).name != filename:
            return None
        candidates = [
            legacy_static_directory / filename,
            persistent_avatar_path(filename),
        ]
    elif "/" not in path and "\\" not in path:
        filename = path
        candidates = [
            persistent_avatar_path(filename),
            legacy_static_directory / filename,
        ]
    else:
        return None

    for candidate in candidates:
        if candidate is None:
            continue
        root = (
            avatar_storage_directory()
            if candidate.parent == avatar_storage_directory()
            else legacy_static_directory
        )
        resolved = contained_regular_file(candidate, root)
        if resolved is not None:
            return resolved
    return None
