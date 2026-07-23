from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

SETTINGS_BACKUP_FORMAT = "mirid-settings-backup"
SETTINGS_BACKUP_VERSION = 1
_SETTINGS_LOCK = threading.RLock()
_PROCESS_LOCK_TIMEOUT_SECONDS = 30.0
_PROCESS_LOCK_RETRY_SECONDS = 0.05


class SettingsStoreError(RuntimeError):
    pass


def get_settings_path() -> Path:
    return Path.home() / ".LiangLocal" / "settings.json"


def get_settings_backup_dir(settings_path: Path | None = None) -> Path:
    path = settings_path or get_settings_path()
    return path.parent / "settings-backups"


def _internal_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _process_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _process_settings_lock(path: Path) -> Iterator[None]:
    lock_path = _process_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()

        deadline = time.monotonic() + _PROCESS_LOCK_TIMEOUT_SECONDS
        while not acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise SettingsStoreError(
                        "Timed out waiting for another Mirid process to finish saving settings."
                    ) from exc
                time.sleep(_PROCESS_LOCK_RETRY_SECONDS)

        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                logger.exception("Failed to release settings lock %s", lock_path)
        handle.close()


@contextmanager
def _settings_transaction(path: Path) -> Iterator[None]:
    with _SETTINGS_LOCK:
        with _process_settings_lock(path):
            yield


def _validate_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SettingsStoreError("Settings must be a JSON object.")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise SettingsStoreError(f"{path.name} is empty.")
    try:
        return _validate_settings(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise SettingsStoreError(f"{path.name} is not valid JSON.") from exc


def _write_json_atomically(
    path: Path,
    payload: dict[str, Any],
    *,
    read_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if read_only:
            path.chmod(stat.S_IREAD)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _load_settings_unlocked(path: Path) -> dict[str, Any]:
    backup = _internal_backup_path(path)
    if path.exists():
        try:
            return _read_json_object(path)
        except SettingsStoreError as primary_error:
            if backup.exists():
                try:
                    recovered = _read_json_object(backup)
                    _write_json_atomically(path, recovered)
                    logger.warning(
                        "Recovered invalid settings file from %s after: %s",
                        backup,
                        primary_error,
                    )
                    return recovered
                except SettingsStoreError:
                    pass
            raise primary_error

    if backup.exists():
        recovered = _read_json_object(backup)
        _write_json_atomically(path, recovered)
        logger.warning("Recovered missing settings file from %s", backup)
        return recovered
    return {}


def _replace_settings_unlocked(path: Path, settings: dict[str, Any]) -> None:
    validated = _validate_settings(settings)
    backup = _internal_backup_path(path)
    previous: dict[str, Any] | None = None
    if path.exists():
        try:
            previous = _read_json_object(path)
        except SettingsStoreError:
            if backup.exists():
                try:
                    previous = _read_json_object(backup)
                except SettingsStoreError:
                    previous = None

    _write_json_atomically(backup, previous or validated)
    _write_json_atomically(path, validated)
    _write_json_atomically(backup, validated)


def load_settings(settings_path: Path | None = None) -> dict[str, Any]:
    path = settings_path or get_settings_path()
    with _settings_transaction(path):
        return dict(_load_settings_unlocked(path))


def replace_settings(
    settings: dict[str, Any],
    settings_path: Path | None = None,
) -> dict[str, Any]:
    path = settings_path or get_settings_path()
    validated = dict(_validate_settings(settings))
    with _settings_transaction(path):
        _replace_settings_unlocked(path, validated)
    return validated


def update_settings(
    patch: dict[str, Any],
    settings_path: Path | None = None,
) -> dict[str, Any]:
    path = settings_path or get_settings_path()
    validated_patch = _validate_settings(patch)
    with _settings_transaction(path):
        settings = _load_settings_unlocked(path)
        settings.update(validated_patch)
        _replace_settings_unlocked(path, settings)
        return dict(settings)


def mutate_settings(
    mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
    settings_path: Path | None = None,
) -> dict[str, Any]:
    path = settings_path or get_settings_path()
    with _settings_transaction(path):
        settings = _load_settings_unlocked(path)
        result = mutator(dict(settings))
        next_settings = settings if result is None else _validate_settings(result)
        _replace_settings_unlocked(path, next_settings)
        return dict(next_settings)


def _canonical_settings_bytes(settings: dict[str, Any]) -> bytes:
    return json.dumps(
        settings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _settings_checksum(settings: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_settings_bytes(settings)).hexdigest()


def create_settings_backup(
    overlay: dict[str, Any] | None = None,
    settings_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = settings_path or get_settings_path()
    with _settings_transaction(path):
        settings = _load_settings_unlocked(path)
        if overlay is not None:
            settings.update(_validate_settings(overlay))

        created_at = datetime.now(timezone.utc)
        document = {
            "format": SETTINGS_BACKUP_FORMAT,
            "version": SETTINGS_BACKUP_VERSION,
            "createdAt": created_at.isoformat(),
            "settingsSha256": _settings_checksum(settings),
            "settings": settings,
        }
        filename = f"Mirid-settings-{created_at.strftime('%Y-%m-%d_%H-%M-%S-%fZ')}.json"
        backup_path = get_settings_backup_dir(path) / filename
        _write_json_atomically(backup_path, document, read_only=True)
        return backup_path, document


def restore_settings_backup(
    raw: bytes | str,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SettingsStoreError("The selected file is not valid JSON.") from exc

    if not isinstance(document, dict):
        raise SettingsStoreError("The selected file does not contain a settings object.")

    if document.get("format") == SETTINGS_BACKUP_FORMAT:
        settings = _validate_settings(document.get("settings"))
        expected_checksum = str(document.get("settingsSha256") or "").strip().lower()
        actual_checksum = _settings_checksum(settings)
        if not expected_checksum or expected_checksum != actual_checksum:
            raise SettingsStoreError(
                "The backup checksum does not match. The file may be incomplete or damaged."
            )
    else:
        settings = _validate_settings(document)

    if not settings:
        raise SettingsStoreError("The selected backup contains no settings.")
    return replace_settings(settings, settings_path)
