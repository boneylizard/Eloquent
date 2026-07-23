import os
import uuid

import pytest

from .avatar_store import (
    is_managed_avatar_filename,
    migrate_legacy_avatar_files,
    resolve_stored_avatar_file,
)


def avatar_name(extension=".png"):
    return f"{uuid.uuid4()}{extension}"


def test_managed_avatar_filename_requires_uuid_and_supported_extension():
    assert is_managed_avatar_filename(avatar_name(".png"))
    assert is_managed_avatar_filename(avatar_name(".webm"))
    assert not is_managed_avatar_filename("portrait.png")
    assert not is_managed_avatar_filename(f"folder/{avatar_name()}")
    assert not is_managed_avatar_filename(avatar_name(".svg"))


def test_migrate_legacy_avatars_copies_only_managed_root_files(tmp_path):
    legacy = tmp_path / "legacy-static"
    destination = tmp_path / "persistent-avatars"
    legacy.mkdir()
    filename = avatar_name()
    (legacy / filename).write_bytes(b"avatar")
    (legacy / "packaged.png").write_bytes(b"packaged")
    (legacy / "generated_images").mkdir()
    (legacy / "generated_images" / avatar_name()).write_bytes(b"generated")

    migrated = migrate_legacy_avatar_files(legacy, destination)

    assert migrated == [destination / filename]
    assert (destination / filename).read_bytes() == b"avatar"
    assert not (destination / "packaged.png").exists()


def test_migrate_legacy_avatars_never_overwrites_persistent_copy(tmp_path):
    legacy = tmp_path / "legacy-static"
    destination = tmp_path / "persistent-avatars"
    legacy.mkdir()
    destination.mkdir()
    filename = avatar_name()
    (legacy / filename).write_bytes(b"older")
    (destination / filename).write_bytes(b"current")

    assert migrate_legacy_avatar_files(legacy, destination) == []
    assert (destination / filename).read_bytes() == b"current"


def test_resolve_old_static_url_falls_back_to_persistent_avatar(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy-static"
    persistent = tmp_path / "data" / "avatars"
    legacy.mkdir()
    persistent.mkdir(parents=True)
    filename = avatar_name()
    expected = persistent / filename
    expected.write_bytes(b"avatar")
    monkeypatch.setenv("MIRID_DATA_DIR", str(tmp_path / "data"))

    resolved = resolve_stored_avatar_file(
        f"http://127.0.0.1:8000/static/{filename}",
        legacy,
    )

    assert resolved == expected


def test_resolve_avatar_rejects_nested_or_unmanaged_persistent_paths(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy-static"
    legacy.mkdir()
    monkeypatch.setenv("MIRID_DATA_DIR", str(tmp_path / "data"))

    assert resolve_stored_avatar_file("/static/../secret.txt", legacy) is None
    assert resolve_stored_avatar_file("/avatars/not-a-uuid.png", legacy) is None


def test_resolve_avatar_rejects_file_symlink(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy-static"
    outside = tmp_path / "outside.png"
    legacy.mkdir()
    outside.write_bytes(b"outside")
    filename = avatar_name()
    try:
        os.symlink(outside, legacy / filename)
    except OSError as error:
        pytest.skip(f"File symlinks are unavailable: {error}")
    monkeypatch.setenv("MIRID_DATA_DIR", str(tmp_path / "data"))

    assert resolve_stored_avatar_file(f"/static/{filename}", legacy) is None
