import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .settings_store import (
    SettingsStoreError,
    create_settings_backup,
    load_settings,
    replace_settings,
    restore_settings_backup,
    update_settings,
)


def test_atomic_updates_preserve_existing_keys(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"theme": "dark", "apiKey": "secret"}, path)

    result = update_settings({"ttsEnabled": True}, path)

    assert result == {"theme": "dark", "apiKey": "secret", "ttsEnabled": True}
    assert load_settings(path) == result
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_invalid_primary_recovers_from_last_known_good_backup(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"theme": "dark", "model": "local"}, path)
    path.write_text("", encoding="utf-8")

    recovered = load_settings(path)

    assert recovered == {"theme": "dark", "model": "local"}
    assert json.loads(path.read_text(encoding="utf-8")) == recovered


def test_concurrent_patches_do_not_drop_settings(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"base": True}, path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda number: update_settings({f"key{number}": number}, path), range(24)))

    result = load_settings(path)
    assert result["base"] is True
    assert all(result[f"key{number}"] == number for number in range(24))


def test_concurrent_processes_do_not_drop_settings(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"base": True}, path)
    backend_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(backend_root), environment.get("PYTHONPATH", "")])
    )

    processes = []
    for number in range(12):
        script = (
            "from pathlib import Path\n"
            "from app.settings_store import update_settings\n"
            f"update_settings({{'process{number}': {number}}}, Path({str(path)!r}))\n"
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=backend_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode != 0:
            failures.append(stderr or stdout)

    assert not failures
    result = load_settings(path)
    assert result["base"] is True
    assert all(result[f"process{number}"] == number for number in range(12))


def test_manual_backup_is_read_only_and_restorable(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"theme": "dark", "apiKey": "secret"}, path)

    backup_path, document = create_settings_backup({"ttsEnabled": True}, path)
    restored = restore_settings_backup(backup_path.read_bytes(), path)

    assert backup_path.exists()
    assert backup_path.stat().st_mode & stat.S_IWRITE == 0
    assert document["format"] == "mirid-settings-backup"
    assert restored == {"theme": "dark", "apiKey": "secret", "ttsEnabled": True}
    backup_path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_restore_rejects_damaged_backup(tmp_path):
    path = tmp_path / "settings.json"
    replace_settings({"theme": "dark"}, path)
    _, document = create_settings_backup(settings_path=path)
    document["settings"]["theme"] = "light"

    with pytest.raises(SettingsStoreError, match="checksum"):
        restore_settings_backup(json.dumps(document), path)
