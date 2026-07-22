from pathlib import Path

from .runtime_paths import data_path, runtime_data_root


def test_runtime_data_root_respects_desktop_override(monkeypatch, tmp_path):
    configured = tmp_path / "Mirid Data"
    monkeypatch.setenv("MIRID_DATA_DIR", str(configured))

    assert runtime_data_root() == configured.resolve()
    assert data_path("documents", "index.json") == configured.resolve() / "documents" / "index.json"


def test_runtime_data_root_defaults_to_source_data(monkeypatch):
    monkeypatch.delenv("MIRID_DATA_DIR", raising=False)

    expected = Path(__file__).resolve().parent.parent / "data"
    assert runtime_data_root() == expected
