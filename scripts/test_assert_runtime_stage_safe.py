from pathlib import Path

from assert_runtime_stage_safe import find_unsafe_files


def test_rejects_runtime_databases_and_generated_content(tmp_path):
    database = tmp_path / "backend" / "data" / "chess_auth.db"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"sqlite")
    generated = tmp_path / "backend" / "app" / "static" / "generated_images" / "private.png"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"image")
    gallery = tmp_path / "backend" / "app" / "static" / "room_gallery" / "manifest.json"
    gallery.parent.mkdir(parents=True)
    gallery.write_text('{"images": ["private.png"]}', encoding="utf-8")

    assert set(find_unsafe_files(tmp_path)) == {database, generated, gallery}


def test_allows_dependency_metadata_and_voice_assets(tmp_path):
    metadata = tmp_path / "package" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}", encoding="utf-8")
    voice = tmp_path / "backend" / "app" / "static" / "voice_references" / "default.wav"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"wave")
    certificate = tmp_path / "certifi" / "cacert.pem"
    certificate.parent.mkdir(parents=True)
    certificate.write_text("public roots", encoding="utf-8")

    assert find_unsafe_files(tmp_path) == []
