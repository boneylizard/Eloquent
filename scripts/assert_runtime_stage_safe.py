from __future__ import annotations

import argparse
from pathlib import Path


DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
PRIVATE_SUFFIXES = {".pem", ".key", ".env"}
FORBIDDEN_NAMES = {"settings.json", "outreach_vapid.json", "runtime.ready"}
MUTABLE_DIRECTORIES = (
    Path("backend/data"),
    Path("backend/app/data"),
    Path("backend/app/static/generated_images"),
    Path("backend/app/static/outreach_runtime"),
    Path("backend/app/static/room_gallery"),
)
REQUIRED_RUNTIME_FILES = (
    Path("en_core_web_sm/__init__.py"),
    Path("en_core_web_sm/meta.json"),
    Path("espeakng_loader/__init__.py"),
    Path("espeakng_loader/espeak-ng-data/intonations"),
    Path("espeakng_loader/espeak-ng-data/phondata"),
    Path("espeakng_loader/espeak-ng-data/phontab"),
    Path("misaki/data/gb_gold.json"),
    Path("misaki/data/us_gold.json"),
)
ESPEAK_LIBRARY_NAMES = (
    "espeak-ng.dll",
    "libespeak-ng.so",
    "libespeak-ng.dylib",
)


def find_unsafe_files(root: Path) -> list[Path]:
    unsafe: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        suffix = path.suffix.lower()
        backend_owned = bool(relative.parts) and relative.parts[0].lower() == "backend"
        private_named = "private" in path.name.lower() or "secret" in path.name.lower()
        if (
            suffix in DATABASE_SUFFIXES
            or suffix == ".log"
            or (suffix in PRIVATE_SUFFIXES and (backend_owned or private_named))
            or path.name.lower() in FORBIDDEN_NAMES
        ):
            unsafe.add(path)

    for relative in MUTABLE_DIRECTORIES:
        directory = root / relative
        if directory.is_dir():
            unsafe.update(path for path in directory.rglob("*") if path.is_file())

    return sorted(unsafe)


def find_missing_runtime_files(root: Path) -> list[Path]:
    missing = [relative for relative in REQUIRED_RUNTIME_FILES if not (root / relative).is_file()]
    if not any(root.glob("en_core_web_sm-*.dist-info/METADATA")):
        missing.append(Path("en_core_web_sm-*.dist-info/METADATA"))
    model_root = root / "en_core_web_sm"
    if not any(model_root.glob("en_core_web_sm-*/config.cfg")):
        missing.append(Path("en_core_web_sm/en_core_web_sm-*/config.cfg"))
    library_directory = root / "espeakng_loader"
    if not any((library_directory / name).is_file() for name in ESPEAK_LIBRARY_NAMES):
        missing.append(Path("espeakng_loader/<eSpeak shared library>"))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject mutable or secret-bearing files from a Mirid runtime stage.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Runtime stage does not exist: {root}")

    unsafe = find_unsafe_files(root)
    if unsafe:
        rendered = "\n".join(f"- {path.relative_to(root)}" for path in unsafe)
        raise SystemExit(f"Runtime stage contains mutable or private files:\n{rendered}")

    missing = find_missing_runtime_files(root)
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Runtime stage is missing required files:\n{rendered}")

    print(f"Runtime stage safety check passed: {root}")


if __name__ == "__main__":
    main()
