from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "runtime" / "runtime-packages.lock.json"
DEFAULT_OUTPUT = ROOT / "build" / "runtime-packages" / "repository"
DEFAULT_CACHE = ROOT / "build" / "downloads" / "runtime-packages"
DATASET_CARD = ROOT / "huggingface" / "mirid-runtime-packs" / "README.md"
USER_AGENT = "Mirid-Runtime-Stager/1"


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def valid_file(path: Path, package: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == package["source"]["size"]
        and sha256(path) == package["source"]["sha256"]
    )


def find_cached(filename: str, cache_roots: list[Path], package: dict[str, Any]) -> Path | None:
    for root in cache_roots:
        if not root.exists():
            continue
        for candidate in root.rglob(filename):
            if valid_file(candidate, package):
                return candidate
    return None


def download(package: dict[str, Any], cache: Path) -> Path:
    source = package["source"]
    destination = cache / source["asset"]
    if valid_file(destination, package):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if not valid_file(partial, package):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed verification: {source['asset']}")
    partial.replace(destination)
    return destination


def repository_path(output: Path, package: dict[str, Any]) -> Path:
    source = package["source"]
    if package["kind"] == "python-wheel":
        return output / "bindings" / package["package"] / package["platform"] / package["accelerator"] / source["asset"]
    engine = package["engine"].replace(".cpp", "-cpp")
    return output / "runners" / package["family"] / engine / package["platform"] / package["accelerator"] / source["asset"]


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def selected_packages(lock: dict[str, Any], platforms: set[str], kinds: set[str]) -> list[dict[str, Any]]:
    return [
        package
        for package in lock["packages"]
        if (not platforms or package["platform"] in platforms)
        and (not kinds or package["kind"] in kinds)
    ]


def built_binding_packages(output: Path, repository: str) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    for receipt_path in output.glob("bindings/**/build-receipts.json"):
        receipts = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        if isinstance(receipts, dict):
            receipts = [receipts]
        for receipt in receipts:
            record = dict(receipt)
            record["kind"] = "python-wheel"
            record["source"] = dict(record["source"])
            record["source"]["asset"] = record["filename"]
            record["source"]["size"] = record["size"]
            record["source"]["sha256"] = record["sha256"]
            record["downloadUrl"] = (
                f"https://huggingface.co/datasets/{repository}/resolve/main/{record['path']}"
            )
            packages.append(record)
    return packages


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Mirid runtime packages.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument("--kind", action="append", choices=["native-archive", "python-wheel"], default=[])
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    packages = selected_packages(lock, set(args.platform), set(args.kind))
    cache_roots = [args.cache, ROOT / "build" / "downloads", ROOT / "wheelhouse"]
    staged: list[dict[str, Any]] = []
    for index, package in enumerate(packages, start=1):
        filename = package["source"]["asset"]
        print(f"[{index}/{len(packages)}] {package['id']}: {filename}", flush=True)
        cached = find_cached(filename, cache_roots, package) or download(package, args.cache)
        destination = repository_path(args.output, package)
        link_or_copy(cached, destination)
        record = dict(package)
        record["path"] = destination.relative_to(args.output).as_posix()
        record["downloadUrl"] = (
            f"https://huggingface.co/datasets/{lock['repositories']['artifacts']}/resolve/main/{record['path']}"
        )
        staged.append(record)

    manifest_path = args.output / "runtime-packages.manifest.json"
    existing: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("packages", [])
    merged = {package["id"]: package for package in existing}
    merged.update({package["id"]: package for package in staged})
    merged.update(
        {
            package["id"]: package
            for package in built_binding_packages(args.output, lock["repositories"]["artifacts"])
        }
    )
    manifest = {
        "schemaVersion": 1,
        "repository": lock["repositories"]["artifacts"],
        "versions": lock["versions"],
        "packages": sorted(merged.values(), key=lambda package: package["id"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if DATASET_CARD.exists():
        shutil.copy2(DATASET_CARD, args.output / "README.md")
    print(f"Staged {len(staged)} verified packages in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
