from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "runtime" / "runtime-packages.sources.json"
DEFAULT_LOCK = ROOT / "runtime" / "runtime-packages.lock.json"
USER_AGENT = "Mirid-Runtime-Resolver/1"


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def github_release(repository: str, revision: str) -> dict[str, Any]:
    suffix = "latest" if revision == "latest" else f"tags/{revision}"
    return request_json(f"https://api.github.com/repos/{repository}/releases/{suffix}")


def matching_asset(release: dict[str, Any], pattern: str) -> dict[str, Any]:
    matches = [asset for asset in release["assets"] if re.match(pattern, asset["name"])]
    if len(matches) != 1:
        names = ", ".join(asset["name"] for asset in matches) or "none"
        raise RuntimeError(f"Expected one asset matching {pattern!r}; found {names}.")
    return matches[0]


def asset_sha256(asset: dict[str, Any]) -> str:
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        return digest.removeprefix("sha256:")
    request = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": USER_AGENT})
    checksum = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        while chunk := response.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def source_record(repository: str, revision: str, asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": repository,
        "revision": revision,
        "asset": asset["name"],
        "size": int(asset["size"]),
        "sha256": asset_sha256(asset),
        "url": asset["browser_download_url"],
    }


def resolve_native_packages(sources: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    releases: dict[str, dict[str, Any]] = {}
    versions: dict[str, str] = {}
    resolved: list[dict[str, Any]] = []
    for package in sources["nativePackages"]:
        upstream_name = package["upstream"]
        upstream = sources["upstreams"][upstream_name]
        if upstream_name not in releases:
            release = github_release(upstream["repository"], upstream["release"])
            releases[upstream_name] = release
            versions[upstream_name] = release["tag_name"]
        release = releases[upstream_name]
        asset = matching_asset(release, package["assetPattern"])
        record = {key: value for key, value in package.items() if key not in {"upstream", "assetPattern"}}
        record["kind"] = "native-archive"
        record["validation"] = "source-sha256-verified"
        record["source"] = source_record(upstream["repository"], release["tag_name"], asset)
        resolved.append(record)
    return resolved, versions


def resolve_binding_wheels(
    sources: dict[str, Any], versions: dict[str, str]
) -> list[dict[str, Any]]:
    upstream = sources["upstreams"]["llamaCppPython"]
    metadata = request_json(upstream["versionSource"])
    version = metadata["info"]["version"]
    versions["llamaCppPython"] = version
    releases: dict[str, dict[str, Any]] = {}
    resolved: list[dict[str, Any]] = []
    for package in sources["bindingWheels"]:
        revision = f"v{version}{package['releaseSuffix']}"
        if revision not in releases:
            releases[revision] = github_release(upstream["repository"], revision)
        asset = matching_asset(releases[revision], package["assetPattern"])
        record = {
            key: value
            for key, value in package.items()
            if key not in {"upstream", "releaseSuffix", "assetPattern"}
        }
        record["kind"] = "python-wheel"
        record["validation"] = "source-sha256-verified"
        record["source"] = source_record(upstream["repository"], revision, asset)
        resolved.append(record)
    return resolved


def resolve_binding_builds(
    sources: dict[str, Any], versions: dict[str, str]
) -> list[dict[str, Any]]:
    upstream = sources["upstreams"]["stableDiffusionCppPython"]
    metadata = request_json(upstream["versionSource"])
    version = metadata["info"]["version"]
    versions["stableDiffusionCppPython"] = version
    source = next(file for file in metadata["urls"] if file["packagetype"] == "sdist")
    resolved: list[dict[str, Any]] = []
    for build in sources["bindingBuilds"]:
        record = {key: value for key, value in build.items() if key != "upstream"}
        record["version"] = version
        record["source"] = {
            "filename": source["filename"],
            "size": int(source["size"]),
            "sha256": source["digests"]["sha256"],
            "url": source["url"],
        }
        resolved.append(record)
    return resolved


def resolve_pypi_wheels(
    sources: dict[str, Any], versions: dict[str, str]
) -> list[dict[str, Any]]:
    metadata_cache: dict[str, dict[str, Any]] = {}
    resolved: list[dict[str, Any]] = []
    for package in sources.get("pypiWheels", []):
        upstream_name = package["upstream"]
        upstream = sources["upstreams"][upstream_name]
        if upstream_name not in metadata_cache:
            metadata_cache[upstream_name] = request_json(upstream["versionSource"])
        metadata = metadata_cache[upstream_name]
        version = metadata["info"]["version"]
        versions[upstream_name] = version
        matches = [file for file in metadata["urls"] if re.match(package["assetPattern"], file["filename"])]
        if len(matches) != 1:
            names = ", ".join(file["filename"] for file in matches) or "none"
            raise RuntimeError(
                f"Expected one PyPI file matching {package['assetPattern']!r}; found {names}."
            )
        file = matches[0]
        record = {
            key: value
            for key, value in package.items()
            if key not in {"upstream", "assetPattern"}
        }
        record["kind"] = "python-wheel"
        record["validation"] = "source-sha256-verified"
        record["source"] = {
            "repository": f"PyPI/{metadata['info']['name']}",
            "revision": version,
            "asset": file["filename"],
            "size": int(file["size"]),
            "sha256": file["digests"]["sha256"],
            "url": file["url"],
        }
        resolved.append(record)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve Mirid runtime sources to immutable release assets.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    native_packages, versions = resolve_native_packages(sources)
    binding_wheels = resolve_binding_wheels(sources, versions)
    binding_builds = resolve_binding_builds(sources, versions)
    pypi_wheels = resolve_pypi_wheels(sources, versions)
    lock = {
        "schemaVersion": 1,
        "resolvedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repositories": sources["repositories"],
        "versions": versions,
        "packages": native_packages + binding_wheels + pypi_wheels,
        "bindingBuilds": binding_builds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Resolved {len(lock['packages'])} packages to {args.output}")


if __name__ == "__main__":
    main()
