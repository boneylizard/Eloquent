from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "runtime" / "runtime-packages.lock.json"
DEFAULT_ARTIFACTS = ROOT / "build" / "runtime-packages" / "repository"
DEFAULT_SPACE = ROOT / "huggingface" / "mirid-runtime-registry"


def run(*arguments: str) -> None:
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Mirid's runtime packages and registry Space.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    args = parser.parse_args()

    if not shutil.which("hf"):
        raise RuntimeError("The Hugging Face CLI is required.")
    run("hf", "auth", "whoami")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    artifact_repo = lock["repositories"]["artifacts"]
    space_repo = lock["repositories"]["registrySpace"]
    run("hf", "repos", "create", artifact_repo, "--type", "dataset", "--public", "--exist-ok")
    run(
        "hf",
        "repos",
        "create",
        space_repo,
        "--type",
        "space",
        "--space-sdk",
        "static",
        "--public",
        "--exist-ok",
    )
    if args.create_only:
        return
    if not args.skip_artifacts:
        run(
            "hf",
            "upload",
            artifact_repo,
            str(args.artifacts),
            ".",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Publish Mirid runtime packages",
        )
    if not args.skip_space:
        run(
            "hf",
            "upload",
            space_repo,
            str(args.space),
            ".",
            "--repo-type",
            "space",
            "--commit-message",
            "Publish Mirid runtime registry",
        )


if __name__ == "__main__":
    main()
