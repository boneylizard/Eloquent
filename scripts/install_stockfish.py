#!/usr/bin/env python3
"""
Download Stockfish from official GitHub releases into tools/stockfish/
so the Chess tab can use it without a system install.

Run from repo root: python scripts/install_stockfish.py
Or: install_stockfish.bat (Windows)
"""
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tools" / "stockfish"
GITHUB_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


def main():
    print("Stockfish installer for Eloquent Chess tab")
    print("Downloading latest release from GitHub...")
    req = Request(GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(req, timeout=15) as r:
        release = json.loads(r.read().decode())
    assets = release.get("assets", [])
    # Prefer Windows AVX2 build (best compatibility/speed), then generic Windows
    win_zips = [a for a in assets if "win" in a["name"].lower() and a["name"].endswith(".zip")]
    avx2 = [a for a in win_zips if "avx2" in a["name"].lower()]
    generic = [a for a in win_zips if "x64" in a["name"].lower() and "avx2" not in a["name"].lower()]
    chosen = (avx2 or generic or win_zips)
    if not chosen:
        print("No Windows zip found in release assets.")
        sys.exit(1)
    asset = chosen[0]
    url = asset["browser_download_url"]
    name = asset["name"]
    print(f"Using: {name}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / name
    print("Downloading...")
    with urlopen(url, timeout=60) as r:
        zip_path.write_bytes(r.read())
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUT_DIR)
    zip_path.unlink(missing_ok=True)
    # Release zips often have stockfish.exe inside a subfolder; move to tools/stockfish/stockfish.exe
    exe_path = OUT_DIR / "stockfish.exe"
    if not exe_path.exists():
        for root, dirs, files in os.walk(OUT_DIR):
            for f in files:
                if f.lower().endswith(".exe") and "stockfish" in f.lower():
                    src = Path(root) / f
                    if src != exe_path:
                        src.rename(exe_path)
                    break
            if exe_path.exists():
                break
        if not exe_path.exists():
            for f in OUT_DIR.iterdir():
                if f.is_file() and f.suffix.lower() == ".exe":
                    f.rename(exe_path)
                    break
    # Remove any empty or leftover subdirs from the zip
    import shutil
    for item in list(OUT_DIR.iterdir()):
        if item.is_dir() and item != OUT_DIR:
            shutil.rmtree(item, ignore_errors=True)
    if exe_path.exists():
        print(f"Done. Stockfish is at: {exe_path}")
    else:
        print("Extraction may have used different structure. Check tools/stockfish/ for stockfish.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
