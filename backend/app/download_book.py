import os
import requests
import sys
from pathlib import Path
import shutil
import threading

# URL provided by the user
BOOK_URL = "https://zipproth.com/Brainfish/Cerebellum_Light_3Merge_200916.7z"
DEST_DIR = Path(__file__).parent / "books"
ARCHIVE_NAME = "chess_book.7z"
TARGET_BIN_NAME = "cerebellum_light.bin"

def install_dependencies():
    """Ensure py7zr is installed."""
    try:
        import py7zr
    except ImportError:
        print("Installing py7zr for chess book extraction...")
        try:
            # unexpected, but we try to install if missing
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
        except Exception as e:
            print(f"Failed to install py7zr: {e}")

def download_file(url, dest_path):
    print(f"Downloading chess book from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_length = r.headers.get('content-length')
            dl = 0
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        dl += len(chunk)
                        f.write(chunk)
                        if total_length:
                            done = int(50 * dl / int(total_length))
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl/1024/1024:.2f} MB")
                            sys.stdout.flush()
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def extract_and_setup(archive_path, dest_dir):
    print(f"Extracting {archive_path}...")
    
    # Create a temporary extraction directory
    extract_temp = dest_dir / "temp_extract"
    extract_temp.mkdir(exist_ok=True)

    try:
        import py7zr
        # Extract using py7zr
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=extract_temp)
        
        # Find the .bin file
        bin_files = list(extract_temp.rglob("*.bin"))
        if not bin_files:
            print("No .bin file found in the archive!")
            return False

        # Move and rename the first .bin file found
        # We rename to standard name to make it easy for the agent to find
        source_bin = bin_files[0]
        target_bin = dest_dir / TARGET_BIN_NAME
        
        print(f"Found book: {source_bin.name}")
        print(f"Installing to: {target_bin}")
        
        if target_bin.exists():
            target_bin.unlink()
            
        shutil.move(str(source_bin), str(target_bin))
        print("Chess book installation successful!")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False
    finally:
        # Cleanup
        if extract_temp.exists():
            shutil.rmtree(extract_temp, ignore_errors=True)
        if archive_path.exists():
            try:
                os.remove(archive_path)
            except OSError:
                pass

def ensure_chess_book():
    """Check if book exists, if not download and install it."""
    target_bin = DEST_DIR / TARGET_BIN_NAME
    if target_bin.exists() and target_bin.stat().st_size > 1024 * 1024: # Check if > 1MB roughly
        print(f"Chess opening book ({TARGET_BIN_NAME}) already exists.")
        return

    print("Chess opening book missing. Starting download/install process...")
    
    install_dependencies()
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DEST_DIR / ARCHIVE_NAME
    
    if download_file(BOOK_URL, archive_path):
        extract_and_setup(archive_path, DEST_DIR)

def ensure_chess_book_background():
    """Run ensure_chess_book in a background thread."""
    thread = threading.Thread(target=ensure_chess_book, daemon=True)
    thread.start()

if __name__ == "__main__":
    ensure_chess_book()
