#!/usr/bin/env python3
"""
Run the Tk file picker in isolation. Prints one path, or JSON array when multiple=true.
Exit code: 0 = selected, 1 = cancelled, 2 = error.
"""
import json
import sys
from pathlib import Path


def main():
    title = None
    initial_directory = None
    filetypes = None
    multiple = False
    if not sys.stdin.isatty():
        try:
            line = sys.stdin.readline()
            if line.strip():
                opts = json.loads(line)
                title = opts.get("title")
                initial_directory = opts.get("initial_directory")
                filetypes = opts.get("filetypes")
                multiple = bool(opts.get("multiple"))
        except Exception:
            pass

    try:
        from tkinter import Tk, filedialog
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    root = Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    options: dict = {}
    if title:
        options["title"] = title
    if initial_directory:
        initial = Path(initial_directory)
        if initial.is_file():
            options["initialdir"] = str(initial.parent)
        elif initial.is_dir():
            options["initialdir"] = str(initial)
    if filetypes:
        options["filetypes"] = filetypes
    else:
        options["filetypes"] = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.opus *.webm *.aac *.mp4 *.mkv"),
            ("All files", "*.*"),
        ]

    try:
        if multiple:
            selected = filedialog.askopenfilenames(**options)
            if not selected:
                return 1
            print(json.dumps(list(selected)))
            return 0
        selected = filedialog.askopenfilename(**options)
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if not selected:
        return 1
    print(selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
