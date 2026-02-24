#!/usr/bin/env python3
"""
Run the Tk directory picker in isolation. Used by the backend so a Tk crash
does not kill the main server process. Prints the selected path to stdout, or
nothing if cancelled. Exit code: 0 = path selected, 1 = cancelled, 2 = error.
"""
import json
import sys
from pathlib import Path


def main():
    # Optional title and initial_directory from stdin (one JSON object)
    title = None
    initial_directory = None
    if not sys.stdin.isatty():
        try:
            line = sys.stdin.readline()
            if line.strip():
                opts = json.loads(line)
                title = opts.get("title")
                initial_directory = opts.get("initial_directory")
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

    options = {}
    if title:
        options["title"] = title
    if initial_directory and Path(initial_directory).is_dir():
        options["initialdir"] = initial_directory

    try:
        selected = filedialog.askdirectory(**options)
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
