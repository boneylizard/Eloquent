# Optional: Polyglot opening book for Deep Analysis

Place a **Polyglot .bin** opening book here to speed up Deep Analysis.

- **Fast**: Book depth is found locally (no API calls for the search); only 1–2 Lichess Explorer calls are made for stats.
- **Names**: Look for `performance.bin`, `book.bin`, or `opening.bin` in this folder.
- **Or** set the env var: `CHESS_OPENING_BOOK=/path/to/your/book.bin`
- **Or** put any `.bin` in `backend/data/books/`.

You can download Polyglot books from:
- [Lichess opening book (performance.bin style)](https://github.com/lichess-org/lila/tree/master/public/explorer) — or build from PGN.
- Many engine packs include a `performance.bin` or `book.bin`.

If no book file is present, Deep Analysis still works using only the Lichess Explorer API (with a few more calls).
