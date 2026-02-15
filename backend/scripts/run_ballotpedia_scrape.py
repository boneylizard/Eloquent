#!/usr/bin/env python3
"""
Run Ballotpedia scraper and optionally import into calibration.
Usage:
  python -m scripts.run_ballotpedia_scrape              # print scraped results only
  python -m scripts.run_ballotpedia_scrape --import   # scrape and add new entries to calibration
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure backend root is on path (parent of scripts/)
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from app import ballotpedia_scraper


def main():
    p = argparse.ArgumentParser(description="Scrape Ballotpedia for special/off-year election results")
    p.add_argument("--import", dest="do_import", action="store_true", help="Merge new results into calibration JSON")
    p.add_argument("--no-nov-filter", action="store_true", help="Include all scraped races, not only since Nov 2025")
    args = p.parse_args()
    if args.do_import:
        out = ballotpedia_scraper.run_and_import_to_calibration(since_nov_2025=not args.no_nov_filter, merge=True)
        print(json.dumps({"added": len(out["added"]), "skipped": len(out["skipped"]), "scraped": len(out["scraped"])}, indent=2))
        print("Added:", [e["label"] for e in out["added"]])
    else:
        results = ballotpedia_scraper.scrape_all(
            since_nov_2025=not args.no_nov_filter,
            include_governors=True,
            include_federal=True,
        )
        print(json.dumps(results, indent=2))
        print("Total:", len(results))


if __name__ == "__main__":
    main()
