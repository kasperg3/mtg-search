"""Step 1a: download MTGJSON files needed for EXP-007 (MIT licence) into data/bench_raw/mtgjson/.

    uv run python bench/precon/collect_mtgjson.py [--refresh]

Files: DeckList.json, SetList.json, Meta.json, AtomicCards.json.gz (first printings),
AllDeckFiles.tar.gz (only Commander Deck files are extracted to decks/). Cached; never re-fetched
unless --refresh.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "bench_raw" / "mtgjson"
BASE = "https://mtgjson.com/api/v5/"
FILES = ["Meta.json", "DeckList.json", "SetList.json", "AtomicCards.json.gz", "AllDeckFiles.tar.gz"]
UA = "mtg-search research (kasper@colourbox.com)"
DECK_TYPES = ("Commander Deck", "MTGO Commander Deck")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": UA}, timeout=600, follow_redirects=True) as c:
        for f in FILES:
            p = DIR / f
            if p.exists() and not a.refresh:
                continue
            with c.stream("GET", BASE + f) as r:
                r.raise_for_status()
                with p.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
            print("downloaded", f, p.stat().st_size)
            time.sleep(2)
    wanted = {d["fileName"] + ".json" for d in json.load(open(DIR / "DeckList.json"))["data"]
              if d["type"] in DECK_TYPES}
    out = DIR / "decks"
    out.mkdir(exist_ok=True)
    with tarfile.open(DIR / "AllDeckFiles.tar.gz") as t:
        members = [m for m in t.getmembers() if m.isfile() and Path(m.name).name in wanted]
        t.extractall(out, members=members, filter="data")
    print(f"extracted {len(members)} commander deck files")


if __name__ == "__main__":
    main()
