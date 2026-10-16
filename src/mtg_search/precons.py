"""Official preconstructed decklists from MTGJSON (MIT licence) as one small table, ``precons.parquet``.

    python -m mtg_search.precons [--tarball AllDeckFiles.tar.gz] [--out data/precons.parquet]

Without ``--tarball`` it downloads MTGJSON's ``AllDeckFiles.tar.gz`` (about 260 MB, once; afterwards only when
MTGJSON has a newer build, via If-Modified-Since) into ``data/raw/mtgjson/`` with a contact User-Agent, and checks
it against the published ``.sha256``. The table keeps names, oracle ids, counts and sections only: MTGJSON's
per-printing fields (prices, purchase links, ranks, foil flags, set numbers) are dropped. The server reads it
from its data directory and never calls MTGJSON itself.

One row per deck: ``file`` (MTGJSON file name), ``name``, ``type``, ``set_code``, ``release_date``, ``source``
(the publisher's decklist page), ``commanders`` (names) and ``cards`` (list of {oracle_id, name, qty, section},
section = commander | main | sideboard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from collections.abc import Iterable, Iterator
from email.utils import formatdate
from pathlib import Path
from typing import Any

import polars as pl

from . import config

MTGJSON = "https://mtgjson.com/api/v5/"
USER_AGENT = "mtg-search (kaspergrontved@proton.me)"
ARCHIVE = "AllDeckFiles.tar.gz"
# card bundles, land packs and redemption lists, not decks anyone plays or upgrades
EXCLUDED_TYPES = frozenset({
    "Secret Lair Drop", "Bundle Land Pack", "MTGO Redemption", "Deck Builder's Toolkit",
    "Booster Battle Pack Packet", "Welcome Booster", "San Diego Comic Con Promos", "Demo Deck",
})
BOARDS = (("commander", "commander"), ("mainBoard", "main"), ("sideBoard", "sideboard"))
# MTGJSON deck type -> the format whose construction rules apply
SOURCE_FORMATS = {
    "Commander Deck": "commander", "MTGO Commander Deck": "commander",
    "Historic Brawl Precon Deck": "brawl", "Pioneer Challenger Deck": "pioneer",
}
SCHEMA = {
    "file": pl.String, "name": pl.String, "type": pl.String, "set_code": pl.String, "release_date": pl.String,
    "source": pl.String, "commanders": pl.List(pl.String),
    "cards": pl.List(pl.Struct({"oracle_id": pl.String, "name": pl.String, "qty": pl.Int32, "section": pl.String})),
}


def deck_record(file_name: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """One MTGJSON Deck (``data`` of a deck file) -> a table row, or None for excluded or empty decks.
    Printings of the same card in a section are merged (foil and non-foil copies are separate entries)."""
    if data.get("type") in EXCLUDED_TYPES:
        return None
    cards: dict[tuple[str, str], dict[str, Any]] = {}
    for board, section in BOARDS:
        for c in data.get(board) or []:
            oid = (c.get("identifiers") or {}).get("scryfallOracleId")
            slot = cards.setdefault((oid or c["name"], section),
                                    {"oracle_id": oid, "name": c["name"], "qty": 0, "section": section})
            slot["qty"] += int(c.get("count") or 1)
    if not any(s["section"] != "sideboard" for s in cards.values()):
        return None
    return {
        "file": file_name, "name": data.get("name") or file_name, "type": data.get("type"),
        "set_code": data.get("code"), "release_date": data.get("releaseDate"), "source": data.get("source"),
        "commanders": list(dict.fromkeys(c["name"] for c in data.get("commander") or [])),
        "cards": list(cards.values()),
    }


def build_table(decks: Iterable[tuple[str, dict[str, Any]]]) -> pl.DataFrame:
    rows = [r for f, d in decks if (r := deck_record(f, d)) is not None]
    return pl.DataFrame(rows, schema=SCHEMA).sort("release_date", "name", descending=[True, False])


def fingerprint(df: pl.DataFrame) -> str:
    """SHA-256 of the table's rows as canonical JSON, in ``file`` order: changes only when a list changes."""
    h = hashlib.sha256()
    for row in df.sort("file").iter_rows(named=True):
        h.update(json.dumps(row, sort_keys=True, default=str).encode())
        h.update(b"\n")
    return h.hexdigest()


def archive_meta(path: str | Path) -> dict[str, Any]:
    """MTGJSON's build ``{date, version}``, from the first deck file in the archive."""
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".json"):
                fh = tf.extractfile(m)
                if fh is not None:
                    return dict(json.load(fh).get("meta") or {})
    return {}


def read_archive(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """(file name, Deck) for every deck file in MTGJSON's AllDeckFiles archive."""
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".json"):
                fh = tf.extractfile(m)
                if fh is not None:
                    yield Path(m.name).stem, json.load(fh)["data"]


def fetch_archive(dest: str | Path, client: Any = None, max_age_days: float = 6.0) -> Path:
    """MTGJSON's AllDeckFiles.tar.gz in ``dest``: reused while younger than ``max_age_days``, then re-requested
    with If-Modified-Since (a 304 keeps the file). Verified against MTGJSON's ``.sha256``."""
    import httpx

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / ARCHIVE
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return path
    own = client is None
    client = client or httpx.Client(timeout=600, follow_redirects=True)
    headers = {"User-Agent": USER_AGENT}
    if path.exists():
        headers["If-Modified-Since"] = formatdate(path.stat().st_mtime, usegmt=True)
    try:
        with client.stream("GET", MTGJSON + ARCHIVE, headers=headers) as r:
            if r.status_code == 304:
                path.touch()
                return path
            r.raise_for_status()
            part = path.with_suffix(".part")
            h = hashlib.sha256()
            with part.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
                    h.update(chunk)
        want = client.get(MTGJSON + ARCHIVE + ".sha256", headers={"User-Agent": USER_AGENT})
        want.raise_for_status()
        if want.text.split()[0].strip().lower() != h.hexdigest():
            part.unlink()
            raise RuntimeError(f"{ARCHIVE}: sha256 does not match MTGJSON's {ARCHIVE}.sha256")
        part.replace(path)  # mtime = download time, the If-Modified-Since date of the next request
        return path
    finally:
        if own:
            client.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m mtg_search.precons", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tarball", type=Path, help=f"a local {ARCHIVE} (default: download it into data/raw/mtgjson/)")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "precons.parquet")
    a = ap.parse_args(argv)
    src = a.tarball or fetch_archive(config.RAW_DIR / "mtgjson")
    df = build_table(read_archive(src))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out, compression="zstd")
    n_cards = int(df["cards"].list.len().sum())
    print(f"{a.out}: {df.height} decks, {n_cards} card rows, {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
