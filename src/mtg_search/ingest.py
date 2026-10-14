"""Scryfall bulk data -> data/cards.parquet, tags.parquet, rulings.parquet, manifest.json.

Usage::

    python -m mtg_search.ingest [--force]

The Scryfall bulk-data endpoint exposes ``jsonl_download_uri`` (gzipped JSONL)
for each bulk type. Downloads are skipped when the ``updated_at`` of the
``oracle_cards`` bulk object matches the existing manifest, unless ``--force``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from mtg_search import config
from mtg_search.doctext import doc_text, faces

DROP_LAYOUTS = frozenset(
    {
        "token",
        "double_faced_token",
        "emblem",
        "art_series",
        "planar",
        "scheme",
        "vanguard",
        "front_card",
    }
)
DROP_SET_TYPES = frozenset({"memorabilia"})

SUPERTYPES = frozenset({"Basic", "Legendary", "Snow", "World", "Ongoing", "Host", "Elite"})

# Formats where a card is playable.
PLAYABLE = ("legal", "restricted")


# --------------------------------------------------------------------------- download


def _headers() -> dict[str, str]:
    return {"User-Agent": config.USER_AGENT, "Accept": "application/json"}


def fetch_bulk_meta(client: httpx.Client, bulk_type: str) -> dict[str, Any]:
    resp = client.get(f"{config.SCRYFALL_API}/bulk-data/{bulk_type}")
    resp.raise_for_status()
    meta = resp.json()
    uri = meta.get("jsonl_download_uri") or meta.get("download_uri")
    if not uri:
        raise RuntimeError(f"bulk object {bulk_type} has no download uri: {meta}")
    meta["_uri"] = uri
    return meta


def stamp(updated_at: str) -> str:
    """'2026-10-14T09:01:55.948+00:00' -> '20261014T090155' (filesystem-safe)."""
    dt = datetime.fromisoformat(updated_at)
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%S")


def download(client: httpx.Client, uri: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", uri, headers={"Accept": "*/*"}) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- normalize


def keep_card(card: dict[str, Any]) -> bool:
    return card.get("layout") not in DROP_LAYOUTS and card.get("set_type") not in DROP_SET_TYPES


def is_funny(card: dict[str, Any]) -> bool:
    """Un-set / playtest cards that are not tournament legal.

    Some cards from ``set_type == "funny"`` sets (e.g. Unfinity non-acorn cards) are
    eternal-legal, so legality in vintage decides.
    """
    if card.get("set_type") != "funny":
        return False
    return card.get("legalities", {}).get("vintage") not in PLAYABLE


def parse_type_line(type_line: str) -> tuple[list[str], list[str], list[str]]:
    """Return (supertypes, types, subtypes) for a type line, handling ' // ' faces."""
    sup: list[str] = []
    typ: list[str] = []
    sub: list[str] = []

    def add(dst: list[str], item: str) -> None:
        if item and item not in dst:
            dst.append(item)

    for part in (type_line or "").split(" // "):
        left, _, right = part.partition("—")
        for word in left.split():
            add(sup if word in SUPERTYPES else typ, word)
        for word in right.split():
            add(sub, word)
    return sup, typ, sub


def _union_colors(card: dict[str, Any]) -> list[str]:
    if card.get("colors") is not None:
        return list(card["colors"])
    seen: list[str] = []
    for f in card.get("card_faces") or []:
        for c in f.get("colors") or []:
            if c not in seen:
                seen.append(c)
    order = "WUBRG"
    return sorted(seen, key=order.index)


def _image_uri(card: dict[str, Any]) -> str | None:
    if card.get("image_uris"):
        return card["image_uris"].get("normal")
    for f in card.get("card_faces") or []:
        if f.get("image_uris"):
            return f["image_uris"].get("normal")
    return None


def _mana_cost(card: dict[str, Any]) -> str:
    if card.get("mana_cost") is not None:
        return card["mana_cost"]
    # Transform/MDFC: top-level cost absent; back faces often have no cost.
    costs = [f.get("mana_cost") for f in card.get("card_faces") or [] if f.get("mana_cost")]
    return " // ".join(costs)


def _type_line(card: dict[str, Any]) -> str:
    if card.get("type_line"):
        return card["type_line"]
    return " // ".join(f.get("type_line") or "" for f in card.get("card_faces") or [])


def normalize_card(card: dict[str, Any], tags: list[str] | None = None) -> dict[str, Any]:
    type_line = _type_line(card)
    sup, typ, sub = parse_type_line(type_line)
    face_list = faces(card)
    return {
        "oracle_id": card["oracle_id"],
        "name": card["name"],
        "face_names": [f.get("name", "") for f in face_list],
        "doc_text": doc_text(card),
        "mana_cost": _mana_cost(card),
        "cmc": float(card.get("cmc") or 0.0),
        "colors": _union_colors(card),
        "color_identity": list(card.get("color_identity") or []),
        "type_line": type_line,
        "supertypes": sup,
        "types": typ,
        "subtypes": sub,
        "keywords": list(card.get("keywords") or []),
        "legalities": dict(card.get("legalities") or {}),
        "edhrec_rank": card.get("edhrec_rank"),
        "game_changer": bool(card.get("game_changer", False)),
        "is_funny": is_funny(card),
        "layout": card.get("layout"),
        "set": card.get("set"),
        "set_type": card.get("set_type"),
        "released_at": card.get("released_at"),
        "scryfall_uri": card.get("scryfall_uri"),
        "image_uri": _image_uri(card),
        "tags": sorted(tags or []),
    }


CARD_SCHEMA_OVERRIDES = {
    "face_names": pl.List(pl.String),
    "colors": pl.List(pl.String),
    "color_identity": pl.List(pl.String),
    "supertypes": pl.List(pl.String),
    "types": pl.List(pl.String),
    "subtypes": pl.List(pl.String),
    "keywords": pl.List(pl.String),
    "tags": pl.List(pl.String),
    "edhrec_rank": pl.Int64,
    "cmc": pl.Float64,
}


def build_cards(
    cards: Iterable[dict[str, Any]], tags_by_oracle: dict[str, list[str]] | None = None
) -> pl.DataFrame:
    tags_by_oracle = tags_by_oracle or {}
    rows = [normalize_card(c, tags_by_oracle.get(c["oracle_id"])) for c in cards if keep_card(c)]
    # legalities: build a struct with a stable, sorted set of format fields.
    formats = sorted({f for r in rows for f in r["legalities"]})
    for r in rows:
        r["legalities"] = {f: r["legalities"].get(f, "not_legal") for f in formats}
    df = pl.DataFrame(rows, schema_overrides=CARD_SCHEMA_OVERRIDES, infer_schema_length=None)
    return df.sort("name")


def build_tags(tag_objs: Iterable[dict[str, Any]], keep_ids: set[str]) -> pl.DataFrame:
    tag_objs = list(tag_objs)
    id_to_slug = {t["id"]: t["slug"] for t in tag_objs}
    rows = []
    for t in tag_objs:
        oids = sorted({g["oracle_id"] for g in t.get("taggings") or [] if g["oracle_id"] in keep_ids})
        rows.append(
            {
                "slug": t["slug"],
                "label": t.get("label"),
                "description": t.get("description"),
                "aliases": list(t.get("aliases") or []),
                "parent_slugs": [id_to_slug[p] for p in t.get("parent_ids") or [] if p in id_to_slug],
                "child_slugs": [id_to_slug[c] for c in t.get("child_ids") or [] if c in id_to_slug],
                "card_count": len(oids),
                "oracle_ids": oids,
            }
        )
    schema = {
        "slug": pl.String,
        "label": pl.String,
        "description": pl.String,
        "aliases": pl.List(pl.String),
        "parent_slugs": pl.List(pl.String),
        "child_slugs": pl.List(pl.String),
        "card_count": pl.Int64,
        "oracle_ids": pl.List(pl.String),
    }
    return pl.DataFrame(rows, schema=schema).sort("slug")


def tags_by_oracle_id(tag_objs: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for t in tag_objs:
        for g in t.get("taggings") or []:
            out[g["oracle_id"]].append(t["slug"])
    return out


def build_rulings(rulings: Iterable[dict[str, Any]], keep_ids: set[str]) -> pl.DataFrame:
    rows = [
        {
            "oracle_id": r["oracle_id"],
            "source": r.get("source"),
            "published_at": r.get("published_at"),
            "comment": r.get("comment"),
        }
        for r in rulings
        if r.get("oracle_id") in keep_ids
    ]
    schema = {"oracle_id": pl.String, "source": pl.String, "published_at": pl.String, "comment": pl.String}
    return (
        pl.DataFrame(rows, schema=schema)
        .with_columns(pl.col("published_at").str.to_date())
        .sort(["oracle_id", "published_at"])
    )


# --------------------------------------------------------------------------- main


def read_manifest(path: Path = config.MANIFEST_JSON) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_from_raw(raw: dict[str, Path], metas: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    tag_objs = list(iter_jsonl(raw["oracle_tags"]))
    cards_df = build_cards(iter_jsonl(raw["oracle_cards"]), tags_by_oracle_id(tag_objs))
    keep_ids = set(cards_df["oracle_id"].to_list())
    tags_df = build_tags(tag_objs, keep_ids)
    rulings_df = build_rulings(iter_jsonl(raw["rulings"]), keep_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    cards_df.write_parquet(out_dir / "cards.parquet", compression="zstd")
    tags_df.write_parquet(out_dir / "tags.parquet", compression="zstd")
    rulings_df.write_parquet(out_dir / "rulings.parquet", compression="zstd")

    manifest = {
        "schema_version": config.SCHEMA_VERSION,
        "scryfall_updated_at": metas["oracle_cards"]["updated_at"],
        "bulk": {k: {"updated_at": m["updated_at"], "uri": m["_uri"]} for k, m in metas.items()},
        "card_count": cards_df.height,
        "tag_count": tags_df.height,
        "ruling_count": rulings_df.height,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run(force: bool = False, data_dir: Path = config.DATA_DIR) -> dict[str, Any]:
    with httpx.Client(headers=_headers(), timeout=httpx.Timeout(30.0, read=300.0), follow_redirects=True) as client:
        metas = {t: fetch_bulk_meta(client, t) for t in config.BULK_TYPES}
        current = read_manifest(data_dir / "manifest.json")
        updated_at = metas["oracle_cards"]["updated_at"]
        if not force and current and current.get("scryfall_updated_at") == updated_at:
            print(f"Up to date (scryfall updated_at {updated_at}); use --force to rebuild.")
            return current

        raw_dir = data_dir / "raw" / stamp(updated_at)
        raw: dict[str, Path] = {}
        for t, meta in metas.items():
            name = re.sub(r"[^\w.-]", "_", Path(httpx.URL(meta["_uri"]).path).name) or f"{t}.jsonl.gz"
            dest = raw_dir / name
            if force or not dest.exists():
                print(f"Downloading {t}: {meta['_uri']}")
                download(client, meta["_uri"], dest)
            raw[t] = dest

    manifest = build_from_raw(raw, metas, data_dir)
    # Keep only the newest raw snapshot.
    for old in (data_dir / "raw").iterdir():
        if old.is_dir() and old != raw_dir:
            shutil.rmtree(old)
    print(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if Scryfall data is unchanged")
    args = ap.parse_args(argv)
    run(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
