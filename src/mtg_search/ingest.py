"""Scryfall bulk data -> data/cards.parquet, tags.parquet, rulings.parquet, manifest.json, plus the
evaluation-only data/eval/popularity.parquet (EDHREC rank; never part of the shipped tables).

Usage::

    python -m mtg_search.ingest [--force]

The Scryfall bulk-data endpoint exposes ``jsonl_download_uri`` (gzipped JSONL)
for each bulk type. Downloads are skipped when the ``updated_at`` of the
``oracle_cards`` bulk object matches the existing manifest, unless ``--force``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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


PIP_COLORS = "WUBRGC"
_SYMBOL_RE = re.compile(r"\{([^}]*)\}")
_X_SYMBOLS = frozenset({"X", "Y", "Z"})


def parse_mana_cost(cost: str | None) -> dict[str, Any]:
    """Parse one face's mana cost (e.g. ``{1}{U}{U}``) into symbol counts.

    Pips use *devotion* semantics: every symbol containing a colour counts once for
    that colour, so hybrid ``{G/W}`` adds to both G and W, ``{2/W}`` and Phyrexian
    ``{W/P}`` add to W. ``{C}`` is counted as ``pips_C``. ``generic`` sums plain
    numeric symbols; ``{X}``/``{Y}``/``{Z}`` set ``has_x``.
    """
    out: dict[str, Any] = {f"pips_{c}": 0 for c in PIP_COLORS}
    out.update(generic=0, has_x=False, has_hybrid=False, has_phyrexian=False)
    for sym in _SYMBOL_RE.findall(cost or ""):
        parts = sym.upper().split("/")
        if len(parts) == 1:
            p = parts[0]
            if p.isdigit():
                out["generic"] += int(p)
            elif p in _X_SYMBOLS:
                out["has_x"] = True
            elif p in PIP_COLORS:
                out[f"pips_{p}"] += 1
            continue
        if "P" in parts:
            out["has_phyrexian"] = True
        if len([q for q in parts if q != "P"]) > 1:
            out["has_hybrid"] = True
        for q in parts:
            if q in PIP_COLORS:
                out[f"pips_{q}"] += 1
    return out


def face_costs(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-face parsed mana costs (one entry for single-faced cards)."""
    face_list = card.get("card_faces") or [card]
    return [
        {"name": f.get("name", ""), "mana_cost": f.get("mana_cost") or "", **parse_mana_cost(f.get("mana_cost"))}
        for f in face_list
    ]


def mana_columns(card: dict[str, Any]) -> dict[str, Any]:
    """Card-level cost columns: per-colour pips / generic are the max over faces,
    flags are true if any face has them; ``face_costs`` keeps the per-face detail."""
    fc = face_costs(card)
    cols: dict[str, Any] = {f"pips_{c}": max(f[f"pips_{c}"] for f in fc) for c in PIP_COLORS}
    cols["generic"] = max(f["generic"] for f in fc)
    for flag in ("has_x", "has_hybrid", "has_phyrexian"):
        cols[flag] = any(f[flag] for f in fc)
    cols["face_costs"] = fc
    return cols


def _type_line(card: dict[str, Any]) -> str:
    if card.get("type_line"):
        return card["type_line"]
    return " // ".join(f.get("type_line") or "" for f in card.get("card_faces") or [])


# --------------------------------------------------------------------------- deck-building fields

PERMANENT_TYPES = frozenset({"Artifact", "Creature", "Enchantment", "Land", "Planeswalker", "Battle"})
_PARTNER_VARIANT_RE = re.compile(r"^Partner\s*[\u2014-]\s*([^(\n]+?)\s*(?:\(|$)", re.MULTILINE)


def _face_is_permanent(type_line: str | None) -> bool:
    _, typ, _ = parse_type_line(type_line or "")
    return bool(PERMANENT_TYPES.intersection(typ))


def _opt_str(val: Any) -> str | None:
    return None if val is None else str(val)


def face_details(card: dict[str, Any]) -> list[dict[str, Any]]:
    """One struct per face (a single-faced card has one) so callers can render each
    face without parsing ``doc_text``. P/T, loyalty and defense stay strings
    (``*``, ``1+*``, ``X`` occur)."""
    return [
        {
            "name": f.get("name", ""),
            "mana_cost": f.get("mana_cost") or "",
            "type_line": f.get("type_line") or "",
            "oracle_text": f.get("oracle_text") or "",
            "power": _opt_str(f.get("power")),
            "toughness": _opt_str(f.get("toughness")),
            "loyalty": _opt_str(f.get("loyalty")),
            "defense": _opt_str(f.get("defense")),
            "is_permanent": _face_is_permanent(f.get("type_line")),
        }
        for f in faces(card)
    ]


def _all_text(card: dict[str, Any]) -> str:
    parts = [card.get("oracle_text") or ""] + [f.get("oracle_text") or "" for f in card.get("card_faces") or []]
    return "\n".join(p for p in parts if p)


def partner_kind(card: dict[str, Any], type_line: str | None = None) -> str | None:
    """How a card pairs as a second commander, or ``None``.

    Values: ``"partner"``, ``"partner with"`` (named pair), ``"friends forever"``,
    other ``"partner—<group>"`` variants (``character select``, ``survivors``,
    ``father & son``), ``"choose a background"``, ``"background"`` (a Background that
    can be the second commander), ``"doctor's companion"`` and ``"doctor"`` (a legendary
    Time Lord Doctor creature that pairs with a companion).
    """
    text = _all_text(card)
    kws = {k.lower() for k in card.get("keywords") or []}
    m = _PARTNER_VARIANT_RE.search(text)
    if m:
        group = m.group(1).strip().lower()
        return "friends forever" if group == "friends forever" else f"partner\u2014{group}"
    if "partner with" in kws:
        return "partner with"
    if "partner" in kws:
        return "partner"
    if "friends forever" in kws or re.search(r"^Friends forever\b", text, re.MULTILINE):
        return "friends forever"
    if "choose a background" in kws:
        return "choose a background"
    if "doctor's companion" in kws:
        return "doctor's companion"
    sup, typ, sub = parse_type_line(type_line if type_line is not None else _type_line(card))
    if "Background" in sub and "Legendary" in sup:
        return "background"
    if "Legendary" in sup and "Creature" in typ and "Doctor" in sub:
        return "doctor"
    return None


def can_be_commander(card: dict[str, Any]) -> bool:
    """Legendary creature on the front face, or text saying it "can be your commander".

    Backgrounds (second commander only via "Choose a Background") are not counted;
    see ``partner_kind``.
    """
    front = faces(card)[0].get("type_line") or _type_line(card).split(" // ")[0]
    sup, typ, _ = parse_type_line(front)
    if "Legendary" in sup and "Creature" in typ:
        return True
    return "can be your commander" in _all_text(card)


def deck_columns(card: dict[str, Any], type_line: str, supertypes: list[str]) -> dict[str, Any]:
    """Fields a deck-building agent needs.

    ``rarity`` comes from the single representative printing that the ``oracle_cards``
    bulk file picks for each card; it is informational (pauper legality already
    accounts for any common printing). Prices are deliberately not stored: they change
    daily and the server never uses them. ``produced_mana`` is Scryfall's card-level
    list (it is never given per face), e.g. ``["C"]`` for Sol Ring.
    """
    fd = face_details(card)
    front = fd[0]
    produced = sorted(set(card.get("produced_mana") or []), key=lambda c: (PIP_COLORS.find(c) % 99, c))
    return {
        "rarity": card.get("rarity"),
        "produced_mana": produced,
        "is_permanent": any(f["is_permanent"] for f in fd),
        "is_legendary": "Legendary" in supertypes,
        "can_be_commander": can_be_commander(card),
        "partner_kind": partner_kind(card, type_line),
        "power": front["power"],
        "toughness": front["toughness"],
        "loyalty": front["loyalty"],
        "defense": front["defense"],
        "faces": fd,
        "reserved": bool(card.get("reserved", False)),
    }


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
        **mana_columns(card),
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
        **deck_columns(card, type_line, sup),
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
    **{f"pips_{c}": pl.Int16 for c in PIP_COLORS},
    "generic": pl.Int32,
    "rarity": pl.String,
    "produced_mana": pl.List(pl.String),
    "partner_kind": pl.String,
    "power": pl.String,
    "toughness": pl.String,
    "loyalty": pl.String,
    "defense": pl.String,
    "faces": pl.List(
        pl.Struct(
            {
                "name": pl.String,
                "mana_cost": pl.String,
                "type_line": pl.String,
                "oracle_text": pl.String,
                "power": pl.String,
                "toughness": pl.String,
                "loyalty": pl.String,
                "defense": pl.String,
                "is_permanent": pl.Boolean,
            }
        )
    ),
    "face_costs": pl.List(
        pl.Struct(
            {
                "name": pl.String,
                "mana_cost": pl.String,
                **{f"pips_{c}": pl.Int16 for c in PIP_COLORS},
                "generic": pl.Int32,
                "has_x": pl.Boolean,
                "has_hybrid": pl.Boolean,
                "has_phyrexian": pl.Boolean,
            }
        )
    ),
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


# --------------------------------------------------------------------------- fingerprint
#
# Deterministic content hashes written to manifest.json. scripts/refresh.py compares them
# with the published manifest to decide what a new Scryfall snapshot requires:
#   text_hash changed            -> re-encode (embeddings are row-aligned to doc_text)
#   gameplay/aux_hash changed    -> publish new tables only (no re-encode)
#   neither                      -> nothing to publish (only volatile fields changed)
# Every cards.parquet column is in exactly one of the three groups below (enforced by
# tests/test_ingest.py), so a new column has to be classified explicitly.

FINGERPRINT_VERSION = 1

# Re-encoding is needed iff this changes: the embedded text and the row order.
TEXT_HASH_COLUMNS: tuple[str, ...] = ("oracle_id", "doc_text")

# Rules facts a deck builder relies on (legalities/bans, tags, costs, types, stats, ...).
GAMEPLAY_HASH_COLUMNS: tuple[str, ...] = (
    "oracle_id",
    "name",
    "face_names",
    "faces",
    "mana_cost",
    "face_costs",
    *(f"pips_{c}" for c in PIP_COLORS),
    "generic",
    "has_x",
    "has_hybrid",
    "has_phyrexian",
    "cmc",
    "colors",
    "color_identity",
    "type_line",
    "supertypes",
    "types",
    "subtypes",
    "keywords",
    "legalities",
    "game_changer",
    "reserved",
    "layout",
    "produced_mana",
    "is_funny",
    "is_permanent",
    "is_legendary",
    "can_be_commander",
    "partner_kind",
    "power",
    "toughness",
    "loyalty",
    "defense",
    "tags",
)

# Not hashed: change without any rules change (links, and the attributes of whichever printing
# Scryfall picks as representative).
VOLATILE_COLUMNS: tuple[str, ...] = (
    "image_uri",
    "scryfall_uri",
    "set",
    "set_type",
    "released_at",
    "rarity",
)


# Crowd/popularity data: extracted by build_cards, then split off by split_eval_only into
# eval/popularity.parquet for offline evaluation (popularity baseline, popularity-drift metric, task
# stratification). Never in cards.parquet, so the shipped server and image cannot contain it.
EVAL_ONLY_COLUMNS: tuple[str, ...] = ("edhrec_rank",)


def split_eval_only(cards_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """-> (shipped cards table without EVAL_ONLY_COLUMNS, eval-only table keyed by oracle_id)."""
    return cards_df.drop(EVAL_ONLY_COLUMNS), cards_df.select("oracle_id", *EVAL_ONLY_COLUMNS)


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def frame_hash(df: pl.DataFrame, columns: Iterable[str] | None = None, *, ordered: bool = True) -> str:
    """SHA-256 over the rows of ``df[columns]`` as canonical JSON, one line per row.

    ``ordered=False`` sorts the row lines first (for tables whose row order carries no
    meaning and may tie, e.g. rulings on the same date)."""
    if columns is not None:
        df = df.select(list(columns))
    lines = (_row_json(r) for r in df.iter_rows(named=True))
    if not ordered:
        lines = iter(sorted(lines))
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(cards_df: pl.DataFrame, tags_df: pl.DataFrame, rulings_df: pl.DataFrame) -> dict[str, Any]:
    """Manifest fingerprint fields (see the comment block above)."""
    oids = hashlib.sha256()
    for oid in cards_df["oracle_id"]:
        oids.update(oid.encode())
        oids.update(b"\n")
    aux = hashlib.sha256()
    aux.update(frame_hash(tags_df, ordered=False).encode())
    aux.update(frame_hash(rulings_df, ordered=False).encode())
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "oracle_ids_sha256": oids.hexdigest(),
        "text_hash": frame_hash(cards_df, TEXT_HASH_COLUMNS),
        "gameplay_hash": frame_hash(cards_df, GAMEPLAY_HASH_COLUMNS),
        "aux_hash": aux.hexdigest(),
    }


def fingerprint_dir(data_dir: Path) -> dict[str, Any]:
    """:func:`fingerprint` of the parquet files in ``data_dir``."""
    return fingerprint(*(pl.read_parquet(data_dir / f"{n}.parquet") for n in ("cards", "tags", "rulings")))


# --------------------------------------------------------------------------- main


def read_manifest(path: Path = config.MANIFEST_JSON) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_from_raw(raw: dict[str, Path], metas: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    tag_objs = list(iter_jsonl(raw["oracle_tags"]))
    cards_df, popularity_df = split_eval_only(build_cards(iter_jsonl(raw["oracle_cards"]), tags_by_oracle_id(tag_objs)))
    keep_ids = set(cards_df["oracle_id"].to_list())
    tags_df = build_tags(tag_objs, keep_ids)
    rulings_df = build_rulings(iter_jsonl(raw["rulings"]), keep_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    cards_df.write_parquet(out_dir / "cards.parquet", compression="zstd")
    tags_df.write_parquet(out_dir / "tags.parquet", compression="zstd")
    rulings_df.write_parquet(out_dir / "rulings.parquet", compression="zstd")
    (out_dir / "eval").mkdir(exist_ok=True)
    popularity_df.write_parquet(out_dir / "eval" / "popularity.parquet", compression="zstd")

    manifest = {
        "schema_version": config.SCHEMA_VERSION,
        "scryfall_updated_at": metas["oracle_cards"]["updated_at"],
        "bulk": {k: {"updated_at": m["updated_at"], "uri": m["_uri"]} for k, m in metas.items()},
        "card_count": cards_df.height,
        "tag_count": tags_df.height,
        "ruling_count": rulings_df.height,
        **fingerprint(cards_df, tags_df, rulings_df),
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
