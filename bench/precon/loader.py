"""EXP-007 evaluation side: dataset loading and the card database used by the scorer/harness.

The dataset contract is ``bench/precon/schema.md`` (configs ``decks``, ``labels``, ``precons``,
``commanders``, ``splits`` plus ``manifest.json`` / ``tag_denylist.json``). It is read either from a
local build directory (``data/bench/precon/<snapshot_date>/``) or from the private HF dataset
``kasper-cbx/mtg-precon-bench`` (``hf:<repo>[@revision]``).

:class:`CardDB` is the oracle-level card lookup (``data/cards.parquet``) the scorer validates
against: exact name resolution, colour identity, commander legality, Game Changer flag, land /
basic / any-number flags, mana value, produced mana, pips and the oracle-tag role closure used
for diagnostics. ``edhrec_rank`` is loaded for *offline* analysis only (popularity-drift metric
and the popularity-oracle reference); it is never rendered for an agent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

HF_REPO = "kasper-cbx/mtg-precon-bench"
CONFIGS = ("decks", "labels", "precons", "commanders", "splits")

# Diagnostic roles (plan §6 structural diagnostics) -> tag roots; a card has the role if any of
# its oracle tags is a root or a descendant of one (closure over tags.parquet parent_slugs).
DIAG_ROLES: dict[str, tuple[str, ...]] = {
    "ramp": ("ramp", "mana-rock", "mana-dork", "land-ramp", "ritual", "cost-reducer"),
    "draw": ("card-advantage", "draw", "draw-engine", "repeatable-card-advantage"),
    "removal": ("spot-removal", "removal-creature", "removal-artifact", "removal-enchantment",
                "removal-planeswalker", "removal-exile", "removal-destroy", "burn-creature"),
    "wipe": ("sweeper",),
}
ROLE_MINIMUMS = {"ramp": 8, "draw": 8, "removal": 6, "wipe": 2}
ANY_NUMBER = re.compile(r"a deck can have any number of cards named", re.IGNORECASE)


def norm_name(s: str) -> str:
    """Name key for exact resolution (same normalisation as ``CardService``): lower case,
    ``//`` and punctuation to spaces, whitespace collapsed. Copied cost/type suffixes that agents
    sometimes paste (``Sol Ring {1}`` / ``Sol Ring — Artifact``) are stripped first."""
    s = str(s)
    s = re.sub(r"\s*\{.*$", "", s)
    s = s.split(" — ")[0]
    s = s.lower().replace("//", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Card:
    oracle_id: str
    name: str
    color_identity: frozenset[str]
    commander_legal: bool
    game_changer: bool
    is_land: bool
    is_basic: bool
    any_number: bool
    cmc: float
    type_line: str
    produced_mana: tuple[str, ...] = ()
    pips: dict[str, int] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    edhrec_rank: int | None = None  # OFFLINE ONLY (popularity drift / oracle reference)
    is_funny: bool = False


class CardDB:
    def __init__(self, cards: pl.DataFrame, tags: pl.DataFrame | None = None):
        self.cards: dict[str, Card] = {}
        self.by_name: dict[str, str] = {}
        cols = set(cards.columns)
        rows = list(cards.iter_rows(named=True))
        rows.sort(key=lambda r: bool(r.get("is_funny")))  # non-funny cards win name ties
        for r in rows:
            tl = r.get("type_line") or ""
            front = tl.split("//")[0]
            leg = (r.get("legalities") or {}).get("commander") if "legalities" in cols else "legal"
            c = Card(
                oracle_id=r["oracle_id"], name=r["name"],
                color_identity=frozenset(r.get("color_identity") or []),
                commander_legal=leg in ("legal", "restricted"),
                game_changer=bool(r.get("game_changer")),
                is_land="Land" in front,
                is_basic="Basic" in front and "Land" in front,
                any_number=bool(ANY_NUMBER.search(r.get("doc_text") or r.get("oracle_text") or "")),
                cmc=float(r.get("cmc") or 0.0), type_line=tl,
                produced_mana=tuple(r.get("produced_mana") or ()),
                pips={k: int(r.get(f"pips_{k}") or 0) for k in "WUBRG"},
                tags=tuple(r.get("tags") or ()),
                edhrec_rank=r.get("edhrec_rank"),
                is_funny=bool(r.get("is_funny")),
            )
            self.cards[c.oracle_id] = c
            self.by_name.setdefault(norm_name(c.name), c.oracle_id)
        # face names only after every full name: "Lightning Bolt" is also a face of
        # "Emeritus of Conflict // Lightning Bolt" and must resolve to the real Lightning Bolt
        for r in rows:
            for nm in r.get("face_names") or []:
                self.by_name.setdefault(norm_name(nm), r["oracle_id"])
        self.parents: dict[str, tuple[str, ...]] = {}
        if tags is not None and "parent_slugs" in tags.columns:
            self.parents = {s: tuple(p or ()) for s, p in zip(tags["slug"].to_list(), tags["parent_slugs"].to_list(), strict=True)}
        self._anc: dict[str, frozenset[str]] = {}

    @classmethod
    def from_parquet(cls, data_dir: Path = ROOT / "data") -> CardDB:
        # edhrec_rank (offline only) comes from the eval-only popularity file, not the shipped card table
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from eval.popularity import with_popularity

        cards = with_popularity(pl.read_parquet(data_dir / "cards.parquet"), data_dir / "eval" / "popularity.parquet")
        tp = data_dir / "tags.parquet"
        tags = pl.read_parquet(tp, columns=["slug", "parent_slugs"]) if tp.exists() else None
        return cls(cards, tags)

    def __contains__(self, oid: str) -> bool:
        return oid in self.cards

    def __getitem__(self, oid: str) -> Card:
        return self.cards[oid]

    def resolve(self, name: Any) -> str | None:
        """Exact (normalised) name or face-name match -> oracle_id, else None."""
        if not isinstance(name, str) or not name.strip():
            return None
        return self.by_name.get(norm_name(name))

    # ---------------------------------------------------------------- tag closure
    def ancestors(self, slug: str) -> frozenset[str]:
        if slug in self._anc:
            return self._anc[slug]
        seen, stack = {slug}, [slug]
        while stack:
            for p in self.parents.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        self._anc[slug] = frozenset(seen)
        return self._anc[slug]

    def tag_closure(self, oid: str) -> frozenset[str]:
        out: set[str] = set()
        for t in self.cards[oid].tags:
            out |= self.ancestors(t)
        return frozenset(out)

    def diag_roles(self, oid: str) -> frozenset[str]:
        """Diagnostic roles (ramp/draw/removal/wipe) from oracle tags. Lands never count as ramp."""
        clo = self.tag_closure(oid)
        out = {r for r, roots in DIAG_ROLES.items() if clo & set(roots)}
        if self.cards[oid].is_land:
            out.discard("ramp")
        return frozenset(out)

    def core_roles(self, oid: str) -> frozenset[str]:
        """Functional core-role families (``tag_filter.CORE_ROLES``) present in the tag closure;
        used for SWAP pair coherence (diagnostic only)."""
        from mtg_search.tag_filter import CORE_ROLES

        clo = self.tag_closure(oid)
        return frozenset(r for r in CORE_ROLES if r in clo) | (frozenset({"land"}) if self.cards[oid].is_land else frozenset())


# =========================================================================== dataset

@dataclass
class Deck:
    precon_id: str
    commander_oids: list[str]
    commander_names: list[str]
    color_identity: frozenset[str]
    cards: list[dict]  # {oracle_id, name, qty, is_land, is_basic, is_commander}

    @cached_property
    def qty(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cards:
            out[c["oracle_id"]] = out.get(c["oracle_id"], 0) + int(c.get("qty") or 1)
        return out

    @property
    def n_cards(self) -> int:
        return sum(self.qty.values())


@dataclass
class Bench:
    decks: dict[str, Deck]
    labels: dict[str, list[dict]]
    precons: dict[str, dict]
    splits: dict[str, dict]
    commanders: dict[str, list[dict]] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    denylist: dict = field(default_factory=dict)
    revision: str = "unknown"
    source: str = ""

    def precon_ids(self, split: str | None = None, scored_only: bool = True) -> list[str]:
        out = []
        for pid in sorted(self.decks):
            s = self.splits.get(pid, {})
            if split and split != "all" and s.get("split") != split:
                continue
            if scored_only and not s.get("scored", True):
                continue
            out.append(pid)
        return out

    def gold_names(self, splits: Iterable[str]) -> set[str]:
        """Names of every gold-labelled card for precons in ``splits`` (leakage pre-flight)."""
        want = set(splits)
        return {lab["name"] for pid, labs in self.labels.items()
                if self.splits.get(pid, {}).get("split") in want for lab in labs}


def _rows(df: pl.DataFrame) -> list[dict]:
    return df.to_dicts()


def bench_from_frames(decks: pl.DataFrame, labels: pl.DataFrame, precons: pl.DataFrame | None = None,
                      splits: pl.DataFrame | None = None, commanders: pl.DataFrame | None = None,
                      manifest: dict | None = None, denylist: dict | None = None,
                      revision: str = "unknown", source: str = "") -> Bench:
    dk = {}
    for r in _rows(decks):
        dk[r["precon_id"]] = Deck(r["precon_id"], list(r["commander_oracle_ids"]), list(r.get("commander_names") or []),
                                  frozenset(r.get("color_identity") or []), list(r["cards"]))
    lab: dict[str, list[dict]] = {}
    for r in _rows(labels):
        lab.setdefault(r["precon_id"], []).append(r)

    def by_pid(df):
        return {r["precon_id"]: r for r in _rows(df)} if df is not None else {}

    cmd: dict[str, list[dict]] = {}
    if commanders is not None:
        for r in _rows(commanders):
            cmd.setdefault(r["precon_id"], []).append(r)
    return Bench(dk, lab, by_pid(precons), by_pid(splits), cmd, manifest or {}, denylist or {}, revision, source)


def _find(root: Path, stem: str) -> Path | None:
    hits = sorted(p for p in root.rglob("*.parquet") if p.stem == stem or p.parent.name == stem)
    return hits[0] if hits else None


def load_bench(source: str | Path | None = None) -> Bench:
    """``source``: a local build dir, ``hf:<repo>[@rev]``, or None = newest
    ``data/bench/precon/<snapshot>/`` if present, else the HF dataset."""
    if source is None:
        local = sorted((ROOT / "data" / "bench" / "precon").glob("*/labels.parquet"))
        source = str(local[-1].parent) if local else f"hf:{HF_REPO}"
    source = str(source)
    if source.startswith("hf:"):
        from huggingface_hub import HfApi, snapshot_download

        spec = source[3:]
        repo, _, rev = spec.partition("@")
        rev = rev or None
        info = HfApi().dataset_info(repo, revision=rev)
        root = Path(snapshot_download(repo, repo_type="dataset", revision=info.sha))
        revision = f"{repo}@{info.sha}"
    else:
        root = Path(source)
        revision = None
    frames = {}
    for cfg in CONFIGS:
        p = _find(root, cfg)
        frames[cfg] = pl.read_parquet(p) if p is not None else None
    if frames["decks"] is None or frames["labels"] is None:
        raise FileNotFoundError(f"no decks/labels parquet under {root}")
    man = next(iter(sorted(root.rglob("manifest.json"))), None)
    manifest = json.loads(man.read_text()) if man else {}
    dl = next(iter(sorted(root.rglob("tag_denylist.json"))), None)
    denylist = json.loads(dl.read_text()) if dl else {}
    if revision is None:
        h = hashlib.sha256(_find(root, "labels").read_bytes()).hexdigest()[:12]
        revision = f"local:{manifest.get('snapshot_date', root.name)}:labels-sha256={h}"
    return bench_from_frames(frames["decks"], frames["labels"], frames["precons"], frames["splits"],
                             frames["commanders"], manifest, denylist, revision, str(root))


def log10_rank(r: int | None) -> float | None:
    return math.log10(r) if r else None
