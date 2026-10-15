"""MCP server (stdio): search and look up every Magic: The Gathering card.

Three read-only tools over ``cards.parquet``: ``search_cards`` (semantic search with lightonai/mLateOn,
``mlateon.py``), ``find_similar`` and ``get_card``. Results carry only rules and print facts; nothing is
ranked or shown by popularity or price.

Environment:
  MTG_SEARCH_DATA_DIR        cards/tags/rulings parquet dir (default ./data; /app/data in the image)
  MTG_SEARCH_EMBEDDINGS_DIR  card token embeddings (default /app/embeddings)
  MTG_SEARCH_MODEL_DIR       ONNX query encoder (default /app/model)
  MTG_SEARCH_THREADS         search worker threads (default min(8, cpu count))

stdout carries only MCP messages; logs go to stderr.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
from pydantic import BeforeValidator, Field

from . import config, filters
from .fusion import rrf
from .render import ALLOWED_FIELDS, all_legalities, render_card, role_tags
from .tag_filter import TagFilter

log = logging.getLogger("mtg_search.server")

MAX_LIMIT = 50
DEFAULT_LIMIT = 15
MAX_QUERIES = 4  # EXP-003: 1-4 rules-text queries per need, fused with RRF

# The only cards.parquet columns the server reads: what may be rendered plus identity/filter columns.
# Popularity and prices are not in cards.parquet at all (EDHREC rank lives in the evaluation-only
# data/eval/popularity.parquet, which the image never contains); this allow-list is a second guard.
SERVER_COLUMNS: frozenset[str] = ALLOWED_FIELDS | {"oracle_id", "is_funny", "scryfall_uri", "supertypes", "types", "subtypes"}


def server_columns(available: Any) -> list[str]:
    """``available`` column names restricted to :data:`SERVER_COLUMNS`, in their stored order."""
    return [c for c in available if c in SERVER_COLUMNS]


def _norm_name(s: str) -> str:
    s = s.lower().replace("//", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class CardService:
    """Tool logic, independent of MCP. ``backend`` has ``search(query, k, allowed_rows)`` and optionally
    ``search_many(queries, k, allowed_rows)`` (``mlateon.MLateOn``; a stub in tests)."""

    def __init__(self, cards: pl.DataFrame, backend: Any, *, tags: pl.DataFrame | None = None,
                 rulings_path: Path | None = None):
        self.cards = cards.select(server_columns(cards.columns))
        self.backend = backend
        tags = tags if tags is not None else pl.DataFrame({"slug": [], "card_count": []})
        self.tag_filter = TagFilter.from_frame(tags)
        self.rulings_path = rulings_path
        self._rows: dict[int, dict] = {}

    def row(self, i: int) -> dict:
        r = self._rows.get(i)
        if r is None:
            r = self.cards.row(i, named=True)
            if len(self._rows) < 5000:
                self._rows[i] = r
        return r

    # ----------------------------------------------------------------- names
    @cached_property
    def _name_index(self) -> tuple[list[str], list[int], dict[str, int]]:
        names = self.cards["name"].to_list()
        faces = self.cards["face_names"].to_list() if "face_names" in self.cards.columns else [[] for _ in names]
        funny = self.cards["is_funny"].to_list() if "is_funny" in self.cards.columns else [False] * len(names)
        order = sorted(range(len(names)), key=lambda i: bool(funny[i]))  # real cards win ties over Un-/playtest cards
        keys, rows, exact = [], [], {}
        for i in order:
            for nm in [names[i], *(faces[i] or [])]:
                keys.append(_norm_name(nm))
                rows.append(i)
        # a card's full name beats another card's face name ("Lightning Bolt" is also a face of an MDFC)
        for i in order:
            exact.setdefault(_norm_name(names[i]), i)
        for i in order:
            for nm in faces[i] or []:
                exact.setdefault(_norm_name(nm), i)
        return keys, rows, exact

    def resolve_name(self, name: str) -> tuple[int | None, list[str]]:
        """Row of the best-matching card (exact, then prefix / whole word, then fuzzy) and close alternatives.
        Deterministic and popularity-free: prefix before whole-word, real before Un-cards, then shortest name."""
        from rapidfuzz import fuzz, process

        keys, rows, exact = self._name_index
        q = _norm_name(name)
        if q in exact:
            return exact[q], []
        names = self.cards["name"]
        if len(q) >= 3:
            pat = re.compile(r"(^|\s)" + re.escape(q) + r"(\s|$)")
            funny = self.cards["is_funny"].to_list() if "is_funny" in self.cards.columns else None
            best: dict[int, tuple] = {}
            for i, k in enumerate(keys):
                tier = 0 if k.startswith(q) else 1 if pat.search(k) else None
                if tier is not None:
                    key = (tier, bool(funny and funny[rows[i]]), len(k), k, rows[i])
                    best[rows[i]] = min(best.get(rows[i], key), key)
            if best:
                order = sorted(best, key=best.__getitem__)
                return order[0], [names[r] for r in order[1:4]]
        ranked, seen = [], set()
        for _k, score, pos in process.extract(q, keys, scorer=fuzz.WRatio, limit=12):
            if rows[pos] not in seen:
                seen.add(rows[pos])
                ranked.append((rows[pos], score))
        if not ranked or ranked[0][1] < 70:
            return None, [names[r] for r, _ in ranked[:5]]
        return ranked[0][0], [names[r] for r, s in ranked[1:4] if s >= 70]

    def _not_found(self, name: str, alts: list[str]) -> str:
        return f'No card named "{name}". Closest names: {", ".join(alts) or "none"}. Retry with one of them.'

    # ----------------------------------------------------------------- shared
    @staticmethod
    def _filters(color_identity, format, types, mana_value_min, mana_value_max) -> dict:
        f = {"color_identity": color_identity or None, "format": format or None, "types": types or None,
             "cmc_min": mana_value_min, "cmc_max": mana_value_max}
        return {k: v for k, v in f.items() if v is not None}

    @staticmethod
    def _describe(f: dict) -> str:
        names = {"cmc_min": "mana_value_min", "cmc_max": "mana_value_max"}
        return ", ".join(f"{names.get(k, k)}={v}" for k, v in f.items()) or "none"

    def _results(self, head: str, hits: list[tuple[int, float]], n_allowed: int, fmt: str | None) -> str:
        lines = [f"{head} · {len(hits)} shown of {n_allowed:,} matching cards"]
        lines += [render_card(self.row(r), rank=i, fmt=fmt, tag_counts=self.tag_filter) for i, (r, _s) in enumerate(hits, 1)]
        return "\n".join(lines)

    # ----------------------------------------------------------------- tools
    def search_cards(self, queries: list[str], color_identity: str | None = None, format: str | None = None,
                     types: list[str] | None = None, mana_value_min: float | None = None,
                     mana_value_max: float | None = None, limit: int = DEFAULT_LIMIT) -> str:
        # Only what the caller sends is searched; the user's own wording is never added (EXP-003).
        qs = list(dict.fromkeys(q.strip() for q in ([queries] if isinstance(queries, str) else queries or []) if q and q.strip()))
        if not qs:
            raise ValueError('give at least one query, e.g. queries=["Destroy target creature with flying."]')
        if len(qs) > MAX_QUERIES:
            raise ValueError(f"at most {MAX_QUERIES} queries per call (got {len(qs)}); keep the ones that describe the most different ways to meet the need")
        limit = max(1, min(MAX_LIMIT, int(limit)))
        f = self._filters(color_identity, format, types, mana_value_min, mana_value_max)
        t0 = time.perf_counter()
        allowed = filters.allowed_indices(self.cards, **f)
        head = f'search: "{" | ".join(qs)}" · filters: {self._describe(f)}'
        if allowed.size == 0:
            return f"{head} · no card matches these filters; loosen them."
        if len(qs) == 1:
            hits = self.backend.search(qs[0], limit, allowed)
        else:  # each query ranks separately; the rankings are fused with RRF (k=60, equal weights)
            depth = max(50, 2 * limit)
            many = getattr(self.backend, "search_many", None)
            lists = many(qs, depth, allowed) if many else [self.backend.search(q, depth, allowed) for q in qs]
            hits = [(int(r), s) for r, s in rrf(lists, limit=limit)]
        log.info("search_cards %r %s -> %d hits in %.0f ms", qs, f, len(hits), (time.perf_counter() - t0) * 1000)
        return self._results(head, hits, allowed.size, f.get("format"))

    def find_similar(self, card_name: str, color_identity: str | None = None, format: str | None = None,
                     types: list[str] | None = None, mana_value_min: float | None = None,
                     mana_value_max: float | None = None, limit: int = DEFAULT_LIMIT) -> str:
        row, alts = self.resolve_name(card_name)
        if row is None:
            return self._not_found(card_name, alts)
        card = self.row(row)
        f = self._filters(color_identity, format, types, mana_value_min, mana_value_max)
        allowed = filters.allowed_indices(self.cards, **f)
        allowed = allowed[(self.cards["name"].to_numpy()[allowed] != card["name"])]  # leave the card itself out
        head = f"similar to: {card['name']} · filters: {self._describe(f)}"
        if allowed.size == 0:
            return f"{head} · no card matches these filters; loosen them."
        hits = self.backend.search(card["doc_text"], max(1, min(MAX_LIMIT, int(limit))), allowed)
        return self._results(head, hits, allowed.size, f.get("format"))

    def rulings(self, oracle_id: str) -> list[tuple[str, str]]:
        if self.rulings_path is None or not Path(self.rulings_path).exists():
            return []
        df = pl.scan_parquet(self.rulings_path).filter(pl.col("oracle_id") == oracle_id).sort("published_at").collect()
        return [(str(d), c) for d, c in zip(df["published_at"].to_list(), df["comment"].to_list(), strict=True)]

    def get_card(self, name: str) -> str:
        row, alts = self.resolve_name(name)
        if row is None:
            return self._not_found(name, alts)
        c = self.row(row)
        out = [render_card(c, detail="full", tag_counts=self.tag_filter)]
        if alts:
            out.append("   (other close names: " + ", ".join(alts) + ")")
        out += ["legality:", all_legalities(c.get("legalities"))]
        roles = role_tags(c.get("tags"), self.tag_filter, n=30)
        if roles:
            out.append("roles: " + ", ".join(roles))
        rul = self.rulings(c["oracle_id"])
        out += [f"rulings ({len(rul)}):", *(f"   - {d}: {t}" for d, t in rul)] if rul else ["rulings: none"]
        if c.get("scryfall_uri"):
            out.append(f"scryfall: {c['scryfall_uri']}")
        return "\n".join(out)


# --------------------------------------------------------------------------- what the agent sees
# Measured on the precon benchmark before shipping (EXP-009/011/014): the instructions name the situations that
# call for the tools and end with the "before you answer" sentence that raised tool use from 0% to 40% (ADD) and
# 86% (SWAP); the search_cards description opens with when to use it ("rather than listing cards from memory").
# A rewrite without that opening searched less (EXP-014: 10/30 vs 19/30 dev episodes). search_cards and get_card
# load upfront in Claude Code (alwaysLoad).

INSTRUCTIONS = """Magic: The Gathering (MTG) card search and lookup over every card's current official rules text (Scryfall data, which can include cards printed after your training data). Use these tools whenever a task involves choosing, suggesting or checking cards: building, upgrading or cutting a Commander/EDH or other deck, "cards that do X", replacements for a card, or a card's exact text, rulings, colour identity or format legality. Use them even when you feel sure from memory: errata, new cards, and colour-identity or legality slips are the usual errors.
Efficient use: one search_cards call per deck need, with color_identity (and format) set, returns candidates that are already in colour and legal; send 2-4 queries in that call for a broad need. get_card checks one card you named from memory or whose rulings matter. find_similar gives alternatives to a known card, e.g. for a swap. Rankings use rules text only.
Before you answer a deck-building request, use the tools: search for candidate cards and check exact card text, colour identity and legality of your picks."""

SEARCH_DOC = """Search every Magic: The Gathering (MTG) card by what it does: semantic search over the rules text, with filters for Commander colour identity, format legality, card type and mana value. Use it to find candidates for a deck need (removal, ramp, card draw, sacrifice outlets, answers to fliers, ...) rather than listing cards from memory. With color_identity (and format="commander") set, every result is in colour and legal, so no separate check is needed.

Write each query the way the rules text of a matching card reads, not the way a player describes the need: "Destroy target creature with flying." rather than "answers to fliers". For a broad need, send 2-4 queries that each describe a different way to meet it, e.g. ["Reach", "Destroy target creature with flying.", "Creatures you control gain reach until end of turn."]; their rankings are merged. Put colour, format, type and mana-value limits in the filters, not in the queries.

Returns up to `limit` cards, best first, ranked only by how well their text matches your queries: name, mana cost and mana value, colours, type line, power/toughness, rules text (long text is shortened; get_card shows the full card), roles (what the card does in a deck, e.g. removal, ramp) and legality. Filters apply before ranking, so even narrow filters return the best matching cards. 15 cards is about 2,000 tokens."""

SIMILAR_DOC = """Find Magic: The Gathering cards that do what a given card does: replacements, upgrades or redundancy. The named card's rules text is the query and the card itself is left out. Use it for "more cards like X", an in-colour or format-legal stand-in for a card, or alternatives in a swap, e.g. card_name="Divination", color_identity="BR", format="commander". Same filters and output as search_cards."""

GET_DOC = """Show one Magic: The Gathering card in full, by name (fuzzy: "fire ice", "jace mind sculptor" or one face of a double-faced card all work). Returns the complete current rules text of every face, mana cost and mana value, type line, power/toughness, colours and colour identity, legality in every format, markers (Game Changer, Reserved, Commander-eligible), roles, official rulings and the Scryfall link. Use it before recommending a card you know only from memory, or when exact wording, a ruling or legality decides the answer. Cards from a search_cards call with color_identity and format set need no second check."""


def _as_list(v: Any) -> Any:
    return [v] if isinstance(v, str) else v


Queries = Annotated[list[str], BeforeValidator(_as_list), Field(
    min_length=1, max_length=MAX_QUERIES,
    description=f"1-{MAX_QUERIES} queries, each worded like card rules text and each a different way to meet the need. A single string also works.")]
ColorIdentity = Annotated[str | None, Field(
    description="Only cards whose colour identity fits within these colours, e.g. 'BG'; 'C' = colourless only. For a Commander deck, use the commander's colours.")]
Format = Annotated[str | None, Field(description="Only cards legal in this format, e.g. 'commander', 'modern', 'pauper'.")]
Types = Annotated[list[str] | None, Field(
    description="Every entry must match a supertype, card type or subtype, e.g. ['creature', 'elf'] or ['legendary'].")]
MvMin = Annotated[float | None, Field(description="Minimum mana value.")]
MvMax = Annotated[float | None, Field(description="Maximum mana value.")]
Limit = Annotated[int, Field(ge=1, le=MAX_LIMIT, description=f"Number of cards to return (1-{MAX_LIMIT}).")]


def build_mcp(service: CardService, surface: dict[str, str] | None = None):
    """The MCP server. ``surface`` overrides the model-facing text for A/B tests on the precon benchmark
    (``bench/precon/harness/surfaces.py``): keys ``instructions``, ``search_cards``, ``find_similar``, ``get_card``."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from . import __version__

    surface = surface or {}

    def guarded(fn):
        def call(**kw):
            try:
                return fn(**kw)
            except ValueError as e:  # bad arguments: tell the model how to fix the call
                raise ToolError(str(e)) from e
        return call

    def search_cards(queries: Queries, color_identity: ColorIdentity = None, format: Format = None, types: Types = None,
                     mana_value_min: MvMin = None, mana_value_max: MvMax = None, limit: Limit = DEFAULT_LIMIT) -> str:
        return guarded(service.search_cards)(queries=queries, color_identity=color_identity, format=format, types=types,
                                             mana_value_min=mana_value_min, mana_value_max=mana_value_max, limit=limit)

    def find_similar(card_name: Annotated[str, Field(description="Name of the card to match (fuzzy).")],
                     color_identity: ColorIdentity = None, format: Format = None, types: Types = None,
                     mana_value_min: MvMin = None, mana_value_max: MvMax = None, limit: Limit = DEFAULT_LIMIT) -> str:
        return guarded(service.find_similar)(card_name=card_name, color_identity=color_identity, format=format, types=types,
                                             mana_value_min=mana_value_min, mana_value_max=mana_value_max, limit=limit)

    def get_card(name: Annotated[str, Field(description="Card name (fuzzy).")]) -> str:
        return guarded(service.get_card)(name=name)

    mcp = MCPServer("mtg-search", title="MTG card search", version=__version__,
                    instructions=surface.get("instructions", INSTRUCTIONS))
    # Claude Code defers MCP tools behind ToolSearch by default; without alwaysLoad the agent rarely loads them
    # for a task it thinks it can answer from memory (EXP-009: 5% vs 45% of episodes with a tool call).
    always = {"anthropic/alwaysLoad": True}
    for fn, title, doc, meta in ((search_cards, "Search cards", SEARCH_DOC, always), (get_card, "Get card", GET_DOC, always),
                                 (find_similar, "Find similar cards", SIMILAR_DOC, None)):
        mcp.add_tool(fn, title=title, description=surface.get(fn.__name__, doc), structured_output=False, meta=meta,
                     annotations=ToolAnnotations(title=title, read_only_hint=True, destructive_hint=False,
                                                 idempotent_hint=True, open_world_hint=False))
    return mcp


def load_service(backend: Any = None) -> tuple[CardService, Any]:
    data = config.DATA_DIR
    cards = pl.read_parquet(data / "cards.parquet", columns=server_columns(pl.read_parquet_schema(data / "cards.parquet")))
    tp = data / "tags.parquet"
    tags = pl.read_parquet(tp, columns=["slug", "card_count", "parent_slugs"]) if tp.exists() else None
    if backend is None:
        from .mlateon import MLateOn

        backend = MLateOn(cards["oracle_id"].to_list(),
                          embeddings_dir=os.environ.get("MTG_SEARCH_EMBEDDINGS_DIR", "/app/embeddings"),
                          model_dir=os.environ.get("MTG_SEARCH_MODEL_DIR", "/app/model"))
    return CardService(cards, backend, tags=tags, rulings_path=data / "rulings.parquet"), backend


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("MTG_SEARCH_LOG", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    def _term(signum, _frame):  # PID 1 in Docker: exit cleanly on SIGTERM
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    service, backend = load_service()
    try:
        build_mcp(service).run("stdio")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
