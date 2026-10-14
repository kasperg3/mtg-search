"""MCP server (stdio) for semantic search over every Magic: The Gathering card.

Retrieval is lightonai/mLateOn late interaction served by NextPlaid (see ``nextplaid.py``);
structured filters come from ``mtg_search.filters`` (``FILTER_PARAMS`` drives the tool
schemas, ``allowed_indices`` the filtering), so new filters there show up here automatically.

Environment:
  MTG_SEARCH_DATA_DIR        cards/tags/rulings parquet dir (default ./data; /app/data in the image)
  MTG_SEARCH_NEXTPLAID_INDEX NextPlaid --index-dir (default /app/index)
  MTG_SEARCH_MODEL_DIR       ONNX query encoder dir (default /app/model)
  NEXTPLAID_URL              use a running NextPlaid instead of spawning one (dev)

stdout carries only MCP messages; all logs go to stderr.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import signal
import sys
import time
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import polars as pl
from pydantic import Field

from . import config, filters
from .fusion import rrf
from .render import ALLOWED_FIELDS, all_legalities, render_card, role_tags
from .tag_filter import TagFilter

log = logging.getLogger("mtg_search.server")

MAX_LIMIT = 50
DEFAULT_LIMIT = 15
MAX_QUERIES = 4  # EXP-003: 1-4 rules-text rewrites, fused with RRF


class Backend:  # protocol: see nextplaid.NextPlaidBackend
    def search(self, query: str, k: int, allowed_rows: np.ndarray | None = None) -> list[tuple[int, float]]:
        raise NotImplementedError


def _norm_name(s: str) -> str:
    s = s.lower().replace("//", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# The only cards.parquet columns the server reads: the rendered allow-list plus columns used
# purely for identity, filtering and the Scryfall link. Soft / market / crowd-derived metrics
# (edhrec_rank, prices, ...) stay in the parquet for offline analysis but are never loaded,
# shown or used for ordering.
INTERNAL_COLUMNS: frozenset[str] = frozenset({
    "oracle_id", "is_funny", "scryfall_uri",
    "supertypes", "types", "subtypes",
    "pips_W", "pips_U", "pips_B", "pips_R", "pips_G", "pips_C",
    "has_x", "has_hybrid", "has_phyrexian",
})
SERVER_COLUMNS: frozenset[str] = ALLOWED_FIELDS | INTERNAL_COLUMNS


def server_columns(available: Any) -> list[str]:
    """``available`` column names restricted to :data:`SERVER_COLUMNS`, in their stored order."""
    return [c for c in available if c in SERVER_COLUMNS]


class CardService:
    """Tool logic, independent of MCP (unit-tested with a fake backend)."""

    def __init__(self, cards: pl.DataFrame, backend: Backend, *, tags: pl.DataFrame | None = None,
                 rulings_path: Path | None = None):
        self.cards = cards.select(server_columns(cards.columns))
        self.backend = backend
        self.tags = tags if tags is not None else pl.DataFrame({"slug": [], "card_count": []})
        self.rulings_path = rulings_path
        self.tag_counts: dict[str, int] = dict(
            zip(self.tags["slug"].to_list(), self.tags["card_count"].to_list(), strict=True)
        )
        self.tag_filter = TagFilter.from_frame(self.tags)
        self._rows_cache: dict[int, dict] = {}

    # ----------------------------------------------------------------- helpers
    def row(self, i: int) -> dict:
        r = self._rows_cache.get(i)
        if r is None:
            r = self.cards.row(i, named=True)
            if len(self._rows_cache) < 5000:
                self._rows_cache[i] = r
        return r

    @cached_property
    def _funny(self) -> list[bool]:
        if "is_funny" not in self.cards.columns:
            return [False] * self.cards.height
        return [bool(x) for x in self.cards["is_funny"].to_list()]

    @cached_property
    def _name_index(self) -> tuple[list[str], list[int], dict[str, int]]:
        keys, rows = [], []
        exact: dict[str, int] = {}
        names = self.cards["name"].to_list()
        faces = (self.cards["face_names"].to_list() if "face_names" in self.cards.columns
                 else [[] for _ in names])
        funny = self._funny
        # non-funny cards first so they win exact-name ties (e.g. playtest reprints)
        order = sorted(range(len(names)), key=lambda i: funny[i])
        for i in order:
            for nm in [names[i], *(faces[i] or [])]:
                k = _norm_name(nm)
                keys.append(k)
                rows.append(i)
                exact.setdefault(k, i)
        return keys, rows, exact

    def resolve_name(self, name: str) -> tuple[int | None, list[tuple[str, float]]]:
        """Row of the best-matching card (exact, then fuzzy) and runner-up names."""
        from rapidfuzz import fuzz, process

        keys, rows, exact = self._name_index
        q = _norm_name(name)
        if q in exact:
            return exact[q], []
        # prefix / whole-word containment before fuzzy: "delver" -> Delver of Secrets.
        # Deterministic, popularity-free order: prefix matches before whole-word matches,
        # non-funny before Un-/playtest cards, then shortest matching name, then alphabetical
        # (then row index).
        if len(q) >= 3:
            pat = re.compile(r"(^|\s)" + re.escape(q) + r"(\s|$)")
            funny = self._funny
            best: dict[int, tuple[int, bool, int, str, int]] = {}
            for i, k in enumerate(keys):
                tier = 0 if k.startswith(q) else 1 if pat.search(k) else None
                if tier is None:
                    continue
                key = (tier, funny[rows[i]], len(k), k, rows[i])
                if rows[i] not in best or key < best[rows[i]]:
                    best[rows[i]] = key
            if best:
                order = sorted(best, key=best.__getitem__)
                return order[0], [(self.cards["name"][r], 100.0) for r in order[1:4]]
        hits = process.extract(q, keys, scorer=fuzz.WRatio, limit=12)
        seen, ranked = set(), []
        for _k, score, pos in hits:
            r = rows[pos]
            if r not in seen:
                seen.add(r)
                ranked.append((r, score))
        if not ranked or ranked[0][1] < 70:
            return None, [(self.cards["name"][r], s) for r, s in ranked[:5]]
        best = ranked[0][0]
        return best, [(self.cards["name"][r], s) for r, s in ranked[1:4] if s >= 70]

    def allowed(self, flt: dict) -> np.ndarray:
        return filters.allowed_indices(self.cards, **flt)

    @staticmethod
    def active_filters(flt: dict) -> dict:
        out = {}
        for k, v in flt.items():
            spec = filters.FILTER_PARAMS.get(k, {})
            if v is None or v == spec.get("default", None) or v in ([], {}):
                continue
            out[k] = v
        return out

    @staticmethod
    def _fmt_filters(active: dict) -> str:
        if not active:
            return "none"
        return ", ".join(f"{k}={v}" for k, v in active.items())

    def _render_hits(self, hits: list[tuple[int, float]], flt: dict, detail: str) -> list[str]:
        fmt = flt.get("format")
        return [
            render_card(self.row(r), rank=i, fmt=fmt, detail=detail, tag_counts=self.tag_filter)
            for i, (r, _s) in enumerate(hits, 1)
        ]

    @staticmethod
    def _limit(limit: int | None) -> int:
        return max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))

    def _clean_filters(self, flt: dict) -> dict:
        flt = {k: v for k, v in flt.items() if k in filters.FILTER_PARAMS}
        for k, v in list(flt.items()):
            if isinstance(v, str) and not v.strip():
                flt[k] = None
        return flt

    # ----------------------------------------------------------------- tools
    def search_cards(self, query: str | None = None, queries: list[str] | None = None,
                     limit: int = DEFAULT_LIMIT, detail: str = "compact", **flt: Any) -> str:
        # Only what the caller sends is searched; the user's raw request is never added (EXP-003).
        qs = [q.strip() for q in ([query] if query else []) + list(queries or []) if q and q.strip()]
        qs = list(dict.fromkeys(qs))
        if not qs:
            raise ValueError("give `query` (a string) or `queries` (a list of strings)")
        if len(qs) > MAX_QUERIES:
            raise ValueError(f"at most {MAX_QUERIES} distinct queries per call (got {len(qs)}); "
                             "keep the 2-4 that describe the most different mechanisms")
        limit = self._limit(limit)
        flt = self._clean_filters(flt)
        t0 = time.perf_counter()
        allowed = self.allowed(flt)
        active = self.active_filters(flt)
        echo = qs[0] if len(qs) == 1 else " | ".join(qs)
        head = f'search: "{echo}" · filters: {self._fmt_filters(active)}'
        if allowed.size == 0:
            return f"{head} · 0 cards match these filters; loosen them."
        if len(qs) == 1:
            hits = self.backend.search(qs[0], limit, allowed)
        else:
            depth = max(50, 2 * limit)
            lists = [self.backend.search(q, depth, allowed) for q in qs]
            hits = [(int(r), s) for r, s in rrf(lists, limit=limit)]  # k=60, equal weights
        ms = (time.perf_counter() - t0) * 1000
        log.info("search_cards %r %s -> %d hits in %.0f ms", qs, active, len(hits), ms)
        lines = [f"{head} · {len(hits)} shown of {allowed.size:,} matching filters"]
        lines += self._render_hits(hits, flt, detail)
        return "\n".join(lines)

    def find_similar(self, card_name: str, limit: int = DEFAULT_LIMIT, detail: str = "compact",
                     **flt: Any) -> str:
        row, alts = self.resolve_name(card_name)
        if row is None:
            sugg = ", ".join(n for n, _ in alts) or "none"
            return f'No card named "{card_name}". Closest names: {sugg}'
        limit = self._limit(limit)
        flt = self._clean_filters(flt)
        card = self.row(row)
        allowed = self.allowed(flt)
        same_name = set(np.flatnonzero((self.cards["name"] == card["name"]).to_numpy()).tolist())
        allowed = allowed[~np.isin(allowed, list(same_name))]
        active = self.active_filters(flt)
        head = f"similar to: {card['name']} · filters: {self._fmt_filters(active)}"
        if allowed.size == 0:
            return f"{head} · 0 cards match these filters; loosen them."
        hits = self.backend.search(card["doc_text"], limit, allowed)
        lines = [f"{head} · {len(hits)} shown of {allowed.size:,} matching filters"]
        lines += self._render_hits(hits, flt, detail)
        return "\n".join(lines)

    def rulings_for(self, oracle_id: str) -> list[tuple[str, str]]:
        if self.rulings_path is None or not Path(self.rulings_path).exists():
            return []
        df = (pl.scan_parquet(self.rulings_path).filter(pl.col("oracle_id") == oracle_id)
              .sort("published_at").collect())
        return [(str(d), c) for d, c in zip(df["published_at"].to_list(), df["comment"].to_list(), strict=True)]

    def get_card(self, name: str) -> str:
        row, alts = self.resolve_name(name)
        if row is None:
            sugg = ", ".join(n for n, _ in alts) or "none"
            return f'No card named "{name}". Closest names: {sugg}'
        c = self.row(row)
        out = [render_card(c, detail="full", tag_counts=self.tag_filter)]
        if alts:
            out.append("   (also close: " + ", ".join(n for n, _ in alts) + ")")
        out.append("legalities:")
        out.append(all_legalities(c.get("legalities")))
        tags = role_tags(c.get("tags"), self.tag_filter, n=30)
        if tags:
            out.append("oracle tags: " + ", ".join(tags))
        rul = self.rulings_for(c["oracle_id"])
        if rul:
            out.append(f"rulings ({len(rul)}):")
            out += [f"   - {d}: {t}" for d, t in rul]
        else:
            out.append("rulings: none")
        if c.get("scryfall_uri"):
            out.append(f"scryfall: {c['scryfall_uri']}")
        return "\n".join(out)

    def list_tags(self, prefix: str | None = None, limit: int = 60) -> str:
        t = self.tags
        if t.height == 0:
            return "no tag data available"
        p = (prefix or "").strip().lower().replace(" ", "-")
        if p:
            m = t.filter(pl.col("slug").str.starts_with(p))
            if m.height < 10:
                m = pl.concat([m, t.filter(pl.col("slug").str.contains(p, literal=True)
                                           & ~pl.col("slug").str.starts_with(p))])
        else:
            m = t
        trivia = [not self.tag_filter.is_functional(s) for s in m["slug"].to_list()]
        m = m.with_columns(pl.Series("_trivia", trivia, dtype=pl.Boolean)) \
            .sort(["_trivia", "card_count"], descending=[False, True])
        total = m.height
        m = m.head(max(1, min(200, limit)))
        lines = [(f"tags matching '{p or '*'}': {total} (showing {m.height}, functional first, "
                 "then most used; [trivia] = naming/flavour/template tag, not a deck role)")]
        has_desc = "description" in m.columns
        for r in m.iter_rows(named=True):
            d = (r.get("description") or "").strip().replace("\n", " ") if has_desc else ""
            if len(d) > 110:
                d = d[:107].rstrip() + "…"
            mark = " [trivia]" if r["_trivia"] else ""
            lines.append(f"- {r['slug']} ({r['card_count']} cards){mark}" + (f": {d}" if d else ""))
        # exact tag: show some example cards (alphabetical; deterministic, no metric ordering)
        if p and "tags" in self.cards.columns and p in set(m["slug"].to_list()):
            ex = (self.cards.with_row_index("_row")
                  .filter(pl.col("tags").list.contains(p) & ~pl.col("is_funny"))
                  .sort(["name", "_row"]).head(15))
            if ex.height:
                n_all = self.cards.filter(pl.col("tags").list.contains(p) & ~pl.col("is_funny")).height
                lines.append(f"example cards tagged {p} ({ex.height} of {n_all}, alphabetical): " + "; ".join(
                    f"{n} {c or ''}".strip() for n, c in zip(ex["name"], ex["mana_cost"], strict=True)))
        return "\n".join(lines)


# --------------------------------------------------------------------------- MCP wiring

_JSON_TYPES = {"string": str, "number": float, "integer": int, "boolean": bool}


def _annotation(spec: dict) -> Any:
    t = spec.get("type")
    if spec.get("enum"):
        return Literal[tuple(spec["enum"])]
    if t == "array":
        return list[_annotation(spec.get("items", {"type": "string"}))]
    if t == "object":
        return dict[str, _annotation(spec.get("additionalProperties", {"type": "string"}))]
    return _JSON_TYPES.get(t, Any)


def filter_parameters() -> list[inspect.Parameter]:
    """Tool parameters generated from ``filters.FILTER_PARAMS`` (all optional)."""
    params = []
    for name, spec in filters.FILTER_PARAMS.items():
        ann = _annotation(spec)
        default = spec.get("default", None)
        if default is None:
            ann = ann | None
        desc = spec.get("description", "")
        params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default,
                                        annotation=Annotated[ann, Field(description=desc)]))
    return params


def _P(name: str, ann: Any, default: Any = inspect.Parameter.empty, desc: str = "") -> inspect.Parameter:
    return inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default,
                             annotation=Annotated[ann, Field(description=desc)] if desc else ann)


def _make_tool(name: str, impl: Callable[..., str], params: list[inspect.Parameter], doc: str) -> Callable[..., str]:
    def tool(**kwargs: Any) -> str:
        from mcp.server.mcpserver.exceptions import ToolError

        try:
            return impl(**kwargs)
        except (ValueError, RuntimeError) as e:  # bad arguments / backend not ready -> shown to the model
            raise ToolError(str(e)) from e

    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = doc
    tool.__signature__ = inspect.Signature(params, return_annotation=str)  # type: ignore[attr-defined]
    tool.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}
    return tool


FILTER_DOC = """
Filters (all optional, combined with AND; applied *before* ranking, so even very selective
filters return the best matching cards, never an empty post-filtered list):
- color_identity: Commander identity within these colours, e.g. "BG" (Golgari deck); "C" = colourless.
- colors + colors_mode: actual card colours. "includes" (default) = has all given colours;
  "subset" = within them (good for 60-card decks, e.g. colors="RW", colors_mode="subset");
  "exact" = exactly them.
- format: legal (or restricted) in that format: commander, modern, pauper, standard, pioneer,
  legacy, vintage, brawl, oathbreaker, paupercommander, premodern, ...
- types: every entry must match a supertype/type/subtype, e.g. ["creature"], ["legendary", "elf"].
- cmc_min / cmc_max: mana value bounds.
- max_pips / min_pips: per-colour mana-symbol counts in the cost, e.g. max_pips={"U": 1} = no
  double-blue; min_pips={"R": 2}.
- has_x / has_hybrid / has_phyrexian: true = only such costs, false = exclude them.
- include_funny: include Un-set/playtest cards (default false).
"""

OUTPUT_DOC = """
Output: one summary line (query, active filters, "N shown of M matching filters"), then one block
per card: `n. Name — cost (MV) · colours · type line · P/T · rarity [markers]`, oracle text
(each face of a multi-face card as `// Face — cost · type`; "[MDFC: land back face]" flags
spell/land MDFCs), a roles line (functional Scryfall oracle tags, keywords, mana produced) and a
footer (legality in the requested format or the main formats it is legal in, colour identity
when it differs). Only rules and print facts are returned; markers are Legendary,
Commander-eligible / partner kind, Game Changer (WotC's official Commander bracket list),
Reserved and layout.
"""

SEARCH_DOC = f"""Semantic search over every Magic: The Gathering card, matched against Oracle rules text.

Phrase each query the way the card itself is worded, not the way a player describes it, e.g.
"Destroy target creature with flying.", "Whenever another creature you control dies, draw a
card.", "Counter target spell unless its controller pays {{3}}." Put colour, format, type and
mana-value constraints in the filter arguments, not in the query text.
- Narrow effect: send 1 precise query.
- Open-ended deck need: send 2-4 `queries`, each a *different* mechanism that solves it, e.g.
  "I keep losing to fliers" -> ["Reach", "Destroy target creature with flying.",
  "Creatures you control gain reach until end of turn."]. Their rankings are fused (RRF).
Write one clean sentence per query, not a whole invented card (type line plus abilities), and
don't add the user's own wording as a query. Results are ranked by text relevance only, never
by how widely played a card is; call get_card for full text and rulings.

Args: query or queries (1-{MAX_QUERIES} in total); limit = cards to return (default 15, max 50; use
30-50 when surveying options); detail = "compact" (oracle text cut at ~250 chars per face) or "full".
{FILTER_DOC}{OUTPUT_DOC}"""

SIMILAR_DOC = f"""Find cards that play like a given card (functional substitutes, upgrades, redundancy).

`card_name` is fuzzy-matched (e.g. "rhystic study"). The card's own oracle text is used as the
query, the card itself is excluded. Combine with filters to get e.g. a pauper-legal or
in-colour replacement: find_similar("Rhystic Study", format="pauper").
{FILTER_DOC}{OUTPUT_DOC}"""

GET_DOC = """Look up one card by name (fuzzy: "fire ice", "jace mind sculptor", a face name of a
double-faced card all work). Returns the full oracle text of every face, mana cost and value,
types, P/T, markers, all format legalities, functional oracle tags,
official rulings and the Scryfall link."""

TAGS_DOC = """List Scryfall oracle tags (community labels of what cards *do*, e.g. "removal",
"mana-rock", "free-sacrifice-outlet", "draw-engine") with card counts and descriptions.
`prefix` filters by slug prefix (falls back to substring matches), e.g. "sacrifice", "ramp".
When `prefix` is an exact tag slug, also lists up to 15 example cards with that tag (alphabetical). Tags appear in
the `roles:` line of search results; use their wording to phrase better search queries."""


def build_mcp(service: CardService):
    from mcp.server.mcpserver import MCPServer

    from . import __version__

    mcp = MCPServer(
        "mtg-search",
        version=__version__,
        instructions=(
            "Semantic search over all Magic: The Gathering cards for deck building. "
            "search_cards for roles/effects (with filters for colours, format, mana value, types), "
            "find_similar for substitutes of a known card, get_card for exact text/rulings/legality, "
            "list_tags to discover functional tag vocabulary."
        ),
    )
    fp = filter_parameters()
    limit_p = _P("limit", Annotated[int, Field(ge=1, le=MAX_LIMIT)], DEFAULT_LIMIT,
                 f"Number of cards to return (1-{MAX_LIMIT}).")
    detail_p = _P("detail", Literal["compact", "full"], "compact",
                  "compact = oracle text truncated to ~250 chars per face; full = complete text.")
    search = _make_tool("search_cards", service.search_cards, [
        _P("query", str | None, None, "One query phrased like Oracle rules text (shorthand for queries=[query])."),
        _P("queries", Annotated[list[str], Field(max_length=MAX_QUERIES)] | None, None,
           f"1-{MAX_QUERIES} rules-text queries, each a different mechanism; rankings are fused (RRF)."),
        *fp, limit_p, detail_p], SEARCH_DOC)
    similar = _make_tool("find_similar", service.find_similar, [
        _P("card_name", str, desc="Name of the reference card (fuzzy matched)."),
        *fp, limit_p, detail_p], SIMILAR_DOC)
    get = _make_tool("get_card", service.get_card, [_P("name", str, desc="Card name (fuzzy matched).")], GET_DOC)
    tags = _make_tool("list_tags", service.list_tags, [
        _P("prefix", str | None, None, "Tag slug prefix, e.g. 'sacrifice' or 'ramp'."),
        _P("limit", Annotated[int, Field(ge=1, le=200)], 60, "Max tags to list.")], TAGS_DOC)
    for fn in (search, similar, get, tags):
        mcp.add_tool(fn, structured_output=False)
    return mcp


def load_service(backend: Backend | None = None) -> tuple[CardService, Any]:
    data = config.DATA_DIR
    cards = pl.read_parquet(data / "cards.parquet",
                            columns=server_columns(pl.read_parquet_schema(data / "cards.parquet")))
    tags = pl.read_parquet(data / "tags.parquet", columns=["slug", "label", "description", "card_count", "parent_slugs"]) \
        if (data / "tags.parquet").exists() else None
    if backend is None:
        from .artifacts import check_index
        from .nextplaid import NextPlaidBackend

        url = os.environ.get("NEXTPLAID_URL") or None
        index_dir = Path(os.environ.get("MTG_SEARCH_NEXTPLAID_INDEX", "/app/index"))
        oids = cards["oracle_id"].to_list()
        if url is None:
            check_index(index_dir, oids)
        backend = NextPlaidBackend(
            oids,
            url=url,
            index_dir=index_dir if url is None else None,
            model_dir=os.environ.get("MTG_SEARCH_MODEL_DIR", "/app/model"),
            port=int(os.environ.get("NEXTPLAID_PORT", "8080")),
        )
        backend.start_background()
    return CardService(cards, backend, tags=tags, rulings_path=data / "rulings.parquet"), backend


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("MTG_SEARCH_LOG", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    def _term(signum, _frame):  # PID 1 in Docker: make SIGTERM/SIGINT exit cleanly
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    service, backend = load_service()
    try:
        build_mcp(service).run("stdio")
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
