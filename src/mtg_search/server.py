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
                keys.append(_norm_name(nm))
                rows.append(i)
        # exact lookups: a card's full name beats another card's face name ("Lightning Bolt" is also a face of
        # the MDFC "Emeritus of Conflict // Lightning Bolt"), then non-funny before funny
        for i in order:
            exact.setdefault(_norm_name(names[i]), i)
        for i in order:
            for nm in faces[i] or []:
                exact.setdefault(_norm_name(nm), i)
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

    def check_cards(self, cards: list[str], commander: list[str] | None = None, color_identity: str | None = None,
                    format: str = "commander", decklist: list[str] | None = None, bracket: int | None = None) -> str:
        """Rules-only validity of candidate cards for a deck (issue #4, EXP-010 prototype)."""
        def fmt_ci(ci) -> str:
            return "".join(c for c in "WUBRG" if c in (ci or [])) or "C"

        def free(c: dict) -> bool:  # singleton exceptions: basic lands, "any number of cards named"
            return "Basic" in (c.get("type_line") or "") or "any number of cards named" in (c.get("oracle_text") or "")

        if not cards:
            raise ValueError("pass at least one card name in `cards`")
        if len(cards) > 40:
            raise ValueError("at most 40 cards per call")
        head, ci = [], None
        if commander:
            ci = set()
            for nm in commander:
                r = self._name_index[2].get(_norm_name(nm))
                if r is None:
                    raise ValueError(f'unknown commander "{nm}"; use the exact card name')
                ci |= set(self.row(r).get("color_identity") or [])
            head.append(f"commander {' + '.join(commander)} (identity {fmt_ci(ci)})")
        elif color_identity:
            ci = set(color_identity.upper().replace("C", ""))
            head.append(f"identity {fmt_ci(ci)}")
        head.append(f"format {format}")
        limit = {1: 0, 2: 0, 3: 3}.get(bracket) if bracket else None
        deck_rows: dict[int, int] = {}
        gc = 0
        if decklist:
            unknown = []
            for nm in decklist:
                m = re.match(r"^\s*(\d+)\s*x?\s+(.*)$", nm)
                n, nm2 = (int(m.group(1)), m.group(2)) if m else (1, nm)
                r = self._name_index[2].get(_norm_name(nm2))
                if r is None:
                    unknown.append(nm2)
                    continue
                deck_rows[r] = deck_rows.get(r, 0) + n
            gc = sum(1 for r in deck_rows if self.row(r).get("game_changer"))
            head.append(f"deck: {sum(deck_rows.values())} cards, {gc} Game Changers"
                        + (f", {len(unknown)} unrecognised names ignored" if unknown else ""))
        if bracket:
            head.append(f"bracket {bracket}" + (f" (Game Changer limit {limit})" if limit is not None else ""))
        lines, seen, n_ok = [f"check: {len(cards)} cards · " + " · ".join(head)], set(), 0
        for nm in cards:
            r = self._name_index[2].get(_norm_name(nm))
            if r is None:
                best, alts = self.resolve_name(nm)
                names = ([self.cards["name"][best]] if best is not None else []) + [n for n, _ in alts]
                sugg = ", ".join(dict.fromkeys(names[:3])) or "none"
                lines.append(f'- INVALID "{nm}": not an exact card name; closest: {sugg}')
                continue
            c, why = self.row(r), []
            if ci is not None and not set(c.get("color_identity") or []) <= ci:
                why.append(f"outside colour identity (card is {fmt_ci(c.get('color_identity'))})")
            st = (c.get("legalities") or {}).get(format)
            if st not in ("legal", "restricted"):
                why.append(f"{(st or 'unknown').replace('_', ' ')} in {format}")
            if r in deck_rows and not free(c):
                why.append("already in the deck")
            if r in seen and not free(c):
                why.append("listed twice")
            seen.add(r)
            if c.get("game_changer") and not why:
                gc_note = "Game Changer"
                if limit is not None:
                    gc += 1
                    gc_note += f" ({gc} of {limit} allowed)" if gc <= limit else ""
                    if gc > limit:
                        why.append(f"over the Game Changer limit for bracket {bracket} ({gc} > {limit})")
            else:
                gc_note = "Game Changer" if c.get("game_changer") else ""
            if why:
                lines.append(f"- INVALID {c['name']}: " + "; ".join(why))
            else:
                n_ok += 1
                lines.append(f"- ok {c['name']}" + (f" [{gc_note}]" if gc_note else ""))
        lines.append(f"{n_ok} of {len(cards)} valid" + ("" if decklist else
                     " (pass decklist to also check duplicates, cards already in the deck and the Game Changer count)"))
        return "\n".join(lines)

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


# Model-facing text (EXP-009, eval/experiments/mcp_instructions.md): the instructions carry the trigger and a
# procedural "before you answer, use the tools" sentence; each description stands alone, key terms first, and does
# not repeat the parameter schema. Measured on the precon benchmark before changing (bench/precon surfaces).
SEARCH_DOC = """Search every Magic: The Gathering (MTG) card by what it does: semantic search over Oracle rules text, with filters for Commander colour identity, format legality, card type and mana value. Use it to find candidates for a deck need (removal, ramp, card draw, sacrifice outlets, answers to fliers, ...) rather than listing cards from memory. With color_identity (and format="commander") set, every result is in colour and legal, so no separate check is needed.

Phrase each query the way the card itself is worded, not the way a player describes it, e.g. "Destroy target creature with flying.", "Whenever another creature you control dies, draw a card." Put colour, format, type and mana-value constraints in the filter arguments, not in the query text.
- Narrow effect: send 1 precise query.
- Open-ended need: send 2-4 `queries`, each a *different* mechanism, e.g. "I keep losing to fliers" -> ["Reach", "Destroy target creature with flying.", "Creatures you control gain reach until end of turn."]. Their rankings are fused.
Write one clean sentence per query, not a whole invented card, and don't add the user's own wording as a query.

Results are ranked by text relevance only, never by how widely played a card is. Filters apply before ranking, so even narrow filters return the best matches. Default: 15 cards, compact (oracle text cut at ~250 characters per face, plus roles, legality and Game Changer markers), about 2,000 tokens. Use limit 30-50 only to survey a role; use get_card or detail="full" for complete text."""

SIMILAR_DOC = """Find Magic: The Gathering (MTG) cards that do what a given card does: replacements, upgrades and redundancy. Use it when the user wants "more cards like X", a stand-in for a card that is off-colour or not legal in their format, or, in a swap, an in-colour alternative to the card being cut. `card_name` is fuzzy-matched; that card's Oracle text is the query and the card itself is excluded. Takes the same filters as search_cards, e.g. find_similar("Rhystic Study", color_identity="BR", format="commander"). Ranked by rules-text similarity only; same compact output as search_cards."""

GET_DOC = """Look up one Magic: The Gathering (MTG) card by name (fuzzy: "fire ice", "jace mind sculptor" or one face of a double-faced card all work). Returns the current Oracle text of every face, mana cost and value, type line, P/T, colours and colour identity, legality in every format, markers (Game Changer, Reserved, Commander-eligible), functional tags, official rulings and the Scryfall link. Use it before recommending or judging a card you know only from memory, and when exact wording, a ruling or legality decides the answer. About 300-1,500 tokens per card; cards found with filtered search_cards need no second check."""

TAGS_DOC = """List Scryfall oracle tags: community labels for what MTG cards do (e.g. "removal", "mana-rock", "free-sacrifice-outlet", "draw-engine"), with card counts and descriptions. Use it rarely: to find the usual name for a deck role before wording a search_cards query, or to see example cards for one tag. `prefix` filters by slug prefix (falling back to substring matches); an exact slug also lists up to 15 example cards (alphabetical). The same tags appear on the `roles:` line of every search result."""

INSTRUCTIONS = """Magic: The Gathering (MTG) card search and lookup over the full Oracle card database (Scryfall data, which can include cards printed after your training data). Use these tools whenever a task involves choosing, suggesting or checking cards: building, upgrading or cutting a Commander/EDH or other deck, "cards that do X", replacements for a card, or a card's exact text, rulings, colour identity or format legality. Use them even when you feel sure from memory: errata, new cards, and colour-identity or legality slips are the usual errors.
Efficient use: one search_cards call per deck need, with color_identity (and format) set, returns candidates that are already in colour and legal; send 2-4 queries in that call for a broad need. get_card checks one card you named from memory or whose rulings matter. find_similar gives alternatives to a known card, e.g. for a swap. list_tags only helps find role vocabulary. Rankings use rules text only.
Before you answer a deck-building request, use the tools: search for candidate cards and check exact card text, colour identity and legality of your picks."""

CHECK_DOC = """Check candidate Magic: The Gathering cards against a Commander deck's rules in one call: exact card name, \
commander colour identity, format legality and Game Changer status; with `decklist` also cards already in the deck, \
duplicates and the Game Changer count for the bracket (bracket 1-2 allow 0, bracket 3 allows 3). Use it on your \
final adds or swaps before you answer, especially for cards you named from memory. Returns one line per card \
(ok / INVALID with the reason); about 30 tokens per card. Rules facts only."""

# Claude Code defers MCP tool schemas behind ToolSearch by default; without alwaysLoad the agent rarely loads
# them for a task it thinks it can answer from memory (EXP-009: 5% vs 45% of episodes with tool calls).
ALWAYS_LOAD = {"anthropic/alwaysLoad": True}


def build_mcp(service: CardService, surface: dict[str, str] | None = None):
    """``surface`` overrides the model-facing text for A/B tests (EXP-009): keys ``instructions``,
    ``search_cards``, ``find_similar``, ``get_card``, ``list_tags`` (tool descriptions)."""
    from mcp.server.mcpserver import MCPServer

    from . import __version__

    surface = surface or {}
    mcp = MCPServer("mtg-search", version=__version__, instructions=surface.get("instructions", INSTRUCTIONS))
    fp = filter_parameters()
    limit_p = _P("limit", Annotated[int, Field(ge=1, le=MAX_LIMIT)], DEFAULT_LIMIT,
                 f"Number of cards to return (1-{MAX_LIMIT}).")
    detail_p = _P("detail", Literal["compact", "full"], "compact",
                  "compact = oracle text truncated to ~250 chars per face; full = complete text.")
    search = _make_tool("search_cards", service.search_cards, [
        _P("query", str | None, None, "One query phrased like Oracle rules text (shorthand for queries=[query])."),
        _P("queries", Annotated[list[str], Field(max_length=MAX_QUERIES)] | None, None,
           f"1-{MAX_QUERIES} rules-text queries, each a different mechanism; rankings are fused (RRF)."),
        *fp, limit_p, detail_p], surface.get("search_cards", SEARCH_DOC))
    similar = _make_tool("find_similar", service.find_similar, [
        _P("card_name", str, desc="Name of the reference card (fuzzy matched)."),
        *fp, limit_p, detail_p], surface.get("find_similar", SIMILAR_DOC))
    get = _make_tool("get_card", service.get_card, [_P("name", str, desc="Card name (fuzzy matched).")],
                     surface.get("get_card", GET_DOC))
    tags = _make_tool("list_tags", service.list_tags, [
        _P("prefix", str | None, None, "Tag slug prefix, e.g. 'sacrifice' or 'ramp'."),
        _P("limit", Annotated[int, Field(ge=1, le=200)], 60, "Max tags to list.")], surface.get("list_tags", TAGS_DOC))
    for fn in (search, similar, get, tags):
        meta = ALWAYS_LOAD if fn in (search, get) else None
        mcp.add_tool(fn, structured_output=False, meta=meta)
    if surface.get("_check_cards"):  # EXP-010 prototype (issue #4); not registered in the shipped server yet
        check = _make_tool("check_cards", service.check_cards, [
            _P("cards", Annotated[list[str], Field(min_length=1, max_length=40)], desc="Exact names of the cards to check."),
            _P("commander", list[str] | None, None, "Commander name(s); their colour identity is used."),
            _P("color_identity", str | None, None, 'Colour identity if no commander is given, e.g. "UR".'),
            _P("format", str, "commander", "Format for the legality check."),
            _P("decklist", list[str] | None, None, "Current deck card names (optional; enables deck-level checks)."),
            _P("bracket", Annotated[int, Field(ge=1, le=5)] | None, None, "Commander bracket 1-5 (Game Changer limit).")],
            surface.get("check_cards", CHECK_DOC))
        mcp.add_tool(check, structured_output=False, meta=ALWAYS_LOAD)
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
