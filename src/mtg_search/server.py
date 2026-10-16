"""MCP server (stdio): search and look up every Magic: The Gathering card.

Read-only tools over ``cards.parquet``: ``search_cards`` (semantic search with lightonai/mLateOn,
``mlateon.py``), ``find_similar``, ``get_card`` and ``get_deck`` (a pasted decklist, a Moxfield or Archidekt link, or an
official precon from ``precons.parquet``, checked against a format's deck rules: ``decks.py``, ``deck_links.py``,
``precons.py``). Results carry only
rules and print facts; nothing is ranked or shown by popularity or price.

Environment:
  MTG_SEARCH_DATA_DIR        cards/tags/rulings/precons parquet dir (default ./data; /app/data in the image)
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
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
from pydantic import BeforeValidator, Field

from . import config, decks, filters
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
                 rulings_path: Path | None = None, precons_path: Path | None = None,
                 fetch_deck: Callable[[str], Any] | None = None):
        self.cards = cards.select(server_columns(cards.columns))
        self.backend = backend
        tags = tags if tags is not None else pl.DataFrame({"slug": [], "card_count": []})
        self.tag_filter = TagFilter.from_frame(tags)
        self.rulings_path = rulings_path
        self.precons_path = precons_path
        # reads a Moxfield / Archidekt link (deck_links.fetch); None: links are not opened (tests, the benchmark)
        self.fetch_deck = fetch_deck
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

    # ----------------------------------------------------------------- decks
    @cached_property
    def _oid_rows(self) -> dict[str, int]:
        return {o: i for i, o in enumerate(self.cards["oracle_id"].to_list())} if "oracle_id" in self.cards.columns else {}

    @cached_property
    def _precons(self) -> pl.DataFrame | None:
        p = self.precons_path
        return pl.read_parquet(p) if p is not None and Path(p).exists() else None

    @cached_property
    def _precon_keys(self) -> list[tuple[str, list[str], tuple, frozenset[str]]]:
        """Per precon: normalised name, normalised commander names (and their faces), tie-break key, and the
        normalised names of the other legendary creatures in the 99 (a precon's alternative commanders: EDHREC
        files World Shaper under Szarel, MTGJSON under Hearthhull)."""
        df, out = self._precons, []
        for name, cmds, typ, date, cards in zip(df["name"].to_list(), df["commanders"].to_list(), df["type"].to_list(),
                                                df["release_date"].to_list(), df["cards"].to_list(), strict=True):
            keys = {_norm_name(k) for c in cmds or [] for k in [c, *c.split(" // ")]}
            legends = set()
            for c in cards or []:
                r = self._oid_rows.get(c["oracle_id"]) if c["section"] == "main" and c["oracle_id"] else None
                tl = (self.row(r).get("type_line") or "").split("//")[0] if r is not None else ""
                if "Legendary" in tl and "Creature" in tl:
                    legends.add(_norm_name(self.row(r)["name"]))
            # official Commander decks first, regular before Collector's Edition lists (same cards), newest first
            tie = (typ not in ("Commander Deck", "MTGO Commander Deck"), "collector" in name.lower(),
                   tuple(-ord(ch) for ch in date or ""))
            out.append((_norm_name(name), sorted(keys), tie, frozenset(legends)))
        return out

    def find_precons(self, query: str) -> list[int]:
        """Precon rows matching ``query`` by deck name or commander name, best first: exact, then prefix, then
        whole word, then a deck or commander name inside the query, then the name of another legendary creature
        in the deck, then a misspelling (edit similarity >= 85). Retried without "my", "precon", "deck" and the
        like."""
        from rapidfuzz import fuzz

        keys = self._precon_keys
        plain = re.sub(r"\b(my|the|precon|preconstructed|commander deck|deck)\b", " ", _norm_name(query))
        for q in dict.fromkeys([_norm_name(query), re.sub(r"\s+", " ", plain).strip()]):
            if not q:
                continue
            pat = re.compile(r"(^|\s)" + re.escape(q) + r"(\s|$)")
            tiers: dict[int, int] = {}
            for i, (name, cmds, _tie, _leg) in enumerate(keys):
                if q == name or q in cmds:
                    tiers[i] = 0
                elif len(q) >= 3 and (name.startswith(q) or any(c.startswith(q) for c in cmds)):
                    tiers[i] = 1
                elif len(q) >= 3 and (pat.search(name) or any(pat.search(c) for c in cmds)):
                    tiers[i] = 2
            if not tiers:  # the deck or commander name inside a longer query ("<deck> <commander>")
                padded = f" {q} "
                tiers = {i: 3 for i, (name, cmds, _tie, _leg) in enumerate(keys)
                         if any(len(k) >= 6 and f" {k} " in padded for k in [name, *cmds])}
            if not tiers:
                tiers = {i: 4 for i, k in enumerate(keys) if any(g == q or g.startswith(q + " ") for g in k[3])}
            if not tiers:
                for i, (name, cmds, _tie, _leg) in enumerate(keys):
                    if max(fuzz.ratio(q, k) for k in [name, *cmds]) >= 85:
                        tiers[i] = 5
            if tiers:
                return sorted(tiers, key=lambda i: (tiers[i], keys[i][2]))
        return []

    def _deck_name(self, name: str, strict: bool) -> tuple[int | None, bool, list[str]]:
        """(row, read fuzzily, closest names). ``strict``: exact name or face name only."""
        q = _norm_name(name)
        exact = self._name_index[2]
        if q in exact:
            return exact[q], False, []
        if strict or not q:
            return None, False, []
        row, alts = self.resolve_name(name)
        return row, row is not None, alts

    def get_deck(self, deck: str, format: str | None = None) -> str:
        text = (deck or "").strip()
        if not text:
            raise ValueError('give a precon name or a decklist, e.g. deck="Calling All Angels" or '
                             'deck="1 Sol Ring\\n1 Arcane Signet\\n..."')
        fmt = decks.normalize_format(format, self._formats())
        one_line = "\n" not in text
        if one_line and re.match(r"https?://", text):
            m = re.match(r"https?://(?:www\.)?mtgjson\.com/api/v5/decks/([^/?#]+?)\.json", text)
            if m and self._precons is not None:
                hits = [i for i, f in enumerate(self._precons["file"].to_list()) if f == m.group(1)]
                if hits:
                    return self._precon_deck(hits, fmt)
            return self._linked_deck(text, fmt)
        named = one_line and self._precons is not None and any(k[0] == _norm_name(text) for k in self._precon_keys)
        if named or (one_line and not re.match(r"(SB:\s*)?\d", text, re.I) and not text.startswith("<")):
            return self._precon_lookup(text, fmt)  # "20 Ways to Win" is a deck name, not 20 copies of a card
        return self._pasted_deck(text, fmt)

    def _precon_lookup(self, text: str, fmt: str | None) -> str:
        if self._precons is None:
            return ('Official precon lists are not in this card index. Pass the decklist as text instead: one card '
                    'per line, e.g. "1 Sol Ring".')
        hits = self.find_precons(text)
        if hits:
            return self._precon_deck(hits, fmt)
        from rapidfuzz import fuzz, process

        names = list(dict.fromkeys(self._precons["name"].to_list()))
        close = [names[i] for _n, _s, i in process.extract(_norm_name(text), [_norm_name(n) for n in names],
                                                            scorer=fuzz.token_set_ratio, limit=5)]
        card = self._deck_name(text, strict=True)[0]
        hint = f" {self.row(card)['name']} is a card: get_card shows one card." if card is not None else ""
        return (f'No precon named "{text}" or led by a commander of that name.{hint} Closest precon names: '
                f'{", ".join(close) or "none"}. Retry with one of them, or pass the decklist as text (one card per line).')

    def _precon_deck(self, hits: list[int], fmt: str | None) -> str:
        from .precons import SOURCE_FORMATS

        df = self._precons
        d = df.row(hits[0], named=True)
        entries, unresolved = [], []
        for c in d["cards"]:
            r = self._oid_rows.get(c["oracle_id"]) if c["oracle_id"] else None
            if r is None:
                r = self._deck_name(c["name"], strict=True)[0]
            if r is None:
                unresolved.append((c["name"], 0, []))
                entries.append(decks.Entry(c["qty"], c["section"], c["name"]))
            else:
                entries.append(decks.Entry(c["qty"], c["section"], self.row(r)["name"], self.row(r)))
        title = f'{d["name"]} · {d["type"]} · {d["set_code"]} · {d["release_date"]} · list from MTGJSON'
        others = list(dict.fromkeys(f'{df["name"][i]} ({df["set_code"][i]}, {(df["release_date"][i] or "")[:4]})'
                                    for i in hits[1:] if df["name"][i] != d["name"]))[:3]
        extra = ["other matches: " + "; ".join(others)] if others else []
        return self._deck_report(entries, fmt, SOURCE_FORMATS.get(d["type"]), title, extra=extra, unresolved=unresolved)

    def _linked_deck(self, url: str, fmt: str | None) -> str:
        from .deck_links import LinkError

        host = re.sub(r"^www\.", "", re.match(r"https?://([^/?#]+)", url).group(1))
        if self.fetch_deck is None:
            return (f"This server does not open deck links here. Export the deck from {host} as text (one card per "
                    'line, e.g. "1 Sol Ring") and pass that text as deck.')
        try:
            linked = self.fetch_deck(url)
        except LinkError as e:
            return str(e)
        if not linked.decklist.lines:
            return f"The {linked.site} deck at {url} has no cards."
        try:
            source = decks.normalize_format(linked.format, list(decks.RULES) + self._formats())
        except ValueError:
            source = None
        return self._listed_deck(linked.decklist, fmt, f"{linked.name} · {linked.site} · {url}",
                                 source_format=source, source_label=linked.site)

    def _formats(self) -> list[str]:
        return ([f.name for f in self.cards.schema["legalities"].fields] if "legalities" in self.cards.columns
                else list(decks.RULES))

    def _pasted_deck(self, text: str, fmt: str | None) -> str:
        parsed = decks.parse_decklist(text)
        if not parsed.lines:
            raise ValueError('no card lines found; give one card per line, e.g. "4 Lightning Bolt"')
        return self._listed_deck(parsed, fmt, parsed.name or "pasted list")

    def _listed_deck(self, parsed: decks.Decklist, fmt: str | None, title: str, *, source_format: str | None = None,
                     source_label: str = "the deck type") -> str:
        merged: dict[tuple[str, str], decks.Entry] = {}
        unresolved, read_as, ignored = [], [], list(parsed.ignored)
        for ln in parsed.lines:
            r, fuzzy, alts = self._deck_name(ln.name, strict=not ln.counted)
            if r is None:
                if ln.counted:
                    unresolved.append((ln.name, ln.lineno, alts))
                    merged.setdefault((ln.name, ln.section), decks.Entry(0, ln.section, ln.name)).qty += ln.qty
                else:
                    ignored.append(ln.name)
                continue
            card = self.row(r)
            if fuzzy:
                read_as.append((ln.name, card["name"]))
            merged.setdefault((card["name"], ln.section), decks.Entry(0, ln.section, card["name"], card)).qty += ln.qty
        return self._deck_report(list(merged.values()), fmt, source_format, title, source_label=source_label,
                                 unresolved=unresolved, read_as=read_as, ignored=ignored)

    def _deck_report(self, entries, fmt, source_format, title, *, source_label="the deck type", extra=(),
                     unresolved=(), read_as=(), ignored=()) -> str:
        if fmt:
            rules, how = decks.RULES.get(fmt), "given"
        elif source_format and source_format not in decks.RULES:  # e.g. a Moxfield Oathbreaker deck: legality only
            fmt, rules, how = source_format, None, f"from {source_label}"
        else:
            fmt, rules, how = decks.detect(entries, source_format, source_label)
        rep = decks.check(entries, fmt, rules, how)
        return decks.render(rep, title=title, extra=extra, unresolved=unresolved, read_as=read_as, ignored=ignored,
                            tag_counts=self.tag_filter)

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
# The slim three-tool schemas searched less than the old 17-parameter ones (EXP-014: 10/30 vs 19/30 dev episodes;
# EXP-010 e4slim the same), so both core descriptions open with the trigger sentence tested in EXP-010/011
# (e1desc / e6desc_slim: 60-79% of ADD episodes searched). search_cards and get_card
# load upfront in Claude Code (alwaysLoad).

INSTRUCTIONS = """Magic: The Gathering (MTG) card search and lookup over every card's current official rules text (Scryfall data, which can include cards printed after your training data). Use these tools whenever a task involves choosing, suggesting or checking cards: building, upgrading or cutting a Commander/EDH or other deck, "cards that do X", replacements for a card, or a card's exact text, rulings, colour identity or format legality. Use them even when you feel sure from memory: errata, new cards, and colour-identity or legality slips are the usual errors.
Efficient use: one search_cards call per deck need, with color_identity (and format) set, returns candidates that are already in colour and legal; send 2-4 queries in that call for a broad need. get_card checks one card you named from memory or whose rulings matter. find_similar gives alternatives to a known card, e.g. for a swap. Rankings use rules text only.
Before you answer a deck-building request, use the tools: search for candidate cards and check exact card text, colour identity and legality of your picks."""

SEARCH_DOC = """Before recommending cards for a deck, call this to find candidates instead of listing them from memory. Search every Magic: The Gathering (MTG) card by what it does: semantic search over the rules text, with filters for Commander colour identity, format legality, card type and mana value. Use it to find candidates for a deck need (removal, ramp, card draw, sacrifice outlets, answers to fliers, ...) rather than listing cards from memory. With color_identity (and format="commander") set, every result is in colour and legal, so no separate check is needed.

Write each query the way the rules text of a matching card reads, not the way a player describes the need: "Destroy target creature with flying." rather than "answers to fliers". For a broad need, send 2-4 queries that each describe a different way to meet it, e.g. ["Reach", "Destroy target creature with flying.", "Creatures you control gain reach until end of turn."]; their rankings are merged. Put colour, format, type and mana-value limits in the filters, not in the queries.

Returns up to `limit` cards, best first, ranked only by how well their text matches your queries: name, mana cost and mana value, colours, type line, power/toughness, rules text (long text is shortened; get_card shows the full card), roles (what the card does in a deck, e.g. removal, ramp) and legality. Filters apply before ranking, so even narrow filters return the best matching cards. 15 cards is about 2,000 tokens."""

SIMILAR_DOC = """Find Magic: The Gathering cards that do what a given card does: replacements, upgrades or redundancy. The named card's rules text is the query and the card itself is left out. Use it for "more cards like X", an in-colour or format-legal stand-in for a card, or alternatives in a swap, e.g. card_name="Divination", color_identity="BR", format="commander". Same filters and output as search_cards."""

GET_DOC = """Before recommending a card you know only from memory, call this to check its current text, colour identity and legality. Shows one Magic: The Gathering card in full, by name (fuzzy: "fire ice", "jace mind sculptor" or one face of a double-faced card all work). Returns the complete current rules text of every face, mana cost and mana value, type line, power/toughness, colours and colour identity, legality in every format, markers (Game Changer, Reserved, Commander-eligible), roles, official rulings and the Scryfall link. Also use it when exact wording, a ruling or legality decides the answer. Cards from a search_cards call with color_identity and format set need no second check."""

GET_DECK_DOC = """Before adding, cutting or swapping cards in a deck the user names, links or pastes, call this to load its exact list instead of recalling it from memory. Reads one Magic: The Gathering deck: a Moxfield or Archidekt deck link, an official preconstructed deck by its name or its commander's name, or a decklist as text, one card per line ("1 Sol Ring", "20 Island (M21) 264"; Arena, MTGO, Moxfield and Archidekt exports with Commander, Companion and Sideboard sections all work). For a deck on another site or in a file, pass its exported text.

Returns every card with mana cost, type line and roles, grouped by section, the commander's full text, the format (as given, from the deck's source, or detected from its size and sections) and every rule the deck breaks in that format: deck size, copies, sideboard size, commander and partner rules, colour identity, banned, restricted or not legal cards, and the Game Changer count. It does not suggest cards: call search_cards with the deck's colour identity and format for that. A 100-card deck is about 2,500 tokens."""


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
DeckText = Annotated[str, Field(
    description="A Moxfield or Archidekt deck link, an official precon's name or its commander's name, or a whole "
                "decklist as text (one card per line).")]
DeckFormat = Annotated[str | None, Field(
    description="Format to check the deck against, e.g. 'commander', 'modern', 'brawl'. Leave empty to detect it.")]


def build_mcp(service: CardService, surface: dict[str, str] | None = None):
    """The MCP server. ``surface`` overrides the model-facing text for A/B tests on the precon benchmark
    (``bench/precon/harness/surfaces.py``): keys ``instructions``, ``search_cards``, ``find_similar``, ``get_card``,
    ``get_deck`` (tool descriptions), ``omit`` (comma-separated names of tools to leave out) and ``always_load``
    (deferred tools to load up front)."""
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

    def get_deck(deck: DeckText, format: DeckFormat = None) -> str:
        return guarded(service.get_deck)(deck=deck, format=format)

    mcp = MCPServer("mtg-search", title="MTG card search", version=__version__,
                    instructions=surface.get("instructions", INSTRUCTIONS))
    # Claude Code defers MCP tools behind ToolSearch by default; without alwaysLoad the agent rarely loads them
    # for a task it thinks it can answer from memory (EXP-009: 5% vs 45% of episodes with a tool call).
    always = {"anthropic/alwaysLoad": True}
    omit = {t.strip() for t in surface.get("omit", "").split(",") if t.strip()}
    load = {t.strip() for t in surface.get("always_load", "").split(",") if t.strip()}
    # get_deck is deferred, like find_similar: an extra always-loaded tool can cost searching (EXP-010's check_cards
    # replaced it; EXP-014's slimmer surface searched less). Test before changing: eval/experiments/deck_tool_plan.md
    for fn, title, doc, meta in ((search_cards, "Search cards", SEARCH_DOC, always), (get_card, "Get card", GET_DOC, always),
                                 (find_similar, "Find similar cards", SIMILAR_DOC, None),
                                 (get_deck, "Get deck", GET_DECK_DOC, None)):
        if fn.__name__ in omit:
            continue
        # get_deck opens the Moxfield / Archidekt link it is given (when the service has a fetcher)
        web = fn is get_deck and service.fetch_deck is not None
        mcp.add_tool(fn, title=title, description=surface.get(fn.__name__, doc), structured_output=False,
                     meta=always if fn.__name__ in load else meta,
                     annotations=ToolAnnotations(title=title, read_only_hint=True, destructive_hint=False,
                                                 idempotent_hint=not web, open_world_hint=web))
    return mcp


def index_paths(revision: str | None = None, home: str | None = None) -> tuple[Path, Path, Path]:
    """(data, embeddings, model) directories. Explicit MTG_SEARCH_EMBEDDINGS_DIR / MTG_SEARCH_MODEL_DIR (the
    Docker image, a dev checkout) win; otherwise the pinned index in the user cache, downloaded on first use."""
    emb, model = os.environ.get("MTG_SEARCH_EMBEDDINGS_DIR"), os.environ.get("MTG_SEARCH_MODEL_DIR")
    if emb and model:
        return config.DATA_DIR, Path(emb), Path(model)
    from . import artifacts

    revision = revision or config.INDEX_REVISION
    base = Path(home) if home else config.index_base()
    have = artifacts.complete_revisions(base)
    if revision not in have and have and artifacts._model_ready(base):
        # an update pinned a snapshot that is not here yet: serve the newest one we have now (a client
        # gives a server only seconds to start) and fetch the new one in the background for the next start
        import threading

        log.warning("index %s not downloaded yet; serving %s and downloading %s in the background", revision, have[-1], revision)
        threading.Thread(target=artifacts.ensure_index, args=(revision, base), daemon=True).start()
        revision = have[-1]
    root, model_dir = artifacts.ensure_index(revision, base)
    return root / "data", root / config.EMBEDDINGS_SUBDIR, model_dir


def load_service(backend: Any = None, *, data_dir: str | Path | None = None, embeddings_dir: str | Path | None = None,
                 model_dir: str | Path | None = None) -> tuple[CardService, Any]:
    data = Path(data_dir) if data_dir else config.DATA_DIR
    cards = pl.read_parquet(data / "cards.parquet", columns=server_columns(pl.read_parquet_schema(data / "cards.parquet")))
    tp = data / "tags.parquet"
    tags = pl.read_parquet(tp, columns=["slug", "card_count", "parent_slugs"]) if tp.exists() else None
    if backend is None:
        from .mlateon import MLateOn

        backend = MLateOn(cards["oracle_id"].to_list(),
                          embeddings_dir=embeddings_dir or os.environ.get("MTG_SEARCH_EMBEDDINGS_DIR", "/app/embeddings"),
                          model_dir=model_dir or os.environ.get("MTG_SEARCH_MODEL_DIR", "/app/model"))
    from .deck_links import fetch

    return CardService(cards, backend, tags=tags, rulings_path=data / "rulings.parquet",
                       precons_path=data / "precons.parquet", fetch_deck=fetch), backend


def main(argv: list[str] | None = None) -> None:
    import argparse

    from . import __version__

    ap = argparse.ArgumentParser(prog="mtg-search-mcp",
                                 description="MCP server over stdio: semantic search over every Magic: The Gathering card.")
    ap.add_argument("--download", action="store_true",
                    help="download the card index (about 1.9 GB, once) and exit; run this before adding the server to a client")
    ap.add_argument("--index-revision", default=None, help=f"index snapshot to use (default {config.INDEX_REVISION})")
    ap.add_argument("--home", default=None, help="index directory (default $MTG_SEARCH_HOME, else ~/.cache/mtg-search/<revision>)")
    ap.add_argument("--version", action="version", version=f"mtg-search-mcp {__version__}")
    args = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("MTG_SEARCH_LOG", "INFO"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub"):  # one line per HTTP request otherwise
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.download:
        from .artifacts import ensure_index

        root, _ = ensure_index(args.index_revision, args.home)
        print(f"mtg-search: index ready at {root}", file=sys.stderr)
        return

    def _term(signum, _frame):  # PID 1 in Docker: exit cleanly on SIGTERM
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    data, emb, model = index_paths(args.index_revision, args.home)
    service, backend = load_service(data_dir=data, embeddings_dir=emb, model_dir=model)
    try:
        build_mcp(service).run("stdio")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
