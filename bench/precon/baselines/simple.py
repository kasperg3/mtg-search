"""EXP-007 non-LLM baselines (plan §9). Each ``propose(task, deck, rep)`` returns the same output
dict an agent would (``{"cuts": [...]}``, ``{"adds": [...]}`` or ``{"swaps": [...]}``), so the
scorer treats every system identically.

* :class:`RandomLegal` (floor, seeded per precon/task/repeat): cuts uniform over the deck's
  non-commander nonbasic cards; adds uniform over the commander-legal, in-identity, non-Un pool
  minus the deck, skipping Game Changers beyond the bracket limit.
* :class:`TextHeuristic` (no LLM): cuts = unconditional tapped lands first (from oracle text, at
  most ``max_taplands``), then the highest-MV nonland cards whose functional oracle tags share no
  role with the commander's (ties by name), then highest MV overall. Adds = top ``search``
  results for the commander's own oracle text (12) plus 2 each from 4 generic role queries,
  filtered to the deck's identity and Commander legality, through the shipped retriever
  (mLateOn) by default.
* :class:`PopularityOracle` (REFERENCE, not a competitor; uses offline ``edhrec_rank`` the agent
  never sees): adds = best global rank in identity not in the deck; cuts = worst-ranked
  non-commander nonbasic cards. Quantifies how much of the ground truth is plain popularity.
"""

from __future__ import annotations

import hashlib
import random
import re

import numpy as np
import polars as pl

from ..loader import ROOT, CardDB, Deck

TAPLAND = re.compile(r"enters (?:the battlefield )?tapped\.", re.IGNORECASE)
CONDITIONAL = re.compile(r"unless|if you control|you may pay|you may reveal", re.IGNORECASE)
ROLE_QUERIES = (
    "add mana or search your library for a land card and put it onto the battlefield",
    "draw cards",
    "destroy or exile target creature or permanent an opponent controls",
    "destroy all creatures",
)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _swaps(cuts: list[str], adds: list[str]) -> dict:
    return {"swaps": [{"cut": c, "add": a, "reason": ""} for c, a in zip(cuts, adds, strict=False)]}


def _legal_pool(db: CardDB, ci: frozenset[str]) -> list[str]:
    return sorted(o for o, c in db.cards.items() if c.commander_legal and not c.is_funny and c.color_identity <= ci)


def _gc_ok(db: CardDB, deck: Deck, gc_limit: int | None):
    n = sum(1 for o in deck.qty if o in db and db[o].game_changer)

    def ok(oid: str) -> bool:
        nonlocal n
        if not db[oid].game_changer:
            return True
        if gc_limit is not None and n >= gc_limit:
            return False
        n += 1
        return True

    return ok


class RandomLegal:
    name = "random"

    def __init__(self, db: CardDB, seed: int = 0, gc_limit: int | None = 3):
        self.db, self.seed, self.gc_limit = db, seed, gc_limit
        self._pools: dict[frozenset, list[str]] = {}

    def cuts(self, deck: Deck, rng: random.Random, n: int = 10) -> list[str]:
        cand = sorted(o for o in deck.qty if o not in deck.commander_oids and o in self.db and not self.db[o].is_basic)
        return [self.db[o].name for o in rng.sample(cand, min(n, len(cand)))]

    def adds(self, deck: Deck, rng: random.Random, n: int = 20) -> list[str]:
        if deck.color_identity not in self._pools:
            self._pools[deck.color_identity] = _legal_pool(self.db, deck.color_identity)
        pool = [o for o in self._pools[deck.color_identity] if o not in deck.qty]
        ok = _gc_ok(self.db, deck, self.gc_limit)
        out: list[str] = []
        for o in rng.sample(pool, min(len(pool), 4 * n)):
            if len(out) >= n:
                break
            if self.db[o].is_basic or not ok(o):
                continue
            out.append(self.db[o].name)
        return out

    def propose(self, task: str, deck: Deck, rep: int = 0) -> dict:
        rng = random.Random(_seed(self.seed, deck.precon_id, task, rep))
        if task == "cut":
            return {"cuts": self.cuts(deck, rng)}
        if task == "add":
            return {"adds": self.adds(deck, rng)}
        return _swaps(self.cuts(deck, rng), self.adds(deck, rng, 10))


class TextHeuristic:
    name = "heuristic"

    def __init__(self, db: CardDB, cards: pl.DataFrame, retriever, gc_limit: int | None = 3, max_taplands: int = 3,
                 n_cmd: int = 12, n_role: int = 2):
        from mtg_search.tag_filter import TagFilter

        self.db, self.cards, self.r, self.gc_limit = db, cards, retriever, gc_limit
        self.max_taplands, self.n_cmd, self.n_role = max_taplands, n_cmd, n_role
        self.oids = cards["oracle_id"].to_list()
        self.doc = dict(zip(self.oids, cards["doc_text"].to_list(), strict=True))
        tags_path = ROOT / "data" / "tags.parquet"
        self.tf = TagFilter.from_frame(pl.read_parquet(tags_path)) if tags_path.exists() else None
        self._cache: dict[tuple, list[int]] = {}

    def _functional(self, oid: str) -> set[str]:
        clo = self.db.tag_closure(oid)
        if self.tf is None:
            return set(clo)
        return {t for t in clo if self.tf.is_functional(t)}

    def cuts(self, deck: Deck, n: int = 10) -> list[str]:
        db = self.db
        cand = [o for o in deck.qty if o not in deck.commander_oids and o in db and not db[o].is_basic]
        tap = sorted((o for o in cand if db[o].is_land and TAPLAND.search(self.doc.get(o, ""))
                      and not CONDITIONAL.search(self.doc.get(o, ""))), key=lambda o: db[o].name)[: self.max_taplands]
        cmd_roles = set().union(*(self._functional(c) for c in deck.commander_oids if c in db))
        nonland = [o for o in cand if not db[o].is_land and o not in tap]
        off = sorted((o for o in nonland if not (self._functional(o) & cmd_roles)), key=lambda o: (-db[o].cmc, db[o].name))
        rest = sorted((o for o in nonland if o not in set(off)), key=lambda o: (-db[o].cmc, db[o].name))
        return [db[o].name for o in (tap + off + rest)[:n]]

    def _search(self, query: str, ci: frozenset[str], k: int) -> list[int]:
        from mtg_search.filters import allowed_indices

        key = (query, ci, k)
        if key not in self._cache:
            mask = np.zeros(len(self.oids), dtype=bool)
            mask[allowed_indices(self.cards, color_identity="".join(sorted(ci)) or "C", format="commander")] = True
            self._cache[key] = [int(r) for r, _ in self.r.search(query, k, mask)]
        return self._cache[key]

    def adds(self, deck: Deck, n: int = 20) -> list[str]:
        db = self.db
        ok = _gc_ok(db, deck, self.gc_limit)
        out: list[str] = []
        seen = set(deck.qty)

        def take(rows: list[int], m: int) -> None:
            got = 0
            for r in rows:
                o = self.oids[r]
                if got >= m or len(out) >= n:
                    return
                if o in seen or db[o].is_basic or not ok(o):
                    continue
                seen.add(o)
                out.append(db[o].name)
                got += 1

        cmd_text = " ".join(self.doc.get(c, "") for c in deck.commander_oids)
        take(self._search(cmd_text, deck.color_identity, 80), self.n_cmd)
        for q in ROLE_QUERIES:
            take(self._search(q, deck.color_identity, 80), self.n_role)
        if len(out) < n:  # top up from the commander query
            take(self._search(cmd_text, deck.color_identity, 200), n - len(out))
        return out[:n]

    def propose(self, task: str, deck: Deck, rep: int = 0) -> dict:
        if task == "cut":
            return {"cuts": self.cuts(deck)}
        if task == "add":
            return {"adds": self.adds(deck)}
        return _swaps(self.cuts(deck), self.adds(deck, 10))


class PopularityOracle:
    name = "popularity"

    def __init__(self, db: CardDB, gc_limit: int | None = 3):
        self.db, self.gc_limit = db, gc_limit
        self._pools: dict[frozenset, list[str]] = {}

    def cuts(self, deck: Deck, n: int = 10) -> list[str]:
        db = self.db
        cand = [o for o in deck.qty if o not in deck.commander_oids and o in db and not db[o].is_basic]
        cand.sort(key=lambda o: (-(db[o].edhrec_rank or 10**9), db[o].name))
        return [db[o].name for o in cand[:n]]

    def adds(self, deck: Deck, n: int = 20) -> list[str]:
        db = self.db
        ci = deck.color_identity
        if ci not in self._pools:
            self._pools[ci] = sorted((o for o in _legal_pool(db, ci) if db[o].edhrec_rank and not db[o].is_basic),
                                     key=lambda o: (db[o].edhrec_rank, db[o].name))
        ok = _gc_ok(db, deck, self.gc_limit)
        out = []
        for o in self._pools[ci]:
            if len(out) >= n:
                break
            if o in deck.qty or not ok(o):
                continue
            out.append(db[o].name)
        return out

    def propose(self, task: str, deck: Deck, rep: int = 0) -> dict:
        if task == "cut":
            return {"cuts": self.cuts(deck)}
        if task == "add":
            return {"adds": self.adds(deck)}
        return _swaps(self.cuts(deck), self.adds(deck, 10))
