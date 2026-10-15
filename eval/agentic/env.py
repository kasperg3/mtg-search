"""Agentic search environment (EXP-006/013): mLateOn search, plain grep, compact card rendering, leak guard.

Conditions use ``search("mlateon", ...)`` (A: search infrastructure) or ``grep`` (G: none). The SparseUp and
BM25 conditions of EXP-006 (B, C, D, F) were removed with those retrievers; their code is in git history
up to commit 61071da and their results stay in eval/agentic/runs/main and agentic_search.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mtg_search.filters import filter_expr  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
TOP_K = 15
ORACLE_CHARS = 180
FILTER_KEYS = ("color_identity", "colors", "colors_mode", "types", "cmc_min", "cmc_max", "format")
# no popularity / price / crowd signal may reach the agent (asserted on every tool result)
LEAK = re.compile(r"edhrec|price|\$|rank|popular|salt", re.IGNORECASE)


def _norm_filters(f: dict | None) -> dict:
    out = {}
    for k, v in (f or {}).items():
        if k in FILTER_KEYS and v not in (None, "", [], {}):
            out[k] = v
    return out


class Env:
    def __init__(self, retrievers: tuple[str, ...] = ("mlateon",)):
        self.cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
        self.oids = self.cards["oracle_id"].to_list()
        # result-cache key includes the corpus order, so a refreshed cards.parquet never reuses stale row ids
        self.corpus_digest = hashlib.sha1("\n".join(self.oids).encode()).hexdigest()[:12]
        self.names = self.cards["name"].to_list()
        self.row_of = {o: i for i, o in enumerate(self.oids)}
        self.base_mask = ~self.cards["is_funny"].to_numpy().astype(bool)
        self._render = [self._render_row(r) for r in self.cards.iter_rows(named=True)]
        self.lock = threading.Semaphore(2)
        self.ret: dict[str, object] = {}
        for name in retrievers:
            if name != "mlateon":
                raise ValueError(f"unknown retriever {name!r} (only mlateon)")
            from mtg_search.mlateon import MLateOn

            self.ret[name] = MLateOn.local(self.oids)
        CACHE_DIR.mkdir(exist_ok=True)
        self._name_index()

    # ------------------------------------------------------------------ rendering

    @staticmethod
    def _render_row(r: dict) -> str:
        faces = r.get("faces") or [{"name": r["name"], "mana_cost": r["mana_cost"], "type_line": r["type_line"], "oracle_text": ""}]
        cost = " // ".join(f["mana_cost"] for f in faces if f.get("mana_cost")) or r.get("mana_cost") or ""
        text = " // ".join((f.get("oracle_text") or "").replace("\n", " ") for f in faces).strip(" /")
        text = re.sub(r"\([^)]*\)", "", text)  # reminder text
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > ORACLE_CHARS:
            text = text[: ORACLE_CHARS - 1].rstrip() + "…"
        pt = f" {faces[0]['power']}/{faces[0]['toughness']}" if faces[0].get("power") is not None else ""
        return f"{r['name']} {cost} — {r['type_line']}{pt} — {text}"

    def render(self, row: int) -> str:
        return self._render[row]

    def assert_no_leak(self, text: str, rows: list[int], snippets=None) -> None:
        """Card renders contain only name, cost, type, P/T and rules text (by construction), whose
        wording may legitimately contain e.g. "rank"/"salt" in names; everything else the harness
        writes (headers, legends, term lists) must not match LEAK."""
        rest = text
        for r in rows:
            rest = rest.replace(self._render[r], "")
        for sn in snippets or []:
            rest = rest.replace(sn, "")
        if LEAK.search(rest):
            raise AssertionError(f"popularity-like signal in tool output: {LEAK.search(rest).group(0)!r}")

    def _name_index(self) -> None:
        self.by_name: dict[str, str] = {}
        for r in self.cards.select("oracle_id", "name", "face_names", "is_funny").iter_rows(named=True):
            if r["is_funny"]:
                continue
            for n in [r["name"], *(r["face_names"] or [])]:
                self.by_name.setdefault(n.lower().strip(), r["oracle_id"])

    def resolve_name(self, name: str) -> str | None:
        import difflib

        key = re.sub(r"\s+", " ", str(name)).lower().strip().strip("\"'")
        key = re.sub(r"\s*\{.*$", "", key)  # agent sometimes copies the cost
        key = key.split(" — ")[0].strip()
        if key in self.by_name:
            return self.by_name[key]
        m = difflib.get_close_matches(key, self.by_name.keys(), n=1, cutoff=0.9)
        return self.by_name[m[0]] if m else None

    # ------------------------------------------------------------------ search

    def mask(self, *filter_dicts: dict) -> np.ndarray:
        m = self.base_mask.copy()
        for f in filter_dicts:
            f = _norm_filters(f)
            if f:
                m &= self.cards.select(filter_expr(self.cards, **f)).to_series().to_numpy().astype(bool)
        return m

    def search(self, retriever: str, query: str, agent_filters: dict | None, task_filters: dict | None, k: int = TOP_K):
        """-> list of rows. Cached on disk per (retriever, query, filters)."""
        key = json.dumps([self.corpus_digest, retriever, query, _norm_filters(agent_filters), _norm_filters(task_filters), k], sort_keys=True)
        path = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text())
        try:
            allowed = self.mask(task_filters or {}, agent_filters or {})
        except Exception as e:  # bad agent filter value
            raise ValueError(f"invalid filters: {e}") from e
        with self.lock:
            hits = self.ret[retriever].search(query, k, allowed)
        rows = [int(r) for r, _ in hits]
        path.write_text(json.dumps(rows))
        return rows

    def grep(self, pattern: str, task_filters: dict | None, k: int = TOP_K, offset: int = 0):
        """Condition G (no search infrastructure): case-insensitive regex over doc_text, matched line by line
        as in grep. -> (n_matches, rows, matched snippets) in card-file order (alphabetical by name; cards.parquet
        is sorted by name), from ``offset`` on. No ranking, no popularity signal."""
        try:
            rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        allowed = self.mask(task_filters or {})
        if not hasattr(self, "_docs"):
            self._docs = self.cards["doc_text"].to_list()
        match = {}
        for i in np.flatnonzero(allowed):
            m = rx.search(self._docs[i])
            if m:
                match[int(i)] = m
        rows = sorted(match)[max(0, offset):][:k]
        return len(match), rows, [self._snippet(match[i], self._docs[i]) for i in rows]

    @staticmethod
    def _snippet(m: re.Match, doc: str) -> str:
        """The matching line with the match in [[ ]] (long lines cut around the match)."""
        a, b = doc.rfind("\n", 0, m.start()) + 1, doc.find("\n", m.end())
        line = doc[a : b if b >= 0 else len(doc)]
        s0, s1 = m.start() - a, m.end() - a
        snip = line[:s0] + "[[" + line[s0:s1] + "]]" + line[s1:]
        return snip if len(snip) <= 200 else "…" + snip[max(0, s0 - 80) : s1 + 100] + "…"



def git_state(paths=("eval/agentic", "src", "bench/precon/harness")) -> dict:
    """Commit and whether the harness has uncommitted changes (recorded in config.json; review m4)."""
    import subprocess

    root = Path(__file__).resolve().parents[2]

    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {"sha": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no", "--", *paths))}
