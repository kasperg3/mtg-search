"""EXP-006 search environment: retrievers, compact card rendering and SparseUp term feedback.

Backends (``retriever`` names used by conditions):
  ``mlateon``          LateInMemory(lightonai/mLateOn)
  ``sparseup``         SparseUpRetriever (card-text index)

Term feedback is always computed with SparseUp, whatever retriever produced the results, so
the same aggregate line can be shown over mLateOn results (condition D) and logged (hidden)
for A/B as a counterfactual baseline for the "did the agent use the terms" analysis.
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

INDEX_DIR = ROOT / "index" / "embeddings"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
TOP_K = 15
ORACLE_CHARS = 180
FILTER_KEYS = ("color_identity", "colors", "colors_mode", "types", "cmc_min", "cmc_max", "format")
_WORD = re.compile(r"^[a-z][a-z'-]{2,}$")
# no popularity / price / crowd signal may reach the agent (asserted on every tool result)
LEAK = re.compile(r"edhrec|price|\$|rank|popular|salt", re.IGNORECASE)
# generic rules words that every card shares; never useful as steering feedback
ENGLISH_STOP = set(
    "a about above after again against all also am an and any are as at be because been before being below between both but by "
    "can could did do does doing down during each few for from further had has have having he her here hers him his how if in "
    "into is it its itself just more most must no nor not now of off on once only or other our out over own same she should so "
    "some such than that their theirs them then there these they this those through to too under until up very was we were what "
    "when where which while who whom why will with would yet you your yours".split()
)
MIN_CORPUS_DF = 5  # a feedback term must occur as a standalone word in at least this many cards' text
TERM_STOP = ENGLISH_STOP | set(
    "the and you your this that with for from into its card cards target each player players creature creatures "
    "turn end may put get gets one two three until control controller spell spells ability abilities battlefield "
    "mana cost costs pay additional instead then also any can't cant don't other another".split()
)


def _norm_filters(f: dict | None) -> dict:
    out = {}
    for k, v in (f or {}).items():
        if k in FILTER_KEYS and v not in (None, "", [], {}):
            out[k] = v
    return out


class Env:
    def __init__(self, retrievers: tuple[str, ...] = ("mlateon", "sparseup")):
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
        from mtg_search.retrievers.sparseup import MODEL_ID, SparseUpRetriever

        self.sparseup = SparseUpRetriever.from_dir(INDEX_DIR, MODEL_ID, doc_texts=self.cards["doc_text"].to_list())
        # memoize query encodings: the same query is encoded for search and for term feedback
        _enc, _memo = self.sparseup.encode_query, {}

        def enc(q: str):
            if q not in _memo:
                _memo[q] = _enc(q)
            return _memo[q]

        self.sparseup.encode_query = enc
        tok = self.sparseup.tokenizer
        vocab = tok.convert_ids_to_tokens(list(range(self.sparseup.csr.shape[1])))
        self.vocab = vocab
        # whole-word, alphabetic, non-stopword vocab dims are eligible as feedback terms
        # Feedback-term filter (identical for every condition; tightened after the pilot showed the
        # aggregate line was dominated by subword fragments): whole-word vocab items only (Ġ-prefixed,
        # alphabetic, >= 3 letters), no stopwords, no popularity/price vocabulary, and the word must
        # occur as a standalone word in >= MIN_CORPUS_DF cards' doc_text.
        from collections import Counter

        corpus_df = Counter()
        for d in self.cards.filter(~pl.col("is_funny"))["doc_text"].to_list():
            corpus_df.update(set(re.findall(r"[a-z]+", d.lower())))
        self.term_ok = np.array(
            [
                t.startswith("Ġ") and bool(re.fullmatch(r"[a-z]{3,}", t[1:].lower())) and t[1:].lower() not in TERM_STOP
                and not LEAK.search(t[1:]) and corpus_df.get(t[1:].lower(), 0) >= MIN_CORPUS_DF
                for t in vocab
            ]
        )
        df = np.diff(self.sparseup.docs.indptr)
        self.idf = np.log(self.sparseup.n / (1.0 + df)).astype(np.float32)
        for name in retrievers:
            self.ret[name] = self._load(name)
        CACHE_DIR.mkdir(exist_ok=True)
        self._name_index()

    def _load(self, name: str):
        if name == "sparseup":
            return self.sparseup
        if name == "bm25":
            from mtg_search.retrievers.bm25 import BM25Retriever

            return BM25Retriever.from_parquet(ROOT / "data" / "cards.parquet")
        if name == "mlateon":
            from mtg_search.retrievers.late import LateInMemory

            return LateInMemory.from_dir(INDEX_DIR, "lightonai/mLateOn")
        raise ValueError(name)

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

    def grep(self, pattern: str, agent_filters: dict | None, task_filters: dict | None, k: int = TOP_K):
        """Case-insensitive regex over doc_text (reminder text included). -> (n_matches, rows, matched
        snippets). Matching cards are ordered by BM25 score of the pattern's literal words (regex
        syntax stripped) over card text, ties broken alphabetically by name. No popularity signal."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        allowed = self.mask(task_filters or {}, agent_filters or {})
        if not hasattr(self, "_docs"):
            self._docs = self.cards["doc_text"].to_list()
        match = {}
        for i in np.flatnonzero(allowed):
            m = rx.search(self._docs[i])
            if m:
                match[int(i)] = m
        words = " ".join(re.findall(r"[A-Za-z][A-Za-z'+/-]*", re.sub(r"\\[a-zA-Z]", " ", pattern)))
        bm = np.zeros(len(self._docs), dtype=np.float32)
        if words and match:
            mask = np.zeros(len(self._docs), dtype=bool)
            mask[list(match)] = True
            for r, sc in self.ret["bm25"].search(words, len(match), mask):
                bm[r] = sc
        order = sorted(match, key=lambda i: (-bm[i], self.names[i]))
        snippets = []
        for i in order[:k]:
            m, doc = match[i], self._docs[i]
            a, b = doc.rfind("\n", 0, m.start()) + 1, doc.find("\n", m.end())
            line = doc[a : b if b >= 0 else len(doc)]
            s0, s1 = m.start() - a, m.end() - a
            snip = line[:s0] + "[[" + line[s0:s1] + "]]" + line[s1:]
            snippets.append(snip if len(snip) <= 200 else "…" + snip[max(0, s0 - 80) : s1 + 100] + "…")
        return len(match), order[:k], snippets

    # ------------------------------------------------------------------ term feedback (SparseUp)

    def _qvec(self, query: str):
        with self.lock:
            return self.sparseup.encode_query(query)  # memoized

    def query_literal(self, query: str) -> set[int]:
        return self.sparseup._literal_dims(query, self.sparseup.query_max_length)

    def _qwords(self, query: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", query.lower()) if len(w) > 2}

    def _same_word(self, term: str, qwords: set[str]) -> bool:
        # crude stemming so "fliers"/"flier" or "deck"/"decks" don't count as new terms
        return any(term[:5] == w[:5] and abs(len(term) - len(w)) <= 3 for w in qwords if len(w) >= 4) or term in qwords

    def per_result_terms(self, query: str, rows: list[int], n: int = 5) -> list[list[tuple[str, bool]]]:
        """Top contributing whole-word SparseUp terms (q[t]*d[t]) per result; flag = expansion
        (term is not a word of the query)."""
        ids, w = self._qvec(query)
        qw = dict(zip(ids.tolist(), w.tolist(), strict=True))
        qlit, qwords = self.query_literal(query), self._qwords(query)
        out = []
        for row in rows:
            rw = self.sparseup.csr.getrow(row)
            contrib = [(qw[t] * d, t) for t, d in zip(rw.indices.tolist(), rw.data.tolist(), strict=True) if t in qw and self.term_ok[t]]
            contrib.sort(reverse=True)
            terms = []
            for _, t in contrib[:n]:
                word = self.vocab[t][1:].lower()
                terms.append((word, t not in qlit and not self._same_word(word, qwords)))
            out.append(terms)
        return out

    def aggregate_terms(self, query: str, rows: list[int], n: int = 15, prior_text: str = "") -> list[tuple[str, int]]:
        """Frequent, distinctive terms in the results' SparseUp document vectors that are not in the
        query: score(t) = idf(t) * sum_r d_r[t], over terms present in >= 2 results."""
        if not rows:
            return []
        sub = self.sparseup.csr[rows]
        dfr = np.asarray((sub > 0).sum(axis=0)).ravel()
        mass = np.asarray(sub.sum(axis=0)).ravel()
        score = mass * self.idf * self.term_ok * (dfr >= 2)
        qlit, qwords = self.query_literal(query), self._qwords(query)
        out = []
        for t in np.argsort(-score):
            if score[t] <= 0 or len(out) >= n:
                break
            word = self.vocab[t][1:].lower()
            if int(t) in qlit or self._same_word(word, qwords) or any(word == o for o, _ in out):
                continue
            out.append((word, int(dfr[t])))
        return out
