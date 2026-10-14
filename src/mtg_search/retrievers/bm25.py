"""BM25 over ``doc_text`` (optionally with Scryfall oracle tags appended)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import top_k

# Keep mana/tap symbols ("{3}", "{T}", "{W/U}") as single tokens and keep 1-char words
# (numbers like "3" matter in card text). No stopword removal: "all"/"each"/"target"
# carry meaning in Magic rules text and BM25's idf already down-weights common words.
TOKEN_PATTERN = r"(?u)\{[^}\s]+\}|\b\w+\b"


class BM25Retriever:
    def __init__(
        self,
        texts: Sequence[str],
        *,
        tags: Sequence[Sequence[str] | None] | None = None,
        stem: bool = True,
        k1: float = 1.2,
        b: float = 0.75,
        name: str | None = None,
    ) -> None:
        import bm25s

        self._bm25s = bm25s
        self.stemmer = None
        if stem:
            import Stemmer

            self.stemmer = Stemmer.Stemmer("english")
        corpus = list(texts)
        if tags is not None:
            corpus = [
                t + ("\n" + " ".join(tg.replace("-", " ") for tg in tg_list) if tg_list else "")
                for t, tg_list in zip(corpus, tags, strict=True)
            ]
        self.n = len(corpus)
        self.name = name or ("bm25+tags" if tags is not None else "bm25")
        tokens = self._tokenize(corpus)
        self.model = bm25s.BM25(k1=k1, b=b)
        self.model.index(tokens, show_progress=False)

    def _tokenize(self, texts: list[str]):
        return self._bm25s.tokenize(
            texts,
            lower=True,
            token_pattern=TOKEN_PATTERN,
            stopwords=None,
            stemmer=self.stemmer,
            return_ids=True,
            show_progress=False,
        )

    @classmethod
    def from_parquet(cls, path, *, with_tags: bool = False, **kw) -> BM25Retriever:
        import polars as pl

        cols = ["doc_text", "tags"] if with_tags else ["doc_text"]
        df = pl.read_parquet(path, columns=cols)
        return cls(df["doc_text"].to_list(), tags=df["tags"].to_list() if with_tags else None, **kw)

    def scores(self, query: str) -> np.ndarray:
        q = self._bm25s.tokenize(
            [query],
            lower=True,
            token_pattern=TOKEN_PATTERN,
            stopwords=None,
            stemmer=self.stemmer,
            return_ids=False,
            show_progress=False,
        )[0]
        # Unknown query tokens would raise inside bm25s; drop them.
        vocab = self.model.vocab_dict
        q = [t for t in q if t in vocab]
        if not q:
            return np.zeros(self.n, dtype=np.float32)
        return self.model.get_scores(q)

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        hits = top_k(self.scores(query), k, allowed)
        return [(i, s) for i, s in hits if s > 0]
