"""Exact late-interaction backend (EXP-012): MaxSim, batching, filtering and safety checks, with a stub
query encoder (no ONNX model needed)."""

import json

import numpy as np
import pytest

from mtg_search.exact import ExactLateBackend


def _backend(n_docs=300, dim=8, seed=0, threads=3):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, 9, n_docs)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tokens = rng.standard_normal((offsets[-1], dim)).astype(np.float32)
    tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)
    be = object.__new__(ExactLateBackend)
    be.tokens, be.offsets, be.n = tokens, offsets, n_docs
    be._init_scoring(threads)
    qs = {}

    def encode(q):
        if q not in qs:
            v = np.random.default_rng(abs(hash(q)) % 2**32).standard_normal((len(q) % 5 + 2, dim)).astype(np.float32)
            qs[q] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return qs[q]

    be.encode_query = encode
    return be


def _brute(be, q):
    v = be.encode_query(q)
    return np.array([(be.tokens[be.offsets[i]:be.offsets[i + 1]] @ v.T).max(axis=0).sum() for i in range(be.n)])


def test_maxsim_matches_brute_force_and_batching():
    be = _backend()
    qs = ["destroy target creature", "draw a card", "x"]
    many = be.scores_many(qs)
    for i, q in enumerate(qs):
        np.testing.assert_allclose(many[i], _brute(be, q), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(be.scores(q), many[i], rtol=1e-6, atol=1e-6)
    assert len(be._chunks) > 1  # really exercised the chunked path


def test_search_filters_before_top_k_and_orders_deterministically():
    be = _backend()
    s = be.scores("removal")
    allowed = np.arange(0, be.n, 7)
    hits = be.search("removal", 5, allowed)
    assert [r for r, _ in hits] == list(allowed[np.argsort(-s[allowed], kind="stable")][:5])
    assert all(r % 7 == 0 for r, _ in hits)
    assert be.search("removal", 5, np.array([], dtype=np.int64)) == []
    assert len(be.search("removal", 10_000)) == be.n
    # exact ties rank by row (chunked BLAS sums can differ by ~1e-7, so test the ranking step directly)
    assert [r for r, _ in be._top(np.zeros(be.n, dtype=np.float32), 4, None)] == [0, 1, 2, 3]
    assert [r for r, _ in be._top(np.zeros(be.n, dtype=np.float32), 3, np.array([9, 4, 7]))] == [4, 7, 9]


def test_search_many_equals_individual_searches():
    be = _backend(seed=3)
    qs = ["a", "bb", "ccc", "dddd"]
    for batched, single in zip(be.search_many(qs, 20), [be.search(q, 20) for q in qs], strict=True):
        assert [r for r, _ in batched] == [r for r, _ in single]
        np.testing.assert_allclose([x for _, x in batched], [x for _, x in single], rtol=1e-5)


def test_refuses_mismatched_embeddings(tmp_path):
    emb, model = tmp_path / "emb", tmp_path / "model"
    emb.mkdir()
    model.mkdir()
    (emb / "meta.json").write_text(json.dumps({"card_count": 2, "oracle_ids_sha256": "nope", "revision": "r1"}))
    with pytest.raises(RuntimeError, match="do not match cards.parquet"):
        ExactLateBackend(["a", "b"], embeddings_dir=emb, model_dir=model)
    from mtg_search.artifacts import oracle_ids_digest
    (emb / "meta.json").write_text(json.dumps({"card_count": 2, "oracle_ids_sha256": oracle_ids_digest(["a", "b"]),
                                               "revision": "r1"}))
    (model / "source.json").write_text(json.dumps({"revision": "r2"}))
    with pytest.raises(RuntimeError, match="one checkpoint"):
        ExactLateBackend(["a", "b"], embeddings_dir=emb, model_dir=model)
