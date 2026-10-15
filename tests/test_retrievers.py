"""Retriever tests on the ~20-card fixture.

Fast tests (BM25, top_k, MaxSim math) run by default. Tests that download/run models are
marked ``models`` and deselected by default; run them with ``uv run pytest -m models``
(needs ``uv sync --extra models`` and network access to the HF Hub).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mtg_search.retrievers.base import model_slug, oracle_ids_digest, top_k

ROOT = Path(__file__).resolve().parents[1]

QUERIES = {
    "counter target spell": "Counterspell",
    "deals 3 damage to any target": "Lightning Bolt",
    "sacrifice a creature: scry 1": "Viscera Seer",
}


def names_of(cards_df, hits):
    return [cards_df["name"][i] for i, _ in hits]


def load_encode_module():
    spec = importlib.util.spec_from_file_location("encode_job", ROOT / "jobs" / "encode.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- fast


def test_top_k_respects_mask_and_order():
    s = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert [i for i, _ in top_k(s, 3)] == [1, 3, 2]
    mask = np.array([True, False, True, True])
    assert [i for i, _ in top_k(s, 10, mask)] == [3, 2, 0]
    assert top_k(s, 0) == []
    assert top_k(s, 5, np.zeros(4, dtype=bool)) == []


def test_slug_and_digest():
    assert model_slug("lightonai/mLateOn") == "lightonai__mLateOn"
    assert oracle_ids_digest(["a", "b"]) != oracle_ids_digest(["b", "a"])
    enc = load_encode_module()
    assert enc.oracle_ids_digest(["x", "y"]) == oracle_ids_digest(["x", "y"])
    assert enc.model_slug("a/b") == model_slug("a/b")


def test_maxsim_matches_bruteforce():
    torch = pytest.importorskip("torch")
    from mtg_search.retrievers.late import maxsim_scores

    rng = np.random.default_rng(0)
    lens = [3, 1, 5, 2]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tokens = rng.standard_normal((offsets[-1], 8)).astype(np.float16)
    q = rng.standard_normal((4, 8)).astype(np.float32)
    got = maxsim_scores(torch.from_numpy(tokens), offsets, q, docs_per_chunk=3)
    want = [(q @ tokens[a:b].astype(np.float32).T).max(axis=1).sum() for a, b in zip(offsets[:-1], offsets[1:])]
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_bucketed_maxsim_matches_reference():
    torch = pytest.importorskip("torch")
    from mtg_search.retrievers.late import build_buckets, maxsim_scores

    rng = np.random.default_rng(1)
    lens = rng.integers(1, 70, size=50)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tokens = torch.from_numpy(rng.standard_normal((offsets[-1], 8)).astype(np.float32))
    q = rng.standard_normal((5, 8)).astype(np.float32)
    want = maxsim_scores(tokens, offsets, q)
    got = np.empty_like(want)
    for rows, block in build_buckets(tokens, offsets, step=16):
        got[rows] = (block @ torch.from_numpy(q).T).amax(dim=1).sum(dim=1).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def bm25(cards_df):
    pytest.importorskip("bm25s")
    from mtg_search.retrievers.bm25 import BM25Retriever

    return BM25Retriever(cards_df["doc_text"].to_list())


def test_bm25_finds_obvious_cards(bm25, cards_df):
    for q, want in QUERIES.items():
        assert names_of(cards_df, bm25.search(q, 3))[0] == want, q


def test_bm25_keeps_mana_symbols_and_numbers(bm25, cards_df):
    import re

    from mtg_search.retrievers.bm25 import TOKEN_PATTERN

    assert re.findall(TOKEN_PATTERN, "pays {3} and {T}: add {C}{C}") == ["pays", "{3}", "and", "{T}", "add", "{C}", "{C}"]
    hits = names_of(cards_df, bm25.search("{T}: Add {C}{C}", 3))
    assert hits[0] == "Sol Ring"


def test_bm25_allowed_mask(bm25, cards_df):
    mask = np.ones(len(cards_df), dtype=bool)
    mask[cards_df["name"].to_list().index("Counterspell")] = False
    assert "Counterspell" not in names_of(cards_df, bm25.search("counter target spell", 10, mask))


def test_bm25_unknown_query_tokens(bm25):
    assert bm25.search("zzzzqqq", 5) == []


def test_bm25_tags_variant(cards_df):
    pytest.importorskip("bm25s")
    from mtg_search.retrievers.bm25 import BM25Retriever

    r = BM25Retriever(cards_df["doc_text"].to_list(), tags=cards_df["tags"].to_list())
    assert r.name == "bm25+tags"
    assert names_of(cards_df, r.search("free sacrifice outlet", 3))[0] == "Viscera Seer"


# --------------------------------------------------------------------------- models


@pytest.fixture(scope="module")
def encoded(tmp_path_factory, cards_df):
    """Encode the fixture with jobs/encode.py's own code path (CPU)."""

    enc = load_encode_module()
    out = tmp_path_factory.mktemp("enc")
    cards = out / "cards.parquet"
    cards_df.select("oracle_id", "doc_text").write_parquet(cards)
    done = set()

    def run(kind, model):
        if (kind, model) not in done:
            enc.main(["--kind", kind, "--model", model, "--cards", str(cards), "--out", str(out), "--no-upload", "--device", "cpu"])
            done.add((kind, model))
        return out

    return run


def check_retriever(r, cards_df):
    for q, want in QUERIES.items():
        hits = r.search(q, 5)
        assert names_of(cards_df, hits)[0] == want, (q, names_of(cards_df, hits))
        assert all(a[1] >= b[1] for a, b in zip(hits, hits[1:]))
    mask = np.ones(len(cards_df), dtype=bool)
    mask[cards_df["name"].to_list().index("Counterspell")] = False
    assert "Counterspell" not in names_of(cards_df, r.search("counter target spell", 50, mask))


@pytest.mark.models
@pytest.mark.parametrize("model", ["answerdotai/answerai-colbert-small-v1", "lightonai/mLateOn"])
def test_late(encoded, cards_df, model):
    from mtg_search.retrievers.late import LateInMemory

    root = encoded("late", model)
    r = LateInMemory.from_dir(root, model)
    assert r.n == len(cards_df)
    check_retriever(r, cards_df)
    # Subset path (gathering allowed docs' tokens) must agree with the full path.
    mask = np.zeros(len(cards_df), dtype=bool)
    mask[[0, 3, 5, 7]] = True
    full = r.scores("destroy target creature")
    sub = r.scores("destroy target creature", mask)
    np.testing.assert_allclose(sub[mask], full[mask], rtol=1e-5)
    assert np.isneginf(sub[~mask]).all()


@pytest.mark.models
def test_dense_embeddinggemma(encoded, cards_df):
    from mtg_search.retrievers.dense import DenseRetriever

    model = "google/embeddinggemma-2"
    root = encoded("dense", model)
    r = DenseRetriever.from_dir(root, model)
    assert r.query_prompt == "task: search result | query: "
    assert r.meta["doc_prompt"] == "title: none | text: "
    check_retriever(r, cards_df)


@pytest.mark.models
def test_sparse_inference_free_query_matches_sentence_transformers(encoded, cards_df):
    from mtg_search.retrievers.sparse import SparseRetriever

    model = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
    root = encoded("sparse", model)
    r = SparseRetriever.from_dir(root, model)
    check_retriever(r, cards_df)
    # Our tokenizer+weights query encoder == SparseEncoder.encode_query.
    enc = load_encode_module()
    st = enc.load_model("sparse", model, "cpu")
    q = "whenever another creature you control dies, draw a card"
    ref = st.encode_query([q], convert_to_sparse_tensor=False)[0].numpy()
    ids, w = r.encoder.encode(q)
    mine = np.zeros_like(ref)
    mine[ids] = w
    np.testing.assert_allclose(mine, ref, rtol=1e-6)
    # And the repaired doc encoder reproduces the model card's reference score.
    sim = st.similarity(st.encode_query(["What's the weather in ny now?"]), st.encode_document(["Currently New York is rainy."]))
    assert abs(float(sim) - 12.5747) < 1e-3


def test_from_dir_checks_card_table_alignment(tmp_path):
    """Review m6: with the card table's ids, loaders refuse embeddings built for another table/order."""
    import json

    from mtg_search.retrievers.base import check_alignment, load_meta, oracle_ids_digest

    ids = ["a", "b", "c"]
    meta = {"card_count": 3, "oracle_ids_sha256": oracle_ids_digest(ids)}
    (tmp_path / "m.json").write_text(json.dumps(meta))
    assert load_meta(tmp_path / "m.json", ids) == meta
    check_alignment(meta, ids)
    with pytest.raises(RuntimeError, match="different card table"):
        check_alignment(meta, ["b", "a", "c"])  # same cards, other row order
    with pytest.raises(RuntimeError, match="different card table"):
        load_meta(tmp_path / "m.json", ["a", "b"])
    with pytest.raises(RuntimeError, match="no oracle_ids_sha256"):
        check_alignment({"card_count": 3}, ids)
