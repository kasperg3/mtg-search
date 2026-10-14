"""EXP-004 explainability demo: per-term decomposition of SPARSEUP scores.

    uv run --extra models python eval/experiments/sparseup_explain.py [--top-n 8] [--json out.json]

For each demo query: the query's own top expansion terms, then for the top-3 cards the
terms contributing most to q·d. Marks: ``[+q]`` = term not literally in the query
(query-side expansion), ``[+d]`` = not literally in the card text (document-side
expansion), ``·x`` = word-internal subword piece. Also prints index statistics
(nnz/doc, index size, query nnz over all eval queries).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.queries import gains, load_queries  # noqa: E402
from mtg_search.filters import filter_expr  # noqa: E402
from mtg_search.retrievers.sparseup import MODEL_ID, SparseUpRetriever, format_explanation  # noqa: E402

INDEX_DIR = ROOT / "index" / "embeddings"
DEMO = [
    ("paraphrase", "hw-soft-counter-para"),  # "counter they can pay around"
    ("paraphrase", "tag-removal-sacrifice-para"),  # "forces them to give up one of their own guys"
    ("deckneeds", None, "I keep losing to fliers in my mono-green deck"),
    ("deckneeds", None, "something to punish opponents who play lots of spells"),
    ("deckneeds", None, "I need ways to get value from creatures dying in an aristocrats deck"),
    ("handwritten", "hw-soft-counter"),
    ("handwritten", "hw-one-blocker-max"),
    ("tags", None, "free way to sacrifice my own creatures whenever I want"),
]


def find(qs: dict, spec) -> tuple[str, dict]:
    st, qid, *text = spec
    for q in qs[st]:
        if (qid and q["id"] == qid) or (text and q["query"] == text[0]):
            return st, q
    raise KeyError(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
    names, texts = cards["name"].to_list(), cards["doc_text"].to_list()
    oids = cards["oracle_id"].to_list()
    r = SparseUpRetriever.from_dir(INDEX_DIR, MODEL_ID, doc_texts=texts)
    qs = {s: load_queries(s) for s in ("tags", "handwritten", "paraphrase", "deckneeds")}

    # ---------------------------------------------------------------- index stats
    nnz = np.diff(r.csr.indptr)
    npz = INDEX_DIR / "sparse" / "Linkup-Platform__linkup-sparseup-embed-v1.npz"
    gte = INDEX_DIR / "sparse" / "opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.npz"
    qnnz = [r.encode_query(q["query"])[0].size for s in qs for q in qs[s]]
    stats = {
        "docs": int(r.n), "vocab": int(r.csr.shape[1]), "active_vocab_dims": int((np.diff(r.docs.indptr) > 0).sum()),
        "nnz_per_doc_mean": float(nnz.mean()), "nnz_per_doc_p50": float(np.median(nnz)), "nnz_per_doc_max": int(nnz.max()),
        "npz_mb": round(npz.stat().st_size / 1e6, 1), "gte_npz_mb": round(gte.stat().st_size / 1e6, 1) if gte.exists() else None,
        "csr_ram_mb": round((r.csr.data.nbytes + r.csr.indices.nbytes + r.csr.indptr.nbytes) / 1e6, 1),
        "query_nnz_mean": float(np.mean(qnnz)), "query_nnz_p50": float(np.median(qnnz)), "query_nnz_max": int(np.max(qnnz)),
    }
    print("index:", json.dumps(stats))

    # literal-vs-expansion share over the whole demo
    out = {"stats": stats, "queries": []}
    exp_d = exp_q = sub = tot = 0
    for spec in DEMO:
        st, q = find(qs, spec)
        allowed = cards.select(filter_expr(cards, **q["filters"])).to_series().to_numpy().astype(bool)
        g = gains(q)
        ids, w = r.encode_query(q["query"])
        order = np.argsort(-w)[:12]
        qlit = r._literal_dims(q["query"], r.query_max_length)
        qterms = [f"{r.display(r.tokenizer.convert_ids_to_tokens(int(ids[i])))}{'' if int(ids[i]) in qlit else '[+q]'} {w[i]:.2f}" for i in order]
        print(f"\n=== {st}/{q['id']}: “{q['query']}”" + (f"  filters={q['filters']}" if q["filters"] else ""))
        print(f"query vector ({ids.size} terms): " + ", ".join(qterms))
        rec = {"set": st, "id": q["id"], "query": q["query"], "query_terms": qterms, "hits": []}
        for rank, (row, score) in enumerate(r.search(q["query"], 3, allowed), 1):
            ex = r.explain_detail(q["query"], row, top_n=args.top_n)
            rel = {2: "✓✓", 1: "✓"}.get(g.get(oids[row], 0), "✗")
            print(f"  {rank}. {names[row]} [{rel}]  " + format_explanation(ex))
            rec["hits"].append({"rank": rank, "name": names[row], "relevant": g.get(oids[row], 0), **ex})
            for t in ex["terms"]:
                tot += 1
                exp_d += t["in_card"] is False
                exp_q += not t["in_query"]
                sub += t["subword"]
        out["queries"].append(rec)
    summ = {"terms_shown": tot, "doc_expansion_share": exp_d / tot, "query_expansion_share": exp_q / tot, "subword_share": sub / tot}
    out["summary"] = summ
    print("\nsummary:", json.dumps(summ))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
