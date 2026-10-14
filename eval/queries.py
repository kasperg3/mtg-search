"""Loader for the retrieval evaluation query sets.

    from eval.queries import load_queries
    qs = load_queries("tags")   # "tags" | "handwritten" | "paraphrase" | "deckneeds"

Each query dict has at least
    id: str, query: str, relevant: list[str]  (oracle_ids, binary relevance)
and optionally
    graded: dict[oracle_id, int]  -- grade 2 = highly relevant; any relevant id
                                      not in `graded` has grade 1.
    filters: dict                 -- kwargs for mtg_search.filters.filter_expr
                                      (color_identity, colors, format, types,
                                      cmc_min, cmc_max); {} when absent. The runner
                                      should restrict retrieval to these cards.
                                      Labels already satisfy the filters.
    origin: str                   -- (paraphrase set) id of the original query.
Use `gains(q)` to get the {oracle_id: grade} mapping for nDCG.
"""

from __future__ import annotations

from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
SETS = {
    "tags": EVAL_DIR / "queries_tags.yaml",
    "handwritten": EVAL_DIR / "queries_handwritten.yaml",
    "paraphrase": EVAL_DIR / "queries_paraphrase.yaml",
    "deckneeds": EVAL_DIR / "queries_deckneeds.yaml",
}


def load_queries(name: str) -> list[dict]:
    """Load a query set by name (see SETS) or by YAML path."""
    path = SETS.get(name, Path(name))
    if not path.exists():
        raise FileNotFoundError(f"unknown query set {name!r}; known: {sorted(SETS)}")
    queries = yaml.safe_load(path.read_text()) or []
    for q in queries:
        q["graded"] = q.get("graded") or {}
        q["filters"] = q.get("filters") or {}
        if not q.get("relevant"):
            raise ValueError(f"{path.name}:{q.get('id')}: no relevant cards")
    return queries


def gains(query: dict) -> dict[str, int]:
    """oracle_id -> graded relevance (1 by default, 2 if listed in `graded`)."""
    g = {o: 1 for o in query["relevant"]}
    g.update({o: int(v) for o, v in (query.get("graded") or {}).items()})
    return g


if __name__ == "__main__":  # sanity check: uv run python eval/queries.py
    import sys

    import polars as pl

    sys.path.insert(0, str(EVAL_DIR.parent / "src"))
    from mtg_search.filters import filter_expr

    cards = pl.read_parquet(EVAL_DIR.parent / "data" / "cards.parquet")
    known = set(cards["oracle_id"])
    funny = set(cards.filter(pl.col("is_funny"))["oracle_id"])
    for name in SETS:
        if not SETS[name].exists():
            print(f"{name}: (missing)")
            continue
        qs = load_queries(name)
        ids = [q["id"] for q in qs]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        missing = {o for q in qs for o in gains(q) if o not in known}
        assert not missing, f"{name}: {len(missing)} unknown oracle_ids"
        bad = {q["id"] for q in qs if set(gains(q)) & funny}
        assert not bad, f"{name}: funny cards labelled in {sorted(bad)}"
        for q in qs:
            if q["filters"]:
                ok = set(cards.filter(filter_expr(cards, **q["filters"]))["oracle_id"])
                assert set(q["relevant"]) <= ok, f"{q['id']}: labels violate filters"
        sizes = sorted(len(q["relevant"]) for q in qs)
        print(f"{name}: {len(qs)} queries, relevant min/median/max = {sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
        for q in qs:
            print(f"  {len(q['relevant']):4d}  {q['id']:34s} {q['query']}")
