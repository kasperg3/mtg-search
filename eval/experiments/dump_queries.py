"""Dump only {set, id, query, filters} for paraphrase + deckneeds, so rewrites are written blind to labels.

    uv run python eval/experiments/dump_queries.py > /tmp/queries_only.txt
"""
from pathlib import Path

import yaml

EVAL = Path(__file__).resolve().parents[1]
for s in ("paraphrase", "deckneeds"):
    for q in yaml.safe_load((EVAL / f"queries_{s}.yaml").read_text()):
        f = q.get("filters") or {}
        print(f"{s}\t{q['id']}\t{q['query']}" + (f"\t{f}" if f else ""))
