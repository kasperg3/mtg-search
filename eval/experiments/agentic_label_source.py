"""EXP-006 follow-up: does the lexical agent (F) beat mLateOn (A) because of how the labels were made?

    uv run python eval/experiments/agentic_label_source.py

The 65 deck tasks have two label sources:
  * ``deckneeds`` (30): Scryfall oracle tags OR regexes over doc_text (eval/build_deckneeds.py);
  * ``paraphrase-need`` (35): regexes over doc_text only (eval/build_handwritten.py).
F's ``grep`` tool searches the same doc_text with regexes, so on regex-labelled tasks it can partly
reproduce the labelling rule. This script splits the reported A-vs-F final recall@20 (analyze.py:
hits in final 20 / |relevant|) and P@10 by label source, paired over tasks both conditions finished.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def boot(d: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(d)
    m = rng.choice(a, (n, len(a))).mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    tasks = {t["id"]: t for t in yaml.safe_load(open(ROOT / "eval/agentic/tasks.yaml")) if t["kind"] == "deck"}
    fin: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    for line in open(ROOT / "eval/agentic/runs/main/episodes.jsonl"):
        e = json.loads(line)
        if e["kind"] == "deck" and e.get("final_oids") is not None:
            fin[e["cond"]][e["task_id"]] = e["final_oids"]
    common = sorted(set(fin["A"]) & set(fin["F"]))
    for source in ("deckneeds", "paraphrase-need", None):
        ids = [t for t in common if source is None or tasks[t]["source"] == source]
        for metric, depth in (("recall@20_final", 20), ("p@10", 10)):
            d = []
            for t in ids:
                rel = set(tasks[t]["relevant"])
                v = {c: sum(o in rel for o in fin[c][t][:depth]) / (len(rel) if depth == 20 else 10) for c in "AF"}
                d.append(100 * (v["F"] - v["A"]))
            m, lo, hi = boot(d)
            print(f"{source or 'all':16s} n={len(ids):2d}  F-A {metric:16s} {m:+.1f} [{lo:+.1f}, {hi:+.1f}]"
                  f"  F better {sum(x > 0 for x in d)}, A better {sum(x < 0 for x in d)}")


if __name__ == "__main__":
    main()
