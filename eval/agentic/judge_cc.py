"""Pooled-judging smoke test with Claude Code (issue #2): can a blind Sonnet judge fix the incomplete labels?

    uv run python eval/agentic/judge_cc.py --run judge-smoke --pool-runs cc-main --tasks <ids> [--workers 4]

The judge is headless Claude Code (``claude -p``, ``claude-sonnet-5-5``, subscription, no API key), with no
tools and an empty working directory (same isolation checks as the precon bridge). It uses the
pre-registered rubric and prompt of ``judge.py``: it sees only the request and the card text, in a seeded
random order, BATCH cards per call, never the condition, the labels or any popularity data.

Unlike ``judge.py`` (which judges only unlabelled pool cards), the smoke test also judges the pooled cards
that the original labels already mark relevant, to measure agreement with them. Every batch is judged
twice, in two different orders, to measure self-consistency. ``--decoys-random N`` / ``--decoys-retrieved N``
mix blind negative controls into the batches: random cards that meet the task's constraints (should be 0)
and cards the agents retrieved but did not pick (harder, relevance unknown).

Outputs in ``eval/agentic/runs/<run>/``: ``judgments.jsonl`` (one row per task, card and pass),
``transcripts/``, ``summary.json`` (agreement, consistency, label gaps, tokens and time, cost projection).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from cc_agent import load_tasks, usage_of  # noqa: E402
from judge import BATCH, JUDGE_SYSTEM, judge_prompt  # noqa: E402

from bench.precon.harness.claude_code import ClaudeCode  # noqa: E402

MODEL = "claude-sonnet-5-5"


def build_pool(run: str, task_ids: list[str]) -> dict[str, list[str]]:
    """Union of every episode's final top-20 per task (all conditions), labelled cards included."""
    p: dict[str, set[str]] = {t: set() for t in task_ids}
    for line in (HERE / "runs" / run / "episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["task_id"] in p:
            p[e["task_id"]] |= {o for o in e["final_oids"][:20] if o}
    return {t: sorted(v) for t, v in p.items()}


def parse_grades(text: str, n: int) -> dict[int, int]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        out = {int(g["i"]): int(g["g"]) for g in json.loads(m.group(0))["grades"]}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {}
    return {i: g for i, g in out.items() if 1 <= i <= n and g in (0, 1, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-runs", default="cc-main")
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--decoys-random", type=int, default=0)
    ap.add_argument("--decoys-retrieved", type=int, default=0)
    args = ap.parse_args()
    tasks = {t["id"]: t for t in load_tasks()}
    ids = args.tasks.split(",")
    pool = {}
    for r in args.pool_runs.split(","):
        for t, oids in build_pool(r, ids).items():
            pool[t] = sorted(set(pool.get(t, [])) | set(oids))
    kind = {(t, o): "pool" for t, oids in pool.items() for o in oids}
    if args.decoys_random or args.decoys_retrieved:
        import numpy as np

        from env import Env

        env = Env((), feedback=False)
        seen = {t: set() for t in ids}
        for r in args.pool_runs.split(","):
            for line in (HERE / "runs" / r / "episodes.jsonl").read_text().splitlines():
                e = json.loads(line)
                if e["task_id"] in seen:
                    seen[e["task_id"]] |= {o for s in e["searches"] for o in s.get("oids", [])}
        for t in ids:
            rel = set(tasks[t]["relevant"])
            rng = random.Random(f"decoy-{t}")
            hard = sorted(seen[t] - set(pool[t]) - rel)
            allowed = [env.oids[i] for i in np.flatnonzero(env.mask(tasks[t].get("filters") or {}))]
            easy = [o for o in rng.sample(allowed, min(len(allowed), 50)) if o not in rel and o not in seen[t] and o not in pool[t]]
            for o, k in [(o, "decoy_retrieved") for o in rng.sample(hard, min(args.decoys_retrieved, len(hard)))] + \
                        [(o, "decoy_random") for o in easy[: args.decoys_random]]:
                kind[(t, o)] = k
                pool[t] = sorted(set(pool[t]) | {o})
    cards = pl.read_parquet(ROOT / "data" / "cards.parquet", columns=["oracle_id", "doc_text"])
    doc = dict(zip(cards["oracle_id"].to_list(), cards["doc_text"].to_list(), strict=True))
    out_dir = HERE / "runs" / args.run
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in range(args.passes):
        for tid in ids:
            oids = list(pool[tid])
            random.Random(f"judge-{tid}-pass{p}").shuffle(oids)
            jobs += [(p, tid, k, oids[i : i + BATCH]) for k, i in enumerate(range(0, len(oids), BATCH))]
    print(f"{sum(map(len, pool.values()))} pooled cards over {len(ids)} tasks; {len(jobs)} judge calls ({args.passes} passes)", flush=True)
    cc = ClaudeCode(model=args.model, timeout=600, retries=2, backoff=30)

    def one(job):
        p, tid, k, oids = job
        tpath = out_dir / "transcripts" / f"{tid}__p{p}__b{k}.jsonl"
        user = judge_prompt(tasks[tid], [doc[o] for o in oids])
        with open(tpath, "w") as f:
            f.write(json.dumps({"event": "config", "task": tid, "pass": p, "batch": k, "oracle_ids": oids,
                                "system": JUDGE_SYSTEM, "user": user, "llm": cc.config()}, ensure_ascii=False) + "\n")
        t0, grades, res = time.time(), {}, {}
        for _ in range(2):  # re-ask once on malformed JSON
            res = cc._retrying(JUDGE_SYSTEM, user, tpath, None)
            grades = parse_grades(res.get("result") or "", len(oids))
            if len(grades) == len(oids):
                break
        return {"pass": p, "task_id": tid, "batch": k, "oids": oids, "grades": grades, "usage": usage_of(res),
                "turns": res.get("num_turns"), "wall_s": round(time.time() - t0, 1)}

    rows, calls = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        for n, r in enumerate(ex.map(one, jobs), 1):
            calls.append({k: r[k] for k in ("pass", "task_id", "batch", "usage", "turns", "wall_s")} | {"n_cards": len(r["oids"]), "n_graded": len(r["grades"])})
            for i, o in enumerate(r["oids"], 1):
                if i in r["grades"]:
                    rows.append({"task_id": r["task_id"], "oracle_id": o, "pass": r["pass"], "grade": r["grades"][i],
                                 "labelled": o in set(tasks[r["task_id"]]["relevant"]), "kind": kind[(r["task_id"], o)],
                                 "judge": args.model})
            print(f"[{n}/{len(jobs)}] {r['task_id']} p{r['pass']} b{r['batch']}: {len(r['grades'])}/{len(r['oids'])} graded, "
                  f"{sum(r['usage'][k] for k in ('input', 'cache_write', 'cache_read', 'output')):,} tokens, {r['wall_s']}s", flush=True)
    (out_dir / "judgments.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (out_dir / "calls.jsonl").write_text("".join(json.dumps(x) + "\n" for x in calls))
    (out_dir / "config.json").write_text(json.dumps({"model": args.model, "cli": cc.version, "pool_runs": args.pool_runs, "tasks": ids,
                                                     "passes": args.passes, "batch": BATCH, "system": JUDGE_SYSTEM}, indent=1))
    print(f"wrote {len(rows)} judgments to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
