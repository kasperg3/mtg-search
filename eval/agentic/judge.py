"""EXP-006 TREC-style pooled judging with GLM-5.3.

    uv run python eval/agentic/judge.py --runs main [--runs pilot] [--dry-run]

Pool, per task: the union of every episode's final top-20 cards (all conditions, all repeats)
minus cards already labelled relevant. Known-item tasks are not judged, because their only
correct answer is the target. Each pooled card is graded 0/1/2 by a blind judge: it sees only
the task text (and any constraints) and the card text, in a seeded random order, batched up to
BATCH cards per call, and never sees the condition. Judgments are cached in
eval/agentic/judgments.jsonl and reused, so only new pool members are judged.
Transcripts go to eval/agentic/judge_transcript.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from agent import constraints_text  # noqa: E402
from glm import GLM, cost_usd  # noqa: E402

BATCH = 15
JUDGMENTS = HERE / "judgments.jsonl"

RUBRIC = """Grade each card for how well it satisfies the player's request:
2 = clearly satisfies it: the card itself performs the requested function directly and meets every stated constraint (colour, card type, mana value, format, cost).
1 = partially satisfies it: a weaker, narrower, conditional or indirect version of the requested function, or a card a reasonable player would still consider for this request; it must still meet the stated constraints.
0 = does not satisfy the request, or violates a stated constraint (wrong colour, card type or mana value, etc.).
Judge only from the card text shown. Do not reward popularity or general power level."""

JUDGE_SYSTEM = f"""You are an expert Magic: The Gathering judge grading search results for relevance to a player's request.

{RUBRIC}

Reply with JSON only: {{"grades": [{{"i": <card number>, "g": <0|1|2>}}, ...]}} with one entry per card."""


def judge_prompt(task: dict, docs: list[str]) -> str:
    req = task["prompt"]
    if task.get("filters"):
        req += f" (constraints: {constraints_text(task['filters'])})"
    cards = "\n\n".join(f"[{i}]\n{d}" for i, d in enumerate(docs, 1))
    return f"Request: {req}\n\nCards:\n\n{cards}"


def load_judgments() -> dict[tuple[str, str], int]:
    out = {}
    if JUDGMENTS.exists():
        for line in JUDGMENTS.read_text().splitlines():
            r = json.loads(line)
            out[(r["task_id"], r["oracle_id"])] = r["grade"]
    return out


def pool(run_dirs: list[Path], tasks: dict) -> dict[str, set[str]]:
    p: dict[str, set[str]] = {}
    for d in run_dirs:
        for line in (d / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            t = tasks.get(e["task_id"])
            if not t or t["kind"] == "known":
                continue
            rel = set(t["relevant"])
            for o in [o for o in e["final_oids"] if o][:20]:
                if o not in rel:
                    p.setdefault(e["task_id"], set()).add(o)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    tasks = {t["id"]: t for t in yaml.safe_load((HERE / "tasks.yaml").read_text())}
    cards = pl.read_parquet(ROOT / "data" / "cards.parquet", columns=["oracle_id", "doc_text"])
    doc = dict(zip(cards["oracle_id"].to_list(), cards["doc_text"].to_list(), strict=True))
    have = load_judgments()
    todo = {tid: sorted(o for o in oids if (tid, o) not in have) for tid, oids in pool([HERE / "runs" / r for r in args.runs], tasks).items()}
    todo = {k: v for k, v in todo.items() if v}
    batches = []
    for tid, oids in sorted(todo.items()):
        random.Random(f"judge-{tid}").shuffle(oids)
        batches += [(tid, oids[i : i + BATCH]) for i in range(0, len(oids), BATCH)]
    print(f"{sum(map(len, todo.values()))} cards to judge over {len(todo)} tasks, {len(batches)} calls")
    if args.dry_run or not batches:
        return
    llm = GLM(temperature=0.0, effort="low")
    tlog = open(HERE / "judge_transcript.jsonl", "a")
    out = open(JUDGMENTS, "a")
    total = [0.0]

    def one(b):
        tid, oids = b
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_prompt(tasks[tid], [doc[o] for o in oids])}]
        grades = {}
        for _attempt in range(3):
            resp = llm.chat(msgs)
            total[0] += cost_usd(resp.get("usage") or {})
            text = resp["choices"][0]["message"].get("content") or ""
            tlog.write(json.dumps({"task_id": tid, "oracle_ids": oids, "messages": msgs, "raw": resp}, ensure_ascii=False) + "\n")
            tlog.flush()
            try:
                m = re.search(r"\{.*\}", text, re.S)
                for g in json.loads(m.group(0))["grades"]:
                    grades[int(g["i"])] = int(g["g"])
                if all(i in grades for i in range(1, len(oids) + 1)):
                    break
            except Exception:  # noqa: BLE001 - retry on malformed JSON
                continue
        return tid, oids, grades

    with ThreadPoolExecutor(args.workers) as ex:
        for n, (tid, oids, grades) in enumerate(ex.map(one, batches), 1):
            for i, o in enumerate(oids, 1):
                if i in grades and grades[i] in (0, 1, 2):
                    out.write(json.dumps({"task_id": tid, "oracle_id": o, "grade": grades[i], "judge": "glm-5.3"}) + "\n")
            out.flush()
            if n % 20 == 0:
                print(f"{n}/{len(batches)} ${total[0]:.3f}", flush=True)
    print(f"judging cost ${total[0]:.3f}")


if __name__ == "__main__":
    main()
