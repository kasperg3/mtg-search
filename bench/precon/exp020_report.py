"""EXP-020 report: the 2 x 2 of eval/experiments/deck_tool_plan.md (pasted/named deck x without/with get_deck).

    uv run python bench/precon/exp020_report.py [--prefix 20261016_exp020]

Reads runs/<prefix>_{PA,PB,NA,NB}/episodes.jsonl, writes runs/<prefix>_summary.json (aggregates only) and prints
the tables the plan's decision rules use.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench.precon import scorer  # noqa: E402
from bench.precon.loader import load_bench  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
CELLS = ("PA", "PB", "NA", "NB")
VALIDITY = {"cut": ["cut.validity"], "add": ["add.validity"], "swap": ["swap.cut.validity", "swap.add.validity"]}
QUALITY = {"cut": ["cut.ndcg@10"], "add": ["add.ndcg@10"], "swap": ["swap.cut.ndcg@10", "swap.add.ndcg@10"]}
REASONS = {"cut": ["cut.invalid.not_in_deck", "cut.invalid.unresolved", "cut.invalid.is_commander"],
           "add": ["add.invalid.in_deck", "add.invalid.off_identity", "add.invalid.not_legal", "add.invalid.unresolved",
                   "add.invalid.game_changer_limit"],
           "swap": ["swap.cut.invalid.not_in_deck", "swap.add.invalid.in_deck", "swap.add.invalid.off_identity"]}
COMPARE = (("NB", "NA", "help"), ("NB", "PA", "recovery"), ("PB", "PA", "guard"))


def load(prefix: str) -> dict[str, list[dict]]:
    return {c: [json.loads(x) for x in (RUNS / f"{prefix}_{c}" / "episodes.jsonl").read_text().splitlines()]
            for c in CELLS}


def tools_used(e: dict) -> set[str]:
    return {t["tool"] for t in (e.get("run") or {}).get("tool_calls") or []}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def cell_summary(eps: list[dict], dev: set[str], new: set[str]) -> dict:
    out = {}
    for task in ("cut", "add", "swap"):
        te = [e for e in eps if e["task"] == task]
        row = {"n": len(te), "errors": sum(1 for e in te if (e.get("run") or {}).get("error"))}
        for m in VALIDITY[task]:
            pp = scorer.per_precon(te, m)
            row[m] = {"all": mean(pp.values()), "dev": mean(v for k, v in pp.items() if k in dev),
                      "new": mean(v for k, v in pp.items() if k in new)}
        for m in QUALITY[task]:
            pp = scorer.per_precon([e for e in te if e["precon_id"] in dev], m)
            row[m + " (dev)"] = mean(pp.values())
        for m in REASONS[task]:
            row[m] = sum(int(e["metrics"].get(m) or 0) for e in te)
        used = [tools_used(e) for e in te]
        for t in ("get_deck", "search_cards", "get_card", "find_similar"):
            row[f"calls {t}"] = sum(t in u for u in used)
        row["get_deck without search_cards"] = sum("get_deck" in u and "search_cards" not in u for u in used)
        runs = [e["run"] for e in te if e.get("run")]
        row["tokens/episode"] = mean(r["usage"]["prompt"] + r["usage"]["completion"] for r in runs)
        row["turns/episode"] = mean(r.get("llm_calls") for r in runs)
        row["tool calls/episode"] = mean(len(r.get("tool_calls") or []) for r in runs)
        row["wall s/episode"] = mean(r.get("wall_s") for r in runs)
        out[task] = row
    return out


def compare(a: list[dict], b: list[dict], ids: set[str]) -> dict:
    out = {}
    for task in ("cut", "add", "swap"):
        for m in VALIDITY[task] + QUALITY[task]:
            pa = scorer.per_precon([e for e in a if e["task"] == task and e["precon_id"] in ids], m)
            pb = scorer.per_precon([e for e in b if e["task"] == task and e["precon_id"] in ids], m)
            r = scorer.paired_bootstrap(pa, pb)
            if r.get("n"):
                out[m] = {k: (v if k == "sign_test_p" else round(v, 4) if isinstance(v, float)
                              else [round(x, 4) for x in v] if isinstance(v, list) else v) for k, v in r.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="20261016_exp020")
    a = ap.parse_args()
    bench = load_bench(None)
    dev = set(bench.precon_ids("dev"))
    cells = load(a.prefix)
    ids = {e["precon_id"] for e in cells["PA"]}
    new = ids - dev
    summary = {"cells": {c: cell_summary(cells[c], dev, new) for c in CELLS}, "comparisons": {}}
    for x, y, name in COMPARE:
        summary["comparisons"][f"{name}: {x} - {y}"] = {
            "all": compare(cells[x], cells[y], ids), "dev": compare(cells[x], cells[y], dev),
            "new": compare(cells[x], cells[y], new)}
    pasted = {c: [tools_used(e) for e in cells[c]] for c in ("PA", "PB")}
    summary["guard"] = {"search episodes PA": sum("search_cards" in u for u in pasted["PA"]),
                        "search episodes PB": sum("search_cards" in u for u in pasted["PB"]),
                        "PB get_deck without search": sum("get_deck" in u and "search_cards" not in u for u in pasted["PB"]),
                        "PB get_deck episodes": sum("get_deck" in u for u in pasted["PB"])}
    nb = [tools_used(e) for e in cells["NB"]]
    summary["NB get_deck share"] = round(sum("get_deck" in u for u in nb) / len(nb), 4)
    summary["n_precons"] = {"all": len(ids), "dev": len(ids & dev), "new": len(new)}
    out = RUNS / f"{a.prefix}_summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    for c in CELLS:
        print(f"== {c}")
        for task, row in summary["cells"][c].items():
            print(f"  {task}: " + "; ".join(f"{k}={v}" for k, v in row.items()))
    for name, comp in summary["comparisons"].items():
        print(f"== {name}")
        for sl in ("all", "dev", "new"):
            for m, r in comp[sl].items():
                print(f"  [{sl}] {m}: {r['mean_diff']:+.3f} {r['ci95']} p={r['sign_test_p']:.3g} n={r['n']}")
    print("guard", summary["guard"], "NB get_deck share", summary["NB get_deck share"])
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
