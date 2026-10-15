"""Rescore saved precon runs with the current scorer, from their stored outputs (no model calls).

    uv run python -m bench.precon.rescore                  # dry run: what would change, per run
    uv run python -m bench.precon.rescore --write          # rewrite run records + episodes.jsonl

Written for review m3 (2026-10-15): land picks no longer cost headline slots (scorer.py, "Headline
vs land"). Safety checks, per run:
  * the run's dataset revision must equal the loaded benchmark data;
  * every episode whose picks contain no land pick must reproduce its stored metrics exactly,
    which proves the card table, labels and validity rules are unchanged since the run.
With ``--write`` each run record gets ``metrics``/``slices`` recomputed and a ``rescored`` block
(date, git, reason, the previous headline means); each episode's ``metrics`` is replaced and the
previous values are kept under ``metrics_before_rescore``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from bench.precon import scorer  # noqa: E402
from bench.precon.loader import CardDB, load_bench  # noqa: E402
from bench.precon.run import RUNS, git_info, slice_fns  # noqa: E402

REASON = "review m3: land picks no longer cost headline slots (headline @k uses min(k, slots - land picks))"


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def rescore_run(rid: str, bench, db, write: bool, n_boot: int = 10_000, seed: int = 0) -> dict:
    rec_path, ep_path = RUNS / f"{rid}.json", RUNS / rid / "episodes.jsonl"
    rec = json.loads(rec_path.read_text())
    if rec["dataset"]["revision"] != bench.revision:
        raise SystemExit(f"{rid}: dataset {rec['dataset']['revision']} != loaded {bench.revision}")
    sc = rec["score_config"]
    cfg = scorer.ScoreConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in sc.items()})
    eps = [json.loads(x) for x in ep_path.read_text().splitlines()]
    n_changed = 0
    for e in eps:
        old = e.get("metrics_before_rescore") or e["metrics"]
        new = scorer.score_episode(e["task"], e["output"], bench.decks[e["precon_id"]], bench.labels.get(e["precon_id"], []), db, cfg)
        land_key = {"cut": "cut_land.n_picks", "add": "add_land.n_picks", "swap": None}[e["task"]]
        n_land = (new.get(land_key) or 0) if land_key else (new.get("swap.cut_land.n_picks") or 0) + (new.get("swap.add_land.n_picks") or 0)
        diff = sorted(k for k in set(old) | set(new) if not _same(old.get(k), new.get(k)))
        if diff and n_land == 0:
            raise SystemExit(f"{rid}: episode {e['precon_id']}/{e['task']}/r{e['rep']} has no land pick but its "
                             f"metrics changed ({diff[:5]}): data or rules differ from the run, not rescoring")
        n_changed += bool(diff)
        e["metrics_before_rescore"], e["metrics"] = old, new
    metrics, slices = {}, {}
    fns = slice_fns(bench)
    for t in sorted({e["task"] for e in eps}):
        te = [e for e in eps if e["task"] == t]
        metrics[t] = scorer.aggregate(te, n_boot=n_boot, seed=seed)
        head = [m for m in scorer.HEADLINE.get(t, []) if m in metrics[t]]
        slices[t] = {name: scorer.slice_aggregate(te, fn, head, n_boot=1000) for name, fn in fns.items()}
    prev = rec.get("rescored", {}).get("previous_metrics") or rec["metrics"]
    head_change = {}
    for t, ms in metrics.items():
        for m in scorer.HEADLINE.get(t, []):
            a, b = (prev.get(t, {}).get(m) or {}).get("mean"), (ms.get(m) or {}).get("mean")
            if not _same(a, b):
                head_change[f"{m}"] = [a, b]
    if write:
        ep_path.write_text("".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in eps))
        rec["rescored"] = {"date": datetime.now(UTC).isoformat(timespec="seconds"), "git": git_info(), "reason": REASON,
                           "episodes_changed": n_changed, "previous_metrics": prev}
        rec["metrics"], rec["slices"] = metrics, slices
        rec_path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return {"run": rid, "episodes": len(eps), "episodes_changed": n_changed, "headline_changes": head_change}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run ids (default: every run record with saved episodes)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=None, help="write the change summary (JSON)")
    args = ap.parse_args(argv)
    rids = args.runs or sorted(p.stem for p in RUNS.glob("*.json") if (RUNS / p.stem / "episodes.jsonl").exists())
    db, benches, out = CardDB.from_parquet(), {}, []
    for r in rids:
        # default: the dataset each run recorded (e.g. hf:<repo>@<sha>), loaded once per distinct spec
        spec = args.data or json.loads((RUNS / f"{r}.json").read_text())["config"].get("data")
        if spec not in benches:
            benches[spec] = load_bench(spec)
        out.append(rescore_run(r, benches[spec], db, args.write))
    for o in out:
        ch = ", ".join(f"{m} {a:.4f}->{b:.4f}" for m, (a, b) in o["headline_changes"].items() if a is not None and b is not None)
        print(f"{o['run']}: {o['episodes_changed']}/{o['episodes']} episodes changed; {ch or 'no headline change'}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
