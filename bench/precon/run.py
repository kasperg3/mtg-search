"""EXP-007 runner: produce answers with a system, score them, write a tracked run record.

    # non-LLM baselines (no API calls)
    uv run python -m bench.precon.run --system random --split all --repeats 3
    uv run python -m bench.precon.run --system heuristic --split all
    uv run python -m bench.precon.run --system popularity --split all
    # harness plumbing dry run with the scripted fake LLM (no API calls)
    uv run python -m bench.precon.run --system agent --llm fake --tools lexical --split test --limit 3
    # real GLM-5.3 runs require explicit approval flags (not run so far):
    uv run python -m bench.precon.run --system agent --llm glm --allow-llm --tools mcp --max-usd 5 ...
    uv run python -m bench.precon.run --system probe --llm glm --allow-llm --split all --max-usd 2
    # paired comparison / gain-function sensitivity between finished runs
    uv run python -m bench.precon.run compare RUN_A RUN_B [RUN_C ...]

Systems: random | heuristic | popularity (reference) | agent (tools mcp|lexical|none,
optional --anonymise-commander) | probe (memorisation probe, product name only).

Outputs:
  bench/precon/runs/<run_id>.json      TRACKED: git sha, dataset revision, config, aggregate
                                        metrics (macro mean, 95% bootstrap CI, repeat sd), slices,
                                        cost. Aggregate scores only (labels stay private).
  bench/precon/runs/<run_id>/           gitignored: episodes.jsonl (per-precon outputs + metrics),
                                        transcripts/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench.precon import scorer  # noqa: E402
from bench.precon.loader import CardDB, load_bench  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
TASKS = ("cut", "add", "swap")


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {"sha": g("rev-parse", "HEAD"), "dirty_bench": bool(g("status", "--porcelain", "--", "bench/precon", "src"))}


def slice_fns(bench) -> dict:
    def meta(pid):
        return bench.precons.get(pid, {})

    def era(pid):
        d = str(meta(pid).get("release_date") or bench.splits.get(pid, {}).get("release_date") or "")
        return d[:4] or "unknown"

    def ndecks(pid):
        n = meta(pid).get("potential_decks")
        if n is None:
            return "unknown"
        return "<1000" if n < 1000 else "1000-2999" if n < 3000 else ">=3000"

    return {
        "split": lambda pid: bench.splits.get(pid, {}).get("split", "unknown"),
        "n_colors": lambda pid: str(len(bench.decks[pid].color_identity)),
        "era": era,
        "universes_beyond": lambda pid: str(meta(pid).get("is_universes_beyond", "unknown")),
        "n_decks": ndecks,
    }


# =========================================================================== systems

def build_system(args, bench, db):
    if args.system == "random":
        from bench.precon.baselines.simple import RandomLegal

        return RandomLegal(db, seed=args.seed, gc_limit=args.gc_limit)
    if args.system == "popularity":
        from bench.precon.baselines.simple import PopularityOracle

        return PopularityOracle(db, gc_limit=args.gc_limit)
    if args.system == "heuristic":
        import polars as pl

        from bench.precon.baselines.simple import TextHeuristic

        cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
        if args.backend == "late":
            from mtg_search.retrievers.late import LateInMemory

            r = LateInMemory.from_dir(ROOT / "index" / "embeddings", "lightonai/mLateOn")
        else:
            from mtg_search.retrievers.bm25 import BM25Retriever

            r = BM25Retriever(cards["doc_text"].to_list())
        return TextHeuristic(db, cards, r, gc_limit=args.gc_limit)
    return None


def make_llm(args):
    if args.llm == "fake":
        from bench.precon.harness.llm import FakeLLM

        return FakeLLM(completion_per_call=args.fake_completion)
    if not args.allow_llm:
        sys.exit("refusing to call a real LLM without --allow-llm (LLM runs need the owner's approval)")
    if args.llm == "claude-code":
        from bench.precon.harness.claude_code import ClaudeCode

        return ClaudeCode(model=args.cc_model, tool_search=args.cc_tool_search)
    from bench.precon.harness.llm import GLM

    return GLM(temperature=args.temperature, effort=args.effort)


class AgentSystem:
    """Wraps the harness: renders the deck, runs pre-flight checks, runs one episode."""

    def __init__(self, args, bench, db, run_dir: Path):
        import polars as pl

        from bench.precon.harness.leakage import identifying_strings
        from bench.precon.harness.llm import Budget
        from bench.precon.harness.tools import make_toolset

        self.args, self.bench, self.db = args, bench, db
        self.llm = make_llm(args)
        self.budget = Budget(args.max_usd)
        kw = {"surface": args.surface} if args.tools == "shipped" else {}
        self.toolset = make_toolset(args.tools, bench.denylist, backend=args.backend, **kw) if args.system == "agent" else None
        self.service = None
        if args.llm == "claude-code" and self.toolset is not None and self.toolset.schemas():
            from bench.precon.harness.claude_code import ToolService

            self.service = ToolService(self.toolset)
        cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
        need = {c["oracle_id"] for d in bench.decks.values() for c in d.cards}
        self.rows = {r["oracle_id"]: r for r in cards.filter(pl.col("oracle_id").is_in(list(need))).iter_rows(named=True)}
        self.forbidden = bench.gold_names(["dev", "test"])
        self.identifying = identifying_strings(bench.precons)
        self.tdir = run_dir / "transcripts"
        self.tdir.mkdir(parents=True, exist_ok=True)

    def config(self) -> dict:
        return {"llm": self.llm.config(), "toolset": self.toolset.config() if self.toolset else None,
                "max_usd": self.args.max_usd, "anonymise_commander": self.args.anonymise_commander,
                "tool_hint": self.args.tool_hint}

    def run(self, task: str, deck, rep: int) -> dict:
        from bench.precon.harness.agent import run_episode
        from bench.precon.harness.leakage import preflight
        from bench.precon.harness.prompts import prompt_hash, render_deck, system_prompt, user_prompt

        tpath = self.tdir / f"{deck.precon_id}__{task}__r{rep}.jsonl"
        if self.args.system == "probe":
            from bench.precon.harness.probe import run_probe

            meta = self.bench.precons.get(deck.precon_id, {})
            if self.args.llm == "claude-code":
                return self.llm.probe(meta.get("deck_name", ""), meta.get("product", ""), transcript_path=tpath)
            return run_probe(self.llm, meta.get("deck_name", ""), meta.get("product", ""), budget=self.budget,
                             transcript_path=tpath)
        ci = "".join(c for c in "WUBRG" if c in deck.color_identity)
        system = system_prompt(task, self.args.tools, ci, gc_limit=self.args.gc_limit, tool_hint=self.args.tool_hint)
        user = user_prompt(task, render_deck(deck, self.rows, self.args.anonymise_commander))
        deck_names = [self.rows[c["oracle_id"]]["name"] for c in deck.cards if c["oracle_id"] in self.rows]
        tool_names = [t["function"]["name"] for t in self.toolset.schemas()]
        preflight(system, user, deck_names=deck_names, forbidden_gold=self.forbidden, identifying=self.identifying,
                  tool_names=tool_names)
        if self.args.llm == "claude-code":
            d = self.llm.episode(task, system, user, tool_url=self.service.url if self.service else None,
                                 transcript_path=tpath, server_name="mtg-search" if self.args.tools == "shipped" else "deck")
        else:
            d = run_episode(self.llm, self.toolset, task, system, user, budget=self.budget, transcript_path=tpath).to_dict()
        d["prompt_hash"] = prompt_hash(system, user)
        d["prompt_chars"] = len(system) + len(user)
        return d


# =========================================================================== main loop

def run(args) -> Path:
    t0 = time.time()
    bench = load_bench(args.data)
    db = CardDB.from_parquet()
    pids = bench.precon_ids(args.split, scored_only=not args.include_unscored)
    if args.limit:
        pids = pids[: args.limit]
    tasks = ["probe"] if args.system == "probe" else [t for t in args.tasks.split(",") if t]
    run_id = args.run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{args.system}" + (
        f"_{args.tools}" if args.system == "agent" else "") + (f"_{args.llm}" if args.system in ("agent", "probe") else "")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    th = {}
    if args.cut_thresholds:
        th["cut_thresholds"] = tuple(float(x) for x in args.cut_thresholds.split(","))
    if args.add_thresholds:
        th["add_thresholds"] = tuple(float(x) for x in args.add_thresholds.split(","))
    cfg = scorer.ScoreConfig(gc_limit=args.gc_limit, **th)
    llm_system = args.system in ("agent", "probe")
    system = AgentSystem(args, bench, db, run_dir) if llm_system else build_system(args, bench, db)
    jobs = [(pid, t, r) for r in range(args.repeats) for pid in pids for t in tasks]
    print(f"{run_id}: {len(pids)} precons x {len(tasks)} tasks x {args.repeats} repeats = {len(jobs)} episodes", flush=True)
    episodes: list[dict] = []
    lock = threading.Lock()
    ep_path = run_dir / "episodes.jsonl"
    ep_f = open(ep_path, "w")

    def one(pid, task, rep):
        deck, labels = bench.decks[pid], bench.labels.get(pid, [])
        if llm_system:
            rec = system.run(task, deck, rep)
            out = rec.pop("output")
        else:
            rec, out = {}, system.propose(task, deck, rep)
        recs = []
        for t in (("cut", "add") if task == "probe" else (task,)):
            m = scorer.score_episode(t, out, deck, labels, db, cfg)
            recs.append({"precon_id": pid, "task": t, "rep": rep, "system": args.system, "output": out,
                         "metrics": m, **({"run": rec} if rec else {})})
        return recs

    stopped, n_skipped = None, 0
    with ThreadPoolExecutor(max(1, args.workers if llm_system else 1)) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            if f.cancelled():
                n_skipped += 1
                continue
            try:
                recs = f.result()
            except Exception as e:
                from bench.precon.harness.llm import BudgetExceeded

                if isinstance(e, BudgetExceeded):
                    stopped = str(e)
                    for g in futs:
                        g.cancel()
                    continue
                print(f"FAILED {futs[f]}: {traceback.format_exc()}", flush=True)
                continue
            with lock:
                for r in recs:
                    episodes.append(r)
                    ep_f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            if i % 50 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    ep_f.close()
    if stopped:
        stopped += f"; {n_skipped} episodes not run"
        print(f"STOPPED: {stopped}", flush=True)

    record = summarise(args, bench, episodes, run_id, cfg, system, stopped, time.time() - t0)
    out = RUNS / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print_summary(record)
    print(f"wrote {out.relative_to(ROOT)} (+ gitignored {run_dir.relative_to(ROOT)}/)")
    return out


def cost_summary(episodes: list[dict]) -> dict | None:
    # one LLM run per (precon, rep, task); the probe's single run is scored as two episodes
    uniq = list({(e["precon_id"], e["rep"], e["run"].get("task", "probe")): e["run"]
                 for e in episodes if "run" in e}.values())
    if not uniq:
        return None
    n = len(uniq)
    tot = {k: sum(r["usage"][k] for r in uniq) for k in ("prompt", "cached", "completion", "reasoning")}
    return {"n_episodes": n, "usd_total": round(sum(r["cost_usd"] for r in uniq), 4),
            "usd_per_episode": round(sum(r["cost_usd"] for r in uniq) / n, 5),
            "llm_calls_per_episode": sum(r["llm_calls"] for r in uniq) / n,
            "tool_calls_per_episode": sum(len(r.get("tool_calls") or []) for r in uniq) / n,
            "tokens_per_episode": {k: v / n for k, v in tot.items()},
            "wall_s_per_episode": sum(r.get("wall_s", 0) for r in uniq) / n,
            "stops": {s: sum(1 for r in uniq if r.get("stop") == s) for s in {r.get("stop") for r in uniq}},
            "parse": {s: sum(1 for r in uniq if r.get("parse") == s) for s in {r.get("parse") for r in uniq}}}


def summarise(args, bench, episodes, run_id, cfg, system, stopped, wall) -> dict:
    metrics, slices = {}, {}
    fns = slice_fns(bench)
    for t in sorted({e["task"] for e in episodes}):
        eps = [e for e in episodes if e["task"] == t]
        metrics[t] = scorer.aggregate(eps, n_boot=args.n_boot, seed=args.seed)
        head = [m for m in scorer.HEADLINE.get(t, []) if m in metrics[t]]
        slices[t] = {name: scorer.slice_aggregate(eps, fn, head, n_boot=1000) for name, fn in fns.items()}
    conf = {k: v for k, v in vars(args).items() if k not in ("func",)}
    rec = {
        "experiment": "EXP-007", "run_id": run_id, "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(), "dataset": {"revision": bench.revision, "snapshot_date": bench.manifest.get("snapshot_date"),
                                        "cards_manifest": bench.manifest.get("cards_parquet") or bench.manifest.get("cards_manifest"),
                                        "cutoff": next((s.get("cutoff") for s in bench.splits.values()), None)},
        "config": conf, "score_config": cfg.to_dict(),
        "system_config": system.config() if hasattr(system, "config") else {"name": getattr(system, "name", args.system)},
        "n_precons": len({e["precon_id"] for e in episodes}),
        "n_precons_by_split": _count_by_split(bench, episodes),
        "n_episodes": len(episodes), "stopped": stopped, "wall_s": round(wall, 1),
        "cost": cost_summary(episodes), "metrics": metrics, "slices": slices,
    }
    return rec


def _count_by_split(bench, episodes) -> dict:
    out: dict[str, int] = {}
    for pid in {e["precon_id"] for e in episodes}:
        s = bench.splits.get(pid, {}).get("split", "unknown")
        out[s] = out.get(s, 0) + 1
    return out


def print_summary(rec: dict) -> None:
    for t, ms in rec["metrics"].items():
        print(f"== {t} ({rec['n_precons']} precons)")
        for m in scorer.HEADLINE.get(t, sorted(ms)):
            if m in ms:
                a = ms[m]
                sd = f" sd={a['repeat_sd']:.3f}" if a.get("repeat_sd") is not None else ""
                print(f"  {m:28s} {a['mean']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] n={a['n_precons']}{sd}")
    if rec.get("cost"):
        c = rec["cost"]
        print(f"  cost: ${c['usd_total']} total, ${c['usd_per_episode']}/episode, tokens/episode {c['tokens_per_episode']}")


# =========================================================================== compare

def compare(run_ids: list[str], n_boot: int = 10_000) -> dict:
    """Paired bootstrap + sign test between the first run and each other run on common precons,
    and Kendall tau between system rankings under thresholded vs continuous gains."""
    eps = {}
    for rid in run_ids:
        p = RUNS / rid / "episodes.jsonl"
        eps[rid] = [json.loads(line) for line in p.read_text().splitlines()]
    out = {"paired": {}, "kendall_tau_raw_vs_cont": {}}
    tasks = sorted({e["task"] for es in eps.values() for e in es})
    for t in tasks:
        for m in scorer.HEADLINE.get(t, []):
            base = scorer.per_precon([e for e in eps[run_ids[0]] if e["task"] == t], m)
            for rid in run_ids[1:]:
                other = scorer.per_precon([e for e in eps[rid] if e["task"] == t], m)
                if base and other:
                    out["paired"].setdefault(t, {}).setdefault(m, {})[f"{run_ids[0]} - {rid}"] = scorer.paired_bootstrap(base, other, n_boot)
        for m in scorer.HEADLINE.get(t, []):
            if m.endswith(".cont") and m.replace(".cont", "") in scorer.HEADLINE[t]:
                raw = [scorer.aggregate([e for e in eps[r] if e["task"] == t], [m.replace(".cont", "")], 10).get(m.replace(".cont", ""), {}).get("mean") for r in run_ids]
                cont = [scorer.aggregate([e for e in eps[r] if e["task"] == t], [m], 10).get(m, {}).get("mean") for r in run_ids]
                if None not in raw and None not in cont:
                    out["kendall_tau_raw_vs_cont"][m.replace(".cont", "")] = scorer.kendall_tau(raw, cont)
    print(json.dumps(out, indent=1))
    return out


# =========================================================================== cost projection

def estimate(dry_runs: dict[str, str], n_precons: int, repeats: int = 3, chars_scale: float = 1.15,
             cached_share: float = 0.22, completion_per_call: int = 1650, probe: bool = True) -> dict:
    """Project GLM-5.3 cost of a full run from fake-LLM dry-run transcripts.

    Per condition: P0 = mean prompt tokens of the first call, T = mean growth per tool call
    (both measured by the fake at ~4 chars/token, scaled by ``chars_scale`` for card text);
    an episode with c tool calls makes c+1 calls with prompt (c+1)*P0 + T*c(c+1)/2 tokens and
    ``completion_per_call`` output tokens per call (EXP-006 GLM-5.3 effort=low mean: 1650, ~93%
    reasoning). ``cached_share`` of input is billed at the cached rate (EXP-006: 22%).
    Scenarios: typical (cut 3, add 6, swap 8 tool calls) and worst case (full budgets)."""
    import glob
    import statistics as st

    from bench.precon.harness.llm import PRICE
    from bench.precon.harness.prompts import BUDGETS

    p_in, p_cached, p_out = PRICE["glm-5.3"]

    def usd(prompt, completion):
        return (prompt * (1 - cached_share) * p_in + prompt * cached_share * p_cached + completion * p_out) / 1e6

    typical = {"cut": 3, "add": 6, "swap": 8}
    out = {"assumptions": {"n_precons": n_precons, "repeats": repeats, "chars_scale": chars_scale,
                           "cached_share": cached_share, "completion_per_call": completion_per_call,
                           "price_per_M": {"input": p_in, "cached": p_cached, "output": p_out}}, "conditions": {}}
    total = {"typical": 0.0, "worst": 0.0}
    for cond, rid in dry_runs.items():
        p0, inc = [], []
        for f in glob.glob(str(RUNS / rid / "transcripts" / "*.jsonl")):
            pts = [json.loads(line)["raw"]["usage"]["prompt_tokens"] for line in open(f)
                   if json.loads(line)["event"] == "response"]
            p0.append(pts[0])
            inc += [b - a for a, b in zip(pts, pts[1:], strict=False)]
        P0, T = st.mean(p0) * chars_scale, (st.mean(inc) * chars_scale if inc else 0.0)
        row = {"P0_tokens": round(P0), "T_tokens_per_tool_call": round(T)}
        for scen, calls in (("typical", typical), ("worst", {t: b[0] for t, b in BUDGETS.items()})):
            per = {}
            for task in ("cut", "add", "swap"):
                c = 0 if cond == "none" else calls[task]
                L = c + 1
                prompt = L * P0 + T * c * (c + 1) / 2
                per[task] = usd(prompt, L * completion_per_call)
            ep = sum(per.values())
            row[scen] = {"usd_per_episode": {k: round(v, 4) for k, v in per.items()},
                         "usd_total": round(ep * n_precons * repeats, 2)}
            total[scen] += ep * n_precons * repeats
        out["conditions"][cond] = row
    if probe:  # 1 call, ~150 prompt tokens
        pr = usd(150, completion_per_call) * n_precons * repeats
        out["conditions"]["probe"] = {"usd_total": round(pr, 2)}
        total = {k: v + pr for k, v in total.items()}
    out["usd_total"] = {k: round(v, 2) for k, v in total.items()}
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        compare(argv[1:])
        return
    if argv and argv[0] == "estimate":
        ap = argparse.ArgumentParser(prog="run.py estimate")
        ap.add_argument("--dry", required=True, help="cond=run_id,... (fake-LLM dry runs), e.g. mcp=X,lexical=Y,none=Z")
        ap.add_argument("--precons", type=int, required=True)
        ap.add_argument("--repeats", type=int, default=3)
        ap.add_argument("--cached-share", type=float, default=0.22)
        a = ap.parse_args(argv[1:])
        estimate(dict(kv.split("=", 1) for kv in a.dry.split(",")), a.precons, a.repeats, cached_share=a.cached_share)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["random", "heuristic", "popularity", "agent", "probe"])
    ap.add_argument("--tasks", default="cut,add,swap")
    ap.add_argument("--split", default="test", help="train|dev|test|gap|all")
    ap.add_argument("--include-unscored", action="store_true", help="also precons failing the §4 gates")
    ap.add_argument("--data", default=None, help="local build dir or hf:<repo>[@rev] (default: newest local, else HF)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gc-limit", type=int, default=3)
    ap.add_argument("--cut-thresholds", default=None, help="grade 3,2,1 lower bounds on u (default .50,.30,.15)")
    ap.add_argument("--add-thresholds", default=None, help="grade 3,2,1 lower bounds on r (default .30,.15,.05)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--tools", default="mcp", choices=["mcp", "lexical", "none", "shipped"],
                    help="shipped = the real MCP server surface (instructions + tool descriptions), see --surface")
    ap.add_argument("--surface", default="v1", help="model-facing text variant for --tools shipped (surfaces.py)")
    ap.add_argument("--backend", default="late", choices=["late", "bm25", "docker"],
                    help="retrieval backend for --tools mcp and the heuristic baseline")
    ap.add_argument("--llm", default="fake", choices=["fake", "glm", "claude-code"],
                    help="claude-code = headless `claude -p` on the Claude Code subscription (no API key)")
    ap.add_argument("--cc-model", default="claude-sonnet-5-5", help="model for --llm claude-code")
    ap.add_argument("--cc-tool-search", action="store_true",
                    help="--llm claude-code: enable ToolSearch (deferred MCP tool loading, the Claude Code default)")
    ap.add_argument("--allow-llm", action="store_true", help="required for --llm glm|claude-code (owner approval)")
    ap.add_argument("--max-usd", type=float, default=5.0, help="hard run budget cap")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--fake-completion", type=int, default=1650, help="fake LLM output tokens per call (EXP-006 GLM mean)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--anonymise-commander", action="store_true")
    ap.add_argument("--tool-hint", action="store_true", help="tell the agent to use its tools before answering")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
