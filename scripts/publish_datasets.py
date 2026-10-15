"""Build (and optionally upload) the citable benchmark datasets on Hugging Face (all private).

    uv run python scripts/publish_datasets.py bench        [--upload] [--tag v1.0]   # kasper-cbx/mtg-search-bench
    uv run python scripts/publish_datasets.py precon-fit   [--upload] [--tag v1.0]   # deck_fit config of kasper-cbx/mtg-precon-bench

``bench``: the search benchmark (363 queries), the 113 agent search tasks, every relevance grade from the
pooled-grading runs (EXP-016, EXP-018, EXP-019) and the resulting relevance labels (qrels). ``precon-fit``: the
EXP-017 deck-fit grades (rubric v3, tuning and test decks) added to the precon benchmark as config ``deck_fit``.
Cards are referenced by Scryfall ``oracle_id``, with names from the card snapshot of
``kasper-cbx/mtg-search-index@scryfall-20261014090155``. Files are written to ``data/hf/<repo>/`` first; ``--upload``
commits them with the dataset card and tags the commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

INDEX_REV = "scryfall-20261014090155"
QUERY_SETS = ("tags", "handwritten", "paraphrase", "deckneeds")
JUDGE_RUNS = {"judge-full": "EXP-016", "judge-ablation": "EXP-018", "judge-promptonly": "EXP-019"}
FIT_RUNS = {"20261015_exp017-dev-r3": "dev", "20261015_exp017-test": "test"}
GRADED = pl.List(pl.Struct({"oracle_id": pl.String, "grade": pl.Int8}))


def names() -> dict[str, str]:
    c = pl.read_parquet(ROOT / "data" / "cards.parquet", columns=["oracle_id", "name"])
    return dict(zip(c["oracle_id"].to_list(), c["name"].to_list(), strict=True))


def graded(d: dict | None) -> list[dict]:
    return [{"oracle_id": o, "grade": int(g)} for o, g in sorted((d or {}).items())]


def build_bench(out: Path) -> dict:
    from eval.queries import load_queries

    nm = names()
    q = [{"set": s, "id": x["id"], "query": x["query"], "relevant": list(x["relevant"]), "graded": graded(x.get("graded")),
          "filters": json.dumps(x.get("filters") or {}, sort_keys=True), "origin": x.get("origin"), "source": x.get("source"),
          "labeling_rule": x.get("labeling_rule")} for s in QUERY_SETS for x in load_queries(s)]
    tasks = [t for f in ("tasks.yaml", "tasks_known_easy.yaml") for t in yaml.safe_load((ROOT / "eval" / "agentic" / f).read_text())]
    t = [{"id": x["id"], "kind": x["kind"], "source": x["source"], "prompt": x["prompt"], "relevant": list(x["relevant"]),
          "graded": graded(x.get("graded")), "filters": json.dumps(x.get("filters") or {}, sort_keys=True),
          "popular": x.get("popular")} for x in tasks]
    g = []
    for run, exp in JUDGE_RUNS.items():
        p = ROOT / "eval" / "agentic" / "runs" / run / "judgments.jsonl"
        for j in map(json.loads, p.read_text().splitlines()):
            g.append({"experiment": exp, "grading_run": run, "task_id": j["task_id"], "oracle_id": j["oracle_id"],
                      "name": nm.get(j["oracle_id"]), "grade": int(j["grade"]), "on_list": bool(j["labelled"]),
                      "kind": j.get("kind", "pool"), "pass": int(j["pass"]), "grader_model": j["judge"]})
    # relevance labels: the task's list (its grade) plus pool cards graded >= 1 (latest grading run wins)
    by_task = {x["id"]: x for x in tasks}
    latest: dict[tuple[str, str], int] = {}
    for r in g:
        if r["kind"] == "pool" and r["pass"] == 0:
            latest[(r["task_id"], r["oracle_id"])] = r["grade"]
    qrels = []
    for tid, x in by_task.items():
        gd = x.get("graded") or {}
        for o in x["relevant"]:
            qrels.append({"task_id": tid, "oracle_id": o, "name": nm.get(o), "relevance": int(gd.get(o, 1)), "source": "list"})
        if x["kind"] != "known":
            on = set(x["relevant"])
            qrels += [{"task_id": tid, "oracle_id": o, "name": nm.get(o), "relevance": gr, "source": "grader"}
                      for (tt, o), gr in sorted(latest.items()) if tt == tid and o not in on and gr >= 1]
    (out / "data").mkdir(parents=True, exist_ok=True)
    frames = {
        "queries": pl.DataFrame(q, schema_overrides={"graded": GRADED}, infer_schema_length=None),
        "agent_tasks": pl.DataFrame(t, schema_overrides={"graded": GRADED}, infer_schema_length=None),
        "relevance_grades": pl.DataFrame(g, schema_overrides={"grade": pl.Int8, "pass": pl.Int8}, infer_schema_length=None),
        "qrels": pl.DataFrame(qrels, schema_overrides={"relevance": pl.Int8}, infer_schema_length=None),
    }
    for k, df in frames.items():
        df.write_parquet(out / "data" / f"{k}.parquet")
    stats = {k: df.height for k, df in frames.items()}
    stats["queries_by_set"] = dict(frames["queries"].group_by("set").len().iter_rows())
    stats["tasks_by_kind"] = dict(frames["agent_tasks"].group_by("kind").len().iter_rows())
    stats["grades_by_run"] = dict(frames["relevance_grades"].group_by("grading_run").len().iter_rows())
    stats["qrels_by_source"] = dict(frames["qrels"].group_by("source").len().iter_rows())
    stats["decoys"] = frames["relevance_grades"].filter(pl.col("kind") == "decoy_random").height
    stats["decoys_accepted"] = frames["relevance_grades"].filter((pl.col("kind") == "decoy_random") & (pl.col("grade") >= 1)).height
    stats["missing_names"] = sum(df["name"].null_count() for k, df in frames.items() if "name" in df.columns)
    return stats


def build_precon_fit(out: Path) -> dict:
    nm = names()
    rows = []
    for run, split in FIT_RUNS.items():
        d = ROOT / "bench" / "precon" / "runs" / run
        cfg = json.loads((d / "config.json").read_text())
        kind = {(pid, o): k for pid, v in json.loads((d / "items.json").read_text()).items() for o, k in v}
        for c in map(json.loads, (d / "calls.jsonl").read_text().splitlines()):
            for i, o in enumerate(c["oids"], 1):
                if str(i) in c["grades"]:
                    rows.append({"precon_id": c["precon_id"], "split": split, "oracle_id": o, "name": nm.get(o),
                                 "kind": kind[(c["precon_id"], o)], "grade": int(c["grades"][str(i)]), "pass": int(c["pass"]),
                                 "rubric": cfg["rubric"], "grader_model": cfg["model"], "grading_run": run,
                                 "proposed_by": ",".join(cfg["runs"]) if kind[(c["precon_id"], o)] == "pool" else None})
    df = pl.DataFrame(rows, schema_overrides={"grade": pl.Int8, "pass": pl.Int8}, infer_schema_length=None)
    (out / "data").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "data" / "deck_fit.parquet")
    return {"rows": df.height, "by_split_kind": {f"{s}/{k}": n for s, k, n in df.group_by("split", "kind").len().iter_rows()},
            "rubric": df["rubric"].unique().to_list(), "missing_names": df["name"].null_count()}


def upload(repo: str, folder: Path, readme: Path | None, message: str, tag: str | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if readme:
        (folder / "README.md").write_text(readme.read_text())
    c = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder), commit_message=message)
    sha = c.oid
    if tag:
        api.create_tag(repo, tag=tag, revision=sha, repo_type="dataset", exist_ok=True)
    return sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["bench", "precon-fit"])
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--readme", default=None, help="dataset card to upload as README.md")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()
    if args.what == "bench":
        repo, out = "kasper-cbx/mtg-search-bench", ROOT / "data" / "hf" / "mtg-search-bench"
        stats = build_bench(out)
    else:
        repo, out = "kasper-cbx/mtg-precon-bench", ROOT / "data" / "hf" / "mtg-precon-bench-deckfit"
        stats = build_precon_fit(out)
    print(json.dumps(stats, indent=1))
    if args.upload:
        sha = upload(repo, out, Path(args.readme) if args.readme else None, args.message or f"{args.what}: dataset files", args.tag)
        print(f"uploaded {repo}@{sha}" + (f", tagged {args.tag}" if args.tag else ""))


if __name__ == "__main__":
    main()
