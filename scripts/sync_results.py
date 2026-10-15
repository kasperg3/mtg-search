"""Mirror experiment outputs to the HF dataset kasper-cbx/mtg-search-experiments.

    uv run python scripts/sync_results.py [-m "EXP-006 main run, partial"]

Git tracks the run summaries (runs/*.json, write-ups); the per-episode outputs and
transcripts are gitignored and only exist locally. This uploads both, so a crashed
machine loses at most the work since the last sync. Uploads are incremental (unchanged
files are skipped by hash). Refuses to upload if any file contains the GLM API key.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "kasper-cbx/mtg-search-experiments"
FOLDERS = [  # local path -> path in the repo (same layout)
    "eval/experiments/runs",
    "eval/experiments/REGISTRY.md",
    "eval/agentic/runs",
    "eval/agentic/results",
    "bench/precon/runs",
]
SKIP = ["**/__pycache__/**", "**/.cache/**", "**/.gitignore"]


def secrets() -> list[str]:
    vals = [os.environ.get("GLM_API_KEY", "")]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GLM_API_KEY="):
                vals.append(line.split("=", 1)[1].strip().strip("'\""))
    return [v for v in vals if len(v) >= 16]


def check_no_secrets(paths: list[Path]) -> None:
    keys = [k.encode() for k in secrets()]
    for p in paths:
        data = p.read_bytes()
        if any(k in data for k in keys):
            raise SystemExit(f"refusing to sync: API key found in {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="sync experiment results")
    args = ap.parse_args()

    files = []
    for f in FOLDERS:
        p = ROOT / f
        files += [p] if p.is_file() else [x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts]
    check_no_secrets(files)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    for f in FOLDERS:
        p = ROOT / f
        if not p.exists():
            continue
        if p.is_file():
            api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO, repo_type="dataset",
                            commit_message=f"{args.message} (git {sha}): {f}")
        else:
            api.upload_folder(folder_path=p, path_in_repo=f, repo_id=REPO, repo_type="dataset",
                              ignore_patterns=SKIP, commit_message=f"{args.message} (git {sha}): {f}")
    print(f"synced {len(files)} files to https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
