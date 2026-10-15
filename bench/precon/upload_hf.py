"""Upload a built EXP-007 snapshot to the HF dataset kasper-cbx/mtg-precon-bench.

    uv run python bench/precon/upload_hf.py --snapshot 2026-10-14

Creates the repo if missing,
writes the dataset card (bench/precon/dataset_card.md, counts filled from manifest.json) as
README.md and uploads data/<config>.parquet + manifest.json + tag_denylist.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
REPO = "kasper-cbx/mtg-precon-bench"
CONFIGS = ["decks", "labels", "precons", "commanders", "splits"]
EXTRA = ["tag_audit.parquet"]


def render_card(m: dict) -> str:
    tpl = (Path(__file__).parent / "dataset_card.md").read_text()
    c = m["counts"]
    sub = {
        "SNAPSHOT": m["snapshot_date"],
        "GIT_SHA": m["built_from_git_sha"],
        "MTGJSON_VERSION": m["sources"]["mtgjson"]["version"],
        "SCRYFALL_DATE": m["sources"]["scryfall_cards_parquet"]["scryfall_updated_at"],
        "EDHREC_BUILD": str(m["sources"]["edhrec"]["build_id"]),
        "EDHREC_REQUESTS": str(m["sources"]["edhrec"]["requests"]),
        "CUTOFF": m["split_rule"]["cutoff"],
        "N_PRECONS": str(c["precons"]),
        "N_MATCHED": str(c["edhrec_matched"]),
        "N_LABELS": str(c["labels"]),
        "LABELS_BY_KIND": ", ".join(f"{k} {v}" for k, v in sorted(c["labels_by_kind"].items())),
        "SPLITS_ALL": ", ".join(f"{k} {v}" for k, v in sorted(c["splits_all"].items())),
        "SPLITS_SCORED": ", ".join(f"{k} {v}" for k, v in sorted(c["splits_scored"].items())),
    }
    for k, v in sub.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    a = ap.parse_args()
    src = ROOT / "data" / "bench" / "precon" / a.snapshot
    m = json.loads((src / "manifest.json").read_text())
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    info = api.dataset_info(REPO)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "data").mkdir()
        for name in CONFIGS:
            shutil.copy(src / f"{name}.parquet", t / "data" / f"{name}.parquet")
        for name in EXTRA:
            shutil.copy(src / name, t / name)
        for name in ["manifest.json", "tag_denylist.json"]:
            shutil.copy(src / name, t / name)
        (t / "README.md").write_text(render_card(m))
        api.upload_folder(folder_path=td, repo_id=REPO, repo_type="dataset",
                          commit_message=f"EXP-007 precon bench snapshot {a.snapshot} (code {m['built_from_git_sha']})")
    info = api.dataset_info(REPO)
    print(f"uploaded; private={info.private}; sha={info.sha}; https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
