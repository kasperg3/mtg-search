"""Point the package at a new index snapshot and bump its patch version, so the publish workflow releases it.

    uv run python scripts/bump_index.py --latest                      # newest scryfall-* tag on the HF dataset
    uv run python scripts/bump_index.py scryfall-20261021060000       # a given tag

Edits ``INDEX_REVISION`` in ``src/mtg_search/config.py`` and ``__version__`` in ``src/mtg_search/__init__.py``.
Prints the new version, or nothing and exit code 3 when the package already uses that snapshot.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "mtg_search" / "config.py"
INIT = ROOT / "src" / "mtg_search" / "__init__.py"
REV = re.compile(r'(INDEX_REVISION = os\.environ\.get\("MTG_SEARCH_INDEX_REVISION", ")(scryfall-\d{14})("\))')
VER = re.compile(r'(__version__ = ")(\d+)\.(\d+)\.(\d+)(")')


def latest_tag() -> str:
    from huggingface_hub import HfApi

    sys.path.insert(0, str(ROOT / "src"))
    from mtg_search import config

    tags = [t.name for t in HfApi().list_repo_refs(config.HF_REPO, repo_type=config.HF_REPO_TYPE).tags]
    snaps = sorted(t for t in tags if re.fullmatch(r"scryfall-\d{14}", t))
    if not snaps:
        raise SystemExit("no scryfall-* tag on the dataset")
    return snaps[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?")
    ap.add_argument("--latest", action="store_true")
    a = ap.parse_args()
    if bool(a.tag) == a.latest:
        ap.error("give a tag or --latest")
    tag = latest_tag() if a.latest else a.tag
    if not re.fullmatch(r"scryfall-\d{14}", tag):
        ap.error(f"not a snapshot tag: {tag}")
    cfg = CONFIG.read_text()
    m = REV.search(cfg)
    if not m:
        raise SystemExit(f"INDEX_REVISION not found in {CONFIG}")
    if m.group(2) == tag:
        print(f"already on {tag}", file=sys.stderr)
        raise SystemExit(3)
    CONFIG.write_text(REV.sub(lambda x: x.group(1) + tag + x.group(3), cfg))
    init = INIT.read_text()
    v = VER.search(init)
    new = f"{v.group(2)}.{v.group(3)}.{int(v.group(4)) + 1}"
    INIT.write_text(VER.sub(lambda x: x.group(1) + new + x.group(5), init))
    print(new)


if __name__ == "__main__":
    main()
