"""Project-wide constants: paths, model ids, remote endpoints."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MTG_SEARCH_DATA_DIR", PROJECT_ROOT / "data"))
INDEX_DIR = Path(os.environ.get("MTG_SEARCH_INDEX_DIR", PROJECT_ROOT / "index"))
RAW_DIR = DATA_DIR / "raw"

CARDS_PARQUET = DATA_DIR / "cards.parquet"
TAGS_PARQUET = DATA_DIR / "tags.parquet"
RULINGS_PARQUET = DATA_DIR / "rulings.parquet"
MANIFEST_JSON = DATA_DIR / "manifest.json"
# Popularity (EDHREC rank) is evaluation-only: written next to, never into, the shipped tables. The server
# and the Docker image never read or contain it (.dockerignore allow-list; HF path eval/, not data/).
POPULARITY_PARQUET = DATA_DIR / "eval" / "popularity.parquet"

SCHEMA_VERSION = 2  # 2: cards.parquet without prices; manifest fingerprint

USER_AGENT = "mtg-search/0.1"
SCRYFALL_API = "https://api.scryfall.com"
BULK_TYPES = ("oracle_cards", "oracle_tags", "rulings")

HF_REPO = "kasper-cbx/mtg-search-index"
HF_REPO_TYPE = "dataset"
# The index snapshot this code release searches (a scryfall-<stamp> tag of HF_REPO). An installed package
# (uvx / pip) downloads it on first use into index_home(); override with MTG_SEARCH_INDEX_REVISION.
INDEX_REVISION = os.environ.get("MTG_SEARCH_INDEX_REVISION", "scryfall-20261014090155")


def index_base() -> Path:
    """Where an installed package keeps its downloads: $MTG_SEARCH_HOME, else <user cache>/mtg-search.
    One folder per index snapshot (``<base>/<revision>/{data,embeddings}``) and one shared query encoder
    (``<base>/models/…``: it is pinned to one model revision, so snapshots never re-download its 1.28 GB)."""
    return Path(os.environ.get("MTG_SEARCH_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mtg-search")


def index_home(revision: str = INDEX_REVISION) -> Path:
    return index_base() / revision

LATE_MODEL = "lightonai/mLateOn"  # the only retriever (mlateon.py)
# where its artifacts sit under an index root (here, so the Docker download stage needs only config + artifacts)
LATE_SLUG = "lightonai__mLateOn"
EMBEDDINGS_SUBDIR = f"embeddings/late/{LATE_SLUG}"
MODEL_SUBDIR = f"models/{LATE_SLUG}-fp32"
