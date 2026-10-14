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

SCHEMA_VERSION = 2  # 2: cards.parquet without prices; manifest fingerprint

USER_AGENT = "mtg-search/0.1"
SCRYFALL_API = "https://api.scryfall.com"
BULK_TYPES = ("oracle_cards", "oracle_tags", "rulings")

HF_REPO = "kasper-cbx/mtg-search-index"
HF_REPO_TYPE = "dataset"

LATE_MODEL = "lightonai/mLateOn"
DENSE_MODEL = "google/embeddinggemma-2"
SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
