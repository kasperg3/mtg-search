import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def raw_cards():
    return _load("cards.jsonl")


@pytest.fixture(scope="session")
def raw_tags():
    return _load("tags.jsonl")


@pytest.fixture(scope="session")
def raw_rulings():
    return _load("rulings.jsonl")


@pytest.fixture(scope="session")
def by_name(raw_cards):
    return {c["name"]: c for c in raw_cards}


@pytest.fixture(scope="session")
def cards_df(raw_cards, raw_tags):
    from mtg_search.ingest import build_cards, tags_by_oracle_id

    return build_cards(raw_cards, tags_by_oracle_id(raw_tags))
