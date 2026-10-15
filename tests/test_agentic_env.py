"""The agentic scorer's name resolution (eval/agentic/env.py), without loading the card data or a model."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "agentic"))

from env import Env  # noqa: E402


def _env(names, faces):
    env = Env.__new__(Env)  # skip __init__: it reads data/cards.parquet and loads the retriever
    env.cards = pl.DataFrame({"oracle_id": [f"o{i}" for i in range(len(names))], "name": names,
                              "face_names": faces, "is_funny": [False] * len(names)})
    env._name_index()
    return env


def test_full_name_beats_other_cards_face_name():
    # "Emeritus of Conflict // Lightning Bolt" sorts first, and its back face is named "Lightning Bolt"
    env = _env(["Emeritus of Conflict // Lightning Bolt", "Lightning Bolt"], [["Emeritus of Conflict", "Lightning Bolt"], []])
    assert env.resolve_name("Lightning Bolt") == "o1"
    assert env.resolve_name("Emeritus of Conflict") == "o0"
    assert env.resolve_name("Emeritus of Conflict // Lightning Bolt") == "o0"


def test_face_name_still_resolves_when_no_card_has_it_as_full_name():
    env = _env(["Grave Researcher // Reanimate"], [["Grave Researcher", "Reanimate"]])
    assert env.resolve_name("Reanimate") == "o0"
