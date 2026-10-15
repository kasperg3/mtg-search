"""EXP-017 deck-aware judge: pool and control construction, prompts and leak checks, batching, scoring.
No network, no data files, no model calls."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.precon.harness.leakage import PromptLeak, preflight  # noqa: E402
from bench.precon.harness.tools import LEAK  # noqa: E402
from bench.precon.judge import (  # noqa: E402
    BATCH,
    N_CONTROLS,
    add_side,
    build_pool,
    controls,
    deck_text,
    episode_metrics,
    fit,
    judge_system,
    judge_user,
    kappa,
    make_jobs,
    parse_grades,
)
from bench.precon.loader import CardDB, Deck  # noqa: E402


def card(oid, name, ci="G", *, type_line="Creature — Elf", legal="legal", gc=False, text="", cmc=2.0):
    return {"oracle_id": oid, "name": name, "face_names": [], "color_identity": list(ci), "colors": list(ci),
            "legalities": {"commander": legal}, "game_changer": gc, "type_line": type_line, "doc_text": text,
            "cmc": cmc, "mana_cost": "{1}{G}", "rarity": "rare", "produced_mana": [], "tags": ["ramp"],
            "edhrec_rank": None, "is_funny": False, "is_legendary": False,
            "faces": [{"name": name, "mana_cost": "{1}{G}", "type_line": type_line, "oracle_text": text}],
            **{f"pips_{c}": 0 for c in "WUBRG"}}


CARDS = [card("cmd", "Tana, the Bloodsower", "GR", type_line="Legendary Creature — Elf",
              text="Whenever Tana, the Bloodsower deals combat damage to a player, create that many Saprolings.")]
CARDS += [card(f"d{i}", f"Deck Card {i}") for i in range(8)]
CARDS += [card("forest", "Forest", type_line="Basic Land — Forest", cmc=0), card("dl", "Deck Land", type_line="Land", cmc=0)]
CARDS += [card(f"r{i}", f"Random Card {i}") for i in range(20)]
CARDS += [card(f"t{i}", f"Top Card {i}") for i in range(8)]
CARDS += [card("blue", "Blue Card", "U"), card("banned", "Banned Card", legal="banned"), card("gc", "Big Card", gc=True)]
ROWS = {c["oracle_id"]: c for c in CARDS}


@pytest.fixture(scope="module")
def db():
    return CardDB(pl.DataFrame(CARDS).drop("faces", "mana_cost", "rarity", "colors", "is_legendary"))


@pytest.fixture()
def deck():
    cards = [{"oracle_id": "cmd", "qty": 1}] + [{"oracle_id": f"d{i}", "qty": 1} for i in range(8)] + \
            [{"oracle_id": "dl", "qty": 1}, {"oracle_id": "forest", "qty": 5}]
    return Deck("pc-t", ["cmd"], ["Tana, the Bloodsower"], frozenset("GR"), cards)


LABELS = [{"oracle_id": f"t{i}", "kind": "add", "rate": 0.5 - i / 20, "name": f"Top Card {i}"} for i in range(8)] + \
         [{"oracle_id": "r0", "kind": "add_land", "rate": 0.01, "name": "Random Card 0"}]


def ep(task, names, rep=0):
    out = {"adds": names} if task == "add" else {"swaps": [{"cut": c, "add": a} for c, a in names]}
    return {"precon_id": "pc-t", "task": task, "rep": rep, "output": out, "metrics": {}}


def test_pool_is_statically_valid_add_side(db, deck):
    eps = {"a": [ep("add", ["Top Card 0", "Blue Card", "Deck Card 1", "Banned Card", "Nope", "Big Card"] + ["Random Card 5"] * 20),
                 ep("swap", [("Deck Card 2", "Top Card 1"), ("Deck Card 3", "Deck Card 2")])],
           "b": [ep("cut", [])]}
    assert add_side(eps["a"][1]) == ["Top Card 1", "Deck Card 2"]
    assert len(add_side(ep("add", ["x"] * 30))) == 20
    # off identity, in deck, banned and unresolved are not judged; the Game Changer limit is per episode
    assert build_pool(eps, {"pc-t": deck}, db) == {"pc-t": {"t0", "gc", "r5", "t1"}}


def test_controls(db, deck):
    pool = {"t0", "t1", "r5"}
    c = controls(deck, pool, LABELS, db, "s")
    kinds = [k for _, k in c]
    assert [kinds.count(k) for k in N_CONTROLS] == list(N_CONTROLS.values())
    by = {k: [o for o, kk in c if kk == k] for k in N_CONTROLS}
    assert by["top_label"] == ["t2", "t3", "t4", "t5", "t6", "t7"]  # highest rate first, picked cards skipped
    assert not set(by["random"]) & (pool | {"r0"} | set(deck.qty))  # no labelled card (any rate), none in deck
    assert all(db[o].commander_legal and db[o].color_identity <= deck.color_identity for o in by["random"])
    assert set(by["in_deck"]) <= set(deck.qty) - {"cmd", "forest"}
    assert controls(deck, pool, LABELS, db, "s") == c and controls(deck, pool, LABELS, db, "other") != c


def test_deck_text_anonymised_and_ordered(db, deck):
    t = deck_text(deck, ROWS, db)
    assert "Tana" not in t and "your commander" in t
    assert "THE OTHER 14 CARDS" in t and "5x Forest" in t
    assert t.index("Deck Card 7") < t.index("Deck Land")  # lands last
    assert "roles:" not in t and "ramp" not in t  # no tags
    assert t.rstrip().endswith("Cards marked [Game Changer] in the deck: 0.")


def test_prompts_pass_preflight_and_hide_sources(db, deck):
    system, user = judge_system(deck), judge_user(deck, ["t0", "r1"], ROWS, db)
    assert "(RG)" in system and "[1]\nTop Card 0" in user and "[2]\nRandom Card 1" in user
    names = [db[o].name for o in [*deck.qty, "t0", "r1"]]
    kw = dict(deck_names=names, forbidden_gold={"Top Card 0", "Sol Ring"}, identifying={"Bloodsower Deck"})
    preflight(system, user, **kw)  # gold names are candidates in the user prompt by design
    with pytest.raises(PromptLeak, match="identifier"):
        preflight(system, user + "\nfrom the Bloodsower Deck", **kw)
    with pytest.raises(PromptLeak, match="popularity"):
        preflight(system, user + "\n(EDHREC rank #3)", **kw)
    assert not LEAK.search(system)  # the rubric itself uses no popularity words


def test_batches_cover_each_card_once_and_rejudge(db):
    items = {"p1": [(f"c{i}", "pool") for i in range(31)], "p2": [(f"e{i}", "pool") for i in range(15)]}
    jobs = make_jobs(items, 0.25, "s")
    p0 = [j for j in jobs if j[1] == 0]
    assert sorted(len(j[3]) for j in p0) == [10, 10, 11, 15] and max(len(j[3]) for j in jobs) <= BATCH
    assert sorted(o for j in p0 for o in j[3]) == sorted(o for v in items.values() for o, _ in v)
    p1 = [j for j in jobs if j[1] == 1]
    assert len(p1) == 1
    src = next(j for j in p0 if (j[0], j[2]) == (p1[0][0], p1[0][2]))
    assert sorted(src[3]) == sorted(p1[0][3])
    assert make_jobs(items, 0.25, "s") == jobs


def test_parse_grades():
    assert parse_grades('{"grades": [{"i": 1, "g": 2}, {"i": 2, "g": 0}, {"i": 9, "g": 1}, {"i": 3, "g": 5}]}', 3) == {1: 2, 2: 0}
    assert parse_grades('Note {U}{U}.\n```json\n{"grades": [{"i": 1, "g": 1}]}\n```', 1) == {1: 1}
    assert parse_grades("no json", 2) == {}


def test_fit_and_episode_metrics(db, deck):
    assert fit([2, 1, 0], 4) == pytest.approx(1.5 / 4)  # a missing slot counts 0
    assert fit([2, None, 0], 3) == pytest.approx(1 / 2)  # an ungraded pick is dropped
    assert fit([2, 2, 1], 3, two=True) == pytest.approx(2 / 3)
    grades = {"t0": 2, "t1": 1, "gc": 2}
    m = episode_metrics(ep("add", ["Top Card 0", "Blue Card", "Top Card 1", "Big Card", "Top Card 0"]), deck, db, grades)
    assert m["fit@20"] == pytest.approx((1 + 0 + 0.5 + 1 + 0) / 20)  # off identity and duplicate score 0
    assert m["share2@20"] == pytest.approx(2 / 20) and m["ungraded"] == 0
    m = episode_metrics(ep("swap", [("Deck Card 1", "Top Card 0"), ("Nope", "Top Card 1"), ("Deck Card 2", "Top Card 5")]), deck, db, grades)
    assert m["swap.fit@10"] == pytest.approx(1.5 / 9) and m["ungraded"] == 1


def test_kappa():
    assert kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert kappa([], []) is None
