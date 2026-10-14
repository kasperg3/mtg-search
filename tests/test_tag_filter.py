"""Functional-vs-trivia oracle tag classification and role ranking (mtg_search.tag_filter)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mtg_search.render import role_tags
from mtg_search.tag_filter import TagFilter, deny_reason

# A slice of the real Scryfall hierarchy (slug -> parents) with card counts.
PARENTS = {
    "card-names": [], "portmanteau": ["card-names"], "alliteration": ["card-names"],
    "punny-name": ["card-names"],
    "cycle": [], "cycle-rav-shockland": ["cycle"], "tricycle-land": ["cycle"],
    "un-design": [], "un-set-mechanics": ["un-design"], "art-matters": ["un-set-mechanics"],
    "staple-with-set-s-mechanic": [], "bear-with-set-s-mechanic": ["staple-with-set-s-mechanic"],
    "ramp": [], "ramp-with-set-s-mechanic": ["ramp", "staple-with-set-s-mechanic"],
    "triggered-ability": [], "death-trigger": ["triggered-ability"],
    "death-trigger-self": ["death-trigger"], "leaves-body-behind": ["death-trigger-self"],
    "blue-effect": [], "counterspell": ["blue-effect"], "counterspell-soft": ["counterspell"],
    "removal": [], "removal-exile": ["removal"], "removal-creature": ["removal"],
    "sweeper": ["removal"], "lifegain": [], "opponent-lifegain": [],
    "tutor": [], "tutor-land": ["tutor"], "tutor-land-basic": ["tutor-land"],
    "tutor-land-plains": ["tutor-land"], "tutor-land-swamp": ["tutor-land"],
    "land-ramp": ["ramp"], "sacrifice-outlet": [], "sacrifice-outlet-creature": ["sacrifice-outlet"],
    "free-sacrifice-outlet": ["sacrifice-outlet"], "martyr": [],
    "characteristic-defining-ability": [], "cda-subtype": ["characteristic-defining-ability"],
    "changeling": ["cda-subtype"], "mana-value-matters": [], "scales-with-mana-value": ["mana-value-matters"],
    "synergy-blue": [], "synergy-artifact": [], "hate-regenerate": [],
}
COUNTS = {s: 100 for s in PARENTS} | {"removal-creature": 1898, "lifegain": 858, "removal-exile": 444,
                                       "opponent-lifegain": 58, "sweeper": 724, "hate-regenerate": 140}

TRIVIA = ["portmanteau", "alliteration", "punny-name", "cycle-rav-shockland", "tricycle-land",
          "art-matters", "bear-with-set-s-mechanic", "ramp-with-set-s-mechanic", "triggered-ability",
          "blue-effect", "synergy-blue", "scales-with-mana-value", "cda-subtype"]
FUNCTIONAL = ["ramp", "counterspell", "counterspell-soft", "death-trigger", "death-trigger-self",
              "leaves-body-behind", "removal-exile", "sacrifice-outlet-creature", "free-sacrifice-outlet",
              "tutor-land-basic", "changeling", "mana-value-matters", "synergy-artifact", "martyr"]


@pytest.fixture(scope="module")
def tf():
    return TagFilter(COUNTS, PARENTS)


@pytest.mark.parametrize("slug", TRIVIA)
def test_trivia_denied(tf, slug):
    assert not tf.is_functional(slug), (slug, tf.reasons.get(slug))


@pytest.mark.parametrize("slug", FUNCTIONAL)
def test_functional_kept(tf, slug):
    assert tf.is_functional(slug), (slug, tf.reasons.get(slug))


def test_deny_reasons():
    assert deny_reason("portmanteau", PARENTS) == "subtree:card-names"
    # a child of a self-only denied template tag is judged on its own
    assert deny_reason("triggered-ability", PARENTS) == "self"
    assert deny_reason("death-trigger", PARENTS) is None
    # mixed parents: survives the hierarchy rule, but the set-mechanic pattern catches it
    assert deny_reason("ramp-with-set-s-mechanic", PARENTS) == "pattern"
    # without hierarchy, patterns + lists still work
    assert deny_reason("cycle-ktk-gainland") == "pattern"
    assert deny_reason("synergy-green") == "self"
    assert deny_reason("removal-creature") is None


def test_roles_core_first_and_siblings_deferred(tf):
    swords = ["removal-creature", "lifegain", "removal-exile", "opponent-lifegain", "portmanteau"]
    assert tf.roles(swords) == ["removal-exile", "lifegain", "removal-creature", "opponent-lifegain"]
    wrath = ["removal-creature", "sweeper", "hate-regenerate", "triggered-ability"]
    assert tf.roles(wrath) == ["sweeper", "removal-creature", "hate-regenerate"]
    farseek = ["tutor-land-plains", "tutor-land-swamp", "land-ramp", "tutor-land-basic"]
    out = tf.roles(farseek, n=3)
    assert out[0] == "land-ramp" and sum(t.startswith("tutor-land-") for t in out) == 2


def test_role_tags_accepts_filter_or_counts(tf):
    tags = ["portmanteau", "counterspell", "synergy-blue"]
    assert role_tags(tags, tf) == ["counterspell"]
    assert role_tags(["synergy-blue", "counterspell", "scales-with-mana-value"], COUNTS) == ["counterspell"]


def test_from_frame_without_parents():
    df = pl.DataFrame({"slug": ["meme", "removal"], "card_count": [130, 50]})
    f = TagFilter.from_frame(df)
    assert f.roles(["meme", "removal"]) == ["removal"]
    assert TagFilter.from_frame(None).roles(["removal"]) == []


REAL = Path(__file__).parents[1] / "data" / "tags.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="needs data/tags.parquet")
def test_real_tags_parquet():
    f = TagFilter.from_frame(pl.read_parquet(REAL))
    for s in ["portmanteau", "alliteration", "synergy-blue", "scales-with-mana-value", "interrupt",
              "inscryption-achievement", "type-addition-human", "fun-ruling", "meme", "dnd-character",
              "cycle-rav-shockland", "intervening-if-clause", "unique-type-line", "french-vanilla"]:
        assert not f.is_functional(s), s
    for s in ["removal", "removal-exile", "ramp", "land-ramp", "mana-rock", "counterspell",
              "counterspell-soft", "sacrifice-outlet-creature", "free-sacrifice-outlet", "tutor-card",
              "draw-engine", "card-advantage", "sweeper", "reanimate-creature", "blood-artist-ability",
              "death-trigger-self", "leaves-body-behind", "changeling", "typal-elf", "hate-graveyard"]:
        assert f.is_functional(s), (s, f.reasons.get(s))
