"""EXP-007 precon scorer: validity rules, gains, metrics, aggregation (hand-computed values)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.precon.loader import CardDB, Deck  # noqa: E402
from bench.precon.scorer import (  # noqa: E402
    ScoreConfig,
    aggregate,
    bootstrap_ci,
    dcg,
    f1_at,
    grade,
    kendall_tau,
    ndcg,
    paired_bootstrap,
    precision_at,
    recall_at,
    score_episode,
    sign_test,
    validate_adds,
    validate_cuts,
)

L2 = math.log2


def card(oid, name, ci="G", *, type_line="Creature — Elf", legal="legal", gc=False, tags=(), cmc=2.0,
         produced=(), pips=None, rank=None, faces=None, text=""):
    pips = pips or {}
    return {
        "oracle_id": oid, "name": name, "face_names": faces or [], "color_identity": list(ci),
        "legalities": {"commander": legal}, "game_changer": gc, "type_line": type_line, "doc_text": text,
        "cmc": cmc, "produced_mana": list(produced), "tags": list(tags), "edhrec_rank": rank, "is_funny": False,
        **{f"pips_{c}": pips.get(c, 0) for c in "WUBRG"},
    }


CARDS = [
    card("cmd", "Cmdr", "GU", type_line="Legendary Creature — Elf", pips={"G": 1, "U": 1}),
    card("a", "Alpha", tags=["land-ramp"], pips={"G": 1}, rank=100),
    card("b", "Beta", pips={"G": 1}),
    card("c", "Gamma", "U", tags=["removal-creature"], cmc=3, pips={"U": 1}),
    card("d", "Delta", "U", tags=["sweeper"], cmc=4, pips={"U": 2}),
    card("e", "Epsilon", tags=["draw-engine"], pips={"G": 1}),
    card("j", "Jay", "U", gc=True, cmc=1, pips={"U": 1}),
    card("l1", "Tapped Grove", "GU", type_line="Land", produced=("G", "U"), cmc=0),
    card("forest", "Forest", "G", type_line="Basic Land — Forest", produced=("G",), cmc=0),
    card("island", "Island", "U", type_line="Basic Land — Island", produced=("U",), cmc=0),
    card("x1", "Xone", tags=["ramp"], rank=10),
    card("x2", "Xtwo", "U", rank=1000),
    card("x3", "Fire // Ice", "GU", faces=["Fire", "Ice"], type_line="Instant // Instant"),
    card("x4", "Xfour", gc=True, rank=1),
    card("x5", "Xfive", "U", gc=True, rank=10_000),
    card("x6", "Xsix", gc=True, rank=5),
    card("x7", "Xseven", "R"),
    card("x8", "Xeight", legal="banned"),
    card("rats", "Relentless Rats", "B", text="A deck can have any number of cards named Relentless Rats."),
    card("lx", "Shock Grove", "GU", type_line="Land — Forest Island", produced=("G", "U"), cmc=0),
]
TAGS = pl.DataFrame({
    "slug": ["ramp", "land-ramp", "removal", "removal-creature", "sweeper", "card-advantage", "repeatable-draw", "draw-engine"],
    "parent_slugs": [[], ["ramp"], [], ["removal"], ["removal"], [], ["card-advantage"], ["repeatable-draw"]],
})


@pytest.fixture(scope="module")
def db():
    return CardDB(pl.DataFrame(CARDS), TAGS)


@pytest.fixture()
def deck():
    names = {c["oracle_id"]: c["name"] for c in CARDS}
    cards = [{"oracle_id": o, "name": names[o], "qty": q, "is_land": o in ("l1", "forest"), "is_basic": o == "forest",
              "is_commander": o == "cmd"} for o, q in
             [("cmd", 1), ("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1), ("j", 1), ("l1", 1), ("forest", 3)]]
    return Deck("pc-test", ["cmd"], ["Cmdr"], frozenset("GU"), cards)


def lab(oid, kind, rate, synergy=None, new=False, headline=None):
    return {"precon_id": "pc-test", "oracle_id": oid, "name": oid, "kind": kind, "rate": rate, "synergy": synergy,
            "is_new_in_product": new, "headline": (not kind.endswith("_land")) if headline is None else headline}


CUT_LABELS = [lab("a", "cut", 0.60), lab("b", "cut", 0.20), lab("c", "cut", 0.35), lab("d", "cut", 0.55),
              lab("l1", "cut_land", 0.40), lab("forest", "cut_land", 0.16)]
ADD_LABELS = [lab("x1", "add", 0.35, 0.5), lab("x4", "add", 0.20, -0.1), lab("x5", "add", 0.06, 0.2),
              lab("x6", "add", 0.40, 0.1), lab("x2", "add", 0.16, 0.3, new=True), lab("lx", "add_land", 0.30, -0.31)]


# ------------------------------------------------------------------ gains and primitive metrics

def test_grade_boundaries():
    cut, add = ScoreConfig().cut_thresholds, ScoreConfig().add_thresholds
    assert [grade(x, cut) for x in (0.5, 0.4999, 0.30, 0.15, 0.1499, None)] == [3, 2, 2, 1, 0, 0]
    assert [grade(x, add) for x in (0.30, 0.2999, 0.15, 0.05, 0.0499)] == [3, 2, 2, 1, 0]
    assert grade(float("nan"), add) == 0


def test_primitives():
    assert dcg([3, 2, 1], 10) == pytest.approx(3 + 2 / L2(3) + 1 / 2)
    assert dcg([3, 2, 1], 2) == pytest.approx(3 + 2 / L2(3))
    assert ndcg([3, 2, 1], [1, 2, 3], 10) == pytest.approx(1.0)
    assert ndcg([0, 0], [0, 0], 10) is None
    assert ndcg([1, 3], [3, 1], 10) == pytest.approx((1 + 3 / L2(3)) / (3 + 1 / L2(3)))
    assert precision_at([3, 0, 1], 10) == pytest.approx(0.2)
    assert recall_at(["a", None, "c"], {"a", "b"}, 10) == pytest.approx(0.5)
    assert recall_at(["a"], set(), 10) is None
    assert f1_at(["a", "c"], {"a", "b"}, 10) == pytest.approx(2 * 0.1 * 0.5 / 0.6)


# ------------------------------------------------------------------ name resolution / validity

def test_resolution_exact_and_faces(db):
    assert db.resolve("alpha") == "a"
    assert db.resolve("  ALPHA ") == "a"
    assert db.resolve("Fire // Ice") == "x3"
    assert db.resolve("Ice") == "x3"
    assert db.resolve("Alpha {1}{G}") == "a"  # copied cost stripped
    assert db.resolve("Alph") is None  # exact only, no fuzzy
    assert db.resolve(None) is None


def test_validate_cuts(db, deck):
    ps = validate_cuts(["Alpha", "Xone", "Cmdr", "Alpha", "nonsense", "Forest"], deck, db)
    assert [(p.valid, p.reason) for p in ps] == [(True, None), (False, "not_in_deck"), (False, "is_commander"),
                                                 (False, "duplicate"), (False, "unresolved"), (True, None)]
    assert ps[-1].side == "land"


def test_validate_adds_rules(db, deck):
    ps = validate_adds(["Xone", "Xseven", "Xeight", "Alpha", "Forest", "Island", "Island", "Relentless Rats",
                        "Xone"], deck, db)
    got = [(p.raw, p.valid, p.reason) for p in ps]
    assert got == [
        ("Xone", True, None), ("Xseven", False, "off_identity"), ("Xeight", False, "not_legal"),
        ("Alpha", False, "in_deck"), ("Forest", True, None),  # basics excepted from singleton
        ("Island", True, None), ("Island", True, None),
        ("Relentless Rats", False, "off_identity"),  # any-number, but black
        ("Xone", False, "duplicate"),
    ]


def test_game_changer_limit_in_rank_order(db, deck):
    # deck already holds 1 GC (Jay); limit 3 -> two more GC adds allowed, the third is invalid
    ps = validate_adds(["Xfour", "Xfive", "Xsix"], deck, db, gc_limit=3)
    assert [p.reason for p in ps] == [None, None, "game_changer_limit"]
    # cutting Jay frees a slot
    ps = validate_adds(["Xfour", "Xfive", "Xsix"], deck, db, gc_limit=3, removed=["j"])
    assert all(p.valid for p in ps)
    ps = validate_adds(["Xfour", "Xfive"], deck, db, gc_limit=None)
    assert all(p.valid for p in ps)


# ------------------------------------------------------------------ CUT episode

def test_cut_episode(db, deck):
    out = {"cuts": ["Alpha", "Tapped Grove", "Beta", "Zzz", "Alpha", "Gamma", "Epsilon", "Cmdr"]}
    m = score_episode("cut", out, deck, CUT_LABELS, db)
    # headline list (land pick removed): Alpha g3, Beta g1, Zzz -, Alpha dup -, Gamma g2, Epsilon 0, Cmdr -
    d = 3 + 1 / L2(3) + 2 / L2(6)
    ideal = 3 + 3 / L2(3) + 2 / L2(4) + 1 / L2(5)  # Alpha .60, Delta .55, Gamma .35, Beta .20
    assert m["cut.ndcg@10"] == pytest.approx(d / ideal)
    assert m["cut.ndcg@10"] == pytest.approx(0.696554, abs=1e-6)
    dc = 0.60 + 0.20 / L2(3) + 0.35 / L2(6)
    ic = 0.60 + 0.55 / L2(3) + 0.35 / L2(4) + 0.20 / L2(5)
    assert m["cut.ndcg@10.cont"] == pytest.approx(dc / ic)
    # the land pick (Tapped Grove) does not cost a headline slot: 10 slots - 1 land -> headline @9 (review m3)
    assert m["cut.p@10"] == pytest.approx(3 / 9)
    assert m["cut.r@10"] == pytest.approx(2 / 3)  # G2 = {Alpha, Delta, Gamma}
    assert m["cut.f1@10"] == pytest.approx(2 * (2 / 9) * (2 / 3) / (2 / 9 + 2 / 3))
    assert m["cut.n_g2"] == 3
    # land side: [Tapped Grove g2] vs labels Tapped Grove .40 (g2), Forest .16 (g1)
    assert m["cut_land.ndcg@10"] == pytest.approx(2 / (2 + 1 / L2(3)))
    assert m["cut_land.n_picks"] == 1
    # validity over all 8 submitted picks: 5 valid (Alpha, Grove, Beta, Gamma, Epsilon)
    assert m["cut.n_submitted"] == 8 and m["cut.n_valid"] == 5 and m["cut.n_missing"] == 2
    assert m["cut.validity"] == pytest.approx(5 / 8)
    assert (m["cut.invalid.unresolved"], m["cut.invalid.duplicate"], m["cut.invalid.is_commander"]) == (1, 1, 1)
    # false cuts: Epsilon is valid but unlisted -> 1/5
    assert m["cut.false_cut_rate"] == pytest.approx(0.2)


def test_cut_truncated_to_10_and_empty(db, deck):
    m = score_episode("cut", {"cuts": []}, deck, CUT_LABELS, db)
    assert m["cut.ndcg@10"] == 0.0 and m["cut.validity"] is None and m["cut.n_missing"] == 10
    m = score_episode("cut", {"cuts": ["Beta"] * 3 + ["Alpha"] * 20}, deck, CUT_LABELS, db)
    assert m["cut.n_submitted"] == 10


# ------------------------------------------------------------------ ADD episode

def test_add_episode(db, deck):
    out = {"adds": ["Xone", "Xseven", "Xfour", "Xfive", "Xsix", "Alpha", "Xeight", "Island", "Xone", "Foo"]}
    m = score_episode("add", out, deck, ADD_LABELS, db)
    # headline list (Island -> land side): grades [3,0,2,1,0(GC over limit),0,0,0(dup),0]
    d = 3 + 2 / L2(4) + 1 / L2(5)
    ideal = 3 + 3 / L2(3) + 2 / L2(4) + 2 / L2(5) + 1 / L2(6)  # Xsix, Xone, Xtwo, Xfour, Xfive
    assert m["add.ndcg@10"] == pytest.approx(d / ideal)
    assert m["add.ndcg@10"] == pytest.approx(0.620456, abs=1e-6)
    assert m["add.ndcg@20"] == pytest.approx(d / ideal)
    # 20 slots - 1 land pick (Island) -> headline @19 for the @20 metrics (review m3)
    assert m["add.p@10"] == pytest.approx(0.3) and m["add.p@20"] == pytest.approx(3 / 19)
    assert m["add.r@20"] == pytest.approx(0.5)  # G2 = {Xsix, Xone, Xtwo, Xfour}; hit Xone, Xfour
    # synergy-filtered: Xfour (syn<0) drops to 0; ideal Xsix 3, Xone 3, Xtwo 2, Xfive 1
    ds, isyn = 3 + 1 / L2(5), 3 + 3 / L2(3) + 2 / L2(4) + 1 / L2(5)
    assert m["add.ndcg@10.syn"] == pytest.approx(ds / isyn)
    # thematic-free: Xtwo (new in product) drops; ideal Xsix 3, Xone 3, Xfour 2, Xfive 1
    assert m["add.ndcg@10.nothematic"] == pytest.approx(d / isyn)
    # continuous gains = r
    dc = 0.35 + 0.20 / L2(4) + 0.06 / L2(5)
    ic = 0.40 + 0.35 / L2(3) + 0.20 / L2(4) + 0.16 / L2(5) + 0.06 / L2(6)
    assert m["add.ndcg@10.cont"] == pytest.approx(dc / ic)
    # validity: Xone, Xfour, Xfive, Island valid of 10
    assert m["add.validity"] == pytest.approx(0.4)
    for r in ("off_identity", "game_changer_limit", "in_deck", "not_legal", "duplicate", "unresolved"):
        assert m[f"add.invalid.{r}"] == 1
    assert m["add.gc_final"] == 3
    # land side: [Island] unlisted vs Shock Grove label -> 0
    assert m["add_land.ndcg@10"] == 0.0 and m["add_land.n_picks"] == 1
    # popularity drift: mean log10 rank of valid adds with a rank (Xone 10, Xfour 1, Xfive 1e4)
    assert m["add.pop_drift"] == pytest.approx((1 + 0 + 4) / 3)


# ------------------------------------------------------------------ SWAP episode

def test_swap_episode(db, deck):
    swaps = [{"cut": "Jay", "add": "Xfour"}, {"cut": "Alpha", "add": "Xone"}, {"cut": "Beta", "add": "Xsix"},
             {"cut": "Tapped Grove", "add": "Island"}, {"cut": "Cmdr", "add": "Xtwo"}]
    m = score_episode("swap", {"swaps": swaps}, deck, CUT_LABELS + ADD_LABELS, db)
    # cut side headline [Jay 0, Alpha 3, Beta 1, Cmdr invalid]; land side [Tapped Grove 2]
    d = 3 / L2(3) + 1 / L2(4)
    ideal = 3 + 3 / L2(3) + 2 / L2(4) + 1 / L2(5)
    assert m["swap.cut.ndcg@10"] == pytest.approx(d / ideal)
    assert m["swap.cut.validity"] == pytest.approx(4 / 5)
    # add side: GC count after cutting Jay = 0 -> Xfour, Xsix valid; [Xfour 2, Xone 3, Xsix 3, Xtwo 2]
    da = 2 + 3 / L2(3) + 3 / L2(4) + 2 / L2(5)  # Island is on the land side
    ia = 3 + 3 / L2(3) + 2 / L2(4) + 2 / L2(5) + 1 / L2(6)
    assert m["swap.add.ndcg@10"] == pytest.approx(da / ia)
    assert m["swap.add.validity"] == pytest.approx(1.0)
    assert m["swap.gc_final"] == 2
    # pair coherence over pairs with both sides valid: (Jay,Xfour) no roles; (Alpha land-ramp, Xone ramp) share
    # "ramp"; (Beta, Xsix) none; (Grove, Island) both lands -> 2/4
    assert m["swap.pair_coherence"] == pytest.approx(0.5)
    # structure: lands 4 -> 4 (Grove out, Island in); toy deck 11 cards - 4 valid cuts + 5 valid adds = 12
    assert m["diag.lands_delta"] == 0 and m["diag.lands_flag"] is False
    assert m["diag.deck_size"] == 12
    # ramp: Alpha (land-ramp) out, Xone (ramp) in -> 1 -> 1
    assert m["diag.role.ramp"] == 1 and m["diag.role_delta.ramp"] == 0
    assert m["diag.role.removal"] == 1 and m["diag.role.wipe"] == 1 and m["diag.role.draw"] == 1
    assert m["diag.role_below_min.wipe"] is True
    # sources: G from Forest x3 (+Grove removed); U from Island (Grove removed)
    assert m["diag.sources.G"] == 3 and m["diag.sources_delta.G"] == -1
    assert m["diag.sources.U"] == 1 and m["diag.sources_delta.U"] == 0


def test_swap_gc_limit(db, deck):
    swaps = [{"cut": "Alpha", "add": "Xfour"}, {"cut": "Beta", "add": "Xfive"}]
    m = score_episode("swap", {"swaps": swaps}, deck, ADD_LABELS, db, ScoreConfig(gc_limit=2))
    assert m["swap.add.invalid.game_changer_limit"] == 1


def test_unknown_task(db, deck):
    with pytest.raises(ValueError):
        score_episode("trade", {}, deck, [], db)


# ------------------------------------------------------------------ aggregation

def test_aggregate_repeats_and_ci():
    eps = [{"precon_id": p, "rep": r, "metrics": {"m": v, "flag": True}}
           for p, r, v in [("p1", 0, 0.5), ("p1", 1, 0.7), ("p2", 0, 0.1), ("p2", 1, 0.3)]]
    a = aggregate(eps, n_boot=2000)
    assert set(a) == {"m"}  # booleans are not aggregated
    assert a["m"]["mean"] == pytest.approx(0.4)
    assert a["m"]["n_precons"] == 2
    assert a["m"]["repeat_sd"] == pytest.approx(math.sqrt(0.02))  # rep means 0.3, 0.5
    lo, hi = a["m"]["ci95"]
    assert 0.2 <= lo <= 0.4 <= hi <= 0.6


def test_bootstrap_constant_and_seeded():
    assert bootstrap_ci([0.3] * 5, 500) == pytest.approx((0.3, 0.3))
    assert bootstrap_ci([0.1, 0.9, 0.5], 500, seed=1) == bootstrap_ci([0.1, 0.9, 0.5], 500, seed=1)


def test_sign_test_and_paired():
    assert sign_test([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert sign_test([1, -1, 0]) == pytest.approx(1.0)
    pb = paired_bootstrap({"a": 0.5, "b": 0.6, "c": 0.1}, {"a": 0.4, "b": 0.6, "d": 0.9}, n_boot=500)
    assert pb["n"] == 2 and pb["mean_diff"] == pytest.approx(0.05)


def test_kendall_tau():
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    assert kendall_tau([1], [1]) is None


def test_full_name_beats_face_name():
    rows = [card("bolt2", "Emeritus of Conflict // Lightning Bolt", "R", faces=["Emeritus of Conflict", "Lightning Bolt"]),
            card("bolt", "Lightning Bolt", "R")]
    d = CardDB(pl.DataFrame(rows))
    assert d.resolve("Lightning Bolt") == "bolt"
    assert d.resolve("Emeritus of Conflict") == "bolt2"


# ------------------------------------------------------------------ baselines (synthetic DB)

def test_random_legal_seeded_and_valid(db, deck):
    from bench.precon.baselines.simple import RandomLegal

    rl = RandomLegal(db, seed=7)
    a, b = rl.propose("swap", deck, rep=0), rl.propose("swap", deck, rep=0)
    assert a == b  # seeded per (precon, task, rep)
    cuts = rl.propose("cut", deck)["cuts"]
    assert len(cuts) == 7 and "Cmdr" not in cuts and "Forest" not in cuts  # 7 non-commander nonbasics
    m = score_episode("cut", {"cuts": cuts}, deck, CUT_LABELS, db)
    assert m["cut.validity"] == 1.0
    adds = rl.propose("add", deck)["adds"]
    m = score_episode("add", {"adds": adds}, deck, ADD_LABELS, db)
    assert m["add.validity"] == 1.0 and m["add.gc_final"] <= 3


def test_popularity_oracle_order(db, deck):
    from bench.precon.baselines.simple import PopularityOracle

    po = PopularityOracle(db)
    adds = po.propose("add", deck)["adds"]
    # best global rank first, in identity, not in deck; GC limit: deck has 1 GC -> Xfour (1), Xsix (5) ok
    assert adds[:4] == ["Xfour", "Xsix", "Xone", "Xtwo"] and "Xfive" not in adds  # 3rd GC over the limit
    assert po.propose("cut", deck)["cuts"][-1] == "Alpha"  # ranked cards last (worst rank cut first)


def test_land_picks_do_not_cost_headline_slots(db, deck):
    """Review m3: lands are allowed, so a land pick shrinks the headline cut-off; an empty slot does not."""
    with_land = score_episode("cut", {"cuts": ["Alpha", "Tapped Grove", "Beta"]}, deck, CUT_LABELS, db)
    without = score_episode("cut", {"cuts": ["Alpha", "Beta"]}, deck, CUT_LABELS, db)
    # same 2 headline picks; with a land pick the headline is scored @9, with an empty slot @10
    assert with_land["cut.p@10"] == pytest.approx(2 / 9)
    assert without["cut.p@10"] == pytest.approx(2 / 10)
