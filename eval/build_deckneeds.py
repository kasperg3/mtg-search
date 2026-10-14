"""Build eval/queries_deckneeds.yaml: open-ended deck-building needs.

Each spec labels relevance as a union of
  * ``tags``    – Scryfall oracle tags (closure over child tags, like otag:),
  * ``regex``   – case-insensitive regexes over doc_text (reminder text stripped),
then intersects with ``filters`` (same kwargs as mtg_search.filters.filter_expr,
the eval runner applies them at query time too) and drops funny cards.
``exclude`` / ``include`` hold card names fixed by hand review; ``graded``
(tags/regex) marks grade-2 "core" answers inside the relevant set.

Run:  uv run python eval/build_deckneeds.py [--show N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from eval.build_tag_queries import tag_closure  # noqa: E402
from mtg_search.filters import filter_expr  # noqa: E402

OUT = ROOT / "eval" / "queries_deckneeds.yaml"
REMINDER = re.compile(r"\([^()]*\)")

SPECS: list[dict] = [
    {
        "id": "need-fliers-mono-green",
        "query": "I keep losing to fliers in my mono-green deck",
        "filters": {"color_identity": "G"},
        "tags": ["hate-flying", "removes-flying"],
        "exclude": ["Colossus Hammer"],
        "regex": [r"(destroy|deals? \w+ damage to) (target|each|all) creatures? with flying"],
        "graded": {"regex": [r"(destroy|deals? \w+ damage to) (target|each|all) creatures? with flying"]},
    },
    {
        "id": "need-refill-red-aggro",
        "query": "cheap ways to refill my hand in an aggressive red deck",
        "filters": {"color_identity": "R", "cmc_max": 3},
        "tags": ["impulsive-draw", "repeatable-impulsive-draw", "wheel", "burst-draw"],
    },
    {
        "id": "need-punish-spells",
        "query": "something to punish opponents who play lots of spells",
        "regex": [
            r"whenever (an opponent|a player|each opponent|target opponent|that player|enchanted player) casts (a|an|their)[^.]*spell[^.]*(damage|loses? (\d+|x|that much) life|sacrifices?|mills?|discards?)",
            r"whenever (an opponent|a player) casts (a|an|their) [^.]*spell, (that player|they|its controller) (loses|sacrifices|discards)",
        ],
        "tags": ["storm-count-matters"],
    },
    {
        "id": "need-protect-commander",
        "query": "protect my commander from removal",
        "filters": {"format": "commander", "types": ["Instant"]},
        "tags": ["gives-hexproof", "gives-shroud", "gives-protection", "gives-indestructible", "gives-ward", "protects-all"],
        "graded": {"tags": ["gives-hexproof", "gives-indestructible"]},
    },
    {
        "id": "need-aristocrats-value",
        "query": "I need ways to get value from creatures dying in an aristocrats deck",
        "tags": ["blood-artist-ability", "your-sacrifice-matters"],
        "regex": [r"whenever (a|another|one or more) (other )?(nontoken )?creatures?( you control)? (dies|die)\b"],
        "graded": {"tags": ["blood-artist-ability"]},
    },
    {
        "id": "need-reanimator-enablers",
        "query": "my reanimator deck needs ways to put big creatures in the graveyard",
        "tags": ["tutor-to-graveyard", "loot"],
        "regex": [r"search your library for [^.]*card[^.]* put (it|that card) into your graveyard"],
        "graded": {"tags": ["tutor-to-graveyard"]},
    },
    {
        "id": "need-pauper-black-removal",
        "query": "cheap creature removal for a pauper black deck",
        "filters": {"color_identity": "B", "format": "pauper", "cmc_max": 2},
        "tags": ["removal-creature"],
    },
    {
        "id": "need-white-wraths",
        "query": "board wipes for my white control deck",
        "filters": {"color_identity": "W"},
        "tags": ["sweeper"],
    },
    {
        "id": "need-green-cheap-ramp",
        "query": "cheap early ramp for my green commander deck",
        "filters": {"color_identity": "G", "format": "commander", "cmc_max": 2},
        "tags": ["land-ramp", "mana-dork", "mana-rock", "extra-land", "ritual"],
    },
    {
        "id": "need-blue-vs-reanimator",
        "query": "my opponents keep bringing creatures back from the dead, what can a blue deck play against that",
        "filters": {"color_identity": "U"},
        "tags": ["hate-graveyard", "sweeper-graveyard"],
    },
    {
        "id": "need-green-artifact-enchantment-answers",
        "query": "how do I deal with artifacts and enchantments in green",
        "filters": {"color_identity": "G"},
        "tags": ["removal-artifact", "removal-enchantment"],
    },
    {
        "id": "need-ub-card-advantage",
        "query": "card advantage for my blue-black control deck",
        "filters": {"color_identity": "UB", "cmc_max": 4},
        "tags": ["burst-draw"],
    },
    {
        "id": "need-tokens-anthem",
        "query": "my go-wide selesnya deck needs ways to make the whole army bigger",
        "filters": {"color_identity": "GW"},
        "tags": ["anthem", "gives-pp-counters-to-all", "overrun"],
    },
    {
        "id": "need-white-token-makers",
        "query": "white cards that flood the board with small soldiers or spirits",
        "filters": {"color_identity": "W"},
        "regex": [r"create (two|three|four|five|x|that many|a number of) [^.]*1/1 [^.]*creature tokens"],
    },
    {
        "id": "need-spells-creatures",
        "query": "creatures for my spells deck that get better as I cast instants and sorceries",
        "filters": {"types": ["Creature"]},
        "tags": ["prowess-like", "magecraft"],
        "regex": [r"whenever you cast (an|a|your first) (instant or sorcery|noncreature) spell"],
    },
    {
        "id": "need-lands-recursion",
        "query": "getting lands back from the graveyard for my lands deck",
        "tags": ["reanimate-land", "regrowth-land"],
        "regex": [r"play (lands|land cards|a land) from your graveyard"],
    },
    {
        "id": "need-stop-activated-combo",
        "query": "stop my opponent's combo by shutting off activated abilities",
        "tags": ["hate-activation", "prevent-activation"],
        "regex": [r"activated abilities (of [^.]*)?can't be activated"],
    },
    {
        "id": "need-anti-tokens",
        "query": "my opponent plays a lot of tokens and small creatures, how do I fight that",
        "tags": ["hate-token", "hate-wide"],
        "regex": [
            r"deals? (1|2|one|two) damage to each (creature|other creature|creature your opponents control|creature an opponent controls|creature you don't control)",
            r"(all|each) (other )?creatures?( your opponents control| you don't control)? gets? -(1|2)/-(1|2)",
        ],
    },
    {
        "id": "need-white-survive-aggro",
        "query": "white cards to stop fast aggro decks from killing me in combat",
        "filters": {"color_identity": "W"},
        "tags": ["fog", "pillowfort", "tax-attack", "fog-selective"],
    },
    {
        "id": "need-team-haste",
        "query": "give my whole team haste so new creatures attack right away",
        "regex": [r"creatures you control (have|gain) [^.]*haste", r"(each|other) creatures? you control (has|have|gains?) [^.]*haste"],
    },
    {
        "id": "need-black-drain-finisher",
        "query": "finisher for a mono-black deck that drains the whole table",
        "filters": {"color_identity": "B", "cmc_min": 4},
        "regex": [r"each opponent loses (\d+|x|that much|life equal|half)"],
        "tags": ["drain-life"],
        "graded": {"regex": [r"each opponent loses (\d+|x|that much|life equal|half)"]},
    },
    {
        "id": "need-colorless-draw",
        "query": "colorless card draw any commander deck can run",
        "filters": {"color_identity": "C", "format": "commander"},
        "tags": ["burst-draw", "repeatable-draw"],
    },
    {
        "id": "need-hexproof-answers",
        "query": "how do I kill a big creature I can't target in modern",
        "filters": {"format": "modern"},
        "tags": ["removal-sacrifice"],
        "regex": [r"each (player|opponent) sacrifices (a|an|one) (\w+ )?creature"],
    },
    {
        "id": "need-survive-wipe",
        "query": "protect my creatures from a board wipe",
        "tags": ["protects-all"],
        "regex": [
            r"creatures you control gain indestructible",
            r"regenerate (all|each) creatures? you control",
            r"(permanents|creatures) you control phase out",
        ],
    },
    {
        "id": "need-modern-blue-tempo",
        "query": "cheap interaction for a blue tempo deck in modern",
        "filters": {"color_identity": "U", "format": "modern", "cmc_max": 2},
        "tags": ["counterspell", "removal-bounce"],
    },
    {
        "id": "need-punish-attackers",
        "query": "I want to punish people for attacking me in multiplayer",
        "tags": ["pillowfort", "tax-attack"],
        "regex": [r"whenever (a|another) creature (attacks you|an opponent controls attacks you)", r"whenever (an opponent|a player) attacks you"],
    },
    {
        "id": "need-equipment-support",
        "query": "my equipment deck needs ways to find gear and suit up cheaply",
        "tags": ["tutor-artifact-equipment", "cost-reducer-equip-ability", "auto-equip"],
    },
    {
        "id": "need-orzhov-lifegain-payoffs",
        "query": "lifegain payoffs for a white-black deck",
        "filters": {"color_identity": "WB"},
        "tags": ["lifegain-matters", "lifegain-to-damage"],
    },
    {
        "id": "need-cheap-sac-fodder",
        "query": "cheap cards that make several bodies to sacrifice",
        "filters": {"cmc_max": 3},
        "regex": [r"create (two|three|four|x|that many|a number of) [^.]*creature tokens"],
    },
    {
        "id": "need-red-enchantment-answers",
        "query": "red deck needs some way to deal with enchantments",
        "filters": {"color_identity": "R"},
        "tags": ["removal-enchantment"],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    cards = pl.read_parquet(ROOT / "data" / "cards.parquet")
    base = cards.filter(~pl.col("is_funny"))
    rank = {o: (r if r is not None else 10**9) for o, r in zip(base["oracle_id"], base["edhrec_rank"])}
    texts = {o: REMINDER.sub("", d) for o, d in zip(base["oracle_id"], base["doc_text"])}
    name_of = dict(zip(base["oracle_id"], base["name"]))
    by_name = {n: o for o, n in name_of.items()}
    closure = tag_closure(pl.read_parquet(ROOT / "data" / "tags.parquet"))

    def label(tags=(), regex=()) -> set[str]:
        s: set[str] = set()
        for t in tags:
            assert t in closure, t
            s |= closure[t]
        for r in regex:
            rx = re.compile(r, re.I)
            s |= {o for o, d in texts.items() if rx.search(d)}
        return s & set(texts)

    out = []
    for sp in SPECS:
        filt = sp.get("filters") or {}
        allowed = set(base.filter(filter_expr(base, **filt))["oracle_id"]) if filt else set(texts)
        rel = label(sp.get("tags", []), sp.get("regex", [])) & allowed
        rel -= {by_name[n] for n in sp.get("exclude", [])}
        rel |= {by_name[n] for n in sp.get("include", [])} & allowed
        rel_sorted = sorted(rel, key=lambda o: (rank[o], o))
        rule = []
        if sp.get("tags"):
            rule.append("otag:" + " OR otag:".join(sp["tags"]))
        rule += [f"regex:{r}" for r in sp.get("regex", [])]
        e = {"id": sp["id"], "query": sp["query"]}
        if filt:
            e["filters"] = filt
        e["labeling_rule"] = " | ".join(rule) + (f" | filters:{filt}" if filt else "")
        e["relevant"] = rel_sorted
        if sp.get("graded"):
            g = label(sp["graded"].get("tags", []), sp["graded"].get("regex", [])) & rel
            e["graded"] = {o: 2 for o in rel_sorted if o in g}
        out.append(e)
        extra = f" (grade2={len(e['graded'])})" if "graded" in e else ""
        print(f"{len(rel):4d}  {sp['id']:42s}{extra}")
        if args.show:
            print("       ", [name_of[o] for o in rel_sorted[: args.show]])

    OUT.write_text(
        "# Generated by eval/build_deckneeds.py -- edit SPECS there.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120)
    )
    print(f"\nwrote {len(out)} queries to {OUT.name}")


if __name__ == "__main__":
    main()
