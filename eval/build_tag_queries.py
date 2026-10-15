"""Build eval/queries_tags.yaml from Scryfall oracle tags (Tagger "otag").

Each entry in TAG_QUERIES maps a *functional* oracle tag to a natural-language
query written the way a deck builder would phrase it (deliberately avoiding the
tag slug wording where possible, so BM25 cannot win by matching the slug).

Relevance = every non-funny card tagged with the tag OR any descendant tag
(transitive closure over child_slugs). This mirrors Scryfall's `otag:` search,
which also matches child tags; e.g. `mana-rock` includes `moxen` and
`utility-mana-rock`. data/tags.parquet stores only *direct* taggings, so the
closure is computed here.

Selection rule: closure size (non-funny cards in data/cards.parquet) in
[MIN_RELEVANT, MAX_RELEVANT]. Tags outside that range are reported and skipped.

Run:  uv run python eval/build_tag_queries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for eval.popularity
CARDS = ROOT / "data" / "cards.parquet"
TAGS = ROOT / "data" / "tags.parquet"
OUT = ROOT / "eval" / "queries_tags.yaml"

MIN_RELEVANT = 20
MAX_RELEVANT = 500

# Reviewed tag -> query mapping. Keep in sync with README; edit by hand.
TAG_QUERIES: dict[str, str] = {
    # --- sacrifice / aristocrats -------------------------------------------
    "free-sacrifice-outlet": "free way to sacrifice my own creatures whenever I want",
    "your-sacrifice-matters": "payoff that rewards me every time I sacrifice a permanent",
    "blood-artist-ability": "whenever a creature dies, drain the opponent and gain that much life",
    "removal-sacrifice": "make my opponent sacrifice a creature (edict effect)",
    # --- counterspells ------------------------------------------------------
    "counterspell-soft": "counter a spell unless they pay extra mana, tax counter",
    "counterspell-creature": "counterspell that only stops creature spells",
    "counterspell-noncreature": "cheap counter for noncreature spells",
    "counterspell-ability": "stop an activated or triggered ability on the stack",
    # --- removal --------------------------------------------------------------
    "removal-tuck": "put an opponent's permanent on the bottom or into their library",
    "removal-bounce": "return an opposing creature or permanent to its owner's hand",
    "removal-fight": "my creature fights one of theirs",
    "one-sided-fight": "my creature deals damage equal to its power to their creature without taking any back",
    "doom-blade": "destroy target nonblack creature style removal",
    "pacifism": "aura that stops a creature from attacking or blocking",
    "removal-nonland": "deal with any nonland permanent, flexible answer",
    "removal-land": "destroy target land, land destruction",
    "mass-land-denial": "armageddon effect that wipes out everyone's lands",
    "sweeper-one-sided": "board wipe that only kills my opponents' stuff",
    # --- graveyard ---------------------------------------------------------
    "hate-graveyard": "graveyard hate to stop reanimator and flashback decks",
    "sweeper-graveyard": "exile all cards from every graveyard",
    "mass-reanimation": "bring back all creatures from the graveyard at once",
    "reanimate-from-opponent": "put a creature from an opponent's graveyard onto the battlefield under my control",
    "regrowth-instant": "get an instant back from my graveyard to my hand",
    "regrowth-creature": "return a creature card from my graveyard to my hand",
    "castable-from-graveyard": "spells I can cast again from the graveyard",
    "mill-opponent": "make an opponent put the top cards of their library into their graveyard",
    # --- card advantage / selection ---------------------------------------
    "wheel": "everyone discards their hand and draws a new one",
    "loot": "draw then discard to filter my hand",
    "rummage": "discard a card, then draw a card",
    "thoughtseize": "look at opponent's hand and pick a card for them to discard",
    "repeatable-impulsive-draw": "each turn exile the top card of my library and I may play it",
    "draw-hate": "punish opponents for drawing extra cards",
    # --- tutors --------------------------------------------------------------
    "tutor-card": "search my library for any card and put it into my hand",
    "tutor-creature": "find a specific creature from my deck",
    "wish": "fetch a card from outside the game / my sideboard",
    # --- mana ----------------------------------------------------------------
    "mana-rock": "artifact that taps for mana to ramp",
    "mana-dork": "cheap creature that taps for mana",
    "ritual": "one-shot spell that adds a burst of mana this turn",
    "extra-land": "play more than one land each turn",
    "cost-reducer": "make my spells cheaper to cast",
    "repeatable-treasures": "engine that keeps creating treasure tokens",
    "landfall": "trigger something every time a land enters under my control",
    # --- tokens / counters / go-wide ----------------------------------------
    "gives-pp-counters-to-all": "put a +1/+1 counter on each creature I control",
    "counter-doubler": "double the counters placed on my permanents",
    "overrun": "creatures get a big pump and trample for an alpha strike",
    "clone": "creature that enters as a copy of another creature",
    "copy-spell": "copy an instant or sorcery spell on the stack",
    "flicker": "exile my creature and return it to the battlefield to reuse its enters ability",
    # --- life ---------------------------------------------------------------
    "lifegain-matters": "payoff cards that get stronger when I gain life",
    "soul-warden-ability": "gain 1 life whenever a creature enters the battlefield",
    "hate-lifegain": "stop opponents from gaining life",
    "drain-life": "opponent loses life and I gain the same amount",
    # --- combat / protection / control -----------------------------------
    "fog": "prevent all combat damage this turn to survive an attack",
    "protects-all": "protect my whole board from removal",
    "gives-indestructible": "make my creatures indestructible",
    "gives-unblockable": "make a creature unblockable",
    "threaten": "steal an opponent's creature until end of turn, untap it and give it haste",
    "theft-creature": "gain permanent control of an opponent's creature",
    "force-attacker": "force opposing creatures to attack each combat (goad)",
    "tax-attack": "opponents must pay mana for each creature that attacks me",
    "rhystic": "punisher effect that does something unless an opponent pays",
    "extra-turn": "take an extra turn after this one",
    "extra-combat-phase": "get an additional combat phase",
    "alternate-win-condition": "card that just wins the game under a condition",
    "polymorph": "replace a permanent with a random card revealed from the top of its controller's library",
}


def tag_closure(tags: pl.DataFrame) -> dict[str, set[str]]:
    kids = dict(zip(tags["slug"].to_list(), tags["child_slugs"].to_list()))
    direct = {s: set(ids) for s, ids in zip(tags["slug"].to_list(), tags["oracle_ids"].to_list())}
    memo: dict[str, set[str]] = {}

    def go(slug: str, stack: frozenset[str]) -> set[str]:
        if slug in memo:
            return memo[slug]
        if slug in stack:  # guard against cycles
            return set()
        out = set(direct.get(slug, ()))
        for k in kids.get(slug) or []:
            out |= go(k, stack | {slug})
        memo[slug] = out
        return out

    return {s: go(s, frozenset()) for s in direct}


def main() -> None:
    from eval.popularity import with_popularity

    cards = with_popularity(pl.read_parquet(CARDS, columns=["oracle_id", "name", "is_funny"]))  # listing order
    eligible = set(cards.filter(~pl.col("is_funny"))["oracle_id"])
    rank = {
        o: (r if r is not None else 10**9)
        for o, r in zip(cards["oracle_id"], cards["edhrec_rank"])
    }
    tags = pl.read_parquet(TAGS)
    closure = tag_closure(tags)

    queries, skipped = [], []
    for slug, query in TAG_QUERIES.items():
        if slug not in closure:
            skipped.append((slug, "missing"))
            continue
        rel = closure[slug] & eligible
        if not MIN_RELEVANT <= len(rel) <= MAX_RELEVANT:
            skipped.append((slug, len(rel)))
            continue
        # sort by EDHREC popularity (stable, readable diffs)
        relevant = sorted(rel, key=lambda o: (rank[o], o))
        queries.append(
            {
                "id": f"tag-{slug}",
                "query": query,
                "source": f"otag:{slug}",
                "relevant": relevant,
            }
        )
        print(f"{len(relevant):4d}  otag:{slug:30s} {query}")

    OUT.write_text(
        "# Generated by eval/build_tag_queries.py -- do not edit by hand; edit TAG_QUERIES.\n"
        + yaml.safe_dump(queries, sort_keys=False, allow_unicode=True, width=120)
    )
    print(f"\nwrote {len(queries)} queries to {OUT.relative_to(ROOT)}")
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main()
