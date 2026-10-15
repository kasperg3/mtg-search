"""Build eval/queries_handwritten.yaml: wording-sensitive queries labelled by regex.

Each spec has
  * ``query``     – what a deck builder would type (NL, not the regex),
  * ``regex``     – case-insensitive Python regex matched against ``doc_text`` of
                    every non-funny card in data/cards.parquet (the labeling rule),
  * ``graded``    – optional stricter regex; matches get grade 2 (others grade 1),
  * ``exclude``   – card names dropped after hand review (false positives),
  * ``include``   – card names added after hand review (regex misses),
  * ``scryfall``  – optional Scryfall query used only to *cross-check* the regex
                    (run with --scryfall; differences are printed for review).

Run:  uv run python eval/build_handwritten.py [--scryfall] [--show N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for eval.popularity
CARDS = ROOT / "data" / "cards.parquet"
OUT = ROOT / "eval" / "queries_handwritten.yaml"
CACHE = ROOT / "eval" / ".scryfall_cache"
REMINDER = re.compile(r"\([^()]*\)")

SPECS: list[dict] = [
    {
        "id": "hw-soft-counter",
        "query": "counter target spell unless its controller pays",
        "regex": r"counter (target|up to one target|the chosen) [^.]*?spell[^.]* unless its controller pays|choose target spell\.[^.]*(\.[^.]*)?counter that spell unless its controller pays",
        "graded": r"counter target spell unless its controller pays",
        "scryfall": 'o:"spell unless its controller pays"',
    },
    {
        "id": "hw-another-dies-draw",
        "query": "whenever another creature you control dies, draw a card",
        "regex": r"whenever (another|a) (nontoken )?creature you control dies,.{0,100}?draw (a|\w+) cards?",
        "graded": r"whenever another (nontoken )?creature you control dies, (you may )?draw a card",
        "scryfall": 'o:/whenever (another|a) (nontoken )?creature you control dies, .*draw/',
    },
    {
        "id": "hw-flicker-own",
        "query": "exile a creature you control, then return it to the battlefield right away",
        "regex": r"exile [^.]*\b(creatures?|permanents?|artifacts?)\b[^.]* you (control|own)[^.]*(,|\.) (then )?return (it|that card|those cards|them|the exiled card|each card exiled this way)[^.]* to the battlefield(?! at the beginning)",
        "scryfall": 'o:/exile .*you control.*return (it|that card|those cards) to the battlefield/',
    },
    {
        "id": "hw-flicker-end-step",
        "query": "exile target creature, return it to the battlefield at the beginning of the next end step",
        "regex": r"(?s)exile.{0,160}?return (it|that card|those cards|the exiled cards?|them|each card exiled this way|the exiled permanents?|that permanent|those permanents)[^.]* to the battlefield[^.]* at the beginning of the next end step",
        "scryfall": 'o:/exile .*return .* to the battlefield.* at the beginning of the next end step/',
    },
    {
        "id": "hw-dies-return-end-step",
        "query": "when it dies, return it from the graveyard to the battlefield at the beginning of the next end step",
        "regex": r"(dies|put into (a|your) graveyard)[^.]*, return (it|that card|that creature|those cards|it to the battlefield)[^.]* to the battlefield[^.]* at the beginning of the next end step",
        "scryfall": 'o:/dies.*return .* to the battlefield.* at the beginning of the next end step/',
    },
    {
        "id": "hw-one-blocker-max",
        "query": "creatures that can't be blocked by more than one creature",
        "regex": r"can't be blocked by more than one creature",
        "scryfall": 'o:"can\'t be blocked by more than one creature"',
    },
    {
        "id": "hw-three-or-more-blockers",
        "query": "can't be blocked except by three or more creatures",
        "regex": r"can't be blocked except by (three|four|\w+) or more creatures|can't be blocked except by two or more creatures",
        "graded": r"can't be blocked except by (three|four|five) or more creatures",
        "scryfall": "o:/can't be blocked except by \\w+ or more creatures/",
    },
    {
        "id": "hw-each-opponent-loses",
        "query": "each opponent loses life, not just one target player",
        "regex": r"each opponent loses (\d+|x|that much|life equal|half)",
        "scryfall": 'o:/each opponent loses (\\d+|x|that much|life equal|half)/',
    },
    {
        "id": "hw-target-player-loses",
        "query": "target player loses life",
        "regex": r"target (player|opponent) loses (\d+|x|that much|life equal|half)",
        "scryfall": 'o:/target (player|opponent) loses (\\d+|x|that much|life equal|half)/',
    },
    {
        "id": "hw-wrath-except",
        "query": "destroy all creatures except mine / except a chosen type",
        "regex": r"(destroy|exile) (all|each) [^.]*creatures?[^.]* except|(destroy|exile) (all|each) (other )?creatures? [^.]*(you don't control|your opponents control|that isn't|that aren't|not of the chosen)",
        "scryfall": 'o:/(destroy|exile) (all|each) .*creatures?.* except/',
    },
    {
        "id": "hw-cant-be-countered",
        "query": "this spell can't be countered",
        "regex": r"this spell can't be countered",
        "scryfall": 'o:"this spell can\'t be countered"',
    },
    {
        "id": "hw-your-spells-uncounterable",
        "query": "make my other spells uncounterable",
        "regex": r"(spells? (you control|you cast)[^.]* can't be countered)|(can't be countered[^.]*spells? you (control|cast))|(that spell can't be countered)",
        "scryfall": 'o:/spells? you (control|cast).* can\'t be countered/',
    },
    {
        "id": "hw-double-pp-counters",
        "query": "double the number of +1/+1 counters on a creature",
        "regex": r"double the number of \+1/\+1 counters|double the number of each kind of counter|double the number of counters",
        "graded": r"double the number of \+1/\+1 counters",
        "scryfall": 'o:/double the number of (\\+1\\/\\+1 |each kind of )?counters/',
    },
    {
        "id": "hw-cant-gain-life",
        "query": "players can't gain life",
        "regex": r"(players|your opponents|each opponent|target player|that player|opponents|each player|they|enchanted player|its controller|that creature's controller)[^.]* can't gain life",
        "scryfall": 'o:"can\'t gain life" -o:"you can\'t gain life"',
    },
    {
        "id": "hw-cant-block-this-turn",
        "query": "target creature can't block this turn",
        "regex": r"target creature[^.]* can't block this turn|target creatures? can't block this turn",
        "scryfall": 'o:/target creatures? (\\w+ )?can\'t block this turn/',
    },
    {
        "id": "hw-second-spell",
        "query": "whenever you cast your second spell each turn",
        "regex": r"your second spell each turn",
        "scryfall": 'o:"your second spell each turn"',
    },
    {
        "id": "hw-haste-sacrifice-end",
        "query": "token gains haste, sacrifice it at the beginning of the next end step",
        "regex": r"(haste|attacking)[^.]*\.[^.]*(sacrifice|exile) (it|them|that creature|those creatures|those tokens|that token)[^.]* at the beginning of the next end step",
        "scryfall": 'o:/haste.*(sacrifice|exile) (it|them|that token|those tokens) at the beginning of the next end step/',
    },
    {
        "id": "hw-untap-all-lands",
        "query": "untap all lands you control",
        "regex": r"untap all lands you control|untap all lands\b",
        "scryfall": 'o:/untap all lands/',
    },
    {
        "id": "hw-bite",
        "query": "my creature deals damage equal to its power to target creature I don't control",
        "regex": r"deals damage equal to (its|their) power to (another )?(target|up to one target|each) creature (or planeswalker )?(you don't control|an opponent controls|your opponents control)",
        "scryfall": 'o:/deals damage equal to (its|their) power to (another )?(target|up to one target) creature.*(you don\'t control|an opponent controls)/',
    },
    {
        "id": "hw-reanimate-own",
        "query": "return target creature card from your graveyard to the battlefield",
        "regex": r"return (target|up to one target|another target) creature card[^.]* from your graveyard to the battlefield",
        "scryfall": 'o:/return (target|up to one target|another target) creature card.* from your graveyard to the battlefield/',
    },
    {
        "id": "hw-reanimate-any-gy",
        "query": "put a creature card from any graveyard onto the battlefield under my control",
        "regex": r"(put|return) (target|up to one target) creature card from (a|an opponent's) graveyard (onto|to) the battlefield under your control|creature card from (a|an opponent's) graveyard onto the battlefield under your control",
        "scryfall": 'o:/creature card from (a|an opponent\'s) graveyard (onto|to) the battlefield under your control/',
    },
    {
        "id": "hw-opp-land-enters",
        "query": "punish opponents whenever a land enters under their control",
        "regex": r"whenever a land (an opponent controls enters|enters under an opponent's control|you don't control enters|enters the battlefield under an opponent's control)|whenever an opponent plays a land",
        "scryfall": 'o:/whenever (a land (an opponent controls enters|enters( the battlefield)? under an opponent\'s control)|an opponent plays a land)/',
    },
    {
        "id": "hw-opponent-draws-trigger",
        "query": "whenever an opponent draws a card, punish them",
        "regex": r"whenever (an opponent|a player|each opponent|target opponent|that player|enchanted player) draws (a|their first|their second|\w+) cards?",
        "scryfall": 'o:/whenever (an opponent|a player) draws/',
        "graded": r"whenever an opponent draws a card",
    },
    {
        "id": "hw-play-lands-from-gy",
        "query": "you may play lands from your graveyard",
        "regex": r"play (lands|land cards) from your graveyard|play a land from your graveyard",
        "scryfall": 'o:/play (lands|land cards|a land) from your graveyard/',
    },
    {
        "id": "hw-play-from-top",
        "query": "play cards from the top of my library",
        "regex": r"(play|cast) [^.]*from the top of your library",
        "scryfall": 'o:/(play|cast) .*from the top of your library/',
    },
    {
        "id": "hw-basic-to-bf-tapped",
        "query": "search your library for a basic land card and put it onto the battlefield tapped",
        "regex": r"search your library for [^.]{0,80}basic lands? cards?[^.]*\.?[^.]*onto the battlefield tapped",
        "scryfall": 'o:/search your library for (a|up to \\w+|two) basic lands? cards?.* onto the battlefield tapped/',
    },
    {
        "id": "hw-3-damage-any-target",
        "query": "deals 3 damage to any target",
        "regex": r"deals 3 damage to any target",
        "scryfall": 'o:"deals 3 damage to any target"',
    },
    {
        "id": "hw-each-player-sac-creature",
        "query": "each player sacrifices a creature",
        "regex": r"each (player|opponent) sacrifices (a|an|one) (\w+ )?creature",
        "graded": r"each player sacrifices a creature",
        "scryfall": 'o:/each (player|opponent) sacrifices (a|an|one) (\\w+ )?creature/',
    },
    {
        "id": "hw-reveal-hand-choose-nonland",
        "query": "target opponent reveals their hand, you choose a nonland card from it and they discard it",
        "regex": r"reveals (their|his or her) hand[^.]*[.,] (then )?you (may )?choose [^.]*cards? from (it|among them)[^.]*\. (if you do, )?(that player|they|target player|the player) discards",
        "graded": r"reveals (their|his or her) hand[^.]*\. you (may )?choose a nonland card from it",
        "scryfall": 'o:/reveals (their|his or her) hand.* you choose (a|an) .*card from it/',
    },
    {
        "id": "hw-fog-all-combat",
        "query": "prevent all combat damage that would be dealt this turn",
        "regex": r"prevent all combat damage that would be dealt this turn",
        "scryfall": 'o:"prevent all combat damage that would be dealt this turn"',
    },
    {
        "id": "hw-opp-enter-tapped",
        "query": "creatures and lands my opponents control enter tapped",
        "regex": r"(creatures|permanents|artifacts|lands|nonbasic lands)[^.]* (your opponents control|opponents control|an opponent controls) enter( the battlefield)? tapped",
        "scryfall": 'o:/(your opponents|opponents|an opponent) controls? enters? (the battlefield )?tapped/',
    },
    {
        "id": "hw-threaten",
        "query": "gain control of target creature until end of turn, untap it, it gains haste",
        "regex": r"gain control of [^.]*creatures?[^.]* until end of turn\. untap",
        "scryfall": 'o:/gain control of .*creatures?.* until end of turn\\. untap/',
    },
    {
        "id": "hw-play-until-next-turn",
        "query": "exile the top card of your library, you may play it until the end of your next turn",
        "regex": r"(?s)exile the top [^.]* of your library\..{0,200}?until the end of your next turn, you may play",
        "scryfall": 'o:/exile the top .* of your library\\..*until the end of your next turn, you may play/',
    },
    {
        "id": "hw-lifegain-pp-counter",
        "query": "whenever you gain life, put a +1/+1 counter on a creature",
        "regex": r"whenever you gain life, put (a|an|that many|x|\w+) \+1/\+1 counters?",
        "scryfall": 'o:/whenever you gain life, put .*\\+1\\/\\+1 counter/',
    },
    {
        "id": "hw-die-exile-instead",
        "query": "if a card would be put into a graveyard from anywhere, exile it instead",
        "regex": r"would (die|be put into (a|an opponent's|your opponents'|that player's|a player's) graveyard from anywhere)[^.]*, exile (it|that card) instead",
        "scryfall": 'o:/would (die|be put into .*graveyard from anywhere).*, exile (it|that card) instead/',
    },
    {
        "id": "hw-regrow-instant-sorcery",
        "query": "return target instant or sorcery card from your graveyard to your hand",
        "regex": r"return (target|up to (one|two|three|x) target|another target) instant (or|and/or) sorcery cards? from your graveyard to your hand",
        "scryfall": 'o:/return (target|up to \\w+ target) instant (or|and\\/or) sorcery cards? from your graveyard to your hand/',
    },
    {
        "id": "hw-tax-spells-cost-more",
        "query": "spells my opponents cast cost more to cast",
        "regex": r"spells[^.]* cost \{\d+\} more to cast|spells[^.]* cost \{x\} more to cast",
        "scryfall": 'o:/spells.* cost \\{\\d\\} more to cast/',
    },
    {
        "id": "hw-opp-creature-dies",
        "query": "whenever a creature an opponent controls dies",
        "regex": r"whenever (a|another) (nontoken )?creature (an opponent controls|you don't control|your opponents control) dies",
        "scryfall": 'o:/whenever a (nontoken )?creature (an opponent controls|you don\'t control) dies/',
    },
    {
        "id": "hw-opp-upkeep",
        "query": "at the beginning of each opponent's upkeep",
        "regex": r"at the beginning of each opponent's upkeep",
        "scryfall": 'o:"at the beginning of each opponent\'s upkeep"',
    },
    {
        "id": "hw-each-opponent-mills",
        "query": "each opponent mills cards",
        "regex": r"each opponent mills",
        "scryfall": 'o:"each opponent mills"',
    },
    {
        "id": "hw-land-from-hand",
        "query": "put a land card from your hand onto the battlefield",
        "regex": r"put (a|up to one|up to two|any number of) land cards? from your hand onto the battlefield",
        "scryfall": 'o:/put (a|up to \\w+|any number of) land cards? from your hand onto the battlefield/',
    },
    {
        "id": "hw-donate",
        "query": "give an opponent control of a permanent I own",
        "regex": r"target (opponent|player) gains control of|(an opponent|that player|target opponent) gains control of (it|target|this|that|up to one)",
        "scryfall": 'o:/(target (opponent|player)|an opponent|that player) gains control of/',
    },
    {
        "id": "hw-oblivion-ring",
        "query": "exile target permanent an opponent controls until this leaves the battlefield",
        "regex": r"exile [^.]*until [^.]* leaves the battlefield",
        "scryfall": 'o:/exile .*until .* leaves the battlefield/',
    },
    {
        "id": "hw-earthquake",
        "query": "deal damage to each creature and each player",
        "regex": r"deals? (\d+|x|that much|damage equal[^.]*?) damage to each (creature|other creature|creature without flying|creature with flying|nonartifact creature)[^.]* and each (player|opponent)",
        "graded": r"deals? (\d+|x) damage to each creature (without flying )?and each player",
        "scryfall": 'o:/deals? \\w+ damage to each .*creature.* and each (player|opponent)/',
    },
    {
        "id": "hw-mass-bounce",
        "query": "return all creatures or nonland permanents to their owners' hands",
        "regex": r"return (all|each) [^.]*(creatures|nonland permanents|permanents|artifacts)[^.]* to their owners' hands",
        "scryfall": 'o:/return (all|each) .*(creatures|permanents).* to their owners\' hands/',
    },
    {
        "id": "hw-copy-instant-sorcery",
        "query": "copy target instant or sorcery spell, you may choose new targets",
        "regex": r"copy (target|that|it|the next) instant or sorcery spell|copy target instant (or|and/or) sorcery spell|copy target (instant|sorcery) spell",
        "scryfall": 'o:/copy (target|that) instant or sorcery spell/',
    },
]


def scryfall_oracle_ids(q: str) -> set[str]:
    """All oracle_ids matching a Scryfall query (cached, ≤2 req/s)."""
    CACHE.mkdir(exist_ok=True)
    key = CACHE / (re.sub(r"[^a-z0-9]+", "_", q.lower())[:120] + ".json")
    if key.exists():
        return set(json.loads(key.read_text()))
    url = "https://api.scryfall.com/cards/search?" + urllib.parse.urlencode(
        {"q": q, "unique": "cards"}
    )
    out: set[str] = set()
    while url:
        req = urllib.request.Request(
            url, headers={"User-Agent": "mtg-search/0.1", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = json.load(r)
        except urllib.error.HTTPError as e:  # 404 = no results
            if e.code != 404:
                print(f"      scryfall error {e.code} for {q!r}")
                return set()
            break
        time.sleep(0.55)
        for c in page.get("data", []):
            if c.get("oracle_id"):
                out.add(c["oracle_id"])
            elif c.get("card_faces"):
                out.add(c["card_faces"][0].get("oracle_id"))
        url = page.get("next_page") if page.get("has_more") else None
    key.write_text(json.dumps(sorted(out)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scryfall", action="store_true", help="cross-check with Scryfall API")
    ap.add_argument("--show", type=int, default=0, help="print N sample names per query")
    args = ap.parse_args()

    from eval.popularity import with_popularity

    cards = with_popularity(pl.read_parquet(CARDS, columns=["oracle_id", "name", "doc_text", "is_funny"]))  # listing order
    cards = cards.filter(~pl.col("is_funny")).sort(
        pl.col("edhrec_rank").fill_null(10**9), "oracle_id"
    )
    rows = list(zip(cards["oracle_id"], cards["name"], cards["doc_text"]))
    by_name = {n: o for o, n, _ in rows}
    name_of = {o: n for o, n, _ in rows}
    # Strip reminder text "(...)" like Scryfall's o: search does, so keyword reminder
    # text (menace, extort, unearth, ...) doesn't create spurious matches.
    texts = {o: REMINDER.sub("", d) for o, _, d in rows}

    out = []
    for spec in SPECS:
        rx = re.compile(spec["regex"], re.I)
        rel = [o for o, _, _ in rows if rx.search(texts[o])]
        excl = {by_name[n] for n in spec.get("exclude", [])}
        incl = [by_name[n] for n in spec.get("include", []) if by_name[n] not in rel]
        rel = [o for o in rel if o not in excl] + incl
        entry = {
            "id": spec["id"],
            "query": spec["query"],
            "labeling_rule": f"regex(doc_text, i): {spec['regex']}",
        }
        if spec.get("exclude") or spec.get("include"):
            entry["labeling_rule"] += (
                f" | hand-review: -{spec.get('exclude', [])} +{spec.get('include', [])}"
            )
        if spec.get("scryfall"):
            entry["scryfall_check"] = spec["scryfall"]
        entry["relevant"] = rel
        if spec.get("graded"):
            g = re.compile(spec["graded"], re.I)
            entry["graded"] = {o: 2 for o in rel if g.search(texts[o])}
        out.append(entry)

        line = f"{len(rel):4d}  {spec['id']:30s}"
        if spec.get("graded"):
            line += f" (grade2={len(entry['graded'])})"
        if args.scryfall and spec.get("scryfall"):
            sf = scryfall_oracle_ids(spec["scryfall"]) & set(name_of)
            only_sf = sf - set(rel)
            only_rx = set(rel) - sf
            line += f"  scryfall={len(sf)} only_scryfall={len(only_sf)} only_regex={len(only_rx)}"
            print(line)
            for o in list(only_sf)[:15]:
                print("      +SF", name_of[o])
            for o in list(only_rx)[:15]:
                print("      +RX", name_of[o])
        else:
            print(line)
        if args.show:
            print("      ", [name_of[o] for o in rel[: args.show]])

    OUT.write_text(
        "# Generated by eval/build_handwritten.py -- edit SPECS there, not this file.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120)
    )
    print(f"\nwrote {len(out)} queries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
