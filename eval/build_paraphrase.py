"""Build eval/queries_paraphrase.yaml: two reformulations of every original query.

For each query in queries_tags.yaml and queries_handwritten.yaml:
  * ``<id>-para`` – casual player paraphrase avoiding rules-text wording,
  * ``<id>-need`` – deck-need / role phrasing ("my X deck needs ...").
Labels (relevant / graded) are copied from the original; ``origin`` points back.

Reformulations were checked with eval/lexical_overlap.py and rewritten while
their content-bigram overlap with the relevant cards' rules text stayed high.

Run:  uv run python eval/build_paraphrase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.queries import load_queries  # noqa: E402

OUT = Path(__file__).resolve().parent / "queries_paraphrase.yaml"

# id -> (casual paraphrase, deck-need phrasing)
REFORMULATIONS: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------------- tags
    "tag-free-sacrifice-outlet": (
        "zero-cost altar to throw my guys into at any moment",
        "my aristocrats list needs an engine piece that lets me cash in my own dudes instantly, no mana required",
    ),
    "tag-your-sacrifice-matters": (
        "stuff that gets value whenever I feed something to an altar",
        "I'm building around giving up my own stuff and want cards that reward me each time I do it",
    ),
    "tag-blood-artist-ability": (
        "drain trigger every time anything hits the bin",
        "my death-heavy deck needs a finisher that slowly bleeds the table out as bodies pile up",
    ),
    "tag-removal-sacrifice": (
        "forces them to give up one of their own guys",
        "answers to hexproof threats that don't need to aim at anything",
    ),
    "tag-counterspell-soft": (
        "soft counter that taxes the opponent's spell",
        "my control deck needs cheap ways to stop spells early, but I'm fine if they can pay through it late",
    ),
    "tag-counterspell-creature": (
        "permission that only hits monsters as they're being summoned",
        "blue answers for decks full of big bodies, I only care about stopping beaters",
    ),
    "tag-counterspell-noncreature": (
        "cheap permission for non-monster spells",
        "my tempo deck wants to stop opposing removal and sweepers while my threats stay on board",
    ),
    "tag-counterspell-ability": (
        "blank an ability as it happens",
        "I want to blank searches, arrival effects and planeswalker activations before they happen",
    ),
    "tag-removal-tuck": (
        "shuffle their threat away into the deck instead of killing it",
        "answers for indestructible bombs and graveyard-recursion threats that won't come back",
    ),
    "tag-removal-bounce": (
        "send their stuff back to hand for tempo",
        "my tempo deck wants to temporarily clear blockers and reset tokens or auras",
    ),
    "tag-removal-fight": (
        "green removal where my beast brawls with theirs",
        "green has no kill spells, I need ways for my big guys to take out their stuff",
    ),
    "tag-one-sided-fight": (
        "bite effect, my guy punches theirs and takes nothing back",
        "green removal that doesn't risk my own beater",
    ),
    "tag-doom-blade": (
        "kill spell that can't hit black things",
        "black-ish cheap creature kill that's great except against the mirror",
    ),
    "tag-pacifism": (
        "enchantment that neuters one of their guys so it can't fight",
        "white answers that lock down a big threat without killing it",
    ),
    "tag-removal-nonland": (
        "catch-all answer for basically anything they play",
        "my commander deck wants flexible answers that handle artifacts, enchantments, walkers or beaters",
    ),
    "tag-removal-land": (
        "blow up one of their lands",
        "I need to punish greedy mana bases and knock out their utility lands",
    ),
    "tag-mass-land-denial": (
        "reset of everyone's mana base, nobody keeps their lands",
        "prison deck wants to wreck all mana sources so nobody can cast anything",
    ),
    "tag-sweeper-one-sided": (
        "one-sided wrath",
        "I'm behind on board but want to keep my own army while theirs disappears",
    ),
    "tag-hate-graveyard": (
        "yard hate",
        "sideboard plan against decks that recur stuff from the bin",
    ),
    "tag-sweeper-graveyard": (
        "nuke every discard pile at once",
        "total answer to dredge and reanimator, wipe all the yards clean",
    ),
    "tag-mass-reanimation": (
        "bring my whole dead army back in one go",
        "big finisher for a deck that dumps its own library: everything that died comes back at once",
    ),
    "tag-reanimate-from-opponent": (
        "steal their dead guys and make them fight for me",
        "black deck wants to raid enemy discard piles for their best threats",
    ),
    "tag-regrowth-instant": (
        "rebuy a used trick or counter from the bin",
        "my spells deck wants to reuse its best interaction a second time",
    ),
    "tag-regrowth-creature": (
        "a dead guy back to my grip",
        "attrition deck needs to recoup threats that got killed",
    ),
    "tag-castable-from-graveyard": (
        "spells with a second life after they hit the bin",
        "self-mill deck wants cards that are still useful once they're milled",
    ),
    "tag-mill-opponent": (
        "deck the opponent by dumping their library",
        "I want to win by running them out of cards to draw",
    ),
    "tag-wheel": (
        "everybody tosses their grip and gets a fresh seven-ish",
        "my hand empties fast, I need a full refill even if it helps the table too",
    ),
    "tag-loot": (
        "card filtering, pitch the bad one after getting a new one",
        "my reanimator deck needs to smooth draws and get big guys in the bin",
    ),
    "tag-rummage": (
        "pitch one first, then get a replacement",
        "aggressive red needs to turn excess lands into fresh gas",
    ),
    "tag-thoughtseize": (
        "peek at their grip and rip out the best thing",
        "proactive disruption to strip their key answer before my combo turn",
    ),
    "tag-repeatable-impulsive-draw": (
        "ongoing red card advantage off the top that I have to use right away",
        "red engine for advantage over a long game, every turn something new to use",
    ),
    "tag-draw-hate": (
        "punish people who refill too much",
        "stax deck wants to hurt opponents running lots of card advantage",
    ),
    "tag-tutor-card": (
        "go get literally anything",
        "my combo deck needs to find its missing piece reliably",
    ),
    "tag-tutor-creature": (
        "go fetch a specific guy",
        "toolbox deck wants to grab the right beater for each situation",
    ),
    "tag-wish": (
        "grab something from the sideboard",
        "a toolbox that reaches outside my main 60 for silver bullets",
    ),
    "tag-mana-rock": (
        "colorless trinket that speeds up my mana",
        "colorless acceleration for any commander deck",
    ),
    "tag-mana-dork": (
        "turn one elf that speeds up my mana",
        "green decks want early bodies that speed me into my expensive stuff",
    ),
    "tag-ritual": (
        "temporary burst of extra mana for one big turn",
        "storm deck needs fuel to chain lots of spells in one turn",
    ),
    "tag-extra-land": (
        "drop multiple lands in a single turn",
        "lands deck wants to get the most out of every turn's land drop",
    ),
    "tag-cost-reducer": (
        "discount effects for what I cast",
        "my big spells are too expensive, I need something that makes them affordable",
    ),
    "tag-repeatable-treasures": (
        "steady gold coin factory",
        "my deck wants a recurring stream of one-shot mana to sacrifice for value",
    ),
    "tag-landfall": (
        "rewards for dropping lands",
        "lands-matter deck wants payoffs for fetchlands and extra drops",
    ),
    "tag-gives-pp-counters-to-all": (
        "grow my whole team permanently",
        "counters deck wants a way to buff everyone at once and keep it",
    ),
    "tag-counter-doubler": (
        "make every counter I place count twice",
        "my +1/+1 and planeswalker deck wants to supercharge every counter",
    ),
    "tag-overrun": (
        "team-wide buff so everyone runs over the blockers for the win",
        "go-wide tokens deck needs a finisher to swing for lethal",
    ),
    "tag-clone": (
        "copycat that mimics the best thing on board",
        "I want flexible threats that become whatever the best body around is",
    ),
    "tag-copy-spell": (
        "twin a spell while it is still waiting to resolve",
        "spellslinger deck wants to double up on its best burn and draw",
    ),
    "tag-flicker": (
        "blink effects to reuse arrival triggers",
        "my value deck full of ETB guys wants to retrigger them repeatedly",
    ),
    "tag-lifegain-matters": (
        "cards that care about me going up in life",
        "my lifegain deck needs payoffs that turn small life bumps into a threat",
    ),
    "tag-soul-warden-ability": (
        "tiny life bumps from guys arriving",
        "a cheap early drop for a lifegain deck that triggers off everything",
    ),
    "tag-hate-lifegain": (
        "shut off healing for the other players",
        "stop the healing deck from running away from my burn",
    ),
    "tag-drain-life": (
        "siphon effect, hurt them and heal me",
        "black finisher that swings the life race in both directions",
    ),
    "tag-fog": (
        "fog effect to survive a big swing",
        "my green control deck needs to buy a turn against a huge alpha",
    ),
    "tag-protects-all": (
        "shield the whole team from a wrath or kill spell",
        "I've overextended and want insurance against the boardwipe",
    ),
    "tag-gives-indestructible": (
        "make my guys immune to destroy effects",
        "protect my key threats against wraths and kill spells",
    ),
    "tag-gives-unblockable": (
        "guarantee my attacker connects",
        "my voltron commander needs to get through a board of chump blockers",
    ),
    "tag-threaten": (
        "borrow their best guy for one swing",
        "aristocrats deck wants to take their best body for a turn and then sac it",
    ),
    "tag-theft-creature": (
        "keep their best guy for good",
        "blue deck wants to permanently take over their biggest threat",
    ),
    "tag-force-attacker": (
        "make their guys attack someone else",
        "in multiplayer, I want opponents' armies busy swinging at each other instead of me",
    ),
    "tag-tax-attack": (
        "tax on swinging at me",
        "pillowfort deck needs to make attacking me too expensive",
    ),
    "tag-rhystic": (
        "'pay up or I get value' effects",
        "I want cards that either slow opponents down or hand me advantage",
    ),
    "tag-extra-turn": (
        "go again right after my turn ends",
        "my blue deck wants to chain more turns in a row",
    ),
    "tag-extra-combat-phase": (
        "swing twice in one turn",
        "aggro deck with attack triggers wants another attack step",
    ),
    "tag-alternate-win-condition": (
        "win without dealing damage",
        "a weird finisher that ends the game on its own when a goal is met",
    ),
    "tag-polymorph": (
        "turn their thing into a random surprise from the deck",
        "chaotic answer: swap their bomb for whatever comes off the top",
    ),
    # --------------------------------------------------------- handwritten
    "hw-soft-counter": (
        "counter they can pay around",
        "early-game permission that loses value late, fine for a tempo list",
    ),
    "hw-another-dies-draw": (
        "refill my grip every time one of my guys bites it",
        "aristocrats deck needs card flow off its own guys going to the yard",
    ),
    "hw-flicker-own": (
        "instant blink of my own guy to dodge removal and reuse its arrival",
        "my ETB deck wants to protect a threat and re-trigger it at the same moment",
    ),
    "hw-flicker-end-step": (
        "remove something temporarily and get it back once the turn wraps up",
        "I want to dodge a sweeper or reset a blocker, with the thing coming back later that turn",
    ),
    "hw-dies-return-end-step": (
        "guys that pop back to play after they're killed, later the same turn",
        "resilient threats for a sacrifice deck that come back after I feed them",
    ),
    "hw-one-blocker-max": (
        "attackers immune to gang blocks, only single chump allowed",
        "my big beaters keep getting double-teamed, I want them to face one at a time",
    ),
    "hw-three-or-more-blockers": (
        "needs a whole squad of defenders just to stop it",
        "a huge threat that's nearly impossible to chump",
    ),
    "hw-each-opponent-loses": (
        "drain that hits the whole table at once",
        "my multiplayer commander deck wants life loss that scales with the number of players",
    ),
    "hw-target-player-loses": (
        "point a bit of life loss at one person",
        "focused reach to finish off a single player who's low",
    ),
    "hw-wrath-except": (
        "selective board wipe that leaves some survivors",
        "my typal deck needs to clear the table but keep my own tribe",
    ),
    "hw-cant-be-countered": (
        "uncounterable spells",
        "my threats keep getting stopped by blue players, I want spells permission can't touch",
    ),
    "hw-your-spells-uncounterable": (
        "protect all my spells from permission",
        "big-creature deck playing against blue wants to make its whole team immune to counters",
    ),
    "hw-double-pp-counters": (
        "instantly twice as big, like a hydra pump",
        "counters deck wants a way to explode a single creature's size",
    ),
    "hw-cant-gain-life": (
        "shut off all healing for opponents",
        "my burn plan needs to beat healing decks",
    ),
    "hw-cant-block-this-turn": (
        "stop one of their defenders from stepping in front this turn",
        "red aggro wants to push the final points of damage past a big wall",
    ),
    "hw-second-spell": (
        "bonus for double-spelling in a single turn",
        "my cheap-spell deck rewards me for casting two things in one turn",
    ),
    "hw-haste-sacrifice-end": (
        "temporary hasty attacker that goes away afterwards",
        "cheat deck wants one-shot hits from big bodies that leave afterwards",
    ),
    "hw-untap-all-lands": (
        "refresh my whole mana base mid-turn",
        "I want to cast two big spells in one turn by reusing my mana",
    ),
    "hw-bite": (
        "bite effect, my beast hits theirs one way",
        "green removal where my big guy punches without getting hit back",
    ),
    "hw-reanimate-own": (
        "dig up a dead monster of mine straight into play",
        "my reanimator deck needs cheap ways to cheat a big guy out of my yard",
    ),
    "hw-reanimate-any-gy": (
        "revive spell that can grab from anyone's bin",
        "I want to revive the best dead thing on the table, mine or theirs",
    ),
    "hw-opp-land-enters": (
        "tax people for making their land drop",
        "punish greedy ramp decks every time they hit their land drop",
    ),
    "hw-opponent-draws-trigger": (
        "hurt opponents whenever they pull a new card",
        "a stax piece that makes refilling painful for the table",
    ),
    "hw-play-lands-from-gy": (
        "reuse dead lands",
        "lands deck wants to replay fetchlands from the yard every turn",
    ),
    "hw-play-from-top": (
        "use my library like an extra hand",
        "card advantage engine that lets me use what's on top without drawing",
    ),
    "hw-basic-to-bf-tapped": (
        "cheap ramp that grabs a basic from the deck",
        "green deck needs early acceleration that fixes colors too",
    ),
    "hw-3-damage-any-target": (
        "cheap red burn for three",
        "red aggro wants efficient reach that can kill a guy or go face",
    ),
    "hw-each-player-sac-creature": (
        "symmetrical edict, everyone loses one guy",
        "my token deck can afford to give up a chump while their one big threat goes",
    ),
    "hw-reveal-hand-choose-nonland": (
        "hand attack where I pick what they lose",
        "disruption that lets me see their plan and rip the key piece",
    ),
    "hw-fog-all-combat": (
        "holding off a whole attack for a turn",
        "my control deck needs to survive one lethal swing",
    ),
    "hw-opp-enter-tapped": (
        "make their new stuff show up exhausted",
        "slow down hasty aggro and fresh mana in a stax deck",
    ),
    "hw-threaten": (
        "borrow their guy for a turn",
        "a way to swing with their biggest guy for one turn",
    ),
    "hw-play-until-next-turn": (
        "red pseudo draw that gives a turn and a half to use it",
        "card advantage for red aggro that doesn't waste cards when I'm tapped out",
    ),
    "hw-lifegain-pp-counter": (
        "guys that grow when I heal",
        "lifegain deck needs a way to turn health into a bigger board",
    ),
    "hw-die-exile-instead": (
        "nothing is allowed to stay in the yard",
        "shut off all recursion and flashback for the whole game",
    ),
    "hw-regrow-instant-sorcery": (
        "rebuy an old spell from the bin",
        "spellslinger deck wants to recast its best burn or counter",
    ),
    "hw-tax-spells-cost-more": (
        "tax on what they cast",
        "a hatebear to slow down storm and spell-heavy decks",
    ),
    "hw-opp-creature-dies": (
        "rewards when their guys go down",
        "I want value whenever my removal or combat kills enemy creatures",
    ),
    "hw-opp-upkeep": (
        "something that triggers on the other players' turns",
        "a passive engine that keeps working while it's not my turn",
    ),
    "hw-each-opponent-mills": (
        "dump cards off the top of every enemy library",
        "plan to empty everyone's library in multiplayer",
    ),
    "hw-land-from-hand": (
        "cheat a land into play for free",
        "lands deck wants to dump extra lands from my grip",
    ),
    "hw-donate": (
        "hand them a gift they don't want",
        "a way to push my drawback permanent or bomb onto someone else",
    ),
    "hw-oblivion-ring": (
        "removal that undoes itself if my enchantment goes away",
        "white flexible answer for any problem permanent",
    ),
    "hw-earthquake": (
        "small sweeper that also burns faces",
        "red finisher that clears the ground and ends the game",
    ),
    "hw-mass-bounce": (
        "send everything back to hand at once",
        "a big tempo reset for when I'm far behind",
    ),
    "hw-copy-instant-sorcery": (
        "twin a spell, maybe redirect it",
        "my izzet spells deck wants to double its best burn or draw",
    ),
}


def main() -> None:
    out = []
    originals = load_queries("tags") + load_queries("handwritten")
    missing = [q["id"] for q in originals if q["id"] not in REFORMULATIONS]
    assert not missing, f"no reformulation for: {missing}"
    for q in originals:
        para, need = REFORMULATIONS[q["id"]]
        for kind, text in (("para", para), ("need", need)):
            e = {"id": f"{q['id']}-{kind}", "origin": q["id"], "query": text, "relevant": q["relevant"]}
            if q.get("graded"):
                e["graded"] = q["graded"]
            out.append(e)
    OUT.write_text(
        "# Generated by eval/build_paraphrase.py -- edit REFORMULATIONS there.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120)
    )
    print(f"wrote {len(out)} queries to {OUT.name}")


if __name__ == "__main__":
    main()
