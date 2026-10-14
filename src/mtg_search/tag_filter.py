"""Which Scryfall oracle tags describe what a card *does* in a deck (vs. trivia), and which of a
card's functional tags are the most informative to show.

Scryfall's tagger mixes gameplay roles ("removal-exile", "sacrifice-outlet-creature",
"counterspell-soft") with naming jokes ("portmanteau"), lore ("dnd-character"), print/errata
history ("type-errata-hound"), rules-template observations ("intervening-if-clause",
"scales-with-mana-value") and colour-pie labels ("synergy-blue", "blue-effect"). A tag is
*non-functional* when any of these holds (see docs/tag_filter.md):

1. it is in :data:`DENY_SUBTREES` or descends from one, i.e. every parent of it is itself
   subtree-denied (so "ramp-with-set-s-mechanic", which also has parent "ramp", survives
   rule 1 but is caught by the regex in rule 3);
2. it is in :data:`DENY_SELF` (a template/meta tag whose children may still be functional,
   e.g. "triggered-ability" -> "death-trigger-self");
3. it matches :data:`DENY_PATTERN`;

unless it is in :data:`ALLOW` (overrides all three).

Everything else is functional. Ranking (:meth:`TagFilter.roles`) puts tags from the core deck
roles (:data:`CORE_ROLES` families: removal, ramp, draw, tutor, counterspell, ...) first, and
within each group prefers mid-specificity tags (card counts near :data:`SWEET_SPOT`) over very
generic or very niche ones. Fully deterministic; no data file besides tags.parquet.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

import polars as pl

TAG_MIN, TAG_MAX = 10, 2500  # tags on fewer / more cards are too niche / too generic to show
SWEET_SPOT = 150  # card count of the "ideal" role tag

# Trivia roots: the tag and every tag whose parents are all denied.
DENY_SUBTREES = frozenset({
    "cycle", "card-names", "un-design", "un-set-mechanics", "dnd", "draft-signpost",
    "type-errata", "keyword-errata", "deprecated-mechanics", "deprecated-card-types",
    "staple-with-set-s-mechanic", "flavors-of-vanilla", "aesthetic-counter",
    "planeswalker-deck-staples", "characteristic-defining-ability", "indefinite-effect",
    "unique-type-line", "unique-mana-cost", "doctor-who-episode-name",
})

# Template / meta / colour-pie tags denied themselves only (children are judged on their own).
DENY_SELF = frozenset({
    # rules templating
    "triggered-ability", "activated-ability", "intervening-if-clause", "reflexive-trigger",
    "delayed-trigger", "delayed-replacement-effect", "self-replacement-effect", "state-trigger",
    "trigger-from-exile", "trigger-from-graveyard", "cast-on-resolution", "exile-on-resolution",
    "special-action", "faux-targeting", "division", "named-choice", "multiple-targets",
    "additional-cost", "alternative-cost", "cheaper-than-mv", "more-expensive-than-mv",
    "non-mana-ability-mana", "mana-ability-with-extra-effect", "passive-ability",
    "noted-tracked-information", "x-doesn-t-matter", "exile-self", "manaless-value",
    "spell-with-no-casting-cost", "sneaky-self-trigger", "that-s-how-the-game-works",
    "digital-only-mechanics", "drawback", "repeated-keyword", "keyword-soup", "sliver-stackable",
    "quadratic", "exponential", "transform-improvement", "color-break", "mix-and-match",
    "vanilla", "vanilla-aura", "vanilla-equipment", "french-vanilla-aura",
    "french-vanilla-equipment", "four-loyalty-abilities", "four-plus-creature-types",
    "mixed-subtypes", "multiple-species-types", "class-type-only", "breaks-ktk-morph-rule",
    "distinct-echo-cost", "counts-as-a-type", "noncreature-typal", "cr-107-3f-x-card",
    "offcolor-ability", "offcolor-additional-cost", "offcolor-mana-generation",
    "out-of-color-token", "has-identical-token", "unprinted-token", "token-errata",
    "multicolor-kicker", "multiple-kicker-costs", "single-target-instant-sorcery",
    "hate-set-mechanic", "landhome", "synergy-art-sticker",
    # colour-pie labels ("this is a blue effect") -- the children (counterspell, ...) say more
    "white-effect", "blue-effect", "black-effect", "red-effect", "green-effect",
    "synergy-white", "synergy-blue", "synergy-black", "synergy-red", "synergy-green",
    # print history, products, design trivia, flavour, community meta
    "meme", "fun-ruling", "rules-nightmare", "notorious-templating", "old-banish-templating",
    "day-zero-errata", "fulfilled-futureshift", "mechanical-foreshadow", "references-keyword",
    "invitational-card", "guest-designer", "great-designer-search-3", "playtest-forecast",
    "un-forecast", "digital-to-paper", "paper-compatible", "commander-set-booster-cards",
    "legends-retold", "potentially-black-border", "vanity-card", "40k-model", "the-doctor",
    "multi-character-card", "legendary-team-up", "mirrored-knight", "wordle", "alt-commander",
    "face-commander", "planeswalker-deck-face-card", "specialized", "team-matters",
    "draft-matters", "set-matters", "legacy", "creature-type-guardian", "creature-type-ship",
    "creature-type-townsfolk", "mana-source-type", "change-name", "tutored-by-name",
})

DENY_PATTERN = re.compile(
    r"^(?:"
    r"cycle-.*|.*-cycle|supercycle-.*|"  # cycles
    r".*-name|.*-names?-.*|"  # naming trivia (punny-name, shares-name-with-a-set, ...)
    r"namesake-.*|eponymous.*|"
    r"art-.*|artist-.*|flavor.*|reprint.*|.*-reference|.*-storyline-in-cards|"
    r"virtual-.*|untracked-.*|unnoted-.*|unique-.*|"  # "unique-p-t", "unique-keyword", ...
    r"type-errata.*|type-addition-.*|.*-errata(?:-.*)?|"
    r"deprecated-.*|old-.*|digital-only-.*|dnd-.*|doctor-who-.*|"
    r"scales-with-.*|cda-.*|"
    r".*-with-set-s?-?mechanics?|.*-set-mechanic-.*|"
    r"un-.*|nonbo.*|aesthetic-.*|pwdeck-.*"
    r")$"
)

# Functional tags that a pattern above would otherwise catch.
ALLOW = frozenset({"tutors-by-name", "hate-named", "changeling"})

# Families of tags that name a core deck role; members (descendants or "<root>-..." prefixes)
# are listed before other functional tags.
CORE_ROLES = (
    "removal", "sweeper", "spot-removal", "burn", "counterspell", "hand-disruption", "discard",
    "ramp", "mana-producer", "mana-dork", "mana-rock", "land-ramp", "ritual", "cost-reducer",
    "cost-ignorer", "card-advantage", "draw", "draw-engine", "repeatable-card-advantage",
    "cantrip", "loot", "rummage", "impulse", "impulsive-draw", "tutor", "wish", "recursion",
    "reanimate", "regrowth", "sacrifice-outlet", "repeatable-sacrifice-outlet",
    "repeatable-token-generator", "token-doubler", "counter-doubler", "protection",
    "protects-creature", "copy", "theft", "control-changing-effects", "extra-turn",
    "extra-combat-phase", "wheel", "flicker", "bounce", "tapper", "lockdown", "pacifism",
    "mill", "lifegain", "drain-life", "blood-artist-ability", "group-slug", "pinger",
    "anthem", "overrun", "evasion", "gives-evasion", "fog", "alternate-win-condition",
    "mass-land-denial", "rhystic", "tax", "pillowfort", "untapper", "graveyard-fuel",
    "hate-graveyard",
)
# ... except these sub-families, which are secondary/how-it-works details ("tutor-land-basic" is
# how a ramp spell ramps; "hate-regenerate" is a footnote on a sweeper).
CORE_EXCLUDE = ("tutor-land", "hate-")


def deny_reason(slug: str, parents: Mapping[str, Iterable[str]] | None = None) -> str | None:
    """Why ``slug`` is non-functional ("subtree:<root>", "self", "pattern"), or None."""
    return _Classifier(parents or {}).reason(slug)


class _Classifier:
    def __init__(self, parents: Mapping[str, Iterable[str]]):
        self.parents = {k: list(v or []) for k, v in parents.items()}
        self._sub: dict[str, str | None] = {}

    def subtree_root(self, slug: str, _stack: frozenset[str] = frozenset()) -> str | None:
        if slug in DENY_SUBTREES:
            return slug
        if slug in self._sub:
            return self._sub[slug]
        ps = [p for p in self.parents.get(slug, []) if p not in _stack]
        root = None
        if ps:
            roots = [self.subtree_root(p, _stack | {slug}) for p in ps]
            if all(roots):
                root = roots[0]
        self._sub[slug] = root
        return root

    def reason(self, slug: str) -> str | None:
        if slug in ALLOW:
            return None
        root = self.subtree_root(slug)
        if root:
            return f"subtree:{root}"
        if slug in DENY_SELF:
            return "self"
        if DENY_PATTERN.match(slug):
            return "pattern"
        return None

    def ancestors(self, slug: str) -> set[str]:
        out: set[str] = set()
        todo = list(self.parents.get(slug, []))
        while todo:
            p = todo.pop()
            if p not in out:
                out.add(p)
                todo += self.parents.get(p, [])
        return out


class TagFilter:
    """Functional-tag classification + role ranking for one tags table."""

    def __init__(self, counts: Mapping[str, int], parents: Mapping[str, Iterable[str]] | None = None):
        self.counts = dict(counts)
        cl = _Classifier(parents or {})
        slugs = set(self.counts) | set(cl.parents)
        self.reasons: dict[str, str] = {s: r for s in slugs if (r := cl.reason(s))}
        self.parents = cl.parents
        core = set(CORE_ROLES)
        self.core = {
            s for s in slugs
            if (s in core or any(s.startswith(c + "-") for c in core) or cl.ancestors(s) & core)
            and (s in core or not s.startswith(CORE_EXCLUDE))
        }

    @classmethod
    def from_frame(cls, tags: pl.DataFrame | None) -> TagFilter:
        if tags is None or tags.height == 0:
            return cls({})
        counts = dict(zip(tags["slug"].to_list(), tags["card_count"].to_list(), strict=True))
        parents = None
        if "parent_slugs" in tags.columns:
            parents = dict(zip(tags["slug"].to_list(), tags["parent_slugs"].to_list(), strict=True))
        return cls(counts, parents)

    def is_functional(self, slug: str) -> bool:
        """Gameplay tag (any card count); unknown slugs are judged by the regex/lists only."""
        if slug in self.reasons:
            return False
        if slug in self.counts:
            return True
        return deny_reason(slug) is None

    def is_core(self, slug: str) -> bool:
        if slug in self.core or slug in CORE_ROLES:
            return True
        return not slug.startswith(CORE_EXCLUDE) and any(slug.startswith(c + "-") for c in CORE_ROLES)

    def score(self, slug: str) -> tuple[int, float, str]:
        c = max(self.counts.get(slug, 0), 1)
        return (0 if self.is_core(slug) else 1, round(abs(math.log(c / SWEET_SPOT)), 6), slug)

    def roles(self, tags: Iterable[str] | None, n: int = 5) -> list[str]:
        """Up to ``n`` functional tags of a card with TAG_MIN <= card_count <= TAG_MAX: core deck
        roles first, then mid-specificity first; a tag whose parent set equals that of an
        higher-ranked tag (a sibling: tutor-land-plains / -swamp / ...) goes to the back of its tier."""
        cand = [
            t for t in dict.fromkeys(tags or [])
            if TAG_MIN <= self.counts.get(t, 0) <= TAG_MAX and self.is_functional(t)
        ]
        cand.sort(key=self.score)
        seen: set[tuple[str, ...]] = set()
        sibling: dict[str, bool] = {}
        for t in cand:
            key = tuple(sorted(self.parents.get(t, [])))
            sibling[t] = bool(key) and key in seen
            seen.add(key)
        cand.sort(key=lambda t: (self.score(t)[0], sibling[t]))  # stable: keeps score order
        return cand[:n]
