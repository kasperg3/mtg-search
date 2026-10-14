# Oracle tag filter (`roles:` line)

`src/mtg_search/tag_filter.py` decides which Scryfall oracle tags are *functional* (what a card
does / what it is good for in a deck) and orders a card's functional tags for the `roles:` line
of the MCP renderer. `list_tags` still returns every tag, but lists functional tags first and
marks the rest `[trivia]`.

## Classification (deterministic, data-driven from `tags.parquet`)

A tag is non-functional if any of these holds (unless it is in the small `ALLOW` list, e.g.
`changeling`, `tutors-by-name`):

1. **Hierarchy deny (`DENY_SUBTREES`)**: the tag is a trivia root (`cycle`, `card-names`,
   `un-design`/`un-set-mechanics`, `dnd`, `draft-signpost`, `type-errata`, `keyword-errata`,
   `deprecated-mechanics`, `deprecated-card-types`, `staple-with-set-s-mechanic`,
   `flavors-of-vanilla`, `aesthetic-counter`, `characteristic-defining-ability`, ...) or *all* of
   its parents are subtree-denied. A tag with one functional parent (e.g.
   `ramp-with-set-s-mechanic` under `ramp`) is not denied by this rule.
2. **Self deny (`DENY_SELF`)**: rules-template, meta and colour-pie tags whose children can still
   be functional: `triggered-ability` (child `death-trigger-self` is kept), `intervening-if-clause`,
   `alternative-cost`, `blue-effect` (child `counterspell` is kept), `synergy-<colour>`, `meme`,
   `fun-ruling`, `vanity-card`, `drawback`, `quadratic`, ...
3. **Pattern deny (`DENY_PATTERN`)**: `cycle-*`, `*-name`, `namesake-*`, `art-*`, `flavor*`,
   `*-storyline-in-cards`, `unique-*`, `virtual-*`, `*-errata*`, `type-addition-*`, `scales-with-*`,
   `cda-*`, `*-with-set-s-mechanic`, `un-*`, ...

Result on the current `tags.parquet` (4,577 tags): **2,015 denied / 2,562 kept**
(1,541 by hierarchy, 115 self, 359 pattern). Among the 1,721 tags in the displayable range
(10–2,500 cards): **463 denied / 1,258 kept**.

## Ranking

Of a card's functional tags with 10–2,500 cards:

1. tags in a core deck-role family (`CORE_ROLES`: removal, sweeper, burn, counterspell, discard,
   ramp/mana-rock/mana-dork, draw/card-advantage, tutor, recursion/reanimate, sacrifice-outlet,
   token generators, protection, theft, wheel, flicker, ... — members are descendants in the tag
   hierarchy or `<root>-*` slugs; `tutor-land*` and `hate-*` are excluded as secondary detail)
   come first;
2. within a tier, mid-specificity first: closest card count to 150 on a log scale (so
   `removal-exile` (444) beats `removal-creature` (1,898), `land-ramp` beats `tutor-land-basic`);
3. a tag with the same parent set as a higher-ranked tag (a sibling, e.g. `tutor-land-plains` /
   `tutor-land-swamp`) moves to the back of its tier.

Compact results show 5, `get_card` full view 12, the `oracle tags:` line 30.

## Before / after (compact `roles:` line, 5 tags)

Before = previous regex deny-list, most-used-first. After = this filter.

| Card | Before | After |
|---|---|---|
| Sol Ring | adds-multiple-mana, full-refund, mana-rock | mana-rock, full-refund, adds-multiple-mana |
| Counterspell | counterspell, interrupt | counterspell |
| Swords to Plowshares | removal-creature, lifegain, removal-exile, opponent-lifegain | removal-exile, lifegain, removal-creature, opponent-lifegain |
| Rhystic Study | draw-engine, repeatable-pure-draw, rhystic, cast-trigger-other, cast-tax | rhystic, repeatable-pure-draw, draw-engine, cast-tax, cast-trigger-other |
| Viscera Seer | sacrifice-outlet-creature, martyr, free-sacrifice-outlet, repeatable-scry | free-sacrifice-outlet, sacrifice-outlet-creature, repeatable-scry, martyr |
| Cultivate | tutor-to-hand, land-ramp, tutor-land-basic, tutor-land-to-battlefield | land-ramp, tutor-to-hand, tutor-land-to-battlefield, tutor-land-basic |
| Llanowar Elves | mana-dork | mana-dork |
| Doubling Season | synergy-planeswalker, synergy-token, counter-doubler, token-doubler | counter-doubler, token-doubler, synergy-planeswalker, synergy-token |
| Wrath of God | removal-creature, removal-destroy, symmetrical, sweeper, hate-regenerate | sweeper, removal-destroy, removal-creature, hate-regenerate, symmetrical |
| Demonic Tutor | tutor-to-hand, tutor-card | tutor-card, tutor-to-hand |
| Lightning Bolt | burn-any | burn-any |
| Fire // Ice | multiple-targets, burn-any, cantrip, multi-removal, tapper-permanent | tapper-permanent, multi-removal, cantrip, burn-any |
| Delver of Secrets // Insectile Aberration | synergy-instant, synergy-sorcery, top-matters, peek-library | top-matters, peek-library, synergy-sorcery, synergy-instant |
| Atraxa, Praetors' Voice | repeatable-lifegain, repeatable-proliferate | repeatable-lifegain, repeatable-proliferate |
| Craterhoof Behemoth | toughness-boost-to-all, gains-trample, creature-count-matters, overrun, quadratic | overrun, creature-count-matters, gains-trample, toughness-boost-to-all |
| Arcane Signet | mini-refund, synergy-commander, commander-identity-matters | synergy-commander, mini-refund, commander-identity-matters |
| Command Tower | synergy-commander, rainbow-land, synergy-color-share, commander-identity-matters | synergy-commander, rainbow-land, synergy-color-share, commander-identity-matters |
| Cyclonic Rift | alternative-cost, removal-nonland, removal-bounce, sweeper-one-sided, pseudo-fog | sweeper-one-sided, removal-bounce, removal-nonland, pseudo-fog |
| Path to Exile | removal-creature, land-ramp, removal-exile, tutor-land-basic, tutor-land-to-battlefield | removal-exile, land-ramp, removal-creature, swap-removal, tutor-land-to-battlefield |
| Brainstorm | cantrip, portmanteau, brainstorm | cantrip, brainstorm |
| Ponder | cantrip, soothsaying | cantrip, soothsaying |
| Dark Ritual | adds-multiple-mana, ritual, interrupt | ritual, adds-multiple-mana |
| Thoughtseize | discard, life-payment, thoughtseize | thoughtseize, discard, life-payment |
| Birds of Paradise | mana-dork | mana-dork |
| Smothering Tithe | ramp, rhystic, repeatable-treasures, draw-hate | repeatable-treasures, rhystic, draw-hate, ramp |
| Esper Sentinel | draw-engine, repeatable-pure-draw, scales-with-power, power-matters-self, rhystic | rhystic, repeatable-pure-draw, draw-engine, cast-tax, cast-trigger-other |
| Mana Drain | adds-multiple-mana, scales-with-mana-value, aikido, counterspell, hate-high-mv | counterspell, ritual, aikido, hate-high-mv, adds-multiple-mana |
| Mystic Remora | draw-engine, repeatable-pure-draw, rhystic, upkeep-cost, cast-trigger-other | rhystic, repeatable-pure-draw, draw-engine, cast-tax, cast-trigger-other |
| Ashnod's Altar | sacrifice-outlet-creature, adds-multiple-mana, refund, free-sacrifice-outlet, mana-rock | free-sacrifice-outlet, mana-rock, sacrifice-outlet-creature, refund, adds-multiple-mana |
| Blood Artist | opponent-loses-life, blood-artist-ability, inscryption-achievement | blood-artist-ability, opponent-loses-life |
| Zulaport Cutthroat | opponent-loses-life, group-slug, death-trigger-self, blood-artist-ability | blood-artist-ability, group-slug, death-trigger-self, opponent-loses-life |
| Eternal Witness | regrowth-any | regrowth-any |
| Reanimate | reanimate-creature, scales-with-mana-value, life-payment, reanimate-from-any | reanimate-from-any, reanimate-creature, life-payment |
| Animate Dead | reanimate-creature, reanimate-from-any, notorious-templating, rules-nightmare | reanimate-from-any, reanimate-creature |
| Skullclamp | draw-engine, repeatable-pure-draw, death-trigger, flowstone | repeatable-pure-draw, draw-engine, flowstone, death-trigger |
| Beast Within | removal-destroy, removal-permanent, donate-token, swap-removal | removal-permanent, removal-destroy, swap-removal, donate-token |
| Chaos Warp | removal-permanent, removal-tuck, swap-removal, cost-ignorer, polymorph | removal-tuck, cost-ignorer, removal-permanent, swap-removal, polymorph |
| Kodama's Reach | tutor-to-hand, land-ramp, tutor-land-basic, tutor-land-to-battlefield | land-ramp, tutor-to-hand, tutor-land-to-battlefield, tutor-land-basic |
| Lightning Greaves | protects-creature, gives-haste, french-vanilla-equipment, gives-shroud | gives-shroud, protects-creature, gives-haste |
| Swiftfoot Boots | protects-creature, gives-haste, gives-hexproof, french-vanilla-equipment | gives-hexproof, protects-creature, gives-haste |
| Dockside Extortionist | ramp, hate-artifact, full-refund, hate-enchantment | ramp, hate-artifact, full-refund, hate-enchantment |
| Tarmogoyf | lhurgoyf, card-types-in-graveyard-matter, fulfilled-futureshift, mechanical-foreshadow | lhurgoyf, card-types-in-graveyard-matter |
| Snapcaster Mage | synergy-instant, synergy-sorcery, gives-flashback, invitational-card | gives-flashback, synergy-sorcery, synergy-instant |
| Grave Pact | removal-creature, repeatable-removal, multi-removal, death-trigger, removal-sacrifice | removal-sacrifice, multi-removal, repeatable-removal, removal-creature, death-trigger |
| Avenger of Zendikar | repeatable-pp-counters, gives-pp-counters, landfall, land-count-matters | land-count-matters, landfall, enters-in-company, gives-pp-counters, repeatable-pp-counters |
| Farseek | land-ramp, tutor-land-to-battlefield, tutor-land-plains, tutor-land-mountain, tutor-land-swamp | land-ramp, tutor-land-to-battlefield, tutor-land-plains, tutor-land-mountain, tutor-land-swamp |
| Toxic Deluge | additional-cost, symmetrical, sweeper, removal-toughness, life-payment | removal-toughness, sweeper, life-payment, symmetrical |
| Necropotence | repeatable-card-advantage, life-for-cards, deal-with-the-devil, skip-draw-step | life-for-cards, repeatable-card-advantage, deal-with-the-devil, skip-draw-step |
| Rampant Growth | land-ramp, tutor-land-basic, tutor-land-to-battlefield | land-ramp, tutor-land-to-battlefield, tutor-land-basic |
| Goblin Bombardment | repeatable-removal, burn-any, sacrifice-outlet-creature, pinger, free-sacrifice-outlet | free-sacrifice-outlet, bombard, pinger, sacrifice-outlet-creature, burn-any |
