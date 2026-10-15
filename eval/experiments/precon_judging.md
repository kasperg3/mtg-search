# EXP-017: Deck-aware judging of the precon add picks

**Status: done 2026-10-15.** Pre-registered in `precon_judging_plan.md` (commit `ad23a0c`) before any judge call. The calibration record and the owner's two decisions are in the same file, each committed before the run it governs (`34da386`, `63ca2a6`, `a9e9723`).

**Answer.**
- **Judged on deck fit, the shipped surface's ADD picks are clearly better than v1's.** v1 is the same agent without tool use. Judged fit@20 is 0.651 vs 0.447: **shipped − v1 = +0.204 [+0.117, +0.298], sign p = 0.002**, better on 13 of 14 test precons. Both pre-registered criteria are met, so the result is **established**. On the same picks, the EDHREC key gives +0.087 [+0.015, +0.168], sign p = 0.057, which is not established (EXP-011).
- **It is not only validity.** Counting valid picks only, fit is 0.669 vs 0.519: +0.151 [+0.066, +0.246], sign p = 0.013 (exploratory). The judge rejects 13% of shipped's valid picks and 27% of v1's.
- **SWAP adds:** no established difference. Fit@10 is +0.075 [−0.007, +0.165], sign p = 1.0 (6 better, 7 worse, 1 tie).
- **The judge does not favour popular cards.**
  - Shipped's picks are less popular than v1's (mean log10 EDHREC rank 3.01 vs 2.48), yet judged better.
  - On test, the judge's 2s are, if anything, the less popular pool cards: Spearman ρ(log10 rank, grade) = +0.14.
  - The community key rewards popularity, and so understated the tool-using agent.
- **On dev, the slim server (EXP-014) and the old server tie on judged fit.** All three rubrics give shipped-now − old ≈ 0 on fit@20 (+0.025, +0.005, −0.002). The community key gave −0.066 on the same decks. So the slim server's less popular picks are not worse for the deck. This is descriptive: dev, one repeat, and the rubric was tuned on these decks.
- **Getting a valid judge took three rubrics** (see Calibration):
  - v1 and v2 accepted too many random cards; v3 missed the dev gate by one card.
  - On test, v3 passed its own validity rule: 10 of 84 random controls (11.9%) and 0 of 42 in-deck cards were graded ≥ 1.
- **Cost.** 338 judge calls in all (probe, 3 dev runs, test), 1.36M tokens, about 6 minutes of judging with 4 workers, on the subscription. The test run alone: 98 calls, 411k tokens, 1 turn and 4.0 s per call, 98 s.

## Question

Judged on fit for the deck rather than on EDHREC add rates, are the shipped surface's ADD picks better than v1's (no tool use)? How far do the judge and the EDHREC labels agree?

## Setup

As pre-registered (`precon_judging_plan.md`, including the addendum):
- **Judge.** Headless Claude Code, `claude-sonnet-5-5`, on the subscription. No tools, an empty directory, and the init isolation check (`bench/precon/judge.py`).
- **What the judge sees.** The format rules, the deck and 15 candidate cards per call, blind and in a seeded random order. The deck is the commander's full text with its name replaced by "your commander", plus the other 99 cards as names.
- **What it never sees:** which run proposed a card, any EDHREC data, prices, or the product or deck name. Every prompt passed the leakage pre-flight.
- **Pool.** The statically valid add-side picks of EXP-011 test `shipped` and `v1` (14 precons × 3 repeats, ADD top 20 + SWAP adds): **1,051 cards**.
- **Controls per precon,** mixed blind into the batches (210 in all): 6 random legal in-identity cards with no add label, the 6 highest-rate EDHREC adds no run picked, and 3 non-basic cards already in the deck.
- **Passes.** One pass, plus a seeded 10% of batches judged again in a new order (128 cards), for consistency.
- **Metric.** fit@k is the mean of grade/2 over k slots; invalid picks (scorer rules, Game Changer limit in rank order) and missing slots count 0. Per precon, repeats are averaged; shipped − v1 is paired over the 14 precons, with a 10k bootstrap CI and a sign test.

## Calibration on dev (before any test judgment)

The plan allowed one rubric revision on dev, keeping the validity rule fixed: random and in-deck controls graded ≥ 1 at most 15% each. Pool: the EXP-014 dev runs (old surface `before` vs shipped-now `after3`, 15 precons × 1 repeat), 675 cards + 225 controls.

| rubric | random ≥ 1 | random = 2 | in-deck ≥ 1 | top EDHREC adds ≥ 1 | pool ≥ 1 / = 2 | re-judge κ(≥ 1) | gate |
|---|---|---|---|---|---|---|---|
| v1 "fits this deck" (pre-registered) | 42/90 (46.7%) | 4.4% | 0/45 | 94.4% | 95.0% / 63.4% | 0.71 | fail |
| v2 "improves the deck" (the allowed revision) | 20/90 (22.2%) | 0% | 0/45 | 85.6% | 90.5% / 42.5% | 0.81 | fail |
| v3 "would you add it as an upgrade" (owner-approved) | 14/90 (15.6%) | 0% | 1/45 | 77.8% | 86.1% / 38.1% | 0.79 | fail by one card |

- **v1** graded filler 1 as "does something useful": Eager Construct, Clockwork Servant, Nim Abomination.
- **v2** dropped most of that filler.
- **v3** requires a specific reason for this deck for any credit, and has the judge state the deck's plan first. Its remaining accepted random cards are mostly lands that fit the mana base (Coastal Tower, Captivating Cave, Hive of the Eye Tyrant) and strong cards (Casualties of War).
- The owner chose to try v3, and then to judge the test pool with it. The test run had to pass the same rule on its own fresh controls, plus a guard that the top EDHREC adds stay "mostly fit" (> 50% graded ≥ 1).

**Recognition probe (dev, 15 decks).** Given the same anonymised deck text, the judge named the right deck for 12, the product for 11 and the commander for 10. It said it was confident on 4. So the judge knows most of these products, mostly from the new cards in each list. Its grades could draw on what it knows of them, though the popularity check below does not point that way.

## Results (test, rubric v3, `20261015_exp017-test`)

**Validity.**
- Random controls: 10/84 (11.9%) graded ≥ 1, 1/84 graded 2.
- In-deck controls: 0/42.
- Top EDHREC adds not picked: 86.9% ≥ 1, 52.4% = 2.
- The run is valid and the guard holds.

**ADD** (14 precons, repeats averaged):

| | shipped | v1 | shipped − v1 [95% CI] | sign p |
|---|---|---|---|---|
| **judged fit@20 (primary)** | **0.651** | **0.447** | **+0.204 [+0.117, +0.298]** | **0.002** |
| judged fit@10 | 0.785 | 0.506 | +0.279 [+0.183, +0.375] | 0.002 |
| share of the 20 slots graded 2 | 0.458 | 0.262 | +0.196 [+0.093, +0.315] | 0.002 |
| EDHREC nDCG@10 (EXP-011 key) | 0.262 | 0.174 | +0.087 [+0.015, +0.168] | 0.057 |
| validity | 0.970 | 0.868 | | |

**SWAP adds:**

| | shipped | v1 | shipped − v1 [95% CI] | sign p |
|---|---|---|---|---|
| judged fit@10 | 0.699 | 0.624 | +0.075 [−0.007, +0.165] | 1.0 |
| share of the 10 slots graded 2 | 0.510 | 0.390 | +0.119 [−0.010, +0.252] | 0.79 |
| EDHREC nDCG@10 | 0.217 | 0.219 | −0.002 [−0.085, +0.085] | 0.42 |

**Agreement with the EDHREC labels** (pool cards):
- **Labelled** (an add label with rate ≥ 0.05; n = 173): 91.9% judged ≥ 1, 60.1% judged 2.
- **Unlabelled** (n = 878): 74.3% judged ≥ 1, 29.6% judged 2.
- **κ** (label vs judged ≥ 1) is 0.07. The two sources agree on which cards are best (twice the rate of 2s), but most unlabelled picks are still judged useful for the deck.

**Consistency** (128 re-judged cards): same grade 78.9%, same ≥ 1 92.2%, κ(≥ 1) 0.80.

**Exploratory (not pre-registered).**
- **Valid picks only.**
  - Fit: shipped 0.669 vs v1 0.519, +0.151 [+0.066, +0.246], sign p = 0.013.
  - Grade mix 0/1/2: shipped 13% / 40% / 47%, v1 27% / 43% / 30%. v1 recalls more cards that read worse once the judge sees the actual text.
- **Within shipped ADD, by tool use.**
  - Episodes that searched (17/42): fit on valid picks 0.755.
  - Episodes that did not (25/42): 0.611, against v1's 0.519.
  - Episodes chose whether to search, so this is not a causal split.
- **Popularity.** Spearman ρ(log10 EDHREC rank, grade) on test pool cards is +0.14: median log10 rank 3.18 for grades 0 and 1, and 3.62 for grade 2. So the judge's 2s are less popular cards, not more. On dev the same ρ ranged from −0.11 to +0.15 across rubrics.

## Limits

- **The rubric was tuned on dev three times**, and v3 missed the dev gate by one card; the owner approved judging test with it. The test pool was judged once, with fresh controls, and passed independently. It is still selection on dev.
- **Sonnet judges Sonnet.** Both arms are the same model, so self-preference cancels between them. But the shipped agent read the same card text the judge reads, while v1 worked from memory. A text-reading judge is best placed to catch misremembered cards, which is part of the effect being measured.
- **The judge knows most of these products** (dev probe: 12/15 decks named). Its grades do not track popularity, but it may draw on other knowledge of these decks.
- **The surface here is EXP-011's** (4 tools, before the EXP-014 cleanup). The current slim server was compared with the old one only on dev, where the two tie on judged fit.
- **Small and reused test set.** 14 precons, and this is the 24th analysis on the test split (it re-labels existing outputs and adds no arm). One pool card shares its name with its own precon ("Avengers Assemble!"), a disclosed pre-flight exemption.
- **One pass.** Single grades, used as is. Consistency (κ 0.80) bounds the noise, not the validity of the judgments.

## Reading

The question EXP-011 left open, whether the tool-using surface picks *better* cards, has a clear answer on deck fit: yes for ADD, by a fifth of the scale, on 13 of 14 decks. The EDHREC key could not show this: the tool-using agent picks cards that fit but are less popular, and that key rewards popularity. For SWAP the judged gain is smaller and not established.

On dev, the judge also backs the owner's decision to keep the slim server: its lower community-key score reflects less popular picks, not worse ones.

**Runs:**
- `bench/precon/runs/20261015_exp017-test.json` (primary);
- `20261015_exp017-dev{,-r2,-r3}.json` (calibration);
- `20261015_exp017-probe-dev.json` (recognition).

Judgments, transcripts and items are in the gitignored run folders and synced to `kasper-cbx/mtg-search-experiments`.
