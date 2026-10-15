# Plan: judged labels for the precon deck tasks (EXP-017, proposed)

**Status: plan, awaiting the owner's go-ahead.** Nothing has been run. Once approved, this file is frozen as the pre-registration (commit before any judge call), and the results go in a separate write-up.

## Why

- **The precon labels measure popularity.** The benchmark grades ADD and SWAP picks against EDHREC add/cut rates: what the community adds to each precon. Those are popularity labels.
- **That hits the shipped surface hardest.** Its tool-using agent picks fewer staples (popularity drift 2.48 → 3.01). EXP-011's primary result fell short on exactly that: ADD nDCG@10 +0.087 [+0.015, +0.168], sign p = 0.057, "not established".
- **The labels are also incomplete.** Any good card the community rarely adds scores 0.
- **The agentic tasks showed the size of this effect.** A blind judge found 89% of unlabelled agent picks relevant (EXP-015).

A deck-aware LLM judge gives a second label source that measures **fit for this deck**, with no reference to popularity.

**Question.** Judged on deck fit rather than on EDHREC rates, are the shipped surface's ADD picks better than v1's (no tool use)? And how far do judge and EDHREC labels agree?

## Which picks get judged

**Primary:** EXP-011 phase 2, already run. 14 test precons × 3 repeats, `shipped` vs `v1` (records `bench/precon/runs/20261014_exp011-test_{shipped,v1}`).
- The pool is the add side of both arms over all repeats: ADD picks (top 20) and SWAP adds. That is about **82 unique cards per precon, about 1,150 in total**.
- This re-labels existing outputs. It adds no arm and does no selection, but it counts as test-split analysis 24 in the EXP-007 registry row.

**Secondary (optional, cheap):** EXP-014 dev, today's runs of the old and new MCP surface on 15 dev precons. About 30 unique add-side picks per precon, so about 450 cards.

**Controls, mixed blind into every batch, per precon:**
- 6 random cards that are legal for the deck and in its colour identity (negatives; expected near 0);
- 6 of the precon's top EDHREC add-labelled cards that neither arm picked (positives; expected mostly fit);
- 3 cards already in the deck (should be graded 0, "already in the deck").

## The judge

- **Model:** headless Claude Code, `claude-sonnet-5-5`, on the subscription, isolated as every run (no tools, empty directory, init checked). The same harness as `eval/agentic/judge_cc.py`, as a new `bench/precon/judge.py`.
- **What it sees:**
  - the commander's full rules text, with its name replaced by "the commander";
  - the deck's other 99 cards as one compact name list;
  - the format, colour identity and bracket rules (the agents' system-prompt rules);
  - then 15 candidate cards with their full text, in a seeded random order.
- **What it never sees:** which arm proposed a card, any EDHREC data (labels, rates, ranks), prices, the product or deck name, or the set code.
- **Leak checks:** the existing pre-flight (`bench/precon/harness/leakage.py`) runs on every judge prompt. It covers identifying strings, gold-label names outside the deck, and popularity vocabulary.
- **Recognition probe.** Before judging, 15 dev decks go to the judge in the same anonymised form, with the question "which product is this?". The recognition rate is reported. If the judge recognises most decks, its grades may echo community data, and the write-up says so.
- **Rubric (deck fit, not power or popularity):**
  - 2 = clearly fits this deck: works with the commander or the deck's evident plan, or fills a role the list is short of (ramp, draw, removal, interaction), and is legal, in colour identity and not already in the deck;
  - 1 = playable but generic or weak fit: in colour and legal, does something useful, but neither supports the plan nor fills a clear gap;
  - 0 = poor fit (works against the plan or does nothing for it), illegal, outside colour identity, over the bracket's Game Changer limit, or already in the deck.

  The judge works only from the card text and the deck list, and is told not to reward a card for being a popular staple or for raw power.
- **Passes.** One pass, as for the agentic pool. A random 10% of batches are judged a second time, in a new order, to report consistency.

## Metrics and tests (fixed before running)

Per arm and per precon, repeats averaged, using the judged grade of each add pick:

- **Primary: judged fit@20.** Mean grade of the ADD list on a 0–1 scale (grade / 2), invalid picks counting 0. The comparison is shipped − v1, paired over the 14 precons: mean difference, 10k bootstrap 95% CI, two-sided sign test.
- **Secondary:**
  - share of ADD picks graded 2;
  - judged fit@10;
  - the same for SWAP adds;
  - each arm's EDHREC nDCG@10 next to it, for the gap between the two label sources.
- **Agreement.** Share of EDHREC add-labelled cards in the pool judged ≥ 1, and of unlabelled ones; Cohen's κ (label vs judged ≥ 1).
- **Controls.** Acceptance rates of the three control groups. **Validity rule:** if more than 15% of random decoys or more than 15% of in-deck cards are graded ≥ 1, the judge is too lenient, the run is reported as invalid, and no arm comparison is drawn.
- **Cost.** Calls, tokens, time.

## Cost and time

Each call carries the deck context (about 1.5–2k tokens) plus 15 candidates (about 1.2k tokens), so about 3–3.5k tokens per call.

| part | cards | calls | tokens (mostly cached deck context) |
|---|---|---|---|
| EXP-011 test pool + controls | ~1,150 + 210 | ~95 | ~0.35M |
| 10% consistency re-judge | ~140 | ~10 | ~0.03M |
| recognition probe (15 dev decks) | — | 15 | ~0.03M |
| optional EXP-014 dev pool + controls | ~450 + 225 | ~45 | ~0.15M |

That is about 10–20 minutes with 4 workers. Everything runs on the subscription; there is no API spend.

## Limits to state up front

- **The judge is the same model family as the agents** (Sonnet judging Sonnet). It may favour its own kind of picks. A different-family judge (GLM) would be the robustness check, but it needs API budget, so it is deferred.
- **Deck fit is a judgment call.** Nothing grounds it the way EDHREC rates are grounded in real decks. The controls and the consistency re-judge bound how noisy it is, but cannot prove it right.
- **The 14 test decks are pre-cutoff for Sonnet 5.5.** The recognition probe measures how much the judge might know about them.

## Steps

1. Owner approves or edits this plan.
2. Build `bench/precon/judge.py` (pool, controls, anonymised deck prompt, pre-flight, Claude Code calls, judgments file) and an analysis script; add unit tests for the pool and control construction and the prompt leak checks.
3. Commit this file as the pre-registration; run the recognition probe on dev.
4. Judge the EXP-011 test pool (one pass + 10% re-judge); optionally the EXP-014 dev pool.
5. Analyse; write `eval/experiments/precon_judging.md`; add the registry row; sync to Hugging Face.
