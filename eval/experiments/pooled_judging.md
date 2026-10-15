# EXP-016: Pooled relevance judging of EXP-013 (search vs grep) on the 80 deck and attribute tasks

**Status: done 2026-10-15.** Pre-registered in commit `c53825c` before any judging call. The pre-registration follows unchanged, from its own status line down to "Known limits"; Results and Reading come after it.

**Correction (2026-10-15, after the results).** The scorer resolved some answers to the wrong card: another card's face name beat a card's full name, so the answer "Lightning Bolt" was scored as "Emeritus of Conflict // Lightning Bolt" (fixed in `eval/agentic/env.py`; `eval/agentic/rescore.py` re-resolved every saved episode). In EXP-013's episodes 4 answers changed: Lightning Bolt on attr-1cmc-red-burn-instant and Reanimate on hw-reanimate-own-need, for A and for G. Reanimate is not on its task's list, so it was graded by the same judge in `runs/judge-facefix` (grade 2); Lightning Bolt is on its list. The agent scores below are corrected (`analyze_pooled.py --judge-runs judge-full,judge-facefix`). Before the fix: pooled P@10 A 0.985, G 0.966; attribute stratum A 0.993, G 0.920; original-label P@10 A 0.755, G 0.744. The judge diagnostics and the new-relevant counts describe the judging run and are left as run. No sign-test count, p-value or conclusion changed.

**Answer.**
- **No established difference in precision.** Under pooled labels, P@10 is search (A) 0.986 vs grep (G) 0.968: G − A = −0.019 [−0.051, +0.011], A better on 9 tasks, G on 5, 66 tied, sign p = 0.42. nDCG@10 is level (0.853 vs 0.854).
- **The original labels were the problem, not the agents.** Both agents' top-10s are about 98% relevant under pooled labels, against about 75% under the original labels. For both agents alike, 22–23% of all top-10 slots held relevant cards the labels missed.
- **Precision is saturated.** Among the cards each agent returned in its top 10, the judge found 99.0% (A) and 98.9% (G) relevant. What still separates them is list length: G returned fewer than 10 cards on 5 tasks (20 empty slots), A on 2 (7 slots).
- **Grep's regex-label advantage does not turn into a search advantage** on the 80 tasks. In the paraphrase stratum the gap went from −0.023 (original) to −0.006 (pooled). The 8-task smoke's A lead (0.975 vs 0.912) does not hold at this size.
- **The judge behaved as in the smoke.** 98.9% of labelled cards judged relevant, 92.4% of unlabelled picks, and 1 of 160 random in-constraint decoys (0.6%).
- **Cost.** 203 calls, 448k tokens, about 3.6 minutes with 4 workers, on the subscription (notional $2.13).

**Status: pre-registered 2026-10-15, before any judging call of this run.** Everything in this section is fixed before the judge runs. Results will be added below it without changing it.

## Question (owner)

"Start with one pass and run the mLateOn vs grep experiment on the 80 decks."

EXP-013 compared an agent with the mLateOn search tool (**A**) against the same agent with only grep (**G**) and found them about level on deck and attribute tasks. Those labels are incomplete tag/regex proxies, and the regex-built ones favour grep. The EXP-015 smoke (8 tasks) showed that a blind Sonnet judge is consistent, agrees with the existing labels, rejects random decoys, and finds that most unlabelled agent picks are relevant.

**Question:** with the labels completed by a blind judge over the pool of both agents' picks, does A or G return more relevant cards in its final top 10?

**Expectation (not a decision rule).** The smoke had A ahead under pooled labels (P@10 0.975 vs 0.912 on 8 tasks) after G led slightly under the original labels. We expect A ≥ G under pooled labels, with most of the shift in the regex-labelled paraphrase stratum. The test is two-sided.

## Judge setup (same as EXP-015)

- **Judge.** Headless Claude Code (`claude -p`), `claude-sonnet-5-5`, on the owner's subscription with no API key. No tools, empty working directory, init isolation check (`eval/agentic/judge_cc.py`).
- **Rubric and prompt.** Unchanged from `judge.py` (pre-registered in EXP-006): 2 = clearly satisfies the request and every constraint; 1 = partially satisfies it, or a card a reasonable player would still consider; 0 = does not satisfy it or violates a constraint. Judge only from the card text; do not reward popularity or power level.
- **What the judge sees.** The request (with its constraints) and the plain card text (`doc_text`). It never sees the condition, the original labels or any popularity data.
- **Batching.** 15 cards per call, in a seeded random order per task (`random.Random("judge-<task>-pass0")`). One re-ask on malformed JSON (script default).
- **One pass.** Each card gets one grade, used as is (no consensus rule, unlike the two-pass plan in EXP-015).

## Pool

- **Tasks.** All 80 non-known tasks of `eval/agentic/tasks.yaml`: 65 `deck` (30 source `deckneeds`, 35 source `paraphrase-need`) and 15 `attr`.
- **Pool per task.** The union of the A and G final top-20 lists (`final_oids[:20]`, unresolved names dropped) from `eval/agentic/runs/cc-main/` (EXP-013, 1 repeat). Cards the original labels already mark relevant stay in the pool, so agreement with the labels can be measured.
- **Size (computed before judging).** 2,271 pooled cards: 1,583 labelled relevant, 688 unlabelled. Every card in either agent's top 10 is in the pool.
- **Leniency control.** 2 random cards per task that meet the task's constraints and that no agent retrieved or picked (`--decoys-random 2`, seeded `random.Random("decoy-<task>")`), 160 in total. They are mixed blind into the same batches, flagged `kind = decoy_random` in the judgments, and **excluded from the labels**.
- **Command.**

  ```
  uv run python eval/agentic/judge_cc.py --run judge-full --pool-runs cc-main --tasks <the 80 ids> \
      --passes 1 --decoys-random 2 --workers 4
  ```

  About 200 calls. If calls fail repeatedly, the missing tasks are re-run under a new run name with fewer workers and the judgments files are merged; this does not change any card's judging conditions.

## Labels

- **Original.** As in EXP-013 (`analyze.label_sets`): each relevant card gets gain 1, or its `graded` value where the task has one (6 deck tasks give part of their labelled cards gain 2).
- **Pooled relevant** = original relevant OR judged grade ≥ 1. A judge grade of 0 never removes an original label.
- **Pooled graded** (for nDCG): the original grade if the card is labelled, else the judged grade. In most tasks this mixes gain-1 original labels with judged gain-2 cards; this is accepted and noted.
- **Cards without a grade** (parse failures): keep their original label; unlabelled ones count as not relevant. Their number is reported.
- Cards outside the pool keep their original label only.

## Metrics and tests (fixed now)

All metrics are computed on each agent's final list (deduplicated, as in `analyze.py`), per task, then paired A vs G over tasks.

- **Primary.** P@10 under pooled labels, over all 80 tasks. Reported: mean A, mean G, mean difference **G − A** (same sign convention as EXP-013), 10k-resample percentile bootstrap 95% CI over tasks (seed 0), and the two-sided exact sign test over tasks with a non-zero difference. A difference counts as established only if the CI excludes 0 **and** sign p < 0.05.
- **Secondary (descriptive, same statistics, no separate claims):**
  - nDCG@10 under pooled graded labels;
  - judged share of the top 10: of the cards in each agent's top 10 that have a judgment, the share the judge grades ≥ 1 (judge only, ignores the original labels; short lists are not penalised, P@10 does that).
- **Not reported:** final R@20. Under pooling the denominator depends on what the agents found, so it is not meaningful.
- **Strata.** Every metric is reported for all 80 tasks and per stratum: deck/deckneeds (30), deck/paraphrase (35), attr (15).
- **Original labels side by side.** P@10 and nDCG@10 under the original labels, same tasks and statistics (these reproduce EXP-013).
- **Judge diagnostics:**
  - agreement: share of labelled pool cards judged ≥ 1;
  - unlabelled pool cards judged ≥ 1, and judged 2;
  - decoy acceptance: share of random decoys judged ≥ 1 (and 2). Sanity flag: if more than 10% are judged ≥ 1, the pooled labels are called lenient and the result is reported as unreliable;
  - per-task counts of new relevant cards (unlabelled, judged ≥ 1), and whether A, G or both picked them;
  - cost: calls, complete parses, tokens (input, cache writes, cache reads, output), seconds per call, wall time, notional list price (`calls.jsonl`).

## Known limits (stated in advance)

- **One pass.** The smoke found the relevant / not-relevant call stable in 97% of cards across two passes; grade 1 vs 2 is less stable (77.5% exact agreement), which matters for nDCG more than for P@10.
- **Same model family.** The judge is Sonnet 5.5 and both agents are Sonnet 5.5. A judge may favour picks that match its own reading of a request. This affects A and G alike but could raise both.
- **Labels exist only for pooled cards.** Relevant cards that neither agent picked stay unlabelled. Comparisons are between A and G on this pool, not absolute quality.
- **1 repeat of each agent** (EXP-013): run-to-run noise of the agents is unmeasured.

## Results

Run data: `eval/agentic/runs/judge-full/` (`judgments.jsonl`, `calls.jsonl`, `config.json`, `summary.json`; transcripts stay local). Analysis: `eval/agentic/analyze_pooled.py --judge-runs judge-full`. Judge: `claude-sonnet-5-5` on Claude Code 2.1.300, one pass, 2026-10-15.

### Run

- 2,271 pooled cards (1,583 labelled relevant, 688 unlabelled) plus 160 random decoys: 2,431 cards in 203 calls.
- 202 of 203 replies parsed completely. One batch of 7 cards (attr-2cmc-blue-counter) failed twice: the judge appended second thoughts after its JSON, and the mana symbols in that text (`{U}{U}`) broke the script's JSON extraction. All 7 cards are labelled relevant, so under the pre-registered rule they keep their labels and **no metric below is affected**. They are missing only from the agreement counts (1,576 instead of 1,583 labelled cards).
- No retries, no rate limits.

### Primary: P@10 under pooled labels, A vs G

Differences are G − A, 95% bootstrap CI over tasks; "G / A / tie" counts tasks.

| stratum | n | A | G | G − A [95% CI] | G / A / tie | sign p |
|---|---|---|---|---|---|---|
| **all 80 (primary)** | 80 | **0.986** | **0.968** | **−0.019 [−0.051, +0.011]** | 5 / 9 / 66 | **0.42** |
| deck/deckneeds | 30 | 0.990 | 0.983 | −0.007 [−0.037, +0.023] | 1 / 3 / 26 | 0.63 |
| deck/paraphrase | 35 | 0.977 | 0.971 | −0.006 [−0.069, +0.043] | 4 / 2 / 29 | 0.69 |
| attr | 15 | 1.000 | 0.927 | −0.073 [−0.167, −0.007] | 0 / 4 / 11 | 0.13 |

The CI includes 0 and p = 0.42, so by the pre-registered rule **no difference is established**. In the attribute stratum the CI excludes 0, but only 4 tasks differ (p = 0.13), and 3 of those 4 are G returning a short list (4, 7 and 9 cards).

Same tasks, **original labels** (these reproduce EXP-013):

| stratum | n | A | G | G − A [95% CI] | G / A / tie | sign p |
|---|---|---|---|---|---|---|
| all 80 | 80 | 0.756 | 0.744 | −0.013 [−0.065, +0.040] | 23 / 24 / 33 | 1.0 |
| deck/deckneeds | 30 | 0.793 | 0.823 | +0.030 [−0.043, +0.107] | 12 / 9 / 9 | 0.66 |
| deck/paraphrase | 35 | 0.651 | 0.629 | −0.023 [−0.120, +0.071] | 10 / 10 / 15 | 1.0 |
| attr | 15 | 0.927 | 0.853 | −0.073 [−0.147, −0.013] | 1 / 5 / 9 | 0.22 |

### Where the top-10 slots go

Each agent has 80 × 10 = 800 top-10 slots.

| slot holds | A | G |
|---|---|---|
| a card the original labels mark relevant | 605 | 595 |
| an unlabelled card the judge grades ≥ 1 | 184 | 179 |
| a card judged not relevant (grade 0, unlabelled) | 4 | 6 |
| nothing (final list shorter than 10) | 7 | 20 |

Pooled labels move both agents up by the same amount (+184 vs +179 slots). What is left between them is mostly short lists. G returned fewer than 10 cards on 5 tasks:
- hw-each-opponent-loses-need: 2;
- attr-white-anthem-enchantment: 4;
- attr-white-instant-blink: 7;
- hw-tax-spells-cost-more-need: 8;
- attr-blue-creature-bounce: 9.

A did so on 2: tag-free-sacrifice-outlet-need (5) and need-red-enchantment-answers (8).

### Secondary metrics

**nDCG@10, graded labels:**

| stratum | n | original A | original G | pooled A | pooled G | pooled G − A [95% CI] | G / A / tie | sign p |
|---|---|---|---|---|---|---|---|---|
| all 80 | 80 | 0.765 | 0.751 | 0.853 | 0.854 | +0.001 [−0.028, +0.030] | 23 / 28 / 29 | 0.58 |
| deck/deckneeds | 30 | 0.805 | 0.824 | 0.821 | 0.829 | +0.007 [−0.045, +0.058] | 10 / 13 / 7 | 0.68 |
| deck/paraphrase | 35 | 0.658 | 0.631 | 0.855 | 0.868 | +0.013 [−0.031, +0.053] | 12 / 10 / 13 | 0.83 |
| attr | 15 | 0.938 | 0.884 | 0.913 | 0.874 | −0.039 [−0.107, +0.011] | 1 / 5 / 9 | 0.22 |

Pooled nDCG is lower than pooled P@10 because the ideal ranking for a task is now full of grade-2 cards. Those are judged grade-2 cards, or labelled cards of the 6 graded tasks. Most labelled cards count only gain 1. As pre-registered, this mixes two scales.

**Judged share of the top 10** (of the returned top-10 cards, the share the judge grades ≥ 1; ignores the original labels and short lists):

| stratum | n | A | G | G − A [95% CI] | G / A / tie | sign p |
|---|---|---|---|---|---|---|
| all 80 | 80 | 0.990 | 0.989 | −0.001 [−0.013, +0.009] | 7 / 4 / 69 | 0.55 |
| deck/deckneeds | 30 | 0.986 | 0.973 | −0.013 [−0.040, +0.010] | 3 / 3 / 24 | 1.0 |
| deck/paraphrase | 35 | 0.989 | 1.000 | +0.011 [+0.003, +0.023] | 4 / 0 / 31 | 0.13 |
| attr | 15 | 1.000 | 0.993 | −0.007 [−0.020, +0.000] | 0 / 1 / 14 | 1.0 |

**Exploratory, not pre-registered: strict pooled P@10.** Here a new card counts only if judged grade 2, which tests whether the result rests on the judge's more lenient grade 1.

| stratum | n | A | G | G − A [95% CI] | G / A / tie | sign p |
|---|---|---|---|---|---|---|
| all 80 | 80 | 0.901 | 0.903 | +0.001 [−0.043, +0.044] | 20 / 17 / 43 | 0.74 |
| deck/deckneeds | 30 | 0.900 | 0.943 | +0.043 [−0.010, +0.097] | 10 / 3 / 17 | 0.09 |
| deck/paraphrase | 35 | 0.871 | 0.866 | −0.006 [−0.086, +0.069] | 9 / 10 / 16 | 1.0 |
| attr | 15 | 0.973 | 0.907 | −0.067 [−0.147, +0.000] | 1 / 4 / 10 | 0.38 |

Strict or lenient, the agents are level overall.

### Judge diagnostics

| cards | n | judged ≥ 1 | grade 2 |
|---|---|---|---|
| pooled, labelled relevant | 1,576 | **98.9%** | 72.0% |
| pooled, unlabelled | 688 | **92.4%** | 43.5% |
| random decoys: meet the constraints, never retrieved | 160 | **0.6%** (1) | 0% |

| stratum | labelled judged ≥ 1 | unlabelled n | unlabelled judged ≥ 1 | decoys judged ≥ 1 |
|---|---|---|---|---|
| deck/deckneeds | 98.2% (of 665) | 275 | 92.4% | 1 of 60 |
| deck/paraphrase | 99.4% (of 614) | 365 | 93.7% | 0 of 70 |
| attr | 99.7% (of 297) | 48 | 83.3% | 0 of 30 |

- **Decoys.** Decoy acceptance is far under the pre-registered 10% flag, so the pooled labels are not called lenient. The one decoy accepted was Nylea's Oracle (grade 1) on need-tokens-anthem.
- **Labelled cards the judge rejects.** 17 labelled cards (1.1%) got grade 0, for example:
  - Grand Abolisher for need-stop-activated-combo;
  - Akroma's Blessing for need-survive-wipe;
  - three cards on tag-wish-need.

  Under the pre-registered rule they stay relevant.
- **Same as the smoke.** The rates match EXP-015: 98.7% labelled, 89.4% unlabelled, 4.2% decoys.

**New relevant cards** (unlabelled pool cards judged ≥ 1):

| stratum | tasks | pooled | labelled | new relevant | of which grade 2 | picked by A only | G only | both |
|---|---|---|---|---|---|---|---|---|
| deck/deckneeds | 30 | 940 | 665 | 254 | 107 | 125 | 120 | 9 |
| deck/paraphrase | 35 | 979 | 614 | 342 | 177 | 165 | 135 | 42 |
| attr | 15 | 352 | 304 | 40 | 15 | 22 | 8 | 10 |
| **all** | 80 | 2,271 | 1,583 | **636** | 299 | 312 | 263 | 61 |

- **How many per task.** The median task gains 6 new relevant cards. 17 tasks gain none, and 26 gain 10 or more.
- **Where the labels were thinnest.**

  | task | new relevant | labelled cards in its pool |
  |---|---|---|
  | need-ub-card-advantage | 34 | 5 |
  | hw-land-from-hand-need | 29 | 7 |
  | tag-rhystic-need | 29 | 0 |
  | hw-dies-return-end-step-need | 28 | 0 |
  | hw-each-opponent-mills-need | 28 | 7 |

  Per-task counts are in `summary.json` (`per_task`).
- **Overlap between the agents.** A and G rarely find the same new cards: only 61 of 636 were picked by both. A's top-20 lists hold more of them (312 + 61 vs 263 + 61), partly because A's lists are longer.

### Cost

| | value |
|---|---|
| calls | 203 (202 complete parses) |
| tokens | 448k total: 388k cache writes, 2k cache reads, 58k output, 0.4k uncached input |
| per call | 2,209 tokens, 4.2 s |
| elapsed | about 3.6 minutes with 4 workers (10:04:18–10:07:57) |
| notional list price | $2.13 (subscription, nothing spent) |

## Reading

1. **On these 80 tasks, search and grep pick equally relevant cards.** Once the labels are completed, both agents' top 10s are about 98% relevant. The primary test finds no difference (−0.019 [−0.051, +0.011], p = 0.42), and neither do nDCG@10 or the judged share. This fits EXP-013's reading: with a strong model, the value of the search infrastructure is efficiency (1.46× fewer tokens, fewer turns, faster), not precision.
2. **The original labels understated both agents by about a quarter.** Under the original labels, P@10 was about 0.75. Nearly all of the gap to 1 was relevant cards the tag/regex labels miss, and it was the same size for both agents (+184 vs +179 slots). The regex-built paraphrase labels were the most incomplete: 342 new relevant cards on 35 tasks, with two tasks having no labelled card in their pool at all. They did not, however, hide a search advantage: the paraphrase gap stayed near 0.
3. **What remains is list length, not judgement.** G's remaining deficit, mainly on attribute tasks, comes from returning short lists: 4, 7 or 9 cards where 10 or more qualify. Grep without ranking sometimes stops before it has filled the list. It does not pick wrong cards: 98.9% of what it returns is relevant, against 99.0% for A. A has the same failure on two tasks.
4. **P@10 is now at its ceiling** (66 of 80 tasks tied at the pooled level) and can no longer tell good agents apart on these tasks. A future comparison needs a measure with headroom:
   - recall against a deeper, judged pool (for example the agents' retrieved-but-not-picked cards);
   - harder tasks;
   - or graded nDCG with a more stable grade (two passes).

### Limits

- **One pass.** Each card has one grade. The relevant / not-relevant call was 97% stable across passes in the smoke; grade 1 vs 2 was less stable (77.5%). That makes pooled nDCG noisier than pooled P@10.
- **Same model family.** The judge is Sonnet 5.5 and so are both agents, so it may share their reading of a request. It accepted 92% of the agents' unlabelled picks but only 0.6% of random decoys. Random decoys are easy negatives, though; this run had no hard control. In the smoke, cards an agent retrieved but did not pick were accepted 23% of the time. Any leniency applies to both agents alike, so it inflates levels more than it biases the comparison.
- **Labels exist only for pooled cards.** Relevant cards that neither agent put in its top 20 stay unlabelled. The comparison is between A and G on this pool. The absolute levels depend on the judge's leniency and say nothing about recall.
- **One repeat of each agent** (EXP-013), so the agents' run-to-run noise is unmeasured. With 66 of 80 tasks tied, a few short lists decide the difference.
- **Deviations from the pre-registration.** There were two, and neither changes a pre-registered number:
  - one batch of 7 cards went unparsed; they were all labelled relevant, so the metrics are unaffected;
  - the strict pooled P@10 was added as an exploratory check and is marked as such.
