# EXP-016: Pooled relevance judging of EXP-013 (search vs grep) on the 80 deck and attribute tasks

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
