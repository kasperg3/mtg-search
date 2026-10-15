# EXP-015: Pooled-judging smoke test (issue #2) with Claude Code (Sonnet)

**Question.** The agentic labels (deck needs, attributes) are incomplete tag/regex proxies. Some are built from regexes, which favour grep (EXP-006 label-source check). Can a blind LLM judge supply the missing labels reliably and cheaply on the owner's subscription?

**Answer (smoke, 8 tasks): yes.**
- The judge is consistent: 97% of cards get the same relevant / not-relevant call in two blind passes.
- It agrees with the existing labels: 99% of labelled cards are judged relevant.
- It discriminates: blind random decoys are judged relevant only 4% of the time.
- **The labels are badly incomplete.** 89% of the agents' picks that the labels miss are judged relevant.
- **The search-vs-grep comparison shifts.** With pooled labels, P@10 on these 8 tasks is search 0.975 vs grep 0.912. With the original labels it was 0.738 vs 0.750.
- **Cost is small.** Judging every agentic task would take about 150 calls per pass, about 0.35M mostly-cached tokens.

## Setup

- **Judge.** Headless Claude Code (`claude -p`), `claude-sonnet-5-5`, on the subscription with no API key. It runs with no tools in an empty working directory, with the same isolation checks as every Claude Code run (`eval/agentic/judge_cc.py`).
- **Rubric and prompt.** These are the ones pre-registered in `judge.py` (EXP-006 write-up):
  - 2 = clearly satisfies the request and every constraint;
  - 1 = partially satisfies it, or a card a reasonable player would still consider;
  - 0 = does not satisfy it, or violates a constraint.

  The rubric adds: judge only from the card text, and do not reward popularity or power level.
- **What the judge sees.** The request (with constraints) and the plain card text: name, cost, type, rules text, P/T, keywords. Cards come 15 per call in a seeded random order. The judge never sees the condition, the labels or any popularity data.
- **Pool.** The union of the final top-20 lists of both EXP-013 agents, search (A) and grep (G), per task. Unlike `judge.py`, the pool includes cards the labels already mark relevant, so agreement can be measured.
- **Tasks.** 8, seeded (`random.Random(2)`):
  - 3 deckneeds: orzhov-lifegain-payoffs, punish-spells, refill-red-aggro;
  - 3 paraphrase-need (regex labels): each-opponent-mills, one-blocker-max, tag-fog;
  - 2 attribute: green-cheap-fog, white-anthem-enchantment.

  That gives 218 pooled cards: 152 labelled, 66 unlabelled.
- **Passes.** Two passes in different orders (36 calls), then a third pass (25 calls). The third pass mixes in blind controls:
  - 48 random cards that meet each task's constraints (negatives);
  - 48 cards the agents retrieved but did not pick (harder; true relevance unknown).

## Results

Data: `eval/agentic/runs/judge-smoke/` (summary in `summary.json`) and `eval/agentic/runs/judge-smoke-decoys/`. Analysis: `judge_smoke_analyze.py`.

**Consistency** (pass 1 vs pass 2, 218 cards):

| | value |
|---|---|
| same grade (0/1/2) | 77.5% |
| same relevant / not relevant | **97.2%** |
| Cohen's κ, 3 grades (unweighted / weighted) | 0.57 / 0.58 |
| pass 3 (with decoys mixed in) vs passes 1–2 consensus, relevant / not | 94% |

Grade 1 vs 2 is the unstable call. Whether a card is relevant at all is stable.

**Agreement with labels and blind controls.** The consensus grade is the lower of the two passes.

| cards | n | judged relevant (≥1) | grade 2 |
|---|---|---|---|
| pooled, labelled relevant | 152 | **98.7%** | — |
| pooled, unlabelled | 66 | **89.4%** | 18% |
| control: random cards meeting the constraints | 48 | **4.2%** | 0% |
| control: retrieved by an agent, not picked | 48 | 22.9% | 2% |

The judge separates relevant from irrelevant cards cleanly. The two random decoys it accepted were both grade 1, on the Orzhov lifegain task. So the high rate on unlabelled pool cards means **the original labels miss most of the good cards the agents pick**, not that the judge is lenient. The gap varies by task:
- each-opponent-mills: 29 of 29 unlabelled picks judged relevant (regex labels are narrow);
- refill-red-aggro: 14;
- punish-spells: 8;
- orzhov-lifegain-payoffs, tag-fog, one-blocker-max: 0–1 (the labels are already near-complete).

**Effect on the EXP-013 comparison** (P@10 of the final lists, the same 8 tasks):

| labels | search (A) | grep (G) |
|---|---|---|
| original | 0.738 | 0.750 |
| pooled (original ∪ judged relevant) | **0.975** | 0.912 |

Only 8 tasks, so this is illustrative. Pooled labels lift both agents and reverse the order. That matches the label-source check: grep's apparent edge came from regex-built labels.

**Cost**:
- 36 calls (passes 1–2): 2,245 tokens per call on average, 81k tokens in total (11k output);
- 4.5 s per call;
- every reply was valid, complete JSON;
- notional list price $0.39 (subscription, so nothing was spent).

A full run over the 80 deck and attribute tasks would cover about 2,200 pooled cards: about 150 calls and 0.35M tokens per pass.

## Next step (owner decision)

Judge the full EXP-013 pool (A and G, 80 non-known tasks) with two passes. Then report EXP-013's deck and attribute strata under pooled labels next to the original ones. This judge setup, rubric and consensus rule (minimum of two passes) would be fixed before the run.
