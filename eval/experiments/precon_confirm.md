# EXP-011: Confirmation run: select the MCP surface on dev, test once

**Status: done.** Pre-registered 2026-10-14 (commit `6bd2a29`) before any EXP-011 episode ran; the protocol was not changed. Results are below.

**Why.** The independent review ([`docs/review_2026-10-14.md`](../../docs/review_2026-10-14.md)) found two problems with EXP-009/010:
- they chose between surfaces adaptively on the same 14 test precons (M3);
- at n=14 × 1 repeat, their quality claims partly fail the sign test (M2).

This run follows the benchmark plan's rule instead: select on dev, freeze, run test once with 3 repeats, with one primary metric fixed in advance.

## Common setup

- **Agent.** Headless Claude Code 2.1.299, `claude-sonnet-5-5`, on the owner's subscription (no API).
  - Isolated sessions (`--tools ""`, strict MCP config, init check).
  - **Deferred tool loading on** (`--cc-tool-search`, the Claude Code default).
- **Server surface.** `--tools shipped` (the real `build_mcp` surface). Retrieval is the exact in-process mLateOn ranker. The shipped NextPlaid/int8 path is not measured here (review M1).
- **Data.** `kasper-cbx/mtg-precon-bench@d6bb8ba`. Scorer and validity rules unchanged from EXP-007.
- **Code.** Git commit recorded in each run JSON; runs are started from a clean tree. `--experiment EXP-011`.
- **Known, accepted biases**, applied equally to all arms:
  - the `find_similar` description example names *Rhystic Study*, a dev label (review m1);
  - land picks score 0 on the headline ADD metric (review m3; fixed and rescored 2026-10-15, primary metric unchanged).

## Phase 1: selection on dev

- **Arms.** Each arm differs only in the server's text, schemas and tool list. See [`mcp_surface_smoke.md`](mcp_surface_smoke.md) for the variants.
  - `shipped`: v3 + alwaysLoad, as in server.py;
  - `e1desc`: shipped + trigger sentence at the start of the `search_cards`/`get_card` descriptions;
  - `e6desc_slim`: e1desc + slim JSON schemas, no `list_tags`.
- **Episodes.** The 15 scored **dev** precons × ADD and SWAP × 1 repeat = 30 episodes per arm, 90 in total.
- **Selection rule** (mechanical):
  1. Take the arm with the highest mean **ADD nDCG@10** over the 15 dev precons.
  2. If another arm is within 0.02 of it, prefer the one with the higher share of episodes with at least one tool call.
  3. If that is also tied (within 5 points), prefer the smaller tool surface.

## Phase 2: test once

- **Comparison.** The frozen phase-1 winner vs the currently shipped surface. If the winner *is* `shipped`, the comparison is `shipped` vs `v1` (the text before EXP-009), which confirms the EXP-009 change instead.
- **Episodes.** The 14 scored **test** precons × CUT, ADD and SWAP × **3 repeats** = 126 episodes per arm, 252 in total.
- **Primary metric (one).** ADD nDCG@10, averaged over repeats per precon, paired over the 14 precons. It is reported with a percentile-bootstrap 95% CI **and** the exact sign test. A difference counts as established only if the CI excludes 0 **and** sign p < 0.05.
- **Secondary (exploratory, no claims):**
  - ADD validity and popularity drift;
  - tool-use share (Fisher exact);
  - CUT nDCG@10, SWAP add-side nDCG@10 and validity;
  - turns, wall time and prompt/output tokens per episode;
  - repeat-to-repeat spread.
- **Test-split evaluations.** This run adds 2 arms (now 23 in total), counted in the EXP-007 registry row.

## Results

### Phase 1: selection on dev (15 precons, 1 repeat)

| arm | ADD searched | ADD nDCG@10 | ADD validity | ADD popularity drift | SWAP searched | SWAP add nDCG@10 | surface chars |
|---|---|---|---|---|---|---|---|
| **shipped** | 20% | **0.314** | 0.973 | 2.67 | 93% | 0.156 | 12,646 |
| e1desc | 93% | 0.207 | 0.997 | 3.47 | 100% | 0.233 | 12,871 |
| e6desc_slim | 60% | 0.279 | 0.990 | 3.05 | 93% | 0.177 | 10,321 |

The rule picks **`shipped`**: highest ADD nDCG@10, with no other arm within 0.02. So phase 2 compares `shipped` vs `v1`.

On dev, the arms that search more score *lower* on ADD nDCG@10, while they pick fewer staples and make fewer invalid picks.
- The dev precons (2025) are certainly in the model's training data.
- The labels are EDHREC adds, which lean towards popular cards.
- Answering from memory therefore does well there. Searching moves picks away from popular staples (the owner's no-hype goal), and the labels penalise that.
- This is a property of the labels, worth keeping in mind for every quality number in this benchmark.

### Phase 2: test once (14 precons × CUT/ADD/SWAP × 3 repeats; `shipped` vs `v1`)

Run records: `bench/precon/runs/20261014_exp011-test_{shipped,v1}.json`, both at code commit `e679214`. The v1 record shows `dirty_bench: true` only because the shipped arm's run record was still untracked when v1 finished; no code differed.

**Primary (pre-registered): ADD nDCG@10, shipped − v1 = +0.087 [+0.015, +0.168], sign test p = 0.057.** The bootstrap CI excludes 0, but the sign test misses the pre-registered p < 0.05 bar, so **the quality gain is not established** under the protocol. The direction agrees with EXP-009 (+0.139, p=0.42) and is now much closer to significance with 3 repeats.

Secondary (exploratory):

| metric | v1 | shipped | shipped − v1 | sign p |
|---|---|---|---|---|
| ADD nDCG@10 (primary) | 0.174 | 0.262 | +0.087 [+0.015, +0.168] | 0.057 |
| ADD nDCG@20 | 0.158 | 0.223 | +0.066 [+0.008, +0.129] | 0.057 |
| ADD R@20 | — | — | +0.030 [−0.002, +0.068] | 0.039 |
| **ADD validity** | 0.868 | 0.970 | **+0.102 [+0.062, +0.152]** | **0.002** |
| ADD popularity drift | 2.48 | 3.01 | +0.54 [+0.21, +0.92] | 0.18 |
| CUT nDCG@10 | 0.362 | 0.356 | −0.007 [−0.033, +0.018] | 0.79 |
| SWAP add nDCG@10 | 0.219 | 0.217 | −0.002 [−0.085, +0.085] | 0.42 |
| SWAP add validity | 0.950 | 0.990 | +0.040 [+0.007, +0.079] | 0.070 |
| **episodes with a tool call**, ADD / SWAP / CUT | 0 / 0 / 0 of 42 | 17 / 36 / 0 of 42 | Fisher p = 2e-6 (ADD), 1.5e-17 (SWAP) | |

ADD nDCG@20, CUT and SWAP are **rescored 2026-10-15** (review m3: land picks no longer cost headline slots). Before, they read +0.067 [+0.010, +0.129]; 0.340 / 0.336, −0.004; and 0.208 / 0.215, +0.007. The primary metric, the dev selection and every conclusion are unchanged.

Repeat-to-repeat SD of the macro mean is small: ADD nDCG@10 0.010 (v1) and 0.019 (shipped). The variance is between precons, not between repeats. All 53 tool calls were `search_cards`.

**Cost of using the tools, per episode (ADD / SWAP):**
- turns: 3.0 → 3.4 / 3.0 → 3.9;
- wall time: 9 → 12 s / 16 → 14 s;
- prompt tokens: 38k → 52k / 40k → 55k, mostly cached.

In deferred mode, both arms spend one turn on `ToolSearch` for `final_answer`.

## Conclusion

- **Established on held-out data:**
  - The shipped surface (v3 + alwaysLoad) makes the agent search: 0% → 40% of ADD and 86% of SWAP episodes.
  - It cuts invalid adds from 13.2% to 3.0% (sign p = 0.002), at about 3 extra seconds and 14k mostly-cached tokens per episode.
- **Likely but not established:** better ADD quality (+0.087 nDCG@10, p = 0.057).
- **No effect:** CUT, where no tools are used, and SWAP quality.
- **Selection:** the dev selection kept the shipped surface; the description-trigger variants searched more but scored lower against EDHREC's popularity-shaped labels.
- **Test-split evaluations:** 23 in total (EXP-007 5, EXP-009 8, EXP-010 8, EXP-011 2).
