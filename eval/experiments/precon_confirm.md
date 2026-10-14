# EXP-011: Confirmation run: select the MCP surface on dev, test once

**Status: pre-registered 2026-10-14, before any EXP-011 episode ran.** The results section is filled in afterwards; the protocol below does not change.

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
  - land picks score 0 on the headline ADD metric (review m3).

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

Pending.
