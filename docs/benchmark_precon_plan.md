# Plan: precon cut/add benchmark (issue #3)

Status: **plan only**. No data has been collected and no experiment has been run. This document covers terms, data model, tasks, metrics, leakage control, splits, baselines and implementation. Open questions for the owner are listed at the end.

## 1. Motivation

Our existing evals (`eval/README.md`, EXP-006 in `eval/agentic/`) measure *retrieval*: given a described need, does the agent find cards that satisfy it? They do not measure *deck-building judgement*. That means deciding what a real deck lacks, which of its cards are weakest, and which concrete cards to bring in.

Commander preconstructed decks ("precons") are a natural testbed. Every player starts from the same fixed 100-card list, and thousands of them publish their upgraded versions, so we can observe what they cut and what they add. EDHREC aggregates these choices on its precon pages.

That gives us a benchmark grounded in real decisions, with three properties:
- **Many valid answers.** This needs graded metrics.
- **Fixed, verifiable inputs.** Precon decklists are official.
- **Natural time-based split.** New precons come out every few months, so some are guaranteed to post-date any model's training data.

The user's constraint shapes the whole design. The agent must choose cards from **card text and function, not hype**. The search tool already exposes no EDHREC rank (commit `619f4cc`), and the benchmark must not let popularity back in through any other channel (see §7).

This creates a deliberate tension. The ground truth *is* crowd behaviour, but the agent is forbidden from seeing crowd signals. The benchmark therefore measures how far text-based reasoning recovers what experienced players converge on. The ground truth is a *proxy* for merit, and two things handle that:
- **Gains are graded** (§6).
- **The popularity share of the score is measured explicitly** with a popularity-oracle baseline (§9).

## 2. Data sources and terms (checked 2026-10-14)

The checks were manual. I fetched each `robots.txt`, each terms page, the EDHREC `/precon` index and one precon page (`/precon/avengers-assemble`), once each, to learn the structure. Nothing was crawled, and none of the fetched pages are committed.

| Source | What it offers | robots.txt | Terms of service | Verdict |
|---|---|---|---|---|
| **EDHREC** (edhrec.com) | `/precon` index: 56 product groups, 176 decks. Each precon page embeds its full data in the Next.js `__NEXT_DATA__` JSON: `deck` (the official list, with set and collector number), `Cards to Add` / `Lands to Add` (`num_decks`, `potential_decks`, `synergy`, `lift`), `Cards to Cut` / `Lands to Cut` (`unpopularity`), and `precon_commander_counts` (number of decks per commander). `json.edhrec.com` serves the same data as JSON; it is undocumented, unofficial and has no key. Its `robots.txt` returns 403. | `/precon/` is **not** disallowed. Only `/articles/preview`, `/articles/search`, `/deckpreview`, `/puzzlebookvegas` are disallowed. | (Effective 2024-08-06.) Grants a licence "solely for your own **personal, noncommercial use**". Prohibits "use software or **automated agents or scripts** … to generate automated searches, requests, or queries to the Site". Also: "no part of the Site may be copied, reproduced, distributed, republished, **downloaded** … unless otherwise indicated". | **Automated collection is not permitted without written permission.** robots.txt allowing a path does not override the ToS. Even manual saving is grey: the purpose is a benchmark for a product, which is arguably not personal use. We must **ask EDHREC first**: describe the project, about 180 page fetches, a private derived dataset, no redistribution, attribution. |
| **Archidekt** (archidekt.com) | Public decklists, and an undocumented JSON API used by the site. Precon upgrades could be found as before/after diffs. EDHREC's precon payload itself carries an `archidekt` field, which suggests EDHREC sources part of its deck data from Archidekt. | Only `/login?`, `/partialCompare`, `/playtester-v2/`, `/sandbox?` are disallowed. | Same template as EDHREC: personal non-commercial licence, no automated queries, no copying or downloading. | **Ask permission** before any API use. |
| **Moxfield** (moxfield.com) | Public decklists. The API is not public. | `api2.moxfield.com/robots.txt`: `Disallow: /` for all agents. The site's robots.txt disallows `/search/*` among others. | The terms page is a JavaScript-only SPA, so a plain fetch could not read it; I did not render it. Moxfield is widely understood to grant API access only by agreement. | **Do not use without an explicit API agreement.** The robots rule on the API host is already a clear no. |
| **MTGJSON** (mtgjson.com) | `AllDeckFiles` / `DeckList` includes every official precon decklist with `uuid`s and Scryfall ids. | n/a (bulk downloads) | MIT licence: permission to use, copy, modify, publish and distribute. | **Permitted.** Use it for the *input* decklists, so that inputs do not depend on EDHREC. |
| **Scryfall** (already used) | Oracle text, legality, colour identity, prices, oracle tags (`data/*.parquet`). | n/a | Bulk data; we already comply (User-Agent, rate limits, private dataset). WotC Fan Content Policy. | Permitted, as today. |

**Redistribution.** Derived EDHREC data (cut and add rates) stays **private**: it is never pushed to the HF dataset `kasper-cbx/mtg-search-index` and never published in write-ups beyond aggregate scores. If EDHREC grants permission, the snapshot goes under a gitignored path, `eval/precon/gold/` (add it to `.gitignore`), or in a separate private store. Card names and text already fall under the Fan Content Policy situation described in `docs/hf_dataset_card.md`.

**What to ask EDHREC for.**
1. Permission to fetch about 180 precon pages, or their `json.edhrec.com` equivalents, once per snapshot, at most about 4 snapshots a year, at 1 request every 5 s or slower, with an identifying User-Agent and contact address.
2. Permission to store the derived cut and add rates privately for evaluation only, with no redistribution.
3. Clarification of the semantics of `unpopularity`, and of whether `potential_decks` counts all decks with that commander or only precon-derived decks.
4. Optionally, whether they have an official export or partner channel.

Mention that the use is a product evaluation, because the ToS limits use to non-commercial. **The owner decides who sends this and under which identity.**

**Fallback if permission is refused or slow.** Use a *small hand-annotated merit set*. One or more experienced players mark cuts and adds for about 8 post-cutoff precons using only card text, under the same rules as the agent. This set is smaller, but fully merit-based and unencumbered (§11 Q8).

## 3. What one EDHREC precon page gives us (structure from the single fetch)

These figures come from `avengers-assemble`, a Marvel Super Heroes Commander precon released in June 2026:
- **Commander.** `potential_decks` = 3,702 decks with this commander. The page also lists alternative commanders from the same list: Director Nick Fury (241 decks) and others.
- **Cards to Add.** 50 nonland cards. Each has `num_decks/potential_decks` (inclusion rate *r*, from 0.66 down to 0.08; the list is truncated near r ≈ 0.08), `synergy` (commander-specific lift over the colour baseline) and `lift`. The top adds are mostly new Marvel cards from the same product line, such as Agent Phil Coulson and two Captain America variants. These are thematic adds, a caveat for §6.
- **Lands to Add.** 8 cards. Shocklands with *negative* synergy (−0.31), which are generic and price-driven.
- **Cards to Cut.** 21 cards. **Lands to Cut.** 10 cards. Each has `unpopularity` *u* (0.77 down to 0.05). The precon itself holds 86 distinct cards, 99 counting basics. Cards not in the cut lists have no reported rate.
- **Card ids.** EDHREC `id`s are Scryfall **printing** ids, not `oracle_id`s. Our `cards.parquet` is oracle-level, so mapping goes by name, which is exact and unique for non-funny cards. Face names cover DFCs, and a set/collector-number check is the fallback. Unmapped names are logged and fail the build.

## 4. Data model (private, versioned)

`eval/precon/` (code is tracked; `gold/` is gitignored):

```
precons.parquet        # one row per precon deck (from MTGJSON; public-safe)
  precon_id            # e.g. "msc-avengers-assemble"
  product, set_code, release_date
  commander_oracle_ids # list (partners / backgrounds)
  color_identity       # list
  deck                 # list<struct{oracle_id, qty, is_land}>, 99 or 100 cards
  split                # train | dev | test (see §8)

gold/<snapshot_date>/labels.parquet   # one row per (precon, card, kind); PRIVATE
  precon_id, oracle_id, name
  kind                 # cut | cut_land | add | add_land
  rate                 # u for cuts; r = num_decks/potential_decks for adds
  synergy, lift        # adds only
  potential_decks      # sample size behind the rate
  in_precon            # sanity: cuts must be True, adds False
  is_new_in_product    # add was first printed in the same product/main set (thematic add flag)
gold/<snapshot_date>/manifest.json    # fetch timestamps, source URLs, EDHREC buildId, counts, sha256
```

**Snapshot rules.**
- **Pin labels to a snapshot date.** Results always cite `snapshot_date` together with the `cards.parquet` manifest date.
- **Require enough decks.** A precon enters a scored split only once `potential_decks ≥ 300` and it is at least 8 weeks past release, so the rates are stable. For example, Reality Fracture (`frc`, 2026-10-02) does not qualify yet.
- **Keep labels mappable.** Every label must map to an `oracle_id` and be commander-legal in colour identity. Cards printed after the `cards.parquet` snapshot block the build until the data is refreshed.

## 5. Tasks: exact agent inputs and outputs

Every task uses the same system prompt template, `temperature=0`, and the same tool server as the shipped MCP (mLateOn, popularity-free). **Input deck rendering:** each card is shown as `name — cost · type`, followed by oracle text, using our `render_card` in compact mode with **no price and no tags**. The commander is shown in full. The precon's product name and deck name are **not** given; only the cards are (§7, T2).

| Task | Input | Output (via `final_answer` tool, JSON) | Budget |
|---|---|---|---|
| **CUT** | Commander, plus the 99-card list rendered as above. Instruction: "identify the 10 cards whose removal most improves the deck; you will replace them with stronger cards for this commander". | `{"cuts": [10 names, best cut first]}`. Each name must be in the deck and must not be the commander. Basic lands are allowed but discouraged. | Up to 5 tool calls (`get_card` / `search_cards` / `find_similar`), 10 turns. |
| **ADD** | Same as CUT. Instruction: "propose 20 cards to add (not already in the deck)". | `{"adds": [20 names, best first]}`, nonland or land; scored separately (§6). | Up to 10 search calls, 15 turns. |
| **SWAP** | Same as CUT. Instruction: "propose 10 one-for-one swaps that keep the deck at 100 cards". | `{"swaps": [{"cut": n, "add": n, "reason": "..."} × 10]}`. `reason` is free text for qualitative review only and is not scored. | Up to 12 tool calls, 20 turns. |

**Shared rules (stated in the prompt, checked by the scorer):**
- Commander format: singleton, except basic lands and cards that allow any number.
- Every card within the commander's colour identity, and commander-legal.
- **Bracket:** an optional parameter (default bracket 3, so at most 3 Game Changers in the final deck). It uses `game_changer` from `cards.parquet`, which reflects WotC's list rather than popularity.
- **Budget:** off by default (§7, T6).

Names are resolved exactly, as in `CardService.resolve_name`. Unresolved, illegal or out-of-identity names count as **invalid**. An invalid answer gains nothing and is reported in the validity rate; it does not crash the scoring. Duplicates count once, and missing slots count as zero gain. Scoring is fully deterministic given the output.

## 6. Metrics and gain functions

### Gains (primary: thresholded; sensitivity: continuous)

| | gain 3 | gain 2 | gain 1 | gain 0 |
|---|---|---|---|---|
| **Cut**, by *u* (`unpopularity`) | u ≥ 0.50 | 0.30 ≤ u < 0.50 | 0.15 ≤ u < 0.30 | u < 0.15 or not listed |
| **Add**, by *r* (`num_decks/potential_decks`) | r ≥ 0.30 | 0.15 ≤ r < 0.30 | 0.05 ≤ r < 0.15 | below 0.05 or not listed |

- The thresholds are fixed on the **train split** before any test run. They are chosen so that about 10 cards per precon have gain ≥ 2.
- **Sensitivity check:** use continuous gains (g = u, and g = r). Report Kendall τ between system rankings under the two gain functions. If they disagree, the threshold choice is driving the conclusions, and we say so.
- **Generic-staple discount (secondary add metric, "add-synergy"):** g' = g · 1[synergy > 0]. This keeps only adds that are more common *for this commander* than for its colours in general, which removes most generic, popularity- or price-driven staples (such as shocklands, with synergy −0.31). It is the closer proxy to "functional fit for this deck".
- **Thematic adds:** labels with `is_new_in_product` are reported as a separate slice. For Universes Beyond decks especially, many top adds are same-franchise cards. Finding them by text is legitimate, but it is not evidence of judgement.

### Metrics
- **CUT** (output of exactly 10):
  - graded **nDCG@10**;
  - **P@10** (share of picks with gain ≥ 1);
  - **R@10** over G₂ = {gain ≥ 2} (cards strongly agreed to be cut);
  - **false-cut rate**: share of picks with no listed cut rate. This is noisy, because EDHREC lists only about 31 cuts.

  Cuts are close to a set decision, so set-based **graded F1@10** against G₂ is reported next to nDCG.
- **ADD** (output of 20):
  - graded **nDCG@10** and **nDCG@20**;
  - **P@10** and **P@20** (gain ≥ 1);
  - **R@20** over G₂.

  These are computed separately for nonland adds (primary) and land adds. Land adds are mostly price and popularity driven, so they are a diagnostic only. The add-synergy variant is reported too.

  Because labels are truncated at about the top 50, these are **lower bounds**. A valid card that did not make EDHREC's list scores 0.
- **SWAP:** cut-side and add-side nDCG@10, computed as above on the 10 cuts and 10 adds. **Pair-level ground truth does not exist on EDHREC.** It would need per-deck diffs from Archidekt or Moxfield, which are blocked on permission (§2). Until then, pair coherence is a *diagnostic* only. It measures the share of pairs whose cut and add share at least one functional oracle-tag role (ramp, draw, removal and so on, using the role closure from `docs/tag_filter.md`). It is never part of the score.
- **Constraint validity (deterministic, merit-side):**
  - share of valid names;
  - colour-identity and legality violations;
  - singleton violations;
  - Game Changer count against the bracket.
- **Structural sanity diagnostics on the final deck after SWAP:**
  - change in land count, flagged if beyond ±2;
  - number of coloured sources per colour, against the pips of the commander and the deck;
  - change in average mana value;
  - role coverage: counts of ramp, card draw, targeted removal and board wipes from oracle tags, against minimums (8 / 8 / 6 / 2) and against the precon.

  These are reported but **not folded into the headline score**. Oracle tags are community labels and are used here only to diagnose; they are never ground truth.
- **Cost:** tool calls, input and output tokens, USD per episode, and wall time.
- **Aggregation:**
  - **unit = precon**: macro mean, with 95% bootstrap CIs over precons (10k resamples);
  - **paired comparisons** between systems on the same precons (paired bootstrap / sign test);
  - **slices** by split, colour count, era, Universes Beyond vs in-universe, and number of decks.
- **Nondeterminism:** scoring is deterministic, but provider sampling is not perfectly so even at temperature 0. Run 3 repeats per (precon, task, system) and report mean ± sd. Tool outputs are cached per call for replay.

## 7. Leakage threat model and mitigations

| # | Channel | Risk | Mitigation | Check |
|---|---|---|---|---|
| T1 | **Popularity in tool output**: EDHREC rank, "most played" ordering, staple labels | The agent picks by hype. | Already done in `619f4cc`: the EDHREC rank has been removed from rendering, ordering, name resolution and docstrings. **Still to do:** `eval/agentic/env.py:172` orders `grep` results by `edhrec_rank`. Switch to alphabetical or oracle_id order for the lexical baseline. | Existing `tests/test_server.py` asserts no EDHREC text. Extend it with a probe sweep over all 4 tools × 50 queries asserting no `edhrec`, `rank #`, `popular`, `staple` or `most played` strings. |
| T2 | **Parametric memory of EDHREC / upgrade guides**: the model has read "Precon upgrade guide: cut X, add Y" articles. EDHREC itself publishes one per precon; the Avengers page links one dated 2026-06-15. | Scores measure recall, not reasoning. | (a) **Primary test split = precons released after the model's training cutoff + 2 months** (§8). Spoilers and guides appear around release. (b) Do not name the product or deck in the prompt; give only the card list. (c) **Memorisation probe**: a no-tools model gets *only* the product and deck name ("Avengers Assemble, Marvel Super Heroes Commander") and is asked to list the common EDHREC cuts and adds. Score it with the same metrics. (d) **Anonymised-commander ablation**: the commander's name is replaced with "your commander" and only its rules text is shown. The 99 still identify the deck, so this is only a partial mitigation. | If probe scores on train/dev are much higher than on test, contamination is confirmed and train/dev scores are reported as "contaminated". If the probe scores near random on test, the test split is clean for that model. Report probe scores next to every headline result. |
| T3 | **Oracle tags or our data encode the ground truth** | A tag like `precon-filler` or an evaluative tag leaks the answer. | Deny-list evaluative or popularity-like tags in the roles line in eval mode. Found so far: `meme` (130), `weaker-in-singleton-formats` (89), `stronger-in-singleton-formats` (6), `staple-with-set-s-mechanic` (63), `planeswalker-deck-staples` (0). Also, `cycle-*-face-commander` / `*-precon-*` tags identify precon membership. They are harmless for the add and cut targets, but they belong on the deny-list too. | **Tag-enrichment audit** on train labels: for each tag, compute lift = P(tag \| gold cut) / P(tag \| kept card), and the same for adds against the candidate pool. Review by hand any tag with lift > 3 and support ≥ 5. Functional tags with high lift, such as `tapland` for cuts, are expected merit signals and are fine. Non-functional ones go on the deny-list. |
| T4 | **Test data in prompts or few-shots** | Direct leakage. | Few-shot examples (if any) come only from the **train** split. The prompt template is hashed per run. | Scorer pre-flight: assert that no oracle_id or name from any dev/test gold set appears in the rendered system prompt or few-shots. This excludes the episode's own deck, which is input. |
| T5 | **Prompt tuning on test** | Overfitting. | Iterate on train (prompt), select on dev, and run test **once per frozen config**. Test results are logged in REGISTRY with the config hash. | Process rule; REGISTRY records the number of test evaluations. |
| T6 | **Price as a popularity proxy**: the price shown in tool output | Price correlates with demand, which is mostly EDHREC demand. | In the benchmark's primary condition, run the server with price **hidden** (new flag `MTG_SEARCH_HIDE_PRICE=1`). The budget constraint, if used, is enforced by the *scorer* only. Ablation: price shown. | Compare add-side scores with price on and off. |
| T7 | **Rarity / reprint markers** | Weaker proxies, such as "mythic". | Keep them: they are printed facts on the card. Note this as a known weak proxy. | Diagnostic: correlation of picks with rarity, per condition. |
| T8 | **The agent prefers popular cards anyway** (parametric "staple" prior) | Not leakage of *labels*, but the bias the user wants to avoid. | Measure it: mean log `edhrec_rank` of the agent's adds, computed offline by the scorer and never shown to the agent, compared with the popularity-oracle baseline. | Reported as "popularity drift" per condition. |
| T9 | **Web access** | Direct lookup. | The agent has only the mtg-search tools; there is no web or fetch tool in the harness. | Harness allow-list assert. |

## 8. Splits

- **Split key: precon release date** (MTGJSON `releaseDate`), relative to each evaluated model's training cutoff *C*:
  - **train:** released before C − 12 months. Use it for prompt iteration, threshold fixing and few-shot sources.
  - **dev:** released from C − 12 months to C. Use it for config selection. It is likely contaminated, so read it relatively.
  - **test:** released after C + 2 months, meeting the §4 gates (8 weeks out, ≥ 300 decks). This is the **headline**.
- **Candidate test pool today** (EDHREC index, 2026): Lorwyn Eclipsed (2 decks), TMNT Eternal (1), Secrets of Strixhaven (5), Marvel Super Heroes (4), Reality Fracture (1, not yet gated), plus about 3 Secret Lair decks. That is **about 12–16 decks**, a small *n*: CIs will be wide, and the plan says so. The pool grows by about 4–6 decks per quarter, for example `trc` in November 2026.
- **Splits depend on the model.** We need the cutoff for every model we evaluate. For GLM-5.3, which EXP-006 uses, the cutoff is unknown. Estimate it empirically with the T2 probe across release dates: the date where recall drops to the random baseline marks the effective cutoff.
- **Exclusions:** the Magic Online theme decks (`td0`), and precons with fewer than 300 decks. Secret Lair decks are optional (Q9).

## 9. Baselines and conditions

All are run on the same precons, budgets and scorer.

1. **Random-legal** (seeded). Cuts: uniform over non-commander nonbasic cards. Adds: uniform over the commander-legal, in-identity pool, excluding the deck. This is the floor.
2. **Text heuristic, no LLM.** Cuts: tapped lands first (from oracle text), then highest MV among cards whose oracle tags have no role overlap with the commander's tags. Adds: top results from `search_cards` on the commander's own oracle text plus 4 generic role queries. This shows how far a simple text rule gets.
3. **Popularity oracle (reference, not a competitor).** Adds: best global `edhrec_rank` in the commander's identity, not in the deck. Cuts: worst `edhrec_rank` cards in the deck. **This quantifies how much of the ground truth is plain global popularity.** If it beats the agents, the add-synergy metric (§6) becomes the more meaningful headline.
4. **Lexical agent** (EXP-006 condition F: `grep` + `bm25`), with grep ordering fixed (T1).
5. **mLateOn agent** (the shipped MCP, popularity-free, price hidden). This is the **main system**.
6. **mLateOn agent with price shown** (ablation T6).
7. **No-tools LLM** with the same deck input. It measures parametric deck-building; adds must still be real cards and are validated.
8. **Memorisation probe** (T2c): no tools, product name only.
9. **Anonymised-commander** variant of system 5 (T2d).

## 10. Implementation steps and effort

| # | Step | Effort | Depends on |
|---|---|---|---|
| 0 | Owner decides on and sends the permission request to EDHREC (and optionally Archidekt). Draft text from §2. | 0.5 h, then waiting | — |
| 1 | `MTG_SEARCH_HIDE_PRICE` flag and eval-mode tag deny-list in `render.py` / `server.py`; the probe-sweep test (T1, T3, T6). | 0.5 day | — |
| 2 | Fix `eval/agentic/env.py` grep ordering (T1). | 1 h | — |
| 3 | `eval/precon/build_precons.py`: MTGJSON precon decklists → `precons.parquet`, with oracle_id mapping and tests. | 0.5 day | — |
| 4 | `eval/precon/fetch_gold.py`: polite single-pass fetcher (UA, 5 s delay, cache, manifest), plus a parser into `labels.parquet`. **Run only after written permission.** | 0.5 day | 0 |
| 5 | `eval/precon/score.py`: gains, metrics, validity and structural diagnostics, bootstrap. Unit tests on synthetic labels. | 1 day | 3 |
| 6 | Harness: task kinds `precon_cut`, `precon_add` and `precon_swap` in `eval/agentic/agent.py` (reuse GLM client, transcripts, resumable runs), `final_answer` schemas, prompt-leak pre-flight (T4). | 1 day | 1, 5 |
| 7 | Leakage tooling: memorisation probe, cutoff-estimation sweep, tag-enrichment audit. | 0.5 day | 4, 6 |
| 8 | Pilot: 5 precons (3 train, 2 test), all baselines, 1 repeat. Sanity-check thresholds; estimate cost. | 0.5 day + about $1–3 API | 4–7 |
| 9 | Full run (3 repeats), write-up `eval/experiments/precon_benchmark.md`, REGISTRY row. | 1 day | 8 |

Total: **about 6 developer-days**, excluding the wait for permission. Steps 1–3 and 5–6 do not need EDHREC data and can start now, using a hand-made toy label file for tests.

## 11. Open questions for the owner

1. **Permission:** may we contact EDHREC (and Archidekt)? Who sends the request, under which identity, and should it describe the use as commercial (a product evaluation)? EDHREC's ToS is personal and non-commercial only, and forbids automated requests and downloading.
2. **Models under test, and their training cutoffs:** GLM-5.3 only, or Claude models too? The test split is defined per cutoff.
3. **Price:** hide price in the tool for the benchmark (proposed), or for the shipped server too? Do you want a budget-constrained task variant?
4. **Bracket:** target bracket and Game Changer limit (proposed: bracket 3, at most 3 GCs)?
5. **Headline add metric:** raw inclusion-rate gains, or synergy-filtered gains (proposed as secondary, but arguably closer to "merit")?
6. **Lands:** keep land cuts and adds out of the headline score (proposed), or include them?
7. **Small test n** (about 12–16 decks): accept wide CIs and grow quarterly, or also lean on dev with a contamination caveat?
8. **Merit-based second ground truth:** would you, or other experienced players, hand-label cuts and adds for about 8 post-cutoff precons using text only? This is also the fallback if EDHREC says no.
9. **Scope:** include Secret Lair precons and Universes Beyond decks? UB top adds are often same-franchise cards; they are reported as a slice either way.
10. **Pair-level SWAP ground truth** needs per-deck before/after diffs (Archidekt or Moxfield) and therefore their permission. Pursue it, or keep SWAP scored on its cut and add sides only?
