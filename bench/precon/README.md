# EXP-007: precon cut/add benchmark

Spec: [`docs/benchmark_precon_plan.md`](../../docs/benchmark_precon_plan.md) (issue #3).
Schema contract: [`schema.md`](schema.md). Registry row: EXP-007 in
[`eval/experiments/REGISTRY.md`](../../eval/experiments/REGISTRY.md).

## Data

Private HF dataset: [`kasper-cbx/mtg-precon-bench`](https://huggingface.co/datasets/kasper-cbx/mtg-precon-bench)
(configs `decks`, `labels`, `precons`, `commanders`, `splits`; card: [`dataset_card.md`](dataset_card.md)).

| File | Role |
|---|---|
| `collect_mtgjson.py` | Downloads MTGJSON (MIT) `DeckList`, `SetList`, `AtomicCards`, `AllDeckFiles` into `data/bench_raw/mtgjson/` (cached) and extracts the Commander deck files. |
| `edhrec_fetch.py` | Polite fetcher: robots.txt check per URL, 1 request / 5 s, identifying User-Agent with contact, on-disk cache under `data/bench_raw/edhrec/<snapshot>/` (never re-fetched), back-off on errors / 429, `fetch_log.jsonl`. |
| `collect_edhrec.py` | One pass: the `/precon` index plus every precon page (main commander only, ~176 pages). Run only with EDHREC's permission (the owner states it was granted). |
| `build_dataset.py` | Offline: maps cards to `oracle_id`, joins MTGJSON decks with EDHREC pages, writes the parquet configs, splits, tag deny-list / audit and `manifest.json` to `data/bench/precon/<snapshot>/`. |
| `upload_hf.py` | Creates the dataset repo **private**, refuses to upload to a public repo, uploads parquet + manifest + card. |

Rebuild:

```bash
uv run python bench/precon/collect_mtgjson.py
uv run python bench/precon/collect_edhrec.py --snapshot 2026-10-14   # network, ~15 min, cached
uv run python bench/precon/build_dataset.py --snapshot 2026-10-14 [--cutoff 2025-11-01]
uv run python bench/precon/upload_hf.py --snapshot 2026-10-14
# splits for another model cutoff (writes splits_<cutoff>.parquet next to the build):
uv run python bench/precon/build_dataset.py --snapshot 2026-10-14 --splits-only --cutoff 2025-06-01
```

Defaults for the plan's open questions (recorded in `schema.md` and `manifest.json`): cutoff
`C = 2025-11-01` (GLM-5.3's is unknown), bracket 3, land cuts/adds kept but out of the headline,
Universes Beyond and Secret Lair included and flagged, raw gain as the headline add gain with
`gain_syn` secondary, plan §6 gain thresholds.

Raw EDHREC pages and all builds live under `data/` (gitignored) and are never committed.

### Snapshot 2026-10-14 (HF revision `6f71dac`, built from git `8c41e35`)

- **Fetch:** 178 EDHREC requests (robots.txt, `/precon` index, 176 precon pages), all HTTP 200,
  0 errors / retries / 429s, 13:31–13:47 UTC (~16 min at 1 request / 5 s). MTGJSON 5.3.0+20261013.
- **Precons:** 199 MTGJSON Commander decks − 16 Collector's Edition duplicates = **183**; 175 matched
  to an EDHREC page (171 by commander + set code, 4 by card-overlap Jaccard where EDHREC's face
  commander differs from MTGJSON's; the task uses EDHREC's commander there). Every matched EDHREC
  list equals the MTGJSON list (Jaccard 1.0). Unmatched: 8 CMA/CM2 reprint decks (no EDHREC page)
  and EDHREC `odds-and-ends` (SLD, no MTGJSON list).
- **Mapping:** 15,189 deck card rows, 0 unmapped (MTGJSON `scryfallOracleId`); 16,719 EDHREC
  labels, 0 unmapped (5 UB in-universe names resolved via MTGJSON `flavorName`).
- **Labels:** add 8,370 · add_land 2,182 · cut 3,675 · cut_land 2,492; ~20 headline cuts and ~21
  headline adds with gain >= 2 per scored precon.
- **Splits (C = 2025-11-01):** train 148 / dev 15 / test 20; scored train 106 / dev 15 / test 14.
  Unscored: 32 with < 300 decks, 8 without an EDHREC page, 6 released < 8 weeks ago (FDC, FRC), 2 `TD0`.
- **Leakage:** `decks` columns asserted at build; 186 oracle tags on the deny-list (incl.
  `cycle-<precon set>-*`); the train tag audit flags 59 tags (lift > 3, support >= 5), 7 of them
  precon cycles now denied, the rest functional.


## Evaluation (scorer, harness, baselines)

| File | Role |
|---|---|
| `loader.py` | `load_bench(source)` reads a local build (`data/bench/precon/<snapshot>/`, newest by default) or `hf:kasper-cbx/mtg-precon-bench[@rev]`; records the dataset revision (HF commit sha, or snapshot + labels sha256). `CardDB` = oracle-level lookup from `data/cards.parquet` (exact name / face-name resolution, full names win over face names; identity, Commander legality, Game Changer, land/basic/any-number, MV, pips, produced mana, oracle-tag closure). `edhrec_rank` is loaded only for offline metrics. |
| `scorer.py` | Deterministic scoring: validity, gains, metrics, aggregation (below). Tests: `tests/test_precon_scorer.py` (hand-computed values). |
| `harness/` | Agent loop (`agent.py`), prompts and deck rendering (`prompts.py`), toolsets (`tools.py`), GLM-5.3 client, scripted fake LLM and budget cap (`llm.py`), pre-flight leakage checks (`leakage.py`), memorisation probe (`probe.py`). Tests: `tests/test_precon_harness.py`. |
| `baselines/simple.py` | `RandomLegal` (seeded), `TextHeuristic` (no LLM), `PopularityOracle` (reference only). |
| `run.py` | Runs a system over a split, scores it and writes `runs/<run_id>.json`; `compare` (paired bootstrap, sign test, Kendall tau raw vs continuous gains) and `estimate` (GLM cost projection from a dry run) subcommands. |

### Scoring (`score_episode(task, output, deck, labels, db, ScoreConfig())`)

- **Validity.** Cuts: the name resolves exactly, the card is in the deck, it is not a commander, and it is not a duplicate. Adds: the name resolves; the card is not in the deck (basics and "any number" cards excepted); it is within the commanders' identity, Commander-legal and not a duplicate; and the final deck's Game Changer count stays within `gc_limit` (default 3, bracket 3). The limit is checked in rank order after the valid cuts are removed. An invalid pick keeps its slot with gain 0, and the reasons are counted (`<task>.invalid.<reason>`, `<task>.validity`).
- **Headline vs lands.** Picks are split into a nonland list and a land list, keeping their order. Each list is scored against its own label kinds (`cut`/`add` vs `cut_land`/`add_land`; `headline` in `labels`). Land metrics carry a `_land` prefix and are diagnostic only.
- **Gains.** Grades come from `rate` with `cut_thresholds` (.50/.30/.15) and `add_thresholds` (.30/.15/.05) (plan §6; overridable with `--cut-thresholds` / `--add-thresholds`). Variants: `.cont` (g = rate), `.syn` (g·1[synergy>0], adds) and `.nothematic` (g·1[not is_new_in_product], adds). Unlisted cards have gain 0.
- **Metrics.** DCG@k = Σ g_i/log2(i+1). nDCG@k uses the k best label gains of the precon as its ideal, and is None when the precon has no labels. P@k = #(grade ≥ 1)/k. R@k = |top-k ∩ G2|/|G2|, where G2 = labels with grade ≥ 2. F1@10 (cuts) is set-based against G2. False-cut rate = share of valid cuts that have no cut label. CUT: nDCG@10, P@10, R@10, F1@10. ADD: nDCG@10/20, P@10/20, R@20. SWAP: the CUT and ADD metrics at k=10 on each side, plus pair coherence (the share of valid pairs that share a core oracle-tag role; diagnostic).
- **Offline diagnostics** (never in the headline):
  - popularity drift: mean log10 `edhrec_rank` of valid adds;
  - after SWAP, on the final deck vs the precon: land-count change (flagged beyond ±2), coloured sources and pip shares per colour, change in average MV, and ramp/draw/removal/wipe counts from the oracle-tag closure against the 8/8/6/2 minimums.
- **Aggregation.** `aggregate` averages repeats per precon, then takes the macro mean over precons, with a 95% percentile bootstrap CI over precons (10k resamples, seeded) and the sd of the per-repeat macro means. `slice_aggregate` breaks results down by split, colour count, era, Universes Beyond and the number of decks behind the labels. `paired_bootstrap` and `sign_test` compare systems on the precons they share; `kendall_tau` compares system rankings under the two gain functions.

### Harness

- **Input** (plan §5): the commander in full, then the other 99 cards via `render_card` (compact, `fmt="commander"`). The cards are sorted by land/MV/name, with oracle tags removed, and no price or popularity is ever shown. The product, deck name, set and date are never passed to the prompt code. `--anonymise-commander` replaces the commander's name with "your commander" (T2d).
- **Toolsets.**
  - `mcp`: the shipped `CardService` tools `search_cards`, `find_similar` and `get_card`, run in-process over in-memory mLateOn (`--backend late`; `bm25` is a cheap stand-in for tests). `--backend docker` runs the built image over MCP stdio instead.
  - `lexical`: `grep` + `bm25` in EXP-006 F style.
  - `none`: no tools.

  Tags on the dataset's `tag_denylist.json` are removed before rendering. Every tool result must pass `check_tool_output`: no EDHREC, price, `$n`, popularity, staple or rank strings. The pattern has no false positives on any card text.
- **Budgets** (plan §5): up to 5 tool calls / 10 turns for CUT, 10 / 15 for ADD and 12 / 20 for SWAP. Once the budget is spent, only `final_answer` is offered. `final_answer` has a JSON schema for each task. `parse_final` also accepts JSON in the text, `"A -> B"` swaps and numbered lists, and records how each answer was parsed.
- **LLM.** GLM-5.3 runs through the Z.ai OpenAI-compatible API at T=0, with thinking effort low. The key comes from `GLM_API_KEY` / `.env` and never enters a message or transcript. A real call needs `--llm glm --allow-llm`. The default is `--llm fake`, a scripted policy that calls tools and answers, for dry runs.
- **Accounting.** Tokens (prompt, cached, completion, reasoning), USD per episode and wall time are recorded. `--max-usd` is a hard cap shared by the whole run: it is checked before every call, can overshoot by at most one call per worker, and episodes it skips are reported in `stopped`.
- **Pre-flight** (`leakage.preflight`, run before every episode):
  - no dev/test gold name in the system prompt or few-shots (the episode's own deck is exempt);
  - few-shots only from train;
  - no product name, deck name, `(SET)` code or EDHREC slug anywhere in the prompt;
  - no popularity or price strings;
  - tools only from the allow-list (no web).

  The sweep over all decks, plain and anonymised, raised 0 false positives.
- **Probe** (`--system probe`, T2c): no tools, given only the product and deck name; its output is scored as a CUT and an ADD episode.

### Runs

```bash
uv run python -m bench.precon.run --system random --split all --repeats 3
uv run python -m bench.precon.run --system heuristic --split all          # mLateOn retrieval
uv run python -m bench.precon.run --system popularity --split all         # reference only
uv run python -m bench.precon.run --system agent --llm fake --tools mcp --split test   # dry run, no API
uv run python -m bench.precon.run estimate --dry mcp=<run>,lexical=<run>,none=<run> --precons 20
```

`runs/<run_id>.json` is tracked and holds the git sha, dataset revision, config, aggregate metrics, slices and cost. `runs/<run_id>/` is gitignored and holds the per-precon outputs and metrics (`episodes.jsonl`) and the transcripts.

### Baseline results (snapshot 2026-10-14, HF `@d6bb8ba`, all 135 scored precons, git `13681d7`)

Macro mean over precons, 95% bootstrap CI. Random uses 3 seeds. No LLM was called.

| system | CUT nDCG@10 | CUT F1@10 | ADD nDCG@10 | ADD nDCG@20 | ADD R@20 | SWAP cut nDCG@10 | SWAP add nDCG@10 |
|---|---|---|---|---|---|---|---|
| random-legal | 0.250 [0.24, 0.26] | 0.169 | 0.002 | 0.002 | 0.001 | 0.243 | 0.003 |
| text heuristic (mLateOn) | 0.320 [0.29, 0.35] | 0.204 | 0.148 [0.12, 0.18] | 0.116 | 0.059 | 0.320 | 0.147 |
| popularity oracle (reference only) | 0.681 [0.65, 0.70] | 0.450 | 0.231 [0.21, 0.25] | 0.173 | 0.083 | 0.681 | 0.201 |

- Random cuts already score 0.25 nDCG@10, because many precon cards have a cut label. The scale for cuts therefore starts well above 0.
- Adds are the hard side. Random is ≈0 against about 50 labelled adds out of ~30k legal cards.
- The popularity oracle orders by EDHREC rank, so it is the ceiling a popularity-following system could reach. It is never exposed to the agent.
- All picks were valid for the three baselines (validity 1.0).

Run records: `runs/20261014_{random,heuristic,popularity}_all.json`.

### LLM cost estimate (not run: needs approval)

Token profile from fake-LLM dry runs on the test split (`runs/20261014_dry_{mcp,lexical,none}.json`), projected for GLM-5.3 at $1.40 / $0.26 cached / $4.40 per 1M tokens. The plan is 20 precons × 3 tasks × 3 repeats:

| toolset | typical | worst case |
|---|---|---|
| mcp | $26.98 | $45.30 |
| lexical | $25.85 | $43.58 |
| none | $2.83 | $2.83 |
| memorisation probe | $0.45 | — |
| **total** | **$56.11** | **$92.16** |

`uv run python -m bench.precon.run estimate --dry mcp=20261014_dry_mcp,lexical=20261014_dry_lexical,none=20261014_dry_none --precons 20`
