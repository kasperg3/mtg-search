# scripts

## Index refresh (`refresh.py`)

Keeps the HF dataset `kasper-cbx/mtg-search-index` in sync with Scryfall's
`oracle_cards` bulk file.

```bash
uv run python scripts/refresh.py --dry-run      # exit 0 = up to date, 10 = metadata-only, 11 = re-encode
uv run python scripts/refresh.py                # refresh what the new snapshot requires
uv run python scripts/refresh.py --force        # full refresh (re-encode) regardless
uv run python scripts/refresh.py --build-image  # ... then run docker/build.sh (if present)
```

Price fluctuations never trigger a corpus update; legality changes always do. Scryfall's
`updated_at` changes almost daily (mostly prices), so it only decides whether to look:

1. Compare Scryfall `updated_at` with `data/manifest.json` on the HF dataset's `main`. Equal: stop (exit 0, nothing downloaded).
2. Otherwise download and ingest into the local `data/` (`python -m mtg_search.ingest`; `--dry-run` does this too, it never touches the hub), check that the card count is between 33k and 36k, build `data/precons.parquet` from MTGJSON's AllDeckFiles (downloaded into `data/raw/mtgjson/`, again only when MTGJSON has a newer build), and compute the content fingerprint.
3. Compare the fingerprint with the published manifest to pick a tier:

   | Tier | When | What runs | `--dry-run` exit |
   |---|---|---|---|
   | no-op | all hashes equal: only volatile fields changed (prices, EDHREC rank, printing metadata) | nothing uploaded or tagged; logs "up to date (only volatile fields changed)" | 0 |
   | metadata-only | `gameplay_hash`, `aux_hash` or `precons_hash` changed, `text_hash` and `oracle_ids_sha256` equal (bans/legalities, tags, rulings, precon lists, ...) | upload tables + manifest, check the published embeddings still match `oracle_ids_sha256`, tag. No HF encode job, no eval smoke | 10 |
   | re-encode | `text_hash` or `oracle_ids_sha256` changed (oracle text, cards added/removed), or the published manifest has no fingerprint | the full pipeline below | 11 |

Full (re-encode) pipeline:
1. Tag the snapshot currently on `main` as `scryfall-<YYYYMMDDHHMMSS>` if it has no tag yet.
2. Upload `data/*.parquet` plus `manifest.json`.
3. Launch the HF Job `hf jobs uv run --flavor t4-small ... jobs/encode.py --model lightonai/mLateOn --kind late --require-gpu` and poll it until it finishes.
4. Check that the `oracle_ids_sha256` in `embeddings/late/lightonai__mLateOn/meta.json` matches the new `cards.parquet`.
5. Run the eval smoke test (`scripts/eval_smoke.py`):
   - It runs late-mLateOn on the handwritten and paraphrase sets.
   - It fails if nDCG@10 drops more than 3 points below the `late-mLateOn` row in `eval/results.md`.
   - Labels are oracle_ids from the snapshot the eval sets were built on. `--rebuild-labels` re-runs `eval/build_handwritten.py` and `eval/build_paraphrase.py` first, which rewrites the committed YAML files.
6. Tag the new snapshot `scryfall-<YYYYMMDDHHMMSS>`.

Both publishing tiers need a new Docker image (`docker/build.sh`): the image bakes `cards.parquet` and the published embeddings, so a metadata-only refresh needs no new embeddings.

### Content fingerprint (`manifest.json`)

Computed by `mtg_search.ingest.fingerprint`. Each hash is SHA-256 over the rows as canonical JSON (sorted keys), one line per row, in `cards.parquet` order (`name`). Every `cards.parquet` column is in exactly one group; `tests/test_ingest.py` fails if a new column is not classified.

- `text_hash`: `oracle_id`, `doc_text`. A change means re-encoding.
- `oracle_ids_sha256`: the row-alignment digest the embeddings carry (same as in `meta.json`).
- `gameplay_hash`: `oracle_id`, `name`, `face_names`, `faces` (per face: name, mana cost, type line, oracle text, P/T, loyalty, defense, is_permanent), `mana_cost`, `face_costs`, `pips_W`/`U`/`B`/`R`/`G`/`C`, `generic`, `has_x`, `has_hybrid`, `has_phyrexian`, `cmc`, `colors`, `color_identity`, `type_line`, `supertypes`, `types`, `subtypes`, `keywords`, `legalities`, `game_changer`, `reserved`, `layout`, `produced_mana`, `is_funny`, `is_permanent`, `is_legendary`, `can_be_commander`, `partner_kind`, `power`, `toughness`, `loyalty`, `defense`, `tags`.
- `aux_hash`: all of `tags.parquet` and `rulings.parquet` (row order ignored).
- `precons_hash`: all of `precons.parquet`, canonical JSON per row in `file` order (`mtg_search.precons.fingerprint`). The manifest's `precons` field records the MTGJSON build and the deck count.
- Not hashed (volatile): `edhrec_rank`, `image_uri`, `scryfall_uri`, and the representative printing's `set`, `set_type`, `released_at`, `rarity`. Prices are not stored at all.

Old builds stay reproducible: pin `revision="scryfall-..."` in `hf_hub_download` or `snapshot_download`. After a refresh, the dense, sparse and answerai embeddings on `main` still belong to the previous snapshot. They are not shipped; re-encode them by hand if you need them.

To run it in CI, use `.github/workflows/refresh.yml`. It runs weekly (Monday 06:00 UTC) and on manual dispatch, runs the dry run to pick the tier (step output `tier` = `none` / `metadata` / `reencode`) and then the refresh, and needs the repository secret `HF_TOKEN`. That token must have write access to the dataset and must be allowed to run HF Jobs.

## Experiment results sync (`sync_results.py`)

```bash
uv run python scripts/sync_results.py -m "EXP-00X done"
```

Mirrors run outputs (`eval/experiments/runs`, `REGISTRY.md`, `eval/agentic/runs` + `results`,
`bench/precon/runs`) to the HF dataset `kasper-cbx/mtg-search-experiments`. This includes the
gitignored per-episode outputs and transcripts. Run it after every experiment (and periodically
during long runs), next to the git push. It refuses to upload if any file contains the GLM API key.

## Releases (`bump_index.py`, `.github/workflows/publish.yml`)

The package `mtg-search-mcp` pins one index snapshot (`INDEX_REVISION` in `src/mtg_search/config.py`) and is
released whenever its version changes on `main`:

```bash
uv run python scripts/bump_index.py --latest   # point at the newest scryfall-* tag, bump the patch version
git commit -am "Release X.Y.Z" && git push      # publish.yml: tests, PyPI, GitHub release, Docker image
```

`publish.yml` skips versions already on PyPI, so other pushes are no-ops. The refresh workflow runs the same
bump after a refresh that tagged a new snapshot and dispatches `publish.yml` itself (a push made with
`GITHUB_TOKEN` does not trigger workflows). PyPI uses trusted publishing (no token): the project's trusted
publisher is this repo, workflow `publish.yml`, environment `pypi`.

