# scripts

## Index refresh (`refresh.py`)

Keeps the HF dataset `kasper-cbx/mtg-search-index` in sync with Scryfall's
`oracle_cards` bulk file.

```bash
uv run python scripts/refresh.py --dry-run      # exit 0 = up to date, 10 = stale
uv run python scripts/refresh.py                # full refresh if stale
uv run python scripts/refresh.py --force        # full refresh regardless
uv run python scripts/refresh.py --build-image  # ... then run docker/build.sh (if present)
```

Steps:
1. Compare Scryfall `updated_at` with `data/manifest.json` locally and on the HF dataset's `main`.
2. Tag the snapshot currently on `main` as `scryfall-<YYYYMMDDHHMMSS>` if it has no tag yet.
3. Ingest (`python -m mtg_search.ingest`), check that the card count is between 33k and 36k, and upload `data/*.parquet` plus `manifest.json`.
4. Launch the HF Job `hf jobs uv run --flavor t4-small ... jobs/encode.py --model lightonai/mLateOn --kind late --require-gpu` and poll it until it finishes.
5. Check that the `oracle_ids_sha256` in `embeddings/late/lightonai__mLateOn/meta.json` matches the new `cards.parquet`.
6. Run the eval smoke test (`scripts/eval_smoke.py`, which needs `--extra models`):
   - It runs late-mLateOn on the handwritten and paraphrase sets.
   - It fails if nDCG@10 drops more than 3 points below the `late-mLateOn` row in `eval/results.md`.
   - Labels are oracle_ids from the snapshot the eval sets were built on. `--rebuild-labels` re-runs `eval/build_handwritten.py` and `eval/build_paraphrase.py` first, which rewrites the committed YAML files.
7. Tag the new snapshot `scryfall-<YYYYMMDDHHMMSS>`.

Old builds stay reproducible: pin `revision="scryfall-..."` in `hf_hub_download` or `snapshot_download`. After a refresh, the dense, sparse and answerai embeddings on `main` still belong to the previous snapshot. They are not shipped; re-encode them by hand if you need them.

To run it in CI, use `.github/workflows/refresh.yml`. It runs weekly (Monday 06:00 UTC) and on manual dispatch, and needs the repository secret `HF_TOKEN`. That token must have write access to the dataset and must be allowed to run HF Jobs.
