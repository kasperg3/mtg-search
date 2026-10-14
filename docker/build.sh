#!/usr/bin/env bash
# Build the mtg-search MCP image (see docker/README.md).
#
#   docker/build.sh                      # artifacts from HF dataset kasper-cbx/mtg-search-index@main
#   docker/build.sh --revision v2026...  # ... at a tag / branch / commit
#   docker/build.sh --local              # artifacts from ./data and ./index (dev); fetches the
#                                        # query-encoder model into ./index/models if missing
#   docker/build.sh --tag myname [-- extra docker build args]
#
# The HF token (HF_TOKEN env, else `hf auth token` / ~/.cache/huggingface/token) is passed as a
# BuildKit secret and never stored in an image layer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="kasper-cbx/mtg-search-index"
SOURCE=hf
REVISION=main
TAG=mtg-search
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) SOURCE=local; shift ;;
    --revision) REVISION="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(hf auth token 2>/dev/null | tail -n1 || true)"
  if [[ ! "$HF_TOKEN" =~ ^hf_ ]] && [[ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
    HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")"
  fi
  [[ "$HF_TOKEN" =~ ^hf_ ]] || HF_TOKEN=""
fi
export HF_TOKEN

ARGS=(-f "$ROOT/docker/Dockerfile" -t "$TAG" --build-arg "SOURCE=$SOURCE")
if [[ "$SOURCE" == local ]]; then
  for f in data/cards.parquet data/tags.parquet data/rulings.parquet data/manifest.json \
           index/embeddings/late/lightonai__mLateOn/tokens.npy; do
    [[ -f "$ROOT/$f" ]] || { echo "missing $ROOT/$f" >&2; exit 1; }
  done
  if [[ ! -f "$ROOT/index/models/lightonai__mLateOn-fp32/source.json" ]]; then
    echo "fetching query-encoder model into index/models/ ..." >&2
    (cd "$ROOT" && uv run python -m mtg_search.artifacts download --parts model --dest index)
  fi
else
  [[ -n "$HF_TOKEN" ]] || { echo "no HF token: set HF_TOKEN or run 'hf auth login'" >&2; exit 1; }
  # resolve to a commit sha so Docker's layer cache is invalidated when the dataset changes
  SHA="$(cd "$ROOT" && uv run --quiet python -c "
from huggingface_hub import HfApi
print(HfApi().dataset_info('$REPO', revision='$REVISION').sha)" 2>/dev/null || true)"
  [[ -n "$SHA" ]] || SHA="$REVISION"
  echo "building from $REPO@$REVISION ($SHA)" >&2
  ARGS+=(--secret id=hf_token,env=HF_TOKEN --build-arg "HF_REVISION=$SHA")
fi

DOCKER_BUILDKIT=1 docker build "${ARGS[@]}" "${EXTRA[@]}" "$ROOT"
docker image ls "$TAG" --format 'built {{.Repository}}:{{.Tag}} ({{.Size}})' >&2
