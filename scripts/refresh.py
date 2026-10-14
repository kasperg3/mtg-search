"""Refresh pipeline: Scryfall updated_at check -> ingest -> HF upload -> mLateOn encode job
-> eval smoke -> tag the HF dataset revision -> (optional) Docker image build.

    uv run python scripts/refresh.py --dry-run      # exit 0 = up to date, 10 = stale
    uv run python scripts/refresh.py                # full refresh if stale
    uv run python scripts/refresh.py --force        # full refresh even if up to date
    uv run python scripts/refresh.py --build-image  # ... and run docker/build.sh at the end

Staleness is judged against the manifest published in the HF dataset
(``kasper-cbx/mtg-search-index``, ``data/manifest.json`` on ``main``): that is what images
and servers are built from. The local ``data/manifest.json`` is reported too; if it is
already current, ingest is a no-op.

Every successful refresh tags the dataset with ``scryfall-<YYYYMMDDHHMMSS>`` (from the
Scryfall ``updated_at``), so older builds stay reproducible via ``revision=<tag>``.
Before overwriting ``main``, the snapshot currently on ``main`` gets its tag too if it
does not have one yet.

Requires ``HF_TOKEN`` (or a logged-in ``hf`` CLI) with write access to the dataset and
permission to run HF Jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtg_search import config  # noqa: E402

EXIT_UP_TO_DATE = 0
EXIT_STALE = 10
EXIT_FAILED = 1

LATE_MODEL = config.LATE_MODEL
LATE_SLUG = LATE_MODEL.replace("/", "__")
ENCODE_CMD = [
    "hf", "jobs", "uv", "run",
    "--flavor", "t4-small",
    "--secrets", "HF_TOKEN",
    "--timeout", "1h",
    "-d",
    str(ROOT / "jobs" / "encode.py"),
    "--model", LATE_MODEL,
    "--kind", "late",
    "--require-gpu",
]  # fmt: skip
DATA_FILES = ["cards.parquet", "tags.parquet", "rulings.parquet", "manifest.json"]
POLL_SECONDS = 30
JOB_TIMEOUT_SECONDS = 75 * 60


def log(*a: Any) -> None:
    print("[refresh]", *a, flush=True)


# --------------------------------------------------------------------------- staleness


@dataclass
class Status:
    scryfall: str
    local: str | None
    hub: str | None

    @property
    def hub_stale(self) -> bool:
        return self.hub != self.scryfall

    @property
    def local_stale(self) -> bool:
        return self.local != self.scryfall

    def describe(self) -> str:
        def mark(v: str | None) -> str:
            return f"{v or 'missing'} ({'current' if v == self.scryfall else 'STALE'})"

        return f"scryfall oracle_cards updated_at: {self.scryfall}\n  local data/manifest.json:  {mark(self.local)}\n  HF {config.HF_REPO}:  {mark(self.hub)}"


def manifest_updated_at(manifest: dict | None) -> str | None:
    if not manifest:
        return None
    return manifest.get("scryfall_updated_at") or manifest.get("bulk", {}).get("oracle_cards", {}).get("updated_at")


def fetch_scryfall_updated_at(client=None) -> str:
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=30, headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"})
    try:
        r = client.get(f"{config.SCRYFALL_API}/bulk-data/oracle_cards")
        r.raise_for_status()
        return r.json()["updated_at"]
    finally:
        if own:
            client.close()


def read_local_manifest(path: Path = config.MANIFEST_JSON) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_hub_manifest(revision: str = "main") -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        p = hf_hub_download(config.HF_REPO, "data/manifest.json", repo_type=config.HF_REPO_TYPE, revision=revision,
                            force_download=True)
    except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
        return None
    return json.loads(Path(p).read_text())


def check(client=None) -> Status:
    return Status(
        scryfall=fetch_scryfall_updated_at(client),
        local=manifest_updated_at(read_local_manifest()),
        hub=manifest_updated_at(read_hub_manifest()),
    )


def tag_name(updated_at: str) -> str:
    """``2026-10-14T09:01:55.948+00:00`` -> ``scryfall-20261014090155`` (same stamp as ingest raw dirs)."""
    dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    return "scryfall-" + dt.strftime("%Y%m%d%H%M%S")


# --------------------------------------------------------------------------- pipeline steps


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def oracle_ids_digest(cards_parquet: Path) -> str:
    # Must match mtg_search.retrievers.base.oracle_ids_digest / jobs/encode.py
    import polars as pl

    h = hashlib.sha256()
    for oid in pl.read_parquet(cards_parquet, columns=["oracle_id"])["oracle_id"]:
        h.update(oid.encode())
        h.update(b"\n")
    return h.hexdigest()


def ensure_tag(api, updated_at: str | None, revision: str = "main") -> None:
    if not updated_at:
        return
    tag = tag_name(updated_at)
    existing = {t.name for t in api.list_repo_refs(config.HF_REPO, repo_type=config.HF_REPO_TYPE).tags}
    if tag in existing:
        log(f"tag {tag} already exists")
        return
    api.create_tag(config.HF_REPO, repo_type=config.HF_REPO_TYPE, tag=tag, revision=revision,
                   tag_message=f"Scryfall oracle_cards updated_at {updated_at}", exist_ok=True)
    log(f"tagged {config.HF_REPO}@{revision} as {tag}")


def upload_data(api, updated_at: str) -> None:
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=f"data/{f}", path_or_fileobj=str(config.DATA_DIR / f)) for f in DATA_FILES]
    api.create_commit(config.HF_REPO, repo_type=config.HF_REPO_TYPE, operations=ops,
                      commit_message=f"data: Scryfall snapshot {updated_at}")
    log("uploaded data/*.parquet + manifest.json")


def parse_job_id(stdout: str) -> str:
    """`hf jobs uv run -d` prints the job id (possibly with other lines / a URL)."""
    m = re.search(r"Job ID:\s*([0-9a-f]{16,})", stdout) or re.search(r"/jobs/[^/\s]+/([0-9a-f]{16,})", stdout)
    if m:
        return m.group(1)
    ids = re.findall(r"\b[0-9a-f]{20,}\b", stdout)
    if not ids:
        raise RuntimeError(f"could not find a job id in hf output:\n{stdout}")
    return ids[-1]


def launch_encode_job() -> str:
    out = sh(ENCODE_CMD, capture_output=True, text=True)
    print(out.stdout, out.stderr, sep="", end="")
    job_id = parse_job_id(out.stdout + "\n" + out.stderr)
    log(f"encode job {job_id} launched")
    return job_id


def wait_for_job(api, job_id: str) -> None:
    t0 = time.monotonic()
    last = None
    while True:
        info = api.inspect_job(job_id=job_id)
        stage = str(getattr(info.status, "stage", info.status))
        stage = stage.split(".")[-1].upper()
        if stage != last:
            log(f"job {job_id}: {stage} ({time.monotonic() - t0:.0f}s)")
            last = stage
        if stage == "COMPLETED":
            return
        if stage in ("ERROR", "CANCELED", "DELETED"):
            msg = getattr(info.status, "message", "")
            raise RuntimeError(f"encode job {job_id} ended in {stage}: {msg}  (see `hf jobs logs {job_id}`)")
        if time.monotonic() - t0 > JOB_TIMEOUT_SECONDS:
            raise RuntimeError(f"encode job {job_id} still {stage} after {JOB_TIMEOUT_SECONDS}s")
        time.sleep(POLL_SECONDS)


def verify_embeddings(expected_digest: str) -> dict:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(config.HF_REPO, f"embeddings/late/{LATE_SLUG}/meta.json", repo_type=config.HF_REPO_TYPE,
                        force_download=True)
    meta = json.loads(Path(p).read_text())
    got = meta.get("oracle_ids_sha256")
    if got != expected_digest:
        raise RuntimeError(f"meta oracle_ids_sha256 {got} != cards.parquet digest {expected_digest}")
    log(f"embeddings meta matches cards.parquet ({meta.get('card_count', '?')} docs, "
        f"{meta.get('n_tokens', '?')} tokens)")
    return meta


def download_late_embeddings() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(config.HF_REPO, repo_type=config.HF_REPO_TYPE, local_dir=str(config.INDEX_DIR),
                      allow_patterns=[f"embeddings/late/{LATE_SLUG}/*"])
    log(f"downloaded embeddings/late/{LATE_SLUG} -> {config.INDEX_DIR}")


def eval_smoke(rebuild_labels: bool) -> None:
    if rebuild_labels:
        # Regex-labelled handwritten set + paraphrase set (copies labels from originals).
        sh(["uv", "run", "python", "eval/build_handwritten.py"])
        sh(["uv", "run", "python", "eval/build_paraphrase.py"])
    sh(["uv", "run", "--extra", "models", "python", "scripts/eval_smoke.py"])


def build_image() -> None:
    script = ROOT / "docker" / "build.sh"
    if not script.exists():
        log("docker/build.sh not found; skipping image build")
        return
    sh(["bash", str(script)])


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only report staleness (exit 0 = up to date, 10 = stale)")
    ap.add_argument("--force", action="store_true", help="refresh even if the HF dataset is current")
    ap.add_argument("--rebuild-labels", action="store_true",
                    help="re-run eval/build_handwritten.py + build_paraphrase.py before the eval smoke "
                         "(rewrites the committed query YAMLs)")
    ap.add_argument("--skip-eval", action="store_true", help="skip the eval smoke test")
    ap.add_argument("--build-image", action="store_true", help="run docker/build.sh at the end (if present)")
    args = ap.parse_args(argv)

    status = check()
    log(status.describe())
    if args.dry_run:
        if status.hub_stale:
            log("rebuild needed")
            return EXIT_STALE
        log("up to date")
        return EXIT_UP_TO_DATE
    if not status.hub_stale and not args.force:
        log("up to date; nothing to do (use --force to rebuild)")
        return EXIT_UP_TO_DATE

    from huggingface_hub import HfApi

    api = HfApi()
    # 0) keep the snapshot currently on main reachable before overwriting it
    ensure_tag(api, status.hub)

    # 1) ingest
    sh(["uv", "run", "python", "-m", "mtg_search.ingest"] + (["--force"] if args.force else []))
    manifest = read_local_manifest()
    updated_at = manifest_updated_at(manifest)
    if not manifest or updated_at is None:
        raise RuntimeError("ingest produced no manifest")
    if not 33_000 <= manifest.get("card_count", 0) <= 36_000:
        raise RuntimeError(f"card_count {manifest.get('card_count')} outside the sane range 33k-36k")
    digest = oracle_ids_digest(config.CARDS_PARQUET)

    # 2) upload data
    upload_data(api, updated_at)

    # 3) encode on HF Jobs, wait, verify
    job_id = launch_encode_job()
    wait_for_job(api, job_id)
    verify_embeddings(digest)
    log("note: dense/sparse/answerai embeddings on main are now from an older snapshot (not shipped)")

    # 4) eval smoke
    if not args.skip_eval:
        download_late_embeddings()
        eval_smoke(args.rebuild_labels)

    # 5) tag this snapshot
    ensure_tag(api, updated_at)

    # 6) image
    if args.build_image:
        build_image()
    else:
        log("next step: build the image with `bash docker/build.sh` (or rerun with --build-image)")
    log(f"refresh done: {tag_name(updated_at)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as e:
        log(f"FAILED: {e}")
        sys.exit(EXIT_FAILED)
