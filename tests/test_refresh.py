"""Staleness logic and helpers of scripts/refresh.py + scripts/eval_smoke.py (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


refresh = _load("refresh")
eval_smoke = _load("eval_smoke")

NEW = "2026-10-21T09:02:11.123+00:00"
OLD = "2026-10-14T09:01:55.948+00:00"


def scryfall_client(updated_at: str, seen: list | None = None) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(req)
        assert req.url.path == "/bulk-data/oracle_cards"
        return httpx.Response(200, json={"object": "bulk_data", "type": "oracle_cards", "updated_at": updated_at})

    return httpx.Client(transport=httpx.MockTransport(handler),
                        headers={"User-Agent": "mtg-search/0.1", "Accept": "application/json"})


@pytest.fixture
def manifests(monkeypatch):
    state = {"local": None, "hub": None}
    monkeypatch.setattr(refresh, "read_local_manifest", lambda *a, **k: state["local"])
    monkeypatch.setattr(refresh, "read_hub_manifest", lambda *a, **k: state["hub"])
    return state


def test_fetch_scryfall_sends_headers():
    seen: list[httpx.Request] = []
    assert refresh.fetch_scryfall_updated_at(scryfall_client(NEW, seen)) == NEW
    assert seen[0].headers["User-Agent"] == "mtg-search/0.1"
    assert seen[0].headers["Accept"] == "application/json"


def test_scryfall_error_raises():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        refresh.fetch_scryfall_updated_at(client)


@pytest.mark.parametrize(
    ("local", "hub", "hub_stale", "local_stale"),
    [
        (NEW, NEW, False, False),
        (OLD, OLD, True, True),
        (NEW, OLD, True, False),  # ingested locally but never published
        (OLD, NEW, False, True),  # published from elsewhere (e.g. CI)
        (None, None, True, True),  # first run / empty repo
    ],
)
def test_check(manifests, local, hub, hub_stale, local_stale):
    manifests["local"] = local and {"scryfall_updated_at": local}
    manifests["hub"] = hub and {"scryfall_updated_at": hub}
    st = refresh.check(scryfall_client(NEW))
    assert (st.hub_stale, st.local_stale) == (hub_stale, local_stale)
    assert "STALE" in st.describe() if (hub_stale or local_stale) else "STALE" not in st.describe()


def test_manifest_updated_at_fallback_to_bulk():
    assert refresh.manifest_updated_at({"bulk": {"oracle_cards": {"updated_at": OLD}}}) == OLD
    assert refresh.manifest_updated_at(None) is None
    assert refresh.manifest_updated_at({}) is None


@pytest.mark.parametrize(("hub", "code"), [(NEW, 0), (OLD, 10), (None, 10)])
def test_dry_run_exit_codes(manifests, monkeypatch, hub, code):
    manifests["local"] = {"scryfall_updated_at": NEW}
    manifests["hub"] = hub and {"scryfall_updated_at": hub}
    monkeypatch.setattr(refresh, "fetch_scryfall_updated_at", lambda client=None: NEW)
    monkeypatch.setattr(refresh, "sh", lambda *a, **k: pytest.fail("dry-run must not run commands"))
    assert refresh.main(["--dry-run"]) == code


def test_up_to_date_does_nothing(manifests, monkeypatch):
    manifests["local"] = manifests["hub"] = {"scryfall_updated_at": NEW}
    monkeypatch.setattr(refresh, "fetch_scryfall_updated_at", lambda client=None: NEW)
    monkeypatch.setattr(refresh, "sh", lambda *a, **k: pytest.fail("must not run commands"))
    assert refresh.main([]) == 0


def test_tag_name():
    assert refresh.tag_name(OLD) == "scryfall-20261014090155"
    assert refresh.tag_name("2026-10-14T09:01:55Z") == "scryfall-20261014090155"


@pytest.mark.parametrize(
    "out",
    [
        "Job started with ID: 68ee0f1ab2c3d4e5f6a7b8c9\nView at: https://huggingface.co/jobs/kasper-cbx/68ee0f1ab2c3d4e5f6a7b8c9\n",
        "68ee0f1ab2c3d4e5f6a7b8c9\n",
        "Job ID: 68ee0f1ab2c3d4e5f6a7b8c9",
    ],
)
def test_parse_job_id(out):
    assert refresh.parse_job_id(out) == "68ee0f1ab2c3d4e5f6a7b8c9"


def test_parse_job_id_missing():
    with pytest.raises(RuntimeError):
        refresh.parse_job_id("something went wrong")


class FakeApi:
    def __init__(self, stages):
        self.stages = list(stages)

    def inspect_job(self, job_id):
        return SimpleNamespace(status=SimpleNamespace(stage=self.stages.pop(0), message="boom"))


def test_wait_for_job(monkeypatch):
    monkeypatch.setattr(refresh.time, "sleep", lambda s: None)
    refresh.wait_for_job(FakeApi(["SCHEDULING", "RUNNING", "RUNNING", "COMPLETED"]), "x")
    with pytest.raises(RuntimeError, match="ERROR"):
        refresh.wait_for_job(FakeApi(["RUNNING", "ERROR"]), "x")


def test_oracle_ids_digest_matches_retrievers(tmp_path):
    import polars as pl

    from mtg_search.retrievers.base import oracle_ids_digest

    ids = ["b", "a", "c"]
    pl.DataFrame({"oracle_id": ids}).write_parquet(tmp_path / "cards.parquet")
    assert refresh.oracle_ids_digest(tmp_path / "cards.parquet") == oracle_ids_digest(ids)


RESULTS = """
| system | late? | tags | handwritten | paraphrase | deckneeds | para+deck | macro | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| rrf(dense+mLateOn) | yes | 61.0 | 63.4 | 14.6 | 24.1 | **16.0** | 40.8 | 225 | 380 |
| late-mLateOn | yes | 60.5 | 70.7 | 11.9 | 21.1 | **13.0** | 41.1 | 105 | 217 |
"""


def test_eval_baseline_and_regressions():
    b = eval_smoke.baseline_from_results(RESULTS)
    assert b == {"tags": 60.5, "handwritten": 70.7, "paraphrase": 11.9, "deckneeds": 21.1}
    assert eval_smoke.regressions({"handwritten": 68.0, "paraphrase": 9.0}, b, 3.0) == []
    bad = eval_smoke.regressions({"handwritten": 67.6, "paraphrase": 11.9}, b, 3.0)
    assert len(bad) == 1 and bad[0].startswith("handwritten")


def test_eval_baseline_from_repo_results():
    b = eval_smoke.baseline_from_results((ROOT / "eval" / "results.md").read_text())
    assert set(b) == {"tags", "handwritten", "paraphrase", "deckneeds"}


def test_manifest_roundtrip(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"scryfall_updated_at": OLD}))
    assert refresh.manifest_updated_at(refresh.read_local_manifest(p)) == OLD
    assert refresh.read_local_manifest(tmp_path / "missing.json") is None
