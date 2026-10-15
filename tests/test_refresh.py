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


def test_dry_run_up_to_date_does_not_ingest(manifests, monkeypatch):
    manifests["local"] = manifests["hub"] = {"scryfall_updated_at": NEW}
    monkeypatch.setattr(refresh, "fetch_scryfall_updated_at", lambda client=None: NEW)
    monkeypatch.setattr(refresh, "sh", lambda *a, **k: pytest.fail("dry-run must not run commands when current"))
    assert refresh.main(["--dry-run"]) == 0


def test_up_to_date_does_nothing(manifests, monkeypatch):
    manifests["local"] = manifests["hub"] = {"scryfall_updated_at": NEW}
    monkeypatch.setattr(refresh, "fetch_scryfall_updated_at", lambda client=None: NEW)
    monkeypatch.setattr(refresh, "sh", lambda *a, **k: pytest.fail("must not run commands"))
    assert refresh.main([]) == 0


# --------------------------------------------------------------------------- tiers


def _fp(raw_cards, raw_tags, raw_rulings):
    from mtg_search import ingest

    cards = ingest.build_cards(raw_cards, ingest.tags_by_oracle_id(raw_tags))
    keep = set(cards["oracle_id"])
    return ingest.fingerprint(cards, ingest.build_tags(raw_tags, keep), ingest.build_rulings(raw_rulings, keep))


def _edit(raw_cards, name, fn):
    return [fn(dict(c)) if c["name"] == name else c for c in raw_cards]


def _price_only(c):
    c["prices"] = {"usd": "1234.56", "usd_foil": None, "eur": "1.00", "tix": "0.01"}
    c["edhrec_rank"] = 99999
    c["image_uris"] = {"normal": "https://example.invalid/new.jpg"}
    return c


def _banned(c):
    c["legalities"] = {**c["legalities"], "commander": "banned"}
    return c


def _errata(c):
    c["oracle_text"] = c["oracle_text"] + " You gain 1 life."
    return c


SCENARIOS = [
    ("price-only", "Sol Ring", _price_only, refresh.TIER_NONE, 0),
    ("legality", "Sol Ring", _banned, refresh.TIER_METADATA, 10),
    ("oracle text", "Lightning Bolt", _errata, refresh.TIER_REENCODE, 11),
]


@pytest.fixture
def tier_setup(manifests, monkeypatch, raw_cards, raw_tags, raw_rulings):
    """Hub manifest = fingerprint of the fixtures at OLD; Scryfall reports NEW."""
    base = _fp(raw_cards, raw_tags, raw_rulings)
    manifests["hub"] = {"scryfall_updated_at": OLD, **base}
    manifests["local"] = {"scryfall_updated_at": OLD}
    monkeypatch.setattr(refresh, "fetch_scryfall_updated_at", lambda client=None: NEW)
    calls: list[str] = []
    for name in ("upload_data", "launch_encode_job", "wait_for_job", "verify_embeddings", "ensure_tag",
                 "download_late_embeddings", "eval_smoke", "build_image"):
        monkeypatch.setattr(refresh, name, lambda *a, _n=name, **k: calls.append(_n) or "job")

    def use(cards):
        fp = _fp(cards, raw_tags, raw_rulings)
        monkeypatch.setattr(refresh, "ingest_local", lambda force: ({"scryfall_updated_at": NEW, **fp}, fp))

    return SimpleNamespace(calls=calls, use=use)


@pytest.mark.parametrize(("label", "card", "fn", "tier", "code"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_classify(raw_cards, raw_tags, raw_rulings, label, card, fn, tier, code):
    base = _fp(raw_cards, raw_tags, raw_rulings)
    got, reasons = refresh.classify(_fp(_edit(raw_cards, card, fn), raw_tags, raw_rulings), base)
    assert got == tier and reasons
    assert refresh.TIER_EXIT[tier] == code


def test_classify_card_added_or_missing_fingerprint(raw_cards, raw_tags, raw_rulings):
    base = _fp(raw_cards[1:], raw_tags, raw_rulings)
    fp = _fp(raw_cards, raw_tags, raw_rulings)
    assert refresh.classify(fp, base)[0] == refresh.TIER_REENCODE
    assert refresh.classify(fp, {"scryfall_updated_at": OLD})[0] == refresh.TIER_REENCODE  # pre-fingerprint hub
    assert refresh.classify(fp, None)[0] == refresh.TIER_REENCODE


@pytest.mark.parametrize(("label", "card", "fn", "tier", "code"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_dry_run_exit_codes(tier_setup, raw_cards, label, card, fn, tier, code):
    tier_setup.use(_edit(raw_cards, card, fn))
    assert refresh.main(["--dry-run"]) == code
    assert tier_setup.calls == []  # dry-run never publishes


def test_price_only_change_is_noop(tier_setup, raw_cards):
    tier_setup.use(_edit(raw_cards, "Sol Ring", _price_only))
    assert refresh.main([]) == 0
    assert tier_setup.calls == []  # no upload, no tag, no encode


def test_legality_change_is_metadata_only(tier_setup, raw_cards):
    tier_setup.use(_edit(raw_cards, "Sol Ring", _banned))
    assert refresh.main(["--build-image"]) == 0
    assert "upload_data" in tier_setup.calls and "build_image" in tier_setup.calls
    assert "verify_embeddings" in tier_setup.calls  # published embeddings still aligned
    assert "launch_encode_job" not in tier_setup.calls and "eval_smoke" not in tier_setup.calls
    assert tier_setup.calls.count("ensure_tag") == 2  # old main + new snapshot


def test_oracle_text_change_reencodes(tier_setup, raw_cards):
    tier_setup.use(_edit(raw_cards, "Lightning Bolt", _errata))
    assert refresh.main([]) == 0
    assert tier_setup.calls == ["ensure_tag", "upload_data", "launch_encode_job", "wait_for_job",
                                "verify_embeddings", "download_late_embeddings", "eval_smoke", "ensure_tag"]


def test_force_turns_noop_into_reencode(tier_setup, raw_cards):
    tier_setup.use(raw_cards)
    assert refresh.main(["--force", "--skip-eval"]) == 0
    assert "launch_encode_job" in tier_setup.calls


def test_ingest_local_backfills_fingerprint(tmp_path, monkeypatch):
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"scryfall_updated_at": NEW, "card_count": 34_000}))
    fp = {"text_hash": "t", "gameplay_hash": "g", "aux_hash": "a", "oracle_ids_sha256": "o", "fingerprint_version": 1}
    monkeypatch.setattr(refresh.config, "MANIFEST_JSON", mpath)
    monkeypatch.setattr(refresh, "read_local_manifest", lambda *a, **k: json.loads(mpath.read_text()))
    monkeypatch.setattr(refresh, "sh", lambda *a, **k: None)
    monkeypatch.setattr(refresh, "local_fingerprint", lambda: fp)
    manifest, got = refresh.ingest_local(force=False)
    assert got == fp and json.loads(mpath.read_text())["text_hash"] == "t" and manifest["aux_hash"] == "a"


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


def test_oracle_ids_digest_matches_search_engine(tmp_path):
    import polars as pl

    from mtg_search.mlateon import oracle_ids_digest

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
