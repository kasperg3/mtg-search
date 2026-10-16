"""Running as an installed package (uvx / pip): where the server finds its index, and the downloads."""

import json
import threading
from pathlib import Path

import pytest

from mtg_search import artifacts, config, server


@pytest.fixture
def home(monkeypatch, tmp_path):
    """No explicit dirs, an empty cache, and fake downloads that record what they fetched."""
    monkeypatch.delenv("MTG_SEARCH_EMBEDDINGS_DIR", raising=False)
    monkeypatch.delenv("MTG_SEARCH_MODEL_DIR", raising=False)
    monkeypatch.setenv("MTG_SEARCH_HOME", str(tmp_path))
    calls = []

    def fake_download(dest, *, revision, parts):
        calls.append(("index", revision, parts))
        (Path(dest) / ".cache").mkdir(parents=True, exist_ok=True)
        (Path(dest) / "data").mkdir(parents=True, exist_ok=True)
        (Path(dest) / "data" / "manifest.json").write_text(json.dumps({"schema_version": config.SCHEMA_VERSION}))

    def fake_model(dest):
        calls.append(("model",))
        out = Path(dest) / artifacts.FP32_MODEL_DIR
        out.mkdir(parents=True, exist_ok=True)
        (out / "source.json").write_text(json.dumps({"revision": artifacts.QUERY_MODEL_REVISION}))

    monkeypatch.setattr(artifacts, "download", fake_download)
    monkeypatch.setattr(artifacts, "download_query_model", fake_model)
    return tmp_path, calls


def test_explicit_dirs_win(monkeypatch, tmp_path):
    # the Docker image and dev checkouts set these; nothing is downloaded then
    monkeypatch.setenv("MTG_SEARCH_EMBEDDINGS_DIR", str(tmp_path / "emb"))
    monkeypatch.setenv("MTG_SEARCH_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(artifacts, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    data, emb, model = server.index_paths()
    assert (emb, model) == (tmp_path / "emb", tmp_path / "model") and data == config.DATA_DIR


def test_first_start_downloads_the_pinned_snapshot_and_the_encoder_once(home):
    base, calls = home
    data, emb, model = server.index_paths()
    root = base / config.INDEX_REVISION
    assert calls == [("model",), ("index", config.INDEX_REVISION, ("data", "embeddings"))]
    assert (data, emb, model) == (root / "data", root / config.EMBEDDINGS_SUBDIR, base / config.MODEL_SUBDIR)
    assert (root / artifacts.COMPLETE).exists() and not (root / ".cache").exists()
    server.index_paths()
    assert len(calls) == 2  # complete: nothing downloaded again


def test_an_update_serves_the_old_snapshot_and_fetches_the_new_one_in_the_background(home, monkeypatch):
    base, calls = home
    artifacts.ensure_index("scryfall-20260101000000", base)  # what an earlier version downloaded
    calls.clear()
    started = []

    class SyncThread:
        def __init__(self, target, args, daemon):
            started.append(args)
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(threading, "Thread", SyncThread)
    data, emb, model = server.index_paths(revision="scryfall-20261231000000")
    assert data == base / "scryfall-20260101000000" / "data"  # served at once from what is here
    assert started == [("scryfall-20261231000000", base)]  # the new snapshot, fetched for next time
    assert calls == [("index", "scryfall-20261231000000", ("data", "embeddings"))]  # the encoder is shared


def test_an_old_snapshot_in_another_format_is_not_served(home):
    base, calls = home
    artifacts.ensure_index("scryfall-20260101000000", base)
    (base / "scryfall-20260101000000" / "data" / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    assert artifacts.complete_revisions(base) == []


def test_interrupted_download_is_resumed(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "download_query_model", lambda dest: None)
    monkeypatch.setattr(artifacts, "download", lambda dest, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        artifacts.ensure_index(home=tmp_path)
    assert not (tmp_path / config.INDEX_REVISION / artifacts.COMPLETE).exists()  # the next start downloads again


def test_default_cache_location(monkeypatch, tmp_path):
    monkeypatch.delenv("MTG_SEARCH_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert config.INDEX_REVISION.startswith("scryfall-")
    assert config.index_base() == tmp_path / "mtg-search"
    assert config.index_home() == tmp_path / "mtg-search" / config.INDEX_REVISION
