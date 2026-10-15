"""Integration test: the built image speaks MCP over stdio (`docker run -i --rm mtg-search`).

Run with ``uv run pytest -m docker`` after ``docker/build.sh``. Image name: $MTG_SEARCH_IMAGE
(default mtg-search). Skipped when Docker or the image is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pytest

pytestmark = pytest.mark.docker

IMAGE = os.environ.get("MTG_SEARCH_IMAGE", "mtg-search")


@pytest.fixture(scope="module")
def proc():
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    if subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True).returncode != 0:
        pytest.skip(f"image {IMAGE} not built (docker/build.sh)")
    p = subprocess.Popen(["docker", "run", "-i", "--rm", IMAGE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    yield p
    if p.poll() is None:
        p.stdin.close()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


def _send(p, msg):
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()


def _recv(p, rid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = p.stdout.readline()
        assert line, f"server exited ({p.poll()}): {p.stderr.read()[-2000:]}"
        msg = json.loads(line)  # every stdout line must be a JSON-RPC message
        assert msg["jsonrpc"] == "2.0"
        if msg.get("id") == rid:
            return msg
    raise TimeoutError


def test_initialize_list_and_call(proc):
    _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "pytest", "version": "0"}}})
    init = _recv(proc, 1)["result"]
    assert init["serverInfo"]["name"] == "mtg-search"
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = {t["name"] for t in _recv(proc, 2)["result"]["tools"]}
    assert tools == {"search_cards", "find_similar", "get_card", "list_tags"}

    _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "search_cards",
        "arguments": {"query": "counter target spell unless its controller pays",
                      "format": "pauper", "color_identity": "U", "cmc_max": 2, "limit": 10}}})
    res = _recv(proc, 3)["result"]
    assert not res.get("isError"), res
    text = res["content"][0]["text"]
    assert "10 shown of" in text
    assert any(n in text for n in ("Mana Leak", "Quench", "Force Spike", "Spell Pierce"))
    assert text.count("legal: pauper ✓") == 10

    t = time.perf_counter()
    _send(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "get_card", "arguments": {"name": "fire ice"}}})
    res = _recv(proc, 4)["result"]
    assert res["content"][0]["text"].startswith("Fire // Ice")
    assert time.perf_counter() - t < 5

    _send(proc, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "search_cards", "arguments": {"query": "x", "format": "nope"}}})
    res = _recv(proc, 5)["result"]
    assert res["isError"] and "unknown format" in res["content"][0]["text"]

    proc.stdin.close()
    assert proc.wait(30) == 0


def test_image_has_no_popularity_data():
    """EDHREC rank and prices are evaluation-only: neither the image's card table nor its files contain them."""
    if subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True).returncode != 0:
        pytest.skip(f"image {IMAGE} not built (docker/build.sh)")
    code = ("import pathlib, polars as pl; c = pl.read_parquet('/app/data/cards.parquet'); "
            "bad = sorted({'edhrec_rank', 'prices', 'penny_rank', 'edhrec_saltiness'} & set(c.columns)); "
            "files = sorted(str(p) for p in pathlib.Path('/app').rglob('*popularity*')); print(bad, files)")
    out = subprocess.run(["docker", "run", "--rm", "--entrypoint", "python", IMAGE, "-c", code],
                         capture_output=True, text=True, check=True, timeout=120).stdout.strip()
    assert out == "[] []", out
