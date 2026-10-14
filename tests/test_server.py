"""MCP server tool logic + rendering, with a fake retrieval backend (no NextPlaid/Docker)."""

from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import pytest

from mtg_search import filters, server
from mtg_search.render import faces_of, legality_text, render_card, role_tags


class FakeBackend:
    """Ranks allowed rows by word overlap with the query; records every call."""

    def __init__(self, cards: pl.DataFrame):
        self.docs = [set(t.lower().split()) for t in cards["doc_text"].to_list()]
        self.calls: list[tuple[str, int, np.ndarray | None]] = []

    def search(self, query, k, allowed_rows=None):
        self.calls.append((query, k, None if allowed_rows is None else np.asarray(allowed_rows)))
        rows = np.arange(len(self.docs)) if allowed_rows is None else np.asarray(allowed_rows)
        q = set(query.lower().split())
        scored = sorted(((len(q & self.docs[r]), int(r)) for r in rows), key=lambda x: (-x[0], x[1]))
        return [(r, float(s)) for s, r in scored[:k]]


@pytest.fixture()
def svc(cards_df, raw_tags, raw_rulings, tmp_path):
    from mtg_search.ingest import build_rulings, build_tags

    keep = set(cards_df["oracle_id"])
    tags = build_tags(raw_tags, keep)
    rp = tmp_path / "rulings.parquet"
    build_rulings(raw_rulings, keep).write_parquet(rp)
    return server.CardService(cards_df, FakeBackend(cards_df), tags=tags, rulings_path=rp)


def names_in(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        head = line.split(". ", 1)
        if len(head) == 2 and head[0].isdigit():
            out.append(head[1].split(" — ")[0])
    return out


def test_search_passes_only_allowed_rows(svc, cards_df):
    out = svc.search_cards("counter target spell", color_identity="U", format="pauper", limit=5)
    _q, k, allowed = svc.backend.calls[-1]
    expect = filters.allowed_indices(cards_df, color_identity="U", format="pauper")
    assert k == 5 and allowed.tolist() == expect.tolist()
    first = out.splitlines()[0]
    assert first.startswith('search: "counter target spell" · filters: color_identity=U, format=pauper')
    assert f"of {len(expect)} matching filters" in first
    assert "Counterspell" in names_in(out)
    assert "legal: pauper ✓" in out


def test_selective_filters_never_post_filter(svc, cards_df):
    out = svc.search_cards("draw", color_identity="G", cmc_max=2, limit=50)
    got = names_in(out)
    sub = cards_df.filter(pl.col("name").is_in(got))
    assert got and sub["cmc"].max() <= 2
    assert all(set(ci) <= {"G"} for ci in sub["color_identity"].to_list())


def test_empty_filter_result(svc):
    out = svc.search_cards("anything", color_identity="W", cmc_min=20)
    assert "0 cards match these filters" in out
    assert not svc.backend.calls


def test_multiple_queries_fused(svc):
    out = svc.search_cards(queries=["counter target spell", "draw a card"], limit=4)
    assert [c[0] for c in svc.backend.calls] == ["counter target spell", "draw a card"]
    assert all(c[1] >= 50 for c in svc.backend.calls)
    assert len(names_in(out)) == 4
    with pytest.raises(ValueError):
        svc.search_cards()


def test_limit_clamped(svc):
    svc.search_cards("x", limit=500)
    assert svc.backend.calls[-1][1] == server.MAX_LIMIT


def test_find_similar_excludes_itself(svc, cards_df):
    out = svc.find_similar("rhystic study", limit=50)
    q, _k, allowed = svc.backend.calls[-1]
    row = cards_df["name"].to_list().index("Rhystic Study")
    assert q == cards_df["doc_text"][row]
    assert row not in allowed.tolist()
    assert out.startswith("similar to: Rhystic Study")
    assert "Rhystic Study" not in names_in(out)


def test_get_card_fuzzy_full(svc):
    out = svc.get_card("fire ice")
    assert out.startswith("Fire // Ice — {1}{R} // {1}{U}")
    assert "// Fire — {1}{R} · Instant" in out and "// Ice — {1}{U} · Instant" in out
    assert "legalities:" in out and "scryfall: https://scryfall.com" in out
    assert svc.get_card("delver").startswith("Delver of Secrets // Insectile Aberration")
    assert svc.get_card("Insectile Aberration").startswith("Delver of Secrets")
    assert svc.get_card("zzzzqqqq").startswith('No card named "zzzzqqqq"')


def test_get_card_rulings(svc, cards_df, raw_rulings):
    oid = raw_rulings[0]["oracle_id"]
    name = cards_df.filter(pl.col("oracle_id") == oid)["name"][0]
    out = svc.get_card(name)
    assert "rulings (" in out and raw_rulings[0]["comment"][:40] in out


def test_list_tags(svc):
    out = svc.list_tags()
    assert out.startswith("tags matching '*'")
    t = svc.tags.sort("card_count", descending=True)["slug"][0]
    out = svc.list_tags(t[:4])
    assert t in out


def test_render_mdfc_land_back(cards_df):
    row = cards_df.filter(pl.col("name").str.starts_with("Emeria's Call")).row(0, named=True)
    out = render_card(row, rank=1)
    assert "[MDFC: land back face]" in out
    assert "// Emeria, Shattered Skyclave · Land" in out


def test_render_compact_truncates_and_full_does_not(cards_df):
    row = cards_df.filter(pl.col("name") == "Jace, the Mind Sculptor").row(0, named=True)
    assert "…" in render_card(row, detail="compact")
    full = render_card(row, detail="full")
    assert "…" not in full and "loyalty 3" in full


def test_render_identity_and_legality(cards_df):
    row = cards_df.filter(pl.col("name") == "Lightning Bolt").row(0, named=True)
    out = render_card(row)
    assert out.startswith("Lightning Bolt — {R} (MV 1) · Red · Instant")
    assert "legal: modern, legacy, vintage, pauper, commander" in out
    assert legality_text(row["legalities"], "standard").startswith("legal: standard ✗")
    land = cards_df.filter(pl.col("name") == "Snow-Covered Island").row(0, named=True)
    assert "identity: U" in render_card(land)


def test_faces_fallback_without_faces_column(cards_df):
    row = cards_df.filter(pl.col("name") == "Fire // Ice").row(0, named=True)
    row.pop("faces", None)
    fs = faces_of(row)
    assert [f["name"] for f in fs] == ["Fire", "Ice"] and fs[1]["mana_cost"] == "{1}{U}"


def test_role_tags_drop_generic():
    counts = {"meme": 130, "counterspell": 98, "activated-ability": 9000, "rare-thing": 3}
    assert role_tags(["meme", "counterspell", "activated-ability", "rare-thing"], counts) == ["counterspell"]


# --------------------------------------------------------------------------- MCP layer


def test_tool_schemas_follow_filter_params(svc):
    mcp = server.build_mcp(svc)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {"search_cards", "find_similar", "get_card", "list_tags"}
    for name in ("search_cards", "find_similar"):
        props = tools[name].input_schema["properties"]
        assert set(filters.FILTER_PARAMS) <= set(props)
        assert {"limit", "detail"} <= set(props)
        assert props["limit"]["maximum"] == server.MAX_LIMIT
    props = tools["search_cards"].input_schema["properties"]
    assert {"query", "queries"} <= set(props)
    assert props["colors_mode"]["enum"] == ["includes", "subset", "exact"]
    assert tools["find_similar"].input_schema["required"] == ["card_name"]
    assert "color_identity" in tools["search_cards"].description


def test_mcp_call_tool_roundtrip(svc):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = server.build_mcp(svc)

    async def go():
        ok = await mcp.call_tool("search_cards", {"query": "counter target spell", "max_pips": {"U": 1},
                                                  "colors": "U", "colors_mode": "subset", "limit": 3})
        with pytest.raises(ToolError, match="unknown format"):  # sent to the client as isError
            await mcp.call_tool("search_cards", {"query": "x", "format": "notaformat"})
        return ok

    ok = asyncio.run(go())
    text = ok.content[0].text
    assert not ok.is_error and "colors_mode=subset" in text and "max_pips" in text
    assert "Counterspell" not in names_in(text)  # {U}{U} has two blue pips
