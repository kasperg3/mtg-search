"""MCP server tool logic, the agent-facing surface and rendering, with a fake search backend."""

from __future__ import annotations

import asyncio
import re

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
    out = svc.search_cards(["counter target spell"], color_identity="U", format="pauper", limit=5)
    _q, k, allowed = svc.backend.calls[-1]
    expect = filters.allowed_indices(cards_df, color_identity="U", format="pauper")
    assert k == 5 and allowed.tolist() == expect.tolist()
    first = out.splitlines()[0]
    assert first.startswith('search: "counter target spell" · filters: color_identity=U, format=pauper')
    assert f"of {len(expect)} matching cards" in first
    assert "Counterspell" in names_in(out)
    assert "legal: pauper ✓" in out


def test_selective_filters_never_post_filter(svc, cards_df):
    out = svc.search_cards(["draw"], color_identity="G", mana_value_max=2, limit=50)
    got = names_in(out)
    sub = cards_df.filter(pl.col("name").is_in(got))
    assert got and sub["cmc"].max() <= 2
    assert all(set(ci) <= {"G"} for ci in sub["color_identity"].to_list())
    assert "mana_value_max=2" in out.splitlines()[0]


def test_empty_filter_result(svc):
    out = svc.search_cards(["anything"], color_identity="W", mana_value_min=20)
    assert "no card matches these filters" in out
    assert not svc.backend.calls


def test_multiple_queries_fused(svc):
    from mtg_search.fusion import rrf

    qs = ["counter target spell", "draw a card"]
    out = svc.search_cards(qs, color_identity="U", limit=4)
    calls = svc.backend.calls
    assert [c[0] for c in calls] == qs  # exactly what was sent, nothing appended
    assert all(c[1] >= 50 for c in calls)
    assert calls[0][2].tolist() == calls[1][2].tolist()  # shared filter mask
    lists = [FakeBackend(svc.cards).search(q, calls[0][1], calls[0][2]) for q in qs]
    expect = [svc.cards["name"][int(r)] for r, _ in rrf(lists, k=60, limit=4)]
    assert names_in(out) == expect
    assert out.splitlines()[0].startswith('search: "counter target spell | draw a card"')
    with pytest.raises(ValueError, match="at least one query"):
        svc.search_cards([])


def test_single_string_and_duplicates(svc):
    assert svc.search_cards("counter target spell", limit=5) == svc.search_cards(["counter target spell"], limit=5)
    svc.backend.calls.clear()
    svc.search_cards(["draw a card", " draw a card "], limit=5)  # duplicates collapse to one search
    assert [c[0] for c in svc.backend.calls] == ["draw a card"]


def test_queries_capped_at_four(svc):
    four = ["reach", "destroy target creature with flying", "draw a card", "counter target spell"]
    svc.search_cards(four, limit=5)
    assert len(svc.backend.calls) == 4
    svc.backend.calls.clear()
    with pytest.raises(ValueError, match=r"at most 4 queries per call \(got 5\)"):
        svc.search_cards([*four, "gain life"])
    assert not svc.backend.calls


def test_mcp_rejects_too_many_queries_and_accepts_a_string(svc):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = server.build_mcp(svc)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert tools["search_cards"].input_schema["properties"]["queries"]["maxItems"] == server.MAX_QUERIES
    with pytest.raises(ToolError):
        asyncio.run(mcp.call_tool("search_cards", {"queries": ["a", "b", "c", "d", "e"]}))
    assert not svc.backend.calls
    ok = asyncio.run(mcp.call_tool("search_cards", {"queries": "counter target spell", "limit": 3}))
    assert not ok.is_error and svc.backend.calls[-1][0] == "counter target spell"


def test_limit_clamped(svc):
    svc.search_cards(["x"], limit=500)
    assert svc.backend.calls[-1][1] == server.MAX_LIMIT


def test_find_similar_excludes_itself(svc, cards_df):
    out = svc.find_similar("rhystic study", limit=50)
    q, _k, allowed = svc.backend.calls[-1]
    row = cards_df["name"].to_list().index("Rhystic Study")
    assert q == cards_df["doc_text"][row]
    assert row not in allowed.tolist()
    assert out.startswith("similar to: Rhystic Study")
    assert "Rhystic Study" not in names_in(out)
    assert svc.find_similar("zzzzqqqq").startswith('No card named "zzzzqqqq"')


def test_get_card_fuzzy_full(svc):
    out = svc.get_card("fire ice")
    assert out.startswith("Fire // Ice — {1}{R} // {1}{U}")
    assert "// Fire — {1}{R} · Instant" in out and "// Ice — {1}{U} · Instant" in out
    assert "legality:" in out and "scryfall: https://scryfall.com" in out
    assert svc.get_card("delver").startswith("Delver of Secrets // Insectile Aberration")
    assert svc.get_card("Insectile Aberration").startswith("Delver of Secrets")
    assert svc.get_card("zzzzqqqq").startswith('No card named "zzzzqqqq"')


def test_get_card_rulings(svc, cards_df, raw_rulings):
    oid = raw_rulings[0]["oracle_id"]
    name = cards_df.filter(pl.col("oracle_id") == oid)["name"][0]
    out = svc.get_card(name)
    assert "rulings (" in out and raw_rulings[0]["comment"][:40] in out


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


# --------------------------------------------------------------------------- MCP layer / what the agent sees


def _listed(svc=None):
    svc = svc or server.CardService(pl.DataFrame({"name": []}), backend=None)
    return {t.name: t for t in asyncio.run(server.build_mcp(svc).list_tools())}


def test_three_read_only_tools_with_small_schemas():
    tools = _listed()
    assert set(tools) == {"search_cards", "find_similar", "get_card"}
    flt = {"color_identity", "format", "types", "mana_value_min", "mana_value_max", "limit"}
    assert set(tools["search_cards"].input_schema["properties"]) == {"queries", *flt}
    assert set(tools["find_similar"].input_schema["properties"]) == {"card_name", *flt}
    assert set(tools["get_card"].input_schema["properties"]) == {"name"}
    assert tools["search_cards"].input_schema["required"] == ["queries"]
    assert tools["find_similar"].input_schema["required"] == ["card_name"]
    assert tools["search_cards"].input_schema["properties"]["limit"]["maximum"] == server.MAX_LIMIT
    for t in tools.values():
        a = t.annotations
        assert a.read_only_hint and a.idempotent_hint and not a.destructive_hint and not a.open_world_hint, t.name
        assert t.title


def test_agent_facing_text_avoids_jargon_and_soft_metrics():
    """Everything the agent reads (instructions, descriptions, parameter descriptions) is plain MTG language:
    no "oracle" jargon and no popularity / price vocabulary."""
    import json

    text = server.INSTRUCTIONS + json.dumps([t.model_dump(mode="json") for t in _listed().values()])
    assert not re.search(r"oracle", text, re.I)
    assert not re.search(r"edhrec|price|\$|€|\btix\b|popular|salt", text, re.I)


def test_search_doc_has_query_guidance():
    for s in ('"Destroy target creature with flying."', '"Reach"', "2-4 queries", "filters, not in the queries",
              "how well their text matches"):
        assert s in server.SEARCH_DOC, s


def test_mcp_call_tool_roundtrip(svc):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = server.build_mcp(svc)

    async def go():
        ok = await mcp.call_tool("search_cards", {"queries": ["counter target spell"], "color_identity": "U",
                                                  "mana_value_max": 2, "limit": 3})
        with pytest.raises(ToolError, match="unknown format"):  # sent to the client as isError
            await mcp.call_tool("search_cards", {"queries": ["x"], "format": "notaformat"})
        return ok

    ok = asyncio.run(go())
    text = ok.content[0].text
    assert not ok.is_error and "color_identity=U, mana_value_max=2" in text


# ------------------------------------------- only rules/print facts reach the agent

SOFT_METRIC = re.compile(r"edhrec|price|\$|€|tix|rank|popular|salt", re.I)
# sentinel values planted in every column outside the server allow-list
SENTINEL_NUM = 987654.25
SENTINEL_STR = "zzleakzz"


def _poisoned(cards_df: pl.DataFrame) -> pl.DataFrame:
    """cards_df plus sentinel-valued soft-metric / print-metadata columns."""
    from mtg_search.render import ALLOWED_FIELDS

    n = cards_df.height
    df = cards_df.with_columns(
        pl.lit(987654, dtype=pl.Int64).alias("edhrec_rank"),
        pl.lit(987654, dtype=pl.Int64).alias("penny_rank"),
        pl.lit(SENTINEL_NUM).alias("edhrec_saltiness"),
        pl.lit(987654, dtype=pl.Int64).alias("inclusion_count"),
        pl.Series("prices", [{"usd": SENTINEL_NUM, "usd_foil": SENTINEL_NUM, "eur": SENTINEL_NUM,
                              "tix": SENTINEL_NUM}] * n),
    )
    for c in df.columns:  # string columns outside the allow-list that the server may not render
        if c not in ALLOWED_FIELDS and c not in ("oracle_id", "scryfall_uri") and df[c].dtype == pl.String:
            df = df.with_columns(pl.lit(SENTINEL_STR).alias(c))
    return df


def _card_text(cards_df: pl.DataFrame, tags: pl.DataFrame) -> list[str]:
    """Card- and tag-supplied text (names, oracle text, tag descriptions) that may legitimately
    contain words like "salt" or "$"; masked before the soft-metric regex is applied."""
    out: list[str] = []
    for r in cards_df.iter_rows(named=True):
        out += [r["name"], *(r.get("face_names") or []), r.get("type_line") or "", r.get("doc_text") or ""]
        for f in r.get("faces") or []:
            out += [f.get("name") or "", f.get("type_line") or "", f.get("oracle_text") or ""]
            out += (f.get("oracle_text") or "").split("\n")
    if "description" in tags.columns:
        out += [d for d in tags["description"].to_list() if d]
    return sorted({t for t in out if t}, key=len, reverse=True)


def _all_tool_outputs(svc, names: list[str]) -> list[str]:
    outs = [
        svc.search_cards(["counter target spell"], limit=50),
        svc.search_cards(["draw a card", "deal damage"], limit=50),
        svc.search_cards(["creature"], format="commander", limit=50),
    ]
    for n in names:
        outs.append(svc.get_card(n))
        outs.append(svc.find_similar(n, limit=10))
    return outs


@pytest.fixture()
def poisoned_svc(cards_df, raw_tags, raw_rulings, tmp_path):
    from mtg_search.ingest import build_rulings, build_tags

    df = _poisoned(cards_df)
    keep = set(df["oracle_id"])
    rp = tmp_path / "rulings.parquet"
    build_rulings(raw_rulings, keep).write_parquet(rp)
    return server.CardService(df, FakeBackend(df), tags=build_tags(raw_tags, keep), rulings_path=rp)


def test_server_keeps_only_allowed_columns(poisoned_svc):
    assert set(poisoned_svc.cards.columns) <= server.SERVER_COLUMNS
    for c in ("edhrec_rank", "penny_rank", "edhrec_saltiness", "inclusion_count", "prices",
              "set", "released_at", "image_uri", "set_type"):
        assert c not in poisoned_svc.cards.columns


def test_load_service_reads_only_allowed_columns(cards_df, monkeypatch, tmp_path):
    from mtg_search import config

    _poisoned(cards_df).write_parquet(tmp_path / "cards.parquet")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    read: list = []
    orig = pl.read_parquet

    def spy(path, *a, columns=None, **kw):
        read.append((str(path), columns))
        return orig(path, *a, columns=columns, **kw)

    monkeypatch.setattr(server.pl, "read_parquet", spy)
    svc, _ = server.load_service(backend=FakeBackend(cards_df))
    cols = next(c for p, c in read if p.endswith("cards.parquet"))
    assert cols is not None and set(cols) <= server.SERVER_COLUMNS
    assert set(svc.cards.columns) <= server.SERVER_COLUMNS


def test_tool_output_has_only_allowed_fields(poisoned_svc, cards_df):
    names = cards_df.filter(~pl.col("is_funny"))["name"].to_list()
    outs = _all_tool_outputs(poisoned_svc, names)
    for row in _poisoned(cards_df).iter_rows(named=True):  # render_card filters even if passed raw
        outs.append(render_card(row, rank=1, detail="full"))
    mask = _card_text(cards_df, pl.DataFrame())
    for out in outs:
        assert "987654" not in out and SENTINEL_STR not in out
        masked = out
        for t in mask:
            masked = masked.replace(t, " ")
        assert not SOFT_METRIC.search(masked), SOFT_METRIC.search(masked).group(0) + " in:\n" + out


def test_instructions_trigger_tool_use():
    """EXP-009/011: the procedural sentence is what made agents search (0% -> 40% of ADD and 86% of SWAP
    episodes with alwaysLoad); keep it unless a new benchmark run says otherwise."""
    ins = server.INSTRUCTIONS
    assert "Before you answer a deck-building request, use the tools" in ins
    for s in ("search_cards", "get_card", "find_similar", "colour identity", "even when you feel sure"):
        assert s in ins, s
    assert len(ins) <= 2048  # older Claude Code versions truncate instructions at 2,048 chars


def test_core_tools_always_load_and_descriptions_fit():
    tools = _listed()
    for name in ("search_cards", "get_card"):  # Claude Code defers MCP tools behind ToolSearch otherwise
        assert (tools[name].meta or {}).get("anthropic/alwaysLoad") is True, name
    assert not (tools["find_similar"].meta or {}).get("anthropic/alwaysLoad")
    for t in tools.values():  # Claude Code truncates descriptions at 2,048 chars before v2.1.296
        assert len(t.description or "") <= 2048, t.name


def _name_svc(names, ranks, perm):
    df = pl.DataFrame({
        "name": [names[i] for i in perm],
        "face_names": [[] for _ in perm],
        "is_funny": [False for _ in perm],
        "edhrec_rank": [ranks[i] for i in perm],
    })
    return server.CardService(df, backend=None)


def test_get_card_name_order_deterministic_and_popularity_free():
    names = ["Lightning Helix", "Lightning Bolt", "Lightning Axe", "Chain Lightning", "Bolt Bend"]
    ranks = [5, 1, 900, 2, None]  # Bolt is "most popular"; must not matter
    results = set()
    for perm in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        s = _name_svc(names, ranks, perm)
        row, alts = s.resolve_name("lightning")
        results.add((s.cards["name"][row], tuple(alts)))
        # whole-word match only when there is no prefix match
        row, _ = s.resolve_name("helix")
        assert s.cards["name"][row] == "Lightning Helix"
        # exact beats prefix
        row, _ = s.resolve_name("lightning bolt")
        assert s.cards["name"][row] == "Lightning Bolt"
        # prefix "bolt" (Bolt Bend) beats the whole-word match (Lightning Bolt, a runner-up)
        row, alts = s.resolve_name("bolt")
        assert s.cards["name"][row] == "Bolt Bend" and alts == ["Lightning Bolt"]
    # prefix matches (shortest name first, then alphabetical), then whole-word matches
    assert results == {("Lightning Axe", ("Lightning Bolt", "Lightning Helix", "Chain Lightning"))}


def test_full_name_beats_other_cards_face_name():
    df = pl.DataFrame({"name": ["Emeritus of Conflict // Lightning Bolt", "Lightning Bolt"],
                       "face_names": [["Emeritus of Conflict", "Lightning Bolt"], []],
                       "is_funny": [False, False]})
    svc = server.CardService(df, backend=None)
    assert svc.resolve_name("Lightning Bolt")[0] == 1
    assert svc.resolve_name("Emeritus of Conflict")[0] == 0
