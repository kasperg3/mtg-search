"""get_deck: decklist parsing, format rules, MTGJSON precon table, and the tool on the fixture cards (no network)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from mtg_search import decks, precons, server
from mtg_search.decks import parse_decklist

FIXTURES = Path(__file__).parent / "fixtures"
DECK_JSON = json.loads((FIXTURES / "mtgjson_deck.json").read_text())

ATRAXA_99 = ["Counterspell", "Llanowar Elves", "Sol Ring", "Rhystic Study", "Thoughtseize", "Kitchen Finks",
             "Spectral Procession", "Ornithopter", "Mana Leak", "Cryptic Command", "Jace, the Mind Sculptor",
             "Delver of Secrets", "Viscera Seer", "Eldrazi Displacer"]  # + 85 Snow-Covered Island


def commander_list(commander="Atraxa, Praetors' Voice", main=ATRAXA_99, islands=85, header=True) -> str:
    body = [f"1 {n}" for n in main] + [f"{islands} Snow-Covered Island"]
    return (f"Commander\n1 {commander}\n\nDeck\n" if header else "") + "\n".join(body)


def sections(d: decks.Decklist) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ln in d.lines:
        out.setdefault(ln.section, {})[ln.name] = out.get(ln.section, {}).get(ln.name, 0) + ln.qty
    return out


# ------------------------------------------------------------------ parsing


def test_parse_arena_export_with_about_and_sections():
    d = parse_decklist("About\nName Mono Blue Tempo\n\nCompanion\n1 Lurrus of the Dream-Den (IKO) 226\n\n"
                       "Deck\n4 Delver of Secrets (MID) 47\n4 Counterspell (MH2) 267a\n\n"
                       "Sideboard\n2 Mana Leak (M12) 64\n1 Lurrus of the Dream-Den (IKO) 226\n")
    assert d.name == "Mono Blue Tempo" and d.sections_marked
    assert sections(d) == {"companion": {"Lurrus of the Dream-Den": 1},
                           "main": {"Delver of Secrets": 4, "Counterspell": 4},
                           "sideboard": {"Mana Leak": 2, "Lurrus of the Dream-Den": 1}}


def test_parse_mtgo_blank_line_sideboard_and_commander_first():
    sixty = "\n".join(["4 Lightning Bolt"] * 15) + "\n\n3 Mana Leak\n2 Counterspell"
    assert sections(parse_decklist(sixty))["sideboard"] == {"Mana Leak": 3, "Counterspell": 2}
    first = parse_decklist(commander_list(header=False).replace("1 Counterspell", "1 Atraxa, Praetors' Voice\n\n1 Counterspell", 1))
    assert sections(first)["commander"] == {"Atraxa, Praetors' Voice": 1}


def test_parse_archidekt_moxfield_and_other_markers():
    d = parse_decklist("1x Atraxa, Praetors' Voice (2x2) 190 [Commander{top}]\n"
                       "1x Sol Ring (cmr) 472 *F* [Ramp]\n"
                       "1x Mystic Remora (ice) 87 [Maybeboard{noDeck}{noPrice}]\n"
                       "1x Fire // Ice (dmr) 215 [Removal,Draw] ^Have,#37d67a^\n"
                       "1 Rhystic Study (PCY) 45 #draw #!staple\n"
                       "1 Thrasios, Triton Hero *CMDR*\n"
                       "SB: 2 Mana Leak\n"
                       "Counterspell x3\n")
    assert sections(d) == {"commander": {"Atraxa, Praetors' Voice": 1, "Thrasios, Triton Hero": 1},
                           "main": {"Sol Ring": 1, "Fire // Ice": 1, "Rhystic Study": 1, "Counterspell": 3},
                           "maybe": {"Mystic Remora": 1}, "sideboard": {"Mana Leak": 2}}


def test_parse_headers_comments_and_uncounted_lines():
    d = parse_decklist("// Deck by someone\nCommander: Atraxa, Praetors' Voice\nCreatures (2)\n1 Llanowar Elves\n"
                       "Delver of Secrets\nLands (36)\n36 Snow-Covered Island\n# note\n---\nTokens\n1 Tyranid\n")
    assert sections(d) == {"commander": {"Atraxa, Praetors' Voice": 1},
                           "main": {"Llanowar Elves": 1, "Delver of Secrets": 1, "Snow-Covered Island": 36}}
    assert [ln.counted for ln in d.lines] == [True, True, False, True]  # a name without a quantity is a guess
    assert d.ignored == ["---"]


def test_parse_forge_dek_and_csv():
    forge = parse_decklist("[metadata]\nName=Forge Deck\n[Commander]\n1 Atraxa, Praetors' Voice|2X2\n[Main]\n1 Sol Ring|C21|1\n")
    assert forge.name == "Forge Deck"
    assert sections(forge) == {"commander": {"Atraxa, Praetors' Voice": 1}, "main": {"Sol Ring": 1}}
    dek = parse_decklist('<?xml version="1.0"?>\n<Deck>\n<Cards CatID="1" Quantity="4" Sideboard="false" Name="Lightning Bolt" />\n'
                         '<Cards CatID="2" Quantity="2" Sideboard="true" Name="Mana Leak" />\n</Deck>')
    assert sections(dek) == {"main": {"Lightning Bolt": 4}, "sideboard": {"Mana Leak": 2}}
    csv = parse_decklist('"Count","Tradelist Count","Name","Edition"\n"4","0","Lightning Bolt","2x2"\n"2","0","Fire // Ice","dmr"\n')
    assert sections(csv) == {"main": {"Lightning Bolt": 4, "Fire // Ice": 2}}


def test_parse_rejects_huge_input():
    with pytest.raises(ValueError, match="too long"):
        parse_decklist("1 Sol Ring\n" * 2000)


def test_copy_limits(by_name, cards_df):
    rows = {r["name"]: r for r in cards_df.iter_rows(named=True)}
    assert decks.copy_limit(rows["Snow-Covered Island"], 4) is None
    assert decks.copy_limit(rows["Lightning Bolt"], 4) == 4
    assert decks.copy_limit({"doc_text": "A deck can have any number of cards named Relentless Rats."}, 1) is None
    assert decks.copy_limit({"doc_text": "A deck can have up to seven cards named Seven Dwarves."}, 4) == 7
    assert decks.copy_limit({"doc_text": "A deck can have only one card named Once More with Feeling."}, 4) == 1


def test_normalize_format():
    known = ["commander", "modern", "standardbrawl", "brawl"]
    assert decks.normalize_format("EDH", known) == "commander"
    assert decks.normalize_format("Standard Brawl", known) == "standardbrawl"
    assert decks.normalize_format(" Modern ", known) == "modern"
    assert decks.normalize_format("", known) is None
    with pytest.raises(ValueError, match="unknown format"):
        decks.normalize_format("chess", known)


# ------------------------------------------------------------------ the tool on fixture cards


@pytest.fixture()
def precon_path(tmp_path):
    p = tmp_path / "precons.parquet"
    precons.build_table([("TestProliferation_TST", DECK_JSON["data"])]).write_parquet(p)
    return p


@pytest.fixture()
def svc(cards_df, raw_tags, precon_path):
    from mtg_search.ingest import build_tags

    return server.CardService(cards_df, None, tags=build_tags(raw_tags, set(cards_df["oracle_id"])),
                              precons_path=precon_path)


def problems(out: str) -> list[str]:
    return [ln[5:] for ln in out.splitlines() if ln.startswith("   - ")]


def test_valid_commander_deck(svc):
    out = svc.get_deck(commander_list())
    assert out.splitlines()[1] == ("format: Commander (detected: a commander and 100 cards) · 100 cards · "
                                   "colour identity WUBG")
    assert "problems: none found" in out
    assert "Game Changers (1): Rhystic Study" in out
    assert "Atraxa, Praetors' Voice — {G}{W}{U}{B}" in out  # the commander in full
    assert "1 Rhystic Study {2}{U} · Enchantment [Game Changer]" in out
    assert "85 Snow-Covered Island · Basic Snow Land — Island" in out
    assert out.endswith('call search_cards with color_identity="WUBG", format="commander"; every card above is already in it.')


def test_commander_violations(svc):
    main = [*ATRAXA_99, "Lightning Bolt", "Sol Ring", "Phyrexian Broodstar"]
    got = problems(svc.get_deck(commander_list(main=main)))
    assert got == ["103 cards (commander included); Commander needs exactly 100",
                   "outside the colour identity WUBG (1): Lightning Bolt (R)",
                   "Sol Ring: 2 copies; at most 1",
                   "not legal in Commander (1): Phyrexian Broodstar"]


def test_commander_pairs_and_eligibility(svc):
    def cmd_problems(text, fmt=None):
        return [p for p in problems(svc.get_deck(text, format=fmt)) if "cards (commander included)" not in p
                and "outside the colour identity" not in p]

    pair = "Commander\n1 {}\n1 {}\n\nDeck\n98 Snow-Covered Island"
    assert cmd_problems(pair.format("Wilson, Refined Grizzly", "Raised by Giants")) == []
    assert cmd_problems(pair.format("Thrasios, Triton Hero", "Sophina, Spearsage Deserter")) == [
        "Thrasios, Triton Hero and Sophina, Spearsage Deserter cannot both be commanders (partner / friends forever)"]
    one = "Commander\n1 {}\n\nDeck\n99 Snow-Covered Island"
    assert cmd_problems(one.format("Clara Oswald")) == []  # Doctor's companion can lead alone
    assert cmd_problems(one.format("Teferi, Temporal Archmage")) == []  # "can be your commander"
    assert cmd_problems(one.format("Jace, the Mind Sculptor")) == [
        "Jace, the Mind Sculptor cannot be a commander (not a legendary creature and not marked as able to be one)"]
    assert cmd_problems(one.format("Jace, the Mind Sculptor"), "brawl") == []  # planeswalkers lead in Brawl


def test_commander_from_sideboard_and_unmarked(svc):
    mtgo = commander_list(header=False) + "\n\n1 Atraxa, Praetors' Voice"
    out = svc.get_deck(mtgo)
    assert "detected: 100 cards with the commander in the sideboard" in out and "problems: none found" in out
    assert "commander read from the sideboard (MTGO export)" in out
    flat = svc.get_deck(commander_list(header=False) + "\n1 Atraxa, Praetors' Voice")
    assert problems(flat) == ["no commander marked: put it under a line that says Commander"]


def test_constructed_rules(svc):
    text = ("Deck\n4 Lightning Bolt\n4 Gitaxian Probe\n1 Sol Ring\n4 Counterspell\n46 Snow-Covered Island\n\n"
            "Sideboard\n1 Lightning Bolt\n15 Snow-Covered Island\n")
    out = svc.get_deck(text, format="modern")
    assert out.splitlines()[1] == "format: Modern (given) · main 59 · sideboard 16"
    assert problems(out) == ["main deck has 59 cards; at least 60", "sideboard has 16 cards; at most 15",
                             "Lightning Bolt: 5 copies (main 4, sideboard 1); at most 4",
                             "banned in Modern (1): Gitaxian Probe", "not legal in Modern (1): Sol Ring"]
    vintage = problems(svc.get_deck("2 Sol Ring\n58 Snow-Covered Island", format="vintage"))
    assert vintage == ["Sol Ring: 2 copies; restricted in Vintage, at most 1"]


def test_constructed_without_format_lists_legal_formats(svc):
    out = svc.get_deck("4 Lightning Bolt\n4 Counterspell\n52 Snow-Covered Island\n\n2 Mana Leak")
    assert "format: 60-card constructed (detected from the deck size) · main 60 · sideboard 2" in out
    assert "every card is legal in modern, legacy, vintage, pauper · not in standard (4 cards), pioneer (3 cards)" in out
    assert "To find cards" not in out  # no format, no search hint


def test_companion_counts_in_a_60_card_sideboard(svc):
    text = "Companion\n1 Kitchen Finks\n\nDeck\n60 Snow-Covered Island\n\nSideboard\n15 Snow-Covered Island"
    assert problems(svc.get_deck(text, format="modern")) == ["sideboard has 16 cards; at most 15"]
    arena = "Companion\n1 Kitchen Finks\n\nDeck\n60 Snow-Covered Island\n\nSideboard\n14 Snow-Covered Island\n1 Kitchen Finks"
    assert problems(svc.get_deck(arena, format="modern")) == []  # Arena lists it in both: counted once


def test_names_read_fuzzily_unresolved_and_ignored(svc):
    out = svc.get_deck("Deck\n4 Lightnig Bolt\n4 Zzqxv Wwkj\nDelver of Secrets\nnot a card line\n52 Snow-Covered Island")
    assert 'read (1): "Lightnig Bolt" as Lightning Bolt' in out
    assert 'not found (1): "Zzqxv Wwkj" (line 3)' in out and "Fix these names and call again." in out
    assert 'ignored lines (1): "not a card line"' in out
    assert "main 61" in out  # the unresolved 4 still count towards the size
    assert "cards that were not found count towards the deck size only" in out


def test_precon_by_name_commander_and_url(svc):
    by_name = svc.get_deck("Test Proliferation")
    assert by_name.splitlines()[0] == "deck: Test Proliferation · Commander Deck · TST · 2026-01-02 · list from MTGJSON"
    assert "format: Commander (from the deck type) · 100 cards · colour identity WUBG" in by_name
    assert "problems: none found" in by_name and "85 Snow-Covered Island" in by_name  # foil + non-foil merged
    assert svc.get_deck("my atraxa precon") == by_name
    assert svc.get_deck("Test Prolifferation") == by_name  # fuzzy
    assert svc.get_deck("Test Proliferation Atraxa, Praetors' Voice") == by_name  # EXP-020: name + commander
    assert svc.get_deck("https://mtgjson.com/api/v5/decks/TestProliferation_TST.json") == by_name
    assert "problems (1):" in svc.get_deck("Test Proliferation", format="modern")


def test_precon_name_starting_with_a_number(cards_df, tmp_path):
    p = tmp_path / "precons.parquet"
    precons.build_table([("Twenty_TST", {**DECK_JSON["data"], "name": "20 Ways to Prolif"})]).write_parquet(p)
    s = server.CardService(cards_df, None, precons_path=p)
    assert s.get_deck("20 Ways to Prolif").startswith("deck: 20 Ways to Prolif · Commander Deck")
    assert s.get_deck("4 Lightning Bolt").startswith("deck: pasted list")


def test_precon_by_another_legend_in_the_deck(cards_df, tmp_path):
    # EDHREC may file a precon under a legend that MTGJSON lists in the 99 (World Shaper: Szarel vs Hearthhull)
    oid = dict(zip(cards_df["name"].to_list(), cards_df["oracle_id"].to_list(), strict=True))
    data = {**DECK_JSON["data"], "name": "Swapped Faces",
            "commander": [{"name": "Thrasios, Triton Hero", "count": 1,
                           "identifiers": {"scryfallOracleId": oid["Thrasios, Triton Hero"]}}],
            "mainBoard": DECK_JSON["data"]["commander"] + DECK_JSON["data"]["mainBoard"]}
    p = tmp_path / "precons.parquet"
    precons.build_table([("Swapped_TST", data)]).write_parquet(p)
    s = server.CardService(cards_df, None, precons_path=p)
    for q in ("Atraxa, Praetors' Voice", "atraxa", "Thrasios"):
        assert s.get_deck(q).startswith("deck: Swapped Faces"), q
    assert s.get_deck("Praetors").startswith('No precon named "Praetors"')  # a legend matches by its name only


def test_precon_not_found_sites_and_missing_table(svc, cards_df):
    assert svc.get_deck("Zzqx Wvut").startswith('No precon named "Zzqx Wvut"')
    assert "Sol Ring is a card: get_card shows one card." in svc.get_deck("sol ring")
    site = svc.get_deck("https://www.moxfield.com/decks/abc123")  # no fetcher: the benchmark, tests
    assert site.startswith("This server does not open deck links here. Export the deck from moxfield.com as text")
    bare = server.CardService(cards_df, None)
    assert bare.get_deck("Test Proliferation").startswith("Official precon lists are not in this card index.")
    with pytest.raises(ValueError, match="give a precon name or a decklist"):
        svc.get_deck("  ")
    with pytest.raises(ValueError, match="unknown format"):
        svc.get_deck("1 Sol Ring", format="chess")


def test_get_deck_tool_surface(svc):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = server.build_mcp(svc)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    t = tools["get_deck"]
    assert set(t.input_schema["properties"]) == {"deck", "format"} and t.input_schema["required"] == ["deck"]
    assert not (t.meta or {}).get("anthropic/alwaysLoad")  # deferred, like find_similar
    assert t.description.startswith("Before adding, cutting or swapping cards in a deck the user names, links or pastes")
    assert t.annotations.open_world_hint is False  # no fetcher, no links opened
    assert "It does not suggest cards: call search_cards" in t.description
    ok = asyncio.run(mcp.call_tool("get_deck", {"deck": "Test Proliferation"}))
    assert not ok.is_error and ok.content[0].text.startswith("deck: Test Proliferation")
    with pytest.raises(ToolError, match="unknown format"):
        asyncio.run(mcp.call_tool("get_deck", {"deck": "1 Sol Ring", "format": "chess"}))
    three = {t.name for t in asyncio.run(server.build_mcp(svc, {"omit": "get_deck"}).list_tools())}
    assert three == {"search_cards", "get_card", "find_similar"}


# ------------------------------------------------------------------ deck links (Moxfield, Archidekt)

MOXFIELD_JSON = {"name": "Atraxa Proliferate", "format": "commander", "boards": {
    "commanders": {"count": 1, "cards": {"a": {"quantity": 1, "card": {"name": "Atraxa, Praetors' Voice"}}}},
    "mainboard": {"count": 99, "cards": {
        **{f"m{i}": {"quantity": 1, "card": {"name": n, "prices": {"usd": 9.99}}} for i, n in enumerate(ATRAXA_99)},
        "isl": {"quantity": 85, "card": {"name": "Snow-Covered Island"}}}},
    "maybeboard": {"count": 1, "cards": {"x": {"quantity": 1, "card": {"name": "Mana Leak"}}}},
    "tokens": {"count": 1, "cards": {"t": {"quantity": 1, "card": {"name": "Spirit"}}}}}}
ARCHIDEKT_JSON = {"name": "Proliferate", "deckFormat": 3,
                  "categories": [{"name": "Commander", "includedInDeck": True}, {"name": "Ramp", "includedInDeck": True},
                                 {"name": "Maybeboard", "includedInDeck": False},
                                 {"name": "Sideboard", "includedInDeck": False}],
                  "cards": [{"quantity": 1, "categories": ["Commander"], "card": {"oracleCard": {"name": "Atraxa, Praetors' Voice"}}},
                            *({"quantity": 1, "categories": ["Ramp"], "card": {"oracleCard": {"name": n}}} for n in ATRAXA_99),
                            {"quantity": 85, "categories": ["Land"], "card": {"oracleCard": {"name": "Snow-Covered Island"}}},
                            {"quantity": 1, "categories": ["Maybeboard"], "card": {"oracleCard": {"name": "Mana Leak"}}},
                            {"quantity": 1, "categories": ["Sideboard"], "card": {"oracleCard": {"name": "Ponder"}}},
                            {"quantity": 1, "categories": ["Ramp"], "deletedAt": "2026-01-01",
                             "card": {"oracleCard": {"name": "Ponder"}}}]}


def link_client(requests: list):
    import httpx

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        if req.url.host == "api2.moxfield.com" and req.url.path == "/v3/decks/all/AbC-dEf_123":
            return httpx.Response(200, json=MOXFIELD_JSON)
        if req.url.host == "archidekt.com" and req.url.path == "/api/decks/4242/":
            return httpx.Response(200, json=ARCHIDEKT_JSON)
        if req.url.path.endswith("/429/"):
            return httpx.Response(429)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_deck_links_read_moxfield_and_archidekt(svc, cards_df, raw_tags, precon_path):
    from mtg_search import deck_links
    from mtg_search.ingest import build_tags

    reqs: list = []
    s = server.CardService(cards_df, None, tags=build_tags(raw_tags, set(cards_df["oracle_id"])),
                           precons_path=precon_path, fetch_deck=lambda u: deck_links.fetch(u, link_client(reqs)))
    mox = s.get_deck("https://moxfield.com/decks/AbC-dEf_123")
    assert mox.splitlines()[:3] == ["deck: Atraxa Proliferate · Moxfield · https://moxfield.com/decks/AbC-dEf_123",
                                    "format: Commander (from Moxfield) · 100 cards · colour identity WUBG",
                                    "problems: none found"]
    assert "maybeboard: 1 cards (not part of the deck, not checked)" in mox and "Spirit" not in mox and "9.99" not in mox
    arc = s.get_deck("https://www.archidekt.com/decks/4242/proliferate")
    assert arc.splitlines()[1] == "format: Commander (from Archidekt) · 100 cards · colour identity WUBG"
    assert "problems: none found" in arc and "sideboard (1)" in arc  # Ponder in Sideboard; the deleted copy dropped
    assert "problems (1):" in s.get_deck("https://moxfield.com/decks/AbC-dEf_123", format="modern")  # given wins
    assert [str(r.url) for r in reqs[:2]] == ["https://api2.moxfield.com/v3/decks/all/AbC-dEf_123",
                                             "https://archidekt.com/api/decks/4242/"]
    assert reqs[0].headers["user-agent"] == precons.USER_AGENT
    gone = s.get_deck("https://moxfield.com/decks/private1")
    assert gone.startswith("Moxfield did not return this deck (HTTP 404)") and "export the deck from Moxfield" in gone
    assert "HTTP 429" in s.get_deck("https://archidekt.com/decks/429/")
    other = s.get_deck("https://tappedout.net/mtg-decks/some-deck/")
    assert other.startswith("This server reads Moxfield and Archidekt deck links, not tappedout.net.")
    t = {t.name: t for t in asyncio.run(server.build_mcp(s).list_tools())}["get_deck"]
    assert t.annotations.open_world_hint is True and t.annotations.idempotent_hint is False


def test_deck_link_formats():
    from mtg_search import deck_links

    def mox(fmt, n):
        return deck_links._moxfield({"format": fmt, "boards": {"mainboard": {"cards": {
            "a": {"quantity": n, "card": {"name": "Island"}}}}}}, "u").format

    assert [mox("pauperEdh", 100), mox("historicBrawl", 100), mox("brawl", 60), mox("brawl", 100),
            mox("modern", 60)] == ["paupercommander", "brawl", "standardbrawl", "brawl", "modern"]
    assert deck_links._archidekt({"deckFormat": 7, "cards": []}, "u").format is None  # Custom
    assert deck_links.site_of("https://moxfield.com/decks/AbC-dEf_123") == "Moxfield"
    assert deck_links.site_of("https://archidekt.com/decks/1/x") == "Archidekt"
    assert deck_links.site_of("https://moxfield.com/users/someone") is None


# ------------------------------------------------------------------ MTGJSON precon table


def test_deck_record_keeps_rules_facts_only():
    rec = precons.deck_record("TestProliferation_TST", DECK_JSON["data"])
    assert rec["commanders"] == ["Atraxa, Praetors' Voice"] and rec["type"] == "Commander Deck"
    islands = [c for c in rec["cards"] if c["name"] == "Snow-Covered Island"]
    assert len(islands) == 1 and islands[0]["qty"] == 85 and islands[0]["section"] == "main"
    assert sum(c["qty"] for c in rec["cards"]) == 100  # tokens are not deck cards
    blob = json.dumps(rec)
    assert "edhrec" not in blob.lower() and "purchase" not in blob.lower() and "example.invalid/buy" not in blob
    assert set(rec["cards"][0]) == {"oracle_id", "name", "qty", "section"}
    assert precons.deck_record("x", {**DECK_JSON["data"], "type": "Secret Lair Drop"}) is None


def _archive(tmp_path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        raw = json.dumps(DECK_JSON).encode()
        info = tarfile.TarInfo("AllDeckFiles/TestProliferation_TST.json")
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def test_build_from_archive(tmp_path):
    p = tmp_path / precons.ARCHIVE
    p.write_bytes(_archive(tmp_path))
    df = precons.build_table(precons.read_archive(p))
    assert df.height == 1 and df["file"][0] == "TestProliferation_TST" and df.schema == precons.SCHEMA


def test_fetch_archive_polite_cached_and_verified(tmp_path):
    import httpx

    body = _archive(tmp_path)
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith(".sha256"):
            return httpx.Response(200, text=hashlib.sha256(body).hexdigest() + "  AllDeckFiles.tar.gz\n")
        if "if-modified-since" in req.headers:
            return httpx.Response(304)
        return httpx.Response(200, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = precons.fetch_archive(tmp_path, client=client)
    assert p.read_bytes() == body
    assert all(r.headers["user-agent"] == "mtg-search (kaspergrontved@proton.me)" for r in seen)
    assert str(seen[0].url) == "https://mtgjson.com/api/v5/AllDeckFiles.tar.gz"
    n = len(seen)
    precons.fetch_archive(tmp_path, client=client)  # fresh: no request at all
    assert len(seen) == n
    precons.fetch_archive(tmp_path, client=client, max_age_days=0)  # stale: conditional request, 304 keeps it
    assert len(seen) == n + 1 and "if-modified-since" in seen[-1].headers and p.read_bytes() == body

    bad = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="0" * 64) if r.url.path.endswith(".sha256") else httpx.Response(200, content=body)))
    with pytest.raises(RuntimeError, match="sha256"):
        precons.fetch_archive(tmp_path / "other", client=bad)
    assert not (tmp_path / "other" / precons.ARCHIVE).exists()
