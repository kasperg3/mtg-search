import json

import polars as pl

from mtg_search import ingest


def row(df, name):
    return df.filter(pl.col("name") == name).row(0, named=True)


def test_dropped_layouts_and_memorabilia(raw_cards, cards_df):
    names = set(cards_df["name"])
    dropped = [c for c in raw_cards if not ingest.keep_card(c)]
    assert {c["layout"] for c in dropped} & {"token", "art_series"}
    assert any(c["set_type"] == "memorabilia" for c in dropped)
    assert not names & {c["name"] for c in dropped}
    assert cards_df.height == len(raw_cards) - len(dropped)
    assert cards_df["oracle_id"].is_unique().all()


def test_funny_flag(cards_df):
    funny = set(cards_df.filter("is_funny")["name"])
    assert "Phyrexian Broodstar" in funny
    # Unfinity card that is eternal-legal is not flagged
    assert not row(cards_df, "Celebr-8000")["is_funny"]
    assert not row(cards_df, "Lightning Bolt")["is_funny"]


def test_multiface_columns(cards_df):
    delver = row(cards_df, "Delver of Secrets // Insectile Aberration")
    assert delver["face_names"] == ["Delver of Secrets", "Insectile Aberration"]
    assert delver["colors"] == ["U"]
    assert delver["mana_cost"] == "{U}"
    assert delver["types"] == ["Creature"]
    assert set(delver["subtypes"]) == {"Human", "Wizard", "Insect"}
    assert delver["image_uri"].startswith("https://")

    fi = row(cards_df, "Fire // Ice")
    assert fi["mana_cost"] == "{1}{R} // {1}{U}"
    assert "Ice" in fi["doc_text"] and "Fire" in fi["doc_text"]

    valki = row(cards_df, "Valki, God of Lies // Tibalt, Cosmic Impostor")
    assert valki["types"] == ["Creature", "Planeswalker"]
    assert valki["supertypes"] == ["Legendary"]


def test_parse_type_line():
    assert ingest.parse_type_line("Legendary Snow Creature — Elf Druid") == (
        ["Legendary", "Snow"],
        ["Creature"],
        ["Elf", "Druid"],
    )
    assert ingest.parse_type_line("Instant // Sorcery — Adventure") == ([], ["Instant", "Sorcery"], ["Adventure"])


def test_legalities_struct(cards_df):
    bolt = row(cards_df, "Lightning Bolt")
    leg = bolt["legalities"]
    assert leg["modern"] == "legal" and leg["legacy"] == "legal" and leg["pauper"] == "legal"
    assert leg["standard"] == "not_legal"
    assert row(cards_df, "Sol Ring")["legalities"]["vintage"] == "restricted"


def test_tags_and_rulings(raw_cards, raw_tags, raw_rulings, cards_df):
    keep = set(cards_df["oracle_id"])
    tags = ingest.build_tags(raw_tags, keep)
    assert tags.height == len(raw_tags)
    for r in tags.iter_rows(named=True):
        assert set(r["oracle_ids"]) <= keep
        assert r["card_count"] == len(r["oracle_ids"])
    # Card-level tags are consistent with the tag table
    exploded = tags.select("slug", "oracle_ids").explode("oracle_ids").drop_nulls()
    pairs = set(zip(exploded["oracle_ids"], exploded["slug"]))
    for r in cards_df.iter_rows(named=True):
        for t in r["tags"]:
            assert (r["oracle_id"], t) in pairs

    rulings = ingest.build_rulings(raw_rulings, keep)
    assert rulings.height > 0
    assert set(rulings["oracle_id"]) <= keep
    assert rulings.schema["published_at"] == pl.Date


def test_build_from_raw(tmp_path, raw_cards, raw_tags, raw_rulings):
    import gzip

    raw = {}
    for key, rows in (("oracle_cards", raw_cards), ("oracle_tags", raw_tags), ("rulings", raw_rulings)):
        p = tmp_path / f"{key}.jsonl.gz"
        with gzip.open(p, "wt") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        raw[key] = p
    metas = {k: {"updated_at": "2026-01-01T00:00:00+00:00", "_uri": "x"} for k in raw}
    out = tmp_path / "out"
    manifest = ingest.build_from_raw(raw, metas, out)
    assert manifest["card_count"] == pl.read_parquet(out / "cards.parquet").height
    assert json.loads((out / "manifest.json").read_text())["schema_version"] == 2
    assert (out / "tags.parquet").exists() and (out / "rulings.parquet").exists()
    fp = ingest.fingerprint_dir(out)
    assert {k: manifest[k] for k in fp} == fp
    assert manifest["fingerprint_version"] == ingest.FINGERPRINT_VERSION


def test_fingerprint_columns_cover_cards_schema(cards_df):
    text, game, vol = set(ingest.TEXT_HASH_COLUMNS), set(ingest.GAMEPLAY_HASH_COLUMNS), set(ingest.VOLATILE_COLUMNS)
    assert not (game | text) & vol
    assert set(cards_df.columns) == text | game | vol, "classify every new cards.parquet column"
    assert not set(ingest.EVAL_ONLY_COLUMNS) & (text | game | vol)


def test_popularity_is_never_in_the_shipped_card_table(raw_cards, raw_tags, raw_rulings, tmp_path):
    """EDHREC rank is evaluation-only: build_from_raw splits it into eval/popularity.parquet."""
    import gzip

    raw = {}
    for name, rows in (("oracle_cards", raw_cards), ("oracle_tags", raw_tags), ("rulings", raw_rulings)):
        p = tmp_path / f"{name}.jsonl.gz"
        with gzip.open(p, "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        raw[name] = p
    metas = {k: {"updated_at": "2026-01-01T00:00:00+00:00", "_uri": f"https://x/{k}"} for k in raw}
    ingest.build_from_raw(raw, metas, tmp_path / "out")
    cards = pl.read_parquet(tmp_path / "out" / "cards.parquet")
    pop = pl.read_parquet(tmp_path / "out" / "eval" / "popularity.parquet")
    assert not {"edhrec_rank", "prices", "penny_rank"} & set(cards.columns)
    assert pop.columns == ["oracle_id", *ingest.EVAL_ONLY_COLUMNS]
    assert pop["oracle_id"].to_list() == cards["oracle_id"].to_list()
    sol = next(c for c in raw_cards if c["name"] == "Sol Ring")
    assert pop.filter(pl.col("oracle_id") == sol["oracle_id"])["edhrec_rank"].item() == sol.get("edhrec_rank")


def _fp(raw_cards, raw_tags, raw_rulings):
    cards = ingest.build_cards(raw_cards, ingest.tags_by_oracle_id(raw_tags))
    keep = set(cards["oracle_id"])
    return ingest.fingerprint(cards, ingest.build_tags(raw_tags, keep), ingest.build_rulings(raw_rulings, keep))


def _edit(raw_cards, name, **changes):
    out = []
    for c in raw_cards:
        if c["name"] == name:
            c = {**c, **changes}
        out.append(c)
    return out


def test_fingerprint_is_deterministic_and_ignores_volatile(raw_cards, raw_tags, raw_rulings):
    base = _fp(raw_cards, raw_tags, raw_rulings)
    assert base == _fp(raw_cards, raw_tags, list(reversed(raw_rulings)))
    sol = next(c for c in raw_cards if c["name"] == "Sol Ring")
    volatile = _edit(raw_cards, "Sol Ring", prices={"usd": "999.99"}, edhrec_rank=1, released_at="2099-01-01",
                     image_uris={"normal": "https://example.invalid/x.jpg"}, scryfall_uri="https://example.invalid",
                     set="zzz", rarity="mythic" if sol.get("rarity") != "mythic" else "common")
    assert _fp(volatile, raw_tags, raw_rulings) == base


def test_fingerprint_legality_change_is_gameplay_only(raw_cards, raw_tags, raw_rulings):
    base = _fp(raw_cards, raw_tags, raw_rulings)
    sol = next(c for c in raw_cards if c["name"] == "Sol Ring")
    banned = _edit(raw_cards, "Sol Ring", legalities={**sol["legalities"], "commander": "banned"})
    fp = _fp(banned, raw_tags, raw_rulings)
    assert fp["text_hash"] == base["text_hash"] and fp["oracle_ids_sha256"] == base["oracle_ids_sha256"]
    assert fp["gameplay_hash"] != base["gameplay_hash"]


def test_fingerprint_oracle_text_change_changes_text_hash(raw_cards, raw_tags, raw_rulings):
    base = _fp(raw_cards, raw_tags, raw_rulings)
    errata = _edit(raw_cards, "Lightning Bolt", oracle_text="Lightning Bolt deals 4 damage to any target.")
    fp = _fp(errata, raw_tags, raw_rulings)
    assert fp["text_hash"] != base["text_hash"] and fp["gameplay_hash"] != base["gameplay_hash"]
    assert _fp(raw_cards[1:], raw_tags, raw_rulings)["text_hash"] != base["text_hash"]  # card removed


def test_fingerprint_ruling_change_is_aux_only(raw_cards, raw_tags, raw_rulings):
    base = _fp(raw_cards, raw_tags, raw_rulings)
    rulings = [{**raw_rulings[0], "comment": "changed"}, *raw_rulings[1:]]
    fp = _fp(raw_cards, raw_tags, rulings)
    assert (fp["text_hash"], fp["gameplay_hash"]) == (base["text_hash"], base["gameplay_hash"])
    assert fp["aux_hash"] != base["aux_hash"]


def test_stamp():
    assert ingest.stamp("2026-10-14T09:01:55.948+00:00") == "20261014T090155"


def test_parse_mana_cost():
    p = ingest.parse_mana_cost
    cc = p("{1}{U}{U}{U}")
    assert (cc["pips_U"], cc["generic"], cc["has_x"], cc["has_hybrid"]) == (3, 1, False, False)
    fb = p("{X}{R}")
    assert fb["has_x"] and fb["pips_R"] == 1 and fb["generic"] == 0
    kf = p("{1}{G/W}{G/W}")
    assert kf["has_hybrid"] and kf["pips_G"] == 2 and kf["pips_W"] == 2 and not kf["has_phyrexian"]
    gp = p("{U/P}")
    assert gp["has_phyrexian"] and not gp["has_hybrid"] and gp["pips_U"] == 1
    sp = p("{2/W}{2/W}{2/W}")
    assert sp["has_hybrid"] and sp["pips_W"] == 3 and sp["generic"] == 0
    assert p("{W/U/P}")["has_hybrid"] and p("{W/U/P}")["has_phyrexian"]
    assert p("{C}{C}{2}")["pips_C"] == 2
    assert p("") == p(None) and sum(v for k, v in p("").items() if k.startswith("pips_")) == 0


def test_mana_columns_per_face(cards_df):
    fi = row(cards_df, "Fire // Ice")
    assert [f["mana_cost"] for f in fi["face_costs"]] == ["{1}{R}", "{1}{U}"]
    assert (fi["pips_R"], fi["pips_U"], fi["generic"]) == (1, 1, 1)
    emeria = row(cards_df, "Emeria's Call // Emeria, Shattered Skyclave")
    assert [f["pips_W"] for f in emeria["face_costs"]] == [3, 0]
    assert emeria["pips_W"] == 3 and emeria["generic"] == 4
    bolt = row(cards_df, "Lightning Bolt")
    assert len(bolt["face_costs"]) == 1 and bolt["pips_R"] == 1
    assert row(cards_df, "Fireball")["has_x"]
    assert row(cards_df, "Gitaxian Probe")["has_phyrexian"]


def test_deck_columns_schema(cards_df):
    s = cards_df.schema
    assert s["rarity"] == pl.String and s["partner_kind"] == pl.String
    assert s["produced_mana"] == pl.List(pl.String)
    assert "prices" not in s
    for col in ("is_permanent", "is_legendary", "can_be_commander", "reserved"):
        assert s[col] == pl.Boolean
    for col in ("power", "toughness", "loyalty", "defense"):
        assert s[col] == pl.String


def test_mdfc_land_back_face(cards_df):
    emeria = row(cards_df, "Emeria's Call // Emeria, Shattered Skyclave")
    assert [f["is_permanent"] for f in emeria["faces"]] == [False, True]
    assert [f["type_line"] for f in emeria["faces"]] == ["Sorcery", "Land"]
    assert emeria["faces"][1]["mana_cost"] == "" and "{T}: Add {W}." in emeria["faces"][1]["oracle_text"]
    assert emeria["is_permanent"] and emeria["produced_mana"] == ["W"]
    assert not emeria["can_be_commander"] and emeria["rarity"] == "mythic"

    valki = row(cards_df, "Valki, God of Lies // Tibalt, Cosmic Impostor")
    assert (valki["power"], valki["toughness"], valki["loyalty"]) == ("2", "1", None)
    assert [f["loyalty"] for f in valki["faces"]] == [None, "5"]
    assert valki["is_legendary"] and valki["can_be_commander"]

    invasion = row(cards_df, "Invasion of Ixalan // Belligerent Regisaur")
    assert invasion["defense"] == "4" and invasion["faces"][1]["power"] == "4"


def test_mana_rock(cards_df):
    sol = row(cards_df, "Sol Ring")
    assert sol["produced_mana"] == ["C"] and sol["is_permanent"] and not sol["is_legendary"]
    assert row(cards_df, "Llanowar Elves")["produced_mana"] == ["G"]
    assert row(cards_df, "Lightning Bolt")["produced_mana"] == []
    assert not row(cards_df, "Lightning Bolt")["is_permanent"]
    assert row(cards_df, "Phyrexian Broodstar")["power"] == "*"
    assert len(row(cards_df, "Lightning Bolt")["faces"]) == 1


def test_commanders_and_partners(cards_df):
    kinds = {r["name"]: r["partner_kind"] for r in cards_df.iter_rows(named=True)}
    assert kinds["Thrasios, Triton Hero"] == "partner"
    assert kinds["Wilson, Refined Grizzly"] == "choose a background"
    assert kinds["Raised by Giants"] == "background"
    assert kinds["Sophina, Spearsage Deserter"] == "friends forever"
    assert kinds["Clara Oswald"] == "doctor's companion"
    assert kinds["Atraxa, Praetors' Voice"] is None and kinds["Sol Ring"] is None
    for name in ("Thrasios, Triton Hero", "Wilson, Refined Grizzly", "Atraxa, Praetors' Voice"):
        assert row(cards_df, name)["can_be_commander"]
    # planeswalker commander via oracle text; plain legendary planeswalker / Background are not
    assert row(cards_df, "Teferi, Temporal Archmage")["can_be_commander"]
    assert not row(cards_df, "Jace, the Mind Sculptor")["can_be_commander"]
    assert not row(cards_df, "Raised by Giants")["can_be_commander"]
    assert row(cards_df, "Raised by Giants")["is_legendary"]
    # flip card: legendary back face doesn't make the front a commander
    assert not row(cards_df, "Akki Lavarunner // Tok-Tok, Volcano Born")["can_be_commander"]


def test_partner_kind_variants():
    pk = ingest.partner_kind
    base = {"type_line": "Legendary Creature — Human", "keywords": ["Partner"]}
    assert pk({**base, "oracle_text": "Partner—Survivors (You can have two commanders if both have this ability.)"}) == (
        "partner—survivors"
    )
    assert pk({**base, "keywords": ["Partner with", "Partner"], "oracle_text": "Partner with Rowan Kenrith"}) == (
        "partner with"
    )
    assert pk({"type_line": "Legendary Creature — Time Lord Doctor", "keywords": [], "oracle_text": ""}) == "doctor"
