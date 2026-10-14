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
    assert json.loads((out / "manifest.json").read_text())["schema_version"] == 1
    assert (out / "tags.parquet").exists() and (out / "rulings.parquet").exists()


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
