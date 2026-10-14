import pytest

from mtg_search.filters import allowed_indices, allowed_oracle_ids


def names(df, **kw):
    return set(df[allowed_indices(df, **kw)]["name"])


def test_default_excludes_funny(cards_df):
    assert "Phyrexian Broodstar" not in names(cards_df)
    assert "Phyrexian Broodstar" in names(cards_df, include_funny=True)


def test_color_identity_subset(cards_df):
    ub = names(cards_df, color_identity="UB")
    assert {"Counterspell", "Thoughtseize", "Sol Ring", "Rhystic Study", "Jace, the Mind Sculptor"} <= ub
    assert "Delver of Secrets // Insectile Aberration" in ub
    assert "Lightning Bolt" not in ub
    assert "Fire // Ice" not in ub  # UR identity
    assert "Atraxa, Praetors' Voice" not in ub
    assert names(cards_df, color_identity="C") >= {"Sol Ring"}
    assert "Counterspell" not in names(cards_df, color_identity="c")


def test_colors_contains(cards_df):
    assert "Fire // Ice" in names(cards_df, colors="U")
    assert "Fire // Ice" in names(cards_df, colors="UR")
    assert "Counterspell" not in names(cards_df, colors="UR")
    assert names(cards_df, colors="C") >= {"Sol Ring"}


def test_format(cards_df):
    pauper = names(cards_df, format="pauper")
    assert {"Lightning Bolt", "Counterspell", "Fire // Ice"} <= pauper
    assert "Jace, the Mind Sculptor" not in pauper
    assert "Sol Ring" in names(cards_df, format="vintage")  # restricted counts
    with pytest.raises(ValueError):
        names(cards_df, format="notaformat")


def test_types_and_cmc(cards_df):
    creatures = names(cards_df, types=["creature"])
    assert "Llanowar Elves" in creatures and "Lightning Bolt" not in creatures
    assert "Llanowar Elves" in names(cards_df, types=["Creature", "elf"])
    assert names(cards_df, types=["legendary", "planeswalker"]) >= {"Jace, the Mind Sculptor"}
    cheap = names(cards_df, cmc_max=1)
    assert "Lightning Bolt" in cheap and "Counterspell" not in cheap
    assert "Lightning Bolt" not in names(cards_df, cmc_min=2)


def test_combined_and_oracle_ids(cards_df):
    ids = allowed_oracle_ids(cards_df, color_identity="UB", format="commander", types=["instant"])
    got = set(cards_df.filter(cards_df["oracle_id"].is_in(list(ids)))["name"])
    assert got == {"Counterspell", "Mana Leak", "Cryptic Command"}


BOROS_SUBSET = {"Lightning Bolt", "Lightning Helix", "Boros Reckoner", "Sol Ring", "Eldrazi Displacer"}


def test_colors_mode_subset_vs_includes(cards_df):
    subset = names(cards_df, colors="RW", colors_mode="subset")
    assert BOROS_SUBSET <= subset
    assert not subset & {"Counterspell", "Fire // Ice", "Kitchen Finks", "Atraxa, Praetors' Voice"}
    includes = names(cards_df, colors="RW")
    assert {"Lightning Helix", "Boros Reckoner"} <= includes
    assert not includes & {"Lightning Bolt", "Sol Ring", "Eldrazi Displacer"}
    exact = names(cards_df, colors="RW", colors_mode="exact")
    assert {"Lightning Helix", "Boros Reckoner"} <= exact and "Lightning Bolt" not in exact
    assert "Lightning Bolt" in names(cards_df, colors="R", colors_mode="exact")
    assert "Fire // Ice" not in names(cards_df, colors="R", colors_mode="exact")
    colorless = names(cards_df, colors="C", colors_mode="subset")
    assert {"Sol Ring", "Ornithopter"} <= colorless and "Lightning Bolt" not in colorless
    with pytest.raises(ValueError):
        names(cards_df, colors="R", colors_mode="nope")


def test_max_min_pips(cards_df):
    single_blue = names(cards_df, colors="U", max_pips={"U": 1})
    assert {"Mana Leak", "Delver of Secrets // Insectile Aberration", "Gitaxian Probe"} <= single_blue
    assert not single_blue & {"Counterspell", "Cryptic Command"}
    assert names(cards_df, min_pips={"u": 3}) >= {"Cryptic Command"}
    assert "Counterspell" not in names(cards_df, min_pips={"U": 3})
    # hybrid counts for both colours (devotion semantics)
    assert "Kitchen Finks" in names(cards_df, min_pips={"G": 2, "W": 2})
    assert "Kitchen Finks" not in names(cards_df, max_pips={"W": 1})
    # MDFC: Emeria's Call {4}{W}{W}{W} front, uncosted land back
    assert "Emeria's Call // Emeria, Shattered Skyclave" in names(cards_df, min_pips={"W": 3})
    assert "Emeria's Call // Emeria, Shattered Skyclave" not in names(cards_df, max_pips={"W": 2})
    with pytest.raises(ValueError):
        names(cards_df, max_pips={"Q": 1})


def test_symbol_flags(cards_df):
    assert names(cards_df, has_x=True) == {"Fireball"}
    assert "Fireball" not in names(cards_df, has_x=False)
    assert names(cards_df, has_hybrid=True) == {"Kitchen Finks", "Boros Reckoner", "Spectral Procession"}
    assert names(cards_df, has_phyrexian=True) == {"Gitaxian Probe"}
    no_phy = names(cards_df, has_phyrexian=False)
    assert "Gitaxian Probe" not in no_phy and "Counterspell" in no_phy


def test_backward_compatible_defaults(cards_df):
    assert names(cards_df, colors="U") == names(cards_df, colors="U", colors_mode="includes")
    assert len(names(cards_df)) == len(names(cards_df, max_pips={}, min_pips=None))


def test_filter_params_cover_signature():
    import inspect

    from mtg_search.filters import FILTER_PARAMS, filter_expr, has_filters

    sig = [p for p in inspect.signature(filter_expr).parameters if p != "df"]
    assert set(sig) == set(FILTER_PARAMS)
    assert all("description" in v and "type" in v for v in FILTER_PARAMS.values())
    assert not has_filters(colors_mode="subset", include_funny=True, max_pips={})
    assert has_filters(max_pips={"U": 1})
