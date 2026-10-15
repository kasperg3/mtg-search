import pytest

from mtg_search.filters import allowed_indices


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


def test_combined(cards_df):
    assert names(cards_df, color_identity="UB", format="commander", types=["instant"]) == {"Counterspell", "Mana Leak", "Cryptic Command"}


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
