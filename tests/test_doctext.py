from mtg_search.doctext import FACE_SEPARATOR, doc_text


def test_single_face(by_name):
    assert doc_text(by_name["Lightning Bolt"]) == (
        "Lightning Bolt {R}\nInstant\nLightning Bolt deals 3 damage to any target."
    )


def test_split_card_has_both_halves(by_name):
    text = doc_text(by_name["Fire // Ice"])
    fire, ice = text.split(FACE_SEPARATOR)
    assert fire.startswith("Fire {1}{R}\nInstant\n")
    assert ice.startswith("Ice {1}{U}\nInstant\n")
    assert "Draw a card." in ice


def test_transform_both_faces_with_pt_and_keywords(by_name):
    text = doc_text(by_name["Delver of Secrets // Insectile Aberration"])
    assert "Delver of Secrets {U}" in text
    assert "\n//\nInsectile Aberration\n" in text  # back face has no cost
    assert "1/1" in text and "3/2" in text
    assert text.endswith("Keywords: Flying, Transform")


def test_no_flavor_text(by_name):
    card = by_name["Delver of Secrets // Insectile Aberration"]
    flavor = card["card_faces"][1]["flavor_text"]
    assert flavor[:20] not in doc_text(card)


def test_planeswalker_loyalty(by_name):
    text = doc_text(by_name["Jace, the Mind Sculptor"])
    assert "Loyalty: 3" in text
    assert "Legendary Planeswalker — Jace" in text


def test_battle_defense(by_name):
    text = doc_text(by_name["Invasion of Ixalan // Belligerent Regisaur"])
    assert "Defense: 4" in text
    assert "Belligerent Regisaur" in text


def test_adventure_and_mdfc(by_name):
    assert "Stomp {1}{R}\nInstant — Adventure" in doc_text(by_name["Bonecrusher Giant // Stomp"])
    text = doc_text(by_name["Valki, God of Lies // Tibalt, Cosmic Impostor"])
    assert text.count(FACE_SEPARATOR) == 1
    assert "Tibalt, Cosmic Impostor {5}{B}{R}" in text
