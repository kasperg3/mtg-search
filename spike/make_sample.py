"""Build a ~2k card sample from the Scryfall oracle_cards bulk jsonl.gz for the NextPlaid spike.

usage: python make_sample.py oracle.jsonl.gz out.jsonl [n]
"""
import gzip, json, random, sys

DROP_LAYOUTS = {"token", "double_faced_token", "emblem", "art_series", "planar",
                "scheme", "vanguard", "front_card"}
MUST = {  # cards that the sanity queries should surface
    "Mana Leak", "Rhystic Study", "Counterspell", "Force Spike", "Spell Pierce", "Daze",
    "Remand", "Mana Tithe", "Syncopate", "Blood Artist", "Grim Haruspex", "Midnight Reaper",
    "Zulaport Cutthroat", "Viscera Seer", "Ashnod's Altar", "Phyrexian Altar", "Carrion Feeder",
    "Skirk Prospector", "Lightning Bolt", "Shock", "Fire // Ice", "Delver of Secrets // Insectile Aberration",
    "Smothering Tithe", "Swords to Plowshares", "Path to Exile", "Llanowar Elves", "Birds of Paradise",
    "Sol Ring", "Wrath of God", "Doom Blade", "Cultivate", "Rampant Growth", "Sakura-Tribe Elder",
    "Thassa's Oracle", "Dark Ritual", "Esper Sentinel", "Mystic Remora", "Pitiless Plunderer",
    "Bastion of Remembrance", "Goblin Bombardment", "Krark-Clan Ironworks",
}


def face_text(f):
    lines = [f"{f.get('name','')} {f.get('mana_cost','')}".strip(), f.get("type_line", "")]
    if f.get("oracle_text"):
        lines.append(f["oracle_text"])
    if "power" in f:
        lines.append(f"{f['power']}/{f['toughness']}")
    if "loyalty" in f:
        lines.append(f"Loyalty: {f['loyalty']}")
    if "defense" in f:
        lines.append(f"Defense: {f['defense']}")
    return "\n".join(l for l in lines if l)


def doc_text(c):
    faces = c.get("card_faces") or [c]
    body = "\n\n".join(face_text(f) for f in faces)
    if c.get("keywords"):
        body += "\nKeywords: " + ", ".join(c["keywords"])
    return body


def main(src, dst, n=2000):
    cards = []
    with gzip.open(src, "rt") as fh:
        for line in fh:
            c = json.loads(line)
            if c.get("layout") in DROP_LAYOUTS or c.get("set_type") == "memorabilia":
                continue
            cards.append(c)
    random.seed(0)
    must = [c for c in cards if c["name"] in MUST]
    rest = [c for c in cards if c["name"] not in MUST]
    sample = must + random.sample(rest, n - len(must))
    with open(dst, "w") as out:
        for c in sample:
            out.write(json.dumps({
                "oracle_id": c["oracle_id"], "name": c["name"],
                "color_identity": "".join(c.get("color_identity", [])) or "C",
                "cmc": c.get("cmc", 0), "type_line": c.get("type_line", ""),
                "doc_text": doc_text(c),
            }) + "\n")
    print(f"{len(cards)} eligible, wrote {len(sample)} ({len(must)} must-include)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
