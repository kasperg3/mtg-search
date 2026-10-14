"""Lexical overlap between queries and the rules text of their relevant cards.

Metrics per query (relevant = its labelled oracle_ids):
  * ``card_hit``  – share of relevant cards whose rules text contains at least one
                    of the query's content-word bigrams (main metric),
  * ``bigram``    – mean over relevant cards of |Q ∩ D| / |Q| on content bigrams,
  * ``kw``        – query tokens that are MTG keyword / ability words
                    (``keywords`` column vocabulary, e.g. "ward", "landfall") and
                    appear in ≥5% of the relevant cards' keywords.

Content words: lowercase alphanumerics (+1/+1 kept as one token), light suffix
stemming, English function words removed. Rules words like "target", "each",
"whenever" are deliberately *kept* as content words. Rules text = doc_text minus
the name/cost line, reminder text and the trailing "Keywords:" line.

Run:  uv run python eval/lexical_overlap.py [--flag 0.10] [--list]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.queries import SETS, load_queries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STOP = set(
    """a an the of to in on for with from and or but if then as at by is are be been was were
    can could may might will would should so just only also very really some any ever all-in
    i me my mine we our us you your yours they them their theirs he she his her it its this that
    these those there here what which who whom how why where do does did doing done get gets got
    want wants need needs like more most much many lot lots something things thing stuff way ways
    deck decks good great nice not no into out up over about than too even still keep keeps
    """.split()
)
TOKEN = re.compile(r"\+\d+/\+\d+|-\d+/-\d+|[a-z0-9']+")
REMINDER = re.compile(r"\([^()]*\)")


def stem(w: str) -> str:
    w = w.replace("'s", "").strip("'")
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_tokens(text: str) -> list[str]:
    toks = [stem(t) for t in TOKEN.findall(text.lower())]
    return [t for t in toks if t and t not in STOP]


def bigrams(toks: list[str]) -> set[tuple[str, str]]:
    return set(zip(toks, toks[1:]))


def rules_text(doc: str) -> str:
    lines = doc.split("\n")
    body = [ln for ln in lines[1:] if not ln.startswith("Keywords:")]
    return REMINDER.sub("", "\n".join(body))


class Overlap:
    def __init__(self) -> None:
        cards = pl.read_parquet(ROOT / "data" / "cards.parquet", columns=["oracle_id", "doc_text", "keywords"])
        self.bi = {o: bigrams(content_tokens(rules_text(d))) for o, d in zip(cards["oracle_id"], cards["doc_text"])}
        self.kws = {o: {stem(k.lower()) for k in ks} for o, ks in zip(cards["oracle_id"], cards["keywords"])}
        self.kw_vocab = {k for ks in self.kws.values() for k in ks if " " not in k}

    def score(self, query: str, relevant: list[str]) -> dict:
        toks = content_tokens(query)
        qb = bigrams(toks)
        rel = [o for o in relevant if o in self.bi]
        hits = [len(qb & self.bi[o]) for o in rel]
        card_hit = sum(h > 0 for h in hits) / len(rel) if rel and qb else 0.0
        bigram = sum(hits) / (len(rel) * len(qb)) if rel and qb else 0.0
        kw = sorted(
            t
            for t in set(toks) & self.kw_vocab
            if sum(t in self.kws[o] for o in rel) >= 0.05 * len(rel)
        )
        # the bigrams responsible, most frequent first (for rewriting)
        freq = {b: sum(b in self.bi[o] for o in rel) for b in qb}
        top = [" ".join(b) for b, n in sorted(freq.items(), key=lambda x: -x[1]) if n][:3]
        return {"card_hit": card_hit, "bigram": bigram, "kw": kw, "top_bigrams": top}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", type=float, default=0.10, help="card_hit threshold for 'too lexical'")
    ap.add_argument("--list", action="store_true", help="list every flagged paraphrase query")
    args = ap.parse_args()

    ov = Overlap()
    groups: dict[str, list[dict]] = {"original": [], "para": [], "need": [], "deckneeds": []}
    for name in ("tags", "handwritten"):
        groups["original"] += load_queries(name)
    for q in load_queries("paraphrase"):
        groups["para" if q["id"].endswith("-para") else "need"].append(q)
    if SETS["deckneeds"].exists():
        groups["deckneeds"] = load_queries("deckneeds")

    print(f"{'set':10s} {'n':>4s} {'card_hit':>9s} {'bigram':>8s} {'kw>0':>6s} {'flagged':>8s}")
    flagged = []
    for g, qs in groups.items():
        if not qs:
            continue
        s = [ov.score(q["query"], q["relevant"]) for q in qs]
        n = len(s)
        fl = [(q, x) for q, x in zip(qs, s) if x["card_hit"] > args.flag or x["kw"]]
        if g in ("para", "need"):
            flagged += fl
        print(
            f"{g:10s} {n:4d} {sum(x['card_hit'] for x in s) / n:9.3f} {sum(x['bigram'] for x in s) / n:8.3f}"
            f" {sum(bool(x['kw']) for x in s) / n:6.2f} {len(fl):8d}"
        )
    if args.list:
        for q, x in flagged:
            print(f"  {x['card_hit']:.2f} kw={x['kw']} {x['top_bigrams']}  {q['id']}: {q['query']}")


if __name__ == "__main__":
    main()
