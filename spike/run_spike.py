"""NextPlaid feasibility spike driver (raw REST, no SDK needed).

usage:
  uv run --python 3.12 --with requests,numpy python spike/run_spike.py URL SAMPLE.jsonl CMD [index]
  CMD: index | query | push | sizes
"""
import json
import statistics
import sys
import time

import numpy as np
import requests

QUERIES = [
    "counter target spell unless its controller pays",
    "whenever another creature you control dies, draw a card",
    "sacrifice a creature: add mana",
    "whenever a creature dies, each opponent loses 1 life",
    "free sacrifice outlet",
    "destroy target creature",
    "exile target creature, its controller gains life",
    "deal 3 damage to any target",
    "search your library for a basic land and put it onto the battlefield tapped",
    "add one mana of any color",
    "whenever an opponent draws a card",
    "destroy all creatures",
    "win the game if your library is empty",
    "tap: add G",
    "create a treasure token",
    "counter target noncreature spell",
]


def post(base, path, body, ok=(200, 202)):
    r = requests.post(base + path, json=body, timeout=600)
    if r.status_code not in ok:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else None


def wait_docs(base, index, n, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{base}/indices/{index}")
        if r.ok and r.json().get("num_documents", 0) >= n:
            return r.json()
        time.sleep(1)
    raise TimeoutError(index)


def load(sample):
    return [json.loads(l) for l in open(sample)]


def metadata(c):
    # everything in metadata goes to SQLite (filters) and FTS5 (keyword search)
    return {k: c[k] for k in ("oracle_id", "name", "color_identity", "cmc", "type_line", "doc_text")}


def cmd_index(base, cards, index="cards"):
    requests.delete(f"{base}/indices/{index}")
    post(base, "/indices", {"name": index, "config": {"nbits": 4}})
    t0 = time.time()
    B = 64
    for i in range(0, len(cards), B):
        chunk = cards[i:i + B]
        while True:
            try:
                post(base, f"/indices/{index}/update_with_encoding",
                     {"documents": [c["doc_text"] for c in chunk],
                      "metadata": [metadata(c) for c in chunk]})
                break
            except RuntimeError as e:
                if "503" not in str(e) and "SERVICE_UNAVAILABLE" not in str(e):
                    raise
                time.sleep(2)
    info = wait_docs(base, index, len(cards))
    dt = time.time() - t0
    print(f"indexed {len(cards)} docs in {dt:.1f}s = {len(cards)/dt:.1f} docs/s")
    print(json.dumps(info, indent=1))


def show(res, k=5):
    out = []
    for r in res["results"]:
        out.append([f"{m['name']} ({s:.1f})" for m, s in zip(r["metadata"][:k], r["scores"][:k])])
    return out


def cmd_query(base, cards, index="cards"):
    p = {"top_k": 10}
    # warmup
    post(base, f"/indices/{index}/search_with_encoding", {"queries": ["warmup"], "params": p})
    lat = []
    for rep in range(3):
        for q in QUERIES:
            t0 = time.perf_counter()
            res = post(base, f"/indices/{index}/search_with_encoding", {"queries": [q], "params": p})
            lat.append((time.perf_counter() - t0) * 1000)
            if rep == 0:
                print(f"\nQ: {q}\n   " + "\n   ".join(show(res)[0]))
    lat.sort()
    print(f"\nsemantic search_with_encoding latency over {len(lat)} sequential queries: "
          f"p50={statistics.median(lat):.0f}ms p95={lat[int(0.95*len(lat))-1]:.0f}ms max={lat[-1]:.0f}ms")

    # encoding alone vs search alone
    enc_lat, srch_lat = [], []
    for q in QUERIES:
        t0 = time.perf_counter()
        e = post(base, "/encode", {"texts": [q], "input_type": "query"})
        enc_lat.append((time.perf_counter() - t0) * 1000)
        emb = e["embeddings"][0]
        t0 = time.perf_counter()
        post(base, f"/indices/{index}/search", {"queries": [{"embeddings": emb}], "params": p})
        srch_lat.append((time.perf_counter() - t0) * 1000)
    print(f"encode-only p50={statistics.median(enc_lat):.0f}ms; search-only (precomputed query emb) "
          f"p50={statistics.median(srch_lat):.0f}ms; query tokens={len(emb)}")

    print("\n--- filtered: color_identity subset of BG (mono-B, mono-G, BG, colorless) ---")
    q = "whenever a creature dies, each opponent loses 1 life"
    res = post(base, f"/indices/{index}/search/filtered_with_encoding",
               {"queries": [q], "params": p,
                "filter_condition": "color_identity IN (?, ?, ?, ?)",
                "filter_parameters": ["B", "G", "BG", "C"]})
    print(q, "\n   " + "\n   ".join(show(res)[0]))
    res = post(base, f"/indices/{index}/search/filtered_with_encoding",
               {"queries": ["counter target spell unless its controller pays"], "params": p,
                "filter_condition": "cmc <= ? AND type_line LIKE ?",
                "filter_parameters": [2, "%Instant%"]})
    print("counter... cmc<=2 instants\n   " + "\n   ".join(show(res)[0]))

    print("\n--- keyword only (FTS5 over metadata) ---")
    res = post(base, f"/indices/{index}/search", {"text_query": ["unless its controller pays"], "params": p})
    print("\n   ".join(show(res)[0]))
    print("\n--- hybrid (semantic+FTS, relative_score alpha .75 / rrf) ---")
    emb = post(base, "/encode", {"texts": ["sacrifice a creature: add mana"], "input_type": "query"})["embeddings"][0]
    for fusion in ("relative_score", "rrf"):
        res = post(base, f"/indices/{index}/search",
                   {"queries": [{"embeddings": emb}], "text_query": ["sacrifice creature add"],
                    "alpha": 0.75, "fusion": fusion, "params": p})
        print(fusion, "\n   " + "\n   ".join(show(res)[0]))


def cmd_push(base, cards, index="cards_pre"):
    """Simulate GPU-precomputed embeddings: encode via /encode, then push via /update (no encoding)."""
    t0 = time.time()
    embs = []
    for i in range(0, len(cards), 64):
        e = post(base, "/encode", {"texts": [c["doc_text"] for c in cards[i:i+64]], "input_type": "document"})
        embs += e["embeddings"]
    print(f"/encode {len(cards)} docs: {time.time()-t0:.1f}s")
    toks = np.concatenate([np.asarray(e, dtype=np.float16) for e in embs])
    print(f"doc tokens total={len(toks)}, avg/doc={len(toks)/len(cards):.1f}, fp16 raw={toks.nbytes/1e6:.1f}MB")
    requests.delete(f"{base}/indices/{index}")
    post(base, "/indices", {"name": index, "config": {"nbits": 4}})
    t0 = time.time()
    for i in range(0, len(cards), 256):
        body = {"documents": [{"embeddings": e} for e in embs[i:i+256]],
                "metadata": [metadata(c) for c in cards[i:i+256]]}
        while True:
            try:
                post(base, f"/indices/{index}/update", body)
                break
            except RuntimeError as e:
                if "503" not in str(e):
                    raise
                time.sleep(1)
    wait_docs(base, index, len(cards))
    print(f"/update push of precomputed embeddings: {time.time()-t0:.1f}s")
    for q in QUERIES[:3]:
        a = post(base, f"/indices/cards/search_with_encoding", {"queries": [q], "params": {"top_k": 10}})
        b = post(base, f"/indices/{index}/search_with_encoding", {"queries": [q], "params": {"top_k": 10}})
        na = [m["name"] for m in a["results"][0]["metadata"]]
        nb = [m["name"] for m in b["results"][0]["metadata"]]
        print(f"{q!r}: top10 overlap text-index vs pushed-index = {len(set(na) & set(nb))}/10")


if __name__ == "__main__":
    base, sample, cmd = sys.argv[1:4]
    extra = sys.argv[4:]
    cards = load(sample)
    {"index": cmd_index, "query": cmd_query, "push": cmd_push}[cmd](base, cards, *extra)
