"""Spike helper: produce tokens.npy/offsets.npy/cards.jsonl (the planned HF-artifact format)
from a running next-plaid (fp32 model == PyLate to 1e-6) for the PoC build context.
usage: python export_embeddings.py URL sample.jsonl outdir [n]"""
import json, os, sys
import numpy as np, requests

url, sample, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
cards = [json.loads(l) for l in open(sample)][:n]
os.makedirs(out, exist_ok=True)
embs = []
for i in range(0, n, 32):
    r = requests.post(url + "/encode", json={"texts": [c["doc_text"] for c in cards[i:i+32]], "input_type": "document"})
    r.raise_for_status()
    embs += [np.asarray(e, dtype=np.float16) for e in r.json()["embeddings"]]
np.save(f"{out}/tokens.npy", np.concatenate(embs))
np.save(f"{out}/offsets.npy", np.cumsum([0] + [len(e) for e in embs]).astype(np.int64))
with open(f"{out}/cards.jsonl", "w") as f:
    for c in cards:
        f.write(json.dumps({k: c[k] for k in ("oracle_id", "name", "color_identity", "cmc", "type_line")}) + "\n")
print("wrote", n, "docs")
