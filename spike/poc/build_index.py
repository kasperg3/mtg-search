"""Build-time indexer: start next-plaid-api WITHOUT a model, push precomputed token embeddings
via POST /indices/{name}/update, wait until all docs are indexed, shut the server down.

data dir: tokens.npy (fp16 [N_tokens,128]), offsets.npy (int64 [N_docs+1]), cards.jsonl (metadata)
"""
import json, subprocess, sys, time
import numpy as np, requests

data, index_dir = sys.argv[1], sys.argv[2]
URL, NAME = "http://127.0.0.1:8080", "cards"
srv = subprocess.Popen(["next-plaid-api", "--host", "127.0.0.1", "--port", "8080", "--index-dir", index_dir])
try:
    for _ in range(120):
        try:
            if requests.get(URL + "/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            time.sleep(0.5)
    toks = np.load(f"{data}/tokens.npy").astype(np.float32)
    offs = np.load(f"{data}/offsets.npy")
    meta = [json.loads(l) for l in open(f"{data}/cards.jsonl")]
    requests.post(URL + "/indices", json={"name": NAME, "config": {"nbits": 4}}).raise_for_status()
    B = 256
    for i in range(0, len(meta), B):
        docs = [{"embeddings": toks[offs[j]:offs[j + 1]].tolist()} for j in range(i, min(i + B, len(meta)))]
        while True:
            r = requests.post(f"{URL}/indices/{NAME}/update", json={"documents": docs, "metadata": meta[i:i + B]})
            if r.status_code != 503:
                r.raise_for_status()
                break
            time.sleep(1)
    while requests.get(f"{URL}/indices/{NAME}").json().get("num_documents", 0) < len(meta):
        time.sleep(1)
    # let the async updater finish writing metadata / merged files
    while any(u.get("status") == "running" for u in requests.get(URL + "/health").json().get("updates", [])):
        time.sleep(1)
    print("indexed", requests.get(f"{URL}/indices/{NAME}").json(), flush=True)
finally:
    srv.terminate()
    srv.wait(30)
