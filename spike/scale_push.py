"""Scale test (argv[4]=start_from_scratch, default 999): push 2000 real doc embeddings replicated x17 (~34k docs, +small noise) via /update
to an embeddings-only next-plaid, time it and report index size. usage: URL embdir reps"""
import json, sys, time
import numpy as np, requests

url, d, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
sfs = int(sys.argv[4]) if len(sys.argv) > 4 else 999  # start_from_scratch
toks = np.load(f"{d}/tokens.npy").astype(np.float32); offs = np.load(f"{d}/offsets.npy")
meta = [json.loads(l) for l in open(f"{d}/cards.jsonl")]
rng = np.random.default_rng(0)
requests.delete(f"{url}/indices/scale")
requests.post(f"{url}/indices", json={"name": "scale", "config": {"nbits": 4, "start_from_scratch": sfs}}).raise_for_status()
t0 = time.time(); n = 0
for r in range(reps):
    t = toks + rng.normal(0, 0.02, toks.shape).astype(np.float32)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    for i in range(0, len(meta), 1000):
        js = range(i, min(i + 1000, len(meta)))
        body = {"documents": [{"embeddings": t[offs[j]:offs[j+1]].astype(np.float64).round(4).tolist()} for j in js],
                "metadata": [{**meta[j], "rep": r} for j in js]}
        while True:
            x = requests.post(f"{url}/indices/scale/update", json=body)
            if x.status_code != 503: x.raise_for_status(); break
            time.sleep(1)
        n += len(js)
print(f"sent {n} docs in {time.time()-t0:.0f}s")
while requests.get(f"{url}/indices/scale").json().get("num_documents", 0) < n: time.sleep(2)
while any(u.get("status") == "running" for u in requests.get(url + "/health").json().get("updates", [])): time.sleep(2)
print(f"indexed {n} docs in {time.time()-t0:.0f}s", requests.get(f"{url}/indices/scale").json())
q = np.load(f"{d}/tokens.npy")[:8].astype(np.float32).tolist()
lat = []
for _ in range(20):
    t = time.perf_counter()
    requests.post(f"{url}/indices/scale/search", json={"queries": [{"embeddings": q}], "params": {"top_k": 10}}).raise_for_status()
    lat.append((time.perf_counter() - t) * 1000)
lat.sort(); print(f"search-only p50={lat[10]:.1f}ms max={lat[-1]:.1f}ms")
lat = []
for _ in range(20):
    t = time.perf_counter()
    requests.post(f"{url}/indices/scale/search", json={"queries": [{"embeddings": q}], "params": {"top_k": 10},
                  "filter_condition": "color_identity = ?", "filter_parameters": ["B"]}).raise_for_status()
    lat.append((time.perf_counter() - t) * 1000)
lat.sort(); print(f"filtered search-only p50={lat[10]:.1f}ms")
