"""Stand-in for the stdio MCP server: spawns next-plaid-api on localhost, then answers one
query per stdin line on stdout. All next-plaid output goes to stderr (stdout is the protocol)."""
import json, subprocess, sys, time
import requests

URL = "http://127.0.0.1:8080"
t0 = time.time()
srv = subprocess.Popen(
    ["next-plaid-api", "--host", "127.0.0.1", "--port", "8080", "--index-dir", "/app/index",
     "--model", "/app/model", "--int8", "--query-length", "64"],
    stdout=sys.stderr, stderr=sys.stderr)
try:
    while True:
        try:
            if requests.get(URL + "/health", timeout=1).ok:
                break
        except requests.ConnectionError:
            if srv.poll() is not None:
                sys.exit(f"next-plaid-api exited with {srv.returncode}")
            time.sleep(0.2)
    print(f"[poc] next-plaid ready in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    for line in sys.stdin:
        q = line.strip()
        if not q:
            continue
        t = time.perf_counter()
        r = requests.post(f"{URL}/indices/cards/search_with_encoding",
                          json={"queries": [q], "params": {"top_k": 5}}).json()["results"][0]
        names = [m["name"] for m in r["metadata"]]
        print(json.dumps({"query": q, "ms": round((time.perf_counter() - t) * 1000), "top5": names}), flush=True)
finally:
    srv.terminate()
    srv.wait(10)
