"""End-to-end smoke test of the image over MCP stdio (raw JSON-RPC, no SDK needed).

    python docker/mcp_smoke.py [--image mtg-search] [--repeat 3]

Spawns ``docker run -i --rm <image>``, initializes, lists tools, runs the sample calls below,
prints results and per-call latency (warm = best of the repeats), and fails if any stdout line
is not a JSON-RPC message (stdout must carry nothing but MCP).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CALLS = [
    ("search_cards", {"query": "blue instants that counter a spell unless its controller pays", "format": "pauper"}),
    ("search_cards", {"query": "free way to sacrifice my creatures", "color_identity": "BG", "format": "commander"}),
    ("find_similar", {"card_name": "Rhystic Study"}),
    ("get_card", {"name": "fire ice"}),
    ("search_cards", {"query": "ramp", "color_identity": "G", "format": "pauper", "cmc_max": 2, "limit": 5}),
    ("search_cards", {"queries": ["destroy target artifact", "exile target artifact or enchantment"],
                      "colors": "RW", "colors_mode": "subset", "format": "modern", "limit": 5}),
    ("list_tags", {"prefix": "free-sac"}),
    ("search_cards", {"queries": ["Reach", "Destroy target creature with flying.",
                                  "Creatures you control gain reach until end of turn.",
                                  "Whenever a creature with flying attacks you, it deals no combat damage."],
                      "color_identity": "G", "format": "commander", "limit": 15}),
]


class StdioClient:
    def __init__(self, cmd: list[str]):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.next_id = 0

    def send(self, msg: dict) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 300) -> dict:
        self.next_id += 1
        rid = self.next_id
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        assert self.p.stdout is not None
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"server closed stdout (exit {self.p.poll()})")
            msg = json.loads(line)  # raises on any non-JSON stdout noise
            assert msg.get("jsonrpc") == "2.0", line
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]
        raise TimeoutError(method)

    def initialize(self) -> dict:
        res = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "mcp-smoke", "version": "0"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return res

    def call(self, name: str, args: dict) -> tuple[str, bool, float]:
        t = time.perf_counter()
        res = self.request("tools/call", {"name": name, "arguments": args})
        ms = (time.perf_counter() - t) * 1000
        text = "\n".join(c.get("text", "") for c in res.get("content", []))
        return text, bool(res.get("isError")), ms

    def close(self) -> int:
        assert self.p.stdin is not None
        self.p.stdin.close()
        return self.p.wait(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="mtg-search")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    name = f"mtg-smoke-{os.getpid()}"
    c = StdioClient(["docker", "run", "-i", "--rm", "--name", name, a.image])
    info = c.initialize()
    print(f"initialize: {info['serverInfo']} in {time.time() - t0:.1f}s")
    tools = c.request("tools/list")["tools"]
    print("tools:", ", ".join(t["name"] for t in tools))
    # EXP-009: the shipped surface must carry the tool-use trigger and keep the core tools always loaded
    ins = info.get("instructions") or ""
    always = sorted(t["name"] for t in tools if (t.get("_meta") or {}).get("anthropic/alwaysLoad"))
    print(f"instructions: {len(ins)} chars; alwaysLoad: {always}")
    if "Before you answer a deck-building request, use the tools" not in ins or always != ["get_card", "search_cards"]:
        print("SMOKE FAIL: instructions trigger or alwaysLoad meta missing")
        sys.exit(1)
    lat = []
    for name, args in CALLS:
        best = None
        for i in range(a.repeat):
            text, err, ms = c.call(name, args)
            if i == 0:
                first = ms
            best = ms if best is None else min(best, ms)
        lat.append((name, best))
        print(f"\n===== {name}({json.dumps(args)})  first {first:.0f} ms, warm {best:.0f} ms"
              + ("  [ERROR]" if err else ""))
        print(text)
    mem = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         capture_output=True, text=True).stdout.strip()
    print(f"container memory after the calls: {mem}")
    rc = c.close()
    print(f"\nwarm latency (best of {a.repeat}): " + ", ".join(f"{n} {ms:.0f}ms" for n, ms in lat))
    print(f"server exit code after stdin EOF: {rc}")
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
