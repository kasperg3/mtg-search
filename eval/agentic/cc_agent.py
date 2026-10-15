"""EXP-013: an agent with search infrastructure vs one without vs all cards in context (Claude Code).

    uv run python eval/agentic/cc_agent.py --run cc-main --conds A,G --workers 4
    uv run python eval/agentic/cc_agent.py --run cc-main --conds I --ids <task ids> --workers 1

Same tasks (``tasks.yaml`` plus the ``tasks_known_easy.yaml`` stratum) and final-answer protocol as
EXP-006: up to 5 tool calls, then a ranked list of at most 20 card names. The agent is headless Claude
Code (``claude -p``, Sonnet 5.5 with the 1M context window, subscription, no API key), isolated as in
the precon bridge (:mod:`bench.precon.harness.claude_code`): empty temp dir, no built-in tools (no web,
files or shell), and only our MCP proxy (A, G) or nothing at all (I). The ``init`` event of every
session is checked.

  A  search infrastructure: ``search(query, filters?)`` = mLateOn + metadata filters, top 15
     (EXP-006 condition A).
  G  no search infrastructure: ``grep(pattern, offset?)``, a case-insensitive regex over the plain
     card text, matches in file order (alphabetical), 15 per call, no ranking, no filter parameters.
  A0 / G0  (EXP-018, prompt ablation) A and G with identical minimal prompts: the task protocol only, no
     search advice, tool descriptions limited to mechanics (no examples, no file layout), and G0's grep
     matched truly line by line.
  I  no tools, everything in context: the plain card text of every card the task allows, in the
     system prompt. All cards are ~2.2M tokens, more than one 1M context, so the file is read in
     alphabetical parts of at most SHARD_CHARS; each part returns up to 20 candidates and, with more
     than one part, a merge call ranks the candidates (with their full text) into the final list.

Task constraints (e.g. colour identity) are applied by the environment in every condition: to every
search (A, G) or to the card file (I). "The card text" is the same ``doc_text`` everywhere: the
text mLateOn indexes, grep searches and I reads.

Every episode records turns, wall time and tokens (input, cache writes, cache reads, output,
thinking), summed over all sessions of the episode, from Claude Code's per-model usage.
Outputs: ``runs/<run>/episodes.jsonl`` and ``runs/<run>/transcripts/`` (as agent.py; resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from agent import MAX_FINAL, MAX_SEARCHES, SYSTEM, SYSTEM_GREP, TOOLS, TOOLS_GREP, constraints_text, format_results, user_message  # noqa: E402

from bench.precon.harness.claude_code import ClaudeCode  # noqa: E402

MODEL = "claude-sonnet-5-5[1m]"
SERVER = "cards"
SHARD_CHARS = 1_800_000  # ~0.5M tokens per part; the whole card file is ~7.5M characters
CC_CONDS = ("A", "G", "I", "A0", "G0")
TASK_FILES = ("tasks.yaml", "tasks_known_easy.yaml")
SHARD_DIR = HERE / ".cache" / "cc_shards"  # I's system prompts (card-file parts), git-ignored and not synced

SYSTEM_I = """You are a deck-building search assistant for Magic: The Gathering. A player describes what they need; you find real cards for it in the card file below.

{scope}Each card is listed with its name and mana cost, type line, rules text, power/toughness and keywords; cards are separated by blank lines.

Pick the cards in this file that genuinely satisfy the player's request, as many as fit, best first. Only cards listed in this file count; copy their names exactly. If the request is for one specific card, put your best guess first. Reply with only a JSON array of up to 20 card names{empty}.

<card_file>
{cards}
</card_file>"""

SYSTEM_I_MERGE = """You are a deck-building search assistant for Magic: The Gathering. A player describes what they need; you find real cards for it.

The full card file{constraint} was too long to read at once, so it was read in {k} parts and up to 20 candidate cards were picked from each part. All {n} candidates are listed below with their full text (name and mana cost, type line, rules text, power/toughness, keywords; separated by blank lines).

Pick the candidates that genuinely satisfy the player's request, best first. Copy their names exactly. If the request is for one specific card, put your best guess first. Reply with only a JSON array of up to 20 card names.

<candidates>
{cards}
</candidates>"""


# EXP-018: the same protocol for both tools, no advice on how to search, mechanics-only tool descriptions
SYSTEM_MIN = """You are a deck-building search assistant for Magic: The Gathering. A player describes what they need; you find real cards for it with the `{tool}` tool.

You may make at most 5 `{tool}` calls. When you are done, call `final_answer` with up to 20 card names that satisfy the request, best first. Only include cards that appeared in your `{tool}` results, and copy their names exactly. If the request is for one specific card, put your best guess first."""
CARD_FIELDS = "name, mana cost, type line, rules text, power/toughness, keywords"
FILTER_SCHEMA_MIN = {
    "color_identity": {"type": "string", "description": "Colour identity as WUBRG letters; the card's colour identity must be within them. C = colourless."},
    "colors": {"type": "string", "description": "Colours as WUBRG letters; the card must have all of them."},
    "types": {"type": "array", "items": {"type": "string"}, "description": "Card types or subtypes; the card must match every entry."},
    "cmc_min": {"type": "number", "description": "Minimum mana value."},
    "cmc_max": {"type": "number", "description": "Maximum mana value."},
    "format": {"type": "string", "description": "Format name; the card must be legal in it."},
}
TOOLS_MIN = [
    {"type": "function", "function": {
        "name": "search",
        "description": f"Ranked search over the text of every Magic card ({CARD_FIELDS}). Returns the 15 best-matching cards, each with name, mana cost, type and truncated rules text.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query."},
            "filters": {"type": "object", "properties": FILTER_SCHEMA_MIN, "description": "Optional hard filters."}},
            "required": ["query"]}}},
    TOOLS[1],
]
TOOLS_GREP_MIN = [
    {"type": "function", "function": {
        "name": "grep",
        "description": f"Case-insensitive Python regular expression, matched separately against each line of every Magic card's text ({CARD_FIELDS}). Returns the number of matching cards and 15 of them in file order (alphabetical by name), each with name, mana cost, type, truncated rules text and the matching line, from `offset` on.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression."},
            "offset": {"type": "integer", "description": "Number of matches to skip (default 0)."}},
            "required": ["pattern"]}}},
    TOOLS[1],
]
SYSTEMS = {"A": SYSTEM, "G": SYSTEM_GREP, "A0": SYSTEM_MIN.format(tool="search"), "G0": SYSTEM_MIN.format(tool="grep")}
TOOLSETS = {"A": TOOLS, "G": TOOLS_GREP, "A0": TOOLS_MIN, "G0": TOOLS_GREP_MIN}


def load_tasks(files=TASK_FILES) -> list[dict]:
    out = []
    for f in files:
        p = HERE / f
        if p.exists():
            out += yaml.safe_load(p.read_text())
    assert len({t["id"] for t in out}) == len(out)
    return out


def parse_names(text: str) -> list[str]:
    """Card names from a reply that should be a JSON array (falls back to one name per line)."""
    for m in reversed(list(re.finditer(r"\[[^\[\]]*\]", text or "", re.S))):
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return [x.strip() for x in v if x.strip()]
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", x).strip().strip('"') for x in (text or "").splitlines()]
    return [x for x in lines if x and len(x) < 120 and not x.endswith(":")]


def usage_of(result: dict) -> dict:
    """Session totals over every model Claude Code used (modelUsage)."""
    u = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "thinking": 0, "notional_usd": 0.0}
    for m in (result.get("modelUsage") or {}).values():
        u["input"] += m.get("inputTokens") or 0
        u["cache_write"] += m.get("cacheCreationInputTokens") or 0
        u["cache_read"] += m.get("cacheReadInputTokens") or 0
        u["output"] += m.get("outputTokens") or 0
        u["thinking"] += m.get("thinkingTokens") or 0
        u["notional_usd"] += m.get("costUSD") or 0.0
    return u


def add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


class Service:
    """Localhost HTTP tool backend for the per-episode MCP proxy (``bench.precon.harness.cc_proxy``).
    ``task`` in each request is ``<cond>:<task id>``; ``n`` is the call number within the episode."""

    def __init__(self, env, tasks: dict[str, dict]):
        self.env, self.tasks = env, tasks
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                cond = parse_qs(urlparse(self.path).query).get("task", ["A:"])[0].split(":", 1)[0]
                self._send({"tools": TOOLSETS[cond], "instructions": None})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                cond, tid = req["task"].split(":", 1)
                try:
                    self._send({"text": svc.call(cond, svc.tasks[tid], req["name"], req.get("args") or {}, int(req.get("n") or 0))[0]})
                except AssertionError as e:  # leak guard
                    self._send({"leak": True, "error": str(e)})
                except (ValueError, TypeError) as e:
                    self._send({"error": str(e)})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, cond: str, task: dict, name: str, args: dict, n: int) -> tuple[str, dict]:
        """-> (tool output, search record). Deterministic, so episodes are replayed from the proxy log."""
        env = self.env
        if cond in ("A", "A0") and name == "search":
            q = str(args.get("query") or "").strip()
            filt = args.get("filters") if isinstance(args.get("filters"), dict) else {}
            if not q:
                raise ValueError("empty query")
            rows = env.search("mlateon", q, filt, task.get("filters"))
            rec = {"tool": "search", "query": q, "filters": filt}
            content = format_results(env, q, n, rows, tool="search", last_note=False)
            snippets = []
        elif cond in ("G", "G0") and name == "grep":
            q = str(args.get("pattern") or "")
            if not q.strip():
                raise ValueError("empty pattern")
            offset = max(0, int(args.get("offset") or 0))
            n_match, rows, snippets = env.grep(q, task.get("filters"), offset=offset, per_line=cond == "G0")
            rec = {"tool": "grep", "query": q, "filters": {}, "offset": offset, "n_match": n_match}
            content = format_results(env, q, n, rows, snippets, n_match, "grep", offset, last_note=False)
        else:
            raise ValueError(f"unknown tool {name}")
        env.assert_no_leak(content, rows, [*snippets, q])
        rec.update(rows=rows, oids=[env.oids[r] for r in rows])
        return content, rec

    def close(self) -> None:
        self.httpd.shutdown()


class Runner:
    def __init__(self, env, tasks: list[dict], run_dir: Path, model: str):
        self.env, self.run_dir = env, run_dir
        self.tasks = {t["id"]: t for t in tasks}
        self.cc = ClaudeCode(model=model, timeout=1800, retries=2, backoff=60)
        self.svc = Service(env, self.tasks)
        (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        self.wlock = threading.Lock()
        self.docs = env.cards["doc_text"].to_list()

    # -------------------------------------------------------------- A / G: tools over MCP

    def tool_episode(self, task: dict, cond: str, rep: int) -> dict:
        tpath = self.run_dir / "transcripts" / f"{task['id']}__{cond}__r{rep}.jsonl"
        system = SYSTEMS[cond]
        user = user_message(task)
        with open(tpath, "w") as f:
            f.write(json.dumps({"event": "config", "task": task["id"], "cond": cond, "llm": self.cc.config(),
                                "system": system, "user": user}, ensure_ascii=False) + "\n")
        mcp = {"type": "stdio", "command": sys.executable, "args": ["-m", "bench.precon.harness.cc_proxy"],
               "env": {"PYTHONPATH": str(ROOT), "CC_TOOL_URL": self.svc.url, "CC_TASK": f"{cond}:{task['id']}",
                       "CC_MAX_CALLS": str(MAX_SEARCHES), "CC_LOG": str(tpath), "CC_SERVER_NAME": SERVER}}
        t0 = time.time()
        result = self.cc._retrying(system, user, tpath, mcp)
        wall = time.time() - t0
        log = [json.loads(x) for x in tpath.read_text().splitlines()]
        last = max(i for i, x in enumerate(log) if x.get("event") == "attempt")
        proxy = [x for x in log[last:] if x.get("event") in ("tool_call", "final_answer", "over_budget", "leak")]
        if any(x["event"] == "leak" for x in proxy):
            raise AssertionError(f"leak guard fired in {tpath.name}")
        searches = []
        for x in proxy:
            if x["event"] != "tool_call":
                continue
            try:
                rec = self.svc.call(cond, task, x["tool"], x.get("args") or {}, x["n"])[1]
            except (ValueError, TypeError) as e:
                rec = {"tool": x["tool"], "query": str((x.get("args") or {}).get("query") or (x.get("args") or {}).get("pattern") or ""),
                       "filters": {}, "rows": [], "oids": [], "error": str(e)}
            searches.append({"t": x["n"], **rec})
        final_args = next((x["args"] for x in proxy if x["event"] == "final_answer"), None)
        if final_args is not None:
            cards = final_args.get("cards")
            final = [str(c) for c in cards] if isinstance(cards, list) else parse_names(json.dumps(cards))
            stop = "final_answer"
        else:
            final, stop = parse_names(result.get("result") or ""), "text_answer"
        return self._record(task, cond, rep, searches, final[:MAX_FINAL], stop, result.get("num_turns") or 0,
                            usage_of(result), wall, n_over=sum(x["event"] == "over_budget" for x in proxy))

    # -------------------------------------------------------------- I: everything in context

    def corpus(self, task: dict) -> list[int]:
        """Rows of every card the task allows, in file order (cards.parquet is sorted by name)."""
        return [int(r) for r in np.flatnonzero(self.env.mask(task.get("filters") or {}))]

    def shards(self, rows: list[int]) -> list[list[int]]:
        out, cur, size = [], [], 0
        for r in rows:
            n = len(self.docs[r]) + 2
            if cur and size + n > SHARD_CHARS:
                out.append(cur)
                cur, size = [], 0
            cur.append(r)
            size += n
        return out + ([cur] if cur else [])

    def _write_system(self, text: str) -> Path:
        h = hashlib.sha1(text.encode()).hexdigest()[:16]
        p = SHARD_DIR / f"{h}.txt"
        if not p.exists():
            p.write_text(text)
        return p

    def context_episode(self, task: dict, rep: int) -> dict:
        tpath = self.run_dir / "transcripts" / f"{task['id']}__I__r{rep}.jsonl"
        filt = task.get("filters") or {}
        cons = constraints_text(filt) if filt else ""
        user = f"Request: {task['prompt']}\n\nReply with only a JSON array of card names, best first."
        parts = self.shards(self.corpus(task))
        k = len(parts)
        with open(tpath, "w") as f:
            f.write(json.dumps({"event": "config", "task": task["id"], "cond": "I", "llm": self.cc.config(), "user": user,
                                "n_cards": sum(map(len, parts)), "parts": [{"cards": len(p), "chars": sum(len(self.docs[r]) + 2 for r in p),
                                                                            "first": self.env.names[p[0]], "last": self.env.names[p[-1]]} for p in parts]},
                               ensure_ascii=False) + "\n")
        usage, turns, calls, cands, t0 = {}, 0, [], [], time.time()
        for i, p in enumerate(parts, 1):
            scope = ("This is part {i} of {k} of a plain-text file of all Magic cards{c}, in alphabetical order. " if k > 1 else
                     "This is a plain-text file of all Magic cards{c}, in alphabetical order. ").format(
                i=i, k=k, c=f" that meet the request's constraints ({cons})" if cons else "")
            system = SYSTEM_I.format(scope=scope, empty=" (an empty array if no card in this part fits)" if k > 1 else "",
                                     cards="\n\n".join(self.docs[r] for r in p))
            sf = self._write_system(system)
            with open(tpath, "a") as f:
                f.write(json.dumps({"event": "part", "i": i, "k": k, "system_file": sf.name, "system_chars": len(system)}) + "\n")
            tc = time.time()
            res = self.cc._retrying("", user, tpath, None, system_file=sf)
            u = usage_of(res)
            names = parse_names(res.get("result") or "")[:MAX_FINAL]
            calls.append({"part": i, "names": names, "usage": u, "wall_s": round(time.time() - tc, 1), "turns": res.get("num_turns") or 1})
            usage, turns = add_usage(usage, u), turns + (res.get("num_turns") or 1)
            cands += [n for n in names if n not in cands]
        final, stop = (calls[0]["names"] if k == 1 else []), "single_part"
        if k > 1:
            rows = [self.env.row_of[o] for o in dict.fromkeys(self.env.resolve_name(n) for n in cands) if o]
            system = SYSTEM_I_MERGE.format(constraint=f" (cards that meet the request's constraints: {cons})" if cons else "",
                                           k=k, n=len(rows), cards="\n\n".join(self.docs[r] for r in rows))
            sf = self._write_system(system)
            tc = time.time()
            res = self.cc._retrying("", user, tpath, None, system_file=sf)
            u = usage_of(res)
            final, stop = parse_names(res.get("result") or "")[:MAX_FINAL], "merged"
            calls.append({"part": "merge", "names": final, "usage": u, "wall_s": round(time.time() - tc, 1),
                          "turns": res.get("num_turns") or 1, "n_candidates": len(rows)})
            usage, turns = add_usage(usage, u), turns + (res.get("num_turns") or 1)
        rec = self._record(task, "I", rep, [], final, stop, turns, usage, time.time() - t0)
        rec.update(parts=k, calls=calls, candidate_oids=[o for o in (self.env.resolve_name(n) for n in cands) if o])
        return rec

    # -------------------------------------------------------------- common

    def _record(self, task, cond, rep, searches, final, stop, turns, usage, wall, **extra) -> dict:
        prompt_tokens = usage["input"] + usage["cache_write"] + usage["cache_read"]
        return {"task_id": task["id"], "kind": task["kind"], "cond": cond, "rep": rep, "searches": searches,
                "final": final, "final_oids": [self.env.resolve_name(n) for n in final], "stop": stop,
                "llm_calls": turns, "wall_s": round(wall, 1),
                "usage": {"prompt": prompt_tokens, "cached": usage["cache_read"], "completion": usage["output"],
                          "reasoning": usage["thinking"]},
                "usage_detail": usage, "total_tokens": prompt_tokens + usage["output"],
                "cost_usd": 0.0, "notional_usd": round(usage["notional_usd"], 4), "billing": "subscription",
                "model": self.cc.model, "cli": self.cc.version, **extra}

    def episode(self, task: dict, cond: str, rep: int) -> dict:
        return self.context_episode(task, rep) if cond == "I" else self.tool_episode(task, cond, rep)

    def run(self, tasks, conds, repeats, workers):
        out = self.run_dir / "episodes.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add((r["task_id"], r["cond"], r["rep"]))
        jobs = [(t, c, r) for r in range(repeats) for t in tasks for c in conds if (t["id"], c, r) not in done]
        print(f"{len(jobs)} episodes to run ({len(done)} already done)", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(self.episode, t, c, r): (t["id"], c, r) for t, c, r in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                key = futs[f]
                try:
                    rec = f.result()
                except Exception:
                    print(f"FAILED {key}:\n{traceback.format_exc()}", flush=True)
                    continue
                with self.wlock, open(out, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(jobs)}] {key} calls={len(rec['searches'])} final={len(rec['final'])} stop={rec['stop']} "
                      f"turns={rec['llm_calls']} tokens={rec['total_tokens']:,} wall={rec['wall_s']}s ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conds", default="A,G")
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--kinds", default=None, help="comma-separated task kinds (deck, attr, known)")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of a stratified order")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted harness changes (recorded)")
    ap.add_argument("--experiment", default="EXP-013")
    args = ap.parse_args()
    from env import git_state

    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        sys.exit("uncommitted changes in eval/agentic, src or bench/precon/harness: commit first (or --allow-dirty)")
    conds = args.conds.split(",")
    assert set(conds) <= set(CC_CONDS), conds
    tasks = load_tasks()
    if args.ids:
        want = args.ids.split(",")
        tasks = [t for t in tasks if t["id"] in want]
        assert len(tasks) == len(want), set(want) - {t["id"] for t in tasks}
    if args.kinds:
        tasks = [t for t in tasks if t["kind"] in args.kinds.split(",")]
    if args.limit:
        by: dict[str, list] = {}
        for t in tasks:
            by.setdefault(t["kind"], []).append(t)
        order = []
        while any(by.values()):
            for k in list(by):
                if by[k]:
                    order.append(by[k].pop(0))
        tasks = order[: args.limit]
    from env import Env

    env = Env(("mlateon",) if {"A", "A0"} & set(conds) else ())
    run_dir = HERE / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_dir / "config.json"
    prev = json.loads(cfg.read_text()) if cfg.exists() else {}
    cfg.write_text(json.dumps({**prev, "experiment": args.experiment, "git": git, "model": args.model, "max_calls": MAX_SEARCHES,
                               "shard_chars": SHARD_CHARS, "conds": sorted(set(prev.get("conds", [])) | set(conds)),
                               "system": SYSTEM, "system_grep": SYSTEM_GREP, "system_in_context": SYSTEM_I,
                               "system_in_context_merge": SYSTEM_I_MERGE,
                               **({"systems_min": {c: SYSTEMS[c] for c in ("A0", "G0")},
                                   "tools_min": {c: TOOLSETS[c] for c in ("A0", "G0")}} if {"A0", "G0"} & set(conds) else {})},
                              indent=1))
    runner = Runner(env, tasks, run_dir, args.model)
    try:
        runner.run(tasks, conds, args.repeats, args.workers)
    finally:
        runner.svc.close()


if __name__ == "__main__":
    main()
