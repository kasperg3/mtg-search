"""Agentic multi-step search with GLM-5.3 (EXP-006 protocol): run episodes over the task set.

    uv run python eval/agentic/agent.py --run <name> --conds A,G --limit 10 --repeats 1

Conditions (identical system prompt apart from the tool paragraph):
  A  search infrastructure: `search(query, filters?)`, mLateOn, top 15 per call.
  G  none (EXP-013): one tool, `grep(pattern, offset?)`, a regex over the plain card text returning the match
     count and 15 matches in file order (alphabetical by name); no ranking and no filter parameters.
Task constraints are applied by the environment in both. The EXP-006 SparseUp/BM25 conditions (B, C, D, F)
were removed with those retrievers (code up to commit 61071da; results in runs/main). The Claude Code version
of this experiment is cc_agent.py.

Outputs in eval/agentic/runs/<run>/:
  episodes.jsonl                  one structured record per (task, cond, rep)
  transcripts/<task>__<cond>__r<rep>.jsonl   every request delta and raw response
Episodes already in episodes.jsonl are skipped, so a run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from glm import GLM, MODEL, cost_usd  # noqa: E402

MAX_SEARCHES = 5
MAX_TURNS = 10
MAX_FINAL = 20
CONDITIONS = {"A": {"retriever": "mlateon"}, "G": {"retriever": "grep"}}

SYSTEM = """You are a deck-building search assistant for Magic: The Gathering. A player describes what they need; you find real cards for it with the `search` tool. It queries a card search engine over all Magic cards (it understands natural language; it is not Scryfall syntax) and returns the top 15 matches with name, mana cost, type and truncated rules text. Optional filters restrict colours, types, mana value and format.

You have a budget of 5 searches. Use them to find as many cards that genuinely satisfy the request as possible. After each search, look at what came back and decide what to search next: refine, rephrase, or explore a different angle that the results suggest. A search that returns the same cards again is wasted.

When you are done (after at most 5 searches), call `final_answer` with up to 20 card names that satisfy the request, best first. Only include cards that appeared in your search results, and copy their names exactly. If the request is for one specific card, put your best guess first."""

SYSTEM_GREP = SYSTEM.replace(
    "you find real cards for it with the `search` tool. It queries a card search engine over all Magic cards (it understands natural language; it is not Scryfall syntax) and returns the top 15 matches with name, mana cost, type and truncated rules text. Optional filters restrict colours, types, mana value and format.",
    "you find real cards for it with the `grep` tool. There is no search engine: `grep` runs a case-insensitive regular expression over a plain-text file of all Magic cards, where each card is listed with its name, mana cost, type line, rules text, power/toughness and keywords. It returns how many cards match and the first 15 matching cards in file order (alphabetical by name), each with name, mana cost, type and truncated rules text plus the matching line with the match in [[ ]]. Pass `offset` to see later matches. Matching works line by line, as in grep: `.` does not cross a line break, and `^` and `$` match at the start and end of a line. One entry of the file looks like this:\n\nSerra Angel {3}{W}{W}\nCreature — Angel\nFlying\nVigilance (Attacking doesn't cause this creature to tap.)\n4/4\nKeywords: Flying, Vigilance",
).replace("You have a budget of 5 searches.", "You have a budget of 5 grep calls (a call with an offset counts too).")
assert SYSTEM_GREP != SYSTEM and "budget of 5 grep calls" in SYSTEM_GREP

FILTER_SCHEMA = {
    "color_identity": {"type": "string", "description": "Commander colour identity, card must be within these colours, e.g. 'UB'; 'C' = colourless."},
    "colors": {"type": "string", "description": "Card colours, e.g. 'R'. Card must have all of them (may have more)."},
    "types": {"type": "array", "items": {"type": "string"}, "description": "Every entry must match a type/subtype, e.g. ['creature','elf']."},
    "cmc_min": {"type": "number", "description": "Minimum mana value."},
    "cmc_max": {"type": "number", "description": "Maximum mana value."},
    "format": {"type": "string", "description": "Format the card must be legal in, e.g. 'commander', 'modern', 'pauper'."},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search Magic cards with a natural-language query. Returns the top 15 cards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the cards should do or be."},
                    "filters": {"type": "object", "properties": FILTER_SCHEMA, "description": "Optional hard filters."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the final list of up to 20 card names that satisfy the request, best first.",
            "parameters": {
                "type": "object",
                "properties": {"cards": {"type": "array", "items": {"type": "string"}, "description": "Exact card names."}},
                "required": ["cards"],
            },
        },
    },
]


TOOLS_GREP = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Case-insensitive regular expression over the plain text of every card (name, cost, type line, rules text, P/T, keywords). Returns the number of matching cards and 15 of them in file order (alphabetical), from `offset` on.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression, e.g. 'counter target spell unless'."},
            "offset": {"type": "integer", "description": "Skip this many matches (default 0) to page through a long match list."}},
            "required": ["pattern"]}}},
    TOOLS[1],
]


def constraints_text(f: dict) -> str:
    parts = []
    if f.get("color_identity"):
        parts.append("colourless only" if f["color_identity"] == "C" else f"colour identity within {f['color_identity']}")
    if f.get("format"):
        parts.append(f"legal in {f['format']}")
    if f.get("types"):
        parts.append("type: " + ", ".join(f["types"]))
    if f.get("cmc_min") is not None:
        parts.append(f"mana value ≥ {f['cmc_min']}")
    if f.get("cmc_max") is not None:
        parts.append(f"mana value ≤ {f['cmc_max']}")
    return "; ".join(parts)


def user_message(task: dict) -> str:
    msg = f"Request: {task['prompt']}"
    if task.get("filters"):
        msg += f"\n(Constraints already applied to every search automatically: {constraints_text(task['filters'])}.)"
    return msg


def format_results(env, query: str, n_done: int, rows, snippets=None, n_match=None, tool="search",
                   offset: int | None = None, last_note: bool = True) -> str:
    head = f"{tool} {n_done}/{MAX_SEARCHES}: "
    if n_match is not None:
        head += f"{n_match} cards match /{query}/; showing {len(rows)}"
        if offset is not None and rows:  # condition G: file order, paged
            head += f" (matches {offset + 1}-{offset + len(rows)} in file order)"
    else:
        head += f"{len(rows)} results for \"{query}\""
    lines = [head]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {env.render(row)}")
        if snippets:
            lines.append(f"   match: {snippets[i - 1]}")
    if n_done >= MAX_SEARCHES and last_note:
        lines.append("That was your last search. Call final_answer now.")
    return "\n".join(lines)


class Runner:
    def __init__(self, env, run_dir: Path, temperature: float, effort: str):
        self.env, self.run_dir = env, run_dir
        self.llm = GLM(temperature=temperature, effort=effort)
        (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        self.wlock = threading.Lock()

    def episode(self, task: dict, cname: str, rep: int) -> dict:
        cond = CONDITIONS[cname]
        env = self.env
        tpath = self.run_dir / "transcripts" / f"{task['id']}__{cname}__r{rep}.jsonl"
        tf = open(tpath, "w")

        def log(rec):
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.flush()

        grep_only = cond["retriever"] == "grep"
        all_tools = TOOLS_GREP if grep_only else TOOLS
        system = SYSTEM_GREP if grep_only else SYSTEM
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_message(task)}]
        sent = 0
        searches, final, usage_tot, cost, stop = [], None, {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}, 0.0, "max_turns"
        n_calls, nudged = 0, False
        for turn in range(MAX_TURNS):
            tools = all_tools if len(searches) < MAX_SEARCHES else all_tools[-1:]
            log({"event": "request", "turn": turn, "new_messages": messages[sent:], "tools": [t["function"]["name"] for t in tools],
                 "model": MODEL, "temperature": self.llm.temperature, "effort": self.llm.effort})
            sent = len(messages)
            resp = self.llm.chat(messages, tools)
            n_calls += 1
            log({"event": "response", "turn": turn, "raw": resp})
            u = resp.get("usage") or {}
            usage_tot["prompt"] += u.get("prompt_tokens", 0)
            usage_tot["cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            usage_tot["completion"] += u.get("completion_tokens", 0)
            usage_tot["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            cost += cost_usd(u)
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            # reasoning_content is logged but not sent back (keeps context small; same for all conditions)
            amsg = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                amsg["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in calls]
            messages.append(amsg)
            if not calls:
                if nudged:
                    stop = "no_tool_call"
                    break
                nudged = True
                messages.append({"role": "user", "content": "Please call final_answer with your list of card names now."})
                continue
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "final_answer" and final is None:
                    final = [str(x) for x in (args.get("cards") or [])][:MAX_FINAL]
                    content = "Received."
                elif name in ("search", "grep") and name in {t["function"]["name"] for t in all_tools} and len(searches) < MAX_SEARCHES:
                    q = str(args.get("query") or args.get("pattern") or "").strip()
                    filt = args.get("filters") if isinstance(args.get("filters"), dict) and not grep_only else {}
                    offset = max(0, int(args.get("offset") or 0)) if grep_only else None
                    rec = {"t": len(searches) + 1, "tool": name, "query": q, "filters": filt}
                    if grep_only:
                        rec["offset"] = offset
                    try:
                        snippets = n_match = None
                        if name == "grep":
                            n_match, rows, snippets = env.grep(q, task.get("filters"), offset=offset or 0) if q else (0, [], [])
                        else:
                            rows = env.search("mlateon", q, filt, task.get("filters")) if q else []
                        rec.update(rows=rows, oids=[env.oids[r] for r in rows], n_match=n_match)
                        searches.append(rec)
                        content = format_results(env, q, len(searches), rows, snippets, n_match, name, offset)
                        # the echoed query is the agent's own text (it may name e.g. "Illness in the Ranks"), not a harness signal
                        env.assert_no_leak(content, rows, [*(snippets or []), q])
                    except (ValueError, TypeError) as e:
                        rec.update(rows=[], oids=[], error=str(e))
                        searches.append(rec)
                        content = f"Search failed: {e}. ({len(searches)}/{MAX_SEARCHES} searches used.)"
                elif name in ("search", "grep") and len(searches) >= MAX_SEARCHES:
                    content = "Search budget exhausted. Call final_answer now."
                else:
                    content = f"Unknown tool {name}."
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            if final is not None:
                stop = "final_answer"
                log({"event": "tool_results", "new_messages": messages[sent:]})
                break
        tf.close()
        final = final or []
        final_oids = [env.resolve_name(n) for n in final]
        return {
            "task_id": task["id"], "kind": task["kind"], "cond": cname, "rep": rep, "searches": searches,
            "final": final, "final_oids": final_oids, "stop": stop, "llm_calls": n_calls,
            "usage": usage_tot, "cost_usd": round(cost, 6), "model": MODEL,
        }

    def run(self, tasks, conds, repeats, workers, budget=None):
        out = self.run_dir / "episodes.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add((r["task_id"], r["cond"], r["rep"]))
        jobs = [(t, c, r) for r in range(repeats) for t in tasks for c in conds if (t["id"], c, r) not in done]
        print(f"{len(jobs)} episodes to run ({len(done)} already done)", flush=True)
        total, t0 = 0.0, time.time()
        prior = sum(json.loads(x)["cost_usd"] for x in out.read_text().splitlines()) if out.exists() else 0.0
        stop_flag = threading.Event()

        def guarded(t, c, r):
            # budget guard: stop starting new episodes once this run's spend reaches the cap
            if stop_flag.is_set() or (budget is not None and prior + total >= budget):
                stop_flag.set()
                return None
            return self.episode(t, c, r)

        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(guarded, t, c, r): (t["id"], c, r) for t, c, r in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                key = futs[f]
                try:
                    rec = f.result()
                    if rec is None:
                        continue
                except Exception:
                    print(f"FAILED {key}:\n{traceback.format_exc()}", flush=True)
                    continue
                total += rec["cost_usd"]
                with self.wlock, open(out, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(jobs)}] {key} searches={len(rec['searches'])} final={len(rec['final'])} "
                      f"stop={rec['stop']} ${rec['cost_usd']:.4f} total=${total:.3f} {time.time() - t0:.0f}s", flush=True)
        if stop_flag.is_set():
            print(f"BUDGET STOP: run spend ${prior + total:.3f} reached cap ${budget}", flush=True)
        return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default=str(HERE / "tasks.yaml"))
    ap.add_argument("--conds", default="A,G")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of a stratified order")
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=float, default=None, help="USD cap on this run dir's total spend")
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted harness changes (recorded)")
    args = ap.parse_args()
    from env import git_state

    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        sys.exit("uncommitted changes in eval/agentic, src or bench/precon/harness: commit first (or --allow-dirty)")

    tasks = yaml.safe_load(Path(args.tasks).read_text())
    if args.ids:
        want = set(args.ids.split(","))
        tasks = [t for t in tasks if t["id"] in want]
    if args.limit:
        # interleave kinds so a pilot covers every task type
        by = {}
        for t in tasks:
            by.setdefault(t["kind"], []).append(t)
        order = []
        while any(by.values()):
            for k in list(by):
                if by[k]:
                    order.append(by[k].pop(0))
        tasks = order[: args.limit]
    conds = args.conds.split(",")
    from env import Env

    assert set(conds) <= set(CONDITIONS), f"conditions: {sorted(CONDITIONS)}"
    env = Env(("mlateon",) if "A" in conds else ())
    run_dir = HERE / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "git": git, "model": MODEL, "max_searches": MAX_SEARCHES,
                                                     "system": SYSTEM, "system_grep": SYSTEM_GREP,
                                                     "conditions": CONDITIONS}, indent=1))
    total = Runner(env, run_dir, args.temperature, args.effort).run(tasks, conds, args.repeats, args.workers, args.budget)
    print(f"done; cost this invocation ${total:.3f}")


if __name__ == "__main__":
    main()
