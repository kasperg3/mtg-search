"""EXP-006 agentic multi-step search: run GLM-5.3 episodes over the task set.

    uv run --extra models python eval/agentic/agent.py --run pilot --conds A,C --limit 10 --repeats 1

Conditions (identical system prompt; only the tool output differs):
  A  mLateOn, card info only
  B  SparseUp, card info only
  C  SparseUp + per-result matched terms (expansion marked *) + aggregate "frequent terms" line
  D  mLateOn + the same aggregate line, computed with SparseUp over mLateOn's results
  F  lexical-only agent: tools `grep(pattern)` (regex over card text; matches ordered by BM25 of the
     pattern's literal words, ties alphabetical; match highlighted) and `bm25(query)`; same budget (5 calls) and final-answer protocol. Its system
     prompt differs only in the paragraph describing the tools.

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
CONDITIONS = {
    "A": {"retriever": "mlateon", "per_result_terms": False, "aggregate": False},
    "B": {"retriever": "sparseup", "per_result_terms": False, "aggregate": False},
    "C": {"retriever": "sparseup", "per_result_terms": True, "aggregate": True},
    "D": {"retriever": "mlateon", "per_result_terms": False, "aggregate": True},
    "F": {"retriever": "lexical", "per_result_terms": False, "aggregate": False},  # grep + bm25 tools only
}

SYSTEM = """You are a deck-building search assistant for Magic: The Gathering. A player describes what they need; you find real cards for it with the `search` tool. It queries a card search engine over all Magic cards (it understands natural language; it is not Scryfall syntax) and returns the top 15 matches with name, mana cost, type and truncated rules text. Optional filters restrict colours, types, mana value and format.

You have a budget of 5 searches. Use them to find as many cards that genuinely satisfy the request as possible. After each search, look at what came back and decide what to search next: refine, rephrase, or explore a different angle that the results suggest. A search that returns the same cards again is wasted.

When you are done (after at most 5 searches), call `final_answer` with up to 20 card names that satisfy the request, best first. Only include cards that appeared in your search results, and copy their names exactly. If the request is for one specific card, put your best guess first."""

SYSTEM_LEXICAL = SYSTEM.replace(
    "you find real cards for it with the `search` tool. It queries a card search engine over all Magic cards (it understands natural language; it is not Scryfall syntax) and returns the top 15 matches with name, mana cost, type and truncated rules text. Optional filters restrict colours, types, mana value and format.",
    "you find real cards for it with two lexical tools over the full text of all Magic cards (name, cost, type line, rules text): `grep` runs a case-insensitive regular expression and returns the number of matching cards plus 15 of them (ordered by keyword relevance to the words in the pattern, then alphabetically) with the matching line highlighted in [[ ]]; `bm25` is keyword search returning the top 15 cards by BM25 score. Both show name, mana cost, type and truncated rules text. Optional filters restrict colours, types, mana value and format.",
).replace("You have a budget of 5 searches.", "You have a budget of 5 searches (grep and bm25 calls both count).")
assert SYSTEM_LEXICAL != SYSTEM

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


def _lex_tool(name: str, arg: str, desc: str, argdesc: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {arg: {"type": "string", "description": argdesc},
                                         "filters": {"type": "object", "properties": FILTER_SCHEMA, "description": "Optional hard filters."}},
        "required": [arg]}}}


TOOLS_LEXICAL = [
    _lex_tool("grep", "pattern", "Case-insensitive regex over card text. Returns the match count and 15 matching cards (keyword-relevance order).",
              "Python regular expression, e.g. 'counter target spell unless'."),
    _lex_tool("bm25", "query", "BM25 keyword search over card text. Returns the top 15 cards.", "Keywords."),
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


def format_results(env, cond: dict, query: str, n_done: int, rows, per_terms, agg, snippets=None, n_match=None, tool="search") -> str:
    head = f"{tool} {n_done}/{MAX_SEARCHES}: "
    if n_match is not None:
        head += f"{n_match} cards match /{query}/; showing {len(rows)}"
    else:
        head += f"{len(rows)} results for \"{query}\""
    lines = [head]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {env.render(row)}")
        if snippets:
            lines.append(f"   match: {snippets[i - 1]}")
        if cond["per_result_terms"]:
            lines.append("   matched terms: " + ", ".join(w + ("*" if x else "") for w, x in per_terms[i - 1]))
    if cond["per_result_terms"]:
        lines.append("(matched terms = the search engine's top-weighted terms linking your query to the card; * = expansion term, not a word in your query)")
    if cond["aggregate"]:
        lines.append("Frequent terms in these results not in your query: " + ", ".join(f"{w} ({c})" for w, c in agg))
    if n_done >= MAX_SEARCHES:
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

        lexical = cond["retriever"] == "lexical"
        all_tools = TOOLS_LEXICAL if lexical else TOOLS
        messages = [{"role": "system", "content": SYSTEM_LEXICAL if lexical else SYSTEM}, {"role": "user", "content": user_message(task)}]
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
                elif name in ("search", "grep", "bm25") and name in {t["function"]["name"] for t in all_tools} and len(searches) < MAX_SEARCHES:
                    q = str(args.get("query") or args.get("pattern") or "").strip()
                    filt = args.get("filters") if isinstance(args.get("filters"), dict) else {}
                    rec = {"t": len(searches) + 1, "tool": name, "query": q, "filters": filt}
                    try:
                        snippets = n_match = None
                        if name == "grep":
                            n_match, rows, snippets = env.grep(q, filt, task.get("filters")) if q else (0, [], [])
                        else:
                            rows = env.search("bm25" if name == "bm25" else cond["retriever"], q, filt, task.get("filters")) if q else []
                        # terms are computed for every non-grep search (shown only in C/D) as a counterfactual baseline
                        per = env.per_result_terms(q, rows) if rows and name != "grep" else []
                        agg = env.aggregate_terms(q, rows) if rows and name != "grep" else []
                        rec.update(rows=rows, oids=[env.oids[r] for r in rows], per_terms=per, agg_terms=agg,
                                   shown_terms=cond["per_result_terms"] or cond["aggregate"], n_match=n_match)
                        searches.append(rec)
                        content = format_results(env, cond, q, len(searches), rows, per, agg, snippets, n_match, name)
                        # the echoed query is the agent's own text (it may name e.g. "Illness in the Ranks"), not a harness signal
                        env.assert_no_leak(content, rows, [*(snippets or []), q])
                    except ValueError as e:
                        rec.update(rows=[], oids=[], per_terms=[], agg_terms=[], error=str(e))
                        searches.append(rec)
                        content = f"Search failed: {e}. ({len(searches)}/{MAX_SEARCHES} searches used.)"
                elif name in ("search", "grep", "bm25") and len(searches) >= MAX_SEARCHES:
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
    ap.add_argument("--conds", default="A,B,C,D")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of a stratified order")
    ap.add_argument("--ids", default=None, help="comma-separated task ids")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=float, default=None, help="USD cap on this run dir's total spend")
    args = ap.parse_args()

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

    retr = set()
    for c in conds:
        retr |= {"bm25"} if CONDITIONS[c]["retriever"] == "lexical" else {CONDITIONS[c]["retriever"]}
    retr = tuple(sorted(retr))
    env = Env(retr)
    run_dir = HERE / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "model": MODEL, "max_searches": MAX_SEARCHES,
                                                     "system": SYSTEM, "system_lexical": SYSTEM_LEXICAL, "conditions": CONDITIONS}, indent=1))
    total = Runner(env, run_dir, args.temperature, args.effort).run(tasks, conds, args.repeats, args.workers, args.budget)
    print(f"done; cost this invocation ${total:.3f}")


if __name__ == "__main__":
    main()
