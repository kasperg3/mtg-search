"""Re-resolve every saved episode's final answer with the current ``Env.resolve_name``.

    uv run python eval/agentic/rescore.py            # dry run: list every answer whose card changes
    uv run python eval/agentic/rescore.py --write    # rewrite final_oids in runs/*/episodes.jsonl

Written for the face-name fix (2026-10-15): the name index used to let another card's face name win over a
card's full name, so "Lightning Bolt" scored as "Emeritus of Conflict // Lightning Bolt". Only ``final_oids``
is recomputed; ``candidate_oids`` (arm I) has no stored names and fed the run itself, so it is left as run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env import Env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    env = Env(retrievers=())
    name_of = dict(zip(env.oids, env.names))
    for path in sorted((HERE / "runs").glob("*/episodes.jsonl")):
        lines = path.read_text().splitlines()
        out, n_changed = [], 0
        for line in lines:
            e = json.loads(line)
            new = [env.resolve_name(n) for n in e["final"]]
            for i, (n, a, b) in enumerate(zip(e["final"], e["final_oids"], new)):
                if a != b:
                    n_changed += 1
                    arm = e.get("condition") or e.get("arm") or e.get("cond")
                    print(f"{path.parent.name}\t{e.get('task_id')}\t{arm}\tr{e.get('rep', e.get('repeat'))}\t#{i + 1}\t"
                          f"{n!r}: {name_of.get(a, a)} -> {name_of.get(b, b)}")
            e["final_oids"] = new
            out.append(json.dumps(e, ensure_ascii=False))
        print(f"# {path.parent.name}: {n_changed} answers change")
        if args.write and n_changed:
            path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
