"""Does the cache really cover the dataset? To be run BEFORE scoring.

`cares eval --scores-only` rebuilds its records with a filter, never with a
length comparison, so a truncated cache scores silently and plausibly. This
tool compares the cache against the `items_<task>.json` witness written by
`cares eval --dump-items`, and refuses on missing keys, intruding keys, records
whose `gold`/`options` diverge, mixed run settings, or records in error.

    python tools/verifier_fusion.py <eval_output> <backend>
    python tools/verifier_fusion.py other/data/data_test/eval_output moss-audio --tolerate-errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: The five counts of the test split, shown as a landmark only: the truth stays
#: `items_<task>.json`, which follows the data-dir it is given.
EXPECTED_COUNTS_TEST_SPLIT = {
    "scene": 925, "sounds-mcq": 2123, "reactions": 2123,
    "grounding": 1782, "summary": 2673,
}


def _run_stamps(cache: dict) -> list[tuple]:
    """The sets of invisible run settings present in the cache, sorted."""
    seen = set()
    for r in cache.values():
        if isinstance(r, dict) and isinstance(r.get("run"), dict):
            seen.add(tuple(sorted(r["run"].items())))
    return sorted(seen)


def verifier(out: Path, backend: str, tolerate_errors: bool,
             tasks: list[str] | None = None) -> int:
    witnesses = sorted(out.glob("items_*.json"))
    if tasks:
        # A campaign may cover a single task, while the witness always covers
        # all five: without this filter a partial run would be refused.
        wanted = {f"items_{t}.json" for t in tasks}
        witnesses = [f for f in witnesses if f.name in wanted]
    if not witnesses:
        print(f"No items_*.json in {out}.")
        print("  Write the witness first, with THE SAME options as the shards:")
        print("  cares eval --data-dir <d> --backend fake --banks <v3> "
              "--counterfactual-dirs <cf> --task all --dump-items")
        return 2

    rc = 0
    for witness in witnesses:
        task = witness.name[len("items_"):-len(".json")]
        path = out / f"raw_{backend}_{task}.json"
        if not path.exists():
            print(f"  {task:12s} : NO cache ({path.name})")
            rc = 1
            continue

        items = {it["item_id"]: it for it in json.loads(witness.read_text())}
        cache = json.loads(path.read_text())
        missing = sorted(set(items) - set(cache))
        intruders = sorted(set(cache) - set(items))
        divergent = sorted(
            k for k, r in cache.items()
            if k in items and isinstance(r, dict)
            and (r.get("gold") != items[k].get("gold")
                 or r.get("options") != items[k].get("options")))
        errors = sorted(k for k, r in cache.items()
                        if isinstance(r, dict) and r.get("error"))
        settings = _run_stamps(cache)

        blocking = bool(missing or intruders or divergent or len(settings) > 1)
        blocking = blocking or (bool(errors) and not tolerate_errors)
        expected = EXPECTED_COUNTS_TEST_SPLIT.get(task)
        marker = "" if expected in (None, len(items)) else f" [!= {expected} expected]"
        print(f"  {task:12s} : {len(cache):5d}/{len(items):5d}{marker}  "
              f"{'REFUSED' if blocking else 'ok'}  missing={len(missing)} "
              f"intruders={len(intruders)} divergent={len(divergent)} "
              f"errors={len(errors)}")
        for label, keys in (("missing", missing), ("intruder", intruders),
                            ("divergent", divergent), ("in error", errors)):
            if keys:
                print(f"      e.g. {label}: {keys[0]}"
                      f"{'' if len(keys) == 1 else f'  (+{len(keys) - 1})'}")
        if len(settings) > 1:
            print(f"      MIXED RUN SETTINGS: {len(settings)} distinct sets")
            for r in settings:
                print(f"        {dict(r)}")
        rc = 1 if blocking else rc

    if rc == 1:
        print()
        print("DO NOT SCORE AS IS. A score computed on an incomplete cache is")
        print("well formed, plausible and indistinguishable from a complete one.")
        print("  missing   -> relaunch the shards concerned")
        print("  errors    -> cares eval --retry-errors on the merged folder")
        print("  divergent -> a shard ran with another bank: throw it away")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_output", type=Path)
    ap.add_argument("backend")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="check only these tasks (default: every task with an "
                         "items_*.json)")
    ap.add_argument("--tolerate-errors",
                    action="store_true",
                    help="do not fail on items in error (use only once it is "
                         "established that they come from the MODEL and not "
                         "from the infrastructure)")
    args = ap.parse_args()
    return verifier(args.eval_output, args.backend, args.tolerate_errors,
                    args.tasks)


if __name__ == "__main__":
    raise SystemExit(main())
