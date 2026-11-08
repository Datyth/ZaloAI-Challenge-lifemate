"""Merge the per-shard submissions of a run (outputs/runs/<prefix>_s*/submission.jsonl) into one file in split order.
  python analysis/merge_submission.py pub_sft_asrlower --split public_test --output result/submission_s1_lower.jsonl
  --override PREFIX replaces the rows of the samples that PREFIX's shards re-ran (e.g. after a rule change that touches only them).
Refuses to write when a shard is unfinished, an id is missing or duplicated, or a row has the wrong shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/train+public_test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--override", help="run prefix whose rows replace the main run's rows for the same ids")
    a = ap.parse_args()
    order = [json.loads(l)["id"] for l in (ROOT / a.split / f"{a.split}.jsonl").open(encoding="utf-8") if l.strip()]
    shards = sorted((PROJECT / "outputs/runs").glob(a.prefix + "_s*"))
    rows: dict[str, dict] = {}
    for d in shards:
        f = d / "submission.jsonl"
        if not f.is_file():
            raise SystemExit(f"unfinished shard (no submission.jsonl): {d}")
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            if set(r) != {"id", "tool_calls", "state_hash"} or not isinstance(r["tool_calls"], list) or len(r["state_hash"]) != 64:
                raise SystemExit(f"bad row shape in {f}: {r.get('id')}")
            if r["id"] in rows:
                raise SystemExit(f"duplicate id {r['id']}")
            rows[r["id"]] = r
    replaced = 0
    if a.override:
        over = sorted((PROJECT / "outputs/runs").glob(a.override + "_s*"))
        for d in over:
            f = d / "submission.jsonl"
            if not f.is_file():
                raise SystemExit(f"unfinished override shard: {d}")
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                if set(r) != {"id", "tool_calls", "state_hash"} or r["id"] not in rows:
                    raise SystemExit(f"bad override row in {f}: {r.get('id')}")
                rows[r["id"]] = r
                replaced += 1
        shards += over
    missing = [i for i in order if i not in rows]
    extra = [i for i in rows if i not in set(order)]
    if missing or extra:
        raise SystemExit(f"missing {len(missing)} ids (first {missing[:3]}), unexpected {len(extra)}")
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as out:
        for i in order:
            out.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    errors = sum(1 for d in shards for l in (d / "runtime_errors.jsonl").open(encoding="utf-8") if l.strip())
    print(f"wrote {len(order)} rows from {len(shards)} shards ({replaced} overridden) to {a.output}; samples with runtime errors kept as executed: {errors}")


if __name__ == "__main__":
    main()
