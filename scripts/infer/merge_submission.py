"""Merge the per-shard submissions of a run (outputs/runs/<prefix>_s<k>/submission.jsonl) into one file in split order.
  python scripts/infer/merge_submission.py pub_q35e1 --split public_test --output result/submission.jsonl
  --override PREFIX replaces the rows of the samples that PREFIX's shards re-ran (e.g. after a rule change that touches only them).
Refuses to write when a shard is unfinished, an id is missing or duplicated, a sample failed at runtime (unless --allow-failures:
its row then holds the calls executed before the error), or the merged file does not replay through the official ToolEnv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import DATA_ROOT, items  # noqa: E402
from lifemate.submission import shard_dirs, validate_file  # noqa: E402

RUNS = PROJECT / "outputs/runs"


def read_run(prefix: str) -> tuple[list[Path], dict[str, dict], list[str]]:
    """Rows and runtime-failed ids of every shard of a run prefix."""
    dirs = shard_dirs(RUNS / prefix)
    if not dirs:
        raise SystemExit(f"no run directories for {RUNS / prefix}[_s<k>]")
    rows: dict[str, dict] = {}
    failed: list[str] = []
    for d in dirs:
        f = d / "submission.jsonl"
        if not f.is_file():
            raise SystemExit(f"unfinished shard (no submission.jsonl): {d}")
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in rows:
                raise SystemExit(f"duplicate id {r['id']} in {prefix}")
            rows[r["id"]] = r
        errors = d / "runtime_errors.jsonl"
        failed += [json.loads(l)["sample_id"] for l in errors.open(encoding="utf-8") if l.strip()] if errors.is_file() else []
    return dirs, rows, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--override", help="run prefix whose rows replace the main run's rows for the same ids")
    ap.add_argument("--allow-failures", action="store_true", help="write even if samples failed at runtime (their rows are partial)")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    order = [x["id"] for x in items(DATA_ROOT, a.split)]
    dirs, rows, failed = read_run(a.prefix)
    replaced = 0
    if a.override:
        over_dirs, over, over_failed = read_run(a.override)
        if unknown := set(over) - set(rows):
            raise SystemExit(f"override rows not in the main run: {sorted(unknown)[:3]}")
        rows.update(over)
        replaced = len(over)
        failed = [i for i in failed if i not in over] + over_failed
        dirs += over_dirs
    missing = [i for i in order if i not in rows]
    extra = set(rows) - set(order)
    if missing or extra:
        raise SystemExit(f"missing {len(missing)} ids (first {missing[:3]}), unexpected {len(extra)}")
    if failed and not a.allow_failures:
        raise SystemExit(f"{len(failed)} samples failed at runtime (see runtime_errors.jsonl): {failed[:10]}; re-run them or pass --allow-failures")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for i in order:
            out.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    try:
        check = validate_file(DATA_ROOT, a.split, tmp)
    except Exception:
        tmp.unlink()
        raise
    tmp.rename(a.output)
    print(f"wrote {len(order)} rows from {len(dirs)} run dirs ({replaced} overridden, {len(failed)} failed kept as executed) "
          f"to {a.output}; ToolEnv replay: {check}")


if __name__ == "__main__":
    main()
