"""Compare two submission files row by row (e.g. the Docker run against the local run that was scored).

    .venv/bin/python scripts/infer/compare_submissions.py result_docker/submission.jsonl result/submission_q35fullp1_asrfinal_20261109.jsonl

Prints the ids in each file, how many rows have the same state_hash and the same tool_calls, and the time field (ms) of each file."""
from __future__ import annotations

import json
import statistics
import sys


def load(path: str) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in open(path, encoding="utf-8") if line.strip())}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    print(f"rows: {len(a)} vs {len(b)}, common ids {len(common)}")
    same_hash = sum(a[i]["state_hash"] == b[i]["state_hash"] for i in common)
    same_calls = sum(a[i]["tool_calls"] == b[i]["tool_calls"] for i in common)
    print(f"same state_hash {same_hash}/{len(common)}, same tool_calls {same_calls}/{len(common)}")
    for name, rows in ((sys.argv[1], a), (sys.argv[2], b)):
        times = [r["time"] for r in rows.values() if isinstance(r.get("time"), (int, float))]
        if times:
            print(f"time {name}: mean {statistics.mean(times):.0f} ms, max {max(times)} ms, over 30 s: {sum(t > 30000 for t in times)}")
    diff = [i for i in common if a[i]["state_hash"] != b[i]["state_hash"]]
    if diff:
        print("different state_hash:", " ".join(diff[:30]) + (" ..." if len(diff) > 30 else ""))


if __name__ == "__main__":
    main()
