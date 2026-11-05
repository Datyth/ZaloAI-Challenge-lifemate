"""Summarize local diagnostics from existing run traces; never inspect hidden state."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.agent import _fake_narration  # noqa: E402
from lifemate.data import items  # noqa: E402


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def summarize(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    predictions = rows(run_dir / "predictions.jsonl")
    llm = rows(run_dir / "llm_turns.jsonl")
    events = rows(run_dir / "tool_events.jsonl")
    terminations = rows(run_dir / "terminations.jsonl")
    dataset = {x["id"]: x for x in items(PROJECT / "data" / "train+public_test", config["split"])}
    bins: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        sample = dataset.get(prediction["id"])
        if sample is None:
            continue
        n = len(sample.get("history") or [])
        bucket = "0" if n == 0 else "1-3" if n < 4 else "4-11" if n < 12 else "12+"
        bins[bucket].append(not prediction.get("tool_calls"))
    fake_turns = [x for x in llm if not (x.get("output") or {}).get("tool_calls") and _fake_narration((x.get("output") or {}).get("content") or "")]
    invalid = [x for x in events if x.get("executed") is False]
    ungrounded = [x for x in invalid if "observed" in str(x.get("validation", {}).get("error", ""))]
    lengths = [x for x in llm if x.get("finish_reason") == "length"]
    latency = sorted(float(x["latency_seconds"]) for x in predictions if "latency_seconds" in x)
    statuses = Counter(x.get("status", "unknown") for x in terminations)
    abnormal = sum(x.get("status") == "error" for x in predictions)
    abnormal += len({x["sample_id"] for x in lengths if not any(p["id"] == x["sample_id"] and p.get("status") == "error" for p in predictions)})
    result = {
        "run_dir": str(run_dir), "split": config["split"], "orchestration": config.get("orchestration", "E0 (legacy)"),
        "samples": len(predictions), "official_score": None,
        "train_state_match_rate": sum(x.get("train_state_match") is True for x in predictions) / len(predictions) if config["split"] == "train" and predictions else None,
        "train_exact_call_f1_surrogate": sum(x.get("train_exact_call_f1_surrogate", 0) for x in predictions) / len(predictions) if config["split"] == "train" and predictions else None,
        "empty_prediction_rate_by_history": {k: {"samples": len(v), "rate": sum(v) / len(v)} for k, v in sorted(bins.items())},
        "fake_tool_narration_turns": len(fake_turns), "fake_tool_narration_samples": len({x["sample_id"] for x in fake_turns}),
        "invalid_proposals": len(invalid), "ungrounded_id_proposals": len(ungrounded),
        "length_finish_turns": len(lengths), "abnormal_termination_samples": abnormal,
        "termination_statuses": dict(statuses),
        "latency_mean_seconds": sum(latency) / len(latency) if latency else None,
        "latency_p95_seconds": latency[math.ceil(.95 * len(latency)) - 1] if latency else None,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = [summarize(path) for path in args.run_dirs]
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
