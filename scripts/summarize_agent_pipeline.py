"""Summarize paired TRAIN oracle runs without reading labels during inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from lifemate.data import items


ROOT = Path(__file__).resolve().parents[1] / "data" / "train+public_test"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run_dirs: list[Path], expected_samples: int = 200) -> dict:
    baseline = None
    baseline_config = None
    report = {}
    gold = {item["id"]: item["label"]["tool_calls"] for item in items(ROOT, "train")}
    per_sample = {}
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text())
        if config["transcript_source"] != "train-reference":
            raise ValueError(f"not a TRAIN reference run: {run_dir}")
        frozen = {key: config.get(key) for key in ("split", "sample_ids", "kit_version", "llm_model", "llm_served_model", "adapter_dir", "device", "parser", "decoding", "deadline_seconds", "max_turns")}
        if baseline_config is None:
            baseline_config = frozen
        elif frozen != baseline_config:
            raise ValueError(f"frozen experiment settings differ: {run_dir}")
        predictions = rows(run_dir / "predictions.jsonl")
        by_id = {row["id"]: row for row in predictions}
        if len(by_id) != len(predictions) or len(predictions) != expected_samples:
            raise ValueError(f"expected {expected_samples} unique samples: {run_dir}")
        if baseline is None:
            baseline = by_id
        if set(by_id) != set(baseline):
            raise ValueError(f"sample IDs differ: {run_dir}")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        events = rows(run_dir / "tool_events.jsonl") if (run_dir / "tool_events.jsonl").exists() else []
        selection_matches = 0
        argument_matches = 0
        gold_calls = 0
        incomplete_multi_step = 0
        for sample_id, row in by_id.items():
            expected = gold[sample_id]
            actual = row["tool_calls"]
            selection_matches += [call["name"] for call in expected] == [call["name"] for call in actual]
            expected_calls = Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in expected)
            actual_calls = Counter(json.dumps(call, sort_keys=True, ensure_ascii=False) for call in actual)
            argument_matches += sum((expected_calls & actual_calls).values())
            gold_calls += len(expected)
            incomplete_multi_step += len(expected) > 1 and len(actual) < len(expected) and not row.get("train_state_match")
            per_sample.setdefault(sample_id, {})[run_dir.name] = {
                "state_match": row.get("train_state_match"), "tool_calls": actual,
                "gold_tool_calls": expected, "termination": row.get("termination"),
            }
        report[run_dir.name] = {
            "state_match_rate": metrics["train_state_match_rate"],
            "exact_tool_call_f1_surrogate": metrics["train_exact_call_f1_surrogate"],
            "latency_mean_seconds": metrics["latency_mean_seconds"],
            "latency_p95_seconds": metrics["latency_p95_seconds"],
            "runtime_errors": metrics["failures"],
            "exact_tool_selection_rate": selection_matches / len(by_id),
            "exact_call_recall": argument_matches / gold_calls if gold_calls else 1.0,
            "incomplete_multi_step_surrogate": incomplete_multi_step,
            "invalid_tool_proposals": sum(not event.get("executed", False) for event in events),
            "tool_execution_failures": sum(event.get("executed") and event.get("outcome") in {"unsuccessful", "rejected"} for event in events),
            "incorrect_to_correct": sorted(sample_id for sample_id, row in by_id.items() if row.get("train_state_match") and not baseline[sample_id].get("train_state_match")),
            "correct_to_incorrect": sorted(sample_id for sample_id, row in by_id.items() if not row.get("train_state_match") and baseline[sample_id].get("train_state_match")),
        }
    best = max(report, key=lambda name: (report[name]["state_match_rate"], report[name]["exact_tool_call_f1_surrogate"], -report[name]["runtime_errors"]))
    return {"runs": report, "best_validated_run": best, "per_sample": per_sample}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dirs, args.expected_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
