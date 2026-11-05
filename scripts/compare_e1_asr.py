"""Compare complete E1 agent runs with different current/history transcripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
FROZEN_IDS = PROJECT / "outputs" / "splits" / "validation_ids.json"
MATCH_KEYS = ("kit_version", "llm_model", "llm_manifest", "device", "prompt_sha256",
              "parser", "orchestration", "decoding", "deadline_seconds", "max_turns", "seed")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_run(name: str, directory: Path, ids: list[str]) -> dict:
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "train" or config.get("sample_ids") != ids or config.get("orchestration") != "E1":
        raise ValueError(f"{name}: wrong split, sample order or agent orchestration in {directory}")
    rows = read_jsonl(directory / "predictions.jsonl")
    if [row.get("id") for row in rows] != ids:
        raise ValueError(f"{name}: missing, duplicate or reordered predictions in {directory}")
    if not (directory / "metrics.json").is_file() or not (directory / "submission.jsonl").is_file():
        raise ValueError(f"{name}: run lacks final metrics or replayable predictions")
    if [row.get("id") for row in read_jsonl(directory / "submission.jsonl")] != ids:
        raise ValueError(f"{name}: incomplete submission artifact")
    recorded = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if recorded.get("replay_hash_matches") != len(ids) or recorded.get("samples") != len(ids):
        raise ValueError(f"{name}: ToolEnv replay was incomplete")
    if not (directory / "llm_turns.jsonl").is_file() or not (directory / "tool_events.jsonl").is_file():
        raise ValueError(f"{name}: execution traces are missing")
    if any(not isinstance(row.get("latency_seconds"), (int, float)) or row["latency_seconds"] < 0 for row in rows):
        raise ValueError(f"{name}: invalid per-sample latency")
    return {"name": name, "directory": directory, "config": config, "rows": rows, "metrics": recorded}


def validate_design(runs: list[dict]) -> None:
    baseline = runs[0]["config"]
    expected_sources = {"A0": "asr", "A1": "cached-asr", "A2": "cached-asr", "A3": "train-reference"}
    for run in runs:
        config = run["config"]
        for key in MATCH_KEYS:
            if config.get(key) != baseline.get(key):
                raise ValueError(f"{run['name']}: {key} differs from A0")
        if config.get("transcript_source") != expected_sources[run["name"]]:
            raise ValueError(f"{run['name']}: unexpected transcript source")
    if runs[1]["config"].get("asr_backend") != "chunkformer" or runs[2]["config"].get("asr_backend") != "gipformer":
        raise ValueError("A1/A2 ASR backends are not ChunkFormer/Gipformer")
    for run in runs[1:3]:
        config = run["config"]
        cache = Path(config["transcripts_jsonl"])
        cache_config = json.loads((cache.parent / "run_config.json").read_text(encoding="utf-8"))
        cache_metrics = json.loads((cache.parent / "metrics.json").read_text(encoding="utf-8"))
        if (cache_config.get("sample_ids") != baseline["sample_ids"] or
            cache_config.get("model_name") != config["asr_backend"] or
            cache_config.get("model") != config["asr_model"] or
            cache_config.get("checkpoint_manifest") != config["asr_manifest"] or
            not cache_metrics.get("complete") or cache_metrics.get("history_failures") != 0 or
            cache_metrics.get("saved_current_transcripts") != len(baseline["sample_ids"]) or
            cache_metrics.get("history_transcripts") != cache_config.get("history_audio")):
            raise ValueError(f"{run['name']}: ASR cache provenance or completeness mismatch")
        cache_rows = read_jsonl(cache)
        expected_rows = len(baseline["sample_ids"]) + cache_config.get("history_audio", -1)
        if len(cache_rows) != expected_rows or len({(row["sample_id"], row["audio"]) for row in cache_rows}) != expected_rows:
            raise ValueError(f"{run['name']}: ASR cache rows are missing or duplicated")


def summarize(run: dict) -> dict:
    rows = run["rows"]
    correct = sum(row.get("status") == "ok" and row.get("train_state_match") is True for row in rows)
    f1 = sum(float(row.get("train_exact_call_f1_surrogate") or 0) if row.get("status") == "ok" else 0 for row in rows) / len(rows)
    errors = sum(row.get("status") != "ok" for row in rows)
    latency = sorted(float(row["latency_seconds"]) for row in rows)
    return {"configuration": run["name"], "run_dir": str(run["directory"]), "samples": len(rows),
            "state_matches": correct, "state_match_rate": correct / len(rows),
            "tool_call_f1": f1, "runtime_errors": errors,
            "latency_mean_seconds": statistics.mean(latency), "latency_p95_seconds": latency[(95 * len(latency) + 99) // 100 - 1],
            "latency_max_seconds": latency[-1]}


def compare(a0: Path, a1: Path, a2: Path, a3: Path, output_dir: Path) -> dict:
    ids = json.loads(FROZEN_IDS.read_text(encoding="utf-8"))
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("frozen TRAIN validation ID file is not exactly 200 unique samples")
    runs = [load_run(name, path, ids) for name, path in (("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3))]
    validate_design(runs)
    summaries = [summarize(run) for run in runs]
    indexed = {run["name"]: dict(zip(ids, run["rows"])) for run in runs}
    transitions = {name: {"correct_to_incorrect": 0, "incorrect_to_correct": 0,
                           "correct_to_correct": 0, "incorrect_to_incorrect": 0} for name in ("A1", "A2", "A3")}
    per_sample = []
    for sample_id in ids:
        row = {"id": sample_id, "models": {}}
        base = indexed["A0"][sample_id]
        base_correct = base.get("status") == "ok" and base.get("train_state_match") is True
        for name in ("A0", "A1", "A2", "A3"):
            pred = indexed[name][sample_id]
            correct = pred.get("status") == "ok" and pred.get("train_state_match") is True
            row["models"][name] = {"state_match": correct, "tool_call_f1": pred.get("train_exact_call_f1_surrogate") if pred.get("status") == "ok" else 0.0,
                                   "status": pred.get("status"), "error": pred.get("error"),
                                   "termination": pred.get("termination"), "latency_seconds": pred["latency_seconds"]}
            if name != "A0":
                label = ("correct" if base_correct else "incorrect") + "_to_" + ("correct" if correct else "incorrect")
                transitions[name][label] += 1
                row["models"][name]["transition_vs_A0"] = label
        per_sample.append(row)
    result = {"sample_ids": len(ids), "configurations": summaries, "transitions_vs_A0": transitions,
              "latency_note": "A1/A2 reuse measured current-utterance transcripts and precompute history transcripts. The saved ASR decode times for both are added to per-sample agent wall time and charged against the 30-second deadline. A0/A3 are preserved earlier measurements; latency comparisons remain indicative because runs happened at different times."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_sample_transitions.jsonl").open("w", encoding="utf-8") as out:
        for row in per_sample:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    lines = ["# E1 Agent × ASR end-to-end TRAIN comparison", "",
             "All four configurations use the same frozen 200 TRAIN IDs, Qwen3-4B-Instruct-2507, E1 prompt and structured history, tool schemas, decoding, and ToolEnv. A0 and A3 reuse preserved runs. State match and exact tool-call F1 are local TRAIN metrics, not an official score.", "",
             "| Configuration | State match | Tool-call F1 | Runtime errors | Mean / P95 latency (s) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["configurations"]:
        lines.append(f"| {row['configuration']} | {row['state_matches']}/200 ({row['state_match_rate']:.1%}) | {row['tool_call_f1']:.3f} | {row['runtime_errors']} | {row['latency_mean_seconds']:.2f} / {row['latency_p95_seconds']:.2f} |")
    lines += ["", "## State transitions relative to A0", "",
              "| Configuration | Correct → incorrect | Incorrect → correct | Correct → correct | Incorrect → incorrect |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, counts in result["transitions_vs_A0"].items():
        lines.append(f"| {name} | {counts['correct_to_incorrect']} | {counts['incorrect_to_correct']} | {counts['correct_to_correct']} | {counts['incorrect_to_incorrect']} |")
    lines += ["", result["latency_note"], "", "Each run directory contains predictions, LLM/tool execution traces, metrics, and replayable tool calls. `per_sample_transitions.jsonl` records the paired outcome for every ID.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0", type=Path, default=PROJECT / "outputs/runs/train200_20261105_115719_E1")
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--a3", type=Path, default=PROJECT / "outputs/runs/agentopt_full_01_E4")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.a0, args.a1, args.a2, args.a3, args.output_dir)
    print(json.dumps({"report": str(args.output_dir / "report.md"), "state_matches": {x["configuration"]: x["state_matches"] for x in result["configurations"]}}, indent=2))
