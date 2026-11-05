"""Paired S0/S1 E1 TRAIN comparison for one PEFT adapter checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import catalog, items  # noqa: E402
from lifemate.metrics import exact_tool_f1  # noqa: E402

SAME_KEYS = ("split", "sample_ids", "kit_version", "llm_model", "llm_manifest", "device",
             "prompt_sha256", "parser", "orchestration", "decoding", "deadline_seconds", "max_turns", "seed")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_run(name: str, path: Path, ids: list[str]) -> dict:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(path / "predictions.jsonl")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if config.get("sample_ids") != ids or config.get("split") != "train" or config.get("orchestration") != "E1":
        raise ValueError(f"{name}: wrong TRAIN IDs or orchestration")
    if [row.get("id") for row in predictions] != ids or metrics.get("replay_hash_matches") != 200:
        raise ValueError(f"{name}: incomplete predictions or ToolEnv replay")
    if not (path / "llm_turns.jsonl").is_file() or not (path / "tool_events.jsonl").is_file():
        raise ValueError(f"{name}: execution traces missing")
    return {"name": name, "path": path, "config": config, "predictions": predictions}


def check_design(runs: list[dict]) -> None:
    base = runs[0]["config"]
    for run in runs:
        for key in SAME_KEYS:
            if run["config"].get(key) != base.get(key):
                raise ValueError(f"{run['name']}: {key} differs from S0")
    s0, s1, oracle0, oracle1 = (run["config"] for run in runs)
    if (s0.get("transcript_source") != "cached-asr" or s1.get("transcript_source") != "cached-asr" or
        oracle0.get("transcript_source") != "train-reference" or oracle1.get("transcript_source") != "train-reference"):
        raise ValueError("wrong S0/S1/oracle transcript sources")
    for key in ("asr_backend", "asr_model", "asr_manifest", "transcripts_jsonl"):
        if s0.get(key) != s1.get(key):
            raise ValueError(f"S1: {key} differs from S0")
    if s0.get("asr_backend") != "gipformer" or s0.get("adapter_dir") or oracle0.get("adapter_dir"):
        raise ValueError("S0 must be the original Gipformer + E1 configuration")
    if not s1.get("adapter_dir") or s1.get("adapter_dir") != oracle1.get("adapter_dir"):
        raise ValueError("S1 and S1-Oracle must use the same adapter")
    if s1.get("adapter_config_sha256") != oracle1.get("adapter_config_sha256"):
        raise ValueError("S1 adapter config hashes differ")
    if not s1.get("adapter_weights_sha256") or s1.get("adapter_weights_sha256") != oracle1.get("adapter_weights_sha256"):
        raise ValueError("S1 adapter weight hashes differ or are missing")
    if s1.get("llm_served_model") != oracle1.get("llm_served_model"):
        raise ValueError("S1 served LoRA model names differ")


def one_sample(pred: dict, gold: list[dict], kinds: dict[str, str]) -> dict:
    calls = pred.get("tool_calls") or []
    pred_names, gold_names = [x["name"] for x in calls], [x["name"] for x in gold]
    matched = sum((Counter(pred_names) & Counter(gold_names)).values())
    aligned = [(p, g) for p, g in zip(calls, gold) if p["name"] == g["name"]]
    eligible_read_write = any(kinds.get(call["name"]) == "write" and
                              any(kinds.get(previous["name"]) == "read" for previous in gold[:index])
                              for index, call in enumerate(gold))
    predicted_read_before_gold_write = any(
        kinds.get(call["name"]) == "write" and call["name"] in gold_names and
        any(kinds.get(previous["name"]) == "read" for previous in calls[:index])
        for index, call in enumerate(calls)) if eligible_read_write else None
    ok = pred.get("status") == "ok"
    return {"state_match": ok and pred.get("train_state_match") is True,
            "tool_call_f1": exact_tool_f1(calls, gold) if ok else 0.0,
            "selection_exact": pred_names == gold_names and ok,
            "selection_matched": matched, "selection_predicted": len(calls), "selection_gold": len(gold),
            "argument_correct": sum(p["arguments"] == g["arguments"] for p, g in aligned) if ok else 0,
            "argument_eligible": len(aligned) if ok else 0,
            "read_to_write_eligible": eligible_read_write,
            "read_to_write_complete": bool(predicted_read_before_gold_write) and ok,
            "runtime_error": not ok, "error": pred.get("error"),
            "latency_seconds": pred["latency_seconds"]}


def summarize(name: str, path: Path, samples: list[dict]) -> dict:
    latencies = sorted(x["latency_seconds"] for x in samples)
    matched = sum(x["selection_matched"] for x in samples)
    selected = sum(x["selection_predicted"] for x in samples)
    gold = sum(x["selection_gold"] for x in samples)
    precision = matched / selected if selected else 0.0
    recall = matched / gold if gold else 0.0
    argument_eligible = sum(x["argument_eligible"] for x in samples)
    read_write_eligible = sum(x["read_to_write_eligible"] for x in samples)
    return {"configuration": name, "run_dir": str(path), "samples": len(samples),
            "state_matches": sum(x["state_match"] for x in samples),
            "state_match_rate": sum(x["state_match"] for x in samples) / len(samples),
            "tool_call_f1": statistics.mean(x["tool_call_f1"] for x in samples),
            "tool_selection_exact": sum(x["selection_exact"] for x in samples) / len(samples),
            "tool_selection_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "argument_accuracy_on_aligned_calls": sum(x["argument_correct"] for x in samples) / argument_eligible if argument_eligible else None,
            "argument_correct": sum(x["argument_correct"] for x in samples), "argument_eligible": argument_eligible,
            "read_to_write_completion": sum(x["read_to_write_complete"] for x in samples) / read_write_eligible if read_write_eligible else None,
            "read_to_write_correct": sum(x["read_to_write_complete"] for x in samples), "read_to_write_eligible": read_write_eligible,
            "runtime_errors": sum(x["runtime_error"] for x in samples),
            "latency_mean_seconds": statistics.mean(latencies),
            "latency_p95_seconds": latencies[(95 * len(latencies) + 99) // 100 - 1]}


def compare(s0: Path, s1: Path, oracle0: Path, oracle1: Path, output_dir: Path) -> dict:
    ids = json.loads((PROJECT / "outputs/splits/validation_ids.json").read_text(encoding="utf-8"))
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("frozen validation split must contain 200 unique IDs")
    runs = [load_run(name, path, ids) for name, path in (("S0", s0), ("S1", s1), ("S0-Oracle", oracle0), ("S1-Oracle", oracle1))]
    check_design(runs)
    root = PROJECT / "data/train+public_test"
    train = {item["id"]: item for item in items(root, "train") if item["id"] in set(ids)}
    kinds = {name: spec.get("kind", "unknown") for name, spec in catalog(root, "train").items()}
    evaluated = {}
    for run in runs:
        evaluated[run["name"]] = {pred["id"]: one_sample(pred, train[pred["id"]]["label"]["tool_calls"], kinds)
                                  for pred in run["predictions"]}
    summaries = [summarize(run["name"], run["path"], [evaluated[run["name"]][sid] for sid in ids]) for run in runs]
    transitions = {}
    rows = []
    for base_name, candidate_name in (("S0", "S1"), ("S0-Oracle", "S1-Oracle")):
        counts = Counter()
        for sid in ids:
            before = evaluated[base_name][sid]["state_match"]
            after = evaluated[candidate_name][sid]["state_match"]
            counts[("correct" if before else "incorrect") + "_to_" + ("correct" if after else "incorrect")] += 1
        transitions[candidate_name] = dict(counts)
    for sid in ids:
        rows.append({"id": sid, "models": {name: evaluated[name][sid] for name in evaluated},
                     "s1_vs_s0": ("correct" if evaluated["S0"][sid]["state_match"] else "incorrect") + "_to_" + ("correct" if evaluated["S1"][sid]["state_match"] else "incorrect"),
                     "s1_oracle_vs_s0_oracle": ("correct" if evaluated["S0-Oracle"][sid]["state_match"] else "incorrect") + "_to_" + ("correct" if evaluated["S1-Oracle"][sid]["state_match"] else "incorrect")})
    result = {"samples": 200, "adapter_dir": runs[1]["config"]["adapter_dir"],
              "summaries": summaries, "transitions": transitions,
              "metric_definitions": {"tool_selection_exact": "exact ordered tool-name sequence per sample",
                  "tool_selection_f1": "micro F1 of tool-name multisets",
                  "argument_accuracy_on_aligned_calls": "exact arguments among same-name calls aligned by sequence position",
                  "read_to_write_completion": "a gold write tool after a predicted read, among gold trajectories with read and write"}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_sample_regressions.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    lines = ["# LifeMate Agent QLoRA SFT comparison", "",
             "All configurations use the same 200 TRAIN validation IDs and E1/ToolEnv settings. S0 is a preserved result; S0-Oracle uses the unmodified base model and may be rerun when the prior raw run is unavailable. S1 and S1-Oracle use the same PEFT adapter. These are local TRAIN metrics, not an official score.", "",
             "| Configuration | State match | Tool-call F1 | Tool selection exact / F1 | Argument accuracy | Read→write | Runtime errors | Mean / P95 latency (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["summaries"]:
        argument = f"{row['argument_correct']}/{row['argument_eligible']} ({row['argument_accuracy_on_aligned_calls']:.1%})" if row["argument_accuracy_on_aligned_calls"] is not None else "n/a"
        read_write = f"{row['read_to_write_correct']}/{row['read_to_write_eligible']} ({row['read_to_write_completion']:.1%})" if row["read_to_write_completion"] is not None else "n/a"
        lines.append(f"| {row['configuration']} | {row['state_matches']}/200 ({row['state_match_rate']:.1%}) | {row['tool_call_f1']:.3f} | {row['tool_selection_exact']:.1%} / {row['tool_selection_f1']:.1%} | {argument} | {read_write} | {row['runtime_errors']} | {row['latency_mean_seconds']:.2f} / {row['latency_p95_seconds']:.2f} |")
    lines += ["", "## Paired state transitions", "", "| Comparison | Correct→incorrect | Incorrect→correct |",
              "| --- | ---: | ---: |"]
    for name, counts in result["transitions"].items():
        lines.append(f"| {name} | {counts.get('correct_to_incorrect', 0)} | {counts.get('incorrect_to_correct', 0)} |")
    base, trained = result["summaries"][0], result["summaries"][1]
    lines += ["", f"S1 changes state matches by {trained['state_matches'] - base['state_matches']:+d}/200 versus S0. Inspect paired regressions before choosing this adapter for deployment.", "",
              "Argument accuracy is conditional on aligned same-name calls; read→write completion is a sequence proxy. Saved Gipformer transcript times are included in S0/S1 latency, while oracle runs have no ASR time. Earlier baseline and current adapter runs occurred at different times, so latency comparisons are indicative.", "",
              "`per_sample_regressions.jsonl` contains all paired outcomes. Each run directory retains predictions, LLM/tool traces, runtime errors and replayable ToolEnv calls.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0", type=Path, default=PROJECT / "outputs/runs/e1_asr_full_01_A2")
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s0-oracle", type=Path, default=PROJECT / "outputs/runs/agentopt_full_01_E4")
    parser.add_argument("--s1-oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.s0, args.s1, args.s0_oracle, args.s1_oracle, args.output_dir)["transitions"], indent=2))
