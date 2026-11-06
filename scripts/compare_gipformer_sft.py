"""Paired TRAIN ASR + S1 Agent comparison; gate adapted checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.asr_eval import critical_slots, slot_hits  # noqa: E402
from lifemate.data import items  # noqa: E402
from lifemate.metrics import error_counts  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def summarize_asr(cache: Path, samples: list[dict]) -> tuple[dict, dict]:
    records = rows(cache / "asr_outputs.jsonl")
    lookup = {(r["sample_id"], r["audio"]): r for r in records}
    expected = {(item["id"], item["audio"]) for item in samples}
    expected |= {(item["id"], turn["audio"]) for item in samples for turn in item["history"]
                 if turn.get("role") == "user" and "audio" in turn}
    if set(lookup) != expected or len(records) != len(expected):
        raise ValueError(f"ASR cache {cache} lacks current/history audio or has duplicates")
    totals = {scope: {"word_errors": 0, "reference_words": 0, "character_errors": 0,
                      "reference_characters": 0, "slots": 0, "slot_correct": 0, "latencies": []}
              for scope in ("current", "history")}
    by_sample = {}
    for item in samples:
        one = {}
        turns = [("current", item["audio"], item["audio_text"])]
        turns += [("history", turn["audio"], turn["audio_text"]) for turn in item["history"]
                  if turn.get("role") == "user" and "audio" in turn]
        for scope, audio, reference in turns:
            record = lookup[(item["id"], audio)]
            counts = error_counts(reference, record["raw_transcript"])
            for key, value in counts.items():
                totals[scope][key] += value
            totals[scope]["latencies"].append(record["latency_seconds"])
            if scope == "current":
                hits = slot_hits(critical_slots(reference, item["label"]["tool_calls"]), record["raw_transcript"])
                totals[scope]["slots"] += len(hits)
                totals[scope]["slot_correct"] += sum(hit["matched"] for hit in hits)
                one = {"current_transcript": record["raw_transcript"], "current_word_errors": counts["word_errors"],
                       "critical_slots": len(hits), "critical_slot_correct": sum(hit["matched"] for hit in hits)}
        by_sample[item["id"]] = one
    summary = {}
    for scope, total in totals.items():
        latencies = sorted(total.pop("latencies"))
        summary[scope] = {"audio": len(latencies), "wer": total["word_errors"] / total["reference_words"],
                          "cer": total["character_errors"] / total["reference_characters"],
                          "critical_slot_accuracy": total["slot_correct"] / total["slots"] if total["slots"] else None,
                          "critical_slot_eligible": total["slots"],
                          "latency_mean_seconds": statistics.mean(latencies),
                          "latency_p95_seconds": latencies[(95 * len(latencies) + 99) // 100 - 1]}
    return summary, by_sample


def summarize_agent(run: Path, ids: list[str]) -> tuple[dict, dict]:
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    predictions = rows(run / "predictions.jsonl")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if config.get("sample_ids") != ids or [r.get("id") for r in predictions] != ids or metrics.get("replay_hash_matches") != 200:
        raise ValueError(f"incomplete 200-row E1/S1 Agent run: {run}")
    good = {r["id"]: r.get("status") == "ok" and r.get("train_state_match") is True for r in predictions}
    return {"state_matches": sum(good.values()), "runtime_errors": sum(r.get("status") != "ok" for r in predictions),
            "latency_mean_seconds": metrics["latency_mean_seconds"],
            "latency_p95_seconds": metrics["latency_p95_seconds"],
            "adapter_weights_sha256": config.get("adapter_weights_sha256"),
            "prompt_sha256": config.get("prompt_sha256")}, good


def compare(base_cache: Path, candidate_cache: Path, base_run: Path, candidate_run: Path,
            output_dir: Path, base_model: Path, candidate_model: Path) -> dict:
    ids = json.loads((PROJECT / "outputs/splits/validation_ids.json").read_text(encoding="utf-8"))
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("frozen TRAIN validation IDs invalid")
    selected = [item for item in items(PROJECT / "data/train+public_test", "train") if item["id"] in set(ids)]
    if [item["id"] for item in selected] != ids:
        raise ValueError("TRAIN IDs reordered")
    base_asr, base_samples = summarize_asr(base_cache, selected)
    candidate_asr, candidate_samples = summarize_asr(candidate_cache, selected)
    base_agent, base_correct = summarize_agent(base_run, ids)
    candidate_agent, candidate_correct = summarize_agent(candidate_run, ids)
    if not base_agent["adapter_weights_sha256"] or base_agent["adapter_weights_sha256"] != candidate_agent["adapter_weights_sha256"] or base_agent["prompt_sha256"] != candidate_agent["prompt_sha256"]:
        raise ValueError("Agent adapter or prompt differs between baseline and candidate")
    checks = {"state_match_improved": candidate_agent["state_matches"] > base_agent["state_matches"],
              "runtime_errors_not_worse": candidate_agent["runtime_errors"] <= base_agent["runtime_errors"],
              "p95_latency_acceptable": candidate_agent["latency_p95_seconds"] <= max(10.0, 1.25 * base_agent["latency_p95_seconds"]),
              "asr_p95_latency_acceptable": candidate_asr["current"]["latency_p95_seconds"] <= max(1.0, 1.5 * base_asr["current"]["latency_p95_seconds"]),
              "wer_not_worse_by_5pct": candidate_asr["current"]["wer"] <= 1.05 * base_asr["current"]["wer"],
              "history_wer_not_worse_by_5pct": candidate_asr["history"]["wer"] <= 1.05 * base_asr["history"]["wer"],
              "slot_accuracy_not_worse_by_1pt": candidate_asr["current"]["critical_slot_accuracy"] >= base_asr["current"]["critical_slot_accuracy"] - 0.01}
    selected_model = candidate_model if all(checks.values()) else base_model
    transitions = {"correct_to_incorrect": sum(base_correct[sid] and not candidate_correct[sid] for sid in ids),
                   "incorrect_to_correct": sum(not base_correct[sid] and candidate_correct[sid] for sid in ids)}
    result = {"baseline": {"asr": base_asr, "agent": base_agent, "model_dir": str(base_model.resolve())},
              "candidate": {"asr": candidate_asr, "agent": candidate_agent, "model_dir": str(candidate_model.resolve())},
              "checks": checks, "transitions": transitions, "selected_asr_dir": str(selected_model.resolve()),
              "selection_note": "candidate only if all gates pass; otherwise preserve original Gipformer + S1"}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as out:
        for sid in ids:
            out.write(json.dumps({"id": sid, "baseline": {**base_samples[sid], "state_match": base_correct[sid]},
                                  "candidate": {**candidate_samples[sid], "state_match": candidate_correct[sid]}},
                                 ensure_ascii=False) + "\n")
    lines = ["# Gipformer LifeMate adaptation", "", "| Configuration | Current WER | Current CER | Slots | History WER | State match | Runtime errors | P95 latency |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("baseline", "candidate"):
        a, g = result[name]["asr"], result[name]["agent"]
        lines.append(f"| {name} | {a['current']['wer']:.1%} | {a['current']['cer']:.1%} | {a['current']['critical_slot_accuracy']:.1%} | {a['history']['wer']:.1%} | {g['state_matches']}/200 | {g['runtime_errors']} | {g['latency_p95_seconds']:.2f}s |")
    lines += ["", f"Selected: `{result['selected_asr_dir']}`", "", f"Gates: `{json.dumps(checks)}`", "",
              f"Paired transitions: {transitions['incorrect_to_correct']} incorrect→correct, {transitions['correct_to_incorrect']} correct→incorrect.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=PROJECT / "outputs/runs/e1_asr_full_01_A2_cache")
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, default=PROJECT / "outputs/runs/sft_01_S1")
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=PROJECT / "models/gipformer1.5-68m-rnnt")
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.base_cache, args.candidate_cache, args.base_run, args.candidate_run,
                             args.output_dir, args.base_model, args.candidate_model)["checks"], indent=2))
