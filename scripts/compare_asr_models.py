"""Compare complete ASR runs on the frozen LifeMate TRAIN validation IDs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.asr_eval import critical_slots, labelled_slices, slot_hits  # noqa: E402
from lifemate.data import items  # noqa: E402
from lifemate.metrics import error_counts  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_model_run(name: str, run_dir: Path, sample_ids: list[str], train: dict[str, dict]) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("split") != "train" or config.get("sample_ids") != sample_ids:
        raise ValueError(f"{name}: run did not use the frozen validation IDs in order")
    if (run_dir / "metrics.json").is_file():
        recorded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if recorded.get("failures", 0):
            raise ValueError(f"{name}: run has {recorded['failures']} inference failures")
    outputs = read_jsonl(run_dir / "asr_outputs.jsonl")
    if [x["sample_id"] for x in outputs] != sample_ids:
        raise ValueError(f"{name}: incomplete or reordered ASR outputs ({len(outputs)}/{len(sample_ids)})")
    rows: list[dict] = []
    for output in outputs:
        sample = train[output["sample_id"]]
        transcript = output["raw_transcript"]
        counts = error_counts(sample["audio_text"], transcript)
        slots = slot_hits(critical_slots(sample["audio_text"], sample["label"]["tool_calls"]), transcript)
        rows.append({"id": sample["id"], "reference": sample["audio_text"], "transcript": transcript,
                     "counts": counts, "critical_slots": slots, "latency_seconds": output["latency_seconds"],
                     "duration_seconds": output["duration_seconds"],
                     "slices": labelled_slices(sample, output)})
    summary = summarize(name, run_dir, rows)
    return summary, rows


def summarize(name: str, run_dir: Path, rows: list[dict]) -> dict:
    counts = {key: sum(row["counts"][key] for row in rows) for key in ("word_errors", "reference_words", "character_errors", "reference_characters")}
    slots = [slot for row in rows for slot in row["critical_slots"]]
    latency = sorted(row["latency_seconds"] for row in rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for label in row["slices"]:
            groups[label].append(row)
    slice_metrics = {}
    for label, members in sorted(groups.items()):
        word_count = sum(x["counts"]["reference_words"] for x in members)
        char_count = sum(x["counts"]["reference_characters"] for x in members)
        eligible = [slot for x in members for slot in x["critical_slots"]]
        slice_metrics[label] = {"samples": len(members),
                                "wer": sum(x["counts"]["word_errors"] for x in members) / word_count if word_count else None,
                                "cer": sum(x["counts"]["character_errors"] for x in members) / char_count if char_count else None,
                                "critical_slot_accuracy": sum(x["matched"] for x in eligible) / len(eligible) if eligible else None,
                                "critical_slot_eligible": len(eligible)}
    return {"model": name, "run_dir": str(run_dir), "samples": len(rows),
            "wer": counts["word_errors"] / counts["reference_words"],
            "cer": counts["character_errors"] / counts["reference_characters"],
            "critical_slot_accuracy": sum(slot["matched"] for slot in slots) / len(slots) if slots else None,
            "critical_slot_correct": sum(slot["matched"] for slot in slots), "critical_slot_eligible": len(slots),
            "latency_mean_seconds": sum(latency) / len(latency),
            "latency_p95_seconds": latency[(95 * len(latency) + 99) // 100 - 1],
            "slices": slice_metrics}


def render_report(summaries: list[dict]) -> str:
    winner = min(summaries, key=lambda x: (x["wer"], -(x["critical_slot_accuracy"] or 0), x["latency_mean_seconds"]))
    lines = ["# LifeMate ASR comparison", "",
             "All models were evaluated on the same frozen 200-sample TRAIN split. WER/CER use lowercase Unicode word normalization without number verbalization. Latency excludes model load and includes the first-sample warmup; the preserved Qwen run was measured earlier, so latency comparisons are indicative. The critical-slot metric is exact lexical preservation of gold argument values explicitly present in the reference utterance; it does not cover derived dates, IDs or paraphrases.", "",
             "| Model | WER | CER | Critical slots | Mean / P95 latency (s) |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        slot = f"{row['critical_slot_accuracy']:.1%} ({row['critical_slot_correct']}/{row['critical_slot_eligible']})" if row["critical_slot_accuracy"] is not None else "unavailable (0 eligible)"
        lines.append(f"| {row['model']} | {row['wer']:.1%} | {row['cer']:.1%} | {slot} | {row['latency_mean_seconds']:.3f} / {row['latency_p95_seconds']:.3f} |")
    lines += ["", f"**WER-first winner: {winner['model']}.** Compare critical-slot accuracy and latency before changing the pipeline ASR model; this benchmark does not measure downstream tool-call accuracy.", "", "## Available slices", ""]
    all_slices = sorted({key for row in summaries for key in row["slices"]})
    if all_slices:
        lines += ["| Slice | Model | Samples | WER | CER | Critical slots |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for label in all_slices:
            for row in summaries:
                x = row["slices"].get(label)
                if x:
                    slot = f"{x['critical_slot_accuracy']:.1%} ({x['critical_slot_eligible']} eligible)" if x["critical_slot_accuracy"] is not None else "unavailable"
                    lines.append(f"| {label} | {row['model']} | {x['samples']} | {x['wer']:.1%} | {x['cer']:.1%} | {slot} |")
    else:
        lines.append("No slice labels were available.")
    observed = set(all_slices)
    missing = [label for label, prefix in (("regional accent", "accent:"), ("noisy speech", "noisy_labelled"), ("truncated utterance", "truncated_labelled")) if not any(x.startswith(prefix) for x in observed)]
    if missing:
        lines += ["", "The TRAIN records have no labels for " + ", ".join(missing) + "; no accuracy claim is made for those groups. Quiet/clipped slices are signal heuristics, not noise labels."]
    lines += ["", "## Fine-tuning suitability", "",
              "ChunkFormer has a local PyTorch checkpoint and published training code; Gipformer 1.5 provides `model.pt` and a separate research training stack. Both are technical fine-tuning candidates, but this benchmark alone does not establish downstream gains. ChunkFormer is CC BY 4.0 and Gipformer is MIT; review competition rules and each training stack before any fine-tuning. No weights were changed here.", "",
              "Raw transcripts, per-sample metrics and error logs are in each model's run directory. `per_sample_comparison.jsonl` contains aligned records for the 200 IDs.", ""]
    return "\n".join(lines)


def compare(qwen: Path, chunkformer: Path, gipformer: Path, output_dir: Path) -> dict:
    sample_ids = json.loads((PROJECT / "outputs" / "splits" / "validation_ids.json").read_text(encoding="utf-8"))
    train = {item["id"]: item for item in items(PROJECT / "data" / "train+public_test", "train") if item["id"] in set(sample_ids)}
    summaries: list[dict] = []
    aligned: dict[str, dict] = {sample_id: {"id": sample_id, "reference": train[sample_id]["audio_text"]} for sample_id in sample_ids}
    for name, run_dir in (("qwen", qwen), ("chunkformer", chunkformer), ("gipformer", gipformer)):
        summary, rows = load_model_run(name, run_dir, sample_ids, train)
        summaries.append(summary)
        for row in rows:
            aligned[row["id"]][name] = {key: value for key, value in row.items() if key not in {"id", "reference"}}
    winner = min(summaries, key=lambda x: (x["wer"], -(x["critical_slot_accuracy"] or 0), x["latency_mean_seconds"]))["model"]
    result = {"sample_ids": len(sample_ids), "winner_by_wer": winner, "models": summaries,
              "limitations": "Critical-slot metric covers only explicitly spoken gold argument values; accent/noise/truncation require labels."}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_sample_comparison.jsonl").open("w", encoding="utf-8") as out:
        for sample_id in sample_ids:
            out.write(json.dumps(aligned[sample_id], ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(render_report(summaries), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=PROJECT / "outputs" / "runs" / "baseline_asr_200")
    parser.add_argument("--chunkformer", type=Path, required=True)
    parser.add_argument("--gipformer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.qwen, args.chunkformer, args.gipformer, args.output_dir)
    print(json.dumps({"winner_by_wer": result["winner_by_wer"], "report": str(args.output_dir / "report.md")}, indent=2))
