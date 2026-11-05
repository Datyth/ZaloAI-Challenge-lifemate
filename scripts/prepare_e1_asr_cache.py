"""Reuse current ASR transcripts and decode E1 history audio offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.asr import load_asr  # noqa: E402
from lifemate.cli import audio_info  # noqa: E402
from lifemate.data import audio_path, items  # noqa: E402


def write(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(model: str, model_dir: Path, source_run: Path, output_dir: Path,
            ids_file: Path, device: str, sample_id: str | None = None) -> dict:
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    source_config = json.loads((source_run / "run_config.json").read_text(encoding="utf-8"))
    if source_config.get("sample_ids") != ids or source_config.get("split") != "train" or source_config.get("model_name") != model:
        raise ValueError("source ASR benchmark differs from frozen TRAIN IDs or requested model")
    data_root = PROJECT / "data" / "train+public_test"
    id_set = set(ids)
    selected = [item for item in items(data_root, "train") if item["id"] in id_set and (sample_id is None or item["id"] == sample_id)]
    expected_ids = [sample_id] if sample_id else ids
    if [item["id"] for item in selected] != expected_ids:
        raise ValueError("selected TRAIN IDs missing")
    model_dir = model_dir.resolve()
    source_rows = [json.loads(line) for line in (source_run / "asr_outputs.jsonl").open(encoding="utf-8") if line.strip()]
    if [row["sample_id"] for row in source_rows] != ids:
        raise ValueError("source ASR transcript file is incomplete or reordered")
    current = {row["sample_id"]: row for row in source_rows}
    for item in selected:
        row = current[item["id"]]
        if row.get("audio") != item["audio"] or row.get("model") != str(model_dir) or not isinstance(row.get("raw_transcript"), str):
            raise ValueError(f"source ASR transcript does not match model/audio for {item['id']}")
    output_dir.mkdir(parents=True, exist_ok=False)
    out = output_dir / "asr_outputs.jsonl"
    errors = output_dir / "errors.jsonl"
    out.touch()
    errors.touch()
    history = [(item, turn["audio"]) for item in selected for turn in item["history"]
               if turn.get("role") == "user" and "audio" in turn]
    config = {"split": "train", "sample_ids": [item["id"] for item in selected],
              "model_name": model, "model": str(model_dir), "device": device,
              "source_asr_run": str(source_run.resolve()), "history_audio": len(history),
              "checkpoint_manifest": json.loads((model_dir / "lifemate_manifest.json").read_text(encoding="utf-8"))}
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in selected:
        write(out, {**current[item["id"]], "source": "saved", "source_run": str(source_run.resolve())})
    backend = load_asr(model, model_dir, device) if history else None
    if backend is not None:
        # The saved current-utterance run already includes its own first decode.
        # Warm this process before timing history to avoid charging startup twice.
        backend.transcribe(audio_path(data_root, "train", selected[0]["audio"]))
    failures = 0
    for index, (item, audio) in enumerate(history, start=1):
        path = audio_path(data_root, "train", audio)
        start = time.monotonic()
        try:
            transcript = backend.transcribe(path)
            write(out, {"sample_id": item["id"], "audio": audio, "model": str(model_dir),
                        "raw_transcript": transcript, "latency_seconds": time.monotonic() - start,
                        "source": "inference", **audio_info(path)})
        except Exception as exc:
            failures += 1
            write(errors, {"sample_id": item["id"], "audio": audio,
                           "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(history):
            print(f"{model} history ASR: {index}/{len(history)}; failures={failures}", file=sys.stderr, flush=True)
    metrics = {"samples": len(selected), "saved_current_transcripts": len(selected),
               "history_transcripts": len(history) - failures, "history_failures": failures,
               "complete": failures == 0}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{failures} history ASR failures; inspect {errors}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chunkformer", "gipformer"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path, default=PROJECT / "outputs/splits/validation_ids.json")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.model_dir, args.source_run, args.output_dir,
                             args.sample_ids, args.device, args.sample_id), indent=2))
