"""Decode every current and history audio turn into a split-specific Gipformer cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.asr import GipformerASR  # noqa: E402
from lifemate.cli import audio_info  # noqa: E402
from lifemate.data import audio_path, items  # noqa: E402


def audio_pairs(item: dict) -> list[str]:
    return list(dict.fromkeys([item["audio"], *(turn["audio"] for turn in item["history"]
                                            if turn.get("role") == "user" and "audio" in turn)]))


def prepare(split: str, model_dir: Path, output_dir: Path, device: str,
            sample_ids: Path | None = None, limit: int | None = None) -> dict:
    root = PROJECT / "data/train+public_test"
    selected = list(items(root, split))
    if sample_ids:
        ids = json.loads(sample_ids.read_text(encoding="utf-8"))
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sample IDs")
        wanted = set(ids)
        selected = [row for row in selected if row["id"] in wanted]
        if [row["id"] for row in selected] != ids:
            raise ValueError("sample IDs missing or reordered")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no samples selected")
    model_dir = model_dir.resolve()
    if not all((model_dir / name).is_file() for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")):
        raise FileNotFoundError(f"incomplete Gipformer ONNX model: {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    pairs = [(row["id"], audio) for row in selected for audio in audio_pairs(row)]
    config = {"split": split, "sample_ids": [row["id"] for row in selected], "model_name": "gipformer",
              "model": str(model_dir), "device": device, "current_audio": len(selected),
              "history_audio": len(pairs) - len(selected), "no_gold_used": True}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    transcripts = output_dir / "asr_outputs.jsonl"
    errors = output_dir / "errors.jsonl"
    backend = GipformerASR(model_dir, device)
    failures = 0
    with transcripts.open("w", encoding="utf-8") as out, errors.open("w", encoding="utf-8") as err:
        for index, (sample_id, audio) in enumerate(pairs, start=1):
            path = audio_path(root, split, audio)
            started = time.monotonic()
            try:
                text = backend.transcribe(path)
                row = {"sample_id": sample_id, "audio": audio, "model": str(model_dir),
                       "raw_transcript": text, "latency_seconds": time.monotonic() - started,
                       **audio_info(path)}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            except Exception as exc:
                failures += 1
                err.write(json.dumps({"sample_id": sample_id, "audio": audio,
                                      "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
                err.flush()
            if index % 50 == 0 or index == len(pairs):
                print(f"Gipformer {split}: {index}/{len(pairs)} audio; failures={failures}", file=sys.stderr, flush=True)
    result = {"samples": len(selected), "audio": len(pairs), "decoded": len(pairs) - failures,
              "failures": failures, "complete": failures == 0}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{failures} ASR failures; inspect {errors}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "public_test"], required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/gipformer1.5-68m-rnnt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--limit", type=int, help="small smoke test only")
    args = parser.parse_args()
    print(json.dumps(prepare(args.split, args.model_dir, args.output_dir, args.device, args.sample_ids, args.limit), indent=2))
