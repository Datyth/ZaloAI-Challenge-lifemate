"""Whisper-family transcripts (Hugging Face transformers, GPU) of every current and history audio turn of a split, in the
same cache format as scripts/infer/asr_cache.py (asr_outputs.jsonl + run_config.json + metrics.json), so analysis/hotword_eval.py
and the agent's cached-asr mode read it unchanged. Raw model text is kept (case, punctuation, digits); scoring renders it.
  .venv/bin/python analysis/whisper_cache.py --split train --ids configs/splits/validation_ids.json \
      --model-dir models/phowhisper-medium --output-dir outputs/asr/phowhisper-medium_zs_val"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import audio_path, items  # noqa: E402

ROOT = PROJECT / "data/train+public_test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "public_test"], required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ids", type=Path, help="only these sample ids")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1)
    a = ap.parse_args()
    rows = list(items(ROOT, a.split))
    if a.ids:
        keep = set(json.loads(a.ids.read_text()))
        rows = [r for r in rows if r["id"] in keep]
    jobs = [(r["id"], audio) for r in rows
            for audio in dict.fromkeys([r["audio"], *(t["audio"] for t in r["history"] if t.get("role") == "user" and "audio" in t)])]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = str(a.model_dir.resolve())
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, dtype=torch.float16).to("cuda").eval()
    n_params = sum(p.numel() for p in model.parameters())
    (a.output_dir / "run_config.json").write_text(json.dumps({
        "split": a.split, "model_dir": model_dir, "params": n_params, "batch": a.batch, "beams": a.beams, "language": "vi",
        "ids": str(a.ids) if a.ids else None}, indent=2))
    t0 = time.time()
    audio_seconds = 0.0
    jobs.sort(key=lambda j: sf.info(str(audio_path(ROOT, a.split, j[1]))).frames)  # similar lengths per batch
    with (a.output_dir / "asr_outputs.jsonl").open("w", encoding="utf-8") as out:
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            waves = []
            for _, audio in chunk:
                w, sr = sf.read(str(audio_path(ROOT, a.split, audio)), dtype="float32")
                assert sr == 16000, (audio, sr)
                waves.append(w if w.ndim == 1 else w.mean(axis=1))
                audio_seconds += len(waves[-1]) / sr
            feats = processor(waves, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = model.generate(feats.input_features.to("cuda", torch.float16), language="vi", task="transcribe",
                                     num_beams=a.beams, max_new_tokens=220)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for (sample_id, audio), text in zip(chunk, texts):
                out.write(json.dumps({"sample_id": sample_id, "audio": audio, "raw_transcript": text.strip(),
                                      "model": model_dir}, ensure_ascii=False) + "\n")
            print(f"{min(b + a.batch, len(jobs))}/{len(jobs)} audio, {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    (a.output_dir / "metrics.json").write_text(json.dumps({"samples": len(rows), "audio": len(jobs), "params": n_params,
                                                          "audio_seconds": round(audio_seconds, 1), "wall_seconds": round(wall, 1),
                                                          "rtf": round(wall / audio_seconds, 4)}, indent=2))
    print(f"done {len(jobs)} audio in {wall:.0f}s, params {n_params / 1e6:.0f}M")


if __name__ == "__main__":
    main()
