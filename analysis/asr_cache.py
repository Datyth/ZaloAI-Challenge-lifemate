"""Gipformer transcripts of every current and history audio turn of a split, decoded on CPU in parallel.
Writes the same cache format as scripts/cache_gipformer_audio.py (asr_outputs.jsonl + run_config.json + metrics.json), so
`lifemate.cli evaluate --transcript-source cached-asr --transcripts-jsonl <dir>/asr_outputs.jsonl` reads it unchanged.
  .venv/bin/python analysis/asr_cache.py --split train --output-dir outputs/asr/gipformer_train --workers 10
Optional --hotwords file (one phrase per line) switches to sherpa-onnx contextual biasing (modified beam search).
--context-hotwords biases each sample with its own phrases (lifemate.context.hotword_phrases: user profile and history tool
calls/results, i.e. only what inference sees), at --hotwords-score."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.cli import audio_info  # noqa: E402
from lifemate.context import hotword_phrases  # noqa: E402
from lifemate.data import audio_path, items  # noqa: E402

ROOT = PROJECT / "data/train+public_test"
_rec = None


def _init(model_dir: str, threads: int, method: str, beam: int, hotwords: str | None = None, score: float = 1.5, per_stream: bool = False) -> None:
    global _rec
    import sherpa_onnx
    m = Path(model_dir)
    _rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(m / "encoder.onnx"), decoder=str(m / "decoder.onnx"), joiner=str(m / "joiner.onnx"),
        tokens=str(m / "tokens.txt"), num_threads=threads, sample_rate=16000, feature_dim=80,
        decoding_method=method, max_active_paths=beam, provider="cpu",
        **({"hotwords_file": hotwords or "", "hotwords_score": score, "modeling_unit": "bpe", "bpe_vocab": str(m / "bpe.vocab")} if hotwords or per_stream else {}))


def _decode(job: tuple) -> dict:
    import soundfile as sf
    split, sample_id, audio, phrases = job
    path = audio_path(ROOT, split, audio)
    t0 = time.monotonic()
    wav, rate = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    stream = _rec.create_stream(hotwords="/".join(phrases)) if phrases else _rec.create_stream()
    stream.accept_waveform(rate, wav)
    _rec.decode_streams([stream])
    return {"sample_id": sample_id, "audio": audio, "raw_transcript": stream.result.text.strip(), **({"hotwords": phrases} if phrases else {}),
            "latency_seconds": time.monotonic() - t0, **audio_info(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "public_test"], required=True)
    ap.add_argument("--model-dir", type=Path, default=PROJECT / "models/gipformer1.5-68m-rnnt")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--method", default="modified_beam_search")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--hotwords", type=Path)
    ap.add_argument("--hotwords-score", type=float, default=1.5)
    ap.add_argument("--ids", type=Path, help="only these sample ids")
    ap.add_argument("--context-hotwords", action="store_true", help="per-sample hotwords from the user profile and history tool data")
    a = ap.parse_args()
    rows = list(items(ROOT, a.split))
    if a.ids:
        keep = set(json.loads(a.ids.read_text()))
        rows = [r for r in rows if r["id"] in keep]
    jobs = [(a.split, r["id"], audio, tuple(hotword_phrases(r)) if a.context_hotwords else ()) for r in rows
            for audio in dict.fromkeys([r["audio"], *(t["audio"] for t in r["history"] if t.get("role") == "user" and "audio" in t)])]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    model = str(a.model_dir.resolve())
    (a.output_dir / "run_config.json").write_text(json.dumps({
        "split": a.split, "model_name": "gipformer", "model": model, "device": "cpu", "method": a.method, "beam": a.beam, "hotwords": str(a.hotwords) if a.hotwords else None, "hotwords_score": a.hotwords_score,
        "context_hotwords": a.context_hotwords, "sample_ids": [r["id"] for r in rows], "current_audio": len(rows), "history_audio": len(jobs) - len(rows), "no_gold_used": True}, indent=2) + "\n")
    done = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(model, a.threads, a.method, a.beam, str(a.hotwords) if a.hotwords else None, a.hotwords_score, a.context_hotwords)) as pool, \
            (a.output_dir / "asr_outputs.jsonl").open("w", encoding="utf-8") as out:
        for row in pool.map(_decode, jobs, chunksize=8):
            out.write(json.dumps({**row, "model": model}, ensure_ascii=False) + "\n")
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"{a.split}: {done}/{len(jobs)}", flush=True)
    (a.output_dir / "metrics.json").write_text(json.dumps({"samples": len(rows), "audio": len(jobs), "decoded": done, "failures": 0,
                                                            "complete": done == len(jobs)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
