"""Synthesise outputs/asr_tts/<v>/texts.jsonl with VieNeu-TTS v3 Turbo (ONNX, CPU), voice cloned from each row's TRAIN clip,
resampled to 16 kHz mono like the TRAIN audio. Shard k of n; resumable (skips existing wavs).
    .venv-tts/bin/python scripts/run_tts.py --texts outputs/asr_tts/v1/texts.jsonl --shard 0 --shards 4 --threads 5"""
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, soxr
import vieneu._v3_turbo_engine.onnx_runtime_lite as L

PROJECT = Path(__file__).resolve().parents[1]
_orig = L.OnnxV3LiteEngine.__init__
def _init(self, *a, **k):
    k.setdefault("codec_dir", str(PROJECT / "models/.tts/moss-audio-tokenizer-nano-onnx"))
    return _orig(self, *a, **k)
L.OnnxV3LiteEngine.__init__ = _init
from vieneu import Vieneu

ap = argparse.ArgumentParser()
ap.add_argument("--texts", type=Path, required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--shards", type=int, default=1)
ap.add_argument("--threads", type=int, default=5)
ap.add_argument("--int8", action="store_true", help="int8 ONNX backbone (needs VNNI; ~3x faster)")
a = ap.parse_args()
rows = [json.loads(l) for l in a.texts.open(encoding="utf-8")][a.shard::a.shards]
out_dir = a.texts.parent / "audio"
out_dir.mkdir(parents=True, exist_ok=True)
tts = Vieneu(backbone_repo=str(PROJECT / "models/.tts/vieneu-v3-turbo"), onnx_dir=str(PROJECT / ("models/.tts/vieneu-v3-turbo/onnx_int8" if a.int8 else "models/.tts/vieneu-v3-turbo/onnx_update")), precision="int8" if a.int8 else "fp32",
             backend="onnx", threads=a.threads)
root = PROJECT / "data/train+public_test/train"
t0 = time.time()
with (a.texts.parent / f"done_s{a.shard}.jsonl").open("a", encoding="utf-8") as log:
    for i, r in enumerate(rows):
        wav = out_dir / f"{r['id']}.wav"
        if wav.exists():
            continue
        try:
            audio = np.asarray(tts.infer(r["tts_text"], ref_audio=str(root / r["voice"])), dtype=np.float32).reshape(-1)
            audio16 = soxr.resample(audio, 48000, 16000)
            peak = float(np.abs(audio16).max() or 1.0)
            sf.write(str(wav), (audio16 / peak * 0.7).astype(np.float32), 16000)
            log.write(json.dumps({"id": r["id"], "seconds": round(len(audio16) / 16000, 2)}) + "\n"); log.flush()
        except Exception as exc:
            log.write(json.dumps({"id": r["id"], "error": f"{type(exc).__name__}: {exc}"}) + "\n"); log.flush()
        if i % 50 == 0:
            print(f"shard {a.shard}: {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
print(f"shard {a.shard} done {time.time() - t0:.0f}s", flush=True)
