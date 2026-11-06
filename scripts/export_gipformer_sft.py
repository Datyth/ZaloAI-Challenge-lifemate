"""Export a strictly checked adapted PyTorch checkpoint to a new ONNX directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))
from gipformer_pt import ORIGINAL, sha256, strict_model  # noqa: E402
from lifemate.asr import GipformerASR  # noqa: E402
from lifemate.data import audio_path, items  # noqa: E402


def export(best: Path, output_dir: Path, icefall_root: Path, device: str) -> dict:
    import torch
    original, _ = strict_model(ORIGINAL, icefall_root)
    saved = torch.load(best, map_location="cpu", weights_only=True)
    if saved.get("source_sha256") != sha256(ORIGINAL / "model.pt"):
        raise ValueError("adapted checkpoint source hash differs from original Gipformer")
    if set(saved["model"]) != set(original.state_dict()):
        raise ValueError("adapted checkpoint keys differ from strictly loaded model")
    original.load_state_dict(saved["model"], strict=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("tokens.txt", "bpe.model", "config.json"):
        shutil.copy2(ORIGINAL / name, output_dir / name)
    torch.save({"model": saved["model"]}, output_dir / "model.pt")
    shutil.copy2(output_dir / "model.pt", output_dir / "epoch-1.pt")
    source = icefall_root / "egs/librispeech/ASR/zipformer/export-onnx.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(icefall_root), str(icefall_root / "egs/librispeech/ASR"),
        str(icefall_root / "egs/librispeech/ASR/zipformer"), env.get("PYTHONPATH", "")])
    subprocess.run([sys.executable, str(source), "--tokens", str(output_dir / "tokens.txt"),
        "--exp-dir", str(output_dir), "--epoch", "1", "--avg", "1",
        "--use-averaged-model", "false", "--enable-int8-quantization", "0",
        "--use-ctc", "true"], check=True, env=env)
    for part in ("encoder", "decoder", "joiner"):
        built = output_dir / f"{part}-epoch-1-avg-1.onnx"
        if not built.is_file() or built.stat().st_size < 1024:
            raise FileNotFoundError(f"ONNX export missing {built}")
        built.rename(output_dir / f"{part}.onnx")
    recognizer = GipformerASR(output_dir, device)
    sample = next(items(PROJECT / "data/train+public_test", "train"))
    transcript = recognizer.transcribe(audio_path(PROJECT / "data/train+public_test", "train", sample["audio"]))
    if not transcript:
        raise RuntimeError("exported ONNX model produced an empty smoke transcript")
    manifest = {"model": "Gipformer 1.5 LifeMate adapted", "source_sha256": saved["source_sha256"],
                "adapted_checkpoint": str(best.resolve()), "adapted_sha256": sha256(best),
                "onnx_sha256": {part: sha256(output_dir / f"{part}.onnx") for part in ("encoder", "decoder", "joiner")},
                "smoke_sample_id": sample["id"], "smoke_transcript": transcript}
    (output_dir / "lifemate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, default=PROJECT / "outputs/asr_sft/gipformer_run_01/best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "models/gipformer1.5-lifemate-sft-01")
    parser.add_argument("--icefall-root", type=Path, default=PROJECT / "models/.research/icefall")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    print(json.dumps(export(args.best, args.output_dir, args.icefall_root, args.device), ensure_ascii=False, indent=2))
