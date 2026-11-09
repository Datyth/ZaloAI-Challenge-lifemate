"""Strictly reconstruct the original Gipformer icefall model from model.pt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL = PROJECT / "models/gipformer1.5-68m-rnnt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(model_dir: Path) -> dict:
    import torch
    path = model_dir / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != {"model"} or not isinstance(raw["model"], dict):
        raise ValueError("expected original icefall model.pt with a single 'model' state dict")
    state = raw["model"]
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("model.pt contains non-tensor or empty state")
    required = ("encoder_embed.", "encoder.", "decoder.", "joiner.", "simple_am_proj.", "simple_lm_proj.")
    if any(not any(key.startswith(prefix) for key in state) for prefix in required):
        raise ValueError("model.pt lacks required Zipformer RNNT modules")
    return {"checkpoint": str(path.resolve()), "sha256": sha256(path), "torch": torch.__version__,
            "tensors": len(state), "tensor_values": sum(value.numel() for value in state.values()),
            "has_ctc_head": any(key.startswith("ctc_output.") for key in state),
            "required_modules": {name: importlib.util.find_spec(name) is not None
                                 for name in ("k2", "kaldifeat", "lhotse", "sentencepiece", "torchaudio")}}


def strict_model(model_dir: Path, icefall_root: Path):
    import torch
    required = [icefall_root / "icefall/__init__.py",
                icefall_root / "egs/librispeech/ASR/zipformer/train.py"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"icefall source missing; expected {required}")
    for name in ("k2", "kaldifeat", "lhotse", "sentencepiece", "torchaudio"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"{name} unavailable; use an isolated icefall environment, not the ONNX inference environment")
    for path in (icefall_root, icefall_root / "egs/librispeech/ASR",
                 icefall_root / "egs/librispeech/ASR/zipformer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import k2
    from train import get_model, get_params, get_parser
    parser = get_parser()
    model_args = parser.parse_args([])
    params = get_params()
    params.update(vars(model_args))
    tokens = k2.SymbolTable.from_file(str(model_dir / "tokens.txt"))
    params.blank_id = tokens["<blk>"]
    params.unk_id = tokens["<unk>"]
    num_tokens = sum(not symbol.startswith("#") for symbol in tokens.symbols)
    if params.blank_id != 0:
        raise ValueError("expected blank token ID 0")
    # Upstream subtracts blank and adds it back for the model vocabulary.
    params.vocab_size = num_tokens
    params.use_ctc = True  # The original checkpoint contains the auxiliary CTC head.
    model = get_model(params)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(checkpoint, strict=True)
    return model, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL)
    parser.add_argument("--icefall-root", type=Path, default=PROJECT / "models/.research/icefall")
    parser.add_argument("--strict-load", action="store_true")
    args = parser.parse_args()
    result = inspect_checkpoint(args.model_dir)
    if args.strict_load:
        model, params = strict_model(args.model_dir, args.icefall_root)
        result["strict_load"] = True
        result["model_parameters"] = sum(p.numel() for p in model.parameters())
        result["icefall_root"] = str(args.icefall_root.resolve())
        result["vocab_size"] = params.vocab_size
    print(json.dumps(result, indent=2))
