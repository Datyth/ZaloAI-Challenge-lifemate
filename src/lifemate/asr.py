"""Offline ASR adapters using only locally downloaded checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class ASRBackend(Protocol):
    def transcribe(self, path: Path) -> str: ...


class QwenASR:
    def __init__(self, model_dir: Path, device: str = "cuda:0") -> None:
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"ASR checkpoint missing: {model_dir}; run download first")
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError("Install the official qwen-asr package in a compatible Python environment") from exc
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            str(model_dir), dtype=dtype, device_map=device, max_new_tokens=256
        )

    def transcribe(self, path: Path) -> str:
        results = self.model.transcribe(audio=str(path), language="Vietnamese")
        return str(results[0].text)


class ChunkFormerASR:
    def __init__(self, model_dir: Path, device: str = "cuda:2") -> None:
        for name in ("config.yaml", "global_cmvn", "pytorch_model.pt", "vocab.txt"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"ChunkFormer checkpoint missing {name}: {model_dir}")
        try:
            from chunkformer import ChunkFormerModel
        except ImportError as exc:
            raise RuntimeError("Install chunkformer==1.2.2 and its inference imports") from exc
        self.model = ChunkFormerModel.from_pretrained(str(model_dir)).to(device).eval()

    def transcribe(self, path: Path) -> str:
        # The upstream API returns plain text when timestamps are disabled.
        result = self.model.endless_decode(
            audio_path=str(path), chunk_size=64, left_context_size=128,
            right_context_size=128, total_batch_duration=1800,
            return_timestamps=False,
        )
        if not isinstance(result, str):
            raise TypeError(f"unexpected ChunkFormer decode result: {type(result).__name__}")
        return result.strip()


class GipformerASR:
    def __init__(self, model_dir: Path, device: str = "cuda:2") -> None:
        for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Gipformer checkpoint missing {name}: {model_dir}")
        if device.startswith("cuda:"):
            physical_index = device.split(":", 1)[1]
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible and visible != physical_index:
                raise RuntimeError(f"Gipformer CUDA provider uses visible GPU 0; expected CUDA_VISIBLE_DEVICES={physical_index}, got {visible}")
            # sherpa-onnx's CUDA provider always uses visible GPU 0.
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            provider = "cuda"
        elif device == "cpu":
            provider = "cpu"
        else:
            raise ValueError(f"unsupported Gipformer device: {device}")
        try:
            import sherpa_onnx
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("Install sherpa-onnx CUDA 12.8 and soundfile for Gipformer") from exc
        self.sf = sf
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(model_dir / "encoder.onnx"), decoder=str(model_dir / "decoder.onnx"),
            joiner=str(model_dir / "joiner.onnx"), tokens=str(model_dir / "tokens.txt"),
            num_threads=int(os.environ.get("LIFEMATE_ASR_THREADS", "4")), sample_rate=16000, feature_dim=80,
            decoding_method="modified_beam_search", provider=provider,
        )

    def transcribe(self, path: Path) -> str:
        samples, rate = self.sf.read(str(path), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            raise ValueError(f"Gipformer expects 16 kHz audio, got {rate} Hz: {path}")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(rate, samples)
        self.recognizer.decode_streams([stream])
        return stream.result.text.strip()


def load_asr(name: str, model_dir: Path, device: str) -> ASRBackend:
    if name == "qwen":
        return QwenASR(model_dir, device=device)
    if name == "chunkformer":
        return ChunkFormerASR(model_dir, device=device)
    if name == "gipformer":
        return GipformerASR(model_dir, device=device)
    raise ValueError(f"unknown ASR model: {name}")
