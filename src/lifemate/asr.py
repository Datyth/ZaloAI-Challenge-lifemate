"""Gipformer RNN-T ASR (sherpa-onnx, modified beam search) from a local ONNX checkpoint."""
from __future__ import annotations

import os
from pathlib import Path


class GipformerASR:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
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
