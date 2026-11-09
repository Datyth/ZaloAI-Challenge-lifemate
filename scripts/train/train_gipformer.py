"""Gipformer v2 adaptation on TRAIN audio (strictly loaded icefall model, separate .venv-asr with k2/kaldifeat).

Differences from the removed v1 trainer (whose run collapsed on <unk> targets before the target fix):
- targets from scripts/train/asr_targets.py: digit spans spelled as the original Gipformer heard them (keeps amount/time rows),
  tone marks moved to the model's own style (HOÀ -> HÒA), unknown-BPE rows skipped;
- the whole model can be trained (--train-all), since the main errors are lexical (dời -> giờ, Zalo -> gia lô), and
  gradient accumulation over several recordings.
ASR v2 (models/gipformer-lm-v2-01; whole model per docs/experiments.md):
  .venv-asr/bin/python scripts/train/train_gipformer.py --data-dir outputs/asr_sft/gipformer_data_v2 --output-dir outputs/asr_sft/gip_v2_01 --physical-gpu 0 --train-all
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import unicodedata

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts" / "train"))
from gipformer_pt import ORIGINAL, sha256, strict_model  # noqa: E402
from asr_targets import target as spelled_target  # noqa: E402
import re as _re  # noqa: E402

_TONE = {"À": ("A", "Ò"), "Á": ("A", "Ó"), "Ả": ("A", "Ỏ"), "Ã": ("A", "Õ"), "Ạ": ("A", "Ọ")}


def model_style(text: str) -> str:
    """Old-style tone placement of the references (HOÀ, KHOÁ, HUỶ, THUỶ) in the model's new style (HÒA, KHÓA, HỦY, THỦY)."""
    text = _re.sub(r"O([ÀÁẢÃẠ])(?=\s|$)", lambda m: {"À": "ÒA", "Á": "ÓA", "Ả": "ỎA", "Ã": "ÕA", "Ạ": "ỌA"}[m.group(1)], text)
    text = _re.sub(r"O([ÈÉẺẼẸ])(?=\s|$)", lambda m: {"È": "ÒE", "É": "ÓE", "Ẻ": "ỎE", "Ẽ": "ÕE", "Ẹ": "ỌE"}[m.group(1)], text)
    text = _re.sub(r"(?<!Q)U([ỲÝỶỸỴ])(?=\s|$)", lambda m: {"Ỳ": "ÙY", "Ý": "ÚY", "Ỷ": "ỦY", "Ỹ": "ŨY", "Ỵ": "ỤY"}[m.group(1)], text)
    return text


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalized_target(text: str) -> str:
    """Match the uppercase, punctuation-free vocabulary of the pretrained BPE.

    Digits require a Vietnamese spoken-form normalizer. Reject those rows rather
    than teaching the model an incorrect or unknown target for the audio.
    """
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char
                   for char in text.upper())
    if any(char.isdigit() for char in text):
        return ""
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "outputs/asr_sft/gipformer_data_20261106")
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL)
    parser.add_argument("--icefall-root", type=Path, default=PROJECT / "models/.research/icefall")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/asr_sft/gipformer_run_01")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-audio", type=int, default=100000)
    parser.add_argument("--max-dev-audio", type=int, default=300)
    parser.add_argument("--train-all", action="store_true", help="train decoder/joiner too, not only the encoder")
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--hyp-cache", type=Path, default=PROJECT / "outputs/asr/gipformer_train/asr_outputs.jsonl")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true", help="one forward/backward pass; do not update weights or save checkpoints")
    args = parser.parse_args()
    if any(x < 1 for x in (args.epochs, args.max_train_audio, args.max_dev_audio, args.eval_every, args.patience)):
        parser.error("epochs, data limits, eval interval and patience must be positive")
    if not 0 < args.learning_rate <= 1e-4:
        parser.error("use a learning rate between 0 and 1e-4")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}; expected physical GPU {args.physical_gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)

    import torch
    import torchaudio
    import sentencepiece as spm
    import k2
    import kaldifeat
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA GPU")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = PROJECT / "configs/splits/validation_ids.json"
    if manifest.get("validation_samples_excluded") != 200 or manifest.get("validation_ids_sha256") != sha256(frozen):
        raise ValueError("SFT data was not prepared against the frozen 200 validation IDs")
    train = read_rows(args.data_dir / "train.jsonl")
    dev = read_rows(args.data_dir / "dev.jsonl")
    if not train or not dev or set(x["audio"] for x in train) & set(x["audio"] for x in dev):
        raise ValueError("empty or overlapping internal train/dev audio")
    model, params = strict_model(args.model_dir, args.icefall_root)
    source_sha = sha256(args.model_dir / "model.pt")
    torch.manual_seed(42)
    random.seed(42)
    model = model.to("cuda:0")
    # A small, low-LR encoder adaptation keeps the original predictor/joiner.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = args.train_all or name.startswith("encoder.") or name.startswith("encoder_embed.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise AssertionError("no encoder parameters selected for adaptation")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    sp = spm.SentencePieceProcessor(model_file=str(args.model_dir / "bpe.model"))
    hyp = {}
    for line in args.hyp_cache.open(encoding="utf-8"):
        r = json.loads(line)
        hyp[r["audio"]] = r["raw_transcript"]
    def valid_targets(rows: list[dict], limit: int) -> tuple[list[dict], int]:
        kept = []
        for row in rows:
            spelled = spelled_target(row["text"], hyp.get(row["audio"], ""))
            target = model_style(spelled) if spelled else ""
            ids = sp.encode(target, out_type=int) if target else []
            if not ids or sp.unk_id() in ids or any(token <= 0 or token >= params.vocab_size for token in ids):
                continue
            kept.append({**row, "target_text": target})
        return kept[:limit], len(rows) - len(kept)
    train, skipped_train = valid_targets(train, args.max_train_audio)
    dev, skipped_dev = valid_targets(dev, args.max_dev_audio)
    if not train or not dev:
        raise ValueError("no TRAIN audio has a fully supported Gipformer BPE target")
    print(f"BPE targets: train={len(train)} dev={len(dev)} skipped_train={skipped_train} skipped_dev={skipped_dev}", flush=True)
    opts = kaldifeat.FbankOptions()
    opts.device = torch.device("cuda:0")
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(opts)
    root = PROJECT / "data/train+public_test/train"
    def sample_loss(row: dict) -> torch.Tensor:
        path = (root / row["audio"]).resolve()
        if not path.is_relative_to((root / "audio").resolve()) or not path.is_file():
            raise ValueError(f"invalid TRAIN audio: {row['audio']}")
        wave, rate = torchaudio.load(str(path))
        if rate != 16000:
            wave = torchaudio.functional.resample(wave, rate, 16000)
        samples = wave.mean(dim=0).contiguous().to("cuda:0")
        feature = fbank([samples])[0]
        if feature.shape[0] < 16:
            raise ValueError(f"audio too short for Zipformer: {row['audio']}")
        tokens = sp.encode(row["target_text"], out_type=int)
        if not tokens or any(token <= 0 or token >= params.vocab_size or token == sp.unk_id() for token in tokens):
            raise ValueError(f"invalid BPE sequence: {row['audio']}")
        y = k2.RaggedTensor([tokens])
        simple, pruned, ctc, _, _ = model(feature.unsqueeze(0),
            torch.tensor([feature.shape[0]], device="cuda:0"), y,
            prune_range=params.prune_range, am_scale=params.am_scale, lm_scale=params.lm_scale)
        loss = 0.5 * simple + pruned + 0.2 * ctc
        return loss / max(1, len(tokens))
    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for row in dev:
                losses.append(float(sample_loss(row)))
        model.train()
        return sum(losses) / len(losses)

    if args.smoke_only:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sample_loss(train[0])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite smoke loss")
        loss.backward()
        gradients = [p.grad for p in trainable if p.grad is not None]
        if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError("missing or nonfinite smoke gradients")
        print(json.dumps({"smoke_only": True, "train_loss": float(loss.detach()),
                          "gradient_tensors": len(gradients), "source_sha256": source_sha}))
        return

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"source_checkpoint": str((args.model_dir / "model.pt").resolve()), "source_sha256": source_sha,
              "data_dir": str(args.data_dir.resolve()), "data_manifest": manifest, "icefall_root": str(args.icefall_root.resolve()),
              "physical_gpu": args.physical_gpu, "learning_rate": args.learning_rate, "epochs": args.epochs,
              "max_train_audio": len(train), "max_dev_audio": len(dev), "eval_every": args.eval_every,
              "patience": args.patience, "trainable_parameters": sum(p.numel() for p in trainable),
              "target_normalization": "upper case, no punctuation, digit spans spelled from the original Gipformer transcript, model tone style; unresolved and unknown-BPE rows skipped",
              "train_all": args.train_all, "accum": args.accum,
              "skipped_train_audio": skipped_train, "skipped_dev_audio": skipped_dev}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best = math.inf
    stale = 0
    step = 0
    with (args.output_dir / "training.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            random.Random(42 + epoch).shuffle(train)
            model.train()
            for index, row in enumerate(train):
                started = time.monotonic()
                loss = sample_loss(row)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite RNNT loss at {row['audio']}")
                (loss / args.accum).backward()
                if (index + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {"epoch": epoch, "step": step, "train_loss": float(loss.detach()),
                          "audio": row["audio"], "seconds": time.monotonic() - started}
                log.write(json.dumps(record) + "\n")
                if step % 25 == 0:
                    log.flush()
                    print(f"Gipformer SFT epoch={epoch} step={step} loss={record['train_loss']:.4f}", flush=True)
                if step % args.eval_every == 0 or (epoch == args.epochs and row is train[-1]):
                    dev_loss = evaluate()
                    checkpoint = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "step": step, "epoch": epoch,
                                  "dev_loss": dev_loss, "source_sha256": source_sha}
                    path = args.output_dir / f"checkpoint-{step:06d}.pt"
                    torch.save(checkpoint, path)
                    improved = dev_loss < best - 1e-4
                    if improved:
                        best = dev_loss
                        stale = 0
                        torch.save(checkpoint, args.output_dir / "best.pt")
                    else:
                        stale += 1
                    log.write(json.dumps({"step": step, "dev_loss": dev_loss,
                                          "checkpoint": str(path), "best": improved}) + "\n")
                    log.flush()
                    print(f"dev step={step} loss={dev_loss:.4f} best={best:.4f} stale={stale}", flush=True)
                    if stale >= args.patience:
                        print("early stopping", flush=True)
                        return


if __name__ == "__main__":
    main()
