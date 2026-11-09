#!/usr/bin/env bash
# Export a fine-tuned Gipformer checkpoint to ONNX (icefall, strict load), add bpe.vocab, decode the audio of the frozen
# validation samples on CPU and compare WER / key-word recall with earlier caches (e.g. the original model's).
#   bash scripts/train/export_gipformer.sh <run_dir> <model_name> [baseline_cache_dir ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
run=$1; name=$2; shift 2
.venv-asr/bin/python scripts/train/export_gipformer.py --best "$run/best.pt" --output-dir "models/$name" --device cpu 2>&1 | grep -vE "custom_fwd|custom_bwd|warn" | tail -5
cp models/gipformer1.5-68m-rnnt/bpe.vocab "models/$name/"
.venv/bin/python scripts/infer/asr_cache.py --split train --ids configs/splits/validation_ids.json --model-dir "models/$name" --output-dir "outputs/asr/${name}_val" --workers 6 > /dev/null
.venv/bin/python analysis/asr_compare.py "$@" "outputs/asr/${name}_val"
