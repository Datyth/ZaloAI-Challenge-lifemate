#!/usr/bin/env bash
# Export a Gipformer v2 checkpoint to ONNX (icefall), add bpe.vocab for hotwords, decode the 200 validation samples' audio
# with and without hotwords, and compare with the original model.  bash downloads/export_gip_v2.sh <run_dir> <model_name>
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1; name=$2
.venv-asr/bin/python scripts/export_gipformer_sft.py --best "$run/best.pt" --output-dir "models/$name" --device cpu 2>&1 | grep -vE "custom_fwd|custom_bwd|warn" | tail -5
cp models/gipformer1.5-68m-rnnt/bpe.vocab "models/$name/"
.venv/bin/python analysis/asr_cache.py --split train --ids outputs/splits/validation_ids.json --model-dir "models/$name" --output-dir "outputs/asr/${name}_val" --workers 6 > /dev/null
.venv/bin/python analysis/asr_cache.py --split train --ids outputs/splits/validation_ids.json --model-dir "models/$name" --output-dir "outputs/asr/${name}_hw_val" --workers 6 \
  --hotwords outputs/asr/hotwords/intent_v2.txt --hotwords-score 4.0 > /dev/null
.venv/bin/python analysis/asr_compare.py outputs/asr/hw_none_val outputs/asr/hw_v2_s4.0_val "outputs/asr/${name}_val" "outputs/asr/${name}_hw_val"
