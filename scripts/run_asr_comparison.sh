#!/usr/bin/env bash
# Compare two new local checkpoints with the preserved Qwen ASR run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The benchmark accepts physical GPU indices. Let PyTorch see all devices;
# Gipformer maps the requested index to its ONNX visible device internally.
unset CUDA_VISIBLE_DEVICES
device="${1:-cuda:2}"
prefix="${2:-asr_compare_$(date +%Y%m%dT%H%M%S)}"
ids="outputs/splits/validation_ids.json"
chunk="outputs/runs/${prefix}_chunkformer"
gip="outputs/runs/${prefix}_gipformer"
report="outputs/reports/${prefix}"
if [ "${CONDA_DEFAULT_ENV:-}" != "datpt_lifemate" ]; then
  echo "Activate datpt_lifemate before running this script" >&2
  exit 2
fi
run_model() {
  local name="$1" run_dir="$2"
  if [ -f "$run_dir/metrics.json" ]; then
    echo "$name: existing metrics found; validating complete output in comparison step"
    return
  fi
  if [ -e "$run_dir" ]; then
    echo "$name: incomplete run directory exists: $run_dir. Use a new prefix after inspecting errors." >&2
    exit 2
  fi
  echo "Starting $name on $device at $(date -Is)"
  python -m lifemate.cli benchmark-asr --model "$name" --sample-ids "$ids" \
    --device "$device" --run-id "${prefix}_${name}"
}
run_model chunkformer "$chunk"
run_model gipformer "$gip"
python scripts/compare_asr_models.py --qwen outputs/runs/baseline_asr_200 \
  --chunkformer "$chunk" --gipformer "$gip" --output-dir "$report"
echo "Complete report: $report/report.md"
