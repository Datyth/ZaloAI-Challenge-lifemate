#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model_dir="${1:-models/qwen3-4b-instruct-2507}"
parser="${2:-hermes}"
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "Missing LLM checkpoint: $model_dir" >&2
  exit 1
fi
# The process sees physical GPU 2 as logical cuda:0.
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec vllm serve "$(realpath "$model_dir")" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser "$parser" \
  --gpu-memory-utilization 0.65 --seed 42
