#!/usr/bin/env bash
# Serve the original Qwen3-4B checkpoint plus one PEFT adapter on GPU 2.
set -euo pipefail
cd "$(dirname "$0")/.."
adapter_dir="${1:-outputs/sft/qwen3_4b_e1_qlora_01/best_adapter}"
if [ ! -f "$adapter_dir/adapter_config.json" ] || [ ! -f "$adapter_dir/adapter_model.safetensors" ]; then
  echo "Missing PEFT adapter: $adapter_dir" >&2
  exit 2
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "datpt_lifemate" ]; then
  echo "Activate datpt_lifemate first" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec vllm serve "$(realpath models/qwen3-4b-instruct-2507)" \
  --host 127.0.0.1 --port 8000 --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.65 --seed 42 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "lifemate_sft=$(realpath "$adapter_dir")"
