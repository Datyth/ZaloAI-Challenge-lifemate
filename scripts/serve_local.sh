#!/usr/bin/env bash
# Local vLLM server for this machine (one RTX PRO 4000 Blackwell): base Qwen3-4B-Instruct-2507 plus the S1 adapter as "lifemate_sft".
# Same model flags as scripts/serve_llm_lora.sh (hermes parser, 16k context, seed 42); TRITON_ATTN as on the DataMate setup.
set -euo pipefail
cd "$(dirname "$0")/.."
adapter_dir="${1:-outputs/sft/qwen3_4b_e1_qlora_01/best_adapter}"
port="${PORT:-8000}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec .venv/bin/vllm serve "$(realpath models/qwen3-4b-instruct-2507)" \
  --host 127.0.0.1 --port "$port" --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization "${GPU_MEM:-0.80}" --seed 42 --attention-backend TRITON_ATTN \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "lifemate_sft=$(realpath "$adapter_dir")"
