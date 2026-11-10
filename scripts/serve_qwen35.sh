#!/usr/bin/env bash
# Serve Qwen3.5-4B (text only) with the qwen3_coder tool parser; optional LoRA adapters as name=path arguments.
#   bash scripts/serve_qwen35.sh [name=adapter_dir ...]
set -euo pipefail
cd "$(dirname "$0")/.."
lora=()
if [ $# -gt 0 ]; then lora=(--enable-lora --max-lora-rank 32 --max-loras "${MAX_LORAS:-$#}" --lora-modules "$@"); fi   # all adapters resident: an ensemble queries them concurrently
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec .venv/bin/vllm serve "$(realpath models/qwen3.5-4b)" --served-model-name qwen35_base "$(realpath models/qwen3.5-4b)" \
  --host 127.0.0.1 --port 8000 --max-model-len 16384 --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --gpu-memory-utilization ${GPU_UTIL:-0.66} --seed 42 ${PREFIX_CACHE:+--no-enable-prefix-caching} ${REASONING_CONFIG:+--reasoning-config "$REASONING_CONFIG"} "${lora[@]}"
