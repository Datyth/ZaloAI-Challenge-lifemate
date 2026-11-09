#!/usr/bin/env bash
# Local vLLM server (127.0.0.1:8000) for Qwen3.5-4B with the release flags of configs/release.sh.
#   bash scripts/infer/serve.sh                      # base model + release LoRA, served as "lifemate"
#   bash scripts/infer/serve.sh name=adapter_dir ... # base model + these adapters instead
#   bash scripts/infer/serve.sh none                 # base model only (served as "qwen35_base")
# NO_PREFIX_CACHE=1 turns prefix caching off; GPU_MEM sets the memory share (default 0.66).
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
base=$(realpath models/qwen3.5-4b)
[ -f "$base/config.json" ] || { echo "missing $base; run scripts/setup/get_models.sh" >&2; exit 1; }
if [ $# -eq 0 ]; then
  adapter=outputs/release/$RELEASE_NAME/llm-lora
  [ -f "$adapter/adapter_config.json" ] || { echo "missing $adapter; run scripts/setup/get_checkpoint.sh" >&2; exit 1; }
  set -- "$LORA_NAME=$(realpath "$adapter")"
fi
lora=(); [ "$1" = none ] || lora=("${VLLM_LORA_FLAGS[@]}" --lora-modules "$@")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec .venv/bin/vllm serve "$base" --served-model-name qwen35_base "$base" --host 127.0.0.1 --port "${PORT:-8000}" \
  "${VLLM_MODEL_FLAGS[@]}" --gpu-memory-utilization "${GPU_MEM:-0.66}" ${NO_PREFIX_CACHE:+--no-enable-prefix-caching} "${lora[@]}"
