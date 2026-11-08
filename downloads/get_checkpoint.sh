#!/usr/bin/env bash
# Fine-tuned checkpoints of the public-58.02 pipeline (q35e1), from the team's private Hugging Face model repo.
# Needs `hf auth login` with an account that has access. Afterwards:
#   outputs/sft/q35_e1_01/adapter -> Qwen3.5-4B LoRA (serve with scripts/serve_qwen35.sh q35_e1=<that dir>)
#   models/gipformer-lm-v2-01     -> Gipformer ASR v2 (ONNX, sherpa-onnx)
# The base LLM comes from downloads/get_models.sh (Qwen/Qwen3.5-4B, pinned revision).
set -euo pipefail
cd "$(dirname "$0")/.."
HF_REPO="${HF_REPO:-hugebenevolence/lifemate-q35e1}"
HF="${HF:-uvx --from huggingface_hub[cli]==0.36.0 hf}"
DEST=outputs/release/lifemate-q35e1
$HF download "$HF_REPO" --local-dir "$DEST"
mkdir -p outputs/sft/q35_e1_01 models
[ -e outputs/sft/q35_e1_01/adapter ] || ln -s "$(realpath "$DEST/llm-lora")" outputs/sft/q35_e1_01/adapter
[ -e models/gipformer-lm-v2-01 ] || ln -s "$(realpath "$DEST/asr-gipformer-v2")" models/gipformer-lm-v2-01
echo "checkpoint done"
