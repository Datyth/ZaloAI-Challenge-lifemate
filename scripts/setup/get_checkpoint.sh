#!/usr/bin/env bash
# Fine-tuned checkpoints of the release (configs/release.sh) from the team's private Hugging Face model repo.
# Needs `hf auth login` with an account that has access. Afterwards:
#   outputs/release/<name>/llm-lora          Qwen3.5-4B LoRA (scripts/infer/serve.sh serves it; the Docker image copies it)
#   outputs/release/<name>/asr-gipformer-v2  Gipformer ASR v2 (ONNX, sherpa-onnx)
# plus the paths the runs and ASR caches were made with: outputs/sft/q35_e1_01/adapter and models/gipformer-lm-v2-01.
# The base LLM comes from scripts/setup/get_models.sh (Qwen/Qwen3.5-4B, pinned revision).
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
HF="${HF:-uvx --from huggingface_hub[cli]==0.36.0 hf}"
DEST=outputs/release/$RELEASE_NAME
$HF download "${HF_REPO:-$RELEASE_HF_REPO}" --local-dir "$DEST"
mkdir -p outputs/sft/q35_e1_01 models
[ -e outputs/sft/q35_e1_01/adapter ] || ln -s "$(realpath "$DEST/llm-lora")" outputs/sft/q35_e1_01/adapter
[ -e models/gipformer-lm-v2-01 ] || ln -s "$(realpath "$DEST/asr-gipformer-v2")" models/gipformer-lm-v2-01
echo "checkpoint done: $DEST"
