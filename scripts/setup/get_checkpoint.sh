#!/usr/bin/env bash
# Fine-tuned checkpoints of the release (configs/release.sh) from the team's Hugging Face model repo, into
# outputs/release/$RELEASE_NAME/:
#   llm-lora-p1/ llm-lora-e1/ llm-lora-synth/   Qwen3.5-4B LoRA adapters (scripts/infer/serve.sh serves them; the image copies them)
#   asr-gipformer/                              Gipformer ASR, ONNX for sherpa-onnx
# The folder is checked against its SHA256SUMS. The base LLM comes from scripts/setup/get_models.sh (Qwen/Qwen3.5-4B, pinned).
# To package checkpoints trained locally instead, run scripts/train/build_release.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
HF="${HF:-uvx --from huggingface_hub[cli]==0.36.0 hf}"
DEST=outputs/release/$RELEASE_NAME
$HF download "${HF_REPO:-$RELEASE_HF_REPO}" --local-dir "$DEST"
(cd "$DEST" && sha256sum --quiet -c SHA256SUMS)
echo "checkpoint done: $DEST"
