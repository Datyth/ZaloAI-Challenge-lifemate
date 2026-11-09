#!/usr/bin/env bash
# Base checkpoints of the release, pinned to the revisions the team used:
#   models/qwen3.5-4b              base LLM (served with the LoRA; copied into the Docker image)
#   models/gipformer1.5-68m-rnnt   original ASR (starting point and BPE model of the ASR v2 fine-tune)
set -euo pipefail
cd "$(dirname "$0")/../.."
HF="${HF:-uvx --from huggingface_hub[cli]==0.36.0 hf}"
$HF download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/qwen3.5-4b
$HF download g-group-ai-lab/gipformer1.5-68M-rnnt --revision dd9227dcd8705c13f33bdbe59728d546ab94480f --local-dir models/gipformer1.5-68m-rnnt
echo "models done"
