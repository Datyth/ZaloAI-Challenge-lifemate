#!/usr/bin/env bash
# Base checkpoints for the LifeMate pipeline, pinned to the revisions the team used.
set -euo pipefail
cd "$(dirname "$0")/.."
HF="uvx --from huggingface_hub[cli]==0.36.0 hf"
$HF download g-group-ai-lab/gipformer1.5-68M-rnnt --revision dd9227dcd8705c13f33bdbe59728d546ab94480f --local-dir models/gipformer1.5-68m-rnnt
$HF download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir models/qwen3-4b-instruct-2507
$HF download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/qwen3.5-4b
$HF download Qwen/Qwen3-4B-Thinking-2507 --local-dir models/qwen3-4b-thinking-2507
echo "models done"
