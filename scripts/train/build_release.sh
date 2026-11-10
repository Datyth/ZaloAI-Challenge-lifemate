#!/usr/bin/env bash
# Collect the released checkpoints into outputs/release/$RELEASE_NAME (configs/release.sh): the folder scripts/infer/serve.sh
# serves, docker/Dockerfile copies to /code/models/ and the Hugging Face release repo holds. Adds a model card and SHA256SUMS.
#   bash scripts/train/build_release.sh
# Sources (override to package other runs): P1, E1, SYNTH = LoRA adapter folders, ASR_SRC = exported Gipformer folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
P1=${P1:-outputs/sft/q35_full_p1/adapter}
E1=${E1:-outputs/sft/q35_full_e1/adapter}
SYNTH=${SYNTH:-outputs/sft/q35_full_synth/adapter}
ASR_SRC=${ASR_SRC:-models/gipformer-lm-final-02}
DEST=outputs/release/$RELEASE_NAME
[ ! -e "$DEST" ] || { echo "$DEST exists; remove it first" >&2; exit 1; }
mkdir -p "$DEST"
for pair in "llm-lora-p1:$P1" "llm-lora-e1:$E1" "llm-lora-synth:$SYNTH"; do
  dir=$DEST/${pair%%:*}; src=${pair#*:}
  mkdir -p "$dir"
  cp -L "$src"/{adapter_config.json,adapter_model.safetensors,tokenizer.json,tokenizer_config.json,chat_template.jinja} "$dir/"
done
mkdir -p "$DEST/$ASR_DIRNAME"
cp -L "$ASR_SRC"/{encoder.onnx,decoder.onnx,joiner.onnx,tokens.txt,bpe.model,config.json,lifemate_manifest.json} "$DEST/$ASR_DIRNAME/"
cat > "$DEST/README.md" <<'EOF'
---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
language: [vi]
tags: [zalo-ai-challenge-2026, lifemate, tool-calling, lora, asr]
---
# LifeMate release checkpoints (Zalo AI Challenge 2026, Track 1)

Fine-tuned weights of the team's LifeMate solution: a Vietnamese spoken request becomes tool calls on the organizer's `ToolEnv`.
Code, training data recipes and the Docker image layout: https://github.com/Datyth/ZaloAI-Challenge-lifemate

| Folder | Model | Trained on |
| --- | --- | --- |
| `asr-gipformer/` | g-group-ai-lab/gipformer1.5-68M-rnnt fine-tuned (all 68M parameters), ONNX for sherpa-onnx | all 5,036 TRAIN recordings |
| `llm-lora-p1/` | LoRA r16 for Qwen/Qwen3.5-4B @ 851bf6e8, E1P prompt, recovery rows, message-text loss weight 0.3 | all 1,545 TRAIN tasks |
| `llm-lora-e1/` | LoRA r16 (QLoRA), E1 prompt | all 1,545 TRAIN tasks |
| `llm-lora-synth/` | LoRA r16 (QLoRA), E1 prompt, plus synthetic apps written for training | TRAIN tasks + synthetic tasks |

The released pipeline serves `llm-lora-p1` alone (public 61.26); the three adapters can also vote on each next call
(`ENSEMBLE` in `configs/release.sh`). Only the organizer's TRAIN split was used; the public and private test sets were never
labelled or trained on. `SHA256SUMS` lists every file.

Licences: the LoRA adapters are released under Apache-2.0, like Qwen3.5. `asr-gipformer/` is fine-tuned from Gipformer,
which is MIT-licensed; its MIT notice applies to that folder. The code is MIT-licensed in the GitHub repository.
EOF
(cd "$DEST" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
du -sh "$DEST"/*
