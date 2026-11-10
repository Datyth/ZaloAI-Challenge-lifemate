#!/usr/bin/env bash
# Train the released checkpoints with the exact settings of the release, from the data of scripts/data/build_final_data.sh,
# then collect them with scripts/train/build_release.sh. Times are for one RTX PRO 4000 Blackwell (24 GB).
#
#   asr    Gipformer 1.5 (68M), all parameters, 3 epochs, lr 2e-5, accum 4, number labels spelled from their value (targets v3),
#          speed/noise/SpecAugment augmentation, all 5,036 TRAIN recordings; last checkpoint; ONNX export      ~1 h
#   p1     Qwen3.5-4B LoRA r16 / alpha 32, 2 epochs, lr 1e-4, max length 16,384, bf16 base, message-text weight 0.3   ~5.6 h
#   e1     same LoRA, data_full_e1, QLoRA on the 4-bit base                                                            ~4.3 h
#   synth  same LoRA, data_full_synth, QLoRA on the 4-bit base                                                         ~5.1 h
#
#   bash scripts/train/train_final.sh asr|p1|e1|synth|all
# Envs: .venv-asr (scripts/setup/setup_asr_env.sh), .venv-train (pip install -e '.[sft]'), .venv (pip install -e '.[infer]').
# Seeds: 42 everywhere (torch/random in train_gipformer.py, random_state and data order in train_agent_sft.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
SFT=${SFT:-outputs/sft}
ASR_DATA=${ASR_DATA:-outputs/asr_sft/asr_full}
LLM=${LLM_PY:-.venv-train/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

asr() {
  .venv-asr/bin/python scripts/train/train_gipformer.py --data-dir "$ASR_DATA" --output-dir outputs/asr_sft/gip_final \
    --physical-gpu "${GPU:-0}" --epochs 3 --learning-rate 2e-5 --train-all --accum 4 --targets v3 --augment --final \
    --hyp-cache outputs/asr/gipformer_train/asr_outputs.jsonl --eval-every 2000 --patience 99 --max-dev-audio 300
  .venv-asr/bin/python scripts/train/export_gipformer.py --best outputs/asr_sft/gip_final/best.pt --output-dir models/gipformer-lm-final
}
lora() {   # name data_dir extra_flags...
  local name=$1 data=$2; shift 2
  $LLM scripts/train/train_agent_sft.py --data-dir "$data" --out "$SFT/q35_full_$name" --model models/qwen3.5-4b \
    --epochs 2 --lr 1e-4 --save-steps 150 "$@"
}
case "${1:-}" in
  asr) asr ;;
  p1) lora p1 "$SFT/data_full_p1" --fuzzy-weight 0.3 --bf16-base --grad-ckpt on ;;
  e1) lora e1 "$SFT/data_full_e1" ;;
  synth) lora synth "$SFT/data_full_synth" ;;
  all) asr; "$0" p1; "$0" e1; "$0" synth; ASR_SRC=models/gipformer-lm-final bash scripts/train/build_release.sh ;;
  *) echo "usage: bash scripts/train/train_final.sh asr|p1|e1|synth|all" >&2; exit 2 ;;
esac
