#!/usr/bin/env bash
# Wait for the baseline submission and an idle physical GPU 2, then train once.
set -euo pipefail
cd "$(dirname "$0")/.."

submission="result/submission_s1_gipformer_base_20261106.jsonl"
baseline_metrics="outputs/runs/public_s1_gipformer_base_20261106/metrics.json"
train_dir="outputs/asr_sft/gipformer_run_01"
train_log="outputs/asr_sft/gipformer_train_01.log"
mkdir -p outputs/asr_sft

exec 9>outputs/asr_sft/gipformer_queue_01.lock
if ! flock -n 9; then
  echo "A Gipformer SFT queue or training process is already running" >&2
  exit 2
fi
if [[ -e "$train_dir" || -e "$train_log" ]]; then
  echo "Training output already exists; refusing to overwrite: $train_dir or $train_log" >&2
  exit 2
fi

gpu_idle() {
  local used pids
  used="$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  used="$(awk '{print $1}' <<< "$used")"
  [[ "$used" =~ ^[0-9]+$ ]] || return 1
  (( used < 1024 )) || return 1
  pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" || return 1
  ! [[ "$pids" =~ [0-9] ]]
}

echo "Waiting for complete baseline submission and idle GPU 2" >&2
while true; do
  if [[ -s "$submission" && -s "$baseline_metrics" ]] && gpu_idle; then
    echo "GPU 2 idle; checking again in 20 seconds" >&2
    sleep 20
    if [[ -s "$submission" && -s "$baseline_metrics" ]] && gpu_idle; then
      break
    fi
  fi
  sleep 30
done

if [[ -e "$train_dir" || -e "$train_log" ]]; then
  echo "Training output appeared while queued; refusing to overwrite" >&2
  exit 2
fi
echo "Starting Gipformer SFT on physical GPU 2 at $(date -Is)" >&2
export CUDA_VISIBLE_DEVICES=2
exec models/.research/gipformer-upstream/.venv/bin/python \
  scripts/train_gipformer_sft.py \
  --data-dir outputs/asr_sft/gipformer_data_20261106 \
  --output-dir "$train_dir" --physical-gpu 2 \
  > "$train_log" 2>&1
