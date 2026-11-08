#!/usr/bin/env bash
# Evaluate one configuration on a TRAIN id list in N parallel shards against the local vLLM server.
#   bash analysis/run_shards.sh <prefix> <ids.json> <shards> <source: asr|oracle> <model: sft|base> [orchestration] [port]
# Shard runs land in outputs/runs/<prefix>_s<k>/; analysis/score.py reads them together.
set -euo pipefail
cd "$(dirname "$0")/.."
prefix=$1; ids=$2; shards=$3; source=$4; model=$5; orch=${6:-E1}; port=${7:-8000}
split=${SPLIT:-train}
cache=${CACHE:-outputs/asr/gipformer_${split/public_test/public}/asr_outputs.jsonl}
mkdir -p outputs/splits/shards outputs/logs
.venv/bin/python - "$ids" "$shards" "$prefix" <<'PY'
import json, sys
ids = json.load(open(sys.argv[1])); n = int(sys.argv[2])
for k in range(n):
    json.dump(ids[k::n], open(f"outputs/splits/shards/{sys.argv[3]}_s{k}.json", "w"))
PY
args=(--split "$split" --device cuda:0 --orchestration "$orch" --allow-failures --deadline "${DEADLINE:-30}" --llm-url "http://127.0.0.1:${port}/v1/chat/completions")
if [ "$source" = asr ]; then args+=(--transcript-source cached-asr --transcripts-jsonl "$cache" --asr-backend gipformer --asr-dir "${ASR_DIR:-models/gipformer1.5-68m-rnnt}")
else args+=(--transcript-source train-reference); fi
# extra flags, e.g. EXTRA="--asr-lowercase --max-turns 12"
if [ -n "${EXTRA:-}" ]; then read -r -a extra <<< "$EXTRA"; args+=("${extra[@]}"); fi
if [ "$model" = sft ]; then args+=(--adapter-dir "${ADAPTER:-outputs/sft/qwen3_4b_e1_qlora_01/best_adapter}" --llm-served-model "${SERVED:-lifemate_sft}"); fi
pids=()
for ((k = 0; k < shards; k++)); do
  PYTHONPATH=src .venv/bin/python -m lifemate.cli evaluate "${args[@]}" --sample-ids "outputs/splits/shards/${prefix}_s${k}.json" \
    --run-id "${prefix}_s${k}" > "outputs/logs/${prefix}_s${k}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$prefix done (status $status)"
