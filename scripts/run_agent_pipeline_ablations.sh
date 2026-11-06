#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Use the same 200 TRAIN IDs and reference transcripts for every mode.
device="${1:-cuda:0}"
port="${2:-8001}"
prefix="${3:-pipeline_$(date +%Y%m%dT%H%M%S)}"
ids="outputs/splits/validation_ids.json"
python_bin="${PYTHON_BIN:-/home/datpt/.conda/envs/datpt_lifemate/bin/python}"
if [[ ! -f "$ids" ]]; then
  echo "Missing frozen validation IDs: $ids" >&2
  exit 2
fi
if ! curl --fail --silent --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null; then
  echo "No Qwen server on port ${port}; start it on a GPU not used by training" >&2
  exit 2
fi
export PYTHONPATH=src
for stage in O0 O1 O2 O3 O4 O5; do
  mode="$stage"
  [[ "$stage" == O0 ]] && mode=E1
  run_id="${prefix}_${stage}"
  run_dir="outputs/runs/${run_id}"
  if [[ -e "$run_dir" ]]; then
    echo "Run already exists: $run_dir" >&2
    exit 2
  fi
  "$python_bin" -m lifemate.cli evaluate --split train \
    --sample-ids "$ids" --transcript-source train-reference \
    --device "$device" --orchestration "$mode" \
    --llm-url "http://127.0.0.1:${port}/v1/chat/completions" \
    --run-id "$run_id" --output "$run_dir/submission.jsonl"
done
"$python_bin" scripts/summarize_agent_pipeline.py "outputs/runs/${prefix}_O0" \
  "outputs/runs/${prefix}_O1" "outputs/runs/${prefix}_O2" \
  "outputs/runs/${prefix}_O3" "outputs/runs/${prefix}_O4" \
  "outputs/runs/${prefix}_O5" --output "outputs/reports/${prefix}_comparison.json"
