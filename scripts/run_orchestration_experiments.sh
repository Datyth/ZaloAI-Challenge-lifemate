#!/usr/bin/env bash
# Run later, when the existing ASR device and local vLLM server are available.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
device="${1:-cuda:2}"
prefix="${2:-orchestration_$(date +%Y%m%dT%H%M%S)}"
ids="${3:-outputs/splits/validation_ids.json}"
mkdir -p outputs/reports
status_file="outputs/reports/${prefix}_run_status.tsv"
printf 'mode\tstatus\n' > "$status_file"
for mode in E0 E1 E2 E3; do
  if python -m lifemate.cli evaluate --split train --sample-ids "$ids" \
      --device "$device" --orchestration "$mode" --run-id "${prefix}_${mode}" \
      --output "outputs/runs/${prefix}_${mode}/submission.jsonl"; then
    printf '%s\tok\n' "$mode" >> "$status_file"
  else
    printf '%s\tfailed; inspect predictions and partial JSONL\n' "$mode" >> "$status_file"
  fi
done
python scripts/summarize_orchestration.py \
  "outputs/runs/${prefix}_E0" "outputs/runs/${prefix}_E1" \
  "outputs/runs/${prefix}_E2" "outputs/runs/${prefix}_E3" \
  --output "outputs/reports/${prefix}_comparison.json"
