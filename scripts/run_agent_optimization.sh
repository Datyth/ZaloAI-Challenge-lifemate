#!/usr/bin/env bash
# Run E4 -> E5 -> E6 -> E6-Oracle on the frozen TRAIN validation set.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
device="${1:-cuda:2}"
prefix="${2:-agentopt_$(date +%Y%m%dT%H%M%S)}"
ids="outputs/splits/validation_ids.json"
e1="outputs/runs/train200_20261105_115719_E1"
e4="outputs/runs/${prefix}_E4"
e6="outputs/runs/${prefix}_E6"
e6_oracle="outputs/runs/${prefix}_E6_oracle"
report_dir="outputs/reports/${prefix}"
mkdir -p "$report_dir"

if [ "${CONDA_DEFAULT_ENV:-}" != "datpt_lifemate" ]; then
  echo "Activate datpt_lifemate before running this script" >&2
  exit 2
fi
if ! curl --fail --silent --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "Local vLLM server unavailable at 127.0.0.1:8000; start scripts/serve_llm.sh on GPU 2" >&2
  exit 2
fi

run_stage() {
  local name="$1" source="$2" mode="$3" run_dir="$4"
  if [ -f "$run_dir/metrics.json" ] && [ -f "$run_dir/submission.jsonl" ]; then
    echo "${name}: complete, reusing $run_dir"
    return
  fi
  if [ -e "$run_dir" ]; then
    echo "${name}: $run_dir already exists without complete metrics; use a new prefix or inspect its partial file" >&2
    exit 2
  fi
  echo "Starting ${name} at $(date -Is)"
  python -m lifemate.cli evaluate --split train --sample-ids "$ids" \
    --device "$device" --orchestration "$mode" --transcript-source "$source" \
    --run-id "${prefix}_${name}" --output "$run_dir/submission.jsonl"
}

run_stage E4 train-reference E1 "$e4"
python scripts/analyze_agent_optimization.py --e1 "$e1" --e4 "$e4" \
  --output-dir "$report_dir" --report docs/agent_optimization_report.md > "$report_dir/e5_summary.json"
run_stage E6 asr E6 "$e6"
run_stage E6_oracle train-reference E6 "$e6_oracle"
python scripts/analyze_agent_optimization.py --e1 "$e1" --e4 "$e4" --e6 "$e6" --e6-oracle "$e6_oracle" \
  --output-dir "$report_dir" --report docs/agent_optimization_report.md > "$report_dir/final_summary.json"
echo "Done: docs/agent_optimization_report.md and $report_dir"
