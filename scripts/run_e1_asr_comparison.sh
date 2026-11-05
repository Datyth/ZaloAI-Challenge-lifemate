#!/usr/bin/env bash
# A1/A2 E1 Agent runs; A0 (Qwen) and A3 (oracle) are preserved TRAIN runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES
device="${1:-cuda:2}"
prefix="${2:-e1_asr_$(date +%Y%m%dT%H%M%S)}"
ids="outputs/splits/validation_ids.json"
a0="outputs/runs/train200_20261105_115719_E1"
a3="outputs/runs/agentopt_full_01_E4"
a1="outputs/runs/${prefix}_A1"
a2="outputs/runs/${prefix}_A2"
a1_cache="outputs/runs/${prefix}_A1_cache"
a2_cache="outputs/runs/${prefix}_A2_cache"
report="outputs/reports/${prefix}"

if [ "${CONDA_DEFAULT_ENV:-}" != "datpt_lifemate" ]; then
  echo "Activate datpt_lifemate before running this script" >&2
  exit 2
fi
if ! curl --fail --silent --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "Qwen3-4B vLLM server unavailable; start scripts/serve_llm.sh first" >&2
  exit 2
fi
python - "$a0" "$a3" <<'PY'
import json
from pathlib import Path
import sys
from urllib.request import urlopen
sys.path.insert(0, 'scripts')
from compare_e1_asr import FROZEN_IDS, MATCH_KEYS, load_run
ids = json.loads(FROZEN_IDS.read_text())
a0, a3 = [load_run(name, Path(path), ids) for name, path in zip(('A0', 'A3'), sys.argv[1:])]
if any(a0['config'].get(key) != a3['config'].get(key) for key in MATCH_KEYS):
    raise SystemExit('A0 and A3 agent settings differ')
with urlopen('http://127.0.0.1:8000/v1/models', timeout=3) as response:
    served = {row['id'] for row in json.load(response)['data']}
expected = str(Path('models/qwen3-4b-instruct-2507').resolve())
if expected not in served:
    raise SystemExit(f'vLLM does not serve required model: {expected}')
print('A0/A3 and Qwen3-4B server verified')
PY

prepare_cache() {
  local stage="$1" backend="$2" checkpoint="$3" source_run="$4" cache_dir="$5"
  if [ -f "$cache_dir/metrics.json" ]; then
    if python - "$cache_dir/metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
sys.exit(0 if m.get('complete') and m.get('samples') == 200 and m.get('history_failures') == 0 else 1)
PY
    then
      echo "$stage cache: complete, reusing $cache_dir"
      return
    fi
  fi
  if [ -e "$cache_dir" ]; then
    echo "$stage cache: incomplete directory $cache_dir; inspect it and use a new prefix" >&2
    exit 2
  fi
  echo "Preparing $stage history ASR cache at $(date -Is)"
  python scripts/prepare_e1_asr_cache.py --model "$backend" --model-dir "$checkpoint" \
    --source-run "$source_run" --output-dir "$cache_dir" --sample-ids "$ids" --device "$device"
}

run_stage() {
  local stage="$1" backend="$2" checkpoint="$3" transcripts="$4" run_dir="$5"
  if [ -f "$run_dir/metrics.json" ] && [ -f "$run_dir/submission.jsonl" ]; then
    echo "$stage: complete run found at $run_dir; final comparison validates it"
    return
  fi
  if [ -e "$run_dir" ]; then
    echo "$stage: incomplete directory $run_dir; inspect it and use a new prefix" >&2
    exit 2
  fi
  echo "Starting $stage at $(date -Is)"
  python -m lifemate.cli evaluate --split train --sample-ids "$ids" \
    --device "$device" --orchestration E1 --parser hermes \
    --transcript-source cached-asr --transcripts-jsonl "$transcripts" \
    --asr-backend "$backend" --asr-dir "$checkpoint" --allow-failures \
    --run-id "${prefix}_${stage}" --output "$run_dir/submission.jsonl"
}

prepare_cache A1 chunkformer models/chunkformer-rnnt-large-vie \
  outputs/runs/asr_compare_final_20261105_chunkformer "$a1_cache"
run_stage A1 chunkformer models/chunkformer-rnnt-large-vie \
  "$a1_cache/asr_outputs.jsonl" "$a1"
prepare_cache A2 gipformer models/gipformer1.5-68m-rnnt \
  outputs/runs/asr_compare_final_20261105_gipformer "$a2_cache"
run_stage A2 gipformer models/gipformer1.5-68m-rnnt \
  "$a2_cache/asr_outputs.jsonl" "$a2"
python scripts/compare_e1_asr.py --a0 "$a0" --a1 "$a1" --a2 "$a2" --a3 "$a3" --output-dir "$report"
echo "Complete report: $report/report.md"
