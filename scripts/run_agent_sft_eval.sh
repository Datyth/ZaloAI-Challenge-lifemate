#!/usr/bin/env bash
# Evaluate one SFT Agent configuration on frozen TRAIN validation IDs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES
source_kind="${1:?Use gipformer, oracle, or base-oracle}"
run_id="${2:?Provide a unique run ID}"
adapter_dir="${3:-outputs/sft/qwen3_4b_e1_qlora_01/best_adapter}"
run_dir="outputs/runs/$run_id"
if [ "${CONDA_DEFAULT_ENV:-}" != "datpt_lifemate" ]; then
  echo "Activate datpt_lifemate first" >&2
  exit 2
fi
if [ -e "$run_dir" ]; then
  echo "Run directory already exists: $run_dir" >&2
  exit 2
fi
python - <<'PY'
import json
from urllib.request import urlopen
with urlopen('http://127.0.0.1:8000/v1/models', timeout=3) as response:
    served = {row['id'] for row in json.load(response)['data']}
if 'lifemate_sft' not in served:
    raise SystemExit('vLLM is not serving the lifemate_sft LoRA adapter')
PY
common=(--split train --sample-ids outputs/splits/validation_ids.json --device cuda:2
        --orchestration E1 --parser hermes --allow-failures --run-id "$run_id"
        --output "$run_dir/submission.jsonl")
if [ "$source_kind" = "gipformer" ]; then
  python -m lifemate.cli evaluate "${common[@]}" --adapter-dir "$adapter_dir" --llm-served-model lifemate_sft \
    --transcript-source cached-asr --asr-backend gipformer \
    --asr-dir models/gipformer1.5-68m-rnnt \
    --transcripts-jsonl outputs/runs/e1_asr_full_01_A2_cache/asr_outputs.jsonl
elif [ "$source_kind" = "oracle" ]; then
  python -m lifemate.cli evaluate "${common[@]}" --adapter-dir "$adapter_dir" --llm-served-model lifemate_sft --transcript-source train-reference
elif [ "$source_kind" = "base-oracle" ]; then
  python -m lifemate.cli evaluate "${common[@]}" --transcript-source train-reference
else
  echo "Unknown source: $source_kind" >&2
  exit 2
fi
