#!/usr/bin/env bash
# Run the agent on a split in N parallel shards against the local vLLM server (scripts/infer/serve.sh), with the release
# agent flags of configs/release.sh. Exits non-zero if any shard fails.
#   bash scripts/infer/run_shards.sh <prefix> <ids.json|all> <shards> <source: asr|oracle> <model: sft|base> [orchestration] [port]
# Shard runs land in outputs/runs/<prefix>_s<k>/; analysis/score.py and scripts/infer/merge_submission.py read them together.
# Env: SPLIT (train), CACHE (ASR transcripts, see below), ADAPTER/SERVED (LoRA), ASR_DIR, DEADLINE (s per sample), EXTRA (more flags).
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/release.sh
prefix=$1; ids=$2; shards=$3; source=$4; model=$5; orch=${6:-}; port=${7:-8000}
split=${SPLIT:-train}
asr_dir=${ASR_DIR:-outputs/release/$RELEASE_NAME/$ASR_DIRNAME}
# Transcripts written by scripts/infer/asr_cache.py with $asr_dir: <name>_val (validation ids) or <name>_public (public_test).
suffix=val; [ "$split" = public_test ] && suffix=public
cache=${CACHE:-outputs/asr/$(basename "$asr_dir")_$suffix/asr_outputs.jsonl}
mkdir -p outputs/splits/shards outputs/logs
.venv/bin/python - "$ids" "$shards" "$prefix" "data/train+public_test/$split/$split.jsonl" <<'PY'
import json, sys
src, n, prefix, split_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
ids = [json.loads(l)["id"] for l in open(split_file, encoding="utf-8") if l.strip()] if src == "all" else json.load(open(src))
for k in range(n):
    json.dump(ids[k::n], open(f"outputs/splits/shards/{prefix}_s{k}.json", "w"))
PY
# DEADLINE 900: four local shards share one GPU; the Docker image uses 27 s per sample, one sample at a time.
args=(--split "$split" --allow-failures --deadline "${DEADLINE:-900}" --llm-url "http://127.0.0.1:${port}/v1/chat/completions" "${AGENT_FLAGS[@]}")
[ -n "$orch" ] && args+=(--orchestration "$orch")
if [ "$source" = asr ]; then
  [ -f "$cache" ] || { echo "missing ASR cache $cache (scripts/infer/asr_cache.py) or set CACHE" >&2; exit 1; }
  args+=(--transcript-source cached-asr --transcripts-jsonl "$cache" --asr-dir "$asr_dir")
else args+=(--transcript-source train-reference); fi
if [ "$model" = sft ]; then args+=(--adapter-dir "${ADAPTER:-outputs/release/$RELEASE_NAME/$LORA_DIRNAME}" --llm-served-model "${SERVED:-$LORA_NAME}"); fi
# extra flags, e.g. EXTRA="--max-turns 12"
if [ -n "${EXTRA:-}" ]; then read -r -a extra <<< "$EXTRA"; args+=("${extra[@]}"); fi
pids=()
for ((k = 0; k < shards; k++)); do
  PYTHONPATH=src .venv/bin/python -m lifemate.cli evaluate "${args[@]}" --sample-ids "outputs/splits/shards/${prefix}_s${k}.json" \
    --run-id "${prefix}_s${k}" > "outputs/logs/${prefix}_s${k}.log" 2>&1 &
  pids+=($!)
done
status=0
for k in "${!pids[@]}"; do wait "${pids[$k]}" || { status=1; echo "shard ${prefix}_s$k failed: outputs/logs/${prefix}_s$k.log" >&2; }; done
echo "$prefix done (status $status)"
exit $status
