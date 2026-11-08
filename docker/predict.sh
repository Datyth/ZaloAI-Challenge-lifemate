#!/usr/bin/env bash
# Entry point of the image: bash /code/predict.sh (no arguments). Reads /data/<split>, writes /result/submission.jsonl
# (one line per sample: id, tool_calls, state_hash, time in ms) and /result/logs/ (trace of every sample, vLLM log).
# 1. start a local vLLM server (loopback only; the container has no network) with the base model and the LoRA adapter
# 2. run the samples one by one: Gipformer ASR on CPU, then the agent loop on that sample's single ToolEnv
set -uo pipefail
cd /code
source ./flags.sh
mkdir -p /result/logs
t0=$(date +%s)
python3 -m vllm.entrypoints.openai.api_server --model "$LLM_DIR" --served-model-name base --host 127.0.0.1 --port 8000 \
  --language-model-only --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  --enable-lora --lora-modules "$LIFEMATE_SERVED_MODEL=$LORA_DIR" "${VLLM_FLAGS[@]}" > /result/logs/vllm.log 2>&1 &
server=$!
until curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
  kill -0 $server 2>/dev/null || { echo "vLLM failed to start, see /result/logs/vllm.log"; tail -20 /result/logs/vllm.log; exit 1; }
  sleep 5
done
echo "lifemate: model loaded in $(( $(date +%s) - t0 ))s"
python3 -m lifemate.docker_predict --data /data --out /result
status=$?
kill $server; wait $server 2>/dev/null
echo "lifemate: done in $(( $(date +%s) - t0 ))s (exit $status)"
exit $status
