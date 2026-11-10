# Every setting of the image (the guide asks for fixed defaults; nothing is passed on the command line).
# Model and agent settings come from configs/release.sh, the file local serving and evaluation read too.
source /code/configs/release.sh
LLM_DIR=/code/models/llm_base
MODELS=/code/models
export LIFEMATE_SERVED_MODEL=$LORA_NAME
export LIFEMATE_ASR_DIR=$MODELS/$ASR_DIRNAME
export LIFEMATE_ASR_THREADS=8
export LIFEMATE_DEADLINE=27                      # seconds per sample, ASR included (limit 30)
export LIFEMATE_ORCHESTRATION=$AGENT_ORCHESTRATION
export LIFEMATE_RETRY_CAP=$AGENT_RETRY_CAP
export LIFEMATE_MAX_TURNS=$AGENT_MAX_TURNS
export LIFEMATE_ENSEMBLE=$ENSEMBLE
LORA_MODULES=()
for pair in "${LORA_ADAPTERS[@]}"; do LORA_MODULES+=("${pair%%=*}=$MODELS/${pair#*=}"); done
VLLM_FLAGS=("${VLLM_MODEL_FLAGS[@]}" "${VLLM_LORA_FLAGS[@]}" --gpu-memory-utilization 0.85 --max-num-seqs 4)

# The vLLM server of predict.sh and predict_notebook.ipynb: loopback only (the container has no network), base model + LoRA(s).
vllm_server() {
  exec python3 -m vllm.entrypoints.openai.api_server --model "$LLM_DIR" --served-model-name base --host 127.0.0.1 --port 8000 \
    --lora-modules "${LORA_MODULES[@]}" "${VLLM_FLAGS[@]}"
}
