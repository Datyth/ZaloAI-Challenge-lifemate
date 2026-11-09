# Every setting of the image (the guide asks for fixed defaults; nothing is passed on the command line).
# Model and agent settings come from configs/release.sh, the file local serving and evaluation read too.
source /code/configs/release.sh
LLM_DIR=/code/models/llm_base
LORA_DIR=/code/models/lora
export LIFEMATE_SERVED_MODEL=$LORA_NAME
export LIFEMATE_ASR_DIR=/code/models/gipformer
export LIFEMATE_ASR_THREADS=8
export LIFEMATE_DEADLINE=27      # seconds per sample, ASR included (limit 30)
VLLM_FLAGS=("${VLLM_MODEL_FLAGS[@]}" "${VLLM_LORA_FLAGS[@]}" --gpu-memory-utilization 0.85 --max-num-seqs 4)

# The vLLM server of predict.sh and predict_notebook.ipynb: loopback only (the container has no network), base model + LoRA.
vllm_server() {
  exec python3 -m vllm.entrypoints.openai.api_server --model "$LLM_DIR" --served-model-name base --host 127.0.0.1 --port 8000 \
    --lora-modules "$LIFEMATE_SERVED_MODEL=$LORA_DIR" "${VLLM_FLAGS[@]}"
}
