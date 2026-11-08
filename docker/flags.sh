# Every setting of the final run, in one place (the guide asks for fixed defaults; nothing is passed on the command line).
# Base model, adapter and parser of the chosen agent (Qwen3.5-4B + LoRA: qwen3_coder parser, one merged system message, no thinking).
LLM_DIR=/code/models/llm_base
LORA_DIR=/code/models/lora
TOOL_PARSER=qwen3_coder
export LIFEMATE_MERGE_SYSTEM=1
export LIFEMATE_LLM_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'
export LIFEMATE_SERVED_MODEL=lifemate
export LIFEMATE_ASR_DIR=/code/models/gipformer
export LIFEMATE_ASR_THREADS=8
export LIFEMATE_DEADLINE=27
VLLM_FLAGS=(--max-model-len 16384 --gpu-memory-utilization 0.85 --seed 42 --max-lora-rank 32 --max-num-seqs 4)   # prefix caching stays on: val50 single stream gave 0/50 rows differing with it, 2/50 without
