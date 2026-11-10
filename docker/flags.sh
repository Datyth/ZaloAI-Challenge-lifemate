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
# Agent options read by lifemate.docker_predict (defaults = the single-model q35e1 setup):
#   LIFEMATE_ORCHESTRATION  E1 | E1D | E1P   prompt/context of the (primary) agent
#   LIFEMATE_RETRY_CAP      block a read tool after this many empty/not-found results (0 = off; 2 used with E1P models)
#   LIFEMATE_ENSEMBLE       served:orchestration,... step-level vote on the one ToolEnv, e.g.
#                           q35_e1:E1,q35_p1:E1P,q35_synth:E1 (needs every adapter in LORA_MODULES below and --max-loras >= 3)
export LIFEMATE_ORCHESTRATION=${LIFEMATE_ORCHESTRATION:-E1}
export LIFEMATE_RETRY_CAP=${LIFEMATE_RETRY_CAP:-0}
export LIFEMATE_ENSEMBLE=${LIFEMATE_ENSEMBLE:-}
# served name=adapter dir pairs; one pair for a single model
LORA_MODULES=("$LIFEMATE_SERVED_MODEL=$LORA_DIR")
