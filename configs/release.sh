# Released pipeline (public 58.02, 2026-11-08): Gipformer ASR v2 on CPU -> Qwen3.5-4B + LoRA q35_e1 in vLLM -> E1 agent loop.
# Single source of the model and agent settings, sourced by scripts/infer/{serve,run_shards}.sh and docker/flags.sh, so local
# evaluation and the Docker image run the same configuration. Paths are not set here: each caller knows its own layout.

# Fine-tuned checkpoints (scripts/setup/get_checkpoint.sh downloads them to outputs/release/$RELEASE_NAME/{llm-lora,asr-gipformer-v2}).
RELEASE_NAME=lifemate-q35e1
RELEASE_HF_REPO=hugebenevolence/lifemate-q35e1
LORA_NAME=lifemate                       # served model id of the LoRA adapter

# vLLM: text layers only, Qwen3.5 tool and reasoning parsers, 16k context, seeded. Prefix caching stays on (two single-stream
# runs of 50 val samples: 0/50 rows differ with it, 2/50 without).
VLLM_MODEL_FLAGS=(--language-model-only --max-model-len 16384 --seed 42
                  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3)
VLLM_LORA_FLAGS=(--enable-lora --max-lora-rank 32)

# Requests (lifemate.llm): Qwen3.5's template takes one leading system message; non-thinking replies.
export LIFEMATE_MERGE_SYSTEM=1
export LIFEMATE_LLM_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'

# Agent loop for `lifemate.cli evaluate|submit`; lifemate.docker_predict hard-codes the same (lower-cased ASR, E1, dedup, 10 turns).
AGENT_FLAGS=(--orchestration E1 --asr-lowercase --dedup-calls --max-turns 10)
