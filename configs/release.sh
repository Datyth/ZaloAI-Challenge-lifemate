# Released pipeline (public 61.26, 2026-11-09):
#   Gipformer ASR "final" (68M, fine-tuned on all TRAIN audio) on CPU
#   -> Qwen3.5-4B + LoRA "p1" (all TRAIN tasks, E1P prompt, recovery rows, message text weight 0.3) in vLLM
#   -> agent loop on the sample's single ToolEnv (schema and x-ref checks, no repeats, retry cap 2, 10 turns).
# Optional: a step-level vote of three LoRA adapters on the same ToolEnv (set ENSEMBLE below; src/lifemate/ensemble.py).
#
# Single source of the model and agent settings. scripts/infer/{serve,run_shards}.sh and docker/flags.sh source it, so local
# evaluation and the Docker image run the same configuration. Paths are not set here: each caller knows its own layout.

# Fine-tuned checkpoints: outputs/release/$RELEASE_NAME/ locally (scripts/setup/get_checkpoint.sh or scripts/train/build_release.sh),
# /code/models/ in the Docker image. One folder per checkpoint:
RELEASE_NAME=lifemate-final
RELEASE_HF_REPO=hugebenevolence/lifemate-final
ASR_DIRNAME=asr-gipformer                # Gipformer ONNX (encoder, decoder, joiner, tokens.txt)
LORA_NAME=lifemate                       # served model id of the main adapter
LORA_DIRNAME=llm-lora-p1                 # main adapter: Qwen3.5-4B LoRA "p1"

# Step-level vote: "served:orchestration,..." with the main adapter first (it breaks ties). Empty = the main adapter alone.
# Public: main adapter alone 61.26; the vote of the three is scored with the same ASR in result/submission_ens3full_*.jsonl.
ENSEMBLE=""
# ENSEMBLE="lifemate:E1P,lifemate_e1:E1,lifemate_synth:E1"

# Adapters the vLLM server loads, as served_name=folder (folder under the checkpoint root of the caller).
LORA_ADAPTERS=("$LORA_NAME=$LORA_DIRNAME")
if [ -n "$ENSEMBLE" ]; then LORA_ADAPTERS+=("lifemate_e1=llm-lora-e1" "lifemate_synth=llm-lora-synth"); fi

# vLLM: text layers only, Qwen3.5 tool and reasoning parsers, 16k context, seeded. Prefix caching stays on (two single-stream
# runs of 50 val samples: 0/50 rows differ with it, 2/50 without). Every adapter stays resident (--max-loras), so a vote
# queries all of them in one step.
VLLM_MODEL_FLAGS=(--language-model-only --max-model-len 16384 --seed 42
                  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3)
VLLM_LORA_FLAGS=(--enable-lora --max-lora-rank 32 --max-loras "${#LORA_ADAPTERS[@]}")

# Requests (lifemate.llm): Qwen3.5's template takes one leading system message; non-thinking replies.
export LIFEMATE_MERGE_SYSTEM=1
export LIFEMATE_LLM_EXTRA='{"chat_template_kwargs": {"enable_thinking": false}}'

# Agent loop. `lifemate.cli evaluate|submit` takes AGENT_FLAGS; lifemate.docker_predict reads the LIFEMATE_* variables that
# docker/flags.sh exports from the same values (ASR text is always lower-cased and identical calls are never repeated there).
AGENT_ORCHESTRATION=E1P                  # prompt of the main adapter (E1P = E1 + weekday calendar + app names + recovery hints)
AGENT_RETRY_CAP=2                        # block a read tool after it came back empty twice
AGENT_MAX_TURNS=10
AGENT_FLAGS=(--orchestration "$AGENT_ORCHESTRATION" --asr-lowercase --dedup-calls --retry-cap "$AGENT_RETRY_CAP" --max-turns "$AGENT_MAX_TURNS")
if [ -n "$ENSEMBLE" ]; then AGENT_FLAGS+=(--ensemble "$ENSEMBLE"); fi
