# Agent E1 QLoRA SFT

## Scope and current status

The pipeline reuses the local Qwen3-4B-Instruct-2507 checkpoint, E1 prompt and execution code, official TRAIN ToolEnv, and saved Gipformer 1.5 transcripts. It saves a PEFT adapter separately; neither ASR nor base LLM weights are changed. Full training and 200-sample evaluation have **not** been started, as requested.

Measured prior results on the frozen 200 TRAIN validation IDs were S0 (Gipformer + E1) **65/200 (32.5%)** and S0-Oracle **88/200 (44.0%)**. Those figures are from earlier reports, not new SFT results. S1 and S1-Oracle results are pending.

The prior S0 run at `outputs/runs/e1_asr_full_01_A2` is intact. The earlier S0-Oracle report cites `outputs/runs/agentopt_full_01_E4`, but that 200-row run directory is absent from this workspace. `outputs/runs/e4_oracle_20261105` contains only 12 rows and cannot support a paired 200-row comparison. The commands below rerun S0-Oracle using the unmodified base model and the same E1 settings. The old aggregate 88/200 remains historical and should not be substituted for per-sample predictions.

## Dataset and training method

`prepare_agent_sft.py` reads only TRAIN samples. It excludes all 200 frozen validation IDs and samples whose history audio points to those IDs. Remaining samples are assigned to internal train/dev by a deterministic SHA-256 rule. Each example includes the gold current transcript, structured gold history, E1 system prompt, user context, available tool schemas, gold native tool calls, and observations returned by the official TRAIN ToolEnv. A call is kept only if ToolEnv validation and replay match the annotation. Empty-call and `sys.ask_missing_info` cases are retained. `skipped.jsonl` explains discarded trajectories.

The training script expands each trajectory into individual assistant decisions, with earlier real observations in the prompt. It uses the checkpoint's Qwen chat template with assistant loss boundaries that leave rendered inference text identical. Completion and assistant masks restrict loss to the current assistant decision. The initial config is NF4 double-quantized QLoRA, rank 16, alpha 32, learning rate 1e-4, three epochs, batch size 1, gradient accumulation 8, max length 8192, gradient checkpointing and BF16 when supported. Internal dev loss drives early stopping and checkpoint saving. Final adapter choice must use the frozen validation results from the evaluation commands, not internal dev loss alone.

The training process maps physical GPU 2 to its local CUDA device and checks for at least 18 GiB free before loading the model. The current execution sandbox could not query the NVIDIA driver, so a GPU training step has not been verified here. Stop any vLLM server occupying GPU 2 before starting training.

## Commands in execution order

Run in the repository terminal. Give new run IDs if these names already exist. Do not use a `--limit` dataset for full training.

### 1. Activate the environment

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
python -m pip install -r requirements-sft.txt
```

### 2. Prepare SFT data

```bash
python scripts/prepare_agent_sft.py --output-dir outputs/sft/data_v1
```

Output: `outputs/sft/data_v1/{train,dev,skipped}.jsonl` and `manifest.json`. Validation examples and their linked history audio are excluded.

### 3. Train QLoRA on physical GPU 2

Ensure the existing vLLM server is stopped so GPU 2 has at least 18 GiB free.

```bash
nohup python scripts/train_agent_sft.py --data-dir outputs/sft/data_v1 --output-dir outputs/sft/qwen3_4b_e1_qlora_01 --physical-gpu 2 > outputs/sft_train_01.log 2>&1 &
echo "Training PID: $!"
```

Output: checkpoints and `best_adapter` under `outputs/sft/qwen3_4b_e1_qlora_01/`; progress in `outputs/sft_train_01.log`. The script will not overwrite the base checkpoint.

### 4. Resume after interruption

Use the latest completed `checkpoint-N` directory from step 3. This example chooses it automatically:

```bash
checkpoint=$(find outputs/sft/qwen3_4b_e1_qlora_01 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
test -n "$checkpoint" || { echo 'No saved checkpoint found' >&2; exit 1; }
nohup python scripts/train_agent_sft.py --data-dir outputs/sft/data_v1 --output-dir outputs/sft/qwen3_4b_e1_qlora_01 --physical-gpu 2 --resume-from-checkpoint "$checkpoint" > outputs/sft_train_resume_01.log 2>&1 &
echo "Resume PID: $!"
```

Run this only if training was interrupted. The adapter is ready when `best_adapter/adapter_model.safetensors` exists.

### Start the LoRA inference server

After training has finished, run in a separate terminal with the same environment activated:

```bash
nohup bash scripts/serve_llm_lora.sh > outputs/vllm_sft_01.log 2>&1 &
echo "vLLM PID: $!"
until curl -fsS http://127.0.0.1:8000/v1/models | grep -q 'lifemate_sft'; do sleep 5; done
```

The server exposes the base model and adapter ID `lifemate_sft` on GPU 2. The following commands use this running server; activate `datpt_lifemate` in each new terminal.

### 5. Evaluate S1 with saved Gipformer transcripts

```bash
nohup bash scripts/run_agent_sft_eval.sh gipformer sft_01_S1 > outputs/sft_01_S1.log 2>&1 &
echo "S1 PID: $!"
```

Output: `outputs/runs/sft_01_S1/` with 200 predictions, execution traces, `metrics.json`, `runtime_errors.jsonl`, and the submission artifact. It reuses `outputs/runs/e1_asr_full_01_A2_cache/asr_outputs.jsonl`.

### 6. Evaluate S1-Oracle and regenerate missing S0-Oracle raw run

Wait for step 5 to finish, then run sequentially to avoid contention on the vLLM server:

```bash
nohup bash scripts/run_agent_sft_eval.sh oracle sft_01_S1_oracle > outputs/sft_01_S1_oracle.log 2>&1 &
echo "S1-Oracle PID: $!"
```

After that run finishes, obtain a complete base Oracle run for per-sample comparison:

```bash
nohup bash scripts/run_agent_sft_eval.sh base-oracle sft_01_S0_oracle > outputs/sft_01_S0_oracle.log 2>&1 &
echo "S0-Oracle PID: $!"
```

Both use the exact frozen 200 TRAIN IDs and gold current/history transcripts. Outputs are separate run directories under `outputs/runs/`.

To compare a different saved checkpoint, pass its directory to `scripts/serve_llm_lora.sh` and as the third argument to `scripts/run_agent_sft_eval.sh`; use new run IDs and compare its frozen-validation state match and paired regressions before choosing the final adapter.

### 7. Compare all four runs

After all three new evaluations have completed:

```bash
python scripts/compare_agent_sft.py --s0 outputs/runs/e1_asr_full_01_A2 --s1 outputs/runs/sft_01_S1 --s0-oracle outputs/runs/sft_01_S0_oracle --s1-oracle outputs/runs/sft_01_S1_oracle --output-dir outputs/reports/sft_01
```

Output: `outputs/reports/sft_01/{comparison.json,per_sample_regressions.jsonl,report.md}`. The report includes state match, strict exact-call F1, tool-name selection, exact arguments on aligned calls, read-to-write sequence completion, runtime errors, latency, and paired state transitions. The read-to-write measure is a sequence proxy; argument accuracy is conditional on aligned calls. These are local TRAIN metrics, not the competition score.

## Lightweight checks completed

- A 50-candidate TRAIN preparation smoke produced 47 train and 3 dev trajectories, with 0 skipped; it included empty-call, multi-step, and `sys.ask_missing_info` cases.
- The CPU training dry run processed eight train and eight dev decisions within the 8192-token limit and confirmed the assistant mask and unchanged rendered chat template.
- Focused SFT tests and shell syntax checks passed. No GPU training, full benchmark, or S1 result was produced.
