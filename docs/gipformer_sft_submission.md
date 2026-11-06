# Gipformer SFT and PUBLIC_TEST submission

Run from the repository root. Every output path below is new; the original
Gipformer and S1 LoRA checkpoints stay intact. Physical GPU 2 is used throughout.

## 1. Valid baseline submission first

The original Gipformer PUBLIC_TEST cache is already complete at
`outputs/runs/gipformer_public_s1_20261106_cache` (634 samples, 1933 current and
history audio files, 0 ASR failures). Start the S1 server in one terminal:

```bash
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
unset CUDA_VISIBLE_DEVICES
nohup bash scripts/serve_llm_lora.sh > outputs/vllm_public_s1_20261106.log 2>&1 &
vllm_pid=$!
echo "vLLM PID: $vllm_pid"
bash scripts/wait_lora_server.sh "$vllm_pid" || { tail -n 50 outputs/vllm_public_s1_20261106.log; exit 1; }
```

Submit using the original ASR and S1 adapter. `--allow-failures` keeps a
replay-valid row for a sample even if its Agent execution fails; inspect
`runtime_errors.jsonl` and `metrics.json` before upload.

```bash
export PYTHONPATH=src
nohup python -m lifemate.cli submit --split public_test --device cuda:2 \
  --orchestration E1 --parser hermes --allow-failures \
  --adapter-dir outputs/sft/qwen3_4b_e1_qlora_01/best_adapter \
  --llm-served-model lifemate_sft --asr-backend gipformer \
  --asr-dir models/gipformer1.5-68m-rnnt --transcript-source cached-asr \
  --transcripts-jsonl outputs/runs/gipformer_public_s1_20261106_cache/asr_outputs.jsonl \
  --run-id public_s1_gipformer_base_20261106 \
  --output result/submission_s1_gipformer_base_20261106.jsonl \
  > outputs/public_s1_gipformer_base_20261106.log 2>&1 &
echo "Submission PID: $!"
submission_pid=$!
wait "$submission_pid"
python -m lifemate.cli validate-submission --split public_test \
  --file result/submission_s1_gipformer_base_20261106.jsonl
PYTHONPATH=data/train+public_test/public_test python -m zaic_tools \
  --split public_test --root data/train+public_test \
  --validate result/submission_s1_gipformer_base_20261106.jsonl
kill "$vllm_pid"
```

`wait "$submission_pid"` only waits for this submission, even if the vLLM
server was started in the same shell. If using another terminal, wait for the
log's `wrote 634 rows` message before validation.

## 2. Optional fine-tuning after baseline is secured

The original `model.pt` is a PyTorch icefall checkpoint. The upstream Gipformer
project pins PyTorch 2.8 and matching k2/kaldifeat CUDA 12.8 wheels, so use its
separate `.venv`. The `datpt_lifemate` environment remains the ONNX/vLLM
inference environment. Stop the vLLM process before training to free GPU 2.

```bash
cd models/.research/gipformer-upstream
uv sync --extra pytorch --python 3.11
uv pip install --python .venv/bin/python lhotse onnx onnxruntime urllib3
cd ../../..
models/.research/gipformer-upstream/.venv/bin/python scripts/gipformer_pt.py \
  --strict-load > outputs/asr_sft/gipformer_strict_load_20261106.json
```

The last command **must succeed** before training. It uses
`load_state_dict(strict=True)` and stops on any architecture mismatch. Do not
switch to `strict=False` or train ONNX files.

One gradient smoke check (no optimizer step and no checkpoint output):

```bash
CUDA_VISIBLE_DEVICES=2 models/.research/gipformer-upstream/.venv/bin/python \
  scripts/train_gipformer_sft.py --physical-gpu 2 --smoke-only \
  --max-train-audio 1 --max-dev-audio 1
```

The prepared TRAIN-only manifest is in
`outputs/asr_sft/gipformer_data_20261106/manifest.json`. To rebuild under a
different output directory:

```bash
python scripts/prepare_gipformer_sft.py \
  --output-dir outputs/asr_sft/gipformer_data_v2_20261106
```

Fine-tune only the encoder at a low learning rate. The default caps are 800
training audio files and 100 development audio files, for at most two epochs;
the script saves checkpoints and stops early on development loss.

```bash
CUDA_VISIBLE_DEVICES=2 nohup models/.research/gipformer-upstream/.venv/bin/python \
  scripts/train_gipformer_sft.py \
  --data-dir outputs/asr_sft/gipformer_data_20261106 \
  --output-dir outputs/asr_sft/gipformer_run_01 --physical-gpu 2 \
  > outputs/asr_sft/gipformer_train_01.log 2>&1 &
echo "Training PID: $!"
```

After `best.pt` exists and training has ended, export to a new model directory:

```bash
CUDA_VISIBLE_DEVICES=2 models/.research/gipformer-upstream/.venv/bin/python \
  scripts/export_gipformer_sft.py \
  --best outputs/asr_sft/gipformer_run_01/best.pt \
  --output-dir models/gipformer1.5-lifemate-sft-01 --device cuda:2
```

This export checks all PyTorch keys strictly and decodes one TRAIN audio file
with the exported ONNX model. The `--device cuda:2` argument identifies the
physical GPU; Sherpa ONNX maps it to visible device 0 internally.

## 3. Frozen TRAIN validation and model selection

Generate candidate ASR for all current and history recordings of the same 200
samples. Keep the LoRA server on GPU 2 for the Agent evaluation; run the ONNX
cache first to avoid GPU contention.

```bash
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
unset CUDA_VISIBLE_DEVICES
python scripts/cache_gipformer_audio.py --split train --device cuda:2 \
  --sample-ids outputs/splits/validation_ids.json \
  --model-dir models/gipformer1.5-lifemate-sft-01 \
  --output-dir outputs/runs/gipformer_sft_val_20261106_cache
nohup bash scripts/serve_llm_lora.sh > outputs/vllm_sft_val_20261106.log 2>&1 &
vllm_pid=$!
bash scripts/wait_lora_server.sh "$vllm_pid" || { tail -n 50 outputs/vllm_sft_val_20261106.log; exit 1; }
export PYTHONPATH=src
nohup python -m lifemate.cli evaluate --split train \
  --sample-ids outputs/splits/validation_ids.json --device cuda:2 \
  --orchestration E1 --parser hermes --allow-failures \
  --adapter-dir outputs/sft/qwen3_4b_e1_qlora_01/best_adapter \
  --llm-served-model lifemate_sft --asr-backend gipformer \
  --asr-dir models/gipformer1.5-lifemate-sft-01 \
  --transcript-source cached-asr \
  --transcripts-jsonl outputs/runs/gipformer_sft_val_20261106_cache/asr_outputs.jsonl \
  --run-id gipformer_sft_s1_val_20261106 \
  > outputs/gipformer_sft_s1_val_20261106.log 2>&1 &
echo "Evaluation PID: $!"
evaluation_pid=$!
wait "$evaluation_pid"
python scripts/compare_gipformer_sft.py \
  --candidate-cache outputs/runs/gipformer_sft_val_20261106_cache \
  --candidate-run outputs/runs/gipformer_sft_s1_val_20261106 \
  --candidate-model models/gipformer1.5-lifemate-sft-01 \
  --output-dir outputs/reports/gipformer_sft_20261106
cat outputs/reports/gipformer_sft_20261106/report.md
kill "$vllm_pid"
```

The report compares WER, CER, critical-slot accuracy, current and history
audio, state matches, runtime errors, latency and paired transitions. Its
`selected_asr_dir` remains the original Gipformer unless **all** selection
gates pass.

## 4. Candidate PUBLIC_TEST submission, only if selected

If `selected_asr_dir` in `comparison.json` names
`models/gipformer1.5-lifemate-sft-01`, produce a separate cache and submission:

```bash
python scripts/cache_gipformer_audio.py --split public_test --device cuda:2 \
  --model-dir models/gipformer1.5-lifemate-sft-01 \
  --output-dir outputs/runs/gipformer_sft_public_20261106_cache
nohup bash scripts/serve_llm_lora.sh > outputs/vllm_sft_public_20261106.log 2>&1 &
vllm_pid=$!
bash scripts/wait_lora_server.sh "$vllm_pid" || { tail -n 50 outputs/vllm_sft_public_20261106.log; exit 1; }
export PYTHONPATH=src
nohup python -m lifemate.cli submit --split public_test --device cuda:2 \
  --orchestration E1 --parser hermes --allow-failures \
  --adapter-dir outputs/sft/qwen3_4b_e1_qlora_01/best_adapter \
  --llm-served-model lifemate_sft --asr-backend gipformer \
  --asr-dir models/gipformer1.5-lifemate-sft-01 \
  --transcript-source cached-asr \
  --transcripts-jsonl outputs/runs/gipformer_sft_public_20261106_cache/asr_outputs.jsonl \
  --run-id public_s1_gipformer_sft_20261106 \
  --output result/submission_s1_gipformer_sft_20261106.jsonl \
  > outputs/public_s1_gipformer_sft_20261106.log 2>&1 &
echo "Submission PID: $!"
submission_pid=$!
wait "$submission_pid"
python -m lifemate.cli validate-submission --split public_test \
  --file result/submission_s1_gipformer_sft_20261106.jsonl
PYTHONPATH=data/train+public_test/public_test python -m zaic_tools \
  --split public_test --root data/train+public_test \
  --validate result/submission_s1_gipformer_sft_20261106.jsonl
kill "$vllm_pid"
```
