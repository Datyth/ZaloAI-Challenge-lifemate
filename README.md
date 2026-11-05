# LifeMate offline baseline

This repository contains the organizer's v3.2.0 TRAIN/PUBLIC_TEST kit and a local Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507 inference runner. See [competition contract](docs/competition_contract.md), [compatibility status](docs/compatibility.md), [measured results](docs/execution_log.md), and [orchestration improvements](docs/orchestration_improvements.md). The model and data directories are ignored by Git.

The E1 QLoRA SFT dataset, training, LoRA evaluation, and paired comparison commands are in [the Agent SFT report](docs/agent_sft_report.md). Full training and validation runs are left for manual execution.

## Compare Vietnamese ASR models on TRAIN

The ASR benchmark compares the preserved Qwen3-ASR-0.6B run with local ChunkFormer-RNNT-Large-Vie and Gipformer 1.5 checkpoints on the same frozen 200 TRAIN IDs. The two new adapters use the existing `benchmark-asr` interface. Run the full comparison on physical GPU 2 with:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
nohup bash scripts/run_asr_comparison.sh cuda:2 asr_compare_manual_01 > outputs/asr_compare_manual_01.log 2>&1 &
echo "ASR benchmark PID: $!"
```

Use a new prefix for each attempt. The script does not require the vLLM server. It writes separate `asr_outputs.jsonl`, `per_sample.jsonl`, and `errors.jsonl` files for each new model under `outputs/runs/<prefix>_<model>/`, then writes `outputs/reports/<prefix>/report.md`. See [ASR benchmark details](docs/asr_benchmark.md) for measured results, metric definitions, and dataset limits.

For the end-to-end A0–A3 comparison with the unchanged E1 Agent, see [the E1 × ASR benchmark instructions](docs/e1_asr_comparison.md). The runner reuses A0 and oracle A3, prepares history audio caches for ChunkFormer and Gipformer, then runs A1/A2 sequentially and reports paired state transitions. One-sample smoke tests passed; the full A1/A2 runs are left for the user to launch.

The baseline checkpoints are downloaded and `datpt_lifemate` ran real-model experiments on physical GPU 2. On the fixed 200-sample TRAIN comparison, E1 had the best completed result: 52/200 final-state matches, exact-call F1 surrogate 0.2395, and zero failures. E2 also matched 52/200 states but failed four samples. E1 is now the CLI default. No official score is available. The existing `result/submission.jsonl` has 634 PUBLIC_TEST rows and passed CPU ToolEnv replay validation with 634 matching hashes; it remains the earlier E0 artifact.

## Run E4–E6 TRAIN optimization

E1 remains the default baseline. `scripts/run_agent_optimization.sh` runs E4 with TRAIN reference transcripts, writes an E5 analysis of the 148 E1 failures, then runs E6 and E6 with oracle transcripts on the same frozen 200 IDs. The script writes separate run directories and regenerates [the optimization report](docs/agent_optimization_report.md). The `agentopt_full_01` run completed all stages with no sample failures.

In terminal 1, activate the requested environment and start the GPU 2 LLM server:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
bash scripts/serve_llm.sh
```

In terminal 2, activate the same environment and run the experiments. Give each full attempt a new prefix; an incomplete stage keeps its `.partial` file and the script will stop rather than overwrite it.

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
bash scripts/run_agent_optimization.sh cuda:2 agentopt_full_01 2>&1 | tee outputs/agentopt_full_01.log
```

The baseline E1 run is `outputs/runs/train200_20261105_115719_E1`. Raw oracle transcript pairs, E5 per-sample evidence, and the four-way comparison are written under `outputs/reports/agentopt_full_01/`. Select E6 for PUBLIC_TEST only if the completed TRAIN report supports it.

## Generate a new PUBLIC_TEST submission when GPUs are available

Use two terminals from this repository. In terminal 1, start the local vLLM server. If `curl http://127.0.0.1:8000/v1/models` already returns this checkpoint, keep that server and skip the start command.

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
bash scripts/serve_llm.sh
```

In terminal 2, run all 634 PUBLIC_TEST samples on physical GPU 2, then validate the complete output with both the project replay check and the organizer's validator. Do not set `CUDA_VISIBLE_DEVICES=2` in terminal 2; that setting is local to the vLLM process in terminal 1.

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
export PYTHONPATH=src
set -e
python -m lifemate.cli submit --split public_test --device cuda:2 --orchestration E1 --output result/submission.jsonl
python -m lifemate.cli validate-submission --split public_test --file result/submission.jsonl
PYTHONPATH=data/train+public_test/public_test python -m zaic_tools --split public_test --root data/train+public_test --validate result/submission.jsonl
```

The runner writes per-sample ASR, LLM, and tool traces to `outputs/runs/<run-id>/`, reports progress every 25 samples, and writes `result/submission.jsonl` only if every sample succeeds and replay validation passes. The full run may take roughly an hour based on the small TRAIN run; this is an estimate, not a measured full-run time. If it stops early, inspect the latest run directory and do not submit a `.partial` file. `set -e` prevents validation commands from running when inference fails.

## Historical recovery procedure for the 2026-11-04 interrupted full run

The run `20261104T192413_submit_570038` produced 634 replayable diagnostic rows but flagged `public_0392` and `public_0421` for exceeding 10 LLM turns. With the local vLLM server running, rerun only those two samples with a higher turn limit, then merge their successful rows with the other 632. This does not rerun the full dataset:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
export PYTHONPATH=src
set -e
python -m lifemate.cli submit --split public_test --sample-id public_0392 --max-turns 20 --device cuda:2 --output result/retry_public_0392.jsonl
python -m lifemate.cli submit --split public_test --sample-id public_0421 --max-turns 20 --device cuda:2 --output result/retry_public_0421.jsonl
python scripts/recover_submission.py --run-dir outputs/runs/20261104T192413_submit_570038 --partial result/submission.jsonl.20261104T192413_submit_570038.partial --replacement public_0392=result/retry_public_0392.jsonl --replacement public_0421=result/retry_public_0421.jsonl --output result/submission.jsonl
python -m lifemate.cli validate-submission --split public_test --file result/submission.jsonl
PYTHONPATH=data/train+public_test/public_test python -m zaic_tools --split public_test --root data/train+public_test --validate result/submission.jsonl
```

If either targeted retry still fails, inspect its new run log and do not run the merge step. The recovery script checks both replacement files and replays all 634 merged rows with the organizer's ToolEnv before writing the final file.

After the first full submission validates, the remaining checkpoint download is unlocked:

```bash
python -m lifemate.cli download --group remaining
```

Model comparison has not run. It must use real TRAIN measurements and check legal parameter totals before pair selection. See [model comparison status](outputs/reports/model_comparison.md), which is a local ignored artifact.
