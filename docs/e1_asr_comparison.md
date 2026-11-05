# E1 Agent with three ASR models

The full end-to-end A0–A3 comparison is ready to run. At the user's request, only one-sample smoke tests for A1 and A2 have been executed; full 200-sample A1/A2 results are pending. Do not interpret the ASR-only winner as the end-to-end winner yet.

| Configuration | Transcript source | Agent |
| --- | --- | --- |
| A0 | Preserved Qwen3-ASR-0.6B run: `outputs/runs/train200_20261105_115719_E1` | E1 |
| A1 | ChunkFormer: current transcript from measured ASR run; history audio decoded into a cache | E1 |
| A2 | Gipformer 1.5: current transcript from measured ASR run; history audio decoded into a cache | E1 |
| A3 | Preserved oracle run: `outputs/runs/agentopt_full_01_E4` | E1 |

The runner checks the frozen 200 TRAIN IDs and verifies that A0/A3 use identical Qwen3-4B-Instruct-2507 checkpoint, E1 prompt, parser, decoding settings, deadline, ToolEnv kit version, and device. A1/A2 use those same agent settings. Gold transcript text and labels are never passed into A1/A2 inference. The caches hold the saved current audio transcripts and independently decoded history audio. Saved ASR decode latency for every audio turn is added to per-sample wall time and charged against the 30-second deadline. This gives an estimated end-to-end latency; A0/A3 were measured at an earlier time.

The preserved A0 result is 52/200 state matches, tool-call F1 0.2395, and zero runtime errors. Preserved A3 is 88/200 state matches, tool-call F1 0.4005, and zero runtime errors. A1/A2 full-run results have not been measured yet.

## Run the complete benchmark

Start the Qwen3-4B vLLM server on physical GPU 2 if it is not already serving the project checkpoint:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  nohup bash scripts/serve_llm.sh > outputs/vllm_e1_asr.log 2>&1 &
fi
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 5; done
```

Once `/v1/models` responds, launch the sequential cache and Agent runs. Use a fresh prefix for a new attempt:

```bash
cd /home/datpt/projects/ZaloAI-Challenge-lifemate
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
nohup bash scripts/run_e1_asr_comparison.sh cuda:2 e1_asr_full_01 > outputs/e1_asr_full_01.log 2>&1 &
echo "Benchmark PID: $!"
tail -f outputs/e1_asr_full_01.log
```

The script uses existing ASR results under `outputs/runs/asr_compare_final_20261105_{chunkformer,gipformer}`. It prepares `outputs/runs/<prefix>_A1_cache/` and `_A2_cache/`, then writes distinct `outputs/runs/<prefix>_A1/` and `_A2/` directories. Each Agent run contains `predictions.jsonl`, `llm_turns.jsonl`, `tool_events.jsonl`, `asr_outputs.jsonl`, `terminations.jsonl`, `runtime_errors.jsonl`, `metrics.json`, `run_config.json`, and replayable `submission.jsonl`. The final report, paired transitions, and summary JSON are under `outputs/reports/<prefix>/`. Completed stages are reused when re-running the same prefix; an incomplete stage requires inspection and a new prefix.

The comparison reports state match, exact tool-call F1 surrogate, runtime error count, mean/P95 latency, and all four correct/incorrect transitions relative to A0 for every configuration. It requires 200 predictions in the frozen order and successful ToolEnv replay before creating the report.

## Verification completed

For `train_0063`, the two ASR cache stages each saved one measured current transcript and decoded one history audio turn. Both E1 smoke runs completed with zero runtime errors and passed ToolEnv replay (`1/1`). The 200-sample A0 and A3 artifacts passed the matching-configuration check. These smoke tests verify execution, not model ranking.
