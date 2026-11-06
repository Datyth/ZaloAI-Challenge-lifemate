# Agent pipeline optimization (TRAIN reference transcripts)

## Scope and evidence

`Review_repo.md` correctly identifies several orchestration risks. The official `zaic_tools` runtime says `validate` checks only the schema; `call` with `ok=true, done=false` means the operation ran but changed nothing. Empty read results can still have `ok=true`. Tool calls rejected by schema are recorded if sent to `env.call`, so the agent must validate before execution.

Implementation is on branch `review-agent-pipeline`; the original E1 inference behavior remains selectable as O0/E1 and is still the CLI default. Existing SFT files, checkpoints, and training processes were left intact.

The frozen validation set is `outputs/splits/validation_ids.json` (200 TRAIN IDs). O0 maps to the original E1 orchestration. O1–O5 are cumulative, selectable with `--orchestration`. Gold labels are used only after inference for evaluation.

## Confirmed root causes and changes

| Stage | Root cause | Change |
|---|---|
| O1 | E1 treated `finish_reason=length` as a normal stop and could append malformed native calls without matching tool responses. Long observations were cut into invalid JSON. | Reject truncated or inconsistent finish reasons before executing calls; reject missing or duplicate call IDs; send a matching validation response for malformed calls with IDs; summarize large observations as valid JSON while retaining status and counts. |
| O2 | Global write flags did not describe individual calls and E1 could retry a side effect. | Record validation, execution, and outcome for each proposed call in `execution_ledger.jsonl`; classify completed, unsuccessful, observed, and waiting results; block duplicate write signatures, including ambiguous attempts. |
| O3 | E6's completion check was skipped after any successful write. | Permit one bounded check after a read or after the first write in a request with multiple explicit action verbs. The prompt asks for remaining work and never issues a write itself. Duplicate protection remains active. |
| O4 | App-wide observed IDs could ground the wrong entity type. | Ground `x-ref` values to the matching named collection in read observations and history; retain literal user-supplied IDs. ToolEnv's own schema validation remains authoritative. |
| O5 | E1 prompt contained concrete calendar/SMS routing examples. | Use a general task-execution prompt; preserve the structured, read-only history. |

`predictions.jsonl` now separates `runtime_status`, `tool_outcome`, and `task_completion`. TRAIN task completion is based on final-state comparison after inference; public-test completion remains unknown. `status=tool_error` and `status=partial_write` no longer appear as `status=ok`.

## Reproduce

First inspect GPU use with `nvidia-smi`. In this session GPU 2 was occupied by training, so inference used GPU 0. Start the existing checkpoint on a separate port:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/datpt/.conda/envs/datpt_lifemate/bin/vllm serve \
  "$PWD/models/qwen3-4b-instruct-2507" --host 127.0.0.1 --port 8001 \
  --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.65 --seed 42
```

In another shell:

```bash
PYTHONPATH=src python -m pytest -q
bash scripts/run_agent_pipeline_ablations.sh cuda:0 8001 pipeline_full_20261105
```

To recreate the disjoint 50-ID holdout selection and its sequential comparison:

```bash
PYTHONPATH=src python - <<'PY'
import json, random
from pathlib import Path
from lifemate.data import items
root = Path('data/train+public_test')
validation = set(json.loads(Path('outputs/splits/validation_ids.json').read_text()))
remaining = sorted(item['id'] for item in items(root, 'train') if item['id'] not in validation)
Path('outputs/splits/pipeline_holdout_ids.json').write_text(json.dumps(random.Random(20261105).sample(remaining, 50), indent=2) + '\n')
PY
for stage in O0 O5; do
  mode="$stage"; [ "$stage" = O0 ] && mode=E1
  PYTHONPATH=src /home/datpt/.conda/envs/datpt_lifemate/bin/python -m lifemate.cli evaluate \
    --split train --sample-ids outputs/splits/pipeline_holdout_ids.json \
    --transcript-source train-reference --device cuda:0 --orchestration "$mode" \
    --llm-url http://127.0.0.1:8001/v1/chat/completions \
    --run-id "pipeline_holdout_20261105_${stage}" \
    --output "outputs/runs/pipeline_holdout_20261105_${stage}/submission.jsonl"
done
PYTHONPATH=src python scripts/summarize_agent_pipeline.py \
  outputs/runs/pipeline_holdout_20261105_O0 outputs/runs/pipeline_holdout_20261105_O5 \
  --expected-samples 50 --output outputs/reports/pipeline_holdout_20261105_comparison.json
```

Each stage writes its own `outputs/runs/<prefix>_O*/` directory with predictions, tool traces, metrics, and configuration. `scripts/summarize_agent_pipeline.py` writes the paired comparison, per-sample outcomes, and correct-to-incorrect and incorrect-to-correct transitions under `outputs/reports/`.

## Results and limits

The historical 88/200 (44.0%) and tool-call F1 0.401 quoted in `Review_repo.md` are from prior reports; their complete 200-row oracle run is absent from this workspace. A different preserved 200-row E1 ASR run has 52/200 state matches and F1 0.240, so it cannot serve as the TRAIN-reference baseline. Fresh O0–O5 results must come from the paired run above.

Fresh results from `pipeline_full_20261105` (all six modes used the same frozen 200 IDs, TRAIN reference transcripts, checkpoint, parser, decoding, and ToolEnv):

| Mode | State matches | Exact-call F1 surrogate | Exact tool selection | Incomplete multi-step surrogate | Invalid proposals | Executed no-ops/failures | Mean / P95 seconds | Runtime errors | Improved / regressed vs O0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O0 (E1) | 89/200 (44.5%) | 0.395 | 35.5% | 26 | 31 | 13 | 4.87 / 9.76 | 0 | 0 / 0 |
| O1 | 84/200 (42.0%) | 0.387 | 35.5% | 28 | 32 | 15 | 5.94 / 11.64 | 0 | 0 / 5 |
| O2 | 83/200 (41.5%) | 0.389 | 34.5% | 30 | 31 | 13 | 5.80 / 11.46 | 0 | 0 / 6 |
| O3 | 87/200 (43.5%) | 0.394 | 35.0% | 25 | 31 | 14 | 6.45 / 14.51 | 0 | 2 / 4 |
| O4 | 87/200 (43.5%) | 0.396 | 35.0% | 26 | 31 | 14 | 6.46 / 14.02 | 0 | 1 / 3 |
| O5 | 90/200 (45.0%) | 0.410 | 40.5% | 30 | 36 | 13 | 7.11 / 19.11 | 0 | 15 / 14 |

The complete per-sample comparison is in `outputs/reports/pipeline_full_20261105_comparison.json`. In O1, all five regressions began with model output differences from O0 despite identical first-turn input messages. The runs shared a vLLM server concurrently; `temperature=0` and seed 42 did not make the server's outputs identical under that scheduling. These small differences therefore do not establish a causal accuracy effect for each code change. O5 has the highest observed state match and F1 on the 200 samples, but its one-sample state-match lead is too small to justify changing the default by itself.

The additional 50-ID TRAIN holdout was selected from IDs outside the frozen 200 using seed 20261105, then O0 and O5 were run sequentially. O0 and O5 both matched 25/50 final states, with 2 improvements and 2 regressions for O5. Exact-call F1 increased from 0.359 to 0.404. Both runs had zero runtime errors. The holdout is disjoint from the frozen validation IDs, although prior work on this repository means it cannot be certified untouched by every earlier experiment. Details are in `outputs/reports/pipeline_holdout_20261105_comparison.json`.

**Recommendation:** Keep E1/O0 as the default for now. O5 is the best observed 200-sample configuration, but its state-match gain did not replicate on the disjoint holdout and its validation P95 latency rose from 9.76 to 19.11 seconds. Use O5 as an opt-in candidate when protocol safety and call F1 matter, then evaluate on more unseen requests before changing the default. O1–O4 individually did not improve the measured 200-sample state-match rate.

The completion check uses explicit Vietnamese action-verb cues and can miss paraphrases. Grounding is conservative when a tool's response omits its collection name. A successful ToolEnv write is evidence of that operation's outcome, not proof that the whole user request is complete. The best configuration should be selected by final-state accuracy first, then checked for regressions in the per-sample report.
