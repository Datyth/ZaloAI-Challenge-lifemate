# Fine-tuned Gipformer: TRAIN validation and submission decision

Evaluated 2026-11-06 on the frozen 200 TRAIN validation IDs in
`outputs/splits/validation_ids.json`. The official PUBLIC_TEST score **30.77**
belongs to the previously submitted original Gipformer + S1 baseline (as
provided in `Comment_and_improvement_4.md`); it is not a score measured by
this local evaluation.

## Decision

**Keep the original Gipformer 1.5 + Qwen3-4B LoRA S1 + E1.** The adapted ASR
fails the end-to-end selection gate. No adapted-model PUBLIC_TEST submission
was generated or uploaded. The valid fallback remains
`result/submission_s1_gipformer_base_20261106.jsonl`.

| TRAIN validation metric | Original Gipformer + S1 | Adapted Gipformer + S1 |
| --- | ---: | ---: |
| Current audio WER | 17.60% | 98.98% |
| Current audio CER | 12.51% | 94.73% |
| Critical slots preserved | 45/64 (70.31%) | 0/64 (0%) |
| History audio WER | 16.62% | 99.86% |
| History audio CER | 11.39% | 92.88% |
| ASR decode failures, current + history | 0/762 | 0/762 |
| ASR current audio latency, mean / P95 | 0.279 / 0.427 s | 0.265 / 0.372 s |
| ASR history audio latency, mean / P95 | 0.182 / 0.258 s | 0.182 / 0.249 s |
| State match, non-error executions | 93/200 (46.5%) | 44/200 (22.0%) |
| Exact tool-call F1, non-error executions | 0.492 | 0.110 |
| Non-OK samples | 7 | 1 |
| Runtime exceptions | 0 | 0 |
| Rejected tool proposals | 6 | 2 |
| Agent latency, mean / P95 | 4.29 / 8.12 s | 1.79 / 3.66 s |

Paired state-match transitions against the original S1 baseline: **7
incorrect→correct, 56 correct→incorrect**, a net loss of 49. The candidate
had 45/200 raw final-state matches, including one tool-error execution; the
table uses the same non-error criterion as the preserved 93/200 baseline.
Shorter Agent latency with the candidate reflects fewer useful actions from
degraded transcripts and does not justify selection. Agent evaluation ran on
an idle RTX A5000 GPU 1 because GPU 2 was occupied; baseline measurements
used another RTX A5000, so latency comparisons are indicative.

The original checkpoint and S1 LoRA adapter hashes remained unchanged:

- Original ASR `model.pt`: `7315ea2e15addfe796e673f0b4c9c5200e80371c029524fcc52f96161620eee2`
- S1 adapter weights: `c9114bff577b190a37a3c786cf436849a0c1a1a8ebe2e6d8670c9dd3e16fe791`
- Baseline PUBLIC_TEST submission: `be1478a99fa7c0745590a52e2069ed36bf6d8ea7487ea49523da78ff964da3a7`

## Checkpoint and failure analysis

The cause of the collapse was found after this run. The pretrained
SentencePiece vocabulary is **uppercase** and has no punctuation tokens, but
`train_gipformer_sft.py` fed the mixed-case, punctuated LifeMate transcript
directly to it. In the first 800 training rows, **10,021/20,540 BPE tokens
(48.79%) were `<unk>`**. The development rows had the same problem
(1,170/2,409, 48.57%). Thus decreasing development loss measured learning of
bad targets, not transcription quality. Uppercasing alone still leaves 11.96%
unknown tokens because of punctuation and digits. The training code now
uppercases, removes punctuation and symbols, and skips rows with digits or
unsupported BPE pieces; 3,388/3,756 prepared TRAIN audio rows remain eligible.
The existing `best.pt` is **not repaired** by this code change and must not be
used for submission.

The checkpoint structure itself is valid: `best.pt` has the same keys and
shapes as the original model, with source SHA matching the original. The
decoder, joiner and auxiliary projections are byte-identical; only encoder
weights changed (encoder relative L2 difference 0.0025 at step 1600). Direct
PyTorch greedy inference on `train_0001` produces a plausible sentence with
the original checkpoint, but `N ⁇ ⁇ ⁇` with `best.pt`. This rules out ONNX
export as the primary cause. Training also updated 63,994,103 parameters on
only 800 audio rows, so another attempt should use an explicit ASR quality
gate before considering any checkpoint for the Agent.

The adapted PyTorch checkpoint is
`outputs/asr_sft/gipformer_run_01/best.pt` (source SHA verified, strict
icefall load). Its ONNX export is
`models/gipformer1.5-lifemate-sft-01/`, with manifest and one-audio export
smoke. Training used 800 TRAIN audio recordings, 100 internal development
recordings, up to two epochs at learning rate `1e-5`; the 200 frozen
validation samples and overlapping history were excluded. The final
development loss was 3.6787 at step 1600.

All **762** frozen validation recordings (200 current and 562 historical)
decoded with **zero ASR runtime failures**, but transcripts collapsed to short
repetitions and an exposed special token. For example, the original model's
`train_0015` transcript is a sentence about paying a bill; the adapted model
outputs `C<sos/eos> À À À À`. This is evidence of severe recognition quality
loss; the development loss alone did not predict task performance. The
original PyTorch checkpoint exported through the same icefall path produced
plausible smoke audio, so the adapted weights warrant investigation before
any future training attempt. No retraining was performed in this evaluation.

The adapted Agent run used the same base checkpoint, S1 adapter hash, E1
prompt hash, Hermes parser, temperature 0, seed 42, max 512 output tokens,
tool schemas and ToolEnv as baseline. It completed 200 rows with zero
exceptions and 200/200 replay hash matches. The organizer validator reported
`200 lines checked, 0 schema errors`. The two rejected tool proposals in the
table are logged attempted calls; they are absent from the valid replayed
submission rows. Candidate maximum per-sample latency was 5.44 s, below the
30 s deadline.

The preserved baseline PUBLIC_TEST file was revalidated: **634/634 replay
hashes match** and the organizer validator reports **634 lines, 0 schema
errors**. Its execution log records 11 sample failures, handled by the
replay-valid fallback rows. No new PUBLIC_TEST score was measured locally.

## Artifacts

- ASR cache, complete current and history transcripts, config and errors:
  `outputs/runs/gipformer_sft_val_20261106_cache/`
- R1 ASR metrics and per-sample errors:
  `outputs/reports/gipformer_sft_r1_20261106/`
- R2 Agent predictions, tool traces, runtime logs and metrics:
  `outputs/runs/gipformer_sft_s1_val_20261106/`
- Paired metrics, per-sample transitions and gate result:
  `outputs/reports/gipformer_sft_20261106/`
- Baseline PUBLIC_TEST submission retained for manual upload:
  `result/submission_s1_gipformer_base_20261106.jsonl`

## Reproduction commands

Run from the repository root. Choose fresh output names when repeating an
experiment; the scripts refuse to overwrite existing run directories.

```bash
source /home/shared/miniconda3/etc/profile.d/conda.sh
conda activate datpt_lifemate
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=src

python scripts/cache_gipformer_audio.py --split train --device cuda:1 \
  --sample-ids outputs/splits/validation_ids.json \
  --model-dir models/gipformer1.5-lifemate-sft-01 \
  --output-dir outputs/runs/gipformer_sft_val_repro_cache

PYTHONPATH=src:. python - <<'PY'
import json
from pathlib import Path
from lifemate.data import items
from scripts.compare_gipformer_sft import summarize_asr
ids = json.loads(Path('outputs/splits/validation_ids.json').read_text())
samples = [row for row in items(Path('data/train+public_test'), 'train') if row['id'] in set(ids)]
assert [row['id'] for row in samples] == ids
baseline, _ = summarize_asr(Path('outputs/runs/e1_asr_full_01_A2_cache'), samples)
candidate, per_sample = summarize_asr(Path('outputs/runs/gipformer_sft_val_repro_cache'), samples)
out = Path('outputs/reports/gipformer_sft_r1_repro')
out.mkdir(exist_ok=False)
(out / 'metrics.json').write_text(json.dumps({'baseline': baseline, 'candidate': candidate, 'samples': 200}, ensure_ascii=False, indent=2) + '\n')
with (out / 'per_sample.jsonl').open('w') as f:
    for row in samples:
        f.write(json.dumps({'id': row['id'], **per_sample[row['id']]}, ensure_ascii=False) + '\n')
PY

LIFEMATE_PHYSICAL_GPU=1 bash scripts/serve_llm_lora.sh \
  > outputs/vllm_gipformer_sft_repro.log 2>&1 &
vllm_pid=$!
bash scripts/wait_lora_server.sh "$vllm_pid"

python -m lifemate.cli evaluate --split train \
  --sample-ids outputs/splits/validation_ids.json --device cuda:1 \
  --orchestration E1 --parser hermes --allow-failures \
  --adapter-dir outputs/sft/qwen3_4b_e1_qlora_01/best_adapter \
  --llm-served-model lifemate_sft --asr-backend gipformer \
  --asr-dir models/gipformer1.5-lifemate-sft-01 \
  --transcript-source cached-asr \
  --transcripts-jsonl outputs/runs/gipformer_sft_val_repro_cache/asr_outputs.jsonl \
  --run-id gipformer_sft_s1_val_repro
kill "$vllm_pid"

python scripts/compare_gipformer_sft.py \
  --candidate-cache outputs/runs/gipformer_sft_val_repro_cache \
  --candidate-run outputs/runs/gipformer_sft_s1_val_repro \
  --candidate-model models/gipformer1.5-lifemate-sft-01 \
  --output-dir outputs/reports/gipformer_sft_repro
```

The selection result is `selected_asr_dir` in the comparison JSON. If a
future candidate passes every gate, the PUBLIC_TEST cache and submission
commands are in `docs/gipformer_sft_submission.md`, section 4. For today's
selected baseline, validate the existing submission before manual upload:

```bash
python -m lifemate.cli validate-submission --split public_test \
  --file result/submission_s1_gipformer_base_20261106.jsonl
PYTHONPATH=data/train+public_test/public_test python -m zaic_tools \
  --split public_test --root data/train+public_test \
  --validate result/submission_s1_gipformer_base_20261106.jsonl
```

## Follow-up: corrected-target fine-tune `run_02`

After fixing the BPE targets, a separate fine-tune was run from the original
checkpoint. It stopped at step 1200 with the best development loss at step
800. The best checkpoint was exported to
`models/gipformer1.5-lifemate-sft-02/` and evaluated with the same 200 frozen
TRAIN validation IDs, 762 current/history audio recordings, E1 Agent and S1
adapter. The Qwen adapter and prompt hashes match the baseline. All 200
submission rows passed replay-hash verification.

| Metric | Original + S1 | `run_01` + S1 | `run_02` + S1 |
| --- | ---: | ---: | ---: |
| Current WER | 17.60% | 98.98% | **14.59%** |
| Current CER | 12.51% | 94.73% | **10.60%** |
| Critical slots | 45/64 | 0/64 | **50/64** |
| History WER | 16.62% | 99.86% | **13.79%** |
| Valid state matches | 93/200 | 44/200 | **102/200** |
| Exact tool-call F1 | 0.492 | 0.110 | **0.512** |
| Non-OK samples | 7 | 1 | 11 |
| Runtime exceptions | 0 | 0 | 1 |

Against the original, `run_02` changes 18 incorrect states to correct and 9
correct states to incorrect, a net gain of 9 valid matches. Its 107 raw
state-hash matches include five non-OK tool executions, so the comparable
count is 102. Ten non-OK rows terminate as `tool_error`; one (`train_0959`)
exceeds the 10-LLM-turn limit. The predeclared selection gate rejects
`run_02` because non-OK samples increased from 7 to 11. The original remains
the submission fallback. No `run_02` PUBLIC_TEST submission or official score
was produced.

Detailed three-way report: `outputs/reports/gipformer_sft02_three_way/`.
Paired per-sample transitions: `outputs/reports/gipformer_sft02_vs_base/`
and `outputs/reports/gipformer_sft02_vs_sft01/`.
