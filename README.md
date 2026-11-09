# LifeMate (Zalo AI Challenge 2026, track 1)

Speech request → tool calls on the organizer's `ToolEnv`. The released pipeline (public **58.02**, 2026-11-08):

1. **ASR.** Gipformer 1.5 (68M RNN-T), fine-tuned on the TRAIN audio ("ASR v2"). It runs on CPU with sherpa-onnx (modified beam
   search, beam 4), and its output is lower-cased.
2. **LLM.** Qwen3.5-4B (text layers only, non-thinking) with the QLoRA adapter `q35_e1`, served by a local vLLM 0.32.0.
3. **Agent.** The E1 loop (`src/lifemate/agent.py`) runs on the sample's single ToolEnv.
   - The model sees structured read-only history.
   - Every proposed call is schema-validated before it is executed.
   - An identical call is never executed twice (`--dedup-calls`).
   - The loop stops after at most 10 LLM turns.

Validation on 200 frozen TRAIN ids: **70.91** with ASR, **82.33** with gold transcripts. All model and agent settings are in
[`configs/release.sh`](configs/release.sh), which the local server, the local runs and the Docker image all source.

The docs are:
- [docs/competition.md](docs/competition.md): rules, scoring, limits and the Docker contract;
- [docs/experiments.md](docs/experiments.md): results and decisions;
- [docs/error_analysis.md](docs/error_analysis.md): error analysis of the release, with the figure in `docs/figures/pipeline.png`;
- [docker/README.md](docker/README.md): the submission image.

## Layout

| Path | Content |
| --- | --- |
| `src/lifemate/` | Inference package: `cli` (evaluate / submit / validate), `docker_predict` (image entry), `agent`, `asr`, `llm`, `toolenv`, `submission` |
| `configs/` | `release.sh` (release settings), `splits/validation_ids.json` (frozen 200 TRAIN ids) |
| `scripts/setup/` | Download base models and the release checkpoints; build the ASR training env |
| `scripts/data/` | Validation split, agent SFT data, Gipformer SFT data, synthetic-app split |
| `scripts/train/` | Agent QLoRA SFT, Gipformer fine-tuning and ONNX export |
| `scripts/infer/` | vLLM server, sharded evaluation, ASR transcript cache, submission merge |
| `analysis/` | Local scorer (`score.py`) and error analyses |
| `synth/` | Synthetic apps and tasks (tried, not in the release) |
| `docker/` | Submission image |
| `tests/` | Unit tests (`python -m pytest`; the tests that need the kit skip without `data/`) |

`data/`, `models/`, `outputs/` and `result/` are git-ignored. The organizer kit goes to `data/train+public_test/`.

## Setup

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e '.[infer,test]'
bash scripts/setup/get_models.sh      # Qwen3.5-4B and the original Gipformer at pinned revisions -> models/
bash scripts/setup/get_checkpoint.sh  # release LoRA + ASR v2 (private HF repo, `hf auth login`) -> outputs/release/lifemate-q35e1/
.venv/bin/python scripts/data/make_validation_split.py   # writes configs/splits/validation_ids.json, or checks it if present
```

The training environments are separate:
- `.venv-train` for agent SFT: `uv pip install -e '.[sft]'`;
- `.venv-asr` for Gipformer fine-tuning with torch 2.8, k2, kaldifeat and icefall: `bash scripts/setup/setup_asr_env.sh`.

## Evaluate

```bash
bash scripts/infer/serve.sh            # terminal 1: vLLM on 127.0.0.1:8000, base + release LoRA served as "lifemate"
# terminal 2: ASR transcripts of the validation samples (CPU), then 4 parallel shards of the agent, then the score
.venv/bin/python scripts/infer/asr_cache.py --split train --ids configs/splits/validation_ids.json \
  --model-dir models/gipformer-lm-v2-01 --output-dir outputs/asr/gipformer-lm-v2-01_val
bash scripts/infer/run_shards.sh v200 configs/splits/validation_ids.json 4 asr sft
bash scripts/infer/run_shards.sh v200_oracle configs/splits/validation_ids.json 4 oracle sft   # gold transcripts
.venv/bin/python analysis/score.py outputs/runs/v200 outputs/runs/v200_oracle
.venv/bin/python analysis/errors.py --asr v200 --oracle v200_oracle    # error groups and per-case files
```

`run_shards.sh` exits non-zero if any shard fails. You can override its defaults through the environment:
- `ADAPTER` / `SERVED` for another LoRA;
- `CACHE` for another transcript file;
- `DEADLINE` (default 900 s, because the shards share one GPU);
- `EXTRA` for more `lifemate.cli evaluate` flags.

To compare adapters, serve them side by side with `bash scripts/infer/serve.sh name=dir ...`.

## Submit (public test)

```bash
.venv/bin/python scripts/infer/asr_cache.py --split public_test --model-dir models/gipformer-lm-v2-01 \
  --output-dir outputs/asr/gipformer-lm-v2-01_public
SPLIT=public_test bash scripts/infer/run_shards.sh pub all 4 asr sft
.venv/bin/python scripts/infer/merge_submission.py pub --split public_test --output result/submission.jsonl
PYTHONPATH=data/train+public_test/public_test .venv/bin/python -m zaic_tools --split public_test \
  --root data/train+public_test --validate result/submission.jsonl
```

`merge_submission.py` refuses to write in these cases:
- a shard is unfinished;
- an id is missing or duplicated;
- a sample failed at runtime (`runtime_errors.jsonl`), unless you pass `--allow-failures`;
- the merged file does not replay through ToolEnv to the same hashes.

It also never overwrites an existing file.

The Docker path (`lifemate.docker_predict`) decodes the audio inline and gave rows identical to the cached run (12/12). It needs a
folder that holds only one split:

```bash
mkdir -p /tmp/pub && ln -sfn "$PWD/data/train+public_test/public_test" /tmp/pub/public_test
source configs/release.sh && PYTHONPATH=src .venv/bin/python -m lifemate.docker_predict --data /tmp/pub --out /tmp/pub_out \
  --asr-dir models/gipformer-lm-v2-01
```

To build and run the image, see [docker/README.md](docker/README.md).

## Train

The 200 validation ids, and every sample that shares a history recording with them, are excluded from all training data. Each
data script records the sha256 of `configs/splits/validation_ids.json` in its manifest.

**Agent adapter (`q35_e1`):**

```bash
.venv/bin/python scripts/infer/asr_cache.py --split train --output-dir outputs/asr/gipformer_train   # original Gipformer, all TRAIN audio
.venv/bin/python scripts/data/prepare_agent_sft.py --orchestration E1 --variants gold,asr --output-dir outputs/sft/data_v2_e1
.venv-train/bin/python scripts/train/train_agent_sft.py --data-dir outputs/sft/data_v2_e1 --out outputs/sft/q35_e1_01 \
  --model models/qwen3.5-4b --epochs 2 --lr 1e-4    # QLoRA r16, max_len 16384; about 3 h on one 24 GB GPU
```

**ASR v2 (`gipformer-lm-v2-01`):**

```bash
.venv/bin/python scripts/data/prepare_gipformer_sft.py --output-dir outputs/asr_sft/gipformer_data
.venv-asr/bin/python scripts/train/train_gipformer.py --data-dir outputs/asr_sft/gipformer_data \
  --output-dir outputs/asr_sft/gipformer_v2 --physical-gpu 0 --train-all   # lr 2e-5, accum 4, 2 epochs (defaults)
bash scripts/train/export_gipformer.sh outputs/asr_sft/gipformer_v2 gipformer-lm-v2-01   # ONNX + WER on the validation audio
```

**Synthetic apps (not in the release):**
- `scripts/data/build_synth_split.py` builds `outputs/synth`;
- then run `prepare_agent_sft.py --split synth --data-root outputs/synth`.
