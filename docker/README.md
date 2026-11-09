# LifeMate solution (Zalo AI Challenge 2026, Track 1)

> Release `lifemate-q35e1` (public 58.02, 2026-11-08). The numbers marked TBD still need a measurement.

## Run

```bash
bash /code/predict.sh        # no arguments: reads /data/<split>, writes /result/submission.jsonl
```

All settings are fixed in `/code/flags.sh`, which sources `/code/configs/release.sh`; local evaluation uses the same file.

- `predict.sh` starts a local vLLM server (loopback only) with the base model and the LoRA adapter.
- It then runs the samples one at a time with `python3 -m lifemate.docker_predict`.
- `predict_notebook.ipynb` runs the same code cell by cell (seed, load, read, predict) and writes `/result/jupyter_submission.jsonl`.
- `start_jupyter.sh` opens JupyterLab on port 9777 (password `zac2026`).

| Output | Content |
| --- | --- |
| `/result/submission.jsonl` | One line per sample: `id`, `tool_calls` (`env.get_tool_calls()`), `state_hash` (`env.get_state_hash()`), `time` in ms from before the audio is read. Flushed after every sample |
| `/result/logs/trace.jsonl` | Every LLM turn, proposed and executed call, and termination of every sample |
| `/result/logs/vllm.log` | vLLM server log (model load time) |

## Pipeline

1. **ASR.**
   - Model: Gipformer 1.5 (68M RNN-T, g-group-ai-lab) fine-tuned on the TRAIN audio, exported to ONNX.
   - Runs on CPU with sherpa-onnx: modified beam search, 8 threads, about 0.07 s per file. The output is identical to the GPU run.
   - The current request and every user turn of the history are transcribed, then lower-cased.
2. **Agent.**
   - The LLM sees one system prompt with the rules and the read-only history (previous requests, prior verified tool calls and
     results, known entities), the request with the context, and the JSON schemas of the installed tools.
   - It calls tools one at a time through the sample's single `ToolEnv`. Every result goes back to the model.
   - The loop stops on a reply without tool calls, after 10 turns, or at the per-sample deadline (27 s).
   - Guards, all generic (no app is named):
     - a call that fails the schema is shown to the model as an error and not executed;
     - an identical call is never executed twice;
     - `x-ref` IDs must come from an earlier result or the history.
3. **Output.** The submission line is exactly `env.get_tool_calls()` and `env.get_state_hash()`.

## Models

| Model | Source | Release | Parameters |
| --- | --- | --- | ---: |
| Base LLM, text layers only (`--language-model-only`) | Qwen/Qwen3.5-4B @ 851bf6e8, Apache-2.0 | 2026-02-27 | TBD (4.66B with the vision tower) |
| LoRA adapter `q35_e1` (ours) | QLoRA r16 on TRAIN trajectories (gold and ASR transcripts), see `training_code/` | 2026-11 | TBD |
| ASR | g-group-ai-lab/gipformer1.5-68M-rnnt, fine-tuned by us | 2026-08-21 | 69,650,000 |
| **Total** | | | **TBD (< 6B; LLM < 5B; context 16,384)** |

## Data

- **Real data.** Only the TRAIN split. Public and private test were never labelled or used for training.
- **Synthetic data.** The released adapter does not use it: an adapter trained with it scored 56.69 on public, against 58.02.
  The generator ships in `training_code/synth/`, as the rules allow during development:
  - new apps written for training only, as zaic_tools-style modules with catalog schemas (`training_code/synth/`);
  - tasks whose gold calls are replayed with the official `Session`;
  - distractor apps and user profiles from TRAIN samples, read through `ToolEnv.get_state()`.
  - None of the synthetic apps copies a TRAIN or PUBLIC app.

## Training code

`/code/training_code/` is a copy of the repository code: `README.md` (setup, train, eval), `pyproject.toml`, `configs/`,
`scripts/{setup,data,train,infer}/`, `synth/` and `src/`. The data is the organizer's TRAIN split. The frozen validation ids
are in `configs/splits/validation_ids.json`.

## Reproducibility and time

- **Decoding and seeds.** Greedy decoding (temperature 0, seed 42), one sample at a time. Two single-stream runs of 50 val samples gave identical rows with prefix caching on (0/50 differ) and 2/50 differing with it off, so it stays on.
- **Measured time.** Single-stream mean about 3 s and maximum about 10 s per sample on an RTX PRO 4000 Blackwell. A30: TBD.
