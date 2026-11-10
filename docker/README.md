# LifeMate solution (Zalo AI Challenge 2026, Track 1)

Release `lifemate-final` (public test **61.26**, 2026-11-09). A spoken Vietnamese request becomes tool calls on the sample's
single `ToolEnv`: Gipformer ASR on CPU, then a Qwen3.5-4B agent with a LoRA adapter, served by a local vLLM.

## Run

```bash
bash /code/predict.sh        # no arguments: reads /data/<split>, writes /result/submission.jsonl
bash /code/start_jupyter.sh  # JupyterLab on port 9777 (password zac2026) for predict_notebook.ipynb
```

Every setting is fixed in `/code/flags.sh`, which sources `/code/configs/release.sh` (the file local evaluation uses too).

1. `predict.sh` starts a vLLM server on 127.0.0.1:8000 (loopback only; the container needs no network) with the base model and
   the LoRA adapter, and prints the model load time.
2. It runs the samples one at a time, in file order, with `python3 -m lifemate.docker_predict`, and prints the mean and
   maximum time per sample.
3. `predict_notebook.ipynb` runs the same code cell by cell (seed, load, read, predict) and writes
   `/result/jupyter_submission.jsonl` with the same rows.

| Output | Content |
| --- | --- |
| `/result/submission.jsonl` | One line per sample: `id`, `tool_calls` (`env.get_tool_calls()`), `state_hash` (`env.get_state_hash()`), `time` in ms measured from before the audio is read. Written and flushed after every sample |
| `/result/logs/trace.jsonl` | Every LLM turn, every proposed and executed call, and how each sample ended |
| `/result/logs/vllm.log` | vLLM server log |

`zaic_tools` is imported from `/data/<split>` (never from a copy in the image), and app data is read and changed only through
`ToolEnv` (one per sample, created once). A sample whose agent loop fails still gets its line: the calls already made on its
ToolEnv and that state's hash.

## Pipeline

1. **ASR.** Gipformer 1.5 (68M RNN-T, g-group-ai-lab), fine-tuned by us on all TRAIN recordings and exported to ONNX.
   - sherpa-onnx on CPU, modified beam search, 8 threads, about 0.07 s per recording (same text as on GPU).
   - The current request and every user turn of the history are transcribed, then lower-cased.
2. **Prompt.** One system message with the rules (when to act, ask back with `sys.ask_missing_info`, or stop), the read-only
   history (earlier requests, the calls already made and their results), the context (time, place, user profile, a weekday
   calendar of the coming days, the Vietnamese name and description of every installed app), the request, and the JSON schemas
   of the installed tools.
3. **Agent loop.** At most 10 turns; each turn the LLM proposes one tool call (non-thinking, greedy). Before it reaches
   `env.call`, the call must pass three generic checks (no app is named in them):
   - it validates against the schema (`env.validate`) and every `x-ref` ID comes from an earlier result, the history or the
     context; otherwise the error goes back to the model and nothing is executed;
   - an identical call is never executed twice;
   - a read tool that came back empty twice is blocked (the model is told to ask back or move on).

   The result of an executed call goes back to the model. The loop ends when the model replies without a call, after 10 turns,
   or at the 27 s deadline of the sample (ASR included).
4. **Output.** The line is exactly `env.get_tool_calls()` and `env.get_state_hash()`.

Optional, off in this release: a step-level vote of the three adapters in `/code/models/` (`ENSEMBLE` in
`/code/configs/release.sh`). Each turn all three propose a call and only the majority call runs, on the same single ToolEnv.

## Models

| Model | Source | Released | Parameters |
| --- | --- | --- | ---: |
| Base LLM, text layers (`--language-model-only`) | Qwen/Qwen3.5-4B @ 851bf6e8, Apache-2.0 | 2026-02 | 4,205,751,296 |
| LoRA adapter `llm-lora-p1` (served) | ours: r16, alpha 32 on attention and MLP of the text layers | 2026-11 | 32,464,896 |
| LoRA adapters `llm-lora-e1`, `llm-lora-synth` (only used by the optional vote) | ours, same shape | 2026-11 | 2 x 32,464,896 |
| ASR `asr-gipformer` | g-group-ai-lab/gipformer1.5-68M-rnnt, fine-tuned by us | 2026-08 | 69,650,000 |
| **Total loaded** | | | **4.31B (LLM 4.24B < 5B; with all three adapters 4.37B < 6B)** |

LLM context: 16,384 tokens (`--max-model-len 16384`). The image holds the base LLM because the LoRA adapters run on top of it at
inference; the vision tower in the checkpoint is not loaded.

## Data and training

- **Real data.** Only the organizer's TRAIN split (1,545 samples). The public and private test sets were never labelled or
  used for training, and no rule or prompt names an app that only appears in the test sets.
- **ASR.** All 5,036 TRAIN recordings (current requests and history turns). Number labels are written the way they are spoken
  ("2.500.000" as "hai triệu rưỡi", two and a half million); speed, noise and SpecAugment augmentation; 3 epochs, lr 2e-5.
- **LLM (`llm-lora-p1`).** Every TRAIN sample becomes a chat trajectory: its gold calls are replayed in `ToolEnv` and must give
  the labelled results. Each sample appears twice, with the gold transcript and with the original Gipformer's transcript (real
  recognition errors on audio that ASR never trained on). 92 "recovery" trajectories add a failed search before the right one;
  the failed call carries no loss, so the model learns the recovery, not the mistake. Loss only on the model's own turns (calls,
  IDs, the decision to stop); free-text message fields (scored loosely by the metric) have weight 0.3. LoRA r16, 2 epochs,
  lr 1e-4, max length 16,384, seed 42.
- **Synthetic data (`llm-lora-synth` only).** Apps written by us for training (`training_code/synth/`), with tasks whose gold
  calls are replayed with the official engine. None copies a TRAIN or test app. The served adapter does not use them.

`/code/training_code/` is the repository code: its `README.md` gives the commands that rebuild the data
(`scripts/data/build_final_data.sh`) and retrain every checkpoint (`scripts/train/train_final.sh`), and the URLs of the base
models and of the released checkpoints (Hugging Face `hugebenevolence/lifemate-final`, frozen after the Docker deadline).

## Reproducibility and time

- Greedy decoding (temperature 0, vLLM seed 42), one sample at a time; seed 42 in the notebook, training and data scripts.
- Prefix caching stays on: two single-stream runs of 50 validation samples differed on 0 rows with it and 2 without.
- Time per sample with this exact code path on the 634 public-test samples, one RTX PRO 4000 Blackwell (24 GB), single stream:

| 634 public samples, one stream, one RTX PRO 4000 Blackwell (24 GB) | Value |
| --- | ---: |
| Model loading (vLLM start to ready) | about 45 s |
| Mean time per sample, ASR included | 2.88 s |
| Slowest sample | 8.71 s |
| Samples over the 30 s limit | 0 |
| Same final state as the submitted 61.26 file | 618 / 634 |

Greedy decoding is not bit-exact across runs in vLLM: batching (the 61.26 file used four parallel streams) and prefix-cache state
change a few low-margin choices. Two single-stream runs of the same 100 public samples agreed on 97; in the 3 others the model got
the identical prompt and picked a different amount at one turn. The Track 1 rules average five organizer runs in that case.
