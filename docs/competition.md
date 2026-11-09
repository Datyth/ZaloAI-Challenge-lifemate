# LifeMate competition rules (Zalo AI Challenge 2026, track 1)

Sources: the portal tabs Description, Evaluation (both updated 2026-10-25), Rule (2026-11-03) and Submission Guidance
(2026-11-05), read through `https://api-challenge.zalo.ai/fe/api/config-detail?name=<tab>&title=1`. The Docker guide
`Docker_Guide_ZAC2026_LifeMate.pdf` and plain-text copies were saved in `docs/portal_2026-11-07/` (git-ignored). The organizer's
kit is in `data/train+public_test` (v3.2.0, 2026-10-24). Where the kit and the text differ, the kit wins: for example,
schema-rejected `env.call` attempts are recorded.

## Data

- TRAIN has 1,545 samples and PUBLIC_TEST has 634.
- Both splits have `id`, `audio`, `context` (time, timezone, location, profile, installed apps and tools) and an optional
  `history` (user audio, prior assistant text, tool calls and results). Audio paths are split-relative.
- Only TRAIN has `audio_text`, `label.tool_calls`, `label.tool_results` and `label.final_state`.
- The tool catalog uses JSON Schema plus the extensions `x-one-of`, `x-ref` and `x-match`.
- The splits use different app sets: 25 apps are shared, 11 exist only in public and 21 only in train. In 65% of public samples
  at least one installed app is absent from train, so the private test will most likely contain unseen apps too.

## ToolEnv

- At inference only `validate`, `call`, `get_tool_calls` and `get_state_hash` may be used. `get_state`, `states`, Executor,
  Simulator, cloning the env and reading files under `<split>_environment/` are not allowed.
- There is exactly one ToolEnv per sample, and every call made on it, schema-rejected ones included, must appear in the
  submission. `lifemate.agent` therefore validates each proposed call first and never sends a rejected proposal to `env.call`.
- History actions are already applied to the initial state and must not be replayed.

## Scoring

- **Per sample.**
  - If the gold call list is empty, the score is 1 only when the submitted list is also empty.
  - Otherwise the score is 0 when the submitted `state_hash` differs from the gold hash or no call matches (M = 0).
  - Otherwise the score is F1, with P = M / (M + U) and R = M / |G|. U counts unmatched submitted calls, schema-rejected ones
    included.
- **Final score.** 100 / N · Σ score.
- **Call matching.**
  - A match needs the same tool name, identical non-fuzzy arguments, and fuzzy arguments with token-F1 > 0.6 after lower-casing
    and NFC.
  - Matching is one-to-one and ignores order. `sys.ask_missing_info` is scored like any other tool.
  - Fuzzy fields do not affect the hash.
- **Replay.** The organizers replay the submitted calls in order with zaic_tools, and their hash is final.
- **Local scorer.** `analysis/score.py` implements these rules. It has fuzzy threshold ≥ 0.5 by default; the strictest reading
  (> 0.6, exact NFC equality) lowers val by at most 0.5 points.

## Limits

- Under 6B parameters in total, the LLM under 5B, an LLM context of at most 16,384 tokens, and at most 30 s per sample. A timeout
  or failure scores 0.
- Fully offline. Models and data must be released before 2026-10-26.
- Synthetic data from external services is allowed during development only. Labelling public or private test is forbidden.
- **Reproducibility.** Seeds must be fixed and decoding deterministic. If a re-run differs, the score is the mean of 5 Docker
  runs. Ties are broken by tool-call F1 without the hash gate, then by lower mean time per sample.
- **Hardware.** The Rule tab says 1 × A30 24 GB, a Xeon Gold 6442Y and 64 GB RAM. The Docker guide says 1 × RTX 5090 32 GB with
  driver 595.58.03 and a base image with CUDA ≤ 13.2. Plan for the slower A30.

## Schedule

- **Public round.** Oct 26 – Nov 16, 5 submissions per day.
- **Docker.** The image, training code and data are due Nov 16 at 23:59 GMT+7. At most 2 uploads are allowed, and the last one
  counts.
- **Private test.** Released Nov 20 at 09:00; results due Nov 23 at 23:59. The top 2 are announced Dec 2.

## Docker contract

- `/code/predict.sh` runs with no arguments, reads `/data/<split>` and writes `/result/submission.jsonl`.
- Each line holds `id`, `tool_calls` and `state_hash`, plus `time` in ms measured from before the audio is read.
- zaic_tools must be imported from `/data/<split>`, not from a copy inside the image.
- `predict_notebook.ipynb` writes `/result/jupyter_submission.jsonl`, and jupyterlab must be installed. `start_jupyter.sh` is
  also required.
- Ship `training_code/` with a README. Base models used only for training stay out of the image.
- Our implementation is in `docker/` (see `docker/README.md`).
