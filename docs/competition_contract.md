# LifeMate competition contract, checked 2026-11-04

Sources: [official LifeMate portal](https://challenge.zalo.ai/portal/lifemate), [competition homepage](https://challenge.zalo.ai/), and the organizer's local `data/train+public_test/README.md`, `VERSION`, split catalogs, `zaic_tools/{__init__,executor,validate,utils}.py`, and sample submission. The portal's interactive Evaluation, Data, Rule, and Submission Guidance tabs could not be read through the available web view, so their current contents remain unverified.

The local kit is v3.2.0, built 2026-10-24, with 1,545 TRAIN and 634 PUBLIC_TEST records. TRAIN contains `audio_text`, `label.tool_calls`, `label.tool_results`, and `label.final_state`; PUBLIC_TEST contains none of those. Both expose `id`, `audio`, `context` (time, timezone, location, profile, installed apps/tools), and optional `history` with user audio and prior assistant text/tool calls/results. The current audio path is split-relative.

`ToolEnv(split, id, root)` loads one sample internally. The agent may use `validate`, `call`, `get_tool_calls`, and `get_state_hash`; it must never inspect files under a split environment directory. All `call` attempts, including schema-rejected attempts, appear in `get_tool_calls`. Proposed calls rejected by our prevalidation do not. History actions are already reflected in the initial environment and must not be replayed. Each submission row has exactly `id`, ordered `tool_calls`, and the state hash obtained after those calls. The installed catalog supplies JSON Schemas and extensions including `x-one-of`, `x-ref`, and `x-match`.

The shipped kit has a CLI submission validator but no standalone official score implementation in the visible non-environment files. Thus official score and final-state match cannot be reported yet. The kit is authoritative where its behavior differs from the task prompt, particularly recording schema-rejected `env.call` attempts.

The task prompt reports SLM limits of under 6B total parameters, under 5B LLM parameters, 16,384 tokens, and 30 seconds per sample on one A30, offline inference, and a pre-2026-10-26 release cutoff. These need confirmation from the live Rule tab before contest submission. No checkpoint has been locally audited for exact parameter count or release eligibility yet.

## Update 2026-11-07: portal tabs verified

The Description, Evaluation, Rule and Submission Guidance tabs were read through the portal's own API (`https://api-challenge.zalo.ai/fe/api/config-detail?name=<tab>&title=1`; the Data tab needs a login). Plain-text copies, the formula image and the Docker guide are in `docs/portal_2026-11-07/`. Last updates: Description and Evaluation 2026-10-25, Rule 2026-11-03, Submission Guidance 2026-11-05.

- **Score per sample.** If the gold list is empty: 1 only when the submitted list is also empty, else 0. Otherwise 0 when the submitted `state_hash` differs from the gold hash or no call matches (M = 0), else F1 with P = M / (M + U) and R = M / |G|. U counts unmatched submitted calls, schema-rejected calls included. Final score = 100 / N · Σ score.
- **Call matching.** Same tool name, identical non-fuzzy arguments, fuzzy arguments with token-F1 > 0.6 after lowercasing and NFC. Matching is one-to-one and order is ignored. `sys.ask_missing_info` is scored like any other tool. Fuzzy fields do not affect the hash. Organisers replay the submitted calls in order with zaic_tools, and their hash is final.
- **Limits.** Under 6B parameters in total, LLM under 5B, LLM context ≤ 16,384 tokens, ≤ 30 s per sample (a timeout or failure scores 0), fully offline. Models and data must be released before 2026-10-26. Synthetic data from other services is allowed only during development. Labelling public or private test is forbidden.
- **ToolEnv.** At inference only `validate`, `call`, `get_tool_calls` and `get_state_hash` are allowed. `get_state`, `states`, Executor, Simulator and cloning the env are not. Exactly one ToolEnv per sample, and every call made on it must appear in the submission.
- **Reproducibility.** Fixed seeds and deterministic decoding. If a re-run differs, the score is the mean of 5 Docker runs. Ties are broken by tool-call F1 without the hash gate, then by lower mean time per sample.
- **Hardware conflict.**
  - The Rule tab says 1 × A30, Xeon Gold 6442Y (16 cores) and 64 GB RAM.
  - The Docker guide says 1 × RTX 5090 32 GB, Xeon Gold 6459C, 64 GB RAM, driver 595.58.03 and a CUDA 13.2 base image (≤ 13.2).
  - Plan for the slower A30.
- **Schedule.**
  - Public round: Oct 26 – Nov 16, 5 submissions per day.
  - Docker plus training code and data due Nov 16 at 23:59 GMT+7: at most 2 uploads, and the last one counts.
  - Private test: released Nov 20 at 09:00, results due Nov 23 at 23:59.
  - Top 2 announced Dec 2.
- **Docker.**
  - `predict.sh` reads `/data/<split>` and writes `/result/submission.jsonl`. Each line also carries `time` in milliseconds, measured from before the audio is read.
  - zaic_tools must be imported from `/data/<split>`, not from a copy inside the image.
  - `predict_notebook.ipynb` writes `/result/jupyter_submission.jsonl`, and jupyterlab must be installed.
  - Ship `training_code/` with a README.
  - Base models used only for training stay out of the image.
