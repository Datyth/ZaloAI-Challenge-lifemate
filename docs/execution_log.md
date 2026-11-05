# Execution log, 2026-11-04

Work used `datpt_lifemate` (Python 3.11.18) and physical GPU 2 (NVIDIA RTX A5000). The default sandbox cannot access the NVIDIA driver or the loopback vLLM server; elevated local execution can. Both baseline checkpoints and model runtime dependencies are now present. The vLLM server responded on `127.0.0.1:8000` at the last check.

| Run | Samples | Result |
| --- | ---: | --- |
| `python -m unittest discover -s tests` | 8 | Passed (rerun after documentation update) |
| `baseline_asr_200` | 200 TRAIN | WER 0.3389, CER 0.2324; mean ASR time 1.105 s |
| `smoke_train_first5` | 5 TRAIN | 0 failures, 5/5 replay hashes, 1/5 final-state matches |
| `baseline_train_validation20` | 20 held-out TRAIN | 0 failures, 20/20 replay hashes, 2/20 final-state matches, exact-call F1 surrogate 0.0786; mean 5.035 s, max 10.177 s |

The 20-sample TRAIN ASR WER was 0.3552 and CER was 0.2433. These use TRAIN `audio_text` with lowercase Unicode word normalization. The final-state match and exact-call F1 are local diagnostics, **not** an official competition score. Errors include ASR mistranscriptions, wrong dates and entities, and missing or unrelated tool calls. No PUBLIC_TEST labels were inspected.

The first PUBLIC_TEST full run was started on 2026-11-04 and interrupted at the user's request before completion. It left a diagnostic run at `outputs/runs/baseline_public_first/` and `result/submission.jsonl.baseline_public_first.partial`; `result/submission.jsonl` does not exist. All five partial rows passed local ToolEnv replay validation (5 calls, 5 matching hashes). Their logs show no sample exceptions or failed tool results. No full-run result or official score is available. The user will run the full command in `README.md`.

Before handing over the full command, log review found a runner issue: a failure after the first sample could previously be written as a row and still published in the final file. The runner now retains the diagnostic `.partial` file and raises instead of publishing if **any** sample fails. It also prints progress every 25 samples. Unit tests (8) and Python compilation passed after this change. The organizer kit emits unclosed-file `ResourceWarning`s during unit tests; these come from its own `zaic_tools` file loading and do not fail the tests.

The first validated PUBLIC_TEST submission, remaining four checkpoint downloads, pair legality audit, and model comparison are still pending.

Update, 2026-11-05: a valid 634-row E0 PUBLIC_TEST submission now exists at `result/submission.jsonl`; replay validation found 681 calls and 634 matching hashes. The earlier pending statement above is historical. A fixed 200-sample TRAIN E0–E3 comparison is complete; E1 is best among completed runs with 52/200 state matches, exact-call F1 surrogate 0.2395, and zero failures. E2 matched 52/200 states but had four failed samples. See `docs/orchestration_improvements.md` for the full comparison. A fresh E1 PUBLIC_TEST submission and official score remain pending.

## Full-run failure and targeted recovery

The user-run PUBLIC_TEST inference `20261104T192413_submit_570038` processed all 634 samples. It recorded 632 successful samples and two `RuntimeError: agent exceeded 10 LLM turns` failures (`public_0392`, `public_0421`). The repeated `pad_token_id` line came from ASR generation and was not the failure. The 634-row diagnostic `.partial` file passed ToolEnv replay validation: 634 matching hashes and 679 executed calls. The runner correctly refused to publish `result/submission.jsonl` because the two rows had not finished their agent loops. The subsequent validators reported `FileNotFoundError` only because the user shell continued after the failed submit command.

`public_0392` was still adding grocery items at turn 10; `public_0421` sent a chat message at turn 10 but had no final no-tool response. The [README recovery commands](../README.md) rerun these two IDs with `--max-turns 20`. `scripts/recover_submission.py` accepts replacements only for the two failed IDs, validates the inputs, and replays the merged 634-row file before writing the final output. A local dry run of the merge using copies of the original failed rows passed 634/634 replay; this verifies the merge procedure, not the quality of new predictions. No targeted GPU retry or final submission has been run by the agent.
