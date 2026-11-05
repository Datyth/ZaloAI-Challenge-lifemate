# Orchestration improvements (2026-11-05)

## Scope and configuration

The runner now exposes `--orchestration E0|E1|E2|E3`. E0 is the archived behavior. E1 adds structured read-only history. E2 adds native-call enforcement, explicit termination logging, and a grounding check that asks for a missing alarm time instead of inventing one. E3 adds stricter tool name and ID grounding, direct user ID support, duplicate write protection, and a guard against turning on a device when the current utterance explicitly forbids it. **E1 is the CLI default** after the fixed 200-sample TRAIN comparison below. ASR, LLM, decoding, ToolEnv, dataset split, and submission format are unchanged. No database or extra model was added.

Structured history contains chronological turn numbers for previous requests, tool actions and results with a verified flag, and IDs from successful historical tool results. Assistant prose is retained as unverified text. The current request is a separate user message. E3 blocks an exact repeat of a verified historical write unless the current request explicitly asks to repeat it. E3 also accepts literal IDs in user speech and IDs exposed in app-scoped context or successful ToolEnv observations. Official `env.validate` checks every proposed call before execution. A rejected proposal is logged but never passed to `env.call`.

Only OpenAI-style `message.tool_calls` execute. Text containing `<tool_call>` or tool-call narration is logged as a protocol event; E2/E3 make one recovery request before any completed write, then fail if no native call arrives. `finish_reason: length` fails with `token_limit`. A step limit, malformed response, timeout, and tool error have separate trace statuses. A no-call answer is logged as `no_action`, or `unsupported` if the model emits `UNSUPPORTED_REQUEST`. These statuses are internal logs; submission rows still have exactly `id`, `tool_calls`, and `state_hash`.

The local Qwen3-4B checkpoint chat template requests `<tool_call>` XML blocks with JSON name and arguments. The existing server uses `--tool-call-parser hermes`; live GPU runs confirmed that it returns structured OpenAI-compatible calls. Tool selection quality remains limited by ASR errors and model decisions.

## Measured TRAIN comparisons

The archived E0 run and new E2 run used the same first 20 IDs of `outputs/splits/validation_ids.json`, ASR transcripts, Hermes parser, 10 turns, and unchanged checkpoints. The shipped kit has no visible official score implementation, so `official_score` is unavailable.

| Measure | E0, 20 TRAIN | E2, same 20 TRAIN |
| --- | ---: | ---: |
| Final-state match | 2/20 (10%) | 3/20 (15%) |
| Exact-call F1 surrogate | 0.0786 | 0.2325 |
| Empty predictions, 4–11 history entries | 6/8 | 1/8 |
| Empty predictions, 12+ history entries | 2/2 | 0/2 |
| Fake narration samples | 7/20 | 0/20 |
| Abnormal termination samples | 0 | 0 |
| Mean / P95 latency | 5.035 / 8.852 s | 5.127 / 8.322 s |

A separate fixed 12-sample TRAIN set in `scripts/orchestration_small_ids.json` covers four history-length groups. Each mode used the same IDs and ASR path. The E2 column is from the final rerun with the alarm-time check. The E3 column predates that check and the subsequent targeted wording change for blocked actions, so E3 remains exploratory rather than a fully matched final configuration.

| Measure | E0 | E1 | E2 | E3 |
| --- | ---: | ---: | ---: | ---: |
| Final-state match | 2/12 | 3/12 | 4/12 | 3/12 |
| Exact-call F1 surrogate | 0.000 | 0.192 | 0.275 | 0.178 |
| Empty predictions, 12+ history entries | 3/3 | 0/3 | 0/3 | 0/3 |
| Fake narration samples | 4/12 | 0/12 | 0/12 | 0/12 |
| Abnormal termination samples | 0 | 0 | 0 | 0 |

E2 improves final-state match and call accuracy on both small comparisons. The alarm-time check corrected `train_0135`: the model first proposed a fabricated 07:00 alarm; the runner rejected it, and the model then called `sys.ask_missing_info` for `time`. E3 blocked an incorrect `smart_home.fan_set` call for a request saying “do not turn on the fan,” but its measured 12-sample F1 was lower; it remains experimental. A targeted E3 rerun then continued to the requested climate action but chose the wrong mode. No gain is claimed from that targeted adjustment. Residual errors include ASR mistranscriptions, wrong dates, wrong tool choice, and incomplete multi-step requests.

The existing 634-sample PUBLIC_TEST trace is used only for aggregate implementation diagnostics. Empty prediction rates rise from 49/215 (22.8%) with no history to 57/58 (98.3%) with 12+ entries. It has 179 samples with tool-like text but no structured call on that turn, 36 invalid proposed calls, 28 ungrounded-ID proposals, six `length` turns, and two step-limit failures. The 179 detector count includes any sample with tool-like text; the earlier 172 count was restricted to empty predictions, so the two counts have different denominators. PUBLIC_TEST labels were not used for optimization.

The 634-row existing [submission](../result/submission.jsonl) was replay-validated on CPU after these changes: 681 calls, 634 matching state hashes. It is an earlier E0 artifact and was preserved as-is. It is not evidence that E1–E3 improve score.

## CPU verification

`PYTHONPATH=src python -m unittest discover -s tests -q` passes 16 tests. The tests cover read-only history verification, fake narration recovery, native call execution, literal versus fabricated IDs, duplicate writes, missing alarm time, explicit device negation, token-limit failure, existing ToolEnv replay, and all four orchestration modes. `python -m compileall -q src scripts tests` and `bash -n scripts/run_orchestration_experiments.sh` pass. The organizer kit emits unclosed-file warnings during tests.

## GPU 2 smoke test (2026-11-05)

Two E3 TRAIN samples ran on physical GPU 2 with the unchanged ASR checkpoint and local vLLM server. Both completed within 30 seconds with zero runtime failures and matching ToolEnv replay hashes. `train_0008` took 4.78 seconds and made one weather read; final state matched, but the call differed from the gold forecast call (exact-call F1 0). `train_0026` took 6.39 seconds and made a calendar read and write; it created an event instead of moving the existing one (state match 0, exact-call F1 0). These are pipeline smoke tests, not an E0–E3 performance comparison. Detailed LLM turns, ASR outputs, tool events, and terminations are in `outputs/runs/smoke_gpu2_e3_train0008/` and `outputs/runs/smoke_gpu2_e3_train0026/`.

## Full 200-sample TRAIN comparison (2026-11-05)

The four runs used the same fixed TRAIN IDs, ASR and LLM checkpoints, decoding settings, and GPU 2. The report is `outputs/reports/train200_20261105_115719_comparison.json`. Final-state match and exact-call F1 are local surrogates; the organizer's official score is unavailable.

| Measure | E0 | E1 | E2 | E3 |
| --- | ---: | ---: | ---: | ---: |
| Final-state match | 49/200 (24.5%) | **52/200 (26.0%)** | 52/200 (26.0%) | 51/200 (25.5%) |
| Exact-call F1 surrogate | 0.143 | **0.240** | 0.239 | 0.237 |
| Failed samples | 1 | **0** | 4 | 3 |
| Empty predictions, 12+ history entries | 33/34 | **1/34** | 1/34 | 1/34 |
| Fake narration samples | 73 | **6** | 6 | 7 |
| Mean / P95 latency (s) | 5.716 / 11.539 | **5.313 / 9.983** | 5.299 / 9.834 | 5.463 / 10.248 |

E0, E2, and E3 each wrote 200 diagnostic rows but retained `.partial` output because at least one sample failed. E0 exceeded 10 LLM turns on one sample. E2 failed three fake-call narration recoveries and one invalid-call recovery; E3 failed the same three narration recoveries. E1 completed all 200 samples, produced a full run submission, and had the highest exact-call F1. **E1 is the best measured configuration on this fixed 200-sample TRAIN set.** The earlier 12- and 20-sample results favoring E2 were less stable and are superseded for configuration selection by this larger run.

The current 634-row `result/submission.jsonl` is still the earlier E0 PUBLIC_TEST artifact. CPU replay validation on 2026-11-05 found 634 rows, 681 calls, and 634 matching hashes; all 16 unit tests passed. No E1 PUBLIC_TEST submission or official competition score has been produced.

## Reproduce the model comparison

Use the same local vLLM endpoint and ASR checkpoint as the E0 baseline, then run `bash scripts/run_orchestration_experiments.sh cuda:2 <unique-prefix>`. This evaluates E0, E1, E2, and E3 sequentially on the same 200 held-out TRAIN IDs, with separate run directories and detailed `llm_turns.jsonl`, `tool_events.jsonl`, `protocol_events.jsonl`, `terminations.jsonl`, `predictions.jsonl`, and submissions. `scripts/summarize_orchestration.py` writes a comparison JSON. If a mode has failed samples, its partial file and traces remain for diagnosis; the script continues with the next mode and records the mode status in `outputs/reports/<unique-prefix>_run_status.tsv`. The kit provides no official scorer, so state match and exact-call F1 are local surrogates.

The measured reports are `outputs/reports/orchestration_small_final_comparison.json`, `outputs/reports/orchestration_validation20_comparison.json`, and `outputs/reports/train200_20261105_115719_comparison.json`; the public aggregate report is `outputs/reports/orchestration_public_diagnostics.json`. For a fresh PUBLIC_TEST run, use E1, which won the fixed 200-sample TRAIN comparison.
