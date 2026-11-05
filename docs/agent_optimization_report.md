# LifeMate E4–E6 agent optimization

## Measured TRAIN results

All rows use the same frozen 200-sample TRAIN split. State match and exact-call F1 are local surrogates; no official score is available.

| Experiment | State match | Exact-call F1 | Invalid proposals | Abnormal terminations | Mean / P95 latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 52/200 (26.0%) | 0.240 | 6.9% | 2.5% | 5.31 / 9.98 |
| E4 | 88/200 (44.0%) | 0.401 | 5.8% | 5.0% | 3.07 / 5.80 |
| E6 | 52/200 (26.0%) | 0.239 | 6.0% | 1.0% | 5.31 / 9.75 |
| E6-Oracle | 80/200 (40.0%) | 0.383 | 5.8% | 3.5% | 3.09 / 6.01 |

Oracle runs skip ASR inference, so their end-to-end latency excludes ASR time; compare latency with that limitation.

## E5 failure analysis

E1 missed the target state in 148 samples. Categories below are rule-based diagnostic labels, not manual ground truth.

| Primary category | Samples |
| --- | ---: |
| Intent / tool selection | 65 |
| ASR errors | 45 |
| Tool grounding | 15 |
| Missing information | 11 |
| Argument extraction | 5 |
| Unnecessary actions | 4 |
| Premature termination | 3 |

Each failed sample, full history, transcript pair, gold and predicted calls, decisions, observations, evidence, and suggested fix is in `e5_errors.jsonl`. Multiple categories can apply.

## Oracle transcript effect

E4 changed state matches by +36/200 (+18.0%) relative to E1. 45 previously failing IDs became correct: train_0026, train_0066, train_0075, train_0144, train_0147, train_0236, train_0239, train_0385, train_0476, train_0478, train_0486, train_0501, train_0508, train_0557, train_0596, train_0597, train_0607, train_0618, train_0624, train_0831, train_0836, train_0870, train_0874, train_0901, train_0972, train_1021, train_1068, train_1071, train_1081, train_1209, train_1284, train_1327, train_1336, train_1343, train_1409, train_1457, train_1474, train_1515, train_1560, train_1578, train_1608, train_1610, train_1617, train_1624, train_1652.
These fixed cases support an ASR contribution; other failures can still involve both transcription and reasoning.

## E6 and next submission

E6 changed state matches by +0/200 and exact-call F1 by -0.000. On the predefined conservative selection rule (at least four more correct states, no material F1, invalid-call or termination regression, and P95 below 30 seconds), use **E1** for the next submission.
E6 adds a prompt for completing dependent actions and one bounded read-to-action check. It does not alter model weights or ToolEnv. The E1 validation traces informed this change, so this comparison is exploratory and may overstate generalization.

## Failure examples

- `train_0015`: Intent / tool selection; missing gold tool names: bill_payment.list_bills, banking.transfer. Gold: bill_payment.list_bills, banking.get_balance, banking.list_beneficiaries, banking.transfer; predicted: banking.get_balance, banking.list_beneficiaries, contacts.search.
- `train_0026`: ASR errors; E1 failed and E4 matched the target state with oracle transcripts. Gold: calendar.list_events, calendar.update_event; predicted: calendar.list_events, calendar.create_event.
- `train_0034`: Intent / tool selection; missing gold tool names: chat_ott.list_chats, chat_ott.send_message. Gold: chat_ott.list_chats, chat_ott.send_message; predicted: (none).
- `train_0041`: Intent / tool selection; missing gold tool names: bill_payment.list_bills, contacts.search, contacts.search, ewallet.topup_mobile, ewallet.transfer. Gold: bill_payment.list_bills, contacts.search, ewallet.topup_mobile, contacts.search, ewallet.transfer; predicted: (none).
- `train_0064`: Intent / tool selection; missing gold tool names: bill_payment.pay. Gold: bill_payment.list_bills, ewallet.get_balance, bill_payment.pay; predicted: ewallet.get_balance, bill_payment.list_bills, ewallet.pay_qr.

## Remaining bottlenecks

Review `e5_errors.jsonl` manually before treating the category counts as causal. If the oracle gap dominates E6 gain, prioritize ASR and Vietnamese entity recognition. If oracle transcripts still leave many wrong tool choices, compare legal LLM checkpoints before considering fine-tuning. Fine-tuning would require a separate data and competition-rule audit.
