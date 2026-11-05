# Codex Task: Implement, Benchmark, and Generate the First Zalo LifeMate 2026 Submission

You are the coding agent working **inside the LifeMate project repository**, with access to its official TRAIN / PUBLIC_TEST development kit. Your job is to **actually implement, test, run, and document** a reproducible offline baseline, generate the first valid public-test submission file, log ASR/LLM/tool behavior for debugging, and then download the remaining checkpoints and run a compact model comparison. Do not stop at a plan or write-only skeletons.

## A. Competition brief, official sources, and goal (READ FIRST)

**Competition:** Zalo AI Challenge 2026, **LifeMate: Agentic AI That Turns Your Voice Into Action**. It is **not** the separate DataMate task, a general speech-to-text benchmark, or a chatbot quality benchmark.

**Official links (source of truth):**
- **LifeMate challenge portal:** https://challenge.zalo.ai/portal/lifemate
  - Read all five tabs on this page: **Description**, **Evaluation**, **Data**, **Rule**, and **Submission Guidance**. Tabs may require interaction or authentication; do not assume URL fragments or direct download URLs.
- **Zalo AI Challenge homepage and timeline:** https://challenge.zalo.ai/
- **Official development kit, TRAIN and PUBLIC_TEST, `zaic_tools`, scorer and exact runtime interfaces:** access through the **Data** tab above and the **organizer-provided files mounted in this repository/environment**. There is no verified standalone URL for the kit in this prompt. Inspect the actual local kit before coding.

### A.1 What the system is required to do

A user speaks **Vietnamese**, possibly with common English words. The recorded request may be colloquial, regional-accented, noisy, slurred, extremely short (1-2 seconds), or cut off mid-utterance. A speaker may contribute multiple **independent** requests; never stitch them together unless the official conversation structure explicitly establishes continuity. An audio-transcription model must not invent a missing sentence ending.

For each sample, the agent receives **only the inputs permitted by the organizer**, conceptually:
1. Latest user request as an **audio recording**.
2. **User context** such as time, timezone, location, and profile (home/work addresses where present).
3. **Installed/connected apps and their callable tool JSON Schemas**, which vary per sample. Example domains include smart home, banking, ride-hailing, food delivery, and email.
4. **Optional conversation history**, which may resolve references to earlier requests.
5. A simulated **application environment** with devices, accounts, reminders, contacts, etc. The agent may inspect or modify that environment **only by invoking official tools through `zaic_tools` / ToolEnv**.

**Success criterion:** Turn the user's spoken request into the **correct sequence of tool actions and intended final application state**. The submission is scored on **tool execution/results, not on an eloquent textual assistant response**. Some requests require several dependent tools (look up state/IDs, observe results, then decide the next action).

Three behaviors are essential:
- **Supported/complete request:** execute necessary tool calls in the correct execution order.
- **Supported but genuinely missing required information:** call `sys.ask_missing_info` with the organizer-required arguments, potentially after earlier valid calls. Try permitted context/read tools first rather than fabricating values.
- **Unsupported by the installed apps:** return an **empty tool-call list** `[]`. Do not hallucinate a capability or invoke unrelated tools.

### A.2 Submission format and state semantics

Output **exactly one JSON object per sample line** in `submission.jsonl`. Its keys are `id`, `tool_calls` (ordered list of `{name, arguments}` objects) and `state_hash`. The following is a **shape illustration only**, not a valid sample or a hard-coded action:

```json
{"id":"<actual_sample_id>","tool_calls":[],"state_hash":"<actual_hash_from_ToolEnv>"}
```

For a real sample, `state_hash` **must** be returned by the organizer's `env.get_state_hash()` after executing its actual sequence of actions in a **fresh ToolEnv for that sample**. Never synthesize a hash, edit environment files, or re-run historical actions already reflected in the environment. Record **every action actually executed** in the output `tool_calls`; persist proposed-but-rejected actions in diagnostic logs, not as successful executions.

### A.3 How LifeMate scoring works (verify against the distributed scorer)

- For ordinary samples with nonempty gold calls, the **final state hash is a gate**: if the predicted final hash differs from the gold final hash, the sample receives **zero**, even if some tool names seem correct.
- When the hash matches, the score uses **tool-call precision/recall/F1** under the organizer's matching rules. Names and non-fuzzy arguments must match; fuzzy fields use the scorer's token-F1 threshold; matching is one-to-one. **Execution order matters for side effects**, although the scorer's gold/predicted call matching may ignore ordering.
- For out-of-scope samples whose gold tool-call list is empty, **an empty predicted tool-call list** is necessary for a correct score.
- The **organizer's actual evaluator is authoritative**. Run it whenever distributed; do not substitute a homemade approximate score and call it official.
- Report tool-call F1, final-state match, argument validity, inference latency and ASR WER/CER **only where valid references exist**. In particular, do not assume every TRAIN item includes a gold audio transcript.

### A.4 Confirmed SLM-track operational restrictions (re-check live rules)

As recorded from the official LifeMate Rule/Submission Guidance screenshots on **2026-10-27**, the track-specific restrictions are:
- Combined model parameters across **ASR + LLM + any other loaded ML models: strictly < 6B**.
- Language-model parameters: **strictly < 5B**. Count the actual complete architectures, not advertised model-name shorthand; quantization reduces memory, **not** parameter count.
- Maximum LLM context: **16,384 tokens**.
- Processing budget: **< 30 seconds per sample** on the organizer's **single NVIDIA A30** (plus 16-core Xeon Gold 6442Y and 64GB system RAM).
- Submitted inference **must be offline**, with no external network API calls. Network may be used for preparation/checkpoint download before packaging.
- Public data/open models must meet the organizer's stated **pre-2026-10-26 release cutoff** and license requirements; development-time teacher/synthetic-data practices must comply with the official Rule tab.
- **Do not open or parse anything under `<split>_environment/`**. Interact with hidden app state exclusively through official ToolEnv methods and allowed tool calls.
- Do **not** label PUBLIC_TEST/PRIVATE_TEST (manually or via external systems) to tune/train models. Use **TRAIN** for development, sample selection and validation.

**Avoid a common mistake:** the competition-wide leaderboard labels `SLM`/`LLM`/`APIs` are *comparison categories*; their general model-size definitions do **not** override LifeMate's stricter **<6B total / <5B LLM** track-specific submission requirements. A public APIs/Others score does not authorize external API inference in the final offline solution.

### A.5 Schedule, source priority, and verification discipline

The organizer's main schedule currently lists: TRAIN/PUBLIC_TEST and kit released **2026-10-26**; public solution deadline **2026-11-16 23:59 GMT+7**; private test released **2026-11-20**; private result deadline **2026-11-23 23:59 GMT+7**. Dates may change; consult the live portal.

**Mandatory source precedence:** (1) newest official organizer Rule/Evaluation/Data/Submission Guidance and exact shipped code, (2) official portal overview, (3) this prompt. If there is a discrepancy, obey the organizer and document what changed. **Never guess file names, exact input keys, ToolEnv method signatures, model eligibility or download locations.** Before implementation, write a concise `docs/competition_contract.md` citing the two official portal links, the local kit version/files examined, verified requirements, unresolved points, and any conflicting instructions. Do not treat inaccessible website tabs as verified simply because a summary appears here.

## B. Operating rules

- Work in the current repository. Inspect existing code and preserve working implementations. Keep the project small and readable, suitable for a research competition, not a production microservice.
- Use Python 3.11, typed Python, structured logging, clear exceptions, pinned dependencies, fixed random seeds, and small tests. Prefer reusing the official SDK and existing libraries over reimplementing agent orchestration.
- Execute safe local setup, downloads, smoke tests, and benchmarks autonomously when access allows. Do not silently claim execution when commands fail. Record exact commands and real outcomes. Do not ask repeated confirmation for routine actions.
- Do not assume dataset keys, paths, ToolEnv constructor/methods, evaluator API, or audio formats. **Inspect the supplied official kit / README / sample schemas first** and adapt to actual interfaces. Never fabricate compatibility.
- **Critical competition rules:** entire submitted inference runs offline; no external API; total loaded model parameters <6B; LLM parameters <5B; LLM context <=16,384; <30 seconds/sample on one NVIDIA A30; checkpoints released before 2026-10-26 with appropriate licenses. Count all weights actually loaded (including ASR encoders, vision modules, etc.); quantization does not reduce parameter count. Verify rules against the latest official kit and obey it if updated.
- **Never open, parse, copy, or inspect files inside `<split>_environment/` (including `state.json`).** Access app state only through official `zaic_tools` / `ToolEnv` methods and permitted tool calls. One new ToolEnv instance per sample. Never replay historical actions already reflected in initial state. Record every actually executed call.
- Use only official TRAIN annotations for tuning/evaluation. Never manually label or externally label PUBLIC_TEST/PRIVATE_TEST for optimization. No gold answers, final state hashes, or hidden environment state inside prompts. No pseudo-labeling public test for training.
- Download model checkpoints and dependencies **during preparation only**, into project-local resources. Inference must work with network disabled. Never commit model weights, datasets, environment files, PII traces, or caches to Git.

## 1. Inspect official kit and hardware before coding

1. Print a concise inventory of the repo, GPU (`nvidia-smi`), CUDA/PyTorch, Python, available disk space, dataset metadata, and official runtime instructions.
2. Identify genuine sample input schema: latest audio path, context, installed apps, tool JSON schemas, conversation history (including whether history contains audio or text), labels available for TRAIN, and required PUBLIC_TEST output.
3. Read `zaic_tools` public API, tool schema extensions (`x-ref`, `x-match`, `x-one-of`, fuzzy fields), and official scorer. **Do not inspect environment state files.** Implement against discovered APIs, not guessed ones.
4. Verify model release dates, licenses, exact parameter counts, parser compatibility, and offline loading. Write a short `docs/compatibility.md` with evidence and any unresolved limitations. Also write `docs/competition_contract.md` as required by section A.5.

## 2. Models and safe download order

Use these exact checkpoint IDs as candidates, subject to verified availability/eligibility:

**ASR**
- `Qwen/Qwen3-ASR-0.6B` — https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- `vinai/PhoWhisper-large` — https://huggingface.co/vinai/PhoWhisper-large
- `FunAudioLLM/Fun-ASR-MLT-Nano-2512` — https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512

**LLM**
- `Qwen/Qwen3-4B-Instruct-2507` — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- `Qwen/Qwen3.5-4B` — https://huggingface.co/Qwen/Qwen3.5-4B
- `Qwen/Qwen3.5-2B` — https://huggingface.co/Qwen/Qwen3.5-2B

**First working pair:** `Qwen/Qwen3-ASR-0.6B` + `Qwen/Qwen3-4B-Instruct-2507` (candidate only; verify combined actual parameters). The checkpoint called '0.6B' may include additional parameters in its actual weight files: audit instead of assuming its name equals its total parameter count.

Download **only the first working pair initially**. Implement and create a valid submission with those two. **After the first submission file is produced and validated**, download the remaining four checkpoints and benchmark them. Before each download check free disk space, use resumable official HF download utilities, store weights in `models/<short-name>/`, and record repository ID, resolved revision/commit SHA, weight sizes, licensing, and measured parameter count. Avoid duplicate Hugging Face caches by setting project-local `HF_HOME` and documenting disk usage. If resources cannot accommodate all six simultaneously, download/run one candidate at a time and preserve results; do not delete user assets.

Important: `Qwen3.5-4B` + `PhoWhisper-large` is likely over the <6B budget: **do not submit an illegal combination**. Recompute all 9 pair counts from actual checkpoints. Exclude any invalid pair from end-to-end comparison and say why.

## 3. Minimal project architecture

Adapt an existing structure if present. Otherwise keep approximately:

```
README.md
pyproject.toml
config.yaml
prompts/agent.txt
src/lifemate/{cli.py,data.py,asr.py,llm.py,agent.py,toolenv.py,metrics.py}
scripts/{download_models.sh,serve_llm.sh}
tests/{test_data.py,test_toolenv.py,test_agent.py,test_submission.py}
models/                # gitignored
outputs/               # gitignored
result/                # public submission
```

Avoid Docker/Kubernetes/UI/server infrastructure until a complete local submission works. A local `vllm serve` process is fine (loopback only); include Docker packaging instructions once baseline works.

## 4. End-to-end inference behavior

For each sample:

1. Load only permitted input data. Resolve audio paths safely and log audio duration, sample rate, channels, clipping/silence summary, and a short/noisy/truncated flag **only when supported by input metadata or objective heuristics**. Do not treat heuristic flags as ground-truth labels.
2. Transcribe the **current utterance** using the selected local ASR with explicit Vietnamese language where supported. Preserve literal speech; **do not invent missing words, complete cut-off utterances, or silently merge unrelated clips**. If prior turns contain audio, transcribe them separately with caching and preserve order.
3. Build agent context from transcript + allowed user context + installed tools + actual conversation history. Do not feed gold transcripts/labels into submission inference.
4. Use **PydanticAI Agent + ExternalToolset + deferred tool execution** where it is compatible with installed version; use the documented `HandleDeferredToolCalls` or explicit deferred resume flow. Verify exact imported API in installed version. If blocked by library incompatibility, implement the thinnest native OpenAI-compatible tool loop as a clearly documented fallback, not a second elaborate framework.
5. Serve LLM locally with vLLM, using model-specific verified tool-call parsers/chat templates. Qwen3 Instruct typically uses `hermes`; Qwen3.5 may require `qwen3_coder`. Verify actual behavior by executing one single-tool and one multi-step tool-use smoke test before running dataset.
6. Dynamically expose only installed app tools from the official schema. Preserve descriptions, enums, required fields, and extensions. Validate tool names and arguments before dispatch. Never fill missing IDs, addresses, payment methods, amounts, dates, or contacts by fabrication.
7. For every model tool request, call the **real official ToolEnv** in emitted order. Return the actual observation (including error) to the model. Keep side effects sequential, and **never blindly retry a write tool after uncertain success**. Only record calls actually executed, never planned-but-unexecuted calls.
8. Support multi-step dependencies, conditional actions, `sys.ask_missing_info` for essential missing information, `[]` for out-of-scope requests, and prompt termination as soon as the requested state is satisfied. Handle ASR uncertainty/conspicuous truncation conservatively rather than hallucinating slots.
9. Set deterministic decoding, context cap <=16,384, configurable max tool/LLM turns, bounded tool output size without losing necessary facts, per-sample deadline, and per-sample error isolation. Record failures, do not fabricate `state_hash` or claim a failed sample succeeded.
10. Derive `state_hash` **only** with `env.get_state_hash()` at end. Emit exactly one JSON object per sample to `result/submission.jsonl`, with `id`, ordered `tool_calls` (`name`, `arguments`), and `state_hash`. Validate schema, unique IDs, order, count, and reproducibility. Do not upload it to the contest portal automatically.

## 5. Mandatory detailed logs (the main research deliverable)

Use per-run directories, e.g. `outputs/runs/<timestamp>_<model-slug>/`, with these files:

- `run_config.json`: git SHA (if available), seed, versions, model IDs/revisions, parameter counts, decoding config, prompt version/hash, parser, hardware, official-kit version, split, selected sample IDs.
- `asr_outputs.jsonl`: `sample_id`, audio path (relative), duration, sample rate, ASR model, **raw transcript**, optionally normalized transcript, ASR inference time, warnings/errors, any confidence/logprob fields only when actually provided by the model. Include history-audio transcripts where applicable. Never invent confidence scores.
- `llm_turns.jsonl`: `sample_id`, turn index, **exact messages sent to LLM or a lossless local trace reference**, model output text and structured tool_calls exactly as returned, finish reason, parser errors, latency, token usage if available. Preserve original text even when tool parsing fails. No assumed hidden chain-of-thought access; log only fields actually exposed.
- `tool_events.jsonl`: each proposed call, validation result, executed call and exact arguments, execution result/error, time, and whether execution occurred. Keep log ordering stable.
- `predictions.jsonl`: `id`, executed `tool_calls`, actual `state_hash`, status/error, latency breakdown.
- `metrics.json` and `summary.csv`: overall and per-model metrics; error categories and aggregate runtime.

Do not log secret env vars, access tokens, raw inaccessible environment state, or commit user-specific PII logs. Keep raw tool observations local and access controlled. Provide an easy `--sample-id` trace inspection command and a short README example.

## 6. Small, scientifically useful benchmark

Create a reproducible **TRAIN-only** held-out subset of ~200 samples if the dataset is large enough. Prefer 150 representative randomly sampled + 50 difficult examples *identified solely from permitted TRAIN data* (short <2s, regional accent if labeled, background noise, slurred speech, interrupted utterances, multi-step and missing-info cases). If TRAIN is smaller, use what exists. Group splits by conversation/session or speaker ID when available to avoid leakage. Save only sample ID lists and seed.

**E0 / smoke (first):** verify 2-5 TRAIN samples, including a single-step and dependent multi-step tool case; verify no invalid tool calls, logs produced, actual hash, and local replay.

**E1 / ASR:** 3 ASR candidates on the same TRAIN audio subset. Cache independent transcripts. Measure corpus WER/CER only when authorized reference transcripts genuinely exist, plus slot-critical transcription error analysis when ground truth available, hallucinated/completed-tail cases, duration-stratified errors, and ASR latency. Never manufacture a reference transcript or derive it from gold tool calls.

**E2 / LLM:** 3 LLM candidates on the **same reference transcripts** if official TRAIN transcripts exist. If no reference transcripts exist, use the **same fixed cached transcript from the baseline ASR** for all LLMs, and explicitly label this as a controlled-LMM comparison rather than an oracle comparison. Hold prompts, tools, config, and ToolEnv behavior constant. Measure official local score, state-match rate, tool-call F1, schema validity, excess tool calls, and time.

**E3 / end-to-end:** run the top 2 ASRs × top 2 LLMs where total parameter count <6B and LLM <5B. Use cached ASR transcripts, replay each test with a **fresh ToolEnv per sample**, and compare actual end-to-end official scores; never pick the winner on WER alone. Log full traces for every variant.

If the official scoring package exists, **call it**. Do not replace official scoring with a guessed metric. If it is unavailable, report surrogate metrics separately and mark official score as unavailable.

## 7. CLI and scripts

Provide documented, real commands with working help output for:

```
python -m lifemate.cli inspect
python -m lifemate.cli download --group baseline
python -m lifemate.cli smoke --split train --limit 5
python -m lifemate.cli evaluate --split train --sample-ids outputs/splits/validation_ids.json
python -m lifemate.cli submit --split public_test --output result/submission.jsonl
python -m lifemate.cli download --group remaining
python -m lifemate.cli benchmark-asr
python -m lifemate.cli benchmark-llm
python -m lifemate.cli benchmark-e2e
python -m lifemate.cli trace --sample-id <id> --run-dir <path>
```

Implement the CLI flags (or equivalent, well documented actual commands); no fake command examples. `--limit` and `--sample-id` must support fast debugging. Respect user-configured project/data/model paths, GPU selection, timeouts, and run IDs. Scripts must fail with clear actionable messages when a checkpoint or dataset path is missing.

## 8. Required execution order and acceptance tests

1. Inspect project/kit/hardware, verify legal model combinations, create environment/config and unit tests.
2. Implement data loader, ASR/LLM adapters, PydanticAI agent + ToolEnv bridge, structured traces, and output writer.
3. Download baseline **ASR + LLM** checkpoint pair locally; smoke-test both inference components. Debug actual tool parsing and execution.
4. Run TRAIN smoke and a small held-out evaluation; validate legal operations and exact hash provenance.
5. Run PUBLIC_TEST **once with the working baseline** to produce `result/submission.jsonl`; run format, sample-ID and replay checks; preserve this first result even if later experiments fail.
6. Only after the first submission file is validated, download the remaining four models (storage permitting), then run E1, E2, E3 and generate comparative reports.
7. Write `README.md` with reproduction commands, the real setup status, model download manifest, limitations, and `outputs/reports/model_comparison.md` with the actual results. If no GPU/data/network is available, finish everything possible, explicitly list blocked commands and why, and never invent scores or claim completion.

**Minimum done criteria:** a single-sample end-to-end call works on local models; official ToolEnv executed calls match the stored ordered log; the JSONL is correctly shaped and generated without gold/test-label leakage; logs clearly show the original ASR transcript and every LLM decision/tool observation; baseline and benchmark results have reproducible commands and honest outcomes.

Before ending, provide a brief report: files changed, packages pinned, models downloaded with paths and exact revision, tests/commands executed and their results, first submission file location and row count, top model comparison (if completed), and any blockers. **Implement and execute, not just describe the design.**
