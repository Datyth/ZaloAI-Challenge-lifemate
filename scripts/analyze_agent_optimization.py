"""TRAIN-only E4/E5/E6 analysis from organizer inputs and recorded traces.

No environment-state files are read. Gold labels are used only here, never in inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import items  # noqa: E402
from lifemate.metrics import error_counts, normalized  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.is_file() else []


def load_run(path: Path) -> dict:
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    if config["split"] != "train":
        raise ValueError(f"analysis requires TRAIN runs: {path}")
    predictions = read_jsonl(path / "predictions.jsonl")
    expected = config["sample_ids"]
    if [row["id"] for row in predictions] != expected or any(row.get("status") != "ok" for row in predictions):
        raise ValueError(f"run is incomplete or has failures: {path}")
    return {"path": str(path), "config": config, "predictions": {r["id"]: r for r in predictions},
            "asr": {(r["sample_id"], r["audio"]): r for r in read_jsonl(path / "asr_outputs.jsonl")},
            "llm": _group(read_jsonl(path / "llm_turns.jsonl")),
            "events": _group(read_jsonl(path / "tool_events.jsonl"))}


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["sample_id"]].append(row)
    return result


def _write_kind(name: str, specs: dict[str, dict]) -> bool:
    return specs.get(name, {}).get("kind") == "write"


def word_edits(oracle: str, asr: str | None) -> list[dict]:
    reference, prediction = normalized(oracle).split(), normalized(asr or "").split()
    return [{"change": tag, "oracle": reference[i:j], "asr": prediction[k:l]}
            for tag, i, j, k, l in SequenceMatcher(None, reference, prediction).get_opcodes() if tag != "equal"]


def classify(sample: dict, base: dict, oracle: dict | None, events: list[dict], specs: dict[str, dict]) -> tuple[str, list[str], list[str]]:
    gold = sample["label"]["tool_calls"]
    predicted = base["tool_calls"]
    gold_names = [x["name"] for x in gold]
    pred_names = [x["name"] for x in predicted]
    labels: list[str] = []
    evidence: list[str] = []
    invalid = [x for x in events if x.get("executed") is False]
    if oracle and oracle.get("train_state_match") is True:
        labels.append("ASR errors")
        evidence.append("E1 failed and E4 matched the target state with oracle transcripts")
    if base.get("status") == "error":
        labels.append("Runtime / parser errors")
        evidence.append(str(base.get("error", base.get("termination"))))
    if invalid:
        labels.append("Tool grounding")
        evidence.append(f"{len(invalid)} rejected proposals: " + "; ".join(str(x.get("validation", {}).get("error", x.get("error"))) for x in invalid[:2]))
    if any(x["name"] == "sys.ask_missing_info" for x in gold) and "sys.ask_missing_info" not in pred_names:
        labels.append("Missing information")
        evidence.append("gold requests sys.ask_missing_info; prediction omits it")
    missing = list((Counter(gold_names) - Counter(pred_names)).elements())
    extra = list((Counter(pred_names) - Counter(gold_names)).elements())
    if missing:
        labels.append("Intent / tool selection")
        evidence.append("missing gold tool names: " + ", ".join(missing))
    if extra:
        labels.append("Unnecessary actions")
        evidence.append("extra predicted tool names: " + ", ".join(extra))
    if (any(_write_kind(name, specs) for name in missing) and pred_names and not extra
        and all(specs.get(name, {}).get("kind") == "read" for name in pred_names)
        and any(name in gold_names for name in pred_names)):
        labels.append("Premature termination")
        evidence.append("agent stopped after a read while a gold write was still missing")
    if len(gold) > 1 and missing:
        labels.append("Multi-step planning")
        evidence.append("multi-call gold sequence was incomplete")
    if gold_names == pred_names and gold != predicted:
        labels.append("Argument extraction")
        evidence.append("tool sequence matches but arguments differ")
    # A history reference is only a clue, not proof of a history error.
    if sample.get("history") and any(word in sample.get("audio_text", "").casefold() for word in ("đó", "ấy", "này", "lúc nãy", "vừa rồi")) and missing:
        labels.append("History resolution")
        evidence.append("request contains a possible backward reference; inspect history and LLM trace")
    priority = ["ASR errors", "Runtime / parser errors", "Tool grounding", "Missing information",
                "Premature termination", "Argument extraction", "Intent / tool selection",
                "Multi-step planning", "Unnecessary actions", "History resolution"]
    primary = next((name for name in priority if name in labels), "Intent / tool selection")
    if not labels:
        evidence.append("state mismatch despite no diagnostic rule; inspect full trace")
    return primary, [x for x in labels if x != primary], evidence


def render_report(result: dict, error_rows: list[dict]) -> str:
    comparison = {row["experiment"]: row for row in result["comparison"]}
    lines = ["# LifeMate E4–E6 agent optimization", "", "## Measured TRAIN results", "",
             "All rows use the same frozen 200-sample TRAIN split. State match and exact-call F1 are local surrogates; no official score is available.", "",
             "| Experiment | State match | Exact-call F1 | Invalid proposals | Abnormal terminations | Mean / P95 latency (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("E1", "E4", "E6", "E6-Oracle"):
        if name in comparison:
            row = comparison[name]
            lines.append(f"| {name} | {row['state_matches']}/{row['samples']} ({row['state_match_rate']:.1%}) | {row['tool_f1_surrogate']:.3f} | {row['invalid_tool_call_rate']:.1%} | {row['abnormal_termination_rate']:.1%} | {row['mean_latency_seconds']:.2f} / {row['p95_latency_seconds']:.2f} |")
    if "E4" not in comparison:
        lines += ["", "Full E4, E6, and E6-Oracle runs are pending; the table currently reports only the preserved E1 baseline."]
    else:
        lines += ["", "Oracle runs skip ASR inference, so their end-to-end latency excludes ASR time; compare latency with that limitation."]
    lines += ["", "## E5 failure analysis", "",
              f"E1 missed the target state in {result['e1_failed_samples']} samples. Categories below are rule-based diagnostic labels, not manual ground truth.", "",
              "| Primary category | Samples |", "| --- | ---: |"]
    for name, count in result["primary_categories"].items():
        lines.append(f"| {name} | {count} |")
    lines += ["", "Each failed sample, full history, transcript pair, gold and predicted calls, decisions, observations, evidence, and suggested fix is in `e5_errors.jsonl`. Multiple categories can apply.", ""]
    if "E4" in comparison:
        delta = comparison["E4"]["state_matches"] - comparison["E1"]["state_matches"]
        lines += ["## Oracle transcript effect", "",
                  f"E4 changed state matches by {delta:+d}/200 ({delta / 200:+.1%}) relative to E1. {len(result['oracle_fixed_ids'])} previously failing IDs became correct: {', '.join(result['oracle_fixed_ids']) or 'none'}.",
                  "These fixed cases support an ASR contribution; other failures can still involve both transcription and reasoning.", ""]
    if "E6" in comparison:
        base, improved = comparison["E1"], comparison["E6"]
        gain = improved["state_matches"] - base["state_matches"]
        choose_e6 = (gain >= 4 and improved["tool_f1_surrogate"] >= base["tool_f1_surrogate"] - .01
                     and improved["abnormal_termination_rate"] <= base["abnormal_termination_rate"]
                     and improved["invalid_tool_call_rate"] <= base["invalid_tool_call_rate"] + .05
                     and improved["p95_latency_seconds"] < 30)
        recommendation = "E6" if choose_e6 else "E1"
        lines += ["## E6 and next submission", "",
                  f"E6 changed state matches by {gain:+d}/200 and exact-call F1 by {improved['tool_f1_surrogate'] - base['tool_f1_surrogate']:+.3f}. On the predefined conservative selection rule (at least four more correct states, no material F1, invalid-call or termination regression, and P95 below 30 seconds), use **{recommendation}** for the next submission.",
                  "E6 adds a prompt for completing dependent actions and one bounded read-to-action check. It does not alter model weights or ToolEnv. The E1 validation traces informed this change, so this comparison is exploratory and may overstate generalization.", ""]
    lines += ["## Failure examples", ""]
    for row in error_rows[:5]:
        lines.append(f"- `{row['id']}`: {row['primary_category']}; {row['evidence'][0]}. Gold: {', '.join(x['name'] for x in row['gold_tool_calls']) or '(none)'}; predicted: {', '.join(x['name'] for x in row['predicted_tool_calls']) or '(none)'}.")
    lines += ["", "## Remaining bottlenecks", "",
              "Review `e5_errors.jsonl` manually before treating the category counts as causal. If the oracle gap dominates E6 gain, prioritize ASR and Vietnamese entity recognition. If oracle transcripts still leave many wrong tool choices, compare legal LLM checkpoints before considering fine-tuning. Fine-tuning would require a separate data and competition-rule audit.", ""]
    return "\n".join(lines)


def analyze(e1: Path, e4: Path | None, e6: Path | None, e6_oracle: Path | None, output_dir: Path, report: Path | None = None) -> dict:
    runs = {"E1": load_run(e1)}
    for label, path in (("E4", e4), ("E6", e6), ("E6-Oracle", e6_oracle)):
        if path:
            runs[label] = load_run(path)
    if "E6-Oracle" in runs and "E6" not in runs:
        raise ValueError("E6-Oracle comparison requires the matching E6 run")
    expected = runs["E1"]["config"]["sample_ids"]
    for label, run in runs.items():
        if run["config"]["sample_ids"] != expected:
            raise ValueError(f"{label} uses a different validation sample set")
        if run["config"].get("orchestration") != ("E6" if label.startswith("E6") else "E1"):
            raise ValueError(f"{label} uses the wrong agent configuration")
        for key in ("llm_model", "llm_manifest", "parser", "decoding", "max_turns", "deadline_seconds"):
            if run["config"].get(key) != runs["E1"]["config"].get(key):
                raise ValueError(f"{label} differs from E1 on {key}")
        if label == "E4" and run["config"].get("prompt_sha256") != runs["E1"]["config"].get("prompt_sha256"):
            raise ValueError("E4 must keep the E1 prompt unchanged")
        if label == "E6-Oracle" and run["config"].get("prompt_sha256") != runs["E6"]["config"].get("prompt_sha256"):
            raise ValueError("E6-Oracle must keep the E6 prompt unchanged")
        if label in {"E1", "E6"} and run["config"]["transcript_source"] != "asr":
            raise ValueError(f"{label} must use ASR transcripts")
        if label in {"E4", "E6-Oracle"} and run["config"]["transcript_source"] != "train-reference":
            raise ValueError(f"{label} must use TRAIN references")
    train = {x["id"]: x for x in items(PROJECT / "data" / "train+public_test", "train") if x["id"] in set(expected)}
    specs_raw = json.loads((PROJECT / "data" / "train+public_test" / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    specs = {t["name"]: t for app in specs_raw["apps"] for t in app["tools"]}
    analysis_rows: list[dict] = []
    oracle_rows: list[dict] = []
    primary_counts: Counter = Counter()
    any_counts: Counter = Counter()
    oracle_fixed: list[str] = []
    for sample_id in expected:
        sample = train[sample_id]
        base = runs["E1"]["predictions"][sample_id]
        asr = runs["E1"]["asr"].get((sample_id, sample["audio"]), {}).get("raw_transcript")
        if "E4" in runs:
            oracle_prediction = runs["E4"]["predictions"][sample_id]
            oracle_rows.append({"id": sample_id, "asr_transcript": asr, "oracle_transcript": sample.get("audio_text"),
                                "asr_error_counts": error_counts(sample.get("audio_text", ""), asr or ""),
                                "asr_word_edits": word_edits(sample.get("audio_text", ""), asr),
                                "history_transcripts": [{"audio": turn["audio"], "asr_transcript": runs["E1"]["asr"].get((sample_id, turn["audio"]), {}).get("raw_transcript"),
                                                         "oracle_transcript": turn.get("audio_text")}
                                                        for turn in sample.get("history", []) if turn.get("role") == "user" and "audio" in turn],
                                "e1_tool_calls": base["tool_calls"], "e4_tool_calls": oracle_prediction["tool_calls"],
                                "e4_tool_observations": runs["E4"]["events"].get(sample_id, []),
                                "e4_state_hash": oracle_prediction["state_hash"], "e4_state_match": oracle_prediction.get("train_state_match"),
                                "e4_tool_f1_surrogate": oracle_prediction.get("train_exact_call_f1_surrogate"),
                                "e4_latency_seconds": oracle_prediction.get("latency_seconds"), "e4_termination": oracle_prediction.get("termination")})
        if base.get("train_state_match") is True:
            continue
        oracle = runs["E4"]["predictions"][sample_id] if "E4" in runs else None
        if oracle and oracle.get("train_state_match") is True:
            oracle_fixed.append(sample_id)
        history = []
        for turn in sample.get("history", []):
            record = dict(turn)
            if "audio" in turn:
                record["asr_transcript"] = runs["E1"]["asr"].get((sample_id, turn["audio"]), {}).get("raw_transcript")
                record["oracle_transcript"] = turn.get("audio_text")
            history.append(record)
        primary, secondary, evidence = classify(sample, base, oracle, runs["E1"]["events"].get(sample_id, []), specs)
        primary_counts[primary] += 1
        any_counts.update([primary, *secondary])
        decisions = [{"turn": x["turn"], "content": (x.get("output") or {}).get("content"),
                      "tool_calls": (x.get("output") or {}).get("tool_calls"), "finish_reason": x.get("finish_reason")}
                     for x in runs["E1"]["llm"].get(sample_id, [])]
        observations = [{"turn": x.get("turn"), "name": x.get("name"), "arguments": x.get("arguments"),
                         "result": x.get("result"), "validation": x.get("validation"), "executed": x.get("executed")}
                        for x in runs["E1"]["events"].get(sample_id, [])]
        fixes = {"ASR errors": "improve transcript quality and Vietnamese entity recognition",
                 "Premature termination": "check whether requested work remains after read observations",
                 "Argument extraction": "ground values in speech and successful tool observations",
                 "Tool grounding": "resolve IDs with read tools before writes",
                 "Missing information": "call sys.ask_missing_info for required unknown values",
                 "Multi-step planning": "continue dependent actions in order",
                 "History resolution": "resolve references from verified chronological context",
                 "Unnecessary actions": "stop after the requested work is complete",
                 "Runtime / parser errors": "inspect native-call protocol and bounded recovery",
                 "Intent / tool selection": "clarify tool choice and app capability in prompt"}
        analysis_rows.append({"id": sample_id, "original_request": sample.get("audio_text"), "history": history,
                              "asr_transcript": asr, "oracle_transcript": sample.get("audio_text"),
                              "asr_error_counts": error_counts(sample.get("audio_text", ""), asr or ""),
                              "asr_word_edits": word_edits(sample.get("audio_text", ""), asr),
                              "gold_tool_calls": sample["label"]["tool_calls"], "predicted_tool_calls": base["tool_calls"],
                              "oracle_tool_calls": oracle["tool_calls"] if oracle else None,
                              "state_hash": base["state_hash"], "oracle_state_match": oracle.get("train_state_match") if oracle else None,
                              "llm_decisions": decisions, "tool_observations": observations,
                              "primary_category": primary, "secondary_categories": secondary,
                              "evidence": evidence, "recommended_fix": fixes[primary],
                              "termination": base.get("termination"), "latency_seconds": base.get("latency_seconds")})
    comparison = []
    for label, run in runs.items():
        predictions = list(run["predictions"].values())
        times = sorted(float(x["latency_seconds"]) for x in predictions)
        events = [event for records in run["events"].values() for event in records]
        abnormal = {"step_limit", "timeout", "token_limit", "invalid_response", "model_error", "tool_error"}
        comparison.append({"experiment": label, "samples": len(predictions),
                           "state_matches": sum(x.get("train_state_match") is True for x in predictions),
                           "state_match_rate": sum(x.get("train_state_match") is True for x in predictions) / len(predictions),
                           "tool_f1_surrogate": sum(x.get("train_exact_call_f1_surrogate", 0) for x in predictions) / len(predictions),
                           "invalid_tool_call_rate": sum(event.get("executed") is False for event in events) / len(events) if events else 0.0,
                           "abnormal_termination_rate": sum(x.get("termination") in abnormal for x in predictions) / len(predictions),
                           "mean_latency_seconds": sum(times) / len(times), "p95_latency_seconds": times[(95 * len(times) + 99) // 100 - 1],
                           "official_score": None})
    result = {"comparison": comparison, "e1_failed_samples": len(analysis_rows),
              "primary_categories": dict(primary_counts.most_common()), "all_categories": dict(any_counts.most_common()),
              "oracle_fixed_ids": oracle_fixed,
              "notes": "State match and exact-call F1 are local surrogates; categories are rule-based diagnostics requiring manual review."}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "e5_errors.jsonl").open("w", encoding="utf-8") as out:
        for row in analysis_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if oracle_rows:
        with (output_dir / "e4_oracle_pairs.jsonl").open("w", encoding="utf-8") as out:
            for row in oracle_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(result, analysis_rows), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e4", type=Path)
    parser.add_argument("--e6", type=Path)
    parser.add_argument("--e6-oracle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.e1, args.e4, args.e6, args.e6_oracle, args.output_dir, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
