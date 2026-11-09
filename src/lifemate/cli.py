"""LifeMate command line. No gold labels enter inference."""
from __future__ import annotations

import argparse
from array import array
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

from .context import normalize_asr
from .agent import execute, prompt_for
from .asr import GipformerASR
from .data import DATA_ROOT as DEFAULT_ROOT, PROJECT, audio_path, catalog, installed_tools, items
from .llm import LocalChat
from .metrics import error_counts, exact_tool_f1
from .toolenv import new_env
from .submission import validate_file

# Released checkpoints (scripts/setup/get_models.sh, get_checkpoint.sh); ASR runs on CPU, the LLM in a local vLLM server.
ASR_DIR = PROJECT / "models" / "gipformer-lm-v2-01"
LLM_DIR = PROJECT / "models" / "qwen3.5-4b"


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


def audio_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        result = {"duration_seconds": duration, "sample_rate": wav.getframerate(), "channels": wav.getnchannels(), "warnings": []}
        if duration < 2:
            result["warnings"].append("short_audio_heuristic")
        if wav.getsampwidth() == 2 and wav.getcomptype() == "NONE":
            samples = array("h")
            samples.frombytes(wav.readframes(wav.getnframes()))
            if samples:
                rms = (sum(x * x for x in samples) / len(samples)) ** 0.5 / 32768
                clipped = sum(abs(x) >= 32760 for x in samples) / len(samples)
                result.update({"rms_fraction": rms, "clipped_fraction": clipped})
                if rms < 0.005:
                    result["warnings"].append("quiet_audio_heuristic")
                if clipped > 0.01:
                    result["warnings"].append("clipped_audio_heuristic")
        return result


def run(args: argparse.Namespace) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = args.data_root.resolve()
    args.asr_dir = args.asr_dir.resolve()
    args.llm_dir = args.llm_dir.resolve()
    if args.adapter_dir:
        args.adapter_dir = args.adapter_dir.resolve()
        if not (args.adapter_dir / "adapter_config.json").is_file() or not (args.adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"PEFT adapter missing config or weights: {args.adapter_dir}")
        if not args.llm_served_model:
            raise ValueError("--adapter-dir requires --llm-served-model from the vLLM LoRA server")
    elif args.llm_served_model:
        raise ValueError("--llm-served-model requires --adapter-dir")
    split = args.split
    selected = list(items(root, split))
    if args.sample_id:
        selected = [x for x in selected if x["id"] == args.sample_id]
        if not selected:
            raise ValueError(f"sample id not found: {args.sample_id}")
    if args.command == "evaluate" and args.sample_ids:
        ids = set(json.loads(args.sample_ids.read_text()))
        selected = [x for x in selected if x["id"] in ids]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no samples selected")
    if args.transcript_source == "train-reference" and (split != "train" or args.command == "submit"):
        raise ValueError("TRAIN reference transcripts are allowed only for TRAIN smoke/evaluate")
    if not (args.llm_dir / "config.json").is_file():
        raise FileNotFoundError(f"LLM checkpoint missing: {args.llm_dir}")
    if args.transcript_source in {"asr", "cached-asr"}:
        if not (args.asr_dir / "encoder.onnx").is_file():
            raise FileNotFoundError(f"ASR checkpoint missing: {args.asr_dir}")
    saved_transcripts: dict[tuple[str, str], dict] = {}
    if args.transcript_source == "cached-asr":
        if args.command not in {"evaluate", "submit", "smoke"} or args.transcripts_jsonl is None:
            raise ValueError("cached-asr requires --transcripts-jsonl")
        cache_config_path = args.transcripts_jsonl.parent / "run_config.json"
        cache_metrics_path = args.transcripts_jsonl.parent / "metrics.json"
        if cache_config_path.is_file():
            cache_config = json.loads(cache_config_path.read_text(encoding="utf-8"))
            if cache_config.get("split") != split or cache_config.get("model") != str(args.asr_dir):
                raise ValueError("ASR cache split or checkpoint differs from this run")
        if cache_metrics_path.is_file():
            cache_metrics = json.loads(cache_metrics_path.read_text(encoding="utf-8"))
            if cache_metrics.get("complete") is False:
                raise ValueError("ASR cache is incomplete")
        expected_model = str(args.asr_dir)
        with args.transcripts_jsonl.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["sample_id"], record["audio"])
                if key in saved_transcripts:
                    raise ValueError(f"duplicate saved ASR transcript: {key}")
                if record.get("model") != expected_model or not isinstance(record.get("raw_transcript"), str) or not isinstance(record.get("latency_seconds"), (int, float)) or record["latency_seconds"] < 0:
                    raise ValueError(f"invalid saved ASR transcript or model: {key}")
                saved_transcripts[key] = record
        required_audio = [(item["id"], item["audio"]) for item in selected]
        required_audio += [(item["id"], turn["audio"]) for item in selected for turn in item["history"]
                           if turn.get("role") == "user" and "audio" in turn]
        missing = [key for key in required_audio if key not in saved_transcripts]
        if missing:
            raise ValueError(f"saved ASR transcripts missing {len(missing)} current/history audio turns; first={missing[0]}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run inference with GPU device access enabled")
    if not args.device.startswith("cuda:") or int(args.device.split(":", 1)[1]) >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPU {args.device} is unavailable; detected {torch.cuda.device_count()} CUDA devices")
    asr = GipformerASR(args.asr_dir, device="cpu") if args.transcript_source == "asr" else None  # as in the Docker image
    llm = LocalChat(args.llm_url, args.llm_served_model or str(args.llm_dir), timeout=args.deadline)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_id = args.run_id or f"{stamp}_{args.command}_{time.time_ns() % 1000000:06d}"
    run_dir = PROJECT / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    versions = {}
    for package in ("torch", "sherpa-onnx", "vllm", "transformers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    def manifest(directory: Path) -> dict | None:
        path = directory / "lifemate_manifest.json"
        return json.loads(path.read_text()) if path.is_file() else None
    use_asr = args.transcript_source in {"asr", "cached-asr"}
    config = {"split": split, "sample_ids": [x["id"] for x in selected], "seed": 42, "git_sha": git_sha, "kit_version": (root / "VERSION").read_text(), "versions": versions, "asr_model": str(args.asr_dir) if use_asr else None, "asr_manifest": manifest(args.asr_dir) if use_asr else None, "transcript_source": args.transcript_source, "transcripts_jsonl": str(args.transcripts_jsonl.resolve()) if args.transcripts_jsonl else None, "llm_model": str(args.llm_dir), "llm_manifest": manifest(args.llm_dir), "llm_served_model": args.llm_served_model or str(args.llm_dir), "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None, "adapter_config_sha256": hashlib.sha256((args.adapter_dir / "adapter_config.json").read_bytes()).hexdigest() if args.adapter_dir else None, "adapter_weights_sha256": hashlib.sha256((args.adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest() if args.adapter_dir else None, "device": args.device, "gpu_name": torch.cuda.get_device_name(int(args.device.split(":", 1)[1])), "prompt_sha256": hashlib.sha256(prompt_for(args.orchestration).encode()).hexdigest(), "orchestration": args.orchestration, "decoding": {"temperature": 0, "seed": 42, "max_tokens": 512}, "deadline_seconds": args.deadline, "max_turns": args.max_turns, "asr_lowercase": args.asr_lowercase, "dedup_calls": args.dedup_calls, "ground_args": args.ground_args}
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (run_dir / "runtime_errors.jsonl").touch()
    specs = catalog(root, split)
    destination = args.output if args.output else run_dir / "submission.jsonl"
    if args.command == "submit" and destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {destination}")
    temp = destination.with_name(destination.name + f".{run_id}.partial")
    word_errors = reference_words = character_errors = reference_characters = 0
    durations: list[float] = []
    failures = 0
    train_state_matches: list[bool] = []
    train_call_f1: list[float] = []
    try:
        for item_index, item in enumerate(selected, start=1):
            started = time.monotonic()
            cached_audio = {item["audio"], *(turn["audio"] for turn in item["history"] if turn.get("role") == "user" and "audio" in turn)}
            cached_asr_seconds = sum(float(saved_transcripts[(item["id"], audio)]["latency_seconds"])
                                     for audio in cached_audio) if args.transcript_source == "cached-asr" else 0.0
            env = new_env(root, split, item["id"])
            termination: dict = {}
            def log(kind: str, record: dict) -> None:
                write_jsonl(run_dir / f"{kind}.jsonl", record)
                if kind == "terminations":
                    termination.update(record)
            transcript_cache: dict[str, str] = {}
            if args.transcript_source == "train-reference":
                transcript_cache[item["audio"]] = item["audio_text"]
                for turn in item.get("history", []):
                    if turn.get("role") == "user" and "audio" in turn and isinstance(turn.get("audio_text"), str):
                        transcript_cache[turn["audio"]] = turn["audio_text"]
            def transcribe(relative: str) -> str:
                if relative in transcript_cache:
                    return transcript_cache[relative]
                saved = saved_transcripts.get((item["id"], relative))
                if saved is not None:
                    content = normalize_asr(saved["raw_transcript"]) if args.asr_lowercase else saved["raw_transcript"]
                    transcript_cache[relative] = content
                    log("asr_outputs", {**saved, "source": "saved", "transcript_file": str(args.transcripts_jsonl)})
                    return content
                if asr is None:
                    raise ValueError(f"TRAIN reference transcript unavailable for {relative}")
                path = audio_path(root, split, relative)
                t0 = time.monotonic()
                content = asr.transcribe(path)
                if args.asr_lowercase:
                    content = normalize_asr(content)
                transcript_cache[relative] = content
                info = audio_info(path)
                log("asr_outputs", {"sample_id": item["id"], "audio": relative, "model": str(args.asr_dir), "raw_transcript": content, "latency_seconds": time.monotonic() - t0, "source": "inference", **info})
                return content
            try:
                transcript = transcribe(item["audio"])
                remaining = args.deadline - (time.monotonic() - started) - cached_asr_seconds
                if remaining <= 0:
                    raise TimeoutError("ASR exhausted per-sample deadline")
                inference_item = {key: item[key] for key in ("id", "audio", "context", "history")}
                allowed_history_keys = {"role", "audio", "text", "tool_calls", "tool_results"}
                inference_item["history"] = [{key: value for key, value in turn.items() if key in allowed_history_keys} for turn in item["history"]]
                result = execute(inference_item, transcript, installed_tools(inference_item, specs), env, llm, transcribe, log, max_turns=args.max_turns, deadline_seconds=remaining, orchestration=args.orchestration, dedup_calls=args.dedup_calls, ground_args=args.ground_args)
                if result["tool_calls"] != env.get_tool_calls() or result["state_hash"] != env.get_state_hash():
                    raise AssertionError("submission row diverged from ToolEnv")
                elapsed = time.monotonic() - started + cached_asr_seconds
                durations.append(elapsed)
                if use_asr and split == "train" and isinstance(item.get("audio_text"), str):
                    counts = error_counts(item["audio_text"], transcript)
                    word_errors += counts["word_errors"]
                    reference_words += counts["reference_words"]
                    character_errors += counts["character_errors"]
                    reference_characters += counts["reference_characters"]
                evaluation = {}
                if split == "train":
                    gold_env = new_env(root, split, item["id"])
                    for gold_call in item["label"]["tool_calls"]:
                        gold_env.call(gold_call["name"], gold_call["arguments"])
                    state_match = result["state_hash"] == gold_env.get_state_hash()
                    call_f1 = exact_tool_f1(result["tool_calls"], item["label"]["tool_calls"])
                    train_state_matches.append(state_match)
                    train_call_f1.append(call_f1)
                    evaluation = {"train_state_match": state_match, "train_exact_call_f1_surrogate": call_f1}
                ended = termination.get("status")
                tool_outcome = "failed" if ended == "tool_error" else "mixed" if ended == "partial_write" else "completed" if ended == "completed_write" else "unverified"
                log("predictions", {**result, **evaluation, "status": ended if ended in {"tool_error", "partial_write"} else "ok",
                                    "runtime_status": "completed", "tool_outcome": tool_outcome,
                                    "task_completion": "matched_gold_state" if evaluation.get("train_state_match") is True else "different_from_gold_state" if evaluation.get("train_state_match") is False else "unknown",
                                    "termination": ended, "latency_seconds": elapsed, "cached_asr_latency_seconds": cached_asr_seconds})
                write_jsonl(temp, result)
            except Exception as exc:
                failures += 1
                elapsed = time.monotonic() - started + cached_asr_seconds
                durations.append(elapsed)
                failed_row = {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
                if split == "train":
                    train_state_matches.append(False)
                    train_call_f1.append(0.0)
                log("predictions", {**failed_row, "status": "error", "runtime_status": "error", "tool_outcome": "unknown", "task_completion": "unknown", "termination": termination.get("status"), "error": f"{type(exc).__name__}: {exc}", "latency_seconds": elapsed, "cached_asr_latency_seconds": cached_asr_seconds, "train_state_match": False if split == "train" else None, "train_exact_call_f1_surrogate": 0.0 if split == "train" else None})
                log("runtime_errors", {"sample_id": item["id"], "error": f"{type(exc).__name__}: {exc}", "termination": termination.get("status"), "latency_seconds": elapsed})
                if len(durations) == 1 and args.command == "submit" and not args.allow_failures:
                    raise RuntimeError(f"first sample failed; refusing to write a full submission: {exc}") from exc
                write_jsonl(temp, failed_row)
            if item_index % 25 == 0 or item_index == len(selected):
                print(f"processed {item_index}/{len(selected)} samples; failures={failures}; diagnostic file={temp}", file=sys.stderr, flush=True)
        if failures and not args.allow_failures:
            raise RuntimeError(f"{failures} sample(s) failed; refusing to publish a final submission. Inspect {run_dir / 'predictions.jsonl'} and {temp}")
        rows = [json.loads(line) for line in temp.open(encoding="utf-8")]
        expected = [item["id"] for item in selected]
        if len(rows) != len(expected) or [r.get("id") for r in rows] != expected:
            raise AssertionError("submission row count, IDs, or order mismatch")
        if any(set(row) != {"id", "tool_calls", "state_hash"} or not isinstance(row["tool_calls"], list) or not isinstance(row["state_hash"], str) or len(row["state_hash"]) != 64 for row in rows):
            raise AssertionError("submission row shape mismatch")
        replay = validate_file(root, split, temp, allow_partial=bool(args.sample_id or args.limit or args.sample_ids or args.command != "submit"))
        ordered_durations = sorted(durations)
        metrics = {"samples": len(rows), "failures": failures, "replay_hash_matches": replay["replay_hash_matches"], "latency_mean_seconds": sum(durations) / len(durations), "latency_p95_seconds": ordered_durations[max(0, (95 * len(ordered_durations) + 99) // 100 - 1)], "latency_max_seconds": max(durations), "official_score": None}
        if reference_words:
            metrics.update({"asr_wer": word_errors / reference_words, "asr_cer": character_errors / reference_characters, "asr_reference": "TRAIN audio_text; lowercase Unicode word normalization"})
        if train_state_matches:
            metrics.update({"train_state_match_rate": sum(train_state_matches) / len(train_state_matches), "train_exact_call_f1_surrogate": sum(train_call_f1) / len(train_call_f1)})
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as summary:
            writer = csv.DictWriter(summary, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        temp.replace(destination)
    finally:
        print(f"run logs: {run_dir}", file=sys.stderr)
    print(f"wrote {len(selected)} rows: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="lifemate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    for name in ("smoke", "evaluate", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--split", choices=["train", "public_test"], default="public_test" if name == "submit" else "train")
        p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--sample-id")
        p.add_argument("--limit", type=int)
        p.add_argument("--sample-ids", type=Path)
        p.add_argument("--output", type=Path, default=PROJECT / "result" / "submission.jsonl" if name == "submit" else None)
        p.add_argument("--llm-url", default="http://127.0.0.1:8000/v1/chat/completions")
        p.add_argument("--asr-dir", type=Path, default=ASR_DIR)
        p.add_argument("--llm-dir", type=Path, default=LLM_DIR)
        p.add_argument("--llm-served-model", help="vLLM LoRA model ID; requires --adapter-dir")
        p.add_argument("--adapter-dir", type=Path, help="PEFT adapter used by the local vLLM server")
        p.add_argument("--deadline", type=float, default=30)
        p.add_argument("--device", default="cuda:0", help="GPU recorded in run_config (the LLM runs in the vLLM server)")
        p.add_argument("--run-id")
        p.add_argument("--max-turns", type=int, default=10)
        p.add_argument("--transcript-source", choices=["asr", "cached-asr", "train-reference"], default="asr")
        p.add_argument("--transcripts-jsonl", type=Path)
        p.add_argument("--allow-failures", action="store_true", help="keep replay-valid rows and error logs for individual failures")
        p.add_argument("--orchestration", choices=["E0", "E1", "E2", "E3", "E6", "O1", "O2", "O3", "O4", "O5", "S2"], default="E1")
        p.add_argument("--asr-lowercase", action="store_true", help="lower-case every ASR transcript (current and history) before the agent")
        p.add_argument("--dedup-calls", action="store_true", help="never execute an identical (name, arguments) call twice in a sample; stop after three repeats")
        p.add_argument("--ground-args", action="store_true", help="writes: reject ID arguments without x-ref that no tool result, history, context or speech contains; drop optional arguments that only copy an unmentioned context value")
    trace = sub.add_parser("trace")
    trace.add_argument("--sample-id", required=True)
    trace.add_argument("--run-dir", type=Path, required=True)
    check = sub.add_parser("validate-submission")
    check.add_argument("--split", choices=["train", "public_test"], default="public_test")
    check.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    check.add_argument("--file", type=Path, required=True)
    check.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        root = DEFAULT_ROOT
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = None
        print(json.dumps({"root": str(root), "train": sum(1 for _ in items(root, "train")), "public_test": sum(1 for _ in items(root, "public_test")), "cuda": cuda}, indent=2))
    elif args.command == "trace":
        for name in ("asr_outputs", "llm_turns", "tool_events", "protocol_events", "terminations", "predictions"):
            path = args.run_dir / f"{name}.jsonl"
            if path.is_file():
                for line in path.open(encoding="utf-8"):
                    row = json.loads(line)
                    if row.get("sample_id", row.get("id")) == args.sample_id:
                        print(name, json.dumps(row, ensure_ascii=False))
    elif args.command == "validate-submission":
        print(json.dumps(validate_file(args.data_root.resolve(), args.split, args.file, args.allow_partial), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
