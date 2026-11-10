"""Private-test inference inside the Docker image (organizers' guide, sections 5 and 10).

    python3 -m lifemate.docker_predict --data /data --out /result [--llm-url http://127.0.0.1:8000/v1/chat/completions --model q35_e1]

- The split is the folder of /data that holds <split>_tools_catalog.json; zaic_tools is imported from /data/<split>, never from a
  copy in the image, and the agent reads app data only through ToolEnv (one per sample, created once).
- Samples run one at a time, in file order: ASR of the current and history audio (Gipformer on CPU, same text as on GPU), then the
  agent loop against the local vLLM server that predict.sh started. Each line carries `time` in ms, measured from before the audio
  is read, as the guide's sample predict.py does. The file is flushed after every line, so a stop keeps every finished sample.
- A sample that fails still gets its line (the calls already made on its ToolEnv and that state's hash)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from .agent import execute
from .asr import GipformerASR
from .context import normalize_asr
from .data import audio_path, catalog, installed_tools, items
from .llm import LocalChat
from .toolenv import new_env


def find_split(root: Path) -> str:
    found = sorted(p.name for p in root.iterdir() if (p / f"{p.name}_tools_catalog.json").is_file())
    if len(found) != 1:
        raise SystemExit(f"expected one split folder with a tools catalog under {root}, found {found}")
    return found[0]


class Predictor:
    """Loaded resources plus one-sample prediction; used by main() and by predict_notebook.ipynb."""

    def __init__(self, data: Path, llm_url: str, model: str, asr_dir: Path, deadline: float, max_turns: int = 10) -> None:
        self.root = data.resolve()
        self.split = find_split(self.root)
        self.specs = catalog(self.root, self.split)
        self.asr = GipformerASR(asr_dir, device="cpu")
        ensemble = os.environ.get("LIFEMATE_ENSEMBLE", "").strip()
        if ensemble:   # step-level vote of several served LoRAs on the one ToolEnv (lifemate.ensemble); first member breaks ties
            from .ensemble import EnsembleChat
            self.llm = EnsembleChat(llm_url, [tuple(m.split(":", 1)) for m in ensemble.split(",")], timeout=deadline)
        else:
            self.llm = LocalChat(llm_url, model, timeout=deadline)
        self.deadline, self.max_turns = deadline, max_turns

    def samples(self) -> list[dict]:
        return list(items(self.root, self.split))

    def predict(self, item: dict, log=lambda kind, record: None) -> dict:
        """One sample: ASR, agent loop on its single ToolEnv; returns the submission line (with time in ms)."""
        t1 = time.perf_counter()                            # before the audio is read
        env = new_env(self.root, self.split, item["id"])    # exactly one ToolEnv for this sample
        cache: dict[str, str] = {}

        def transcribe(relative: str) -> str:
            if relative not in cache:
                cache[relative] = normalize_asr(self.asr.transcribe(audio_path(self.root, self.split, relative)))
            return cache[relative]

        try:
            transcript = transcribe(item["audio"])
            view = {key: item[key] for key in ("id", "audio", "context", "history")}
            if hasattr(self.llm, "bind"):
                self.llm.bind(view, transcript, installed_tools(view, self.specs))
                self.llm.last_votes = []
            execute(view, transcript, installed_tools(view, self.specs), env, self.llm, transcribe, log, max_turns=self.max_turns,
                    deadline_seconds=self.deadline - (time.perf_counter() - t1), orchestration=os.environ.get("LIFEMATE_ORCHESTRATION", "E1"), dedup_calls=True,
                    retry_cap=int(os.environ.get("LIFEMATE_RETRY_CAP", "0")))
        except Exception as exc:  # the line is still written from the ToolEnv
            log("error", {"sample_id": item["id"], "error": f"{type(exc).__name__}: {exc}"})
        return {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(), "time": int((time.perf_counter() - t1) * 1000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/data"))
    ap.add_argument("--out", type=Path, default=Path("/result"))
    ap.add_argument("--llm-url", default=os.environ.get("LIFEMATE_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    ap.add_argument("--model", default=os.environ.get("LIFEMATE_SERVED_MODEL", "lifemate"))
    ap.add_argument("--asr-dir", type=Path, default=Path(os.environ.get("LIFEMATE_ASR_DIR", "/code/models/gipformer")))
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("LIFEMATE_DEADLINE", "27")), help="seconds per sample, ASR included")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--limit", type=int, help="first N samples only (smoke test)")
    a = ap.parse_args()
    random.seed(42)
    t_load = time.perf_counter()
    pred = Predictor(a.data, a.llm_url, a.model, a.asr_dir, a.deadline, a.max_turns)
    print(f"lifemate: split {pred.split}, ASR loaded in {time.perf_counter() - t_load:.1f}s, LLM {a.model} at {a.llm_url}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logs").mkdir(exist_ok=True)
    selected = pred.samples()[: a.limit] if a.limit else pred.samples()
    times = []
    with (a.out / "submission.jsonl").open("w", encoding="utf-8") as fout, (a.out / "logs/trace.jsonl").open("w", encoding="utf-8") as trace:
        def log(kind: str, record: dict) -> None:
            trace.write(json.dumps({"kind": kind, **record}, ensure_ascii=False, default=str) + "\n")
        for k, item in enumerate(selected, 1):
            row = pred.predict(item, log)
            times.append(row["time"])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if k % 50 == 0 or k == len(selected):
                print(f"lifemate: {k}/{len(selected)} samples, mean {sum(times) / len(times):.0f} ms, max {max(times)} ms", flush=True)
    print(f"lifemate: wrote {a.out / 'submission.jsonl'} ({len(selected)} lines)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
