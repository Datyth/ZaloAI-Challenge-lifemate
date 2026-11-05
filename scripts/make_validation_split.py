"""Deterministic TRAIN-only 150 random + 50 difficult sample IDs."""
from __future__ import annotations

import json
import random
import sys
import wave
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import audio_path, items  # noqa: E402

root = PROJECT / "data" / "train+public_test"
rows = list(items(root, "train"))
rng = random.Random(42)
random_ids = {x["id"] for x in rng.sample(rows, min(150, len(rows)))}
difficult = []
for item in rows:
    if item["id"] in random_ids:
        continue
    with wave.open(str(audio_path(root, "train", item["audio"])), "rb") as wav:
        short = wav.getnframes() / wav.getframerate() < 2
    multi_step = len(item["label"]["tool_calls"]) > 1
    missing = any(c["name"] == "sys.ask_missing_info" for c in item["label"]["tool_calls"])
    if short or multi_step or missing:
        difficult.append(item["id"])
rng.shuffle(difficult)
chosen = sorted(random_ids | set(difficult[:50]))
out = PROJECT / "outputs" / "splits" / "validation_ids.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(chosen, indent=2) + "\n")
print(f"{len(chosen)} TRAIN IDs -> {out}")
