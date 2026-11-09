"""Deterministic TRAIN-only 150 random + 50 difficult sample IDs, frozen in configs/splits/validation_ids.json.
Writes the file if absent; otherwise checks that the data still yields the same ids and exits 1 if not."""
from __future__ import annotations

import json
import random
import sys
import wave
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import DATA_ROOT, VALIDATION_IDS, audio_path, items  # noqa: E402

root = DATA_ROOT
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
text = json.dumps(chosen, indent=2) + "\n"
if VALIDATION_IDS.is_file():
    if VALIDATION_IDS.read_text() != text:
        sys.exit(f"{VALIDATION_IDS} differs from the ids this data yields; the frozen split is kept, check the data version")
    print(f"{len(chosen)} TRAIN IDs match {VALIDATION_IDS}")
else:
    VALIDATION_IDS.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_IDS.write_text(text)
    print(f"{len(chosen)} TRAIN IDs -> {VALIDATION_IDS} (commit it)")
