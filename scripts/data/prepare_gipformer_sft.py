"""Leakage-free LifeMate TRAIN audio/text pairs for Gipformer adaptation.

    .venv/bin/python scripts/data/prepare_gipformer_sft.py --output-dir outputs/asr_sft/gipformer_data_v2
    .venv/bin/python scripts/data/prepare_gipformer_sft.py --output-dir outputs/asr_sft/gipformer_data_full --final

Default: the 200 frozen validation samples (and every sample sharing a recording with them) are left out, and the rest is split
into train/dev by connected components of shared audio. --final (the released ASR): train = that train + dev + the left-out
recordings (all 5,036 TRAIN recordings); dev stays the same 518 recordings, used only as a loss monitor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import audio_path, items  # noqa: E402


def pairs(item: dict) -> list[tuple[str, str]]:
    result = [(item["audio"], item["audio_text"])]
    result.extend((turn["audio"], turn["audio_text"]) for turn in item["history"]
                  if turn.get("role") == "user" and "audio" in turn and isinstance(turn.get("audio_text"), str))
    return result


def all_audio(item: dict) -> set[str]:
    return {item["audio"], *(turn["audio"] for turn in item["history"] if "audio" in turn)}


def prepare(output_dir: Path, validation_ids: Path, final: bool = False) -> dict:
    root = PROJECT / "data/train+public_test"
    forbidden = set(json.loads(validation_ids.read_text(encoding="utf-8")))
    if len(forbidden) != 200:
        raise ValueError("frozen validation split must have 200 unique IDs")
    rows = list(items(root, "train"))
    validation_audio = {audio for item in rows if item["id"] in forbidden for audio in all_audio(item)}
    selected = [item for item in rows if item["id"] not in forbidden
                and not (all_audio(item) & validation_audio)]
    # Tie samples that share a history recording into the same internal split.
    parents = {item["id"]: item["id"] for item in selected}
    def find(x: str) -> str:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(a: str, b: str) -> None:
        parents[find(a)] = find(b)
    owner: dict[str, str] = {}
    for item in selected:
        for audio in all_audio(item):
            if audio in owner:
                union(item["id"], owner[audio])
            owner[audio] = item["id"]
    components: dict[str, list[dict]] = {}
    for item in selected:
        components.setdefault(find(item["id"]), []).append(item)
    produced: dict[str, dict[str, dict]] = {"train": {}, "dev": {}}
    for component in components.values():
        signature = min(item["id"] for item in component)
        bucket = "dev" if int(hashlib.sha256(f"gipformer:42:{signature}".encode()).hexdigest(), 16) % 10 == 0 else "train"
        for item in component:
            for audio, transcript in pairs(item):
                audio_path(root, "train", audio)
                previous = produced[bucket].get(audio)
                if previous and previous["text"] != transcript:
                    raise ValueError(f"conflicting TRAIN transcripts for {audio}")
                produced[bucket][audio] = {"sample_id": item["id"], "audio": audio, "text": transcript}
    if not produced["train"] or not produced["dev"]:
        raise ValueError("internal train/dev split is empty")
    if set(produced["train"]) & set(produced["dev"]):
        raise AssertionError("train/dev audio leakage")
    if final:   # every recording: train, dev (also trained), then the validation-side samples in file order
        everything = {**produced["train"], **produced["dev"]}
        for item in rows:
            if item["id"] in forbidden or all_audio(item) & validation_audio:
                for audio, transcript in pairs(item):
                    everything.setdefault(audio, {"sample_id": item["id"], "audio": audio, "text": transcript})
        produced["train"] = everything
    output_dir.mkdir(parents=True, exist_ok=False)
    for bucket in ("train", "dev"):
        with (output_dir / f"{bucket}.jsonl").open("w", encoding="utf-8") as out:
            for row in produced[bucket].values():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"source": "official LifeMate TRAIN only", "validation_samples_excluded": len(forbidden),
                "validation_ids_sha256": hashlib.sha256(validation_ids.read_bytes()).hexdigest(),
                "selected_samples": len(selected), "train_audio": len(produced["train"]),
                "dev_audio": len(produced["dev"]), "validation_audio_excluded": len(validation_audio),
                "split_rule": "connected components of shared audio; 10% hash dev"}
    if final:
        manifest = {"source": "official LifeMate TRAIN only (final: train + dev + validation audio)", "validation_samples_excluded": 0,
                    "train_audio": len(produced["train"]), "dev_audio_monitor_only": len(produced["dev"]),
                    "validation_audio_included": len(validation_audio),
                    "note": "dev rows are also trained; dev loss is a sanity monitor only; the last checkpoint is the model"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, default=PROJECT / "configs/splits/validation_ids.json")
    parser.add_argument("--final", action="store_true", help="released ASR: train on every TRAIN recording (validation included)")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.validation_ids, args.final), indent=2))
