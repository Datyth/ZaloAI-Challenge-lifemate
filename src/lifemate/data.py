"""Read only organizer inputs; never inspect environment directories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "data" / "train+public_test"
# Frozen 200 TRAIN ids used for every validation number; excluded (with their shared history audio) from all training data.
VALIDATION_IDS = PROJECT / "configs" / "splits" / "validation_ids.json"


def items(root: Path, split: str) -> Iterator[dict]:
    if not split.replace("_", "").isalnum():
        raise ValueError(f"unsupported split: {split}")
    path = root / split / f"{split}.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def audio_path(root: Path, split: str, relative: str) -> Path:
    base = (root / split / "audio").resolve()
    path = (root / split / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"invalid or missing audio path: {relative}")
    return path


def catalog(root: Path, split: str) -> dict[str, dict]:
    path = root / split / f"{split}_tools_catalog.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    # "_app": the app's Vietnamese name and description (users name apps that way: "app đi chợ"); read by agent.user_message (E1P)
    return {t["name"]: {**t, "_app": {"id": app["id"], "name": app.get("name_vi", ""), "description": app.get("description", "")}}
            for app in raw["apps"] for t in app["tools"]}


def installed_tools(item: dict, specs: dict[str, dict]) -> list[dict]:
    names = {n for app in item["context"]["installed_apps"].values() for n in app["tools"]}
    names.add("sys.ask_missing_info")
    return [specs[n] for n in sorted(names) if n in specs]
