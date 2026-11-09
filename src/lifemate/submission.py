"""Validate a JSONL by replaying calls through the official ToolEnv API."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import items
from .toolenv import new_env


def shard_dirs(prefix: Path) -> list[Path]:
    """Run directories of a run prefix: <prefix> itself, else its shards <prefix>_s0, _s1, ... (written by run_shards.sh).
    Only exact shard names match, so prefix `pub` never picks up the shards of `pub_synth`."""
    if prefix.is_dir():
        return [prefix]
    shard = re.compile(re.escape(prefix.name) + r"_s\d+")
    return sorted((d for d in prefix.parent.glob(prefix.name + "_s*") if shard.fullmatch(d.name)), key=lambda d: int(d.name.rsplit("_s", 1)[1]))


def validate_file(root: Path, split: str, path: Path, allow_partial: bool = False) -> dict:
    expected = [item["id"] for item in items(root, split)]
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or any(sample_id not in expected for sample_id in ids):
        raise ValueError("submission has duplicate or unknown sample IDs")
    if not allow_partial and ids != expected:
        raise ValueError("submission IDs or order differ from full split")
    if allow_partial and ids != [sample_id for sample_id in expected if sample_id in set(ids)]:
        raise ValueError("partial submission order differs from split order")
    calls = 0
    for row in rows:
        if set(row) != {"id", "tool_calls", "state_hash"} or not isinstance(row["tool_calls"], list) or not isinstance(row["state_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["state_hash"]):
            raise ValueError(f"invalid row shape: {row.get('id')}")
        env = new_env(root, split, row["id"])
        for call in row["tool_calls"]:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["arguments"], dict):
                raise ValueError(f"invalid call shape: {row['id']}")
            validation = env.validate(call["name"], call["arguments"])
            if not validation["ok"]:
                raise ValueError(f"schema error in {row['id']}: {validation['error']}")
            env.call(call["name"], call["arguments"])
            calls += 1
        if env.get_tool_calls() != row["tool_calls"] or env.get_state_hash() != row["state_hash"]:
            raise ValueError(f"ToolEnv replay mismatch: {row['id']}")
    return {"rows": len(rows), "calls": calls, "replay_hash_matches": len(rows)}
