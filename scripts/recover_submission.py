"""Merge successful rows from an interrupted run with targeted retry outputs.

Only failed sample IDs from the run's predictions log may be replaced. Every
input and the final result are replayed with the official ToolEnv validator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from lifemate.cli import DEFAULT_ROOT
from lifemate.submission import validate_file


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recover(run_dir: Path, partial: Path, replacements: list[str], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    original = read_jsonl(partial)
    if [row["id"] for row in predictions] != [row["id"] for row in original]:
        raise ValueError("prediction IDs or order differ from partial submission")
    failed_ids = {row["id"] for row in predictions if row.get("status") == "error"}
    if not failed_ids:
        raise ValueError("run has no failed samples to replace")
    if any(row.get("status") not in {"ok", "error"} for row in predictions):
        raise ValueError("run contains an unknown prediction status")
    validate_file(DEFAULT_ROOT, "public_test", partial)

    replacement_rows: dict[str, dict] = {}
    for item in replacements:
        sample_id, separator, filename = item.partition("=")
        if not separator or not sample_id or not filename:
            raise ValueError(f"replacement must be SAMPLE_ID=FILE: {item}")
        if sample_id in replacement_rows:
            raise ValueError(f"duplicate replacement: {sample_id}")
        path = Path(filename)
        rows = read_jsonl(path)
        if len(rows) != 1 or rows[0].get("id") != sample_id:
            raise ValueError(f"replacement must contain exactly one row for {sample_id}: {path}")
        validate_file(DEFAULT_ROOT, "public_test", path, allow_partial=True)
        replacement_rows[sample_id] = rows[0]
    if set(replacement_rows) != failed_ids:
        raise ValueError(f"must replace exactly failed IDs {sorted(failed_ids)}; got {sorted(replacement_rows)}")

    merged = [replacement_rows.get(row["id"], row) for row in original]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False) as stream:
            temp_name = stream.name
            for row in merged:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = validate_file(DEFAULT_ROOT, "public_test", Path(temp_name))
        if output.exists():
            raise FileExistsError(f"output appeared while validating: {output}")
        os.replace(temp_name, output)
        return result
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--replacement", action="append", required=True, metavar="SAMPLE_ID=FILE")
    parser.add_argument("--output", type=Path, default=Path("result/submission.jsonl"))
    args = parser.parse_args()
    print(json.dumps(recover(args.run_dir, args.partial, args.replacement, args.output), indent=2))


if __name__ == "__main__":
    main()
