"""Read safetensors metadata without loading weights."""
from __future__ import annotations

import json
import math
from pathlib import Path


def stored_parameters(directory: Path) -> tuple[int | None, int]:
    """Return tensor element count and weight bytes; count is not an architecture audit."""
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        return None, 0
    count = 0
    size = 0
    for path in files:
        size += path.stat().st_size
        with path.open("rb") as source:
            header_len = int.from_bytes(source.read(8), "little")
            if header_len <= 0 or header_len > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(source.read(header_len))
        count += sum(math.prod(tensor["shape"]) for key, tensor in header.items() if key != "__metadata__")
    return count, size
