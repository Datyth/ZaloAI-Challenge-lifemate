"""Thin bridge to the exact split-specific organizer runtime."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def new_env(root: Path, split: str, sample_id: str):
    # One split per process. The two kit copies expose the same package name.
    package_dir = str((root / split).resolve())
    loaded = sys.modules.get("zaic_tools")
    if loaded is not None and not str(Path(loaded.__file__).resolve()).startswith(package_dir):
        raise RuntimeError("Run each split in a separate Python process")
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    module = importlib.import_module("zaic_tools")
    return module.ToolEnv(split, sample_id, root=root)
