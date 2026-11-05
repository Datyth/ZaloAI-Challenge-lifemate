"""TRAIN-only ASR diagnostics; gold tool calls are never inference inputs."""
from __future__ import annotations

import re
from collections.abc import Iterable

from .metrics import normalized


CRITICAL_KEYS = {
    "query", "name", "recipient", "recipient_name", "contact", "contact_name",
    "title", "content", "message", "text", "location", "destination",
    "address", "amount", "date", "time", "start", "end", "phone", "phone_number",
}


def _leaves(value: object, key: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _leaves(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _leaves(child, key)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield key, value


def critical_slots(reference: str, gold_calls: list[dict]) -> list[dict]:
    """Keep only gold argument values explicitly spoken in the TRAIN reference.

    This is a strict lexical proxy. Derived dates, IDs and paraphrases are not
    scored, so the eligible-slot count must be reported with accuracy.
    """
    reference_words = normalized(reference).split()
    found: set[tuple[str, str]] = set()
    slots: list[dict] = []
    for call in gold_calls:
        for key, value in _leaves(call.get("arguments", {})):
            if key not in CRITICAL_KEYS:
                continue
            phrase = normalized(str(value)).split()
            if not phrase or len(phrase) > len(reference_words):
                continue
            if not any(reference_words[i:i + len(phrase)] == phrase for i in range(len(reference_words) - len(phrase) + 1)):
                continue
            signature = (key, " ".join(phrase))
            if signature not in found:
                found.add(signature)
                slots.append({"key": key, "value": str(value), "normalized": signature[1]})
    return slots


def slot_hits(slots: list[dict], transcript: str) -> list[dict]:
    words = normalized(transcript).split()
    return [{**slot, "matched": any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))}
            for slot in slots for phrase in [slot["normalized"].split()]]


def labelled_slices(item: dict, audio: dict) -> list[str]:
    labels: list[str] = []
    # The frozen TRAIN validation set has no utterances below 2 s; 4 s
    # captures its shortest 20/200 clips and is stated explicitly in reports.
    if audio["duration_seconds"] < 4:
        labels.append("short_under_4s")
    if "quiet_audio_heuristic" in audio.get("warnings", []):
        labels.append("quiet_proxy")
    if "clipped_audio_heuristic" in audio.get("warnings", []):
        labels.append("clipped_proxy")
    metadata = item.get("metadata") or item.get("labels") or {}
    if isinstance(metadata, dict):
        accent = metadata.get("accent") or metadata.get("region")
        if isinstance(accent, str) and accent.strip():
            labels.append("accent:" + re.sub(r"\s+", "_", accent.strip().casefold()))
        if metadata.get("noisy") is True or metadata.get("noise") is True:
            labels.append("noisy_labelled")
        if metadata.get("truncated") is True:
            labels.append("truncated_labelled")
    return labels
