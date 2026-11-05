"""Transparent TRAIN-only ASR metrics; no claimed official task score."""
from __future__ import annotations

import re
import json
from collections import Counter


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def error_counts(reference: str, prediction: str) -> dict[str, int]:
    ref, pred = normalized(reference), normalized(prediction)
    words_ref, words_pred = ref.split(), pred.split()
    chars_ref, chars_pred = list(ref.replace(" ", "")), list(pred.replace(" ", ""))
    return {
        "word_errors": edit_distance(words_ref, words_pred),
        "reference_words": len(words_ref),
        "character_errors": edit_distance(chars_ref, chars_pred),
        "reference_characters": len(chars_ref),
    }


def exact_tool_f1(predicted: list[dict], gold: list[dict]) -> float:
    """Strict exact-call surrogate; this is not the organizer's fuzzy scorer."""
    if not predicted and not gold:
        return 1.0
    def key(call: dict) -> str:
        return json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False, sort_keys=True)
    matched = sum((Counter(map(key, predicted)) & Counter(map(key, gold))).values())
    return 2 * matched / (len(predicted) + len(gold)) if predicted or gold else 1.0
