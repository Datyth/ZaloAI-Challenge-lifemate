"""Gipformer fine-tuning targets from TRAIN audio_text, in the pretrained BPE's form (upper case, no punctuation).

Digits: the reference writes "45 phút", "218k", "vtv1"; the audio says words, and the way they are read ("bốn lăm" or "bốn mươi
lăm") is not knowable from the text. The original Gipformer transcript of the same audio already spells the number, so a digit
span of the reference takes the aligned span of that transcript when the latter is made only of Vietnamese number words.
Rows whose digits cannot be resolved this way, or whose target has an unknown BPE piece, are skipped."""
from __future__ import annotations

import difflib
import re
import unicodedata

NUMBER_WORDS = {"KHÔNG", "MỘT", "MỐT", "HAI", "BA", "BỐN", "TƯ", "NĂM", "LĂM", "SÁU", "BẢY", "BẨY", "TÁM", "CHÍN", "MƯỜI", "MƯƠI",
                "TRĂM", "NGHÌN", "NGÀN", "TRIỆU", "TỶ", "TỈ", "LINH", "LẺ", "RƯỠI", "CHỤC", "PHẨY", "CHẤM"}


def _plain(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).upper()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return text.split()


def target(reference: str, hypothesis: str) -> str | None:
    """Upper-case target for one recording, or None when a digit span cannot be spelled from the hypothesis."""
    ref = _plain(reference)
    if not any(any(c.isdigit() for c in w) for w in ref):
        return " ".join(ref)
    hyp = _plain(hypothesis)
    out: list[str] = []
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg = ref[i1:i2]
        if not any(any(c.isdigit() for c in w) for w in seg):
            out.extend(seg)
            continue
        if op != "replace":
            return None
        # split mixed tokens like "VTV1" / "RON95": keep the letters, take the spoken number from the hypothesis
        letters = [re.sub(r"\d+", "", w) for w in seg]
        spoken = hyp[j1:j2]
        prefix = [w for w in letters if w and not any(c.isdigit() for c in w)]
        rest = [w for w in spoken if w not in prefix]
        if not rest or not all(w in NUMBER_WORDS or w in {"K", "GIỜ", "PHÚT", "ĐỘ", "PHẦN", "TRĂM"} for w in rest):
            return None
        out.extend(prefix + rest)
    text = " ".join(out)
    return text if not any(c.isdigit() for c in text) else None
