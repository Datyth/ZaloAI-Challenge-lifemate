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


# ---- v3 targets (2026-11-09): every digit token of the reference gets candidate spoken readings of ITS OWN value, and the
# candidate closest (word edit distance) to the aligned span of the original Gipformer transcript wins. v2 copied that span
# whenever it was made of number words, so a misheard number became the label (>= 7 rows), and rows whose span had other
# words were dropped (112 of 3,756 TRAIN audios, i.e. most number-heavy utterances were never trained).
_U = ["KHÔNG", "MỘT", "HAI", "BA", "BỐN", "NĂM", "SÁU", "BẢY", "TÁM", "CHÍN"]


def _spell_up(n: int) -> str:
    """Vietnamese reading of 0 <= n < 10^9 in upper case (same rule as prepare_agent_sft_v2._spell, copied so the ASR venv
    does not import the agent package)."""
    if n == 0:
        return "KHÔNG"
    out: list[str] = []
    for scale, word in ((10**6, "TRIỆU"), (1000, "NGHÌN"), (1, "")):
        k, n = divmod(n, scale)
        if not k:
            continue
        h, r = divmod(k, 100)
        t, u = divmod(r, 10)
        part = []
        if h or out:
            part += [_U[h], "TRĂM"]
        if t == 0 and u and (h or out):
            part.append("LINH")
        if t == 1:
            part.append("MƯỜI")
        elif t:
            part += [_U[t], "MƯƠI"]
        if u:
            part.append("MỐT" if u == 1 and t > 1 else "LĂM" if u == 5 and t else "TƯ" if u == 4 and t > 1 else _U[u])
        out += part + ([word] if word else [])
    return " ".join(out)


def _int_variants(n: int) -> set[str]:
    std = _spell_up(n)
    out = {std, std.replace("NGHÌN", "NGÀN"), std.replace("LINH", "LẺ")}
    if 21 <= n <= 99 and n % 10:
        t, u = divmod(n, 10)
        last = {1: "MỐT", 4: "TƯ", 5: "LĂM"}.get(u, _U[u])
        out |= {f"{_U[t]} {last}", f"{_U[t]} {_U[u]}"}
    if 100 <= n <= 999:
        h, r = divmod(n, 100)
        t, u = divmod(r, 10)
        if r and u == 0:
            out |= {f"{_U[h]} TRĂM {_U[t] if t > 1 else 'MƯỜI'}", f"TRĂM {_U[t]}" if h == 1 and t > 1 else std}
        if t > 1 and u:
            last = {1: "MỐT", 4: "TƯ", 5: "LĂM"}.get(u, _U[u])
            out.add(f"{_U[h]} TRĂM {_U[t]} {last}")
    for scale, word in ((10**6, "TRIỆU"), (1000, "NGHÌN")):
        if n >= scale and n % scale and n % (scale // 10) == 0:
            a, b = divmod(n, scale)
            d = b // (scale // 10)
            for w in (word, "NGÀN" if word == "NGHÌN" else word):
                out.add(f"{_spell_up(a)} {w} {_U[d]}")
                if d == 5:
                    out.add(f"{_spell_up(a)} {w} RƯỠI")
    return {" ".join(x.split()) for x in out}


def _token_variants(tok: str) -> list[str] | None:
    """Spoken readings of one reference token that contains digits (already upper case, punctuation split off)."""
    t = tok.upper()
    m = re.fullmatch(r"(\d+)", t)
    if m:
        d = m.group(1)
        digits = " ".join(_U[int(c)] for c in d)
        if (d.startswith("0") and len(d) > 1) or (len(d) >= 9 and int(d) % 1000):
            return [digits]                                   # phone numbers, codes: digit by digit
        return sorted(_int_variants(int(d))) + ([digits] if len(d) >= 4 else [])
    m = re.fullmatch(r"(\d+)K", t)
    if m:
        base = _int_variants(int(m.group(1)))
        return sorted({f"{b} {w}" for b in base for w in ("NGHÌN", "NGÀN", "CA")})
    m = re.fullmatch(r"(\d+)H(\d*)", t)
    if m:
        h, mi = int(m.group(1)), m.group(2)
        hs = _int_variants(h)
        if not mi or int(mi) == 0:
            return sorted({f"{x} GIỜ" for x in hs} | {f"{x} GIỜ ĐÚNG" for x in hs})
        ms = _int_variants(int(mi))
        out = {f"{x} GIỜ {y}" for x in hs for y in ms} | {f"{x} GIỜ {y} PHÚT" for x in hs for y in ms}
        if int(mi) == 30:
            out |= {f"{x} GIỜ RƯỠI" for x in hs} | {f"{x} RƯỠI" for x in hs}
        return sorted(out)
    m = re.fullmatch(r"([A-ZĐ]+)(\d+)", t) or re.fullmatch(r"(\d+)([A-ZĐ]+)", t)
    if m:
        letters = m.group(1) if not m.group(1).isdigit() else m.group(2)
        num = m.group(2) if not m.group(1).isdigit() else m.group(1)
        nums = _int_variants(int(num)) | {" ".join(_U[int(c)] for c in num)}
        if m.group(1).isdigit():
            return sorted({f"{x} {letters}" for x in nums})
        return sorted({f"{letters} {x}" for x in nums})
    return None


def _wdist(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _join_numeric(ref: list[str], raw: str) -> list[str]:
    """Rejoin what _plain split apart in the raw reference: thousands separators (2.500.000), times (7:30) and dates (15/11)."""
    text = unicodedata.normalize("NFC", raw).upper()
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)                  # 2.500.000 -> 2500000
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1H\2", text)               # 7:30 -> 7H30
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b", r"\1 THÁNG \2", text)   # 15/11 -> 15 THÁNG 11
    text = re.sub(r"\b(\d+),(\d)\b", r"\1 PHẨY \2", text)                  # 1,5 -> 1 PHẨY 5
    return _plain(text)


def target_v3(reference: str, hypothesis: str) -> str | None:
    ref = _join_numeric(_plain(reference), reference)
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
        options: list[list[str]] = [[]]
        for w in seg:
            if any(c.isdigit() for c in w):
                v = _token_variants(w)
                if not v:
                    return None
                options = [o + x.split() for o in options for x in v][:400]
            else:
                options = [o + [w] for o in options]
        spoken = hyp[j1:j2] if op == "replace" else []
        best = min(options, key=lambda o: (_wdist(o, spoken), len(o)))
        out.extend(best)
    text = " ".join(out)
    return text if not any(c.isdigit() for c in text) else None
