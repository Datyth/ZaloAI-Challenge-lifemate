"""Inverse text normalisation for Gipformer output: lower case, spoken Vietnamese numbers to digits.

Gipformer writes upper case without punctuation and spells numbers ("MỘT TRĂM HAI MƯƠI HAI TRIỆU", "BỐN LĂM PHÚT"), while the
agent was trained on references that write digits ("122 triệu", "45 phút"). Only number spans are rewritten; a lone ambiguous
word ("năm" = year, "ba" = father, "tư", "một") is kept unless a unit or a time word makes it a number.
  normalize("CHUYỂN HAI TRĂM NĂM MƯƠI NGHÌN CHO BA") -> "chuyển 250 nghìn cho ba"
"""
from __future__ import annotations

import re

DIGIT = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7,
         "bẩy": 7, "tám": 8, "chín": 9, "linh": None, "lẻ": None}
TEN = {"mười": 10}
SCALE = {"trăm": 100, "nghìn": 1000, "ngàn": 1000, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
NUMWORD = set(DIGIT) | set(TEN) | set(SCALE) | {"mươi", "chục"}
# words after which a single number word really is a quantity, a time or a position
UNIT = {"giờ", "phút", "tiếng", "giây", "ngày", "tuần", "tháng", "chỗ", "người", "cái", "chiếc", "lít", "độ", "phần", "lần",
        "tầng", "km", "cây", "mét", "ký", "kg", "hộp", "gói", "phần", "suất", "tô", "ly", "vé", "tờ", "bộ", "bao", "cuốn", "quyển",
        "viên", "trận", "tập", "buổi", "đêm", "năm", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "k", "đô", "yên", "rưỡi", "kém", "xe",
        "đơn", "món", "căn", "phòng", "lượt", "bước", "điểm", "sao", "chai", "lon", "cốc", "kiện", "thùng", "cân", "giải"}
PRE = {"lúc", "số", "mức", "khoảng", "tầm", "thứ", "ngày", "tháng", "kênh", "vtv", "htv", "ron", "tập", "phòng", "bàn", "tầng",
       "trước", "sau", "đến", "tới", "từ", "còn", "dưới", "trên", "hơn", "giảm", "tăng", "lên", "xuống", "chuyển", "nạp", "trả",
       "mất", "hết", "được", "là", "bằng", "nhiệt", "âm", "lượng", "đúng", "cỡ", "hẹn", "dời", "lùi"}


def _value(words: list[str]) -> int | None:
    """Value of a spoken number span, or None when the span is not a well-formed number."""
    total, current, last = 0, 0, None
    for w in words:
        if w in SCALE:
            s = SCALE[w]
            if s == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * s
                current = 0
            last = "scale"
        elif w in ("mươi", "chục"):
            current = (current % 10 or 1) * 10 + (current - current % 10)
            last = "tens"
        elif w in TEN:
            current += 10
            last = "ten"
        elif DIGIT.get(w) is not None:
            d = DIGIT[w]
            if last == "digit" and current % 10 in range(2, 10) and w in ("lăm", "mốt", "tư", "năm", "một", "hai", "ba", "bốn", "sáu", "bảy", "tám", "chín") and current < 10:
                current = current * 10 + d            # "bốn lăm" = 45, "hai sáu" = 26
            else:
                current += d
            last = "digit"
        elif w in ("linh", "lẻ"):
            last = "linh"
        else:
            return None
    return total + current


def _digits_run(words: list[str]) -> str | None:
    """'không chín tám ba hai ...' -> '0983...': four or more single digits read one by one (phone numbers, codes)."""
    if len(words) >= 4 and all(DIGIT.get(w) is not None and w not in ("lăm", "mốt", "tư") for w in words):
        return "".join(str(DIGIT[w]) for w in words)
    return None


def normalize(text: str) -> str:
    words = text.lower().split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if words[i] not in NUMWORD:
            out.append(words[i])
            i += 1
            continue
        j = i
        while j < len(words) and words[j] in NUMWORD:
            j += 1
        span = words[i:j]
        # two bare digits ("ba một chút") are two words unless they read as a tens shorthand ("bốn lăm", "hai sáu độ")
        if (len(span) == 2 and all(DIGIT.get(w) is not None for w in span) and span[1] not in ("lăm", "mốt", "tư")
                and (words[j] if j < len(words) else "") not in UNIT):
            j = i + 1
            span = words[i:j]
        # keep a trailing scale word as text when the head is a number ("122 triệu", "250 nghìn")
        tail = []
        if len(span) > 1 and span[-1] in ("nghìn", "ngàn", "triệu", "tỷ", "tỉ") and not any(w in ("nghìn", "ngàn", "triệu", "tỷ", "tỉ") for w in span[:-1]):
            tail, span = [span[-1]], span[:-1]
        prev = out[-1] if out else ""
        nxt = words[j] if j < len(words) else ""
        phone = _digits_run(span)
        value = _value(span)
        lone = len(span) == 1 and not tail
        ambiguous = lone and span[0] in ("năm", "ba", "tư", "một", "không", "linh", "lẻ", "chín", "trăm", "nghìn", "ngàn", "triệu", "mươi", "chục", "lăm", "mốt")
        if phone:
            out.append(phone)
        elif value is None or (ambiguous and nxt not in UNIT and prev not in PRE) or (lone and span[0] in SCALE):
            out.extend(span)
        elif span[0] == "năm" and nxt in ("nay", "ngoái", "sau", "trước", "tới", "mới", "học"):
            out.extend(span)                           # "năm nay", "năm sau": the year, not five
        else:
            out.append(str(value))
        out.extend(tail)
        i = j
    return " ".join(out)
