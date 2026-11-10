"""Deterministic helpers added to the agent's view of a sample (orchestration S2); used identically for SFT data and inference."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

WEEKDAY = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def date_reference(current_time: str) -> dict:
    """Weekday names for yesterday .. +13 days and the Monday-Sunday ranges of this and next week.
    A 4B model often misplaces "thứ ba tuần sau" or "thứ sáu" by a day; the table removes that arithmetic."""
    today = datetime.fromisoformat(current_time).date()
    def label(d: date) -> str:
        tag = {-1: " (hôm qua)", 0: " (hôm nay)", 1: " (mai)", 2: " (ngày kia)"}.get((d - today).days, "")
        return f"{d.isoformat()} {WEEKDAY[d.weekday()]}{tag}"
    monday = today - timedelta(days=today.weekday())
    return {"days": [label(today + timedelta(days=k)) for k in range(-1, 14)],
            "this_week": f"{monday.isoformat()} .. {(monday + timedelta(days=6)).isoformat()}",
            "next_week": f"{(monday + timedelta(days=7)).isoformat()} .. {(monday + timedelta(days=13)).isoformat()}"}


def normalize_asr(text: str) -> str:
    """Gipformer writes upper case without punctuation; the agent reads lower-case text (public 30.77 -> 38.80 with this alone)."""
    return " ".join(text.lower().split())


_ID_LIKE = re.compile(r"^[a-z]+_[0-9a-z_]+$")


def hotword_phrases(item: dict, limit: int = 80) -> list[str]:
    """Names and titles the speaker may say, taken only from what inference already sees before decoding: the user profile and
    the history's tool calls and results (contacts, events, programmes, notes ...). Upper case, for Gipformer's BPE vocabulary.
    Full short values that carry a capital letter, plus runs of capitalised words inside longer text ("gặp Peter Hansen góp ý")."""
    strings: list[str] = []

    def walk(v) -> None:
        if isinstance(v, str):
            strings.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk((item.get("context") or {}).get("user_profile") or {})
    for turn in item.get("history") or []:
        walk(turn.get("tool_results") or [])
        for call in turn.get("tool_calls") or []:
            walk(call.get("arguments") or {})
    out: dict[str, None] = {}
    for s in strings:
        s = " ".join(re.sub(r"[^\w\s]", " ", s).split())
        if not s or _ID_LIKE.match(s) or re.search(r"\d", s):
            continue
        words = s.split()
        if len(words) <= 4 and any(w[0].isupper() for w in words):
            out[s.upper()] = None
        run: list[str] = []
        for w in words + [""]:
            if w and w[0].isupper():
                run.append(w)
                continue
            if len(run) >= 2 or (len(run) == 1 and len(run[0]) >= 4 and run[0].isascii()):
                out[" ".join(run).upper()] = None
            for x in run if len(run) >= 2 else []:      # foreign names alone too ("Northwind" of "Northwind Trading Cup")
                if len(x) >= 5 and x.isascii():
                    out[x.upper()] = None
            run = []
    return list(out)[:limit]
