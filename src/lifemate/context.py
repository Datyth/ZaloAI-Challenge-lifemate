"""Deterministic helpers added to the agent's view of a sample (orchestration S2); used identically for SFT data and inference."""
from __future__ import annotations

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
