"""Shared pieces for synthetic tasks: call helper, TRAIN-app states used in cross-app tasks."""
from __future__ import annotations

import copy


def C(_tool: str, **arguments) -> dict:
    return {"name": _tool, "arguments": arguments}


def ASK(tool: str, *params: str) -> dict:
    return C("sys.ask_missing_info", tool_name=tool, params=list(params))


def T(name, time, text, gold, states, **extra):
    return {"name": name, "time": time, "text": text, "gold": gold, "states": copy.deepcopy(states), **extra}


CONTACTS = {"contacts": [
    {"id": "c_01", "name": "Chị Hà", "relation": "chị gái", "phone": "0912345678"},
    {"id": "c_02", "name": "Anh Khoa", "relation": "đồng nghiệp", "phone": "0987654321", "email": "khoa.nguyen@congty.vn"},
    {"id": "c_03", "name": "Mẹ", "relation": "mẹ", "phone": "0903111222"},
    {"id": "c_04", "name": "Bé Na", "relation": "con gái", "phone": "0934555666"},
    {"id": "c_05", "name": "Thầy Long", "relation": "giáo viên", "phone": "0977888999", "email": "long.tran@truong.edu.vn"},
    {"id": "c_06", "name": "Linh", "relation": "bạn", "phone": "0965432109", "email": "linh.pham@mail.vn"},
    {"id": "c_07", "name": "Anh Tuấn", "relation": "hàng xóm", "phone": "0944123123"},
]}
SMS = {"messages": []}
REMINDERS = {"reminders": []}
CALENDAR = {"events": []}
