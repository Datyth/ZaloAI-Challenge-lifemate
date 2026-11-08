"""Sequential tool loop with exact organizer observations."""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from .history import ids_in, outcome, structured_history, successful


PROMPT = """Bạn là trợ lý LifeMate. Chuyển lời nói mới nhất thành đúng các thao tác trên các ứng dụng đã cài.
Chỉ dùng tool được cung cấp. Đọc dữ liệu qua tool trước khi cần ID hoặc trạng thái. Không bịa ID, số tiền, người nhận,
địa chỉ hay phần lời nói bị cắt. Nếu thiếu thông tin bắt buộc sau khi tra cứu được phép, gọi sys.ask_missing_info.
Chỉ truyền tham số tùy chọn khi người dùng nói rõ hoặc khi công cụ thật sự cần; bỏ qua giá trị mặc định đã có trong bối cảnh.
Kết quả đọc không hoàn thành hành động. Nếu điều kiện đúng, tiếp tục gọi các tool ghi cho mọi hành động được yêu cầu.
"Dời lịch" cần tìm sự kiện bằng calendar.list_events rồi calendar.update_event với ID và thời gian thực tế.
"Nhắn [người]" cần tìm người qua contacts.search rồi gửi bằng sms.send nếu SMS đã cài; không thay bằng reminders.create.
Không bao giờ nói đã dời lịch, gửi tin, thanh toán hay tạo việc nếu tool ghi tương ứng chưa trả về thành công.
Trước mỗi tool ghi, xác định động từ và đối tượng tương ứng trong lời nói. Không điều khiển nhà thông minh chỉ vì có điều kiện thời tiết;
phải có yêu cầu nói rõ thiết bị, phòng, hoặc hành động với nhà. Nếu lời nhận dạng mơ hồ, không tự suy ra hành động khác.
Nếu yêu cầu ngoài khả năng các app đã cài, không gọi tool. Dừng ngay khi yêu cầu đã hoàn tất.
Các thao tác trong lịch sử đã được thực hiện; không chạy lại. Một lượt tiếng nói hiện tại là yêu cầu độc lập trừ khi
lịch sử thật sự giải quyết được tham chiếu. Phản hồi văn bản không được chấm điểm.
"""

E6_PROMPT = PROMPT + """Khi yêu cầu có nhiều bước, sau mỗi kết quả đọc hãy đối chiếu yêu cầu mới nhất với các thao tác đã thực sự thành công.
Nếu còn một hành động được yêu cầu nhưng chưa hoàn tất, dùng quan sát vừa nhận để tiếp tục bằng tool phù hợp.
Với yêu cầu có điều kiện, chỉ thực hiện nhánh mà dữ liệu quan sát xác nhận; nếu điều kiện không thể xác định, hỏi thông tin thiếu.
Không thực hiện thao tác ghi chỉ vì đã gọi tool đọc. Với yêu cầu chỉ hỏi thông tin, kết thúc sau khi đọc đủ.
"""

GENERAL_PROMPT = """Bạn là trợ lý LifeMate. Thực hiện yêu cầu hiện tại bằng các tool được cung cấp.
Lịch sử là dữ liệu chỉ đọc: thao tác đã hoàn tất trong lịch sử không phải yêu cầu mới.
Trước khi ghi, dùng kết quả tool đọc hoặc ID do người dùng cung cấp để xác định đúng đối tượng.
Theo dõi từng hành động được yêu cầu. Chỉ xem hành động hoàn tất khi kết quả tool xác nhận thành công;
ok=true, done=false không xác nhận hoàn tất. Với hành động phụ thuộc, chờ kết quả bước trước.
Nếu thiếu tham số bắt buộc sau khi tra cứu hợp lệ, dùng sys.ask_missing_info. Không bịa ID hoặc tham số.
Không lặp lại thao tác ghi đã thành công hoặc chưa rõ kết quả. Nếu chỉ hỏi thông tin, dừng sau khi đọc đủ.
Chỉ gọi native tool_calls; lời mô tả không thực thi tool. Dừng khi yêu cầu đã hoàn tất.
"""


S2_PROMPT = PROMPT + """Lời nói là bản nhận dạng giọng nói, có thể nghe nhầm từ hoặc tên riêng; hãy suy ra ý định hợp lý nhất từ ngữ cảnh và các app đã cài.
Dùng date_reference trong context để đổi "mai", "thứ ba tuần sau", "thứ sáu"... ra ngày cụ thể.
Khi người dùng nói trả hoặc chuyển bằng nguồn A, không đủ thì dùng nguồn B: đọc số dư trước, so với số tiền, rồi chỉ gọi thao tác ghi một lần với nguồn phù hợp.
Không tự điền giá trị người dùng không nói (giờ, ngày, số tiền, mức âm lượng, mục tiêu, đối tượng cụ thể); khi thiếu, dùng sys.ask_missing_info.
Khi tìm liên hệ, giữ nguyên danh xưng như trong lời nói (ví dụ "Sếp Dũng", "Vợ Hằng").
"""


def prompt_for(orchestration: str) -> str:
    if orchestration == "O5":
        return GENERAL_PROMPT
    if orchestration == "S2":
        return S2_PROMPT
    return E6_PROMPT if orchestration == "E6" else PROMPT


def user_message(item: dict, transcript: str, orchestration: str) -> dict:
    """The current request message; S2 adds a weekday table (lifemate.context.date_reference)."""
    context = {k: v for k, v in item["context"].items() if k != "installed_apps"}
    if orchestration == "S2":
        from .context import date_reference
        context["date_reference"] = date_reference(item["context"]["current_time"])
    return {"role": "user", "content": json.dumps({"request": transcript, "context": context}, ensure_ascii=False)}


def _ids_for_reference(result: object, reference: str) -> set[str]:
    """Use only a named collection in an observation, not unrelated IDs in the same app."""
    collection = reference.split(".", 1)[-1]
    singular = collection[:-1] if collection.endswith("s") else collection
    found: set[str] = set()
    if isinstance(result, dict):
        for key, value in result.items():
            if key in {collection, singular}:
                found.update(ids_in(value))
            elif isinstance(value, (dict, list)):
                found.update(_ids_for_reference(value, reference))
    elif isinstance(result, list):
        for value in result:
            found.update(_ids_for_reference(value, reference))
    return found


def _observation_for_model(result: object, limit: int = 16000) -> str:
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= limit:
        return serialized
    if not isinstance(result, dict):
        return json.dumps({"truncated": True, "type": type(result).__name__}, ensure_ascii=False)
    compact: dict = {key: (value[:1000] if isinstance(value, str) and len(value) > 1000 else value)
                     for key, value in result.items() if key in {"ok", "done", "error", "message", "waiting_for_user", "id"}}
    compact["truncated"] = True
    for key, value in result.items():
        if key in compact:
            continue
        if isinstance(value, list):
            compact[key] = []
            for entry in value:
                trial = {**compact, key: [*compact[key], entry]}
                if len(json.dumps(trial, ensure_ascii=False)) > limit:
                    break
                compact[key].append(entry)
            compact[f"{key}_total"] = len(value)
            compact[f"{key}_shown"] = len(compact[key])
        elif len(json.dumps({**compact, key: value}, ensure_ascii=False)) <= limit:
            compact[key] = value
    while len(json.dumps(compact, ensure_ascii=False)) > limit:
        largest = max((key for key, value in compact.items() if isinstance(value, list) and value), key=lambda key: len(compact[key]), default=None)
        if largest is None:
            break
        compact[largest].pop()
        compact[f"{largest}_shown"] = len(compact[largest])
    return json.dumps(compact, ensure_ascii=False)


def _explicit_write_request(transcript: str) -> bool:
    """Conservative trigger for one read-to-action completion check."""
    return bool(re.search(
        r"(?:^|[,.!?;]\s*|\b(?:giúp|hãy|nhờ)\s+)"
        r"(?:chuyển\s+tiền|trả\s+tiền|thanh\s+toán|nhắn|gửi\s+(?:tin|sms)|"
        r"dời|đổi\s+lịch|đặt\s+(?:lịch|báo\s+thức)|tạo|xóa|xoá|hủy|huỷ|bật|tắt|thêm)\b",
        transcript.strip(), re.I,
    ))


def _multiple_actions_requested(transcript: str) -> bool:
    verbs = re.findall(r"\b(?:chuyển\s+tiền|trả\s+tiền|thanh\s+toán|nhắn|gửi\s+(?:tin|sms)|dời|đổi\s+lịch|đặt\s+(?:lịch|báo\s+thức)|tạo|xóa|xoá|hủy|huỷ|bật|tắt|thêm)\b", transcript, re.I)
    return len(verbs) >= 2


def _tool_defs(specs: list[dict]) -> list[dict]:
    # OpenAI-compatible servers often restrict function names to letters, digits,
    # underscores and dashes; official names contain a period.
    return [{"type": "function", "function": {"name": s["name"].replace(".", "__"), "description": f"Official tool {s['name']}. {s.get('description', '')}", "parameters": s.get("parameters") or {"type": "object", "properties": {}}}} for s in specs]


def _history(history: list[dict], transcribe_audio: Callable[[str], str]) -> list[dict]:
    out: list[dict] = []
    for turn in history:
        role = turn.get("role")
        if role == "user":
            content = transcribe_audio(turn["audio"]) if "audio" in turn else turn.get("text", "")
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            content = turn.get("text", "")
            if turn.get("tool_calls"):
                content += "\nĐã gọi tool: " + json.dumps(turn["tool_calls"], ensure_ascii=False)
            if turn.get("tool_results"):
                content += "\nKết quả: " + json.dumps(turn["tool_results"], ensure_ascii=False)
            out.append({"role": "assistant", "content": content})
    return out


def _fake_narration(content: str) -> bool:
    """Detect protocol-looking text, without treating it as an executable call."""
    return bool(re.search(r"<tool_call>|</tool_call>|\"(?:name|tool_name)\"\s*:\s*\"[\w]+[.__][\w]+\"|(?:đã|sẽ)\s+(?:gọi|dùng|thực hiện)\s+(?:tool|công cụ)", content, re.I))


def _user_supplied_id(value: str, transcript: str, history_requests: list[dict]) -> bool:
    # A literal ID must occur as a complete token in user speech, not inside another ID.
    return any(re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", text, re.I)
               for text in [transcript, *(r["text"] for r in history_requests)])


def _contradicts_device_request(name: str, args: dict, transcript: str) -> bool:
    """Reject a device-on write when the current utterance explicitly forbids it."""
    device = {"smart_home.light_set": r"đèn", "smart_home.fan_set": r"quạt",
              "smart_home.climate_set": r"(?:điều hoà|điều hòa|máy lạnh)"}.get(name)
    if not device or args.get("power") != "on":
        return False
    return bool(re.search(rf"(?:đừng|không|chớ)\s+bật\s+{device}\b|{device}\b.{{0,30}}?\b(?:đừng|không|chớ)\s+bật\b", transcript, re.I))


def _alarm_time_unverified(transcript: str, observed_clock: bool) -> bool:
    """An absolute alarm time needs a spoken clock time or a current read observation."""
    if observed_clock or re.search(r"\b(?:giờ đó|lúc đó|như cũ|giờ cũ)\b", transcript, re.I):
        return False
    number = r"(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười(?:\s+(?:một|hai))?)"
    return not bool(re.search(rf"\b{number}\s*(?:giờ|h\b|:\d{{2}})|\blúc\s+{number}\b", transcript, re.I))


def _unreferenced_ids(spec: dict, args: dict) -> list[tuple[str, str]]:
    """(path, value) of ID-like string arguments that the schema gives no x-ref, at any depth (e.g. items[].item_id)."""
    props = (spec.get("parameters") or {}).get("properties", {})
    found: list[tuple[str, str]] = []
    def walk(value: object, path: str, depth: int) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                prop = props.get(key, {}) if depth == 0 else {}
                if (key == "id" or key.endswith("_id")) and isinstance(child, str) and not (isinstance(prop, dict) and prop.get("x-ref")):
                    found.append((path + key, child))
                walk(child, path + key + ".", depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, path.rstrip(".") + "[].", depth + 1)
    walk(args, "", 0)
    return found


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


_TO_PLACE = r"\b(?:về|ở|tới|đến|tại|qua|sang|giao)\s+(?:tận\s+)?"
# "giao về nhà", "ở nhà mình", "tới công ty": a place word after a preposition, not "nhà anh có bốn người"
_PROFILE_PLACE = {"home": _TO_PLACE + r"nhà\b", "work": _TO_PLACE + r"(?:công ty|cơ quan|văn phòng|chỗ làm|nơi làm|công sở)\b"}


def _context_values(context: dict) -> dict[str, str]:
    """Normalised string value -> its key path, for the user's context (profile, defaults), without app list and clock."""
    out: dict[str, str] = {}
    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for child in value:
                walk(child, path)
        elif isinstance(value, str) and len(value) >= 3:
            out.setdefault(_norm(value), path)
    walk({k: v for k, v in context.items() if k not in {"installed_apps", "current_time", "timezone"}}, "")
    return out


def _drop_context_defaults(spec: dict, args: dict, context: dict, speech: str) -> tuple[dict, dict]:
    """Drop optional write arguments that only copy a context value (home address, profile default) the user never referred to.
    TRAIN labels pass such a value 2 times in 1,545 samples, both when the user named the place ("ở nhà mình", "ở công ty")."""
    params = spec.get("parameters") or {}
    required = set(params.get("required") or [])
    props = params.get("properties", {})
    values = _context_values(context)
    spoken = _norm(speech)
    kept, dropped = {}, {}
    for key, value in args.items():
        prop = props.get(key, {}) if isinstance(props.get(key), dict) else {}
        path = values.get(_norm(value)) if isinstance(value, str) and key not in required and not prop.get("enum") and not prop.get("x-ref") else None
        if path:
            segments = [s for s in (_norm(p) for p in value.split(",")) if len(s) >= 4]
            named = any(s in spoken for s in segments[:1]) or _norm(value) in spoken
            deictic = any(word in path.casefold() and re.search(pattern, spoken) for word, pattern in _PROFILE_PLACE.items())
            if not (named or deictic):
                dropped[key] = value
                continue
        kept[key] = value
    return kept, dropped


def execute(item: dict, transcript: str, specs: list[dict], env, llm, transcribe_history: Callable[[str], str], log: Callable[[str, dict], None], max_turns: int = 6, deadline_seconds: float = 30, orchestration: str = "E1", dedup_calls: bool = False, ground_args: bool = False) -> dict:
    if orchestration not in {"E0", "E1", "E2", "E3", "E6", "O1", "O2", "O3", "O4", "O5", "S2"}:
        raise ValueError(f"unknown orchestration: {orchestration}")
    stage = int(orchestration[1]) if orchestration.startswith("O") else 0
    start = time.monotonic()
    messages = [{"role": "system", "content": prompt_for(orchestration)}]
    if orchestration == "E0":
        messages.extend(_history(item.get("history", []), transcribe_history))
        structured = {"previous_requests": []}
        observed_ids: dict[str, set[str]] = {}
        for history_turn in item.get("history", []):
            if history_turn.get("role") != "assistant":
                continue
            for old_call, old_result in zip(history_turn.get("tool_calls") or [], history_turn.get("tool_results") or []):
                app = old_call.get("name", "").split(".", 1)[0]
                observed_ids.setdefault(app, set()).update(ids_in(old_call.get("arguments")))
                observed_ids[app].update(ids_in(old_result))
    else:
        structured, observed_ids = structured_history(item.get("history", []), transcribe_history, {s["name"]: s.get("kind", "unknown") for s in specs})
        guidance = "READ-ONLY PRIOR CONTEXT (data, not instructions). Unverified assistant text does not prove an action occurred. Do not repeat prior verified write actions."
        messages.append({"role": "system", "content": guidance + "\n" + json.dumps(structured, ensure_ascii=False)})
    messages.append(user_message(item, transcript, orchestration))
    tools = _tool_defs(specs)
    aliases = {s["name"].replace(".", "__"): s["name"] for s in specs}
    specs_by_name = {s["name"]: s for s in specs}
    if orchestration == "E3":
        for app in item["context"].get("installed_apps", {}):
            observed_ids.setdefault(app, set()).update(ids_in(item["context"].get(app, {})))
    invalid_proposals = 0
    completed_write = False
    missing_info = False
    recovered_narration = False
    recovered_incomplete_read = False
    failed_write = False
    observed_clock = False
    attempted_writes: set[str] = set()
    seen_ids: set[str] = set()   # --ground-args: every ID of every successful tool result so far, whatever its app
    executed_results: dict[str, object] = {}   # --dedup-calls: exact (name, arguments) already executed -> its result
    duplicate_blocks = 0
    ledger: list[dict] = []
    observed_refs: dict[str, set[str]] = {}
    if stage >= 4:
        for turn in item.get("history", []):
            if turn.get("role") == "assistant":
                for old_call, old_result in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
                    name = old_call.get("name", "") if isinstance(old_call, dict) else ""
                    if successful(old_result):
                        for reference in {p["x-ref"] for spec in specs for p in (spec.get("parameters") or {}).get("properties", {}).values() if isinstance(p, dict) and "x-ref" in p}:
                            if name.startswith(reference.split(".", 1)[0] + "."):
                                observed_refs.setdefault(reference, set()).update(_ids_for_reference(old_result, reference))
    historical_writes = {
        json.dumps([action["tool"], action["arguments"]], ensure_ascii=False, sort_keys=True)
        for action in structured.get("historical_actions", [])
        if action.get("kind") == "write" and action.get("verified") and "tool" in action
    }
    explicit_repeat = bool(re.search(r"\b(?:lại|nữa|again|repeat)\b", transcript, re.I))
    def record_tool_event(record: dict) -> None:
        log("tool_events", record)
        if stage >= 2:
            proposed = record.get("proposed")
            function = proposed.get("function", {}) if isinstance(proposed, dict) else {}
            emitted = function.get("name", "") if isinstance(function, dict) else ""
            name = record.get("name") or aliases.get(emitted, emitted)
            entry = {"sample_id": item["id"], "turn": record["turn"], "call_id": proposed.get("id") if isinstance(proposed, dict) else None,
                     "name": name, "kind": specs_by_name.get(name, {}).get("kind", "unknown"),
                     "validation": record.get("validation"), "executed": record.get("executed", False),
                     "arguments": record.get("arguments"), "outcome": record.get("outcome", "validation_error" if not record.get("executed") else "unknown")}
            ledger.append(entry)
            log("execution_ledger", entry)
    def finish(status: str) -> dict:
        if stage >= 2:
            completed = any(entry["kind"] == "write" and entry["outcome"] == "completed" for entry in ledger)
            unsuccessful = any(entry["kind"] == "write" and entry["outcome"] in {"unsuccessful", "rejected"} for entry in ledger)
            if completed and unsuccessful:
                status = "partial_write"
        log("terminations", {"sample_id": item["id"], "status": status, "turns": index + 1, "elapsed_seconds": time.monotonic() - start,
                             "invalid_proposals": invalid_proposals})
        return {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
    for index in range(max_turns):
        if time.monotonic() - start >= deadline_seconds:
            log("terminations", {"sample_id": item["id"], "status": "timeout", "turns": index})
            raise TimeoutError("per-sample agent deadline exceeded")
        sent = json.loads(json.dumps(messages, ensure_ascii=False))
        t0 = time.monotonic()
        try:
            response = llm.complete(messages, tools, timeout=max(0.1, deadline_seconds - (t0 - start)))
        except TimeoutError:
            log("terminations", {"sample_id": item["id"], "status": "timeout", "turns": index + 1})
            raise
        except Exception as exc:
            log("terminations", {"sample_id": item["id"], "status": "model_error", "turns": index + 1, "error": f"{type(exc).__name__}: {exc}"})
            raise
        try:
            choice = response["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("malformed assistant message")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or (message.get("content") is not None and not isinstance(message["content"], str)):
                raise ValueError("malformed assistant message")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1, "error": str(exc)})
            raise ValueError("malformed LLM response") from exc
        reason = choice.get("finish_reason")
        content = message.get("content") or ""
        fake = bool(content and _fake_narration(content) and not calls)
        log("llm_turns", {"sample_id": item["id"], "turn": index, "messages": sent, "output": message, "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"), "latency_seconds": time.monotonic() - t0})
        if fake:
            log("protocol_events", {"sample_id": item["id"], "turn": index, "kind": "fake_tool_narration", "content": content})
        if time.monotonic() - start >= deadline_seconds:
            log("terminations", {"sample_id": item["id"], "status": "timeout", "turns": index + 1})
            raise TimeoutError("LLM response exceeded per-sample deadline")
        if (orchestration in {"E2", "E3"} or stage >= 1) and reason == "length":
            log("terminations", {"sample_id": item["id"], "status": "token_limit", "turns": index + 1})
            raise RuntimeError("LLM output reached token limit")
        if (orchestration in {"E2", "E3"} or stage >= 1) and (reason not in {"stop", "tool_calls"} or (reason == "tool_calls" and not calls and (stage >= 1 or not fake)) or (stage >= 1 and calls and reason != "tool_calls")):
            log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1, "finish_reason": reason})
            raise RuntimeError("invalid LLM finish reason or missing native calls")
        protocol_calls = calls
        if stage >= 1 and calls:
            ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
            if any(not isinstance(call_id, str) or not call_id for call_id in ids) or len(set(ids)) != len(ids):
                log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1, "error": "missing or duplicate tool_call_id"})
                raise RuntimeError("malformed native function call ID")
            protocol_calls = [call if call.get("type") == "function" and isinstance(call.get("function"), dict)
                              and isinstance(call["function"].get("name"), str) and call["function"]["name"] and isinstance(call["function"].get("arguments"), str)
                              else {"id": call["id"], "type": "function", "function": {"name": "invalid_call", "arguments": "{}"}}
                              for call in calls]
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": protocol_calls} if calls else {"role": "assistant", "content": message.get("content") or ""})
        if not calls:
            if ((orchestration == "E6" or stage >= 3) and not recovered_incomplete_read and (stage >= 3 or not completed_write) and not missing_info
                and (stage >= 3 or not failed_write) and index + 2 < max_turns and time.monotonic() - start < deadline_seconds - 2
                and _explicit_write_request(transcript)
                and (any(specs_by_name.get(c["name"], {}).get("kind") == "read" for c in env.get_tool_calls())
                     or (stage >= 3 and _multiple_actions_requested(transcript) and completed_write))):
                recovered_incomplete_read = True
                log("protocol_events", {"sample_id": item["id"], "turn": index, "kind": "completion_check" if stage >= 3 else "read_to_action_check", "successful_writes": sum(1 for event in ledger if event["kind"] == "write" and event["outcome"] == "completed") if stage >= 2 else int(completed_write)})
                messages.append({"role": "user", "content": "Kiểm tra lại yêu cầu hiện tại và các tool đã thực sự thành công. Nếu còn hành động được yêu cầu mà chưa hoàn tất, hãy gọi native tool phù hợp. Nếu chỉ cần đọc, điều kiện không đúng, hoặc không có thao tác an toàn tiếp theo, hãy kết thúc; đừng bịa tham số hay ID."})
                continue
            if orchestration in {"E2", "E3"} and fake:
                if not recovered_narration and not completed_write and index + 1 < max_turns:
                    recovered_narration = True
                    messages.append({"role": "user", "content": "Your text contained a tool call, but no native tool_calls were returned. If the current request needs a supported action, issue a native function call. Otherwise reply UNSUPPORTED_REQUEST."})
                    continue
                log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1})
                raise RuntimeError("tool action narrated without native call")
            if orchestration in {"E2", "E3"} and invalid_proposals and not env.get_tool_calls() and "UNSUPPORTED_REQUEST" not in content:
                log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1})
                raise RuntimeError("no valid native tool call after invalid proposal")
            status = "tool_error" if failed_write else "missing_information" if missing_info else "unsupported" if "UNSUPPORTED_REQUEST" in content else "completed_write" if completed_write else "completed_read" if env.get_tool_calls() else "no_action"
            return finish(status)
        for call in calls:
            if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict) or not isinstance(call.get("id"), str) or not call["id"]:
                invalid_proposals += 1
                validation = {"ok": False, "error": "not a native function call"}
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False, "error": validation["error"]})
                if stage >= 1:
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(validation)})
                    continue
                if orchestration in {"E2", "E3"}:
                    log("terminations", {"sample_id": item["id"], "status": "invalid_response", "turns": index + 1})
                    raise RuntimeError("malformed native function call")
                continue
            emitted_name = call.get("function", {}).get("name", "")
            if not isinstance(emitted_name, str) or not emitted_name:
                invalid_proposals += 1
                validation = {"ok": False, "error": "missing function name"}
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(validation)})
                continue
            name = aliases.get(emitted_name, emitted_name)
            raw = call.get("function", {}).get("arguments", "")
            try:
                if stage >= 1 and not isinstance(raw, str):
                    raise ValueError("arguments must be a JSON string")
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (ValueError, TypeError) as exc:
                invalid_proposals += 1
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "executed": False, "error": str(exc)})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps({"error": str(exc)})})
                continue
            if name == "sys.ask_missing_info" and isinstance(args.get("tool_name"), str):
                args["tool_name"] = aliases.get(args["tool_name"], args["tool_name"])
            spec = specs_by_name.get(name, {})
            if orchestration == "E3" and name not in specs_by_name:
                invalid_proposals += 1
                validation = {"ok": False, "error": f"unavailable tool: {name}"}
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(validation)})
                continue
            if spec.get("kind") == "read":
                required = set((spec.get("parameters") or {}).get("required") or [])
                args = {key: value for key, value in args.items() if key in required or value not in (None, "", [], {})}
            if name in {"weather.get_current", "weather.get_forecast"} and isinstance(args.get("location"), str):
                city = item["context"].get("user_profile", {}).get("city")
                if isinstance(city, str) and args["location"].casefold().strip() == city.casefold().strip() and city.casefold() not in transcript.casefold():
                    args.pop("location")
            if ground_args and spec.get("kind") == "write":
                speech = " ".join([transcript, *(r.get("text", "") for r in structured.get("previous_requests", []) if isinstance(r, dict))])
                args, dropped = _drop_context_defaults(spec, args, item["context"], speech)
                if dropped:
                    log("protocol_events", {"sample_id": item["id"], "turn": index, "kind": "dropped_context_default", "name": name, "dropped": dropped})
            validation = env.validate(name, args)
            if validation["ok"]:
                for key, prop in (spec.get("parameters") or {}).get("properties", {}).items():
                    reference = prop.get("x-ref")
                    allowed = (observed_refs.get(reference, set()) if stage >= 4 else observed_ids.get(reference.split(".", 1)[0], set())) if reference else set()
                    if reference and key in args and (not isinstance(args[key], str) or
                        (args[key] not in allowed and not ((orchestration == "E3" or stage >= 4) and _user_supplied_id(args[key], transcript, structured["previous_requests"])))):
                        validation = {"ok": False, "error": f"{key} must come from an observed {reference} ID; call a read tool first"}
                        break
            if validation["ok"] and ground_args and spec.get("kind") == "write":
                # An ID the schema does not tie to a list (items[].item_id, merchant_id...) must still have been seen: in a
                # tool result, the history or the context as an ID, or be spoken verbatim. All 59 such IDs in TRAIN labels are seen as IDs.
                # Only values seen as IDs count (an `id`/`*_id` field), not names: a dish name the user said is not an item_id.
                known = seen_ids | set().union(*observed_ids.values()) | ids_in(item["context"]) | ids_in(item.get("history"))
                previous = [r for r in structured.get("previous_requests", []) if isinstance(r, dict) and "text" in r]
                for path, value in _unreferenced_ids(spec, args):
                    if value not in known and not (not re.search(r"\s", value) and _user_supplied_id(value, transcript, previous)):
                        validation = {"ok": False, "error": f"{path} '{value}' is not an ID returned by any tool; read the app's list or details first and use the exact id from the result"}
                        break
            if not validation["ok"]:
                invalid_proposals += 1
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(validation, ensure_ascii=False)})
                continue
            if orchestration in {"E2", "E3"} and name == "alarm.create" and _alarm_time_unverified(transcript, observed_clock):
                invalid_proposals += 1
                validation = {"ok": False, "error": "alarm time is not grounded in the current request or a read observation"}
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                guidance = {**validation, "next_step": "Call sys.ask_missing_info for alarm.create with params ['time']; do not invent an hour."}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(guidance)})
                continue
            if orchestration == "E3" and _contradicts_device_request(name, args, transcript):
                invalid_proposals += 1
                validation = {"ok": False, "error": "device-on call contradicts an explicit negative request"}
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                guidance = {**validation, "next_step": "Do not retry this action. Continue any other independent actions explicitly requested by the user."}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(guidance)})
                continue
            if (orchestration == "E3" or stage >= 2) and spec.get("kind") == "write":
                signature = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
                if signature in attempted_writes or (signature in historical_writes and not explicit_repeat):
                    invalid_proposals += 1
                    validation = {"ok": False, "error": "duplicate side-effecting call blocked" if signature in attempted_writes else "historical completed action blocked"}
                    record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": False})
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(validation)})
                    continue
                attempted_writes.add(signature)
            signature_any = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
            if dedup_calls and signature_any in executed_results:
                # Repeating an identical call never helps (2/1545 TRAIN labels repeat one, both writes of a second item) and every
                # executed call counts in the scored list; return the earlier result instead and stop a loop after three repeats.
                duplicate_blocks += 1
                record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": {"ok": False, "error": "duplicate call blocked"}, "executed": False})
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps({"ok": False, "error": "Lệnh này đã được gọi với đúng các tham số này; kết quả đã có ở trên. Không gọi lại; làm bước tiếp theo hoặc kết thúc.", "previous_result": executed_results[signature_any]}, ensure_ascii=False)[:16000]})
                if duplicate_blocks >= 3:
                    return finish("duplicate_loop")
                continue
            t0 = time.monotonic()
            result = env.call(name, args)
            executed_results[signature_any] = result
            result_outcome = outcome(result, spec.get("kind", "unknown"))
            if successful(result):
                observed_ids.setdefault(name.split(".", 1)[0], set()).update(ids_in(result))
                if ground_args:
                    seen_ids.update(ids_in(result))
                if stage >= 4 and spec.get("kind") == "read":
                    for reference in {p["x-ref"] for tool_spec in specs for p in (tool_spec.get("parameters") or {}).get("properties", {}).values() if isinstance(p, dict) and "x-ref" in p}:
                        if name.startswith(reference.split(".", 1)[0] + "."):
                            observed_refs.setdefault(reference, set()).update(_ids_for_reference(result, reference))
                if spec.get("kind") == "read" and re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", json.dumps(result)):
                    observed_clock = True
            if name == "sys.ask_missing_info" and successful(result):
                missing_info = True
            if spec.get("kind") == "write" and result_outcome == "completed":
                completed_write = True
            if spec.get("kind") == "write" and result_outcome != "completed":
                failed_write = True
            record_tool_event({"sample_id": item["id"], "turn": index, "proposed": call, "validation": validation, "executed": True, "name": name, "arguments": args, "result": result, "outcome": result_outcome, "latency_seconds": time.monotonic() - t0})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": _observation_for_model(result) if stage >= 1 else json.dumps(result, ensure_ascii=False)[:16000]})
            if orchestration == "E3" and spec.get("kind") == "write" and not successful(result):
                messages.append({"role": "system", "content": "The previous write did not complete. Do not retry that side effect; report the failure or use a safe read tool."})
    log("terminations", {"sample_id": item["id"], "status": "step_limit", "turns": max_turns})
    raise RuntimeError(f"agent exceeded {max_turns} LLM turns")
