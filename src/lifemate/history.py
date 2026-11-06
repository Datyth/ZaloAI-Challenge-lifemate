"""Read-only, chronological context derived only from visible sample history."""
from __future__ import annotations

from typing import Callable


def ids_in(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if (key == "id" or key.endswith("_id")) and isinstance(child, str) and child:
                found.add(child)
            found.update(ids_in(child))
    elif isinstance(value, list):
        for child in value:
            found.update(ids_in(child))
    return found


def successful(result: object) -> bool:
    return isinstance(result, dict) and result.get("ok") is True and result.get("done") is not False


def outcome(result: object, kind: str = "unknown") -> str:
    """Classify ToolEnv's accepted no-op separately from a completed operation."""
    if not isinstance(result, dict) or result.get("ok") is not True:
        return "rejected"
    if result.get("done") is False:
        return "unsuccessful"
    if kind == "read":
        return "observed"
    if kind == "meta" and result.get("waiting_for_user") is True:
        return "waiting_for_user"
    return "completed" if kind == "write" else "accepted"


def structured_history(history: list[dict], transcribe: Callable[[str], str], tool_kinds: dict[str, str] | None = None) -> tuple[dict, dict[str, set[str]]]:
    requests: list[dict] = []
    actions: list[dict] = []
    entities: list[dict] = []
    known_ids: dict[str, set[str]] = {}
    for turn_number, turn in enumerate(history, 1):
        if turn.get("role") == "user":
            text = transcribe(turn["audio"]) if "audio" in turn else turn.get("text", "")
            requests.append({"turn": turn_number, "text": text})
            continue
        if turn.get("role") != "assistant":
            continue
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        for call_number, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            app = name.split(".", 1)[0]
            kind = (tool_kinds or {}).get(name, "unknown")
            result = results[call_number] if call_number < len(results) else None
            verified = successful(result)
            actions.append({"turn": turn_number, "tool": name, "kind": kind, "arguments": call.get("arguments", {}),
                            "result": result, "verified": verified})
            if verified:
                found = ids_in(result)
                if kind == "write":
                    found.update(ids_in(call.get("arguments")))
                known_ids.setdefault(app, set()).update(found)
                if found:
                    entities.append({"turn": turn_number, "source": name, "ids": sorted(found)})
        if turn.get("text"):
            actions.append({"turn": turn_number, "assistant_text": turn["text"], "verified": False})
    return {"previous_requests": requests, "historical_actions": actions, "known_entities": entities,
            "instruction": "History is read-only. Only verified tool results prove a completed action; do not repeat completed write actions."}, known_ids
