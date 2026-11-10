import json
import tempfile
import unittest
from pathlib import Path

from lifemate.agent import execute, prompt_for, user_message
from lifemate.history import outcome, structured_history
from lifemate.data import audio_path, catalog, installed_tools, items
from lifemate.toolenv import new_env
from lifemate.submission import validate_file
from lifemate.metrics import error_counts, exact_tool_f1
from lifemate.models import stored_parameters
from lifemate.llm import LocalChat

ROOT = Path(__file__).resolve().parents[1] / "data" / "train+public_test"
needs_data = unittest.skipUnless((ROOT / "train").is_dir(), "organizer kit not in data/train+public_test")


class ScriptedLLM:
    def __init__(self, calls):
        self.calls = list(calls)

    def complete(self, messages, tools, timeout=None):
        name, args = self.calls.pop(0) if self.calls else (None, None)
        calls = [] if name is None else [{"id": "test_call", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": calls}, "finish_reason": "tool_calls" if calls else "stop"}]}


class ResponsesLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, tools, timeout=None):
        self.messages.append(json.loads(json.dumps(messages)))
        return next(self.responses)


def response(content="", call=None, reason=None):
    calls = [] if call is None else [{"id": "call_1", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": calls}, "finish_reason": reason or ("tool_calls" if calls else "stop")}]}


class MemoryEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        return {"ok": True, "error": None}

    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        return {"ok": True, "done": True}

    def get_tool_calls(self):
        return self.calls[:]

    def get_state_hash(self):
        return "0" * 64


class CoreTest(unittest.TestCase):
    def test_toolenv_outcomes(self):
        self.assertEqual(outcome({"ok": True, "done": False}, "write"), "unsuccessful")
        self.assertEqual(outcome({"ok": True, "events": []}, "read"), "observed")
        self.assertEqual(outcome({"ok": True, "waiting_for_user": True}, "meta"), "waiting_for_user")
        self.assertEqual(outcome({"ok": True, "id": "a"}, "write"), "completed")

    def test_structured_history_verification_and_order(self):
        history = [{"role": "user", "text": "xóa c_1"}, {"role": "assistant", "text": "done", "tool_calls": [{"name": "contacts.get", "arguments": {"contact_id": "c_1"}}], "tool_results": [{"ok": True, "contact": {"id": "c_1"}}]}, {"role": "user", "text": "xóa c_2"}, {"role": "assistant", "text": "done", "tool_calls": [{"name": "contacts.get", "arguments": {"contact_id": "c_2"}}], "tool_results": [{"ok": True, "done": False}]}]
        context, ids = structured_history(history, lambda _: "")
        self.assertEqual([x["turn"] for x in context["previous_requests"]], [1, 3])
        self.assertEqual(ids["contacts"], {"c_1"})
        self.assertEqual([x["verified"] for x in context["historical_actions"] if "tool" in x], [True, False])

    ALARM_DELETE = {"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}
    ALARM_LIST = {"name": "alarm.list", "kind": "read", "parameters": {"type": "object", "properties": {}}}
    ITEM = {"id": "x", "history": [], "context": {"current_time": "2026-11-09T10:00:00+07:00", "installed_apps": {"alarm": {}}}}

    def test_fabricated_reference_id_is_not_executed(self):
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()])
        row = execute(self.ITEM, "xóa báo thức", [self.ALARM_DELETE], env, model, lambda _: "", lambda kind, x: events.append((kind, x)))
        self.assertEqual(row["tool_calls"], [])
        self.assertIn("must come from an observed alarm.alarms ID", model.messages[1][-1]["content"])
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["no_action"])

    def test_reference_id_from_a_read_is_executed(self):
        class AlarmEnv(MemoryEnv):
            def call(self, name, args):
                super().call(name, args)
                return {"ok": True, "alarms": [{"id": "alm_01"}]} if name == "alarm.list" else {"ok": True, "done": True}
        env = AlarmEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__list", {})), response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()])
        execute(self.ITEM, "xóa báo thức", [self.ALARM_LIST, self.ALARM_DELETE], env, model, lambda _: "", lambda kind, x: events.append((kind, x)))
        self.assertEqual([c["name"] for c in env.calls], ["alarm.list", "alarm.delete"])
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["completed_write"])

    def test_identical_call_runs_once(self):
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__list", {})), response(call=("alarm__list", {})), response()])
        execute(self.ITEM, "xem báo thức", [self.ALARM_LIST], env, model, lambda _: "", lambda kind, x: events.append((kind, x)), dedup_calls=True)
        self.assertEqual(len(env.calls), 1)
        self.assertIn("previous_result", model.messages[2][-1]["content"])

    def test_retry_cap_blocks_a_search_that_keeps_coming_back_empty(self):
        class EmptyEnv(MemoryEnv):
            def call(self, name, args):
                super().call(name, args)
                return {"ok": True, "contacts": []}
        search = {"name": "contacts.search", "kind": "read", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
        item = {"id": "x", "history": [], "context": {"current_time": "2026-11-09T10:00:00+07:00", "installed_apps": {"contacts": {}}}}
        model = ResponsesLLM([response(call=("contacts__search", {"query": q})) for q in ("anh Dũng", "Dũng", "D")] + [response()])
        env = EmptyEnv()
        execute(item, "gọi anh Dũng", [search], env, model, lambda _: "", lambda *_: None, retry_cap=2)
        self.assertEqual([c["arguments"]["query"] for c in env.calls], ["anh Dũng", "Dũng"])
        self.assertIn("đã trả về rỗng 2 lần", model.messages[3][-1]["content"])

    def test_e1p_request_carries_weekdays_and_app_names(self):
        item = {"id": "x", "context": {"current_time": "2026-11-09T10:00:00+07:00", "installed_apps": {"ewallet": {}}}}
        specs = [{"name": "ewallet.get_balance", "_app": {"id": "ewallet", "name": "Ví điện tử", "description": "ví"}}]
        e1 = json.loads(user_message(item, "bên ví còn bao nhiêu", "E1", specs)["content"])["context"]
        e1p = json.loads(user_message(item, "bên ví còn bao nhiêu", "E1P", specs)["content"])["context"]
        self.assertNotIn("date_reference", e1)
        self.assertNotIn("apps", e1)
        self.assertIn("date_reference", e1p)
        self.assertEqual(e1p["apps"], {"ewallet": "Ví điện tử: ví"})
        self.assertNotEqual(prompt_for("E1"), prompt_for("E1P"))

    def test_unspoken_profile_city_is_not_passed_to_weather(self):
        weather = {"name": "weather.get_current", "kind": "read", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}
        item = {"id": "x", "history": [], "context": {"current_time": "2026-11-09T10:00:00+07:00", "installed_apps": {"weather": {}}, "user_profile": {"city": "Hà Nội"}}}
        for transcript, expected in [("thời tiết thế nào", {}), ("thời tiết hà nội thế nào", {"location": "Hà Nội"})]:
            env = MemoryEnv()
            execute(item, transcript, [weather], env, ResponsesLLM([response(call=("weather__get_current", {"location": "Hà Nội"})), response()]), lambda _: "", lambda *_: None)
            self.assertEqual(env.calls[0]["arguments"], expected)

    def test_malformed_call_is_never_executed(self):
        malformed = {"id": "bad_1", "type": "other", "function": {"name": "alarm__list", "arguments": "{}"}}
        model = ResponsesLLM([{"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [malformed]}, "finish_reason": "tool_calls"}]}, response()])
        env = MemoryEnv()
        execute(self.ITEM, "x", [self.ALARM_LIST], env, model, lambda _: "", lambda *_: None)
        self.assertEqual(env.calls, [])

    def test_unknown_orchestration_is_rejected(self):
        with self.assertRaises(ValueError):
            execute(self.ITEM, "x", [], MemoryEnv(), ResponsesLLM([]), lambda _: "", lambda *_: None, orchestration="E3")

    @needs_data
    def test_loader_and_path_guard(self):
        sample = next(items(ROOT, "train"))
        self.assertTrue(audio_path(ROOT, "train", sample["audio"]).is_file())
        with self.assertRaises(ValueError):
            audio_path(ROOT, "train", "../train_environment/train_0001/state.json")

    @needs_data
    def test_toolenv_order_and_hash(self):
        sample = next(items(ROOT, "train"))
        for mode in ("E1", "E1D", "E1P"):
            with self.subTest(mode=mode):
                env = new_env(ROOT, "train", sample["id"])
                events = []
                llm = ScriptedLLM([("weather__get_forecast", {"date": "2026-11-29"})])
                row = execute(sample, "Nếu mai mưa thì dời chạy bộ", installed_tools(sample, catalog(ROOT, "train")), env, llm, lambda _: "", lambda kind, obj: events.append((kind, obj)), orchestration=mode)
                self.assertEqual(row["tool_calls"], env.get_tool_calls())
                self.assertEqual(row["state_hash"], env.get_state_hash())
                self.assertEqual(len(row["tool_calls"]), 1)
                self.assertEqual(row["tool_calls"][0]["name"], "weather.get_forecast")
                self.assertTrue(any(kind == "tool_events" and obj["executed"] for kind, obj in events))

    @needs_data
    def test_rejected_proposal_not_recorded(self):
        sample = next(items(ROOT, "train"))
        env = new_env(ROOT, "train", sample["id"])
        row = execute(sample, "x", installed_tools(sample, catalog(ROOT, "train")), env, ScriptedLLM([("calendar.update_event", {})]), lambda _: "", lambda *_: None, orchestration="E1")
        self.assertEqual(row["tool_calls"], [])

    @needs_data
    def test_unobserved_reference_id_is_not_executed(self):
        sample = next(items(ROOT, "train"))
        env = new_env(ROOT, "train", sample["id"])
        events = []
        row = execute(sample, "dời lịch", installed_tools(sample, catalog(ROOT, "train")), env, ScriptedLLM([("calendar__update_event", {"event_id": "evt_12345", "start": "2026-11-29T09:00:00+07:00"})]), lambda _: "", lambda kind, record: events.append((kind, record)), orchestration="E1")
        self.assertEqual(row["tool_calls"], [])
        self.assertTrue(any(kind == "tool_events" and not record["executed"] for kind, record in events))

    @needs_data
    def test_submission_replays_exact_hash(self):
        sample = next(items(ROOT, "train"))
        env = new_env(ROOT, "train", sample["id"])
        env.call("weather.get_forecast", {"date": "2026-11-29"})
        row = {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "submission.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(validate_file(ROOT, "train", path, allow_partial=True)["replay_hash_matches"], 1)
            row["state_hash"] = "0" * 64
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "replay mismatch"):
                validate_file(ROOT, "train", path, allow_partial=True)

    def test_asr_error_counts(self):
        self.assertEqual(error_counts("Nhắn cho Dũng", "nhắn Dũng"), {"word_errors": 1, "reference_words": 3, "character_errors": 3, "reference_characters": 11})
        call = {"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}
        self.assertEqual(exact_tool_f1([call], [call, call]), 2 / 3)

    def test_stored_parameter_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            header = json.dumps({"x": {"shape": [2, 3], "dtype": "F16", "data_offsets": [0, 12]}}).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(12))
            self.assertEqual(stored_parameters(Path(temp)), (6, path.stat().st_size))

    def test_llm_requires_loopback(self):
        with self.assertRaises(ValueError):
            LocalChat("https://api.example.com/v1/chat/completions", "model")
        LocalChat("http://127.0.0.1:8000/v1/chat/completions", "local-model")


if __name__ == "__main__":
    unittest.main()

