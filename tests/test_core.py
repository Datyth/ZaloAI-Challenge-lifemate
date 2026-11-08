import json
import tempfile
import unittest
from pathlib import Path

from lifemate.agent import execute, prompt_for, _observation_for_model
from lifemate.history import outcome, structured_history
from lifemate.data import audio_path, catalog, installed_tools, items
from lifemate.toolenv import new_env
from lifemate.submission import validate_file
from lifemate.metrics import error_counts, exact_tool_f1
from lifemate.asr_eval import critical_slots, slot_hits, labelled_slices
from lifemate.models import stored_parameters
from lifemate.llm import LocalChat
from scripts.analyze_agent_optimization import classify

ROOT = Path(__file__).resolve().parents[1] / "data" / "train+public_test"


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

    def test_o1_rejects_length_without_executing_calls(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__delete", {}), reason="length")])
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            execute(item, "x", [], env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="O1")
        self.assertEqual(env.calls, [])
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["token_limit"])

    def test_o1_malformed_call_gets_matching_response(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        malformed = {"id": "bad_1", "type": "other", "function": {"name": "foo", "arguments": "{}"}}
        model = ResponsesLLM([{"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [malformed]}, "finish_reason": "tool_calls"}]}, response()])
        events = []
        execute(item, "x", [], MemoryEnv(), model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="O1")
        previous = model.messages[1]
        self.assertEqual(previous[-2]["tool_calls"][0]["id"], "bad_1")
        self.assertEqual(previous[-2]["tool_calls"][0]["type"], "function")
        self.assertEqual(previous[-1]["tool_call_id"], "bad_1")

    def test_o2_does_not_repeat_unsuccessful_write(self):
        class NoopEnv(MemoryEnv):
            def call(self, name, args):
                super().call(name, args)
                return {"ok": True, "done": False, "message": "not found"}
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        specs = [{"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {}}}]
        model = ResponsesLLM([response(call=("alarm__delete", {})), response(call=("alarm__delete", {})), response()])
        env = NoopEnv(); events = []
        execute(item, "x", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="O2")
        self.assertEqual(len(env.calls), 1)
        self.assertEqual([x["outcome"] for kind, x in events if kind == "execution_ledger"], ["unsuccessful", "validation_error"])

    def test_o4_grounds_collection_not_sibling_collection(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        specs = [{"name": "banking.get_accounts", "kind": "read", "parameters": {"type": "object", "properties": {}}},
                 {"name": "banking.transfer", "kind": "write", "parameters": {"type": "object", "properties": {"beneficiary_id": {"type": "string", "x-ref": "banking.beneficiaries"}}, "required": ["beneficiary_id"]}}]
        class AccountsEnv(MemoryEnv):
            def call(self, name, args):
                super().call(name, args)
                return {"ok": True, "accounts": [{"id": "shared_1"}]} if name == "banking.get_accounts" else {"ok": True}
        model = ResponsesLLM([response(call=("banking__get_accounts", {})), response(call=("banking__transfer", {"beneficiary_id": "shared_1"})), response()])
        env = AccountsEnv(); events = []
        execute(item, "x", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="O4")
        self.assertEqual([x["name"] for x in env.calls], ["banking.get_accounts"])
        self.assertTrue(any(kind == "tool_events" and x.get("validation", {}).get("ok") is False for kind, x in events))

    def test_o3_checks_after_first_of_two_writes(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        specs = [{"name": "alarm.create", "kind": "write", "parameters": {"type": "object", "properties": {}}},
                 {"name": "reminders.create", "kind": "write", "parameters": {"type": "object", "properties": {}}}]
        model = ResponsesLLM([response(call=("alarm__create", {})), response("đã đặt báo thức"),
                              response(call=("reminders__create", {})), response()])
        env = MemoryEnv(); events = []
        execute(item, "Đặt báo thức và tạo nhắc nhở", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="O3")
        self.assertEqual([x["name"] for x in env.calls], ["alarm.create", "reminders.create"])
        self.assertEqual(sum(kind == "protocol_events" and x.get("kind") == "completion_check" for kind, x in events), 1)

    def test_o4_accepts_explicit_user_id(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        specs = [{"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}]
        env = MemoryEnv()
        execute(item, "Xóa alm_01", specs, env, ResponsesLLM([response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()]), lambda _: "", lambda *_: None, orchestration="O4")
        self.assertEqual(len(env.calls), 1)

    def test_o5_prompt_is_general(self):
        self.assertNotIn("calendar.list_events", prompt_for("O5"))
        self.assertIn("done=false", prompt_for("O5"))

    def test_large_observation_remains_json_with_status(self):
        text = _observation_for_model({"ok": True, "done": False, "message": "not found", "events": [{"id": str(i), "title": "x" * 1000} for i in range(100)]})
        self.assertLessEqual(len(text), 16000)
        self.assertEqual(json.loads(text)["done"], False)
        self.assertTrue(json.loads(text)["truncated"])

    def test_structured_history_verification_and_order(self):
        history = [{"role": "user", "text": "xóa c_1"}, {"role": "assistant", "text": "done", "tool_calls": [{"name": "contacts.get", "arguments": {"contact_id": "c_1"}}], "tool_results": [{"ok": True, "contact": {"id": "c_1"}}]}, {"role": "user", "text": "xóa c_2"}, {"role": "assistant", "text": "done", "tool_calls": [{"name": "contacts.get", "arguments": {"contact_id": "c_2"}}], "tool_results": [{"ok": True, "done": False}]}]
        context, ids = structured_history(history, lambda _: "")
        self.assertEqual([x["turn"] for x in context["previous_requests"]], [1, 3])
        self.assertEqual(ids["contacts"], {"c_1"})
        self.assertEqual([x["verified"] for x in context["historical_actions"] if "tool" in x], [True, False])

    def test_fake_narration_recovery_and_native_execution(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"weather": {}}}}
        specs = [{"name": "weather.get_current", "kind": "read", "parameters": {"type": "object", "properties": {}}}]
        model = ResponsesLLM([response('Đã gọi tool: [{"name":"weather__get_current","arguments":{}}]'), response(call=("weather__get_current", {})), response()])
        env = MemoryEnv(); events = []
        row = execute(item, "thời tiết", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E3")
        self.assertEqual(len(row["tool_calls"]), 1)
        self.assertEqual(row["tool_calls"][0]["name"], "weather.get_current")
        self.assertTrue(any(kind == "protocol_events" for kind, _ in events))
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["completed_read"])

    def test_truncated_response_is_failure(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {}}}
        events = []
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            execute(item, "x", [], MemoryEnv(), ResponsesLLM([response("partial", reason="length")]), lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E2")
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["token_limit"])

    def test_e6_checks_read_to_write_once(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"calendar": {}}}}
        specs = [{"name": "calendar.list_events", "kind": "read", "parameters": {"type": "object", "properties": {}}},
                 {"name": "calendar.create_event", "kind": "write", "parameters": {"type": "object", "properties": {}}}]
        model = ResponsesLLM([response(call=("calendar__list_events", {})), response("Đã xem lịch"),
                              response(call=("calendar__create_event", {})), response()])
        env = MemoryEnv(); events = []
        execute(item, "Đặt lịch họp", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E6")
        self.assertEqual([x["name"] for x in env.calls], ["calendar.list_events", "calendar.create_event"])
        self.assertEqual([x["kind"] for kind, x in events if kind == "protocol_events"], ["read_to_action_check"])
        self.assertIn("các tool đã thực sự thành công", model.messages[2][-1]["content"])
        self.assertNotEqual(prompt_for("E1"), prompt_for("E6"))

    def test_e6_does_not_extend_read_only_request(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"calendar": {}}}}
        specs = [{"name": "calendar.list_events", "kind": "read", "parameters": {"type": "object", "properties": {}}}]
        model = ResponsesLLM([response(call=("calendar__list_events", {})), response("Sự kiện đã tìm thấy")])
        env = MemoryEnv(); events = []
        execute(item, "Xem lịch họp", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E6")
        self.assertEqual(len(model.messages), 2)
        self.assertFalse(any(kind == "protocol_events" and x.get("kind") == "read_to_action_check" for kind, x in events))

    def test_e5_classification_uses_oracle_and_read_evidence(self):
        sample = {"label": {"tool_calls": [{"name": "calendar.list_events", "arguments": {}},
                                            {"name": "calendar.update_event", "arguments": {}}]}, "history": [], "audio_text": "Dời lịch họp"}
        base = {"tool_calls": [{"name": "calendar.list_events", "arguments": {}}], "status": "ok"}
        specs = {"calendar.list_events": {"kind": "read"}, "calendar.update_event": {"kind": "write"}}
        primary, secondary, evidence = classify(sample, base, None, [], specs)
        self.assertEqual(primary, "Premature termination")
        self.assertIn("Multi-step planning", secondary)
        self.assertTrue(evidence)
        primary, _, _ = classify(sample, base, {"train_state_match": True}, [], specs)
        self.assertEqual(primary, "ASR errors")

    def test_direct_user_id_is_accepted_but_fabricated_id_is_rejected(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"alarm": {}}}}
        specs = [{"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}]
        for transcript, expected in [("xóa alm_01", 1), ("xóa báo thức", 0)]:
            env = MemoryEnv(); events = []
            model = ResponsesLLM([response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()])
            try:
                execute(item, transcript, specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E3")
            except RuntimeError:
                self.assertEqual(expected, 0)
            self.assertEqual(len(env.calls), expected)

    def test_duplicate_write_is_not_replayed(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"alarm": {}}}}
        specs = [{"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}]
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__delete", {"alarm_id": "alm_01"})), response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()])
        execute(item, "xóa alm_01", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E3")
        self.assertEqual(len(env.calls), 1)
        self.assertTrue(any(kind == "tool_events" and record.get("validation", {}).get("error") == "duplicate side-effecting call blocked" for kind, record in events))

    def test_historical_write_requires_explicit_repeat(self):
        item = {"id": "x", "history": [{"role": "assistant", "tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "alm_01"}}], "tool_results": [{"ok": True, "done": True}]}], "context": {"installed_apps": {"alarm": {}}}}
        specs = [{"name": "alarm.delete", "kind": "write", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}]
        for transcript, count in [("xóa alm_01", 0), ("xóa lại alm_01", 1)]:
            env = MemoryEnv()
            model = ResponsesLLM([response(call=("alarm__delete", {"alarm_id": "alm_01"})), response()])
            try:
                execute(item, transcript, specs, env, model, lambda _: "", lambda *_: None, orchestration="E3")
            except RuntimeError:
                self.assertEqual(count, 0)
            self.assertEqual(len(env.calls), count)

    def test_explicit_device_negation_blocks_write(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"smart_home": {}}}}
        specs = [{"name": "smart_home.fan_set", "kind": "write", "parameters": {"type": "object", "properties": {"power": {"type": "string"}}}}]
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("smart_home__fan_set", {"power": "on"})), response()])
        with self.assertRaisesRegex(RuntimeError, "no valid native tool call"):
            execute(item, "quạt thì đừng bật", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E3")
        self.assertEqual(env.calls, [])
        self.assertTrue(any(kind == "tool_events" and record.get("validation", {}).get("error") == "device-on call contradicts an explicit negative request" for kind, record in events))

    def test_alarm_without_spoken_time_requests_missing_info(self):
        item = {"id": "x", "history": [], "context": {"installed_apps": {"alarm": {}}}}
        specs = [{"name": "alarm.create", "kind": "write", "parameters": {"type": "object", "properties": {"time": {"type": "string"}}, "required": ["time"]}},
                 {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}}}}]
        env = MemoryEnv(); events = []
        model = ResponsesLLM([response(call=("alarm__create", {"time": "2026-11-22T07:00:00+07:00"})),
                              response(call=("sys__ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]})), response()])
        execute(item, "Mai đặt báo thức gọi mình dậy nhé", specs, env, model, lambda _: "", lambda kind, x: events.append((kind, x)), orchestration="E2")
        self.assertEqual([x["name"] for x in env.calls], ["sys.ask_missing_info"])
        self.assertEqual([x["status"] for kind, x in events if kind == "terminations"], ["missing_information"])

    def test_loader_and_path_guard(self):
        sample = next(items(ROOT, "train"))
        self.assertTrue(audio_path(ROOT, "train", sample["audio"]).is_file())
        with self.assertRaises(ValueError):
            audio_path(ROOT, "train", "../train_environment/train_0001/state.json")

    def test_toolenv_order_and_hash(self):
        sample = next(items(ROOT, "train"))
        for mode in ("E0", "E1", "E2", "E3"):
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

    def test_rejected_proposal_not_recorded(self):
        sample = next(items(ROOT, "train"))
        env = new_env(ROOT, "train", sample["id"])
        row = execute(sample, "x", installed_tools(sample, catalog(ROOT, "train")), env, ScriptedLLM([("calendar.update_event", {})]), lambda _: "", lambda *_: None, orchestration="E0")
        self.assertEqual(row["tool_calls"], [])

    def test_unobserved_reference_id_is_not_executed(self):
        sample = next(items(ROOT, "train"))
        env = new_env(ROOT, "train", sample["id"])
        events = []
        row = execute(sample, "dời lịch", installed_tools(sample, catalog(ROOT, "train")), env, ScriptedLLM([("calendar__update_event", {"event_id": "evt_12345", "start": "2026-11-29T09:00:00+07:00"})]), lambda _: "", lambda kind, record: events.append((kind, record)), orchestration="E0")
        self.assertEqual(row["tool_calls"], [])
        self.assertTrue(any(kind == "tool_events" and not record["executed"] for kind, record in events))

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

    def test_critical_slots_use_only_explicit_gold_values(self):
        calls = [{"name": "contacts.search", "arguments": {"query": "Phạm Hùng"}},
                 {"name": "banking.transfer", "arguments": {"amount": 100000, "recipient_id": "c_123"}}]
        slots = critical_slots("Chuyển tiền cho Phạm Hùng nhé", calls)
        self.assertEqual([(x["key"], x["value"]) for x in slots], [("query", "Phạm Hùng")])
        self.assertTrue(slot_hits(slots, "chuyển tiền cho phạm hùng")[0]["matched"])
        self.assertFalse(slot_hits(slots, "chuyển tiền cho Phạm Hoàng")[0]["matched"])

    def test_audio_slices_do_not_invent_accent_or_noise_labels(self):
        self.assertEqual(labelled_slices({}, {"duration_seconds": 1.5, "warnings": []}), ["short_under_4s"])
        self.assertEqual(labelled_slices({"metadata": {"accent": "Miền Trung", "noisy": True, "truncated": True}},
                                         {"duration_seconds": 3, "warnings": []}),
                    ["short_under_4s", "accent:miền_trung", "noisy_labelled", "truncated_labelled"])

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


class GroundArgsTest(unittest.TestCase):
    ORDER = {"name": "food_delivery.place_order", "kind": "write", "parameters": {"type": "object", "required": ["restaurant_id", "items"], "properties": {
        "restaurant_id": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "properties": {"item_id": {"type": "string"}, "quantity": {"type": "integer"}}}},
        "address": {"type": "string"}}}}
    ITEM = {"id": "x", "history": [], "context": {"installed_apps": {"food_delivery": {}}, "user_profile": {"home_address": "25 Láng Hạ, Đống Đa"},
                                                  "food_delivery": {"favourites": [{"restaurant_id": "r1"}]}}}

    def run_calls(self, transcript, calls):
        env = MemoryEnv(); events = []
        model = ResponsesLLM([*(response(call=c) for c in calls), response()])
        execute(self.ITEM, transcript, [self.ORDER], env, model, lambda _: "", lambda kind, x: events.append((kind, x)), ground_args=True)
        return env.calls, events

    def test_spoken_dish_name_is_not_an_item_id(self):
        calls, events = self.run_calls("đặt hai phần mì xào chay", [("food_delivery__place_order", {"restaurant_id": "r1", "items": [{"item_id": "mì xào chay", "quantity": 2}]})])
        self.assertEqual(calls, [])
        self.assertTrue(any("is not an ID returned" in str(x.get("validation")) for kind, x in events if kind == "tool_events"))

    def test_unmentioned_home_address_is_dropped(self):
        calls, _ = self.run_calls("đặt món", [("food_delivery__place_order", {"restaurant_id": "r1", "items": [], "address": "25 Láng Hạ, Đống Đa"})])
        self.assertNotIn("address", calls[0]["arguments"])
        calls, _ = self.run_calls("đặt món giao về nhà", [("food_delivery__place_order", {"restaurant_id": "r1", "items": [], "address": "25 Láng Hạ, Đống Đa"})])
        self.assertEqual(calls[0]["arguments"]["address"], "25 Láng Hạ, Đống Đa")
