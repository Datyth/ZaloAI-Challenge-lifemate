"""Agent SFT data: TRAIN-only trajectories, each sample twice (gold transcripts and lower-cased Gipformer transcripts).
Release data (scripts/data/build_final_data.sh runs it for E1P and E1, validation included):

    python scripts/data/prepare_agent_sft.py --orchestration E1P --include-validation --output-dir outputs/sft/data_v2_e1p_full

Why two copies: the S1 adapter learned on gold `audio_text` only, while inference reads Gipformer text (WER ~17%, intent verbs and
names misheard); on the 200 validation IDs S1 scores 71.4 with gold text and 44.0 with lower-cased ASR. Pairing the same gold calls
with the ASR transcript teaches the model to recover the intended action from noisy speech; the gold copy keeps the clean case.
Leakage rule: the 200 validation IDs and every sample sharing a history recording with them are excluded.
Observations come from replaying the gold calls in the official ToolEnv (checked against label.tool_results)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.agent import HISTORY_GUIDANCE as GUIDANCE, _tool_defs, prompt_for, user_message  # noqa: E402
from lifemate.context import normalize_asr  # noqa: E402
from lifemate.data import catalog, installed_tools, items  # noqa: E402
from lifemate.history import structured_history  # noqa: E402
from lifemate.toolenv import new_env  # noqa: E402


def history_audio_ids(item: dict) -> set[str]:
    """TRAIN ids whose recordings this sample plays (current turn and history), read from the audio paths."""
    paths = [item["audio"], *(turn["audio"] for turn in item["history"] if "audio" in turn)]
    return {part for path in paths for part in (*Path(path).parts[:-1], Path(path).stem) if part.startswith("train_")}


# --rename-apps: other plausible namespaces for each TRAIN app, so a trajectory can be shown under names the model never saw and
# it has to follow the schema (descriptions, x-ref, required fields) instead of recalling a memorised app. Written from the train
# catalog only; no public app id is used as an alias.
ALIASES = {
    "contacts": ["phonebook", "address_book", "people_directory"], "calendar": ["schedule", "agenda", "event_calendar"],
    "smart_home": ["home_control", "iot_hub", "house_devices"], "reminders": ["reminder_app", "memo_alerts", "alerts_planner"],
    "bill_payment": ["bills", "utility_pay", "invoice_pay"], "chat_ott": ["messenger", "chat_app", "instant_chat"],
    "alarm": ["alarm_clock", "wakeup", "clock_alarms"], "sms": ["text_messages", "sms_app", "short_messages"],
    "ewallet": ["wallet", "e_wallet_app", "digital_wallet"], "weather": ["forecast", "weather_app", "climate_info"],
    "banking": ["bank", "mobile_banking", "bank_app"], "tasks": ["todo", "task_manager", "checklist"],
    "ride_hailing": ["taxi", "ride_booking", "car_hailing"], "notes": ["notebook", "memo_pad", "note_app"],
    "food_delivery": ["meal_delivery", "food_order", "restaurant_delivery"], "email": ["mail", "email_client", "inbox"],
    "learning": ["courses", "study_app", "e_learning"], "home_security": ["security_system", "home_guard", "guard_app"],
    "phone": ["dialer", "calls", "phone_app"], "maps": ["navigation", "map_app", "directions"],
    "home_automation": ["automation", "home_rules", "smart_routines"], "door_lock": ["smart_lock", "door_access", "lock_app"],
    "robot_vacuum": ["vacuum_robot", "cleaner_bot", "robo_cleaner"], "gold_price": ["gold_rates", "gold_market", "gold_quotes"],
    "flights": ["air_travel", "airline_booking", "flight_booking"], "video_meeting": ["video_call", "meetings", "conference_app"],
    "fitness": ["workout", "health_tracker", "exercise_app"], "news": ["news_feed", "headlines", "newspaper"],
    "bike_rental": ["bike_share", "bicycle_hire", "cycle_rental"], "subscriptions": ["memberships", "recurring_plans", "subscription_manager"],
    "appliances": ["home_appliances", "devices_app", "appliance_control"], "loyalty": ["rewards", "points_program", "membership_points"],
    "household_chores": ["chores", "housework", "home_tasks"], "podcast_radio": ["radio", "podcasts", "audio_shows"],
    "medication": ["medicine", "pill_reminder", "meds"], "home_energy": ["energy_monitor", "power_usage", "electricity"],
    "music": ["music_player", "songs", "music_app"], "package_tracking": ["parcels", "delivery_tracking", "shipment_tracker"],
    "shopping_list": ["buy_list", "shopping_notes", "purchase_list"], "library": ["library_app", "book_loans", "reading_library"],
    "exchange_rate": ["currency_rates", "forex", "fx_rates"], "calculator": ["calc", "math_tool", "calculator_app"],
    "lottery": ["lotto", "lottery_results", "xo_so"], "tv": ["television", "tv_guide", "tv_remote"],
    "fuel_price": ["petrol_price", "gas_price", "fuel_rates"],
}


_UNITS = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _spell(n: int) -> str:
    """Vietnamese reading of 0 <= n < 10^9, the way Gipformer writes it (one hundred twenty two -> một trăm hai mươi hai)."""
    if n == 0:
        return "không"
    out = []
    for scale, word in ((10**6, "triệu"), (1000, "nghìn"), (1, "")):
        k, n = divmod(n, scale)
        if not k:
            continue
        h, r = divmod(k, 100)
        t, u = divmod(r, 10)
        part = []
        if h or out:
            part += [_UNITS[h] if h else "không", "trăm"]
        if t == 0 and u and (h or out):
            part.append("linh")
        if t == 1:
            part.append("mười")
        elif t:
            part += [_UNITS[t], "mươi"]
        if u:
            part.append("mốt" if u == 1 and t > 1 else "lăm" if u == 5 and t else "tư" if u == 4 and t > 1 else _UNITS[u])
        out += part + ([word] if word else [])
    return " ".join(out)


def pseudo_asr(text: str) -> str:
    """ASR-like copy of a written utterance for samples without audio (synthetic split)."""
    text = re.sub(r"\b\d{1,9}\b", lambda m: _spell(int(m.group())), text.lower())
    return " ".join(re.sub(r"[,.!?;:\"()]", " ", text).split())


def rename_apps(row: dict, apps: set[str], rng: random.Random) -> tuple[dict, dict]:
    """Rename every app namespace of one trajectory consistently: function aliases (app__tool), official names and x-ref values
    (app.tool, app.collection) in tools, prompts, history and observations. sys stays."""
    mapping = {a: rng.choice(ALIASES.get(a, [f"{a}_app", f"my_{a}", f"{a}_service"])) for a in sorted(apps - {"sys"})}
    text = json.dumps(row, ensure_ascii=False)
    for app, new in mapping.items():
        text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(app) + r"(?=\.[a-z_]|__)", new, text)
    return json.loads(text), mapping


def trajectory(item: dict, root: Path, specs: dict, transcripts: dict[str, str], orchestration: str, hidden: frozenset = frozenset(), split: str = "train") -> dict:
    available = [s for s in installed_tools(item, specs) if s["name"].split(".")[0] not in hidden]
    names = {t["name"] for t in available}
    structured, _ = structured_history(item["history"], lambda a: transcripts[a], {s["name"]: s.get("kind", "unknown") for s in available})
    messages = [{"role": "system", "content": prompt_for(orchestration)},
                {"role": "system", "content": GUIDANCE + "\n" + json.dumps(structured, ensure_ascii=False)},
                user_message(item, transcripts[item["audio"]], orchestration, available)]
    env = new_env(root, split, item["id"])
    gold = item["label"]["tool_calls"]
    for k, call in enumerate(gold):
        if call["name"] not in names or not env.validate(call["name"], call["arguments"])["ok"]:
            raise ValueError(f"unavailable or invalid gold call {k}: {call['name']}")
        observed = env.call(call["name"], call["arguments"])
        if observed != item["label"]["tool_results"][k]:
            raise ValueError(f"ToolEnv observation differs from the label at call {k}")
        alias = call["name"].replace(".", "__")
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": alias, "arguments": call["arguments"]}}]})
        messages.append({"role": "tool", "name": alias, "content": json.dumps(observed, ensure_ascii=False)})
    if env.get_tool_calls() != gold:
        raise ValueError("ToolEnv replay diverged from the gold calls")
    messages.append({"role": "assistant", "content": " "})     # the stop decision, as in v1
    return {"messages": messages, "tools": _tool_defs(available)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=PROJECT / "data/train+public_test")
    ap.add_argument("--validation-ids", type=Path, default=PROJECT / "configs/splits/validation_ids.json")
    ap.add_argument("--include-validation", action="store_true", help="final training: do not exclude the 200 validation tasks")
    ap.add_argument("--asr-cache", type=Path, default=PROJECT / "outputs/asr/gipformer_train/asr_outputs.jsonl")
    ap.add_argument("--orchestration", choices=["E1", "E1D", "E1P"], default="E1P")
    ap.add_argument("--variants", default="gold,asr")
    ap.add_argument("--exclude-apps", default="", help="comma list: drop samples whose gold calls use these apps (held-out-app study)")
    ap.add_argument("--strip-held", action="store_true", help="with --exclude-apps: also drop samples whose history calls those apps and remove "
                    "their schemas from every other prompt, so the held apps are truly unseen")
    ap.add_argument("--rename-apps", type=float, default=0.0, help="share of trajectories shown under other app namespaces (ALIASES)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="train", help="train, or synth (outputs/synth built by scripts/data/build_synth_split.py; use --data-root outputs/synth)")
    a = ap.parse_args()
    root = a.data_root
    forbidden = set(json.loads(a.validation_ids.read_text()))
    if a.include_validation:          # final training: the 200 validation tasks are trained too (no held-out val left)
        forbidden = set()
    elif len(forbidden) != 200:
        raise ValueError("validation split must contain exactly 200 unique TRAIN IDs")
    held = {x for x in a.exclude_apps.split(",") if x}
    specs = catalog(root, a.split)
    asr = {}
    for line in (a.asr_cache.open(encoding="utf-8") if a.split == "train" else []):
        r = json.loads(line)
        asr[(r["sample_id"], r["audio"])] = normalize_asr(r["raw_transcript"])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    out = {"train": (a.output_dir / "train.jsonl").open("w", encoding="utf-8"), "dev": (a.output_dir / "dev.jsonl").open("w", encoding="utf-8")}
    counts, skipped = Counter(), []
    for item in items(root, a.split):
        if item["id"] in forbidden or history_audio_ids(item) & forbidden:
            counts["excluded_validation"] += 1
            continue
        if held and {c["name"].split(".")[0] for c in item["label"]["tool_calls"]} & held:
            counts["excluded_held_apps"] += 1
            continue
        if held and a.strip_held and {c["name"].split(".")[0] for t in item["history"] for c in t.get("tool_calls") or []} & held:
            counts["excluded_held_history"] += 1
            continue
        bucket = "dev" if int(hashlib.sha256(f"42:{item['id']}".encode()).hexdigest(), 16) % 10 == 0 else "train"
        audios = [item["audio"], *(t["audio"] for t in item["history"] if t.get("role") == "user" and "audio" in t)]
        gold_text = {item["audio"]: item["audio_text"], **{t["audio"]: t["audio_text"] for t in item["history"] if t.get("role") == "user" and "audio" in t}}
        for variant in a.variants.split(","):
            try:
                if variant == "gold":
                    text = gold_text
                elif variant == "pseudo":       # no audio: ASR-like rendering of the gold text (lower case, no punctuation, numbers spelled)
                    text = {x: pseudo_asr(t) for x, t in gold_text.items()}
                else:
                    text = {x: asr[(item["id"], x)] for x in audios}
                row = trajectory(item, root, specs, text, a.orchestration, frozenset(held) if a.strip_held else frozenset(), split=a.split)
            except (ValueError, KeyError) as exc:
                skipped.append({"sample_id": item["id"], "variant": variant, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            rng = random.Random(f"{item['id']}:{variant}")
            renamed = {}
            if a.rename_apps and rng.random() < a.rename_apps:
                apps = {t["function"]["name"].split("__")[0] for t in row["tools"]}
                row, renamed = rename_apps(row, apps, rng)
                counts["renamed"] += 1
            out[bucket].write(json.dumps({"sample_id": item["id"], "variant": variant, "renamed": renamed, **row}, ensure_ascii=False) + "\n")
            counts[f"{bucket}_{variant}"] += 1
    for f in out.values():
        f.close()
    manifest = {"counts": dict(counts), "skipped": skipped, "orchestration": a.orchestration, "variants": a.variants,
                "exclude_apps": sorted(held), "strip_held": a.strip_held, "rename_apps": a.rename_apps, "asr_cache": str(a.asr_cache),
                "validation_ids_sha256": hashlib.sha256(a.validation_ids.read_bytes()).hexdigest(), "validation_excluded": len(forbidden)}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "skipped"} | {"skipped": len(skipped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
