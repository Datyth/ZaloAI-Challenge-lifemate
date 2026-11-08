"""Build the synthetic split outputs/synth/synth/ in the layout of the organizers' kit, so ToolEnv("synth", id, root) and the SFT
builder work on it unchanged.

    .venv/bin/python scripts/build_synth_split.py [--out outputs/synth] [--exclude-apps a,b] [--seed 7]

- zaic_tools/ is a copy of the TRAIN package plus synth/apps/synth_apps.py; the catalog is the TRAIN catalog plus synth/catalog.py.
- Each task (synth/tasks/*.py, TASKS) gives the app states it needs, the time, the utterance and the gold calls. The other installed
  apps (distractors, ~9 apps per sample as in TRAIN) and the user profile come from a random TRAIN sample outside the 200 validation
  IDs, read through ToolEnv.get_state() (allowed on TRAIN); no environment file is opened directly.
- Gold calls are replayed with the official Session: every call must pass the schema check, writes must not be no-ops, and the
  results / final_state are stored as the label, like TRAIN."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
KIT = PROJECT / "data/train+public_test"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "synth/tasks"))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_contexts(n_needed: int, exclude: set[str], seed: int) -> list[dict]:
    """(context, states) of random TRAIN samples outside validation, via a child process (one zaic_tools copy per process)."""
    code = r'''
import json, sys, random
sys.path.insert(0, sys.argv[1] + "/train")
from zaic_tools import ToolEnv
val = set(json.load(open(sys.argv[2])))
exclude = set(x for x in sys.argv[3].split(",") if x)
items = [json.loads(l) for l in open(sys.argv[1] + "/train/train.jsonl", encoding="utf-8")]
items = [x for x in items if x["id"] not in val]
rng = random.Random(int(sys.argv[4]))
out = []
for x in rng.sample(items, min(int(sys.argv[5]), len(items))):
    st = ToolEnv("train", x["id"], root=sys.argv[1]).get_state()
    apps = {a: v for a, v in x["context"]["installed_apps"].items() if a not in exclude}
    out.append({"from": x["id"], "context": {**x["context"], "installed_apps": apps}, "states": {a: st[a] for a in apps if a in st}})
print(json.dumps(out, ensure_ascii=False))
'''
    res = subprocess.run([sys.executable, "-c", code, str(KIT), str(PROJECT / "outputs/splits/validation_ids.json"), ",".join(sorted(exclude)),
                          str(seed), str(n_needed)], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/synth")
    ap.add_argument("--exclude-apps", default="", help="TRAIN apps never installed as distractors (held-out-app study)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    exclude = {x for x in a.exclude_apps.split(",") if x}
    cat_mod = load(PROJECT / "synth/catalog.py")
    tasks = []
    for f in sorted((PROJECT / "synth/tasks").glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod = load(f)
        for t in mod.TASKS:
            tasks.append({**t, "_file": f.stem})
    split = a.out / "synth"
    if split.exists():
        shutil.rmtree(split)
    (split / "synth_environment").mkdir(parents=True)
    shutil.copytree(KIT / "train/zaic_tools", split / "zaic_tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PROJECT / "synth/apps/synth_apps.py", split / "zaic_tools/apps/synth_apps.py")
    catalog = json.loads((KIT / "train/train_tools_catalog.json").read_text(encoding="utf-8"))
    catalog["apps"] = catalog["apps"] + cat_mod.APPS
    (split / "synth_tools_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    tools_of = {x["id"]: [t["name"] for t in x["tools"]] for x in catalog["apps"]}
    bases = train_contexts(len(tasks), exclude, a.seed)
    rng = random.Random(a.seed)
    items = []
    for k, (t, base) in enumerate(zip(tasks, bases)):
        sid = f"synth_{k + 1:04d}"
        ctx = copy.deepcopy(base["context"])
        states = copy.deepcopy(base["states"])
        for app_id, st in t["states"].items():          # task apps (synthetic or TRAIN) replace the distractor state
            states[app_id] = copy.deepcopy(st)
            ctx["installed_apps"][app_id] = {"tools": tools_of[app_id]}
        for app_id in t.get("uninstall", []):            # out-of-scope tasks: make sure no installed app can do it
            states.pop(app_id, None)
            ctx["installed_apps"].pop(app_id, None)
        ctx["current_time"] = t["time"]
        if t.get("profile"):
            ctx["user_profile"] = {**ctx["user_profile"], **t["profile"]}
        names = sorted({n for v in ctx["installed_apps"].values() for n in v["tools"]})
        env_dir = split / "synth_environment" / sid
        env_dir.mkdir()
        meta = {"question_id": sid, "current_time": t["time"], "tools": names, "city": ctx["user_profile"].get("city")}
        (env_dir / "state.json").write_text(json.dumps({"_meta": meta, **states}, ensure_ascii=False), encoding="utf-8")
        items.append({"id": sid, "audio": f"audio/{sid}/{sid}.wav", "context": ctx, "history": [], "audio_text": t["text"],
                      "label": {"tool_calls": t["gold"]}, "synth": {"task": t.get("name"), "file": t["_file"], "context_from": base["from"]}})
    # replay the gold calls with the official Session of the new package (child process: separate zaic_tools copy)
    (split / "synth.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    code = r'''
import json, sys, copy
sys.path.insert(0, sys.argv[1])
from zaic_tools import ToolEnv, TOOLS
rows, bad = [], []
for line in open(sys.argv[1] + "/synth.jsonl", encoding="utf-8"):
    x = json.loads(line)
    env = ToolEnv("synth", x["id"], root=sys.argv[2])
    before = env.get_state()
    results, err = [], None
    for c in x["label"]["tool_calls"]:
        v = env.validate(c["name"], c["arguments"])
        if not v["ok"]:
            err = f"schema: {c['name']}: {v['error']}"; break
        r = env.call(c["name"], c["arguments"])
        if TOOLS.get(c["name"]) == "write" and r.get("done") is False:
            err = f"no-op write: {c['name']}: {r.get('message')}"; break
        results.append(r)
    if err:
        bad.append({"id": x["id"], "task": x["synth"]["task"], "error": err}); continue
    after = env.get_state()
    x["label"]["tool_results"] = results
    x["label"]["final_state"] = {a: s for a, s in after.items() if s != before.get(a)}
    rows.append(x)
open(sys.argv[1] + "/synth.jsonl", "w", encoding="utf-8").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
print(json.dumps({"ok": len(rows), "bad": bad}, ensure_ascii=False))
'''
    res = subprocess.run([sys.executable, "-c", code, str(split), str(a.out)], capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stderr)
    report = json.loads(res.stdout)
    for b in report["bad"]:
        print("BAD", b)
    print(f"{report['ok']} samples ok, {len(report['bad'])} rejected; split at {split}")


if __name__ == "__main__":
    main()
