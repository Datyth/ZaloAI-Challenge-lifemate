"""Readable dump of public run trajectories (label-free): transcript, context time, apps (unseen marked *), each tool call with a
short result, and the termination. Usage: pub_dump.py <run glob> <out.txt>"""
import ast, glob, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "src")
from lifemate.data import items
DATA = Path("data/train+public_test")
def apps(split): return {a["id"] for a in json.load(open(DATA / split / f"{split}_tools_catalog.json"))["apps"]}
seen = apps("train")
pat, out = sys.argv[1], sys.argv[2]
def load(name):
    rows = []
    for p in sorted(glob.glob(pat)):
        f = Path(p) / name
        if f.exists(): rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows
ev = defaultdict(list)
for e in load("tool_events.jsonl"): ev[e["sample_id"]].append(e)
asr = {(r["sample_id"], r["audio"]): r["raw_transcript"].lower() for r in load("asr_outputs.jsonl")}
term = {r["sample_id"]: r for r in load("terminations.jsonl")}
its = {it["id"]: it for it in items(DATA, "public_test")}
def short(x, n=260):
    s = json.dumps(x, ensure_ascii=False); return s if len(s) <= n else s[:n] + "…"
rows = []
for sid, it in its.items():
    ctx = it["context"]; ctx = ast.literal_eval(ctx) if isinstance(ctx, str) else ctx
    inst = ctx.get("installed_apps", [])
    called = {e["name"].split(".")[0] for e in ev[sid] if e.get("name")}
    uns = sorted(a for a in called if a not in seen and a != "sys")
    empty = any(e.get("result") is not None and (e["result"].get("ok") is False or any(isinstance(v, list) and not v for v in e["result"].values())) for e in ev[sid])
    lines = [f"=== {sid}  time {ctx.get('current_time')}  term {term.get(sid, {}).get('status')}  unseen_called {uns}  empty {empty}",
             f"apps: {' '.join(a + ('*' if a not in seen else '') for a in inst)}"]
    for t in it["history"]:
        if t.get("role") == "user":
            lines.append(f"  H user: {asr.get((sid, t.get('audio')), t.get('audio_text', '?'))}")
        else:
            lines.append(f"  H {t.get('role')}: {short({k: v for k, v in t.items() if k != 'role'}, 200)}")
    lines.append(f"  USER: {asr.get((sid, it['audio']), '?')}")
    for e in ev[sid]:
        v = "" if e["validation"]["ok"] else f" BLOCKED {e['validation']['error']}"
        lines.append(f"  -> {e.get('name') or e['proposed']['function']['name']} {short(e.get('arguments') or e['proposed']['function']['arguments'], 200)}{v}")
        if e.get("result") is not None: lines.append(f"     = {short(e['result'])}")
    rows.append({"id": sid, "uns": bool(uns), "empty": empty, "term": term.get(sid, {}).get("status"), "text": "\n".join(lines)})
json.dump(rows, open(out.replace(".txt", ".json"), "w"), ensure_ascii=False)
open(out, "w").write("\n\n".join(r["text"] for r in rows))
from collections import Counter
print(len(rows), Counter((r["uns"], r["empty"]) for r in rows), Counter(r["term"] for r in rows))
