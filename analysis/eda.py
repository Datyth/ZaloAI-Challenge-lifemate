"""Overview of TRAIN and PUBLIC_TEST inputs and TRAIN labels (never reads *_environment/). python analysis/eda.py"""
import collections
import json
import statistics as st
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/train+public_test"


def load(split):
    return [json.loads(l) for l in open(ROOT / split / f"{split}.jsonl", encoding="utf-8")]


def catalog(split):
    raw = json.loads((ROOT / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    return {a["id"]: a for a in raw["apps"]}, {t["name"]: t for a in raw["apps"] for t in a["tools"]}


def dur(split, rel):
    with wave.open(str(ROOT / split / rel), "rb") as w:
        return w.getnframes() / w.getframerate()


def q(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(st.mean(xs), 2), "p50": xs[len(xs) // 2], "p90": xs[int(len(xs) * .9)], "max": xs[-1]}


tr, pu = load("train"), load("public_test")
apps_tr, tools_tr = catalog("train")
apps_pu, tools_pu = catalog("public_test")
print("apps train", len(apps_tr), "public", len(apps_pu))
print("public-only apps:", sorted(set(apps_pu) - set(apps_tr)))
print("train-only apps:", sorted(set(apps_tr) - set(apps_pu)))
shared = set(apps_tr) & set(apps_pu)
diff_tools = [a for a in shared if {t["name"] for t in apps_tr[a]["tools"]} != {t["name"] for t in apps_pu[a]["tools"]}]
print("shared apps with different tool sets:", diff_tools)
for split, rows in (("train", tr), ("public_test", pu)):
    hist_turns = [len(r["history"]) for r in rows]
    user_turns = [sum(1 for t in r["history"] if t.get("role") == "user") for r in rows]
    napps = [len(r["context"]["installed_apps"]) for r in rows]
    ntools = [sum(len(a["tools"]) for a in r["context"]["installed_apps"].values()) for r in rows]
    durs = [dur(split, r["audio"]) for r in rows]
    used = collections.Counter(a for r in rows for a in r["context"]["installed_apps"])
    print(f"\n== {split}: {len(rows)} samples")
    print(" history turns", q(hist_turns), " user turns", q(user_turns))
    print(" history buckets", collections.Counter(min(u, 6) for u in user_turns))
    print(" installed apps", q(napps), " installed tools", q(ntools))
    print(" current audio sec", q([round(d, 2) for d in durs]), " <2s:", sum(d < 2 for d in durs), " <3s:", sum(d < 3 for d in durs))
    if split == "public_test":
        new = [r for r in rows if set(r["context"]["installed_apps"]) - set(apps_tr)]
        print(" samples with a public-only app installed:", len(new))
calls = [r["label"]["tool_calls"] for r in tr]
print("\n== TRAIN labels")
print(" calls per sample", collections.Counter(min(len(c), 6) for c in calls))
print(" empty:", sum(not c for c in calls), " ask_missing_info:", sum(any(x["name"] == "sys.ask_missing_info" for x in c) for c in calls))
kinds = collections.Counter(tools_tr.get(x["name"], {}).get("kind", "?") for c in calls for x in c)
print(" call kinds", kinds)
print(" samples with a write", sum(any(tools_tr.get(x["name"], {}).get("kind") == "write" for x in c) for c in calls),
      " read-only", sum(bool(c) and all(tools_tr.get(x["name"], {}).get("kind") == "read" for x in c) for c in calls))
apps_used = collections.Counter(len({x["name"].split(".")[0] for x in c if x["name"] != "sys.ask_missing_info"}) for c in calls)
print(" distinct apps per sample", apps_used)
tc = collections.Counter(x["name"] for c in calls for x in c)
print(" top tools", tc.most_common(30))
ac = collections.Counter(x["name"].split(".")[0] for c in calls for x in c)
print(" calls by app", ac.most_common())
fuzzy = collections.Counter()
for c in calls:
    for x in c:
        props = (tools_tr.get(x["name"], {}).get("parameters") or {}).get("properties", {})
        for k in x["arguments"]:
            if props.get(k, {}).get("x-match") == "fuzzy":
                fuzzy[f"{x['name']}.{k}"] += 1
print(" fuzzy args in gold", fuzzy.most_common(20))
# history: do labels relate to history (reference resolution)?
print(" samples with history", sum(bool(r["history"]) for r in tr), " assistant turns with tool calls",
      sum(1 for r in tr for t in r["history"] if t.get("role") == "assistant" and t.get("tool_calls")))
print(" audio_text length words", q([len(r["audio_text"].split()) for r in tr]))
