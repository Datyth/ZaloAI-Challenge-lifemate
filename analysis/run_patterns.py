"""Label-free failure patterns on a run (public or val): not-found lookups, empty search then give up, read-only ends,
blocked duplicates, cart without checkout. For val runs, the mean score of each group is printed too."""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
pat = sys.argv[1]; scores = sys.argv[2] if len(sys.argv) > 2 else None
seen = {a["id"] for a in json.load(open("data/train+public_test/train/train_tools_catalog.json"))["apps"]}
ev = defaultdict(list); term = {}
for d in sorted(glob.glob(pat)):
    for l in open(Path(d) / "tool_events.jsonl"): e = json.loads(l); ev[e["sample_id"]].append(e)
    for l in open(Path(d) / "terminations.jsonl"): t = json.loads(l); term[t["sample_id"]] = t["status"]
sc = {json.loads(l)["id"]: json.loads(l)["score"] for l in open(scores)} if scores else {}
def empty(r):
    return isinstance(r, dict) and (r.get("ok") is False or r.get("done") is False or any(isinstance(v, list) and not v for v in r.values()))
groups = defaultdict(set)
for sid in term:
    es = [e for e in ev[sid] if e.get("executed")]
    apps = {e["name"].split(".")[0] for e in es}
    uns = bool(apps - seen - {"sys"})
    tag = "unseen" if uns else "seen"
    groups["all"].add(sid); groups[tag].add(sid)
    if any("không tìm thấy" in json.dumps(e.get("result"), ensure_ascii=False) for e in es): groups["not_found_msg"].add(sid)
    if any(empty(e.get("result")) for e in es):
        groups["hit_empty"].add(sid)
        last = es[-1] if es else None
        if term[sid] in ("completed_read", "step_limit", "duplicate_loop", "no_action") : groups["empty_then_noncompleted"].add(sid)
        k = next(i for i, e in enumerate(es) if empty(e.get("result")))
        nxt = es[k + 1]["name"] if k + 1 < len(es) else "END"
        groups["after_empty:" + ("same_tool" if nxt == es[k]["name"] else "ask" if nxt == "sys.ask_missing_info" else "end" if nxt == "END" else "other")].add(sid)
    if any(not e["validation"]["ok"] for e in ev[sid]): groups["blocked"].add(sid)
    names = [e["name"] for e in es]
    if any(n.endswith("add_to_cart") for n in names) and not any(n.endswith("checkout") for n in names): groups["cart_no_checkout"].add(sid)
    groups["term:" + term[sid]].add(sid)
n = len(term)
for g, s in sorted(groups.items(), key=lambda x: -len(x[1])):
    m = f"  score {sum(sc[i] for i in s if i in sc) / max(1, len([i for i in s if i in sc])):.1f}" if sc else ""
    print(f"{g:28s} {len(s):4d} ({len(s) / n:5.1%}) unseen {len(s & groups['unseen']):4d}{m}")
