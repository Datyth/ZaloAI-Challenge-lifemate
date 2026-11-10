"""Relative weekday phrases ("thứ bảy tuần sau", "chủ nhật này") in TRAIN gold: which date the label uses, relative to
context current_time; then the same check on q35e1 val (gold known) and public (label-free: does any arg date equal the
TRAIN convention?)."""
import ast, glob, json, re, sys, datetime as dt
from collections import Counter
from pathlib import Path
sys.path.insert(0, "src")
from lifemate.data import items
DATA = Path("data/train+public_test")
WD = {"hai": 0, "ba": 1, "tư": 2, "năm": 3, "sáu": 4, "bảy": 5, "chủ nhật": 6}
PAT = re.compile(r"(thứ (hai|ba|tư|năm|sáu|bảy)|chủ nhật)\s+(tuần\s+(sau|tới|này|nay)|này|tới)", re.I)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
def ctx(it):
    c = it["context"]; return ast.literal_eval(c) if isinstance(c, str) else c
def dates_in(calls):
    return {d for c in calls for d in DATE.findall(json.dumps(c["arguments"], ensure_ascii=False))}
def phrase(text):
    m = PAT.search(text.lower())
    if not m: return None
    wd = WD["chủ nhật"] if m.group(1).startswith("chủ") else WD[m.group(2)]
    rel = m.group(4) or m.group(3)
    rel = "sau" if rel in ("sau", "tới") else "này"
    return wd, rel, m.group(0)
def candidates(now, wd, rel):
    # A: calendar week (Mon-Sun): "này" = this week, "sau" = next week; B: next occurrence (+7 for "sau")
    mon = now - dt.timedelta(days=now.weekday())
    a = mon + dt.timedelta(days=wd + (7 if rel == "sau" else 0))
    nxt = now + dt.timedelta(days=(wd - now.weekday()) % 7 or 7)
    b = nxt + dt.timedelta(days=7) if rel == "sau" else nxt
    return {"week": a.isoformat(), "next_occ": b.isoformat()}
tr = Counter(); ex = []
for it in items(DATA, "train"):
    p = phrase(it["audio_text"])
    if not p: continue
    now = dt.date.fromisoformat(ctx(it)["current_time"][:10])
    cand = candidates(now, *p[:2]); ds = dates_in(it["label"]["tool_calls"])
    if not ds: continue
    hit = tuple(sorted(k for k, v in cand.items() if v in ds)) or ("none",)
    tr[(p[1], hit)] += 1
    if hit == ("none",) and len(ex) < 6: ex.append((it["id"], p[2], now.isoformat(), now.strftime("%a"), cand, sorted(ds)[:4]))
print("TRAIN gold:", sorted(tr.items()))
for e in ex: print("  ", e)
def run(pattern, split, scores=None):
    preds = {}
    for f in glob.glob(pattern + "/predictions.jsonl"):
        for l in open(f): r = json.loads(l); preds[r["id"]] = r
    asr = {}
    for f in glob.glob(pattern + "/asr_outputs.jsonl"):
        for l in open(f): r = json.loads(l); asr[(r["sample_id"], r["audio"])] = r["raw_transcript"].lower()
    sc = {json.loads(l)["id"]: json.loads(l) for l in open(scores)} if scores else {}
    out = Counter(); bad = []
    for it in items(DATA, split):
        if it["id"] not in preds: continue
        text = it.get("audio_text") or asr.get((it["id"], it["audio"]), "")
        p = phrase(text)
        if not p: continue
        now = dt.date.fromisoformat(ctx(it)["current_time"][:10])
        cand = candidates(now, *p[:2]); ds = dates_in(preds[it["id"]]["tool_calls"])
        if not ds: out[(p[1], "no_date")] += 1; continue
        hit = tuple(sorted(k for k, v in cand.items() if v in ds)) or ("none",)
        out[(p[1], hit)] += 1
        if hit == ("none",): bad.append((it["id"], p[2], now.isoformat(), now.strftime("%a"), cand, sorted(ds)[:3], sc.get(it["id"], {}).get("score")))
    return out, bad
# usage: weekday_dates.py [<run prefix> ...]  (outputs/runs/<prefix>_s*; prefixes starting with "pub" are public_test)
runs = sys.argv[1:] or ["v200_q35e1_asrv2_dedup", "pub_q35e1_asrv2_dedup"]
for name, pat, split, sc in [(r, f"outputs/runs/{r}_s*", "public_test" if r.startswith("pub") else "train", None) for r in runs]:
    o, b = run(pat, split, sc)
    print(name, sorted(o.items(), key=str))
    for x in b[:12]: print("  ", x)
