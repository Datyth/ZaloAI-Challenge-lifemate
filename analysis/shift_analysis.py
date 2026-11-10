"""Distribution shift between TRAIN/val and public (label-free on public: inputs and our own predictions only), tied to val scores.

1. Request features from the inputs the agent sees (ASR transcript, history, installed apps): length, multi-intent markers,
   conditionals, relative dates / weekdays, numbers, cancellations ("à thôi"), history turns, number of installed apps.
2. Val score (with labels) per feature bin, and the share of each bin in val and public. The "reweighted val" score is the val
   score per bin averaged with public's bin shares: what val would score if its requests looked like public's on that feature.
3. A linear model of val score on all features, applied to public requests: the expected public score from seen-app behaviour.
4. Confusion of call-list shapes (gold vs predicted) on val, and predicted shapes on public vs gold shapes on TRAIN.
    python analysis/shift_analysis.py --val-run v200_q35e1_asrv2_dedup --pub-run pub_q35e1_asrv2_dedup \
        --val-cache outputs/asr/gipformer-lm-v2-01_val/asr_outputs.jsonl --pub-cache outputs/asr/gipformer-lm-v2-01_public/asr_outputs.jsonl
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from lifemate.data import catalog, items  # noqa: E402

DATA = Path("data/train+public_test")
WORDS = re.compile(r"[^\W_]+", re.U)
MULTI = re.compile(r"\b(rồi|xong|sau đó|với lại|luôn|thêm|còn|và)\b")
COND = re.compile(r"\b(nếu|hễ|trường hợp|lỡ mà|mà .{0,30} thì)\b")
RELDATE = re.compile(r"\b(mai|ngày kia|mốt|tuần sau|tuần tới|tuần này|tháng sau|tối nay|sáng nay|chiều nay|trưa nay|hôm qua|cuối tuần|đầu tháng)\b")
WEEKDAY = re.compile(r"\b(thứ (hai|ba|tư|năm|sáu|bảy)|chủ nhật)\b")
NUMW = re.compile(r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|trăm|nghìn|ngàn|triệu|rưỡi|lăm|mốt|tư)\b")
CANCEL = re.compile(r"\b(à thôi|thôi khỏi|khỏi|thôi đừng|không cần|hủy|huỷ)\b")


def features(text: str, item: dict, seen_apps: set) -> dict:
    t = text.lower()
    w = WORDS.findall(t)
    apps = set(item["context"]["installed_apps"])
    return {"words": len(w), "multi": len(MULTI.findall(t)), "cond": int(bool(COND.search(t))), "reldate": int(bool(RELDATE.search(t))),
            "weekday": int(bool(WEEKDAY.search(t))), "numwords": len(NUMW.findall(t)), "cancel": int(bool(CANCEL.search(t))),
            "history": sum(1 for x in item["history"] if x.get("role") == "user"), "apps": len(apps),
            "unseen_installed": len(apps - seen_apps)}


BINS = {"words": [0, 10, 15, 20, 25, 35, 999], "multi": [0, 1, 2, 3, 99], "numwords": [0, 1, 3, 6, 99], "history": [0, 1, 3, 6, 99],
        "apps": [0, 6, 9, 12, 99], "cond": [0, 1, 2], "reldate": [0, 1, 2], "weekday": [0, 1, 2], "cancel": [0, 1, 2]}


def binlabel(f: str, v: int) -> str:
    edges = BINS[f]
    for a, b in zip(edges, edges[1:]):
        if a <= v < b:
            return f"{a}" if b == a + 1 else f"{a}-{b - 1}"
    return f">={edges[-1]}"


def shape(calls, specs):
    if not calls:
        return "empty"
    k = ["ask" if c["name"] == "sys.ask_missing_info" else specs.get(c["name"], {}).get("kind", "?") for c in calls]
    if k == ["ask"]:
        return "ask_only"
    if "ask" in k:
        return "reads_then_ask"
    if all(x == "read" for x in k):
        return "read_only"
    if all(x == "write" for x in k):
        return "write_only"
    return "read_write"


def load_preds(run):
    out = {}
    for f in glob.glob(f"outputs/runs/{run}_s*/predictions.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r
    return out


def load_cache(path):
    return {(r["sample_id"], r["audio"]): r["raw_transcript"] for r in map(json.loads, open(path, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--pub-run", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--pub-cache", required=True)
    ap.add_argument("--public-score", type=float, help="actual public score of --pub-run, for the gap line")
    a = ap.parse_args()
    seen = {x["id"] for x in json.load(open(DATA / "train/train_tools_catalog.json"))["apps"]}
    specs = {**catalog(DATA, "train"), **catalog(DATA, "public_test")}
    val_ids = set(json.load(open("outputs/splits/validation_ids.json")))
    scores = {json.loads(l)["id"]: json.loads(l)["score"] for l in open(f"outputs/analysis/{a.val_run}.scores.jsonl")}
    vcache, pcache = load_cache(a.val_cache), load_cache(a.pub_cache)
    val, train_feats, train_shapes = [], [], Counter()
    for it in items(DATA, "train"):
        g = it["label"]["tool_calls"]
        if it["id"] in val_ids:
            val.append((it, features(vcache[(it["id"], it["audio"])], it, seen), scores[it["id"]], shape(g, specs)))
        else:
            train_feats.append(features(it["audio_text"], it, seen))
            train_shapes[shape(g, specs)] += 1
    ppred = load_preds(a.pub_run)
    vpred = load_preds(a.val_run)
    pub = [(it, features(pcache[(it["id"], it["audio"])], it, seen)) for it in items(DATA, "public_test")]
    print(f"val {len(val)}, public {len(pub)}, train (non-val) {len(train_feats)}\n")

    print("## 1. Feature means (TRAIN gold text / val ASR / public ASR)")
    for f in BINS:
        tm = np.mean([x[f] for x in train_feats]); vm = np.mean([x[1][f] for x in val]); pm = np.mean([x[1][f] for x in pub])
        print(f"  {f:16s} train {tm:6.2f}  val {vm:6.2f}  public {pm:6.2f}  public/val {pm / vm if vm else float('nan'):5.2f}")
    print(f"  {'unseen_installed':16s} val {np.mean([x[1]['unseen_installed'] for x in val]):.2f}  public {np.mean([x[1]['unseen_installed'] for x in pub]):.2f}")

    print("\n## 2. Val score by bin, with the share of that bin in val and public")
    base = 100 * np.mean([s for _, _, s, _ in val])
    rows = []
    for f in BINS:
        vb = defaultdict(list)
        for _, x, s, _ in val:
            vb[binlabel(f, x[f])].append(s)
        pb = Counter(binlabel(f, x[f]) for _, x in pub)
        reweighted = sum(pb[b] / len(pub) * np.mean(vb[b]) for b in pb if b in vb) / max(1e-9, sum(pb[b] / len(pub) for b in pb if b in vb))
        print(f"  {f}: reweighted val {100 * reweighted:.1f} (val {base:.1f})")
        for b in sorted(set(vb) | set(pb), key=lambda z: float(z.split('-')[0].lstrip('>='))):
            vs = f"{100 * np.mean(vb[b]):5.1f}" if b in vb else "   - "
            print(f"     {b:>7s}: val score {vs}  val share {len(vb.get(b, [])) / len(val):5.1%}  public share {pb[b] / len(pub):5.1%}")
        rows.append((f, 100 * reweighted))

    print("\n## 3. Linear model of val score on all features -> expected public score from seen-app behaviour")
    keys = list(BINS)
    X = np.array([[x[k] for k in keys] for _, x, _, _ in val], float); y = np.array([s for _, _, s, _ in val])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = np.c_[np.ones(len(X)), (X - mu) / sd]
    lam = 5.0
    beta = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ y)
    P = np.array([[x[k] for k in keys] for _, x in pub], float)
    pred_pub = np.clip(np.c_[np.ones(len(P)), (P - mu) / sd] @ beta, 0, 1)
    print("  coefficients (per 1 sd):", {k: round(100 * b, 1) for k, b in zip(keys, beta[1:])})
    uns = np.array([x["unseen_installed"] > 0 and any(c["name"].split(".")[0] not in seen and c["name"].split(".")[0] != "sys"
                                                       for c in ppred.get(it["id"], {}).get("tool_calls", [])) for it, x in pub])
    print(f"  expected public from features: {100 * pred_pub.mean():.1f}  (val actual {base:.1f})")
    print(f"     samples calling an unseen app ({uns.sum()}): expected {100 * pred_pub[uns].mean():.1f}; others ({(~uns).sum()}): {100 * pred_pub[~uns].mean():.1f}")
    if a.public_score:
        print(f"  actual public {a.public_score:.2f}: gap to feature-expected {100 * pred_pub.mean() - a.public_score:.1f} points not explained by these features")

    print("\n## 4. Call-list shapes")
    conf = Counter((g, shape(vpred[it["id"]]["tool_calls"], specs)) for it, _, _, g in val)
    shapes = ["empty", "ask_only", "reads_then_ask", "read_only", "write_only", "read_write"]
    print("  val confusion (rows gold, cols predicted):")
    print("   " + " " * 15 + "".join(f"{s[:10]:>11s}" for s in shapes))
    for g in shapes:
        print(f"   {g:15s}" + "".join(f"{conf[(g, p)]:11d}" for p in shapes))
    vs = defaultdict(list)
    for it, _, s, g in val:
        vs[g].append(s)
    print("  val score by gold shape:", {g: f"{100 * np.mean(v):.1f} (n={len(v)})" for g, v in vs.items()})
    ps = Counter(shape(ppred[it["id"]]["tool_calls"], specs) for it, _ in pub if it["id"] in ppred)
    vp = Counter(shape(vpred[it["id"]]["tool_calls"], specs) for it, _, _, _ in val)
    tot = sum(train_shapes.values())
    print("  shape shares: TRAIN gold | val predicted | public predicted")
    for s in shapes:
        print(f"     {s:15s} {train_shapes[s] / tot:6.1%} | {vp[s] / len(val):6.1%} | {ps[s] / len(pub):6.1%}")


if __name__ == "__main__":
    main()
