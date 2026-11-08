"""Label-free view of a public run split by the apps the agent called (seen in train, unseen, none), next to the same
signals on a val run where the score is known. Uses only run logs, the tool catalogs and installed_apps; no public labels,
no <split>_environment files.

    python analysis/unseen_apps.py --public 'outputs/runs/pub_s1c_asrv2_dedup_s*' --val 'outputs/runs/v200_s1c_asrv2_s*' \
        --val-scores outputs/analysis/v200_s1c_asrv2.scores.jsonl
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data/train+public_test")


def catalog_apps(split):
    cat = json.load(open(DATA / split / f"{split}_tools_catalog.json", encoding="utf-8"))
    return {a["id"] for a in cat["apps"]}


def load(pattern, name):
    rows = []
    for p in sorted(glob.glob(pattern)):
        f = Path(p) / name
        if f.exists():
            rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return rows


def run_view(pattern):
    preds = {r["id"]: r for r in load(pattern, "predictions.jsonl")}
    term = {r["sample_id"]: r for r in load(pattern, "terminations.jsonl")}
    blocked = Counter()
    for e in load(pattern, "tool_events.jsonl"):
        if not e["validation"]["ok"]:
            blocked[e["sample_id"]] += 1
    return preds, term, blocked


def segment(calls, seen):
    apps = {c["name"].split(".")[0] for c in calls} - {"sys"}
    if not calls:
        return "empty"
    if apps - seen:
        return "unseen app called"
    if not apps:
        return "sys only"
    return "seen apps only"


def table(title, ids, preds, term, blocked, seen, scores=None):
    groups = defaultdict(list)
    for i in ids:
        groups[segment(preds[i]["tool_calls"] if i in preds else [], seen)].append(i)
    print(f"\n## {title} ({len(ids)} samples)")
    head = "| segment | n | share | calls/sample | ask_missing | blocked proposals | tool_error | step/dup limit |" + (" score |" if scores else "")
    print(head + "\n|" + "---|" * (head.count("|") - 1))
    for g in ("seen apps only", "unseen app called", "sys only", "empty"):
        xs = groups.get(g, [])
        if not xs:
            continue
        n = len(xs)
        calls = sum(len(preds[i]["tool_calls"]) for i in xs if i in preds) / n
        ask = sum(any(c["name"] == "sys.ask_missing_info" for c in preds[i]["tool_calls"]) for i in xs if i in preds) / n
        blk = sum(blocked[i] > 0 for i in xs) / n
        st = Counter(term[i]["status"] for i in xs if i in term)
        line = (f"| {g} | {n} | {n / len(ids):.0%} | {calls:.2f} | {ask:.0%} | {blk:.0%} | {st['tool_error'] / n:.0%} | "
                f"{(st['step_limit'] + st['duplicate_loop']) / n:.0%} |")
        if scores:
            line += f" {100 * sum(scores[i] for i in xs) / n:.1f} |"
        print(line)
    if scores:
        for flag, test in (("a blocked proposal", lambda i: blocked[i] > 0), ("tool_error status", lambda i: term.get(i, {}).get("status") == "tool_error")):
            xs = [i for i in ids if test(i)]
            if xs:
                print(f"- val samples with {flag}: {len(xs)}, score {100 * sum(scores[i] for i in xs) / len(xs):.1f} "
                      f"(others {100 * sum(scores[i] for i in ids if not test(i)) / max(1, len(ids) - len(xs)):.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-scores", required=True)
    a = ap.parse_args()
    seen = catalog_apps("train") - {"sys"}
    unseen = catalog_apps("public_test") - catalog_apps("train")
    print("unseen apps in public:", ", ".join(sorted(unseen)))

    pub = [json.loads(l) for l in open(DATA / "public_test/public_test.jsonl", encoding="utf-8")]
    inst = {s["id"]: set(s["context"]["installed_apps"]) for s in pub}
    preds, term, blocked = run_view(a.public)
    ids = [s["id"] for s in pub]
    n_inst = sum(bool(inst[i] & unseen) for i in ids)
    print(f"public samples with an unseen app installed: {n_inst}/{len(ids)}")
    table("Public, label-free", ids, preds, term, blocked, seen)
    called = Counter(c["name"] for i in ids for c in preds.get(i, {"tool_calls": []})["tool_calls"] if c["name"].split(".")[0] in unseen)
    print("\nunseen tools called most:", ", ".join(f"{k} {v}" for k, v in called.most_common(15)))

    scores = {r["id"]: r["score"] for r in map(json.loads, open(a.val_scores, encoding="utf-8"))}
    vpreds, vterm, vblocked = run_view(a.val)
    table("Val (all apps seen in train)", list(scores), vpreds, vterm, vblocked, seen, scores)


if __name__ == "__main__":
    main()
