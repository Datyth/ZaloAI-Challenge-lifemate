"""Summary of analysis/token_loss.py output: loss share, mean NLL and teacher-forced token accuracy per category and set/variant;
turn-level teacher-forced exactness; the worst tokens. Usage: token_loss_report.py <out dir> [--worst N]"""
import json, sys
from collections import defaultdict, Counter
d = sys.argv[1]; worst = int(sys.argv[sys.argv.index("--worst") + 1]) if "--worst" in sys.argv else 0
toks = [json.loads(l) for l in open(f"{d}/tokens.jsonl")]
turns = [json.loads(l) for l in open(f"{d}/turns.jsonl")]
key = lambda r: f"{r['set']}/{r['variant']}" if r["set"] in ("dev", "train") else r["set"]
cats = ["decision", "fn_name", "param_key", "value_ref", "value_copy", "value_time", "value_num", "value_from_context", "value_other", "value_struct", "stop", "syntax"]
by = defaultdict(lambda: defaultdict(list))
for r in toks: by[key(r)][r["cat"]].append(r)
for k in sorted(by):
    tot = sum(x["nll"] for c in by[k].values() for x in c); n = sum(len(c) for c in by[k].values())
    rows = sum(1 for _ in {(t["sample_id"], t["variant"]) for t in turns if key(t) == k})
    print(f"\n## {k}: {rows} trajectories, {n} trained tokens, mean NLL {tot / n:.4f}, sum {tot:.1f}")
    print(f"{'category':20s} {'tokens':>7s} {'share':>6s} {'meanNLL':>8s} {'acc':>7s} {'wrong':>6s}")
    for c in cats:
        xs = by[k].get(c, [])
        if not xs: continue
        s = sum(x["nll"] for x in xs); w = sum(not x["ok"] for x in xs)
        print(f"{c:20s} {len(xs):7d} {s / tot:6.1%} {s / len(xs):8.4f} {1 - w / len(xs):7.2%} {w:6d}")
    ts = [t for t in turns if key(t) == k]
    ex = sum(t["wrong"] == 0 for t in ts)
    fw = Counter(t["first_wrong"]["cat"] for t in ts if t["first_wrong"])
    traj = defaultdict(list)
    for t in ts: traj[(t["sample_id"], t["variant"])].append(t["wrong"] == 0)
    print(f"turns exact under teacher forcing {ex}/{len(ts)} ({ex / len(ts):.1%}); trajectories all-exact {sum(all(v) for v in traj.values())}/{len(traj)}; first wrong token by category {dict(fw.most_common())}")
if worst:
    for k in sorted(by):
        print(f"\n## worst first-wrong turns {k}")
        ts = sorted([t for t in turns if key(t) == k and t["first_wrong"]], key=lambda t: -t["nll"])[:worst]
        for t in ts:
            fw = t["first_wrong"]
            print(f"  {t['sample_id']} t{t['turn']}/{t['n_turns']} gold={t['gold_fn']} nll={t['nll']:.2f} {fw['cat']} key={fw.get('key')} gold_tok={fw['gold']!r} pred={fw['pred']!r} value={fw.get('value', '')[:60]!r}")
