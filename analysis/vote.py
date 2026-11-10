"""Majority vote over several agent runs by predicted final state hash (offline simulation and submission builder).

Per sample, the runs are grouped by their predicted state_hash; the hash with the most runs wins (ties: the group holding the
earliest run in the given order). Inside the winning group the earliest run's tool calls are kept. On val the per-run scores
(outputs/analysis/<run>.scores.jsonl) give the voted score exactly, since the chosen prediction is one run's own prediction.
    python analysis/vote.py val v200_q35e1_asrv2_dedup v200_q35p1cap2_asrv2_dedup v200_q35synth_asrv2_dedup
    python analysis/vote.py public result/submission_a.jsonl result/submission_b.jsonl ... --output result/submission_vote.jsonl
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def preds_of_run(run: str) -> dict:
    out = {}
    for f in sorted(glob.glob(f"outputs/runs/{run}_s*/predictions.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r
    return out


def vote(preds: list[dict]) -> tuple[dict, dict]:
    chosen, margin = {}, Counter()
    for sid in preds[0]:
        hashes = [p[sid].get("state_hash") if sid in p else None for p in preds]
        counts = Counter(h for h in hashes if h)
        best = max(counts.values())
        winner = next(h for h in hashes if h and counts[h] == best)   # earliest run among the tied hashes
        k = hashes.index(winner)
        chosen[sid] = k
        margin[f"{best}/{len(preds)}"] += 1
    return chosen, margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "public"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.mode == "val":
        preds = [preds_of_run(r) for r in a.runs]
        scores = [{json.loads(l)["id"]: json.loads(l)["score"] for l in open(f"outputs/analysis/{r}.scores.jsonl")} for r in a.runs]
        chosen, margin = vote(preds)
        n = len(chosen)
        for r, s in zip(a.runs, scores):
            print(f"  {r:40s} {100 * sum(s.values()) / len(s):6.2f}")
        voted = 100 * sum(scores[k][sid] for sid, k in chosen.items()) / n
        oracle = 100 * sum(max(s[sid] for s in scores) for sid in chosen) / n
        print(f"  VOTE {voted:.2f}   (best-of-runs oracle {oracle:.2f}; agreement {dict(sorted(margin.items()))}; picked from run {dict(Counter(chosen.values()))})")
    else:
        preds = [{json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")} for f in a.runs]
        chosen, margin = vote(preds)
        print(f"agreement {dict(sorted(margin.items()))}; picked from run {dict(Counter(chosen.values()))}")
        if a.output:
            with a.output.open("w", encoding="utf-8") as f:
                for sid in preds[0]:
                    f.write(json.dumps(preds[chosen[sid]][sid], ensure_ascii=False) + "\n")
            print(f"wrote {len(chosen)} rows to {a.output}")


if __name__ == "__main__":
    main()
