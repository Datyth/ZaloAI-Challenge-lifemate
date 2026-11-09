"""Samples of a saved run where the --ground-args rules (src/lifemate/agent.py) would change an executed write call.
Label-free: reads only the run logs, the split's inputs (context, history) and its tool catalog. Used to re-run just those
samples after a rule change; the other trajectories would be the same.
    python analysis/ground_fires.py pub_s1c_asrv2_dedup_g --split public_test [--out ids.json]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.agent import _drop_context_defaults, _unreferenced_ids, _user_supplied_id  # noqa: E402
from lifemate.history import ids_in, successful  # noqa: E402
from lifemate.submission import shard_dirs  # noqa: E402

ROOT = PROJECT / "data/train+public_test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (ROOT / a.split / f"{a.split}.jsonl").open(encoding="utf-8"))}
    raw = json.loads((ROOT / a.split / f"{a.split}_tools_catalog.json").read_text(encoding="utf-8"))
    specs = {t["name"]: t for app in raw["apps"] for t in app["tools"]}
    events, speech = defaultdict(list), defaultdict(list)
    for d in shard_dirs(PROJECT / "outputs/runs" / a.prefix):
        for e in map(json.loads, (d / "tool_events.jsonl").open(encoding="utf-8")):
            if e.get("executed"):
                events[e["sample_id"]].append(e)
        for r in map(json.loads, (d / "asr_outputs.jsonl").open(encoding="utf-8")):
            speech[r["sample_id"]].append(r.get("raw_transcript") or "")
    fires = {}
    for sid, evs in sorted(events.items()):
        item = items[sid]
        known = ids_in(item["context"]) | ids_in(item.get("history"))
        text = " ".join(speech[sid]) or item.get("audio_text", "")
        why = []
        for e in evs:
            spec = specs.get(e["name"], {})
            if spec.get("kind") == "write":
                bad = [(p, v) for p, v in _unreferenced_ids(spec, e["arguments"])
                       if v not in known and not (not re.search(r"\s", v) and _user_supplied_id(v, text, []))]
                dropped = _drop_context_defaults(spec, e["arguments"], item["context"], text)[1]
                why += [f"{e['name']} {p}={v!r}" for p, v in bad] + [f"{e['name']} drop {sorted(dropped)}" for _ in [0] if dropped]
            if successful(e.get("result")):
                known |= ids_in(e["result"])
        if why:
            fires[sid] = why
    for sid, why in fires.items():
        print(sid, "; ".join(why))
    print(f"{len(fires)} of {len(events)} samples with executed calls")
    if a.out:
        a.out.write_text(json.dumps(sorted(fires)))


if __name__ == "__main__":
    main()
