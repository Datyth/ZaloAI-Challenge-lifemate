"""Offline test of a post-processing rule: a number the agent put into a call must come from somewhere.

When a numeric argument (integer/number type, or a digit-only string) of a call is not found among the numbers of the
transcripts (current + history, after ITN), the context, the history and the tool observations of the sample, the call is
treated as a guess about a value the user never gave: the calls from there on are dropped and the turn ends with
sys.ask_missing_info(tool_name, params). The labels ask exactly this way when the user leaves out a required value
("chuyển ví cho Hương chút tiền" -> ask amount; "volume mức chị hay coi" -> ask volume).

Re-scores a saved TRAIN run with analysis/score.py (state hash recomputed by replaying the edited calls in ToolEnv).
    python analysis/ground_numbers.py outputs/runs/v200_s1c_asrv2 [--show]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "src"))
import score as S  # noqa: E402
from lifemate.itn import normalize  # noqa: E402
from lifemate.toolenv import new_env  # noqa: E402

SCALE = {"k": 1e3, "nghìn": 1e3, "ngàn": 1e3, "nghin": 1e3, "ngan": 1e3, "trăm": 1e2, "triệu": 1e6, "củ": 1e6, "tr": 1e6, "tỷ": 1e9, "tỉ": 1e9}
NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|nghin|ngan|trăm|triệu|củ|tr|tỷ|tỉ)?\s*(rưỡi)?", re.I)


DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?)?|\b\d{1,2}:\d{2}(?::\d{2})?\b")


def numbers(text: str) -> set[float]:
    text = DATETIME.sub(" ", text)   # dates and clock times are not quantities
    out = set()
    for m in NUM.finditer(text):
        raw = m.group(1)
        vals = {float(raw.replace(",", "."))} | ({float(raw.replace(",", "").replace(".", ""))} if re.search(r"[.,]\d{3}\b", raw) else set())
        for v in vals:
            out.add(v)
            if m.group(2):
                s = SCALE[m.group(2).lower()]
                out |= {v * s, v * s + (s / 2 if m.group(3) else 0)}
    return out


def numeric_args(spec: dict, args: dict) -> list[str]:
    props = (spec.get("parameters") or {}).get("properties", {})
    keys = []
    for k, v in args.items():
        p = props.get(k, {})
        if isinstance(v, bool) or "enum" in p or p.get("x-ref") or k.endswith("_id"):
            continue
        if p.get("type") in ("integer", "number") and isinstance(v, (int, float)):
            keys.append(k)
        elif isinstance(v, str) and re.fullmatch(r"\d{3,}", v):
            keys.append(k)
    return keys


def sources(item: dict, run_dir: Path, sid: str) -> tuple[set[float], list[set[float]]]:
    """(numbers known before any call, numbers of each executed call's observation in order)."""
    texts = []
    for r in map(json.loads, (run_dir / "asr_outputs.jsonl").open(encoding="utf-8")):
        if r["sample_id"] == sid:
            texts.append(normalize(r.get("raw_transcript") or r.get("transcript") or ""))
    context = {k: v for k, v in item["context"].items() if k != "installed_apps"}
    texts.append(json.dumps(context, ensure_ascii=False))
    texts.append(json.dumps(item["history"], ensure_ascii=False))
    observed = [numbers(json.dumps(e.get("result"), ensure_ascii=False)) for e in map(json.loads, (run_dir / "tool_events.jsonl").open(encoding="utf-8"))
                if e["sample_id"] == sid and e.get("executed")]
    out = set()
    for t in texts:
        out |= numbers(t)
    return out, observed


def edit(calls: list[dict], known: tuple[set[float], list[set[float]]], specs: dict) -> tuple[list[dict], str | None]:
    base, observed = known
    for k, c in enumerate(calls):
        grounded = base.union(*observed[:k])   # only what the agent had seen before this call
        if c["name"] == "sys.ask_missing_info":
            return calls, None
        bad = [a for a in numeric_args(specs.get(c["name"], {}), c["arguments"]) if float(c["arguments"][a]) not in grounded]
        if bad:
            return calls[:k] + [{"name": "sys.ask_missing_info", "arguments": {"tool_name": c["name"], "params": bad}}], f"{c['name']}{bad}"
    return calls, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (S.ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    specs = S.catalog()
    preds = S.load_run(a.run)
    ghash = S.gold_hashes([items[i] for i in preds])
    before = after = 0.0
    changed = []
    for sid, p in sorted(preds.items()):
        item = items[sid]
        old = S.score_sample(item, p, ghash[sid], 0.5)["score"]
        calls, why = edit(p.get("tool_calls") or [], sources(item, Path(p["run_dir"]), sid), specs)
        new = old
        if why:
            env = new_env(S.ROOT, "train", sid)
            for c in calls:
                env.call(c["name"], c["arguments"])
            new = S.score_sample(item, {**p, "tool_calls": calls, "state_hash": env.get_state_hash()}, ghash[sid], 0.5)["score"]
            changed.append((sid, why, old, new))
        before += old
        after += new
    n = len(preds)
    print(f"{a.run.name}: {100 * before / n:.2f} -> {100 * after / n:.2f} | edited {len(changed)}: "
          f"better {sum(x[3] > x[2] for x in changed)}, worse {sum(x[3] < x[2] for x in changed)}, same {sum(x[3] == x[2] for x in changed)}")
    if a.show:
        for sid, why, old, new in changed:
            print(f"  {sid} {old:.2f}->{new:.2f} {why} | {items[sid]['audio_text'][:80]!r}")


if __name__ == "__main__":
    main()
