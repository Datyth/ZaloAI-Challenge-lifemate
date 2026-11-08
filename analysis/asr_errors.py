"""Where Gipformer loses information on TRAIN audio (current and history turns), against audio_text.
  python analysis/asr_errors.py [--cache outputs/asr/gipformer_train/asr_outputs.jsonl]
- word alignment on accent-kept lower-case tokens: top substitutions / deletions / insertions
- digit tokens: references write numbers as digits ("1 tiếng", "218k"), Gipformer speaks them ("MỘT"); counted apart
- slot survival: every string argument of the gold calls that occurs in audio_text, checked in the ASR text (accent-folded)"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
import score as S  # noqa: E402

ROOT = S.ROOT


def words(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", s.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=PROJECT / "outputs/asr/gipformer_train/asr_outputs.jsonl")
    ap.add_argument("--ids", type=Path, help="restrict to these sample ids (e.g. the 200 validation ids)")
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    keep = set(json.loads(a.ids.read_text())) if a.ids else None
    ref = {}
    for x in items.values():
        ref[(x["id"], x["audio"])] = x["audio_text"]
        for t in x["history"]:
            if t.get("role") == "user" and "audio" in t:
                ref[(x["id"], t["audio"])] = t["audio_text"]
    subs, dels, ins = collections.Counter(), collections.Counter(), collections.Counter()
    n_words = n_err = n_digit_words = n_digit_err = 0
    current_hyp = {}
    for line in a.cache.open(encoding="utf-8"):
        r = json.loads(line)
        if keep and r["sample_id"] not in keep:
            continue
        rw, hw = words(ref[(r["sample_id"], r["audio"])]), words(r["raw_transcript"])
        if r["audio"].endswith(f"{r['sample_id']}.wav"):
            current_hyp[r["sample_id"]] = r["raw_transcript"]
        sm = difflib.SequenceMatcher(a=rw, b=hw, autojunk=False)
        n_words += len(rw)
        n_digit_words += sum(any(ch.isdigit() for ch in w) for w in rw)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            seg_r, seg_h = " ".join(rw[i1:i2]), " ".join(hw[j1:j2])
            err = max(i2 - i1, j2 - j1)
            digit = any(ch.isdigit() for ch in seg_r)
            n_err += err
            n_digit_err += err if digit else 0
            if op == "replace":
                subs[(seg_r, seg_h)] += 1
            elif op == "delete":
                dels[seg_r] += 1
            else:
                ins[seg_h] += 1
    print(f"aligned word errors {n_err}/{n_words} = {n_err / n_words:.3f}; inside spans with digits {n_digit_err} "
          f"({n_digit_err / n_err:.1%} of errors); reference words with digits {n_digit_words}")
    non_digit = [(k, v) for k, v in subs.items() if not any(ch.isdigit() for ch in k[0])]
    print("\nTop substitutions without digits (ref -> asr):")
    for (r_, h_), c in sorted(non_digit, key=lambda kv: -kv[1])[:45]:
        print(f"  {c:4d}  {r_!r} -> {h_!r}")
    print("\nTop substitutions with digits:")
    for (r_, h_), c in sorted(((k, v) for k, v in subs.items() if any(ch.isdigit() for ch in k[0])), key=lambda kv: -kv[1])[:20]:
        print(f"  {c:4d}  {r_!r} -> {h_!r}")
    print("\nTop deletions:", dels.most_common(20))
    print("\nTop insertions:", ins.most_common(20))
    # slot survival on the current utterance
    survive = collections.Counter()
    lost_examples = collections.defaultdict(list)
    for sid, hyp in current_hyp.items():
        it = items[sid]
        text_f, hyp_f = S.fold(it["audio_text"]), S.fold(hyp)
        for c in it["label"]["tool_calls"]:
            for k, v in (c.get("arguments") or {}).items():
                if not isinstance(v, str) or len(v) < 2 or re.fullmatch(r"[a-z]+_\w+", v):
                    continue
                vf = S.fold(v)
                if vf not in text_f:
                    continue
                key = f"{c['name']}.{k}"
                ok = vf in hyp_f
                survive[(key, ok)] += 1
                if not ok and len(lost_examples[key]) < 4:
                    lost_examples[key].append((v, hyp[:90]))
    keys = sorted({k for k, _ in survive}, key=lambda k: -(survive[(k, True)] + survive[(k, False)]))
    tot_ok = sum(v for (k, ok), v in survive.items() if ok)
    tot = sum(survive.values())
    print(f"\nString slots spoken verbatim in audio_text that survive ASR: {tot_ok}/{tot} = {tot_ok / tot:.1%}")
    for k in keys[:25]:
        ok, bad = survive[(k, True)], survive[(k, False)]
        print(f"  {k:40s} {ok:4d}/{ok + bad:<4d} lost e.g. {lost_examples[k][:2]}")


if __name__ == "__main__":
    main()
