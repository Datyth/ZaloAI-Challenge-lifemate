"""Context-hotword ASR caches against the plain ASR v2 cache on the 200 validation ids (gold transcripts as reference).
    .venv/bin/python analysis/hotword_eval.py outputs/asr/gipformer-lm-v2-01_val outputs/asr/ctxhw_v2_s2.0_val ...
Per cache: word error rate (current request / all user audio), samples with a damaged stated argument, and for the bias phrases:
how many of the phrases the speaker really says come out right, and how many phrases appear in the ASR text without being said."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "src"))
import noise_analysis as N  # noqa: E402
from lifemate.context import hotword_phrases  # noqa: E402


def main() -> None:
    items = {x["id"]: x for x in map(json.loads, (N.S.ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    val = sorted(json.loads((PROJECT / "configs/splits/validation_ids.json").read_text()))
    base = None
    print("| cache | WER current | WER all | stated args damaged | samples with damage | phrases said | said and recognised | phrases inserted, not said | transcripts changed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in map(Path, sys.argv[1:]):
        cache = {(r["sample_id"], r["audio"]): r["raw_transcript"] for r in map(json.loads, (d / "asr_outputs.jsonl").open(encoding="utf-8"))}
        base = base or cache
        rows = [N.damage(items[i], cache) for i in val]
        said = hit = ins = 0
        for i in val:
            gold = " ".join(" ".join(N.render(t)) for t in N.user_audios(items[i]).values())
            hyp = " ".join(" ".join(N.render(cache[(i, a)])) for a in N.user_audios(items[i]))
            for p in hotword_phrases(items[i]):
                r = " ".join(N.render(p))
                in_gold, in_hyp = f" {r} " in f" {gold} ", f" {r} " in f" {hyp} "
                said += in_gold
                hit += in_gold and in_hyp
                ins += in_hyp and not in_gold
        changed = sum(cache[k] != base[k] for k in cache)
        st, dm = sum(len(r["stated"]) for r in rows), sum(len(r["damaged"]) for r in rows)
        print(f"| {d.name} | {N.wer(rows, 'err_current'):.2f} | {N.wer(rows):.2f} | {dm}/{st} | {sum(r['cls'] == 'stated argument damaged' for r in rows)} | {said} | {hit} | {ins} | {changed}/{len(cache)} |")


if __name__ == "__main__":
    main()
