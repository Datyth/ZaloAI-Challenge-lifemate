"""Compare ASR caches on the same TRAIN audio: WER, and recall / false insertions of key words (accent-kept, lower-case).
  python analysis/asr_compare.py outputs/asr/hw_none_val outputs/asr/hw_v1_s15_val [...]"""
import json, re, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.metrics import error_counts  # noqa: E402
KEYS = ["dời", "nhắc", "nhắn", "zalo", "sms", "sjc", "họp", "báo thức", "tắt", "bật", "hủy|huỷ", "nửa tiếng", "một tiếng", "sếp", "chuyển", "trả", "đặt"]
items = {x["id"]: x for x in map(json.loads, open(PROJECT / "data/train+public_test/train/train.jsonl", encoding="utf-8"))}
ref = {}
for x in items.values():
    ref[(x["id"], x["audio"])] = x["audio_text"]
    for t in x["history"]:
        if t.get("role") == "user" and "audio" in t:
            ref[(x["id"], t["audio"])] = t["audio_text"]
def words(s): return " " + " ".join(re.findall(r"[^\W_]+", s.lower())) + " "
for d in sys.argv[1:]:
    rows = [json.loads(l) for l in open(Path(d) / "asr_outputs.jsonl", encoding="utf-8")]
    we = rw = 0
    stats = {k: [0, 0, 0] for k in KEYS}   # ref has, hyp keeps, hyp inserts without ref
    for r in rows:
        R, H = ref[(r["sample_id"], r["audio"])], r["raw_transcript"]
        c = error_counts(R, H); we += c["word_errors"]; rw += c["reference_words"]
        rr, hh = words(R), words(H)
        for k in KEYS:
            pat = re.compile(r" (?:" + k + r") ")
            in_r, in_h = bool(pat.search(rr)), bool(pat.search(hh))
            stats[k][0] += in_r; stats[k][1] += in_r and in_h; stats[k][2] += in_h and not in_r
    print(f"\n{d}: audio={len(rows)} WER={we / rw:.4f}")
    print("  " + "  ".join(f"{k.split('|')[0]} {v[1]}/{v[0]}+{v[2]}" for k, v in stats.items()))
