"""Texts for ASR v4 TTS data: TRAIN (non-val) gold utterances with person names and numbers re-drawn, spelled for TTS.

Why: on val the requests with 6+ number words score 52 (vs 71 overall) and misheard names/numbers cost ~6 of the 10.5 points ASR
loses; public has 26% more number words than val. TTS audio of TRAIN-shaped sentences with new names and new numbers teaches the
acoustic model those words without touching any test data (names come from pools written here, numbers are random).
Each output row: the TTS text (numbers spelled out, so the reading is known), the ASR label (upper case, no digits), a reference
voice (a TRAIN clip), and the source utterance.
    .venv/bin/python scripts/gen_tts_asr_text.py --out outputs/asr_tts/v1/texts.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from asr_targets import _int_variants, _plain, _spell_up  # noqa: E402

SURNAMES = "Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Đinh Trịnh Lương Mai Tô Đoàn Cao Lâm Hà Quách Tạ Châu Kiều".split()
MIDDLES = "Văn Thị Minh Ngọc Quang Thanh Hữu Đức Thu Hoài Gia Bảo Khánh Kim Phương Xuân Hồng Tuấn".split()
GIVEN = ("An Anh Bảo Bình Châu Chi Cường Diệp Dung Duy Dương Đạt Giang Hà Hải Hạnh Hiền Hiếu Hòa Hoàng Huy Hùng Hương Khang Khoa Khôi "
         "Lâm Linh Long Lộc Mạnh My Nam Nga Ngân Nhung Oanh Phát Phong Phúc Quân Quyên Sơn Tâm Thảo Thắng Thịnh Thơ Thủy Tiến Toàn "
         "Trang Trinh Trung Tú Tùng Uyên Vân Việt Vinh Xuân Yến Đông Nhàn Lụa Tài Kiệt Luân Sang").split()
FOREIGN_FIRST = "Peter Ben Lucas Emily Sarah David Kevin Daniel Anna Linda Tom Jack Ryan Liam Olivia Sophia Henry Mark John Lisa Kate Mike Paul Laura".split()
FOREIGN_LAST = "Hansen Taylor Smith Brown Miller Wilson Moore Clark Lewis Walker Hall Young King Wright Scott Green Baker Adams Nelson Carter".split()
HONORIFIC = {"Sếp", "Cô", "Chú", "Anh", "Chị", "Bác", "Thầy", "Ông", "Bà", "Em", "Mr", "Ms", "Mrs", "Dì", "Cậu", "Mợ", "Vợ", "Chồng",
             "Bố", "Mẹ", "Con", "Cháu", "Bé", "Thằng", "Con"}
KEEP = {"Hà", "Nội", "Đà", "Nẵng", "Huế", "Sài", "Gòn", "Việt", "Nam", "SJC", "SMS", "Zalo", "Tây", "Hàn", "Nhật", "Ninh", "Bình",
        "Hồ", "Chí", "Minh", "Cần", "Thơ", "Hải", "Phòng", "Vũng", "Tàu", "Nha", "Trang", "Đồng", "Nai", "Long", "An", "Quận", "Phố",
        "Nhà", "Bún", "Mèo", "Ngày", "Bước", "Grab", "Shopee", "Momo", "MoMo", "Vietcombank", "EVN", "FPT", "VNPT", "Viettel"}
NUM = re.compile(r"\b\d+(?:[.,]\d{3})*(?:k|K|h\d*|H\d*)?\b")


def new_name(rng: random.Random, n_words: int) -> str:
    if rng.random() < 0.25:
        return rng.choice(FOREIGN_FIRST) + ("" if n_words == 1 else " " + rng.choice(FOREIGN_LAST))
    if n_words == 1:
        return rng.choice(GIVEN)
    if n_words == 2:
        return f"{rng.choice(MIDDLES if rng.random() < 0.3 else SURNAMES)} {rng.choice(GIVEN)}"
    return f"{rng.choice(SURNAMES)} {rng.choice(MIDDLES)} {rng.choice(GIVEN)}"


def swap_names(text: str, rng: random.Random) -> tuple[str, int]:
    """Replace runs of capitalised words (an optional honorific + up to 3 name words) not at a sentence start and free of place /
    brand words with a new name from the pools; the honorific and trailing punctuation stay."""
    toks = text.split(" ")
    out, i, swapped = [], 0, 0
    word = lambda t: re.sub(r"[^\w]", "", t)
    while i < len(toks):
        w = word(toks[i])
        sentence_start = i == 0 or toks[i - 1].endswith((".", "!", "?", ":", ";", "..."))
        if not (w and w[0].isupper() and not w.isupper() and not sentence_start):
            out.append(toks[i])
            i += 1
            continue
        j = i
        while j < len(toks) and j - i < 4:
            wj = word(toks[j])
            if not (wj and wj[0].isupper() and not wj.isupper()):
                break
            j += 1
            if toks[j - 1][-1:] in ",.!?;:":
                break
        span = toks[i:j]
        words = [word(t) for t in span]
        hon = 1 if words[0] in HONORIFIC else 0
        if any(x in KEEP for x in words) or len(words) - hon < 1:
            out += span
            i = j
            continue
        trail = span[-1][len(span[-1].rstrip(",.!?;:")):]
        repl = words[:hon] + new_name(rng, min(len(words) - hon, 3)).split()
        out += repl[:-1] + [repl[-1] + trail]
        swapped += 1
        i = j
    return " ".join(out), swapped


def respell_number(tok: str, rng: random.Random) -> str:
    """A new random value in the same format, spelled; one valid reading picked at random."""
    m = re.fullmatch(r"(\d+)[hH](\d*)", tok)
    if m:
        h = rng.randint(1, 12) if int(m.group(1)) <= 12 else rng.randint(13, 23)
        mi = rng.choice([0, 15, 30, 45, 10, 20]) if m.group(2) else 0
        r = _spell_up(h) + " GIỜ" + ("" if not mi else (" RƯỠI" if mi == 30 and rng.random() < 0.6 else " " + _spell_up(mi)))
        return r.lower()
    k = tok.lower().endswith("k")
    d = re.sub(r"[.,kK]", "", tok)
    if d.startswith("0") and len(d) > 1:
        return " ".join(_spell_up(rng.randint(0, 9)) for _ in d).lower()
    n = int(d)
    mag = len(str(n))
    v = rng.randint(10 ** (mag - 1) if mag > 1 else 1, 10 ** mag - 1)
    if n % 1000 == 0 and n >= 1000:
        v = (v // 1000) * 1000 or 1000
    elif n % 100 == 0 and n >= 100:
        v = (v // 100) * 100 or 100
    reading = rng.choice(sorted(_int_variants(v)))
    return (reading + (" nghìn" if k else "")).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--voices", type=int, default=48)
    ap.add_argument("--templates", type=int, default=600)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in (PROJECT / "outputs/asr_sft/gipformer_data_v2/train.jsonl").open(encoding="utf-8")]
    root = PROJECT / "data/train+public_test/train"
    clips = [r["audio"] for r in rows if 3.0 <= sf.info(str(root / r["audio"])).duration <= 8.0]
    voices = rng.sample(clips, a.voices)
    out = []
    for r in rows:
        text, swapped = swap_names(r["text"], rng)
        nums = NUM.findall(text)
        if not swapped and not nums:
            continue
        text = NUM.sub(lambda m: respell_number(m.group(0), rng), text)
        if re.search(r"\d", text):
            continue
        label = " ".join(_plain(text))
        out.append({"id": f"tts_{len(out):05d}", "tts_text": text, "label": label, "voice": voices[len(out) % len(voices)],
                    "source": r["audio"], "names_swapped": swapped, "numbers": len(nums)})
    # number-heavy templates (generic assistant requests; amounts, times, dates, phone numbers, quantities)
    T = ["chuyển cho {hon} {name} {amount} nhé", "nạp {amount} vào ví giúp {me}", "trả tiền điện {amount} bằng ngân hàng nha",
         "đặt báo thức {time} sáng mai", "nhắc {me} lúc {time} chiều nay đi họp", "gọi giúp {me} số {phone}",
         "nhắn tin cho số {phone} là {me} tới trễ {small} phút", "mua {small} hộp sữa với {small} kg gạo",
         "đặt lịch họp ngày {day} tháng {month} lúc {time}", "chuyển {amount} cho {hon} {name} rồi nhắn {hon} ấy là đã chuyển",
         "xem giá vàng hôm nay rồi nếu dưới {amount} thì nhắc {me}", "đặt bàn cho {small} người lúc {time} tối nay",
         "tăng nhiệt độ điều hòa lên {temp} độ", "đặt xe {seats} chỗ lúc {time} từ nhà {hon} {name}",
         "ghi chú là tháng này đã tiêu {amount}", "hẹn {hon} {name} {time} ngày {day} tháng {month} ở quán cà phê"]
    def fill(t):
        amt = rng.choice([rng.randint(1, 999) * 1000, rng.randint(1, 99) * 10000, rng.randint(1, 50) * 100000, rng.randint(1, 30) * 1000000])
        return t.format(hon=rng.choice(["anh", "chị", "cô", "chú", "bác", "em", "sếp"]), name=new_name(rng, rng.choice([1, 1, 2, 3])),
                        me=rng.choice(["anh", "chị", "em", "tôi", "mình"]), amount=rng.choice(sorted(_int_variants(amt))).lower() + rng.choice([" đồng", ""]),
                        time=respell_number(f"{rng.randint(5, 21)}h{rng.choice(['', '15', '30', '45'])}", rng),
                        phone=" ".join(_spell_up(int(c)).lower() for c in "0" + "".join(str(rng.randint(0, 9)) for _ in range(9))),
                        small=_spell_up(rng.randint(2, 15)).lower(), day=_spell_up(rng.randint(1, 31)).lower(), month=_spell_up(rng.randint(1, 12)).lower(),
                        temp=_spell_up(rng.randint(18, 30)).lower(), seats=rng.choice(["bốn", "bảy"]))
    for _ in range(a.templates):
        text = fill(rng.choice(T))
        out.append({"id": f"tts_{len(out):05d}", "tts_text": text, "label": " ".join(_plain(text)), "voice": voices[len(out) % len(voices)],
                    "source": "template", "names_swapped": 0, "numbers": 1})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(out)} texts from {len(rows)} TRAIN utterances; with names {sum(x['names_swapped'] > 0 for x in out)}, "
          f"with numbers {sum(x['numbers'] > 0 for x in out)}; {len(voices)} reference voices")


if __name__ == "__main__":
    main()
