"""ASR noise in the agent's SFT data and at inference, measured against the gold transcripts by word alignment.
    .venv/bin/python analysis/noise_analysis.py [--md docs/noise_analysis.md] [--json out.json]

Every transcript (gold and ASR) is rendered the same way before alignment: lower case, numbers spelled the way Gipformer reads them,
punctuation removed, and the tone mark moved to the end of each syllable (so "hoà" and "hòa" are the same word). A call argument is
"stated" when its rendering appears word for word in the sample's gold transcripts (current request or history), and "damaged" when
no such occurrence survives the ASR alignment: the label then asks for a value the ASR text does not contain.

Tables:
  1. word error rate per (ids, ASR): base Gipformer on the SFT ids, base and ASR v2 on the 200 validation ids, ASR v2 on its own training audio;
  2. damage per sample: stated argument damaged / key word damaged / other word errors / clean, and the per-argument loss rate;
  3. label-free public estimate: base-vs-v2 disagreement and out-of-vocabulary rate on public next to validation;
  4. q35e1 on validation by damage class: gold-text score, ASR score and the points ASR costs in each class;
  5. what q35e1 does with a damaged argument it gets right from the gold transcript.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "scripts" / "data"))
sys.path.insert(0, str(PROJECT / "src"))
import score as S  # noqa: E402
from prepare_agent_sft import _spell  # noqa: E402

TONES = {"̀": 1, "́": 2, "̃": 3, "̉": 4, "̣": 5}
# Words whose loss flips the action or its polarity (intent verbs, negation, on/off), canonical spelling.
KEY_WORDS = ["không", "chưa", "đừng", "bật", "tắt", "xoá", "huỷ", "dời", "nhắn", "nhắc", "gọi", "chuyển", "đặt", "hẹn", "thêm", "sửa",
             "đổi", "mở", "khoá", "gửi", "trả", "nạp", "rút", "mua", "hỏi", "lùi", "hoãn", "bỏ", "xem", "kiểm", "tra", "tìm"]
ASR_DIR = PROJECT / "outputs/asr"


def canon(word: str) -> str:
    d = unicodedata.normalize("NFD", word)
    tone = next((TONES[c] for c in d if c in TONES), 0)
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))
    return f"{base}{tone}" if tone else base


def render(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)
    t = re.sub(r"\b(\d{1,2})[:h](\d{2})\b", lambda m: f"{m.group(1)} giờ" + ("" if m.group(2) == "00" else f" {int(m.group(2))}"), t)
    t = t.replace("%", " phần trăm ")
    t = re.sub(r"(\d)([^\W\d_])", r"\1 \2", t)
    t = re.sub(r"([^\W\d_])(\d)", r"\1 \2", t)
    t = re.sub(r"\b\d{1,9}\b", lambda m: _spell(int(m.group())), t)
    return [canon(w) for w in re.sub(r"[^\w]+", " ", t).split()]


KEY = {canon(w) for w in KEY_WORDS}


def align(ref: list[str], hyp: list[str]) -> tuple[list[bool], int, int, int]:
    """Levenshtein alignment; returns per-ref-token 'recognised' flags and (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    ok, i, j, sub, dele, ins = [False] * n, n, m, 0, 0, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ok[i - 1] = ref[i - 1] == hyp[j - 1]
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return ok, sub, dele, ins


def values(v, path=""):
    if isinstance(v, bool) or v is None:
        return
    if isinstance(v, str):
        yield path, v
    elif isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
        yield path, str(int(v))
    elif isinstance(v, list):
        for k, x in enumerate(v):
            yield from values(x, f"{path}[{k}]")
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from values(x, f"{path}.{k}" if path else k)


def find(seq: list[str], tokens: list[str]) -> list[int]:
    k = len(seq)
    return [i for i in range(len(tokens) - k + 1) if tokens[i:i + k] == seq] if k else []


def load_cache(name: str) -> dict:
    return {(r["sample_id"], r["audio"]): r["raw_transcript"] for r in map(json.loads, (ASR_DIR / name / "asr_outputs.jsonl").open(encoding="utf-8"))}


def user_audios(item: dict) -> dict[str, str | None]:
    out = {item["audio"]: item.get("audio_text")}
    out.update({t["audio"]: t.get("audio_text") for t in item["history"] if t.get("role") == "user" and "audio" in t})
    return out


def damage(item: dict, asr: dict) -> dict | None:
    audios = user_audios(item)
    if any((item["id"], a) not in asr for a in audios):
        return None
    refs, oks, hyps = {}, {}, {}
    sub = dele = ins = nref = sub_c = dele_c = ins_c = nref_c = 0
    for a, gold in audios.items():
        refs[a], hyps[a] = render(gold), render(asr[(item["id"], a)])
        oks[a], s, d, i = align(refs[a], hyps[a])
        sub, dele, ins, nref = sub + s, dele + d, ins + i, nref + len(refs[a])
        if a == item["audio"]:
            sub_c, dele_c, ins_c, nref_c = s, d, i, len(refs[a])
    stated, damaged = [], []
    for c in item["label"]["tool_calls"]:
        for path, v in values(c.get("arguments") or {}):
            seq = render(v)
            if not seq or not re.search(r"[^\W\d_]", " ".join(seq)):
                continue
            hits = [(a, p) for a in refs for p in find(seq, refs[a])]
            if not hits:
                continue
            stated.append((c["name"], path, v))
            if not any(all(oks[a][p:p + len(seq)]) for a, p in hits):
                damaged.append((c["name"], path, v))
    key_lost = sorted({refs[a][k] for a in refs for k in range(len(refs[a])) if refs[a][k] in KEY and not oks[a][k]})
    cls = "stated argument damaged" if damaged else "key word damaged" if key_lost else "other word errors" if sub + dele + ins else "clean"
    return {"id": item["id"], "cls": cls, "stated": stated, "damaged": damaged, "key_lost": key_lost,
            "err": (sub, dele, ins, nref), "err_current": (sub_c, dele_c, ins_c, nref_c), "hyp": {a: " ".join(hyps[a]) for a in hyps}}


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def wer(rows, key="err"):
    s = sum(r[key][0] for r in rows), sum(r[key][1] for r in rows), sum(r[key][2] for r in rows), sum(r[key][3] for r in rows)
    return 100 * (s[0] + s[1] + s[2]) / max(s[3], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-run", default="v200_q35e1_asrv2_dedup")
    ap.add_argument("--oracle-run", default="v200_q35e1_oracle_dedup")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (S.ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    val = set(json.loads((PROJECT / "configs/splits/validation_ids.json").read_text()))
    sft = [i for i in items if i not in val and not ({au.split("/")[1] for au in user_audios(items[i])} & val)]
    base, v2_val, v2_train = load_cache("gipformer_train"), load_cache("gipformer-lm-v2-01_val"), load_cache("gipformer-lm-v2-01_train")
    sets = {
        "SFT ids (1,345), base Gipformer: the noise q35e1 trained on": [damage(items[i], base) for i in sft],
        "SFT ids, ASR v2 on its own training audio": [damage(items[i], v2_train) for i in sft],
        "validation (200), base Gipformer": [damage(items[i], base) for i in sorted(val)],
        "validation (200), ASR v2: what the agent sees": [damage(items[i], v2_val) for i in sorted(val)],
    }
    sets = {k: [r for r in v if r] for k, v in sets.items()}
    parts = ["# ASR noise in SFT data and at inference\n"]

    rows = []
    for name, rs in sets.items():
        per = [100 * (r["err_current"][0] + r["err_current"][1] + r["err_current"][2]) / max(r["err_current"][3], 1) for r in rs]
        rows.append([name, len(rs), f"{wer(rs, 'err_current'):.1f}", f"{wer(rs):.1f}", f"{statistics.median(per):.1f}",
                     f"{100 * sum(p == 0 for p in per) / len(per):.0f}%"])
    parts.append("## 1. Word error rate (rendered words; history = all user audios of the sample)\n")
    parts.append(md_table(["ids, ASR", "samples", "WER current", "WER incl. history", "median WER current", "current exact"], rows) + "\n")

    classes = ["stated argument damaged", "key word damaged", "other word errors", "clean"]
    rows = []
    for name, rs in sets.items():
        cnt = collections.Counter(r["cls"] for r in rs)
        st, dm = sum(len(r["stated"]) for r in rs), sum(len(r["damaged"]) for r in rs)
        rows.append([name] + [f"{100 * cnt[c] / len(rs):.1f}%" for c in classes] + [st, f"{100 * dm / max(st, 1):.1f}%"])
    parts.append("## 2. Damage per sample (first matching class) and stated-argument loss\n")
    parts.append(md_table(["ids, ASR", *classes, "stated args", "stated args damaged"], rows) + "\n")
    lost_words = collections.Counter(w for r in sets["SFT ids (1,345), base Gipformer: the noise q35e1 trained on"] for w in r["key_lost"])
    lost_v2 = collections.Counter(w for r in sets["validation (200), ASR v2: what the agent sees"] for w in r["key_lost"])
    parts.append("Key words damaged, SFT base noise (samples): " + ", ".join(f"{w} {n}" for w, n in lost_words.most_common(15)) + "\n")
    parts.append("Key words damaged, validation ASR v2 (samples): " + ", ".join(f"{w} {n}" for w, n in lost_v2.most_common(15)) + "\n")
    dm_tools = collections.Counter(f"{t}.{p.split('[')[0]}" for r in sets["SFT ids (1,345), base Gipformer: the noise q35e1 trained on"] for t, p, _ in r["damaged"])
    parts.append("Damaged stated arguments in the SFT noise, by tool argument: " + ", ".join(f"{k} {n}" for k, n in dm_tools.most_common(15)) + "\n")

    # 3. label-free public estimate
    pub_base, pub_v2 = load_cache("gipformer_public"), load_cache("gipformer-lm-v2-01_public")
    vocab = collections.Counter(w for i in sft for t in user_audios(items[i]).values() if t for w in render(t))

    def disagreement(pairs):
        s = n = 0
        for ref, hyp in pairs:
            r, h = render(ref), render(hyp)
            _, a1, a2, a3 = align(r, h)
            s, n = s + a1 + a2 + a3, n + len(r)
        return 100 * s / max(n, 1)

    def oov(texts):
        ws = [w for t in texts for w in render(t)]
        return 100 * sum(w not in vocab for w in ws) / max(len(ws), 1)

    val_keys = [(i, au) for i in sorted(val) for au in user_audios(items[i]) if (i, au) in v2_val and (i, au) in base]
    pub_keys = [k for k in pub_v2 if k in pub_base]
    cur = lambda keys: [k for k in keys if k[1].split("/")[-1].startswith(k[0] + ".")]  # noqa: E731
    rows = [
        ["validation", len(cur(val_keys)), f"{disagreement([(v2_val[k], base[k]) for k in cur(val_keys)]):.1f}", f"{oov(v2_val[k] for k in cur(val_keys)):.2f}%",
         f"{oov(items[k[0]]['audio_text'] for k in cur(val_keys)):.2f}%"],
        ["public", len(cur(pub_keys)), f"{disagreement([(pub_v2[k], pub_base[k]) for k in cur(pub_keys)]):.1f}", f"{oov(pub_v2[k] for k in cur(pub_keys)):.2f}%", "n/a"],
    ]
    parts.append("## 3. Label-free noise estimate for public (current request audio)\n")
    parts.append("Disagreement = word error rate of the base Gipformer text measured against the ASR v2 text. OOV = share of ASR v2 words absent from "
                 "the gold transcripts of the SFT ids.\n")
    parts.append(md_table(["split", "audios", "base vs v2 disagreement", "ASR v2 OOV", "gold OOV"], rows) + "\n")

    # 4-5. q35e1 on validation by damage class
    preds = {name: S.load_run(PROJECT / "outputs/runs" / run) for name, run in (("asr", a.asr_run), ("oracle", a.oracle_run))}
    ids = sorted(set(preds["asr"]) & set(preds["oracle"]) & val)
    gh = S.gold_hashes([items[i] for i in ids])
    res = {name: {i: {**S.score_sample(items[i], preds[name][i], gh[i], 0.5), "calls": preds[name][i].get("tool_calls") or []} for i in ids} for name in preds}
    dmg = {r["id"]: r for r in sets["validation (200), ASR v2: what the agent sees"]}
    rows = []
    n = len(ids)
    for c in classes:
        g = [i for i in ids if dmg[i]["cls"] == c]
        if not g:
            continue
        o, s = sum(res["oracle"][i]["score"] for i in g), sum(res["asr"][i]["score"] for i in g)
        only = sum(res["oracle"][i]["score"] == 1 and res["asr"][i]["score"] < 1 for i in g)
        rows.append([c, len(g), f"{100 * o / len(g):.1f}", f"{100 * s / len(g):.1f}", f"{100 * (o - s) / n:.2f}", only])
    parts.append(f"## 4. {a.asr_run} vs {a.oracle_run} by ASR v2 damage class ({n} samples)\n")
    parts.append(md_table(["damage class", "samples", "score, gold text", "score, ASR", "points ASR costs (of 100)", "right with gold text, wrong with ASR"], rows) + "\n")

    behaviour = collections.Counter()
    examples = []
    for i in ids:
        if dmg[i]["cls"] != "stated argument damaged" or res["oracle"][i]["score"] < 1:
            continue
        calls = res["asr"][i]["calls"]
        asked = any(c["name"] == "sys.ask_missing_info" for c in calls)
        hyp_all = " ".join(dmg[i]["hyp"].values())
        for tool, path, v in dmg[i]["damaged"]:
            got = [dict(values(c.get("arguments") or {})).get(path) for c in calls if c["name"] == tool]
            got = [x for x in got if x is not None]
            if any(render(x) == render(v) for x in got):
                kind = "right value anyway"
            elif got and any(" ".join(render(x)) in hyp_all for x in got):
                kind = "copied the ASR words"
            elif got:
                kind = "other value"
            elif asked:
                kind = "asked instead"
            else:
                kind = "call or argument missing"
            behaviour[kind] += 1
            if kind != "right value anyway" and len(examples) < 14:
                examples.append(f"{i} {tool}.{path}: gold '{v}' -> {got[:1] or ('ask' if asked else '-')}")
    parts.append("## 5. Damaged stated arguments of validation samples that q35e1 gets right from gold text: what the ASR run outputs\n")
    parts.append(md_table(["behaviour", "arguments"], [[k, v] for k, v in behaviour.most_common()]) + "\n")
    parts.append("\n".join(f"- {e}" for e in examples) + "\n")
    text = "\n".join(parts)
    print(text)
    if a.md:
        a.md.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps({k: v for k, v in sets.items()}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
