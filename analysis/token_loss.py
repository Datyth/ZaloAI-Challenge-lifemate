"""Teacher-forced per-token loss of an agent adapter on whole trajectories, split by what each trained token encodes, so the
training loss can be read as "where does the model still pay": the call/stop decision, the function name, argument keys,
argument values (copied from speech, IDs from tool results, dates/times, numbers, other), or chat syntax.

Sets (each a list of trajectories in the SFT format, built exactly as scripts/prepare_agent_sft_v2.py builds them):
  train   rows the adapter was trained on (data_v2_e1/train.jsonl, a seeded sample)
  dev     the data_v2_e1 dev bucket: TRAIN tasks q35_e1_01 never saw (gold + base-Gipformer copies)
  val     the 200 validation TRAIN tasks, gold transcripts (val_gold) and ASR v2 lower-cased transcripts (val_asr, what inference reads)
The model is the bf16 base + LoRA (as vLLM serves it). Writes <out>/tokens.jsonl (one row per trained token), turns.jsonl, and
prints the summary.
    .../.venv-train/bin/python analysis/token_loss.py --adapter outputs/sft/q35_e1_01/adapter --out outputs/analysis/token_loss_q35e1
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import torch  # noqa: E402

SPAN = re.compile(r"<\|im_start\|>assistant\n(?:<think>\n\n</think>\n\n)?(.*?<\|im_end\|>)", re.S)
FUNC = re.compile(r"<function=([^>\n]+)>")
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$|^\d{1,2}:\d{2}$")
NUM = re.compile(r"^-?\d+(\.\d+)?$")
ID = re.compile(r"^[a-z]+_[0-9a-z_]+$")


def val_rows(variant: str) -> list[dict]:
    from prepare_agent_sft_v2 import trajectory
    from lifemate.context import normalize_asr
    from lifemate.data import catalog, items
    root = PROJECT / "data/train+public_test"
    ids = set(json.loads((PROJECT / "outputs/splits/validation_ids.json").read_text()))
    specs = catalog(root, "train")
    asr = {}
    if variant == "asr":
        for line in (PROJECT / "outputs/asr/gipformer-lm-v2-01_val/asr_outputs.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            asr[(r["sample_id"], r["audio"])] = normalize_asr(r["raw_transcript"])
    out = []
    for item in items(root, "train"):
        if item["id"] not in ids:
            continue
        audios = [item["audio"], *(t["audio"] for t in item["history"] if t.get("role") == "user" and "audio" in t)]
        if variant == "gold":
            text = {item["audio"]: item["audio_text"], **{t["audio"]: t["audio_text"] for t in item["history"] if t.get("role") == "user" and "audio" in t}}
        else:
            text = {x: asr[(item["id"], x)] for x in audios}
        out.append({"sample_id": item["id"], "variant": variant, **trajectory(item, root, specs, text, "E1")})
    return out


def categorize(text: str, spans: list[tuple[int, int]], user_text: str) -> list[list[tuple[int, int, str, dict]]]:
    """Per assistant span: (start, end, category, info) character regions covering the whole span."""
    out = []
    for s, e in spans:
        body = text[s:e]
        regs: list[tuple[int, int, str, dict]] = []
        if body == "<|im_end|>":
            out.append([(s, e, "stop", {})])
            continue
        f = FUNC.search(body)
        fn = f.group(1) if f else "?"
        if f:
            regs.append((s + f.start(1), s + f.end(1), "fn_name", {"fn": fn}))
        prefix = text[:s]
        for p in PARAM.finditer(body):
            key, val = p.group(1), p.group(2)
            regs.append((s + p.start(1), s + p.end(1), "param_key", {"fn": fn, "key": key}))
            v = val.strip()
            if ID.match(v) and v in prefix:
                kind = "value_ref"
            elif DATE.match(v):
                kind = "value_time"
            elif NUM.match(v):
                kind = "value_num"
            elif v.lower() in user_text.lower():
                kind = "value_copy"
            elif v.startswith(("[", "{")):
                kind = "value_struct"
            elif v in prefix:
                kind = "value_from_context"
            else:
                kind = "value_other"
            regs.append((s + p.start(2), s + p.end(2), kind, {"fn": fn, "key": key, "value": v}))
        regs.sort()
        filled, cur = [], s
        for a, b, c, i in regs:
            if a > cur:
                filled.append((cur, a, "syntax", {"fn": fn}))
            filled.append((a, b, c, i))
            cur = b
        if cur < e:
            filled.append((cur, e, "syntax", {"fn": fn}))
        out.append(filled)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--model", default=str(PROJECT / "models/qwen3.5-4b"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="dev,train,val_gold,val_asr")
    ap.add_argument("--train-sample", type=int, default=300)
    ap.add_argument("--limit", type=int, help="rows per set (smoke test)")
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "outputs/sft/data_v2_e1")
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from lifemate.llm import merge_system

    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    model = PeftModel.from_pretrained(base, str(a.adapter)).eval()
    lm = model.base_model.model
    sets: dict[str, list[dict]] = {}
    for name in a.sets.split(","):
        if name == "dev":
            sets[name] = [json.loads(l) for l in (a.data_dir / "dev.jsonl").open(encoding="utf-8")]
        elif name == "train":
            rows = [json.loads(l) for l in (a.data_dir / "train.jsonl").open(encoding="utf-8")]
            sets[name] = random.Random(0).sample(rows, min(a.train_sample, len(rows)))
        elif name.startswith("val_"):
            sets[name] = val_rows(name[4:])
    if a.limit:
        sets = {k: v[:a.limit] for k, v in sets.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    ftok = (a.out / "tokens.jsonl").open("w", encoding="utf-8")
    fturn = (a.out / "turns.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for name, rows in sets.items():
        for n, row in enumerate(rows):
            msgs = merge_system(row["messages"])
            text = tok.apply_chat_template(msgs, tools=row["tools"], tokenize=False, enable_thinking=False)
            spans = [(m.start(1), m.end(1)) for m in SPAN.finditer(text)]
            user_turns = [m for m in row["messages"] if m["role"] == "user"]
            user_text = user_turns[0]["content"] if user_turns else ""
            regions = categorize(text, spans, user_text if isinstance(user_text, str) else json.dumps(user_text, ensure_ascii=False))
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            # trained tokens: inside an assistant span (same rule as train_agent_sft_q35.encode)
            lab = []
            j = 0
            for t, (s, e) in enumerate(offs):
                while j < len(spans) and s >= spans[j][1]:
                    j += 1
                if j < len(spans) and s >= spans[j][0] and e <= spans[j][1] and e > s:
                    lab.append((t, j))
            if not lab or len(ids) > 16384:
                continue
            pos = torch.tensor([t - 1 for t, _ in lab], device="cuda")
            with torch.inference_mode():
                out = lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                h = out.last_hidden_state[0, pos]
                logits = lm.lm_head(h).float()
                logp = torch.log_softmax(logits, -1)
                tgt = torch.tensor([ids[t] for t, _ in lab], device="cuda")
                nll = (-logp.gather(1, tgt[:, None])[:, 0]).tolist()
                top = logits.argmax(-1).tolist()
            turn_stats: dict[int, dict] = defaultdict(lambda: {"tokens": 0, "wrong": 0, "nll": 0.0, "first_wrong": None})
            first_in_turn = set()
            seen_turn = set()
            for (t, k), l, p in zip(lab, nll, top):
                s, e = offs[t]
                cat, info = "syntax", {}
                for a0, b0, c, i in regions[k]:
                    if a0 <= s < b0 or (s < a0 < e):
                        cat, info = c, i
                        break
                if k not in seen_turn:
                    seen_turn.add(k)
                    first_in_turn.add(t)
                    cat = "decision"
                ok = p == ids[t]
                ts = turn_stats[k]
                ts["tokens"] += 1
                ts["nll"] += l
                if not ok:
                    ts["wrong"] += 1
                    if ts["first_wrong"] is None:
                        ts["first_wrong"] = {"cat": cat, "gold": tok.decode([ids[t]]), "pred": tok.decode([p]), **info}
                ftok.write(json.dumps({"set": name, "sample_id": row["sample_id"], "variant": row.get("variant"), "turn": k,
                                       "n_turns": len(spans), "cat": cat, "tok": tok.decode([ids[t]]), "pred": tok.decode([p]),
                                       "nll": round(l, 5), "ok": ok, **{x: info[x] for x in ("fn", "key") if x in info}},
                                      ensure_ascii=False) + "\n")
            for k, ts in sorted(turn_stats.items()):
                body = text[spans[k][0]:spans[k][1]]
                f = FUNC.search(body)
                fturn.write(json.dumps({"set": name, "sample_id": row["sample_id"], "variant": row.get("variant"), "turn": k,
                                        "n_turns": len(spans), "gold_fn": f.group(1) if f else "STOP", "prompt_tokens": spans[k][0],
                                        **ts, "nll": round(ts["nll"], 4)}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"{name} {n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    ftok.close()
    fturn.close()
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
