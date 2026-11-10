"""Recovery trajectories: TRAIN rows where the first search call fails and the next call recovers, built from real ToolEnv results.

Why: TRAIN gold has ~0 searches that come back empty, so the model never saw what to do after one. On val (q35e1, ASR v2) the 15
samples that hit an empty result score 20 vs 70 overall (contacts.search with "Sếp Vũ", "Cô Thảo", a misheard "Thầy Minh Sang");
on public 75/634 samples hit one (52 with unseen apps) and most then repeat the same search or stop.

For each eligible SFT row (one search call with a query, in data_v2_e1d_all format), one variant:
  honorific   contacts: the model's typical query, honorific + name ("anh Dũng"), when the contact is stored as the bare name;
              recovery = the gold query (drop the honorific)
  asr_name    contacts, ASR copy whose transcript does not contain the gold name: the query is the misheard words aligned to the
              gold name; recovery = its last word (given name) if that finds the gold contact, else an empty query (list all)
  long_query  other search tools (news, maps, podcast_radio, library): the query is the gold query plus the neighbouring spoken
              words; recovery = the gold query
The failed call is executed in ToolEnv (it must return no hit) and kept in the context with "train": false, so
train_agent_sft_q35.py puts no loss on it: the model learns the recovery, never the mistake. The rest of the trajectory is replayed
and must reproduce every original tool result. Rows are written as extra rows (the clean originals stay).
    .venv/bin/python scripts/inject_recovery.py --input outputs/sft/data_v2_e1d_all/train.jsonl --output outputs/sft/recovery_e1d.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.data import items  # noqa: E402
from lifemate.toolenv import new_env  # noqa: E402

HON = {"anh", "chị", "em", "cô", "chú", "bác", "sếp", "thầy", "bạn", "ông", "bà", "dì", "cậu", "mợ", "cháu", "con", "vợ", "chồng",
       "mẹ", "bố", "ba", "má", "xếp", "chế", "o", "ôn"}
STOP = {"rồi", "cho", "của", "nha", "nhé", "với", "và", "là", "thì", "mà", "đi", "giúp", "giùm", "dùm", "em", "anh", "chị", "tôi", "mình", "nào", "đó", "này", "ấy", "nữa", "luôn"}
SEARCH = {"contacts__search", "news__search", "maps__search_place", "podcast_radio__search", "library__search"}
WORD = re.compile(r"[^\W_]+", re.U)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def hits(result: object) -> list[str]:
    """IDs of the entities a search returned (first list of dicts with an id)."""
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]:
                return [x["id"] for x in v]
    return []


def aligned_span(gold_text: str, asr_text: str, phrase: str) -> str | None:
    """ASR words aligned to the words of `phrase` inside the gold transcript."""
    g, a, p = words(gold_text), words(asr_text), words(phrase)
    if not p:
        return None
    start = next((i for i in range(len(g) - len(p) + 1) if g[i:i + len(p)] == p), None)
    if start is None:
        return None
    end = start + len(p)
    sm = difflib.SequenceMatcher(a=g, b=a, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i2 <= start or i1 >= end:
            continue
        if tag == "equal":
            out += a[j1 + max(0, start - i1): j1 + min(i2, end) - i1]
        else:
            out += a[j1:j2]
    return " ".join(out) or None


def call_name(m: dict) -> str:
    return m["tool_calls"][0]["function"]["name"]


def replay(sample_id: str, calls: list[tuple[str, dict]]) -> list:
    env = new_env(PROJECT / "data/train+public_test", "train", sample_id)
    return [env.call(n.replace("__", ".", 1), a) for n, a in calls]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gold_text = {it["id"]: it["audio_text"] for it in items(PROJECT / "data/train+public_test", "train")}
    stats, out = Counter(), []
    for line in a.input.open(encoding="utf-8"):
        row = json.loads(line)
        msgs = row["messages"]
        idx = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls") and call_name(m) in SEARCH
                    and isinstance(m["tool_calls"][0]["function"]["arguments"].get("query"), str)), None)
        if idx is None:
            continue
        stats["eligible"] += 1
        fn = call_name(msgs[idx])
        q = msgs[idx]["tool_calls"][0]["function"]["arguments"]["query"]
        request = json.loads(msgs[2]["content"])["request"]
        target = set(hits(json.loads(msgs[idx + 1]["content"])))
        kind, wrong, recover = None, None, None
        if fn == "contacts__search":
            qw = words(q)
            if row["variant"] == "asr" and " ".join(qw) not in " ".join(words(request)):
                span = aligned_span(gold_text[row["sample_id"]], request, q)
                if span:
                    span = " ".join(span.split()[-max(1, len(qw)):])   # a query as long as the name, not the whole garbled stretch
                if span and span != " ".join(qw):
                    kind, wrong = "asr_name", span
            elif qw and qw[0] not in HON:
                req = words(request)
                prev = next((req[i - 1] for i in range(1, len(req)) if req[i] == qw[0] and req[i - 1] in HON), None)
                if prev:                     # only the honorific the speaker actually used before the name
                    kind, wrong = "honorific", f"{prev} {q}"
        else:
            req = words(request)
            qw = words(q)
            start = next((i for i in range(len(req) - len(qw) + 1) if req[i:i + len(qw)] == qw), None)
            if qw and start is not None:
                left, right = req[max(0, start - 1):start], req[start + len(qw):start + len(qw) + 1]
                for cand in ([*left, q], [q, *right]):
                    if len(cand) > 1 and cand[0] not in STOP and cand[-1] not in STOP:
                        kind, wrong = "long_query", " ".join(cand)
                        break
        if not kind:
            stats["no_variant"] += 1
            continue
        before = [(call_name(m), m["tool_calls"][0]["function"]["arguments"]) for m in msgs[:idx] if m["role"] == "assistant" and m.get("tool_calls")]
        res = replay(row["sample_id"], before + [(fn, {"query": wrong})])
        if hits(res[-1]):
            stats[f"{kind}_not_empty"] += 1
            continue
        if kind == "asr_name":
            last = wrong.split()[-1]
            r_last = replay(row["sample_id"], before + [(fn, {"query": last})])[-1]
            recover = {"query": last} if target & set(hits(r_last)) else {}
        else:
            recover = {"query": q}
        # new call sequence: before, wrong (no loss), recovery, then the original calls after the gold search
        after_msgs = msgs[idx + 2:]
        after = [(call_name(m), m["tool_calls"][0]["function"]["arguments"]) for m in after_msgs if m["role"] == "assistant" and m.get("tool_calls")]
        res = replay(row["sample_id"], before + [(fn, {"query": wrong}), (fn, recover)] + after)
        if not target & set(hits(res[len(before) + 1])):
            stats[f"{kind}_recovery_misses"] += 1
            continue
        orig_after = [json.loads(m["content"]) for m in after_msgs if m["role"] == "tool"]
        if res[len(before) + 2:] != orig_after:
            stats[f"{kind}_replay_differs"] += 1
            continue
        new = msgs[:idx] + [
            {"role": "assistant", "content": "", "train": False, "tool_calls": [{"type": "function", "function": {"name": fn, "arguments": {"query": wrong}}}]},
            {"role": "tool", "name": fn, "content": json.dumps(res[len(before)], ensure_ascii=False)},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": fn, "arguments": recover}}]},
            {"role": "tool", "name": fn, "content": json.dumps(res[len(before) + 1], ensure_ascii=False)},
        ] + after_msgs
        out.append({**row, "messages": new, "recovery": {"kind": kind, "wrong": wrong, "recover": recover, "gold_query": q}})
        stats[kind] += 1
    with a.output.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=1), len(out))


if __name__ == "__main__":
    main()
