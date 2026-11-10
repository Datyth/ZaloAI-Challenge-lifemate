"""Error analysis of TRAIN runs scored by analysis/score.py.
  python analysis/errors.py --asr v200_sft_asr --oracle v200_sft_oracle [--base-asr v200_base_asr --base-oracle v200_base_oracle]
Writes outputs/analysis/<asr>_errors.json (tables) and <asr>_cases.jsonl (one record per failed sample, for reading)."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "src"))
import score as S  # noqa: E402
from lifemate.metrics import error_counts  # noqa: E402

ROOT = S.ROOT
OUT = PROJECT / "outputs/analysis"


def kind(name: str) -> str:
    return S.catalog().get(name, {}).get("kind", "?")


def writes(calls):
    return [c for c in calls if kind(c["name"]) == "write"]


def reads(calls):
    return [c for c in calls if kind(c["name"]) == "read"]


def asks(calls):
    return [c for c in calls if c["name"] == "sys.ask_missing_info"]


def category(item: dict, pred: list[dict], r: dict) -> tuple[str, list[str]]:
    """Primary failure class of a sample with score < 1, plus detail tags."""
    gold = item["label"]["tool_calls"]
    tags = []
    if r["status"] == "error":
        return "runtime_error", [str(r.get("termination") or r.get("error", ""))[:60]]
    if not gold:
        return "acted_on_out_of_scope", sorted({c["name"] for c in pred})
    if not pred:
        return ("no_call_missing_info" if asks(gold) else "no_call"), []
    if asks(gold) and not asks(pred):
        return "missed_ask_missing_info", sorted({c["name"] for c in writes(pred)}) or ["no_write"]
    if asks(pred) and not asks(gold):
        return "asked_instead_of_acting", [a["arguments"].get("tool_name", "?") + ":" + ",".join(a["arguments"].get("params") or []) for a in asks(pred)]
    gw, pw = collections.Counter(c["name"] for c in writes(gold)), collections.Counter(c["name"] for c in writes(pred))
    if asks(gold) and asks(pred):
        ga, pa = asks(gold)[0]["arguments"], asks(pred)[0]["arguments"]
        if ga.get("tool_name") != pa.get("tool_name") or sorted(ga.get("params") or []) != sorted(pa.get("params") or []):
            tags.append(f"ask {pa.get('tool_name')}:{pa.get('params')} vs gold {ga.get('tool_name')}:{ga.get('params')}")
    if not r["hash_ok"]:
        if not pw and gw:
            return "no_write_after_reads", sorted(gw)
        missing, extra = gw - pw, pw - gw
        if missing and extra:
            return "wrong_write_tool", [f"-{k}" for k in sorted(missing)] + [f"+{k}" for k in sorted(extra)]
        if missing:
            return "missed_write", [f"-{k}" for k in sorted(missing)]
        if extra:
            return "extra_write", [f"+{k}" for k in sorted(extra)]
        bad = sorted({f"{n['name']}.{a}" for n in r["gold_notes"] if not n["matched"] and n["bad_args"] for a in n["bad_args"]
                      if kind(n["name"]) == "write"})
        return "write_args", bad or ["state_only"]
    # hash matches but the calls are not a perfect match
    unmatched = [n for n in r["gold_notes"] if not n["matched"]]
    if tags:
        return "ask_params", tags
    if any(kind(n["name"]) == "write" for n in unmatched):
        return "write_text_fields", sorted({f"{n['name']}.{a}" for n in unmatched if kind(n["name"]) == "write" for a in (n["bad_args"] or ["?"])})
    if r["n_pred"] > r["matched"] and not unmatched:
        return "extra_calls", sorted({c["name"] for i, c in enumerate(pred) if i not in set(r["gold_links"])})
    if unmatched:
        return "read_calls", sorted({f"{n['name']}" + (f".{','.join(n['bad_args'])}" if n["bad_args"] else ":missing") for n in unmatched})
    return "other", []


def hist_bucket(item):
    u = sum(1 for t in item["history"] if t.get("role") == "user")
    return "0" if u == 0 else "1-2" if u <= 2 else "3-5" if u <= 5 else "6+"


def ncall_bucket(item):
    n = len(item["label"]["tool_calls"])
    return str(n) if n < 4 else "4+"


def label_type(item):
    g = item["label"]["tool_calls"]
    if not g:
        return "out_of_scope"
    if asks(g):
        return "ask_missing_info"
    if not writes(g):
        return "read_only"
    return "write"


def wer_bucket(w):
    return "0" if w == 0 else "<10%" if w < .1 else "10-25%" if w < .25 else "25-50%" if w < .5 else ">=50%"


def table(rows, key):
    acc = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        k = key(r)
        acc[k][0] += 1
        acc[k][1] += r["score"]
    return {k: {"n": n, "score": round(100 * s / n, 1)} for k, (n, s) in sorted(acc.items(), key=lambda kv: str(kv[0]))}


def run_scores(prefix: str, items: dict) -> dict[str, dict]:
    preds = S.load_run(PROJECT / "outputs/runs" / prefix)
    gh = S.gold_hashes([items[i] for i in preds])
    out = {}
    for i, p in preds.items():
        r = S.score_sample(items[i], p, gh[i], 0.5)
        r["pred_calls"] = p.get("tool_calls") or []
        r["category"], r["tags"] = category(items[i], r["pred_calls"], r) if r["score"] < 1 else ("ok", [])
        out[i] = r
    return out


def transcripts(prefix: str) -> dict[str, str]:
    """Current-utterance transcript each sample was run with (from the run's asr_outputs log)."""
    found = {}
    for d in S.shard_dirs(PROJECT / "outputs/runs" / prefix):
        f = d / "asr_outputs.jsonl"
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                x = json.loads(line)
                if x["audio"].endswith(f"{x['sample_id']}.wav"):
                    found[x["sample_id"]] = x["raw_transcript"]
    return found


def run_events(prefix: str) -> dict[str, collections.Counter]:
    ev = collections.defaultdict(collections.Counter)
    for d in S.shard_dirs(PROJECT / "outputs/runs" / prefix):
        for name, key in (("tool_events.jsonl", "tool"), ("protocol_events.jsonl", "protocol"), ("llm_turns.jsonl", "turn")):
            f = d / name
            if not f.is_file():
                continue
            for line in f.open(encoding="utf-8"):
                x = json.loads(line)
                sid = x["sample_id"]
                if key == "tool":
                    ev[sid]["proposed"] += 1
                    ev[sid]["rejected"] += not x.get("executed")
                    if x.get("executed") and isinstance(x.get("result"), dict) and x["result"].get("done") is False:
                        ev[sid]["not_done"] += 1
                elif key == "protocol":
                    ev[sid][x.get("kind", "protocol")] += 1
                else:
                    ev[sid]["llm_turns"] += 1
                    ev[sid]["prompt_tokens_max"] = max(ev[sid]["prompt_tokens_max"], (x.get("usage") or {}).get("prompt_tokens") or 0)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--base-asr")
    ap.add_argument("--base-oracle")
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    runs = {"asr": run_scores(a.asr, items), "oracle": run_scores(a.oracle, items)}
    if a.base_asr:
        runs["base_asr"] = run_scores(a.base_asr, items)
    if a.base_oracle:
        runs["base_oracle"] = run_scores(a.base_oracle, items)
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    asr_text = transcripts(a.asr)
    wer = {i: (lambda c: c["word_errors"] / max(1, c["reference_words"]))(error_counts(items[i]["audio_text"], asr_text.get(i, ""))) for i in ids}
    events = {k: run_events(v) for k, v in (("asr", a.asr), ("oracle", a.oracle))}
    report: dict = {"n": len(ids), "overall": {}}
    for name, rs in runs.items():
        rows = [rs[i] for i in ids]
        lat = sorted(r["latency"] or 0 for r in rows)
        report["overall"][name] = {"score": round(100 * sum(r["score"] for r in rows) / len(rows), 2), "state_match": sum(r["hash_ok"] for r in rows),
                                   "perfect": sum(r["score"] == 1 for r in rows), "mean_f1": round(sum(r["f1"] for r in rows) / len(rows), 3),
                                   "runtime_errors": sum(r["status"] == "error" for r in rows), "latency_mean": round(sum(lat) / len(lat), 2),
                                   "latency_p95": round(lat[int(.95 * len(lat))], 2), "latency_max": round(lat[-1], 2)}
    # paired attribution: oracle vs ASR with the same model
    pair = collections.Counter()
    for i in ids:
        o, s = runs["oracle"][i]["score"] == 1, runs["asr"][i]["score"] == 1
        pair["both_right" if o and s else "asr_broke_it" if o else "asr_luck" if s else "both_wrong"] += 1
    report["paired_asr_vs_oracle"] = dict(pair)
    report["score_loss_from_asr"] = round(100 * sum(runs["oracle"][i]["score"] - runs["asr"][i]["score"] for i in ids) / len(ids), 2)
    if "base_asr" in runs:
        sft_gain = collections.Counter()
        for i in ids:
            b, s = runs["base_asr"][i]["score"] == 1, runs["asr"][i]["score"] == 1
            sft_gain["both" if b and s else "sft_fixed" if s else "sft_broke" if b else "neither"] += 1
        report["paired_sft_vs_base_asr"] = dict(sft_gain)
    for name in ("asr", "oracle"):
        rows = [runs[name][i] for i in ids]
        fails = [r for r in rows if r["score"] < 1]
        report[f"categories_{name}"] = dict(collections.Counter(r["category"] for r in fails).most_common())
        report[f"category_tags_{name}"] = {c: dict(collections.Counter(t for r in fails if r["category"] == c for t in r["tags"]).most_common(12))
                                            for c in report[f"categories_{name}"]}
        report[f"lost_points_by_category_{name}"] = dict(sorted(((c, round(100 * sum(1 - r["score"] for r in fails if r["category"] == c) / len(ids), 2))
                                                                 for c in report[f"categories_{name}"]), key=lambda kv: -kv[1]))
        report[f"by_label_type_{name}"] = table(rows, lambda r: label_type(items[r["id"]]))
        report[f"by_history_{name}"] = table(rows, lambda r: hist_bucket(items[r["id"]]))
        report[f"by_ncalls_{name}"] = table(rows, lambda r: ncall_bucket(items[r["id"]]))
        report[f"by_napps_{name}"] = table(rows, lambda r: len({c["name"].split(".")[0] for c in items[r["id"]]["label"]["tool_calls"] if c["name"] != "sys.ask_missing_info"}))
        app_rows = collections.defaultdict(list)
        for r in rows:
            for app in {c["name"].split(".")[0] for c in items[r["id"]]["label"]["tool_calls"]}:
                app_rows[app].append(r["score"])
        report[f"by_gold_app_{name}"] = {k: {"n": len(v), "score": round(100 * sum(v) / len(v), 1)} for k, v in sorted(app_rows.items(), key=lambda kv: -len(kv[1]))}
        ev = events[name]
        report[f"protocol_{name}"] = {"rejected_proposals": sum(ev[i]["rejected"] for i in ids), "samples_with_rejection": sum(ev[i]["rejected"] > 0 for i in ids),
                                      "executed_not_done": sum(ev[i]["not_done"] for i in ids), "fake_narration": sum(ev[i]["fake_tool_narration"] for i in ids),
                                      "mean_llm_turns": round(sum(ev[i]["llm_turns"] for i in ids) / len(ids), 2),
                                      "max_prompt_tokens": max(ev[i]["prompt_tokens_max"] for i in ids)}
    report["by_current_wer_asr"] = table([runs["asr"][i] for i in ids], lambda r: wer_bucket(wer[r["id"]]))
    report["by_current_wer_asr_minus_oracle"] = {}
    for b in sorted({wer_bucket(w) for w in wer.values()}):
        sel = [i for i in ids if wer_bucket(wer[i]) == b]
        report["by_current_wer_asr_minus_oracle"][b] = {"n": len(sel), "oracle": round(100 * sum(runs["oracle"][i]["score"] for i in sel) / len(sel), 1),
                                                       "asr": round(100 * sum(runs["asr"][i]["score"] for i in sel) / len(sel), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.asr}_errors.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / f"{a.asr}_cases.jsonl").open("w", encoding="utf-8") as f:
        for i in ids:
            o, s = runs["oracle"][i], runs["asr"][i]
            if o["score"] == 1 and s["score"] == 1:
                continue
            it = items[i]
            f.write(json.dumps({"id": i, "label_type": label_type(it), "history_users": sum(1 for t in it["history"] if t.get("role") == "user"),
                                "gold_text": it["audio_text"], "asr_text": asr_text.get(i), "wer": round(wer[i], 3),
                                "oracle": {"score": o["score"], "hash_ok": o["hash_ok"], "category": o["category"], "tags": o["tags"], "calls": o["pred_calls"]},
                                "asr": {"score": s["score"], "hash_ok": s["hash_ok"], "category": s["category"], "tags": s["tags"], "calls": s["pred_calls"]},
                                "gold": it["label"]["tool_calls"]}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
