"""Local approximation of the LifeMate score on TRAIN runs (the kit ships no scorer; see data/.../zaic_tools/README.md).

Per sample: gold empty -> 1 if the prediction is empty, else 0. Otherwise the final state hash is a gate (0 when it differs from
the hash of replaying the gold calls), and the score is the F1 of a one-to-one matching of predicted and gold calls: same tool
name, equal non-fuzzy arguments (strings lower-cased, whitespace collapsed, number+unit joined, scalar lists as sets) and, for
x-match: fuzzy arguments, token F1 >= --fuzzy (on zaic_tools' accent-folded tokens). The score is the mean x 100.
  python analysis/score.py outputs/runs/v200_sft_oracle [more run prefixes...]
A prefix matches outputs/runs/<prefix> and its shards <prefix>_s<k>."""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/train+public_test"
sys.path.insert(0, str(PROJECT / "src"))
from lifemate.submission import shard_dirs  # noqa: E402
from lifemate.toolenv import new_env  # noqa: E402

GOLD_HASH = PROJECT / "outputs/analysis/gold_hash_train.json"


def fold(s) -> str:
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def tokens(s) -> list[str]:
    return re.findall(r"[^\W_]+", fold(s))


def token_f1(a, b) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    return 0.0 if not inter else 2 * inter / (len(ta) + len(tb))


def canon(v):
    if isinstance(v, str):
        return re.sub(r"(\d)\s+(?=[^\W\d_])", r"\1", " ".join(v.lower().split()))
    if isinstance(v, list):
        out = [canon(x) for x in v]
        return sorted(out, key=lambda x: (type(x).__name__, str(x))) if all(not isinstance(x, (dict, list)) for x in out) else out
    if isinstance(v, dict):
        return {k: canon(x) for k, x in v.items()}
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


@functools.lru_cache(None)
def catalog() -> dict:
    raw = json.loads((ROOT / "train/train_tools_catalog.json").read_text(encoding="utf-8"))
    return {t["name"]: t for a in raw["apps"] for t in a["tools"]}


def fuzzy_keys(name: str) -> set[str]:
    props = (catalog().get(name, {}).get("parameters") or {}).get("properties", {})
    return {k for k, p in props.items() if isinstance(p, dict) and p.get("x-match") == "fuzzy"}


def arg_diff(pred: dict, gold: dict, fuzzy_min: float) -> list[str]:
    """Names of the arguments that keep pred from matching gold (empty list = match)."""
    fz = fuzzy_keys(gold["name"])
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    bad = []
    for k in sorted(set(pa) | set(ga)):
        if k not in pa or k not in ga:
            bad.append(k)
        elif k in fz:
            if token_f1(pa[k], ga[k]) < fuzzy_min:
                bad.append(k)
        elif canon(pa[k]) != canon(ga[k]):
            bad.append(k)
    return bad


def match(pred: list[dict], gold: list[dict], fuzzy_min: float) -> tuple[int, list[int | None], list[dict]]:
    """Greedy one-to-one matching in gold order; returns matches, gold->pred index, and per-gold nearest-miss notes."""
    used, link, notes = set(), [], []
    for g in gold:
        hit, near = None, None
        for j, p in enumerate(pred):
            if j in used or p["name"] != g["name"]:
                continue
            diff = arg_diff(p, g, fuzzy_min)
            if not diff:
                hit = j
                break
            if near is None or len(diff) < len(near[1]):
                near = (j, diff)
        if hit is not None:
            used.add(hit)
        link.append(hit)
        notes.append({"name": g["name"], "matched": hit is not None, "same_name_pred": near[0] if near else None, "bad_args": near[1] if near else None})
    return len(used), link, notes


def gold_hashes(items: list[dict]) -> dict[str, str]:
    cached = json.loads(GOLD_HASH.read_text()) if GOLD_HASH.is_file() else {}
    missing = [x for x in items if x["id"] not in cached]
    for x in missing:
        env = new_env(ROOT, "train", x["id"])
        for c in x["label"]["tool_calls"]:
            env.call(c["name"], c["arguments"])
        cached[x["id"]] = env.get_state_hash()
    if missing:
        GOLD_HASH.parent.mkdir(parents=True, exist_ok=True)
        GOLD_HASH.write_text(json.dumps(cached, indent=0))
    return cached


def load_run(prefix: Path) -> dict[str, dict]:
    dirs = shard_dirs(prefix)
    rows = {}
    for d in dirs:
        for line in (d / "predictions.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["id"]] = {**r, "run_dir": str(d)}
    return rows


def score_sample(item: dict, pred: dict, ghash: str, fuzzy_min: float) -> dict:
    gold = item["label"]["tool_calls"]
    calls = pred.get("tool_calls") or []
    hash_ok = pred.get("state_hash") == ghash
    m, link, notes = match(calls, gold, fuzzy_min)
    f1 = 1.0 if not calls and not gold else (2 * m / (len(calls) + len(gold)))
    score = (1.0 if not calls else 0.0) if not gold else (f1 if hash_ok else 0.0)
    return {"id": item["id"], "score": score, "hash_ok": hash_ok, "f1": f1, "matched": m, "n_pred": len(calls), "n_gold": len(gold),
            "gold_links": link, "gold_notes": notes, "status": pred.get("status"), "termination": pred.get("termination"),
            "error": pred.get("error"), "latency": pred.get("latency_seconds")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--fuzzy", type=float, default=0.5)
    ap.add_argument("--write", action="store_true", help="write <prefix>.scores.jsonl next to the runs")
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    for prefix in a.runs:
        preds = load_run(prefix)
        ghash = gold_hashes([items[i] for i in preds])
        rows = [score_sample(items[i], preds[i], ghash[i], a.fuzzy) for i in sorted(preds)]
        n = len(rows)
        print(f"{prefix.name}: n={n} score={100 * sum(r['score'] for r in rows) / n:.2f} state_match={sum(r['hash_ok'] for r in rows)}/{n} "
              f"mean_f1={sum(r['f1'] for r in rows) / n:.3f} errors={sum(r['status'] == 'error' for r in rows)}")
        if a.write:
            out = PROJECT / "outputs/analysis" / f"{prefix.name}.scores.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
