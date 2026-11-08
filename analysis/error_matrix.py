"""Error matrices of one model on the TRAIN validation ids, gold text (oracle run) next to ASR (asr run).
  python analysis/error_matrix.py --asr v200_q35e1_asrv2_dedup --oracle v200_q35e1_oracle_dedup [--md out.md]

Tables (all from run logs, the TRAIN labels and the TRAIN catalog):
  1. outcome matrix: oracle outcome x ASR outcome (perfect / hash ok but F1 < 1 / hash wrong), and the points each cell loses;
  2. decision matrix: gold action type x predicted action type (write, read only, ask, nothing), per transcript source;
  3. gold write calls: exact, same tool with wrong arguments, missing (and which write replaced it), plus extra writes;
  4. wrong arguments by schema class (ID / date-time / number / enum / fuzzy text / exact string), oracle vs ASR;
  5. points lost per gold app, oracle vs ASR.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "analysis"))
sys.path.insert(0, str(PROJECT / "src"))
import score as S  # noqa: E402

TIME_NAME = re.compile(r"(^|_)(date|time|start|end|due|at|until|from_date|to_date|deadline|schedule_time|datetime)($|_)")


def kind(name: str) -> str:
    return S.catalog().get(name, {}).get("kind", "?")


def arg_class(tool: str, arg: str) -> str:
    props = ((S.catalog().get(tool) or {}).get("parameters") or {}).get("properties", {})
    p = props.get(arg) or {}
    if "x-ref" in p or arg.endswith("_id") or arg.endswith("_ids"):
        return "ID"
    if "enum" in p:
        return "enum"
    if p.get("format") in ("date", "date-time", "time") or TIME_NAME.search(arg):
        return "date/time"
    if p.get("type") in ("number", "integer"):
        return "number"
    if p.get("x-match") == "fuzzy":
        return "fuzzy text"
    if p.get("type") in ("array", "object"):
        return "list/object"
    return "exact string"


def action(calls: list[dict]) -> str:
    if not calls:
        return "nothing"
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return "ask" if not any(kind(c["name"]) == "write" for c in calls) else "ask+write"
    if any(kind(c["name"]) == "write" for c in calls):
        return "write"
    return "read only"


def outcome(r: dict) -> str:
    return "perfect" if r["score"] == 1 else "hash ok, F1<1" if r["hash_ok"] else "hash wrong"


def md_table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    items = {x["id"]: x for x in map(json.loads, (S.ROOT / "train/train.jsonl").open(encoding="utf-8"))}
    runs = {}
    for name, prefix in (("oracle", a.oracle), ("asr", a.asr)):
        preds = S.load_run(PROJECT / "outputs/runs" / prefix)
        gh = S.gold_hashes([items[i] for i in preds])
        runs[name] = {i: {**S.score_sample(items[i], p, gh[i], 0.5), "calls": p.get("tool_calls") or []} for i, p in preds.items()}
    ids = sorted(set(runs["oracle"]) & set(runs["asr"]))
    n = len(ids)
    parts = [f"# Error matrices: {a.asr} vs {a.oracle} ({n} samples)\n"]
    parts.append(f"Score: gold text {100 * sum(runs['oracle'][i]['score'] for i in ids) / n:.2f}, ASR {100 * sum(runs['asr'][i]['score'] for i in ids) / n:.2f}\n")

    # 1. outcome matrix
    cells = collections.defaultdict(lambda: [0, 0.0])
    for i in ids:
        o, s = runs["oracle"][i], runs["asr"][i]
        c = cells[(outcome(o), outcome(s))]
        c[0] += 1
        c[1] += 1 - s["score"]
    keys = ["perfect", "hash ok, F1<1", "hash wrong"]
    rows = [[f"gold text: {ko}"] + [f"{cells[(ko, ks)][0]} ({100 * cells[(ko, ks)][1] / n:.1f})" for ks in keys] for ko in keys]
    parts.append("## 1. Outcome, gold text (rows) x ASR (columns): samples (points the ASR run loses in that cell)\n")
    parts.append(md_table(["", *[f"ASR: {k}" for k in keys]], rows) + "\n")

    # 2. decision matrix
    for name in ("oracle", "asr"):
        m = collections.Counter()
        for i in ids:
            m[(action(items[i]["label"]["tool_calls"]), action(runs[name][i]["calls"]))] += 1
        golds = ["write", "read only", "ask", "ask+write", "nothing"]
        preds = ["write", "read only", "ask", "ask+write", "nothing"]
        rows = [[f"gold: {g}", sum(m[(g, p)] for p in preds)] + [m[(g, p)] for p in preds] for g in golds]
        parts.append(f"## 2. Action type, gold (rows) x predicted (columns), {'gold text' if name == 'oracle' else 'ASR'}\n")
        parts.append(md_table(["", "n", *preds], rows) + "\n")

    # 3. gold write calls
    for name in ("oracle", "asr"):
        res = collections.Counter()
        replaced = collections.Counter()
        extra = collections.Counter()
        for i in ids:
            r, gold, pred = runs[name][i], items[i]["label"]["tool_calls"], runs[name][i]["calls"]
            linked = {j for j in r["gold_links"] if j is not None}
            pw_unlinked = [p for j, p in enumerate(pred) if j not in linked and kind(p["name"]) == "write"]
            for g, note in zip(gold, r["gold_notes"]):
                if kind(g["name"]) != "write":
                    continue
                if note["matched"]:
                    res["exact"] += 1
                elif note["same_name_pred"] is not None:
                    res["same tool, wrong args"] += 1
                else:
                    res["missing"] += 1
                    for p in pw_unlinked:
                        if p["name"] != g["name"]:
                            replaced[f"{g['name']} -> {p['name']}"] += 1
            gold_names = collections.Counter(g["name"] for g in gold)
            for p in pw_unlinked:
                if gold_names[p["name"]] == 0:
                    extra[p["name"]] += 1
        total = sum(res.values())
        parts.append(f"## 3. Gold write calls, {'gold text' if name == 'oracle' else 'ASR'} ({total} calls)\n")
        parts.append(md_table(["outcome", "calls", "share"], [[k, v, f"{100 * v / total:.0f}%"] for k, v in res.most_common()]) + "\n")
        if replaced:
            parts.append("Missing write replaced by: " + ", ".join(f"{k} ({v})" for k, v in replaced.most_common(12)) + "\n")
        if extra:
            parts.append("Writes with no gold call of that tool: " + ", ".join(f"{k} ({v})" for k, v in extra.most_common(12)) + "\n")

    # 4. wrong arguments by schema class
    cls = {name: collections.Counter() for name in runs}
    examples = collections.defaultdict(list)
    for name in runs:
        for i in ids:
            for note in runs[name][i]["gold_notes"]:
                if note["matched"] or not note["bad_args"]:
                    continue
                for arg in note["bad_args"]:
                    c = arg_class(note["name"], arg)
                    cls[name][c] += 1
                    if name == "asr" and len(examples[c]) < 8:
                        examples[c].append(f"{note['name']}.{arg}")
    keys = sorted(set(cls["oracle"]) | set(cls["asr"]), key=lambda k: -(cls["asr"][k] + cls["oracle"][k]))
    parts.append("## 4. Wrong arguments of same-tool calls, by schema class (count of argument mismatches)\n")
    parts.append(md_table(["class", "gold text", "ASR", "examples (ASR)"], [[k, cls["oracle"][k], cls["asr"][k], ", ".join(sorted(set(examples[k])))] for k in keys]) + "\n")

    # 5. points lost per gold app
    lost = {name: collections.Counter() for name in runs}
    count = collections.Counter()
    for i in ids:
        apps = {c["name"].split(".")[0] for c in items[i]["label"]["tool_calls"]} - {"sys"} or {"(none)"}
        for app in apps:
            count[app] += 1
            for name in runs:
                lost[name][app] += (1 - runs[name][i]["score"]) / len(apps)
    rows = [[app, count[app], f"{100 * lost['oracle'][app] / n:.2f}", f"{100 * lost['asr'][app] / n:.2f}"]
            for app in sorted(count, key=lambda k: -lost["asr"][k])[:20]]
    parts.append("## 5. Points lost per gold app (a sample's loss is split over its apps), top 20 by ASR loss\n")
    parts.append(md_table(["app", "samples", "lost, gold text", "lost, ASR"], rows) + "\n")

    text = "\n".join(parts)
    print(text)
    if a.md:
        a.md.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
