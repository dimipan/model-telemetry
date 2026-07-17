#!/usr/bin/env python3
"""
resegment.py — re-derive steps from raw_response without touching the GPU.

Segmentation is a pure text transform over saved raw responses, so changing the
segmenter never requires regenerating. Use this to migrate an existing
original_traces.jsonl to the current segmentation.py, or to re-segment after any
future change to the rule.

    python resegment.py --in data/original_traces.jsonl --out data/original_traces.jsonl

(in-place is fine; it writes to a temp file and moves it over)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import tempfile

from segmentation import segment
from telemetry import render_steps

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(response: str) -> str | None:
    for rx in (BOXED_RE, ANSWER_RE):
        m = rx.findall(response)
        if m:
            nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", m[-1])
            if nums:
                return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else None


def is_correct(pred, gold) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recheck-answer", action="store_true",
                    help="also re-extract the answer and correctness (boxed-aware)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text().splitlines() if l.strip()]

    methods = {}
    changed = 0
    n_correct = 0
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=".", suffix=".jsonl")
    with tmp:
        for r in rows:
            raw = r.get("raw_response", "")
            steps, method = segment(raw)
            canonical, _ = render_steps(steps)

            methods[method] = methods.get(method, 0) + 1
            if steps != r.get("steps"):
                changed += 1

            r["steps"] = steps
            r["segmentation_method"] = method
            r["n_steps"] = len(steps)
            r["canonical_response"] = canonical
            r.pop("used_fallback_segmenter", None)   # retired field

            if args.recheck_answer:
                pred = extract_answer(raw)
                r["pred_answer"] = pred
                r["correct"] = is_correct(pred, r["gold_answer"])
            n_correct += int(r.get("correct", False))

            tmp.write(json.dumps(r) + "\n")

    pathlib.Path(tmp.name).replace(args.out)

    print(f"[done] {args.out}")
    print(f"  traces           : {len(rows)}")
    print(f"  steps changed    : {changed}")
    print(f"  accuracy         : {n_correct/len(rows):.3f}")
    print(f"  segmentation methods:")
    for m, c in sorted(methods.items()):
        print(f"    {m:18s}: {c}")
    non_think = sum(c for m, c in methods.items() if m not in ("step_tags", "think_sentences"))
    print(f"  non-think regime : {non_think/len(rows):.3f}   <- want this LOW")


if __name__ == "__main__":
    main()
