#!/usr/bin/env python3
"""
02_inject.py — synthetic stall conditions with exact step labels.

Runs BEFORE activation extraction. Edited traces get their own teacher-forced
forward pass in 03_extract.py; original activations are never reused for an
edited sequence.

  A. exact_copy    Duplicate steps [i, i+k) verbatim. Pipeline validation and an
                   upper-bound sanity check. Lexical n-grams are EXPECTED to win
                   here. This is not the experiment.

  B. verification  Insert k templated steps that restate an ALREADY-RESOLVED
                   reference intermediate. Semantic recurrence, low lexical
                   identity, zero new reference-path progress. THIS is the
                   experiment.

The no-gain guarantee is asserted on the WRITE PATH — every candidate verification
trace is re-scored end to end before it can be written, and the run ABORTS (assert,
not retry) if an inserted step resolved a reference intermediate. Retrying would
paper over a scorer/injector disagreement; that is a bug, not bad luck.

Correct-only is the DEFAULT. The safest protocol should be the bare command;
use --include-incorrect to opt out.

Usage
-----
    python 02_inject.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from telemetry import claimed_result, parse_reference_intermediates, reference_progress

# Deliberately spread across a wide length range. Uniformly long templates make
# step_length a trivially winning detector, which turns the result into a length
# artefact rather than a stall signal. 04_evaluate.py checks for this explicitly.
VERIFICATION_TEMPLATES = [
    # short
    "Checking: {expr} = {value}.",
    "That is right, {expr} gives {value}.",
    "Re-checking {expr}: still {value}.",
    # medium
    "Let me verify that calculation. {expr} gives {value}, confirming the previous result.",
    "Checking once more: {expr} = {value}. The earlier calculation is consistent.",
    "To be safe, I will re-derive this. Working {expr} once more produces {value}.",
    # long
    "Before continuing, I should confirm the arithmetic here, because an error at this "
    "stage would propagate through everything after it. Evaluating {expr} again yields "
    "{value}, which matches what I already had.",
    "It is worth double-checking this intermediate value rather than assuming it is "
    "right. Recomputing {expr} still comes out to {value}, so the earlier line stands "
    "and I can proceed from there.",
]


def fmt_value(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def resolved_before(step_texts, intermediates, idx) -> list[int]:
    """Indices of reference intermediates resolved strictly before step `idx`.

    Uses the same claimed-result rule as the scorer, so the gate and the
    injection agree by construction.
    """
    if idx == 0:
        return []
    results = [claimed_result(s) for s in step_texts[:idx]]
    out = []
    for i, item in enumerate(intermediates):
        tol = max(1e-6 * abs(item["value"]), 1e-6)
        if any(r is not None and abs(r - item["value"]) <= tol for r in results):
            out.append(i)
    return out


def make_exact_copy(steps, rng, k):
    T = len(steps)
    if T < k + 2:
        return None
    i = rng.randrange(1, T - k)            # never duplicate step 0 (warmup)
    new_steps = steps[: i + k] + list(steps[i : i + k]) + steps[i + k :]
    labels = [0] * (i + k) + [1] * k + [0] * (T - i - k)
    return new_steps, labels, {"span_start": i, "k": k}


def make_verification(steps, intermediates, rng, k):
    """Insert k templated no-gain verification steps."""
    T = len(steps)
    if T < 3 or not intermediates:
        return None

    for _ in range(20):
        i = rng.randrange(2, T)
        avail = resolved_before(steps, intermediates, i)
        if not avail:
            continue

        inserted, tpl_ids, int_ids = [], [], []
        for _ in range(k):
            ii = rng.choice(avail)
            ti = rng.randrange(len(VERIFICATION_TEMPLATES))
            item = intermediates[ii]
            inserted.append(
                VERIFICATION_TEMPLATES[ti].format(
                    expr=item["expr"], value=fmt_value(item["value"])
                )
            )
            tpl_ids.append(ti)
            int_ids.append(ii)

        new_steps = steps[:i] + inserted + steps[i:]
        labels = [0] * i + [1] * k + [0] * (T - i)

        # Re-score the completed trace before it can
        # be written. Injected steps are scored RESULT-ONLY (result_only_steps
        # mask): they restate a value, and must be credited only on that value,
        # never on incidental operand co-occurrence. Under that rule an inserted
        # restatement of an ALREADY-RESOLVED intermediate cannot resolve a new one.
        # If this assertion still fires, the scorer and injector have genuinely
        # drifted apart — a code inconsistency, not an unlucky candidate. Abort;
        # do not retry, which would paper over it.
        ro = [lab == 1 for lab in labels]
        prog = reference_progress(new_steps, intermediates, result_only_steps=ro)
        leaked = [s for s, lab in enumerate(labels)
                  if lab == 1 and prog["n_new"][s] != 0]
        assert not leaked, (
            f"INVARIANT VIOLATED: verification steps {leaked} resolved a new "
            f"reference intermediate even under result-only scoring. The scorer "
            f"and injector disagree. Do not run the study until this is understood."
        )

        return new_steps, labels, {
            "insert_at": i, "k": k,
            "template_ids": tpl_ids,
            "intermediate_indices": int_ids,
        }
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/original_traces.jsonl")
    ap.add_argument("--out", default="data/injected_traces.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=2, help="stall span length in steps")
    ap.add_argument("--variants", type=int, default=2, help="injections per family per trace")
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument(
        "--include-incorrect",
        action="store_true",
        help="also inject into traces with an incorrect final answer "
             "(v0.1 default is correct-only)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text().splitlines() if l.strip()]

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    stats = {"skipped_short": 0, "skipped_incorrect": 0, "skipped_no_interm": 0,
             "clean": 0, "exact_copy": 0, "verification": 0, "verification_failed": 0}
    coverages = []

    with outp.open("w") as fh:
        for r in rows:
            steps = r["steps"]
            if len(steps) < args.min_steps:
                stats["skipped_short"] += 1
                continue
            if not args.include_incorrect and not r["correct"]:
                stats["skipped_incorrect"] += 1
                continue

            interm = parse_reference_intermediates(r["gold_solution"])
            if not interm:
                stats["skipped_no_interm"] += 1
                continue

            # Alignment diagnostic. How much of the reference path does the
            # UNEDITED trace cover? Recorded, never filtered on.
            coverage = float(reference_progress(steps, interm)["upsilon"][-1])
            coverages.append(coverage)

            base = {
                "problem_id": r["problem_id"],
                "question": r["question"],
                "gold_solution": r["gold_solution"],
                "prompt": r["prompt"],
                "correct": r["correct"],
                "reference_coverage": coverage,
                "n_reference_intermediates": len(interm),
            }

            fh.write(json.dumps({**base,
                                 "trace_id": f"{r['problem_id']}::clean",
                                 "condition": "clean",
                                 "steps": steps,
                                 "stall_labels": [0] * len(steps),
                                 "injection": None}) + "\n")
            stats["clean"] += 1

            for v in range(args.variants):
                ec = make_exact_copy(steps, rng, args.k)
                if ec:
                    ns, lb, meta = ec
                    stats["exact_copy"] += 1
                    fh.write(json.dumps({**base,
                                         "trace_id": f"{r['problem_id']}::exact_copy::{v}",
                                         "condition": "exact_copy",
                                         "steps": ns, "stall_labels": lb,
                                         "injection": meta}) + "\n")

                vf = make_verification(steps, interm, rng, args.k)
                if vf:
                    ns, lb, meta = vf
                    stats["verification"] += 1
                    fh.write(json.dumps({**base,
                                         "trace_id": f"{r['problem_id']}::verification::{v}",
                                         "condition": "verification",
                                         "steps": ns, "stall_labels": lb,
                                         "injection": meta}) + "\n")
                else:
                    stats["verification_failed"] += 1

    print(f"[done] {outp}")
    for k, v in stats.items():
        print(f"  {k:22s}: {v}")

    if coverages:
        import statistics as st
        print(f"\n  reference coverage of unedited traces:")
        print(f"    mean {st.mean(coverages):.3f} | median {st.median(coverages):.3f} "
              f"| frac at 1.0: {sum(c >= 0.999 for c in coverages)/len(coverages):.3f}")
        print("    (low coverage => the gate is firing on correct alternative")
        print("     derivations, not just on stalls. that is a finding, not a bug.)")

    if stats["verification"] == 0:
        raise SystemExit("FATAL: no verification stalls could be built.")


if __name__ == "__main__":
    main()