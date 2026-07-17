#!/usr/bin/env python3
"""
tests/smoke_test.py — run the injection + evaluation pipeline on a synthetic
fixture. No model, no GPU, no downloads. Verifies plumbing and the load-bearing
invariants. It says nothing about whether the real signal works.

The fixture plants recurrence deliberately: injected stall steps get hidden
states close to an earlier step. If the pipeline is wired correctly, recurrence
signals must beat chance here. If they do not, the code is broken, not the idea.

    python tests/smoke_test.py
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from segmentation import reasoning_block, segment  # noqa: E402
from telemetry import (  # noqa: E402
    EMBEDDING_DEPTH,
    RECURRENCE_WINDOW,
    claimed_result,
    history_size,
    parse_reference_intermediates,
    reference_progress,
)

RNG = np.random.default_rng(0)
D = 64


# ---------------------------------------------------------------------------
def unit_tests():
    # 1. claimed_result must not fire on OPERANDS or on GIVENS
    assert claimed_result("We divide 48 by 2, giving 24.") == 24.0
    assert claimed_result("Multiplying 8 times 3 equals 24.") == 24.0
    assert claimed_result("8 times 3 is 24.") == 24.0
    assert claimed_result("Subtracting yields 16.") == 16.0
    assert claimed_result("That leaves 42.") == 42.0
    # NO explicit result marker + MULTIPLE numbers => ambiguous => None.
    # This is the false-progress path the earlier version left open: a setup step
    # restating GIVENS ("there are 8 boxes and 3 items") used to claim a result of
    # 3, so a reference intermediate whose value happened to be 3 was marked
    # resolved on a number that was never computed.
    assert claimed_result("First, there are 8 boxes and 3 items each.") is None
    assert claimed_result("He starts with 20 apples and eats 4 of them.") is None
    # single bare number is still a plausible result
    assert claimed_result("24") == 24.0
    print("[ok] scorer: results only; operands and multi-number givens claim nothing")

    # 2. distinct-value coverage: a repeated reference value is ONE slot
    gold = "a <<2*3=6>>6 then <<12/2=6>>6 then <<6+1=7>>7\n#### 7"
    interm = parse_reference_intermediates(gold)
    vals = [i["value"] for i in interm]
    assert vals == [6.0, 7.0], f"expected distinct-value coverage, got {vals}"
    print("[ok] scorer: duplicate reference values collapse to one slot (documented policy)")

    # 3. progress is monotone and only credits claimed results
    steps = ["We start with 12 items and 2 groups.",
             "Dividing gives 12 / 2 = 6.",
             "Adding one yields 7."]
    prog = reference_progress(steps, interm)
    assert prog["n_new"].tolist() == [0, 1, 1], prog["n_new"].tolist()
    assert prog["upsilon"][-1] == 1.0
    print("[ok] scorer: coverage monotone (result mode), operands do not resolve slots")

    # operand-match (Option C): a step stating both operands resolves the op,
    # even without writing the result. This is the default mode.
    gold_om = "compute <<8*3.5=28>>28 then <<52-8-28=16>>16\n#### 16"
    im = parse_reference_intermediates(gold_om)
    steps_desc = [
        "First I note the totals.",
        "I multiply 8 by 3.5 to find the distance.",   # operands 8, 3.5 -> resolves 28
        "Then I subtract 8 and 28 from 52.",           # operands 52,8,28 -> resolves 16
    ]
    r_res = reference_progress(steps_desc, im, match="result")["upsilon"][-1]
    r_op = reference_progress(steps_desc, im, match="operand")["upsilon"][-1]
    assert r_res < r_op, f"operand-match should recover more coverage: {r_res} vs {r_op}"
    assert r_op == 1.0, f"operand-match should fully cover this trace, got {r_op}"
    print(f"[ok] scorer: operand-match recovers coverage {r_res:.2f} -> {r_op:.2f} on described ops")

    # THE GATE MUST NOT LEAK under operand-match. Two cases:
    #
    # (a) simple restatement of an already-resolved op.
    steps_with_verif = steps_desc[:2] + [
        "Checking: 8 x 3.5 = 28, confirming the previous result.",
    ] + steps_desc[2:]
    ro = [False, False, True, False, False]
    prog_v = reference_progress(steps_with_verif, im, match="operand", result_only_steps=ro)
    assert prog_v["n_new"][2] == 0, f"simple restatement leaked (n_new={prog_v['n_new'][2]})"

    # (b) THE CROSS-TALK CASE found at n=300. A template restating "8/2=4" contains
    # the numbers 8, 2, 4. Under naive operand-match the incidental pair {8,4}
    # resolves an UNRELATED intermediate "8+4=12". Result-only scoring of the
    # injected step must prevent that.
    # Minimal faithful reproduction: intermediate 8+4=12 is UNRESOLVED, and an
    # injected step restating a different op ("8/2 ... 4") happens to contain both
    # 8 and 4 — the operands of 8+4. Naive operand-match resolves 12 by cross-talk;
    # result-only scoring of the injected step must not.
    im2 = parse_reference_intermediates("<<8+4=12>>12")
    trace = [
        "We have two groups.",
        "Working 8/2 once more produces 4.",   # INJECTED: contains 8 and 4
    ]
    naive = reference_progress(trace, im2, match="operand")
    assert naive["n_new"][1] == 1, "cross-talk mechanism should reproduce here"
    masked = reference_progress(trace, im2, match="operand",
                                result_only_steps=[False, True])
    assert masked["n_new"][1] == 0, (
        f"cross-talk leak NOT closed: injected step resolved a new intermediate "
        f"under result-only scoring (n_new={masked['n_new'][1]})"
    )
    print("[ok] scorer: result-only masking closes the operand cross-talk leak (n=300 bug)")

    # 4. history_size is the position/candidate-count confound guard
    h = history_size(12)
    assert h.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8], h.tolist()
    assert RECURRENCE_WINDOW == 8 and EMBEDDING_DEPTH == 0
    print("[ok] guards: history_size matches the recurrence candidate count")

    # 5. segmentation: pull the <think> block, drop the polished summary
    raw = ("<think>\nFirst I compute 8 times 3 = 24.\n"
           "Then I add 10, giving 34.\nWait, let me recheck 8 times 3 = 24.\n</think>\n\n"
           "**Step 1:** 8 x 3 = 24.\n**Final Answer:** \\boxed{34}")
    steps, method = segment(raw)
    assert method == "think_sentences", method
    assert len(steps) == 3, steps
    assert "boxed" not in " ".join(steps), "post-answer summary leaked into steps"
    # runaway with no </think> still segments, flagged as raw_sentences
    _, m2 = segment("First do this. Then do that. And finally the other thing.")
    assert m2 == "raw_sentences", m2
    print("[ok] segmentation: think block extracted, summary dropped, runaway handled")


# ---------------------------------------------------------------------------
def fake_originals(n=40):
    rows = []
    for i in range(n):
        a, b = 8 + i, 3
        c, d, e = a * b, a * b + 10, (a * b + 10) * 2
        gold = (
            f"He has {a} boxes with {b} each, so <<{a}*{b}={c}>>{c}.\n"
            f"Then he adds 10: <<{c}+10={d}>>{d}.\n"
            f"Finally he doubles it: <<{d}*2={e}>>{e}.\n#### {e}"
        )
        steps = [
            # deliberately claims NO result: givens only, multiple numbers, no marker
            f"First I identify the quantities: {a} boxes and {b} items each.",
            f"Multiplying {a} times {b} equals {c}.",
            f"Adding ten to that result produces {d}.",
            f"Doubling the total yields {e}.",
            f"So the final total comes out to {e}.",
        ]
        rows.append({
            "problem_id": f"gsm8k_fix{i:04d}",
            "question": f"fixture {i}",
            "gold_solution": gold,
            "gold_answer": str(e),
            "prompt": "PROMPT ",
            "raw_response": "",
            "canonical_response": "",
            "steps": steps,
            "used_fallback_segmenter": False,
            "n_steps": len(steps),
            "pred_answer": str(e),
            "correct": True,
            "gen_params": {},
        })
    return rows


def fake_acts(traces_path, out_path):
    rows = [json.loads(l) for l in pathlib.Path(traces_path).read_text().splitlines() if l.strip()]
    depths = [0, 7, 14, 21, 28]
    H = {L: [] for L in depths}
    Hm = {L: [] for L in depths}
    tids, sidx, ent = [], [], []

    for r in rows:
        T = len(r["steps"])
        base = RNG.normal(size=(T, D))
        for s in range(T):
            if r["stall_labels"][s] == 1 and s > 0:
                j = RNG.integers(0, s)
                base[s] = base[j] + 0.25 * RNG.normal(size=D)
        for L in depths:
            for s in range(T):
                v = (base[s] + 0.15 * RNG.normal(size=D)).astype(np.float16)
                H[L].append(v)
                Hm[L].append(v)
        for s in range(T):
            tids.append(r["trace_id"])
            sidx.append(s)
            ent.append(float(1.5 - 0.4 * r["stall_labels"][s] + 0.1 * RNG.normal()))

    payload = {f"H_{L}": np.stack(H[L]) for L in depths}
    payload.update({f"Hmean_{L}": np.stack(Hm[L]) for L in depths})
    payload["trace_id"] = np.array(tids)
    payload["step_idx"] = np.array(sidx)
    payload["entropy"] = np.array(ent, dtype=np.float32)
    payload["layer_ids"] = np.array(depths)
    np.savez_compressed(out_path, **payload)


# ---------------------------------------------------------------------------
def main():
    unit_tests()

    tmp = pathlib.Path(tempfile.mkdtemp())
    orig, inj = tmp / "original.jsonl", tmp / "injected.jsonl"
    acts, res = tmp / "acts.npz", tmp / "detector_results.csv"

    orig.write_text("\n".join(json.dumps(r) for r in fake_originals()) + "\n")

    # correct-only is the DEFAULT now — no flag needed
    subprocess.run([sys.executable, "02_inject.py", "--in", str(orig), "--out", str(inj)],
                   cwd=ROOT, check=True)

    inj_rows = [json.loads(l) for l in inj.read_text().splitlines() if l.strip()]
    conds = {}
    for r in inj_rows:
        conds[r["condition"]] = conds.get(r["condition"], 0) + 1
    assert conds.get("verification", 0) > 0, "no verification stalls built"
    assert conds.get("exact_copy", 0) > 0, "no exact-copy stalls built"

    # the load-bearing guarantee, re-verified independently of the write path
    for r in inj_rows:
        if r["condition"] != "verification":
            continue
        interm = parse_reference_intermediates(r["gold_solution"])
        prog = reference_progress(r["steps"], interm)
        for s, lab in enumerate(r["stall_labels"]):
            if lab == 1:
                assert prog["n_new"][s] == 0, (
                    f"verification step {s} of {r['trace_id']} resolved a new "
                    f"intermediate — the gate is leaking")
        assert r["injection"]["template_ids"], "template metadata not recorded"
    print("[ok] injection: verification stalls make zero reference-path progress")

    fake_acts(inj, acts)
    subprocess.run(
        [sys.executable, "04_evaluate.py", "--traces", str(inj), "--acts", str(acts),
         "--out", str(res), "--splits", "5"],
        cwd=ROOT, check=True,
    )

    import csv
    with res.open() as fh:
        out = {r["signal"]: r for r in csv.DictReader(fh)}

    assert "history_size" in out, "position confound guard missing from the table"
    gated = float(out["gated_recurrence"]["verification_auprc_mean"])
    base = float(out["gated_recurrence"]["verification_base"])
    assert gated > base, (
        f"gated AUPRC {gated} <= base rate {base} on a fixture where recurrence "
        f"was planted by construction — the pipeline is broken")

    # orientations must be declared, and the arms must not be fitted
    for arm in ("recurrence_only", "gain_only", "gated_recurrence"):
        assert out[arm]["orientation"] == "fixed", f"{arm} orientation was fitted"
    print("[ok] evaluation: core arm orientations are fixed, not fitted")
    print(f"[ok] evaluation: gated AUPRC {gated:.3f} > base rate {base:.3f} on planted fixture")

    print("\nSMOKE TEST PASSED — plumbing and invariants hold.")
    print("This says nothing about the real result.")


if __name__ == "__main__":
    main()