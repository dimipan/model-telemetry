#!/usr/bin/env python3
"""
04_evaluate.py — the whole experiment, in one table.

One claim:
    Latent recurrence does not necessarily indicate stalled reasoning. This
    tests whether recurrence becomes a more SPECIFIC stall signal when
    conditioned on the absence of reference-path progress.

Three arms, identical preprocessing, differing only in the ablation:
    recurrence_only     r_t
    gain_only          -g_t
    gated_recurrence    r_t * 1[n_new_t == 0]      <- proposed signal

Baselines: lexical 3-gram containment, token entropy, step length.

Protocol
--------
* Split by ORIGINAL PROBLEM. Every derivative of a problem — clean, both
  exact-copy variants, both verification variants — stays in one partition.
* REPEATED SPLITS (default 10). Problems are the independent unit; steps within
  a problem and variants of a problem are correlated, so a single split with a
  0.011 AUPRC difference tells you nothing. Reported as mean +/- sd across
  splits, plus the fraction of splits in which the gated arm beats BOTH
  constituents.
* Orientation of the four arms and of lexical overlap is PRESPECIFIED (+1) and
  never fitted. Token entropy and step length have their sign chosen on train
  and are labelled fitted nuisance baselines.
* Layer, centering mean, and thresholds are fitted on TRAIN only, per split.
* AUPRC is primary (stall steps are a minority). AUROC is supplementary.
* Exact-copy and verification are reported SEPARATELY.

Usage
-----
    python 04_evaluate.py                 # raw cosine
    python 04_evaluate.py --center        # if probe_recurrence.py says so
    python 04_evaluate.py --pooling mean  # representation ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from telemetry import (
    ARMS,
    EMBEDDING_DEPTH,
    FITTED_ORIENTATION,
    FIXED_ORIENTATION,
    RECURRENCE_WINDOW,
    WARMUP_STEPS,
    history_size,
    latent_recurrence,
    lexical_recurrence,
    parse_reference_intermediates,
    reference_progress,
    step_length,
)

NOMINAL_FPR = 0.05
PRIMARY_ARM = "gated_recurrence"
CONSTITUENTS = ("recurrence_only", "gain_only")
SIGNAL_ORDER = [
    "lexical_ngram", "token_entropy", "step_length", "history_size",
    "recurrence_only", "gain_only", "gated_recurrence", "gated_recurrence_soft",
]
BASELINES = ("lexical_ngram", "token_entropy", "step_length", "history_size")
EPS = 1e-12


# ---------------------------------------------------------------------------
def build(traces_path, acts_path, pooling):
    """Flat per-step table + the hidden-state matrix, indexed by row."""
    rows = [json.loads(l) for l in pathlib.Path(traces_path).read_text().splitlines() if l.strip()]
    acts = np.load(acts_path, allow_pickle=True)

    depths = acts["layer_ids"].tolist()
    prefix = "H_" if pooling == "boundary" else "Hmean_"
    X = {L: acts[f"{prefix}{L}"].astype(np.float32) for L in depths}
    tid = np.array([str(t) for t in acts["trace_id"]])
    sidx = acts["step_idx"]
    ent = acts["entropy"]

    by_trace = {}
    for i, t in enumerate(tid):
        by_trace.setdefault(t, []).append(i)
    for t in by_trace:
        by_trace[t] = sorted(by_trace[t], key=lambda i: sidx[i])

    recs = []
    for r in rows:
        t = r["trace_id"]
        if t not in by_trace:
            continue
        idx = by_trace[t]
        steps = r["steps"]
        if len(idx) != len(steps):
            continue

        interm = parse_reference_intermediates(r["gold_solution"])
        # Injected steps are scored result-only, identically to 02_inject, so the
        # gate the evaluator computes matches the gate the injector guaranteed.
        # (stall_labels == 1 marks injected steps in both exact_copy and
        # verification conditions.)
        ro = [lab == 1 for lab in r["stall_labels"]]
        prog = reference_progress(steps, interm, result_only_steps=ro)
        lex = lexical_recurrence(steps)
        slen = step_length(steps)
        hist = history_size(len(steps))

        for s in range(len(steps)):
            recs.append({
                "problem_id": r["problem_id"],
                "trace_id": t,
                "condition": r["condition"],
                "step": s,
                "row": idx[s],                       # index into X[L]
                "eligible": s >= WARMUP_STEPS,
                "label": int(r["stall_labels"][s]),
                "gain": float(prog["gain"][s]),
                "n_new": int(prog["n_new"][s]),
                "lexical_ngram": float(lex[s]),
                "token_entropy": float(ent[idx[s]]),
                "step_length": float(slen[s]),
                "history_size": float(hist[s]),
            })
    return recs, X, depths


def metrics(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def fit_threshold(s, nominal=NOMINAL_FPR):
    """Smallest value whose realized FPR is <= nominal.

    Quantiles break on sparse signals: lexical overlap is exactly 0.0 for most
    steps, so quantile(0.95) IS zero, every step clears it, and the reported FPR
    is 1.0. Any signal with a point mass at its minimum has this problem.
    """
    if len(s) == 0:
        return float("nan")
    for v in np.unique(s):
        if float((s >= v).mean()) <= nominal:
            return float(v)
    return float(np.unique(s)[-1] + 1e-9)


# ---------------------------------------------------------------------------
def run_split(recs, X, depths, seed, test_frac, center):
    """One problem-level split. Returns {signal: {metric: value}}."""
    pids = sorted({r["problem_id"] for r in recs})
    random.Random(seed).shuffle(pids)
    test_pids = set(pids[: int(round(test_frac * len(pids)))])

    tr_all = [r for r in recs if r["problem_id"] not in test_pids]
    te_all = [r for r in recs if r["problem_id"] in test_pids]

    # centering mean: TRAIN CLEAN steps only
    mus = {L: None for L in depths}
    if center:
        cl = [r["row"] for r in tr_all if r["condition"] == "clean"]
        for L in depths:
            mus[L] = X[L][cl].mean(0).astype(np.float64)

    # recurrence per trace, per depth (all steps; masking happens at scoring)
    by_trace = {}
    for r in recs:
        by_trace.setdefault(r["trace_id"], []).append(r)
    for t in by_trace:
        by_trace[t].sort(key=lambda r: r["step"])

    R = {L: {} for L in depths}
    for t, rs in by_trace.items():
        rows = [r["row"] for r in rs]
        for L in depths:
            R[L][t] = latent_recurrence(X[L][rows].astype(np.float64), mu=mus[L])

    def arm_vec(rows, arm, L):
        r = np.array([R[L][x["trace_id"]][x["step"]] for x in rows])
        g = np.array([x["gain"] for x in rows])
        n = np.array([x["n_new"] for x in rows])
        return ARMS[arm](r, g, n)

    def base_vec(rows, key):
        return np.array([x[key] for x in rows])

    # scoring uses eligible steps only
    tr = [r for r in tr_all if r["eligible"]]
    te = [r for r in te_all if r["eligible"]]
    sub = lambda rows, c: [r for r in rows if r["condition"] == c]

    # Layer: chosen on TRAIN, primary arm, verification condition.
    #
    # DEPTH 0 IS EXCLUDED FROM SELECTION. It is the embedding output, not a
    # transformer layer. A signal that already exists there reflects token
    # identity, punctuation or template style — not model computation — and would
    # gut the interpretation the study exists to support. It is retained as a
    # NEGATIVE CONTROL and reported separately.
    tr_ver = sub(tr, "verification")
    y_tr_ver = np.array([r["label"] for r in tr_ver])
    candidates = [L for L in depths if L != EMBEDDING_DEPTH] or list(depths)
    layer = max(candidates, key=lambda L: metrics(y_tr_ver, arm_vec(tr_ver, PRIMARY_ARM, L))[0])

    signals = {a: (lambda rows, a=a: arm_vec(rows, a, layer)) for a in ARMS}
    for b in BASELINES:
        signals[b] = lambda rows, k=b: base_vec(rows, k)

    # orientation: FIXED for the arms + lexical; fitted only for the nuisance pair
    tr_inj = [r for r in tr if r["condition"] != "clean"]
    y_tr_inj = np.array([r["label"] for r in tr_inj])
    orient = dict(FIXED_ORIENTATION)
    for name in FITTED_ORIENTATION:
        s = signals[name](tr_inj)
        orient[name] = 1.0 if metrics(y_tr_inj, s)[0] >= metrics(y_tr_inj, -s)[0] else -1.0

    tr_clean = sub(tr, "clean")
    thr = {n: fit_threshold(orient[n] * signals[n](tr_clean)) for n in signals}

    te_clean = sub(te, "clean")
    out = {"_layer": layer}
    for name in SIGNAL_ORDER:
        d = {"orientation": int(orient[name])}
        for cond in ("exact_copy", "verification"):
            rc = sub(te, cond)
            y = np.array([r["label"] for r in rc])
            s = orient[name] * signals[name](rc)
            ap, auc = metrics(y, s)
            d[f"{cond}_auprc"] = ap
            d[f"{cond}_auroc"] = auc
            d[f"{cond}_base"] = float(y.mean())
            d[f"{cond}_tpr"] = float((s[y == 1] >= thr[name]).mean())
        sc = orient[name] * signals[name](te_clean)
        d["fpr_clean"] = float((sc >= thr[name]).mean())
        out[name] = d

    # embedding-output negative control: the primary arm read at depth 0
    if EMBEDDING_DEPTH in depths:
        rc = sub(te, "verification")
        y = np.array([r["label"] for r in rc])
        ap0, _ = metrics(y, arm_vec(rc, PRIMARY_ARM, EMBEDDING_DEPTH))
        out["_embed_control_auprc"] = ap0

    # position diagnostic: are positives structurally later than negatives?
    rc = sub(te, "verification")
    pos = [r["step"] for r in rc if r["label"] == 1]
    neg = [r["step"] for r in rc if r["label"] == 0]
    out["_pos_mean_step"] = float(np.mean(pos)) if pos else float("nan")
    out["_neg_mean_step"] = float(np.mean(neg)) if neg else float("nan")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/injected_traces.jsonl")
    ap.add_argument("--acts", default="data/acts_injected.npz")
    ap.add_argument("--out", default="results/detector_results.csv")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--pooling", choices=["boundary", "mean"], default="boundary")
    ap.add_argument("--center", action="store_true",
                    help="mean-centre hidden states before cosine "
                         "(run probe_recurrence.py first to decide)")
    args = ap.parse_args()

    recs, X, depths = build(args.traces, args.acts, args.pooling)
    n_prob = len({r["problem_id"] for r in recs})
    print(f"[data] {sum(r['eligible'] for r in recs)} scored steps "
          f"({len(recs)} total) over {n_prob} problems")
    print(f"[cfg]  pooling={args.pooling} | centering={args.center} | "
          f"splits={args.splits} | nominal FPR={NOMINAL_FPR}\n")

    runs = [run_split(recs, X, depths, seed, args.test_frac, args.center)
            for seed in range(args.splits)]

    layers = [r["_layer"] for r in runs]
    print(f"[layer] selected on train, per split: {layers}")

    # ---- aggregate -----------------------------------------------------------
    rows_out = []
    for name in SIGNAL_ORDER:
        row = {"signal": name,
               "orientation": "fixed" if name in FIXED_ORIENTATION else "fitted"}
        for cond in ("exact_copy", "verification"):
            for m in ("auprc", "auroc", "tpr"):
                v = np.array([r[name][f"{cond}_{m}"] for r in runs])
                row[f"{cond}_{m}_mean"] = round(float(np.nanmean(v)), 4)
                row[f"{cond}_{m}_sd"] = round(float(np.nanstd(v)), 4)
            row[f"{cond}_base"] = round(float(np.mean([r[name][f"{cond}_base"] for r in runs])), 4)
        row["fpr_clean_mean"] = round(float(np.mean([r[name]["fpr_clean"] for r in runs])), 4)
        rows_out.append(row)

    # gated minus best constituent, per split — the actual claim
    deltas = np.array([
        r[PRIMARY_ARM]["verification_auprc"] - max(r[c]["verification_auprc"] for c in CONSTITUENTS)
        for r in runs
    ])
    wins = int((deltas > EPS).sum())
    ties = int((np.abs(deltas) <= EPS).sum())
    losses = args.splits - wins - ties

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with outp.with_name("split_deltas.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "layer", "gated_minus_best_constituent_verification_auprc"])
        for i, (r, d) in enumerate(zip(runs, deltas)):
            w.writerow([i, r["_layer"], round(float(d), 4)])

    # ---- table ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"TEST RESULTS  — mean +/- sd over {args.splits} problem-level splits")
    print("=" * 92)
    print(f"| {'Signal':<23} | {'sign':>6} | {'Copy AUPRC':>15} | "
          f"{'Verif AUPRC':>15} | {'Verif AUROC':>15} | {'FPR':>5} |")
    print("|" + "-" * 25 + "|" + "-" * 8 + "|" + "-" * 17 + "|" + "-" * 17
          + "|" + "-" * 17 + "|" + "-" * 7 + "|")
    for r in rows_out:
        mark = "**" if r["signal"] == PRIMARY_ARM else "  "
        print(f"| {mark}{r['signal']:<21} | {r['orientation']:>6} | "
              f"{r['exact_copy_auprc_mean']:>7.3f}±{r['exact_copy_auprc_sd']:<7.3f} | "
              f"{r['verification_auprc_mean']:>7.3f}±{r['verification_auprc_sd']:<7.3f} | "
              f"{r['verification_auroc_mean']:>7.3f}±{r['verification_auroc_sd']:<7.3f} | "
              f"{r['fpr_clean_mean']:>5.3f} |")
    print("=" * 92)
    print(f"chance AUPRC (verification base rate) = {rows_out[0]['verification_base']}")

    # ---- embedding-output negative control -----------------------------------
    if "_embed_control_auprc" in runs[0]:
        ec = float(np.nanmean([r["_embed_control_auprc"] for r in runs]))
        sel = float(np.nanmean([r[PRIMARY_ARM]["verification_auprc"] for r in runs]))
        print(f"\nEMBEDDING-OUTPUT CONTROL (depth 0, excluded from selection):")
        print(f"  primary arm at embedding output : {ec:.3f}")
        print(f"  primary arm at selected depth   : {sel:.3f}  (depths {layers})")
        if ec >= sel - 0.02:
            print("  !! the signal is ALREADY PRESENT at the embedding output. It reflects")
            print("     token identity / template style, not model computation. The")
            print("     internal-representation reading does NOT hold. Say so plainly.")
        else:
            print(f"  ok  the signal improves by {sel - ec:+.3f} inside the transformer")

    # ---- the claim, with uncertainty ----------------------------------------
    print("\n" + "-" * 92)
    print("THE CLAIM: gated recurrence beats BOTH constituents on verification stalls")
    print("-" * 92)
    print(f"  gated - best constituent, per split: {np.round(deltas, 3).tolist()}")
    print(f"  mean delta : {deltas.mean():+.4f} +/- {deltas.std():.4f}")
    print(f"  win / tie / loss : {wins} / {ties} / {losses}  (of {args.splits})")
    print("\n  NOTE: this is a STABILITY CHECK, not a confidence test. The ten test")
    print("  sets overlap heavily, so the splits are not independent. A grouped")
    print("  bootstrap would be needed for a real interval.")

    if wins >= 0.8 * args.splits and deltas.mean() > 0:
        print(f"\n  => STABLE ADVANTAGE UNDER THE PRESPECIFIED SPLIT CHECK.")
        print(f"     Gated beat both constituents in {wins}/{args.splits} problem-level")
        print(f"     splits, mean verification-AUPRC improvement "
              f"{deltas.mean():+.3f} ± {deltas.std():.3f}.")
    elif wins <= 0.2 * args.splits:
        print(f"\n  => NO ADVANTAGE. The gate does not help ({wins}/{args.splits}). Report this.")
    else:
        print(f"\n  => NOT STABLE across the repeated problem-level splits "
              f"({wins}/{args.splits}).")
        print(f"     Do not quote the mean without the split count.")

    # ---- artefact guards -----------------------------------------------------
    g = {r["signal"]: r["verification_auprc_mean"] for r in rows_out}
    print("\nARTEFACT GUARDS:")
    for name, msg in (
        ("step_length",
         "the templates are a LENGTH cue, not a stall cue"),
        ("history_size",
         "recurrence is rising because later steps have MORE CANDIDATES to\n"
         "     maximise over, not because the geometry means anything"),
        ("lexical_ngram",
         "the latent telemetry is adding nothing over surface repetition"),
    ):
        if g[name] >= g[PRIMARY_ARM] - 0.02:
            print(f"  !! {name} ({g[name]:.3f}) is competitive with the gated arm "
                  f"({g[PRIMARY_ARM]:.3f}).")
            print(f"     {msg}. Fix this before reporting anything else in the table.")
        else:
            print(f"  ok  {name} ({g[name]:.3f}) does not reach the gated arm "
                  f"({g[PRIMARY_ARM]:.3f})")

    pm = float(np.nanmean([r["_pos_mean_step"] for r in runs]))
    nm = float(np.nanmean([r["_neg_mean_step"] for r in runs]))
    print(f"\n  position check: mean step index of positives {pm:.2f} vs negatives {nm:.2f}")
    if abs(pm - nm) > 1.5:
        print("     positives are structurally offset in position. history_size above is")
        print("     the guard; if it is also non-competitive, position is not the story.")

    print(f"\n[done] {outp}")
    print(f"[done] {outp.with_name('split_deltas.csv')}")


if __name__ == "__main__":
    main()