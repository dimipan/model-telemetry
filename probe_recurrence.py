#!/usr/bin/env python3
"""
probe_recurrence.py — chooses the PRIMARY representation. Run before the study.

Transformer residual streams are strongly anisotropic: a handful of rogue
dimensions dominate the norm, so the raw cosine between ANY two hidden states
can sit at 0.98-0.999. In that regime r_t technically varies but carries almost
nothing, and every downstream AUPRC is noise wearing a result's clothes.

That risk is INVISIBLE in the results table, so it is checked directly.

IMPORTANT: this probe chooses which analysis is PRIMARY. It does not decide
whether the study exists. A compressed cosine band is not proof of absent signal
— a small but consistent rank shift can still discriminate — and a wide band is
not proof of present signal, since it could be driven by position or formatting
(that is what the history_size and step_length rows in the results table test).

Rule of thumb: if the raw p05-p95 band on CLEAN steps is narrower than ~0.05,
CENTERED recurrence becomes the prespecified primary analysis and raw is reported
as a robustness comparison. Both use the same extracted activations, so run both:

    python 04_evaluate.py --out results/raw.csv
    python 04_evaluate.py --center --out results/centered.csv

Log the choice in NOTES.md BEFORE looking at the results table.

Usage
-----
    python 03_extract.py --in data/injected_traces.jsonl --out data/acts_probe.npz \
                         --condition clean --limit 40
    python probe_recurrence.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from telemetry import latent_recurrence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts_probe.npz")
    ap.add_argument("--traces", default="data/injected_traces.jsonl")
    ap.add_argument("--pooling", choices=["boundary", "mean"], default="boundary")
    args = ap.parse_args()

    acts = np.load(args.acts, allow_pickle=True)
    depths = acts["layer_ids"].tolist()
    prefix = "H_" if args.pooling == "boundary" else "Hmean_"
    tid = np.array([str(t) for t in acts["trace_id"]])
    sidx = acts["step_idx"]

    cond = {}
    for line in pathlib.Path(args.traces).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cond[r["trace_id"]] = r["condition"]

    clean_rows = np.array([cond.get(t) == "clean" for t in tid])
    print(f"[probe] {clean_rows.sum()} clean step vectors, "
          f"{len(np.unique(tid[clean_rows]))} clean traces\n")

    if clean_rows.sum() < 20:
        raise SystemExit("not enough clean steps to probe; extract more traces")

    hdr = (f"{'depth':>6} {'mode':>10} {'p05':>8} {'p50':>8} {'p95':>8} "
           f"{'spread':>8}   verdict")
    print(hdr)
    print("-" * len(hdr))

    verdicts = {}
    for L in depths:
        X = acts[f"{prefix}{L}"].astype(np.float64)
        mu = X[clean_rows].mean(0)

        for mode, centre in (("raw", None), ("centered", mu)):
            vals = []
            for t in np.unique(tid[clean_rows]):
                rows = np.where(tid == t)[0]
                rows = rows[np.argsort(sidx[rows])]
                r = latent_recurrence(X[rows], mu=centre)
                vals.extend(r[1:])          # step 0 has no history
            v = np.array(vals)
            p05, p50, p95 = np.percentile(v, [5, 50, 95])
            spread = p95 - p05
            flag = "COMPRESSED" if spread < 0.05 else "ok"
            if mode == "raw":
                verdicts[L] = flag
            print(f"{L:>6} {mode:>10} {p05:>8.4f} {p50:>8.4f} {p95:>8.4f} "
                  f"{spread:>8.4f}   {flag}")
        print()

    n_bad = sum(v == "COMPRESSED" for v in verdicts.values())
    print("=" * 70)
    if n_bad >= len(depths) - 1:
        print("CHOICE: raw cosine is strongly compressed at nearly every depth.")
        print("  -> CENTERED recurrence is the prespecified PRIMARY analysis")
        print("  -> raw is still reported, as a robustness comparison")
    elif n_bad:
        print(f"CHOICE: raw cosine is compressed at {n_bad}/{len(depths)} depths.")
        print("  -> centered is the safer primary; report both, note any disagreement")
    else:
        print("CHOICE: raw cosine has usable spread at every depth.")
        print("  -> RAW is the primary analysis; centered is the robustness comparison")
    print("=" * 70)
    print("\nRun BOTH either way (same activations, seconds apart):")
    print("  python 04_evaluate.py           --out results/raw.csv")
    print("  python 04_evaluate.py --center  --out results/centered.csv")
    print("\nWrite the primary choice into NOTES.md BEFORE reading either table.")
    print("\nSpread is necessary, not sufficient: a wide band can be driven by step")
    print("position or length rather than recurrence. The history_size and")
    print("step_length rows in the results table are what test that.")


if __name__ == "__main__":
    main()
