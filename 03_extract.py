#!/usr/bin/env python3
"""
03_extract.py — teacher-forced activation extraction at step boundaries.

Pass B. One forward pass over prompt + CANONICALISED response with
output_hidden_states=True and use_cache=False, then slice step-boundary
representations.

"Canonicalised" is the honest word: extraction runs over the segmented steps
re-wrapped in fresh <step> tags (telemetry.render_steps), not over the raw
generated string. Untagged preamble and the <answer> block are dropped. This is
deliberate — clean and injected traces are then rendered by one identical rule —
but it means the saved token ids from 01_generate.py do not reproduce this
forward pass.

Per step:
  boundary hidden state (final token of the step TEXT, before </step>)  [PRIMARY]
  mean-pooled hidden state over the step's tokens                       [ABLATION]
  mean predictive entropy over the step's tokens
  token span

Representation depths are taken at fractions of model depth. Index 0 is the
EMBEDDING output, not a transformer layer — so this is five representation
depths, four of which are layers.

Usage
-----
    python 03_extract.py --in data/injected_traces.jsonl --out data/acts_injected.npz

    # for probe_recurrence.py — filter THEN limit, or you get ~8 clean traces
    python 03_extract.py --in data/injected_traces.jsonl --out data/acts_probe.npz \
                         --condition clean --limit 40
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from modeling import DEFAULT_MODEL, load
from telemetry import render_steps

LAYER_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
ENTROPY_CHUNK = 128     # positions per chunk; bounds the fp32 logits allocation


def token_spans(offsets, char_spans, shift):
    """Character spans -> inclusive token index ranges."""
    out = []
    for (c0, c1) in char_spans:
        c0, c1 = c0 + shift, c1 + shift
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0]
        out.append((idx[0], idx[-1]) if idx else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--condition", choices=["all", "clean", "exact_copy", "verification"],
                    default="all",
                    help="filter BEFORE --limit. The injected file interleaves conditions, "
                         "so a bare --limit 40 yields ~8 clean traces, not 40.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N traces (0 = all)")
    args = ap.parse_args()

    import torch

    tok, model = load(args.model, args.device, args.dtype)
    assert tok.is_fast, "need a fast tokenizer for offset mapping"

    n_layers = model.config.num_hidden_layers
    depths = sorted({int(round(f * n_layers)) for f in LAYER_FRACTIONS})
    print(f"[model] {n_layers} transformer layers; hidden_states indices {depths} "
          f"(index 0 = embedding output, not a layer)")

    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text().splitlines() if l.strip()]
    if args.condition != "all":
        rows = [r for r in rows if r["condition"] == args.condition]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {len(rows)} traces (condition={args.condition}, limit={args.limit or 'none'})")

    H = {L: [] for L in depths}
    Hm = {L: [] for L in depths}
    meta_trace, meta_step, meta_ent = [], [], []
    skipped = 0

    for n, r in enumerate(rows):
        response, char_spans = render_steps(r["steps"])
        full = r["prompt"] + response
        shift = len(r["prompt"])

        enc = tok(full, return_tensors="pt", return_offsets_mapping=True,
                  truncation=True, max_length=args.max_len)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}

        tspans = token_spans(offsets, char_spans, shift)
        if any(s is None for s in tspans):
            skipped += 1
            continue

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states                 # (n_layers+1) x (1, T, d)
        logits = out.logits[0]                 # (T, V) — keep MODEL dtype

        # Predictive entropy. The fp32 cast happens INSIDE the chunk loop.
        # Casting the whole (T, 151k) logits tensor to fp32 up front costs
        # several GB on a long sequence and makes the chunking pointless.
        T = logits.shape[0]
        ent = torch.empty(T, dtype=torch.float32, device=logits.device)
        for a in range(0, T, ENTROPY_CHUNK):
            b = min(a + ENTROPY_CHUNK, T)
            lp = torch.log_softmax(logits[a:b].float(), dim=-1)
            ent[a:b] = -(lp.exp() * lp).sum(-1)
        ent = ent.cpu().numpy()

        for si, (t0, t1) in enumerate(tspans):
            for L in depths:
                h = hs[L][0]
                H[L].append(h[t1].float().cpu().numpy().astype(np.float16))
                Hm[L].append(h[t0 : t1 + 1].float().mean(0).cpu().numpy().astype(np.float16))
            lo = max(0, t0 - 1)
            meta_ent.append(float(ent[lo:t1].mean()) if t1 > lo else float(ent[t1]))
            meta_trace.append(r["trace_id"])
            meta_step.append(si)

        del out, hs, logits
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(rows)} traces", flush=True)

    payload = {f"H_{L}": np.stack(H[L]) for L in depths}
    payload.update({f"Hmean_{L}": np.stack(Hm[L]) for L in depths})
    payload["trace_id"] = np.array(meta_trace)
    payload["step_idx"] = np.array(meta_step)
    payload["entropy"] = np.array(meta_ent, dtype=np.float32)
    payload["layer_ids"] = np.array(depths)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **payload)

    print(f"\n[done] {outp}")
    print(f"  traces kept   : {len(rows) - skipped}/{len(rows)}")
    print(f"  traces skipped: {skipped}   <- report this (offset mapping / truncation)")
    print(f"  step vectors  : {len(meta_step)}")


if __name__ == "__main__":
    main()
