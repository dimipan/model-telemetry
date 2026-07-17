# Gain-Gated Telemetry for Reasoning Stalls

**Latent recurrence does not necessarily indicate stalled reasoning.** This repository
tests whether recurrence becomes a more *specific* stall signal when conditioned on the
absence of reference-path progress.

The experiment compares **recurrence alone**, **progress alone**, and **gain-gated
recurrence** on exact-copy and templated no-gain verification stalls in GSM8K reasoning
traces from `DeepSeek-R1-Distill-Qwen-1.5B`.

That is the entire claim. Nothing here identifies deception, hallucination, or
overthinking in general, and nothing here is causal.

> The gain-gated recurrence measure is adapted from Acquisition-State Telemetry, where it
> was developed to monitor external evidence-acquisition processes. Here it is applied to
> a model-internal reasoning substrate.

---

## The signal

For each reasoning step *t*, with `h_t` the boundary-token hidden state:

| | |
|---|---|
| reference coverage | `υ_t = (1/\|M\|) · Σ_i 1[reference intermediate m_i resolved by step t]` |
| reference-path progress | `g_t = υ_t − υ_{t−1}` |
| latent recurrence | `r_t = max_{j ∈ [t−W, t−1]} cos(h_t, h_j)`, W = 8 |

Three arms, identical preprocessing, differing only in the ablation:

```
RecurrenceOnly_t   = r_t
GainOnly_t         = −g_t
GatedRecurrence_t  = r_t · 1[g_t = 0]      ← the proposed signal
```

All three are **prespecified** to point the same way — higher means more stall-like — and
their orientation is never fitted. The claim survives only if the gated combination
separates stalls better than **both** of its constituents, **and** that advantage
persists across problem-level splits. If it does not, this README says so.

`M` is the set of **distinct reference-result values** parsed from GSM8K's inline
`<<48/2=24>>` annotations. An intermediate is resolved by the first step whose *claimed
result* — the right-hand side of an equality or result verb — equals its value. Operands
do not count. This is **reference-intermediate coverage**, not unrestricted "knowledge
gain"; see [Limitations](#limitations).

---

## Two stall conditions, reported separately

**A. Exact-copy.** Duplicate steps *[i, i+k)* verbatim. Pipeline validation and an
upper-bound sanity check. Lexical n-grams are *expected* to win here. **Not the
experiment.**

**B. Templated no-gain verification.** Insert *k* deterministic steps restating an
already-resolved reference intermediate, drawn from eight templates spanning short to
long:

> *Checking once more: 48/2 = 24. The earlier calculation is consistent.*

Semantic recurrence, low lexical identity, and zero new reference-path progress — the
latter **checked on the write path**: every verification trace is re-scored end to end
before it is written, and the run aborts if an inserted step ever resolves an
intermediate. **This is the experiment.**

The easy condition is never averaged into the hard one.

---

## Results

n = 300 GSM8K problems -> 857 traces (199 clean / 398 exact-copy / 260 verification),
6,086 scored steps. Mean +/- sd over 10 problem-level splits. Primary analysis is
**centered** (chosen by the anisotropy probe before results were seen; see NOTES); raw agrees.
Layer 21 selected on train in all 10 splits. AUPRC is primary; stall base rate 0.256.

| Signal | Sign | Exact-copy AUPRC | **Verification AUPRC** | Verification AUROC | FPR |
|---|---|---:|---:|---:|---:|
| Lexical 3-gram overlap | fixed | 1.000 ± 0.000 | 0.300 ± 0.013 | 0.368 ± 0.015 | 0.046 |
| Token entropy †  | *fitted* | 0.348 ± 0.010 | **0.893 ± 0.019** | 0.967 ± 0.005 | 0.054 |
| Step length *(length guard)* | *fitted* | 0.257 ± 0.010 | 0.390 ± 0.061 | 0.516 ± 0.077 | 0.045 |
| History size *(position guard)* | *fitted* | 0.288 ± 0.009 | 0.274 ± 0.010 | 0.573 ± 0.015 | 0.034 |
| Recurrence only | fixed | 0.460 ± 0.034 | 0.409 ± 0.040 | 0.656 ± 0.015 | 0.066 |
| Gain only | fixed | 0.358 ± 0.009 | 0.358 ± 0.017 | 0.692 ± 0.013 | 0.000 |
| **Gain-gated recurrence** | fixed | **0.527 ± 0.038** | **0.483 ± 0.047** | 0.772 ± 0.017 | 0.065 |
| Gain-gated recurrence (soft) | fixed | 0.525 ± 0.038 | 0.475 ± 0.046 | 0.754 ± 0.017 | 0.065 |

Embedding-output control (primary arm at depth 0, excluded from selection): 0.400 vs 0.483 at
the selected transformer depth — the signal **improves inside the transformer** (+0.083), so it
is not token identity at the input.

### Finding 1 — the claim holds

**Gain-gated recurrence beats both of its constituents on verification stalls in 10/10
problem-level splits** (+0.074 ± 0.011 AUPRC over the best constituent; raw analysis +0.103).
It also beats them on exact-copy (0.527 vs 0.460 / 0.358). Gating a latent recurrence signal on
the absence of reference-path progress makes it a more specific stall marker than either
recurrence or progress alone. The three confound guards — step length, history size, lexical
overlap — all score below the gated arm, and the position of positives (mean step 4.96) and
negatives (4.56) is nearly matched, so the effect is not a position or length artefact.

This is a *stability* result across overlapping splits, not a formal confidence interval; a
grouped bootstrap is left to v0.2.

### Finding 2 — the entropy baseline wins, but only by detecting injected text †

**Token entropy scores 0.893 on verification — higher than any latent signal.** This is
reported, not hidden, and it is a **synthetic-injection artefact**, identified by the
exact-copy control:

- On **verification** stalls (text we authored) entropy scores 0.893.
- On **exact-copy** stalls (the model's *own* prior steps, duplicated) entropy scores **0.348** —
  near chance.

The difference is not length (verification inserts average 16.0 words, identical to genuine
steps; step length scores only 0.390) and not template uniformity (84% of inserts are distinct;
their entropy variance is *larger* than genuine steps, not smaller). It is **distributional
foreignness**: the injected prose is fluent but is not this 1.5B model's terse reasoning voice,
so it carries high per-token perplexity. Entropy is detecting *who wrote the text*, not whether
the model is stalling. When the inserted stall is the model's own text (exact-copy), the tell
vanishes.

Gain-gated recurrence, by contrast, scores consistently across both conditions (0.527 / 0.483)
because it detects repetition structure, not provenance. **It is the strongest signal that is
robust to how the stall was produced.**

### What this means, and what comes next

The honest one-sentence summary: *gain-gating reliably improves latent recurrence and is the
strongest injection-robust stall signal; a predictive-entropy baseline scores higher but only
by detecting that the injected text is out-of-distribution, as the exact-copy control shows.*

The entropy result is the argument for the **natural-stall** experiment (v0.2): the only way to
remove the distributional tell is to study stalls the model generates itself, with nothing
injected. The synthetic benchmark here motivates that follow-up rather than substituting for it.

## Run it

```bash
pip install -r requirements.txt

python tests/smoke_test.py                 # no GPU, no downloads — plumbing + invariants

python 01_generate.py --n 300 --seed 0     # resumable; per-problem seeds
# if you change the segmenter later, re-derive steps without regenerating:
#   python resegment.py --in data/original_traces.jsonl --out data/original_traces.jsonl
python 02_inject.py                        # correct-only is the DEFAULT protocol

# choose the PRIMARY representation before looking at any results
python 03_extract.py --in data/injected_traces.jsonl --out data/acts_probe.npz \
                     --condition clean --limit 40
python probe_recurrence.py

python 03_extract.py --in data/injected_traces.jsonl --out data/acts_injected.npz
python 04_evaluate.py          --out results/raw.csv
python 04_evaluate.py --center --out results/centered.csv
```

**Why the probe exists.** Transformer residual streams are strongly anisotropic — a few
rogue dimensions dominate the norm, so raw cosine between *any* two hidden states can sit
at 0.98–0.999. In that regime `r_t` varies but may carry almost nothing, and the AUPRCs
below would be noise wearing a result's clothes. That risk is **invisible in the results
table**, so it is checked directly.

The probe chooses which analysis is **primary**; it does not decide whether the study
exists. A compressed band is not proof of absent signal, and a wide band is not proof of
present signal. Both analyses run on the same activations — record the primary choice in
`NOTES.md` *before* reading either table.

Model dtype is resolved from the device (bf16 where supported, else fp16, else fp32), not
hard-coded. Runtime and memory are **not** quoted here until the first full run has
happened; see `NOTES.md`.

---

## Protocol (the parts that would otherwise invalidate the result)

**Two-pass extraction.** `01_generate.py` produces text only. Activations come from a
separate teacher-forced forward pass (`output_hidden_states=True`, `use_cache=False`) in
`03_extract.py`. Injection happens *between* them, so edited traces get their own forward
pass and never inherit an unedited trace's activations.

**Teacher-forced extraction over canonicalised step traces.** Extraction runs over the
segmented steps re-wrapped in fresh `<step>` tags — not the raw generated string. Untagged
preamble and the `<answer>` block are dropped. This is deliberate: clean and injected
traces are then rendered by one identical rule. It does mean the token ids saved by
`01_generate.py` do not reproduce this forward pass. Both forms are stored
(`raw_response`, `canonical_response`).

**Problem-level splits, repeated.** Every derivative of a GSM8K problem — clean trace,
both exact-copy variants, both verification variants — stays in one partition. Problems
are the independent unit; steps within a problem and variants of a problem are correlated,
so a single split with a 0.011 AUPRC difference tells you nothing. Ten splits, mean ± sd,
and the win count.

**Prespecified orientations.** The four arms and lexical overlap have fixed sign (+1).
Token entropy, step length and history size have their sign chosen on train and are
labelled *fitted nuisance baselines* in the table.

**Depth 0 is excluded from layer selection.** It is the embedding output. A signal that
already exists there is not a fact about the model's computation, and letting it win the
headline would gut the interpretation this study exists to support. It is retained and
reported as a negative control.

**Train-only fitting.** Layer selection, centering mean, thresholds, and the two fitted
signs are all fitted on the train split of each repeat.

**Representation.** Five **representation depths** at fractions of model depth — index 0 is
the *embedding output*, not a transformer layer, so four are layers and one is a control.
The boundary token (last token of the step text, before `</step>`) is primary because it is
available online and summarises the complete prefix. `--pooling mean` repeats everything
with mean-pooled step representations.

---

## Limitations

- GSM8K reference intermediates describe **one valid solution path**, not a unique correct
  reasoning state. A model may combine operations, use algebra where the reference uses
  arithmetic, or omit an annotated value it nonetheless used.
- Consequently, **correct alternative derivations can appear to make no reference-path
  progress**, which pushes them toward the stall side of the gate. `02_inject.py` prints
  the reference-coverage distribution so this is visible rather than hidden. If coverage is
  low across the board, that is itself the headline finding.
- `M` is the set of **distinct** reference-result values. Numeric matching cannot tell two
  identical result values apart, so slot-level coverage is not honestly available here.
- Steps come from the `<think>` reasoning block, not the post-answer summary, because
  whole-response splitting duplicates the reasoning and poisons a recurrence study. The model
  often plans a calculation in `<think>` and writes the result only in the discarded summary.
- To recover the progress that implies, an intermediate is resolved when EITHER its result is
  claimed OR both of its operands appear in a reasoning step (`match="operand"`, the default).
  This is a deliberately looser, documented definition; it lifted measured coverage from 0.32
  (result-only) to ~0.74 on the sanity run. The strict `match="result"` mode is retained for
  comparison. Operand-match never inflates injected verification steps, which restate
  already-resolved values (asserted).
- A step is credited with a *result* only when it carries an explicit result marker
  (`= / equals / gives / yields / produces / is / therefore / …`), or contains exactly one
  number. Multiple numbers with no marker claim **nothing**. This is deliberately
  conservative and *lowers* measured coverage: undercounting progress opens the gate too
  often (a false-positive stall), whereas overcounting closes the gate on real stalls and
  silently destroys the thing being measured.
- **Synthetic verification stalls may differ from naturally emerging overthinking.**
  Natural-stall identification is out of scope for v0.1.
- This studies **hidden-state association, not a causal mechanism.** No intervention is
  claimed.
- One small distilled reasoning model. Results may not transfer.

## Scope

**v0.1 (this repo):** 300 GSM8K problems, one model, five representation depths,
deterministic reference-progress scorer, two stall families, non-causal baselines,
repeated problem-level splits, complete negative-result reporting.

**v0.2 (not started):** natural stalls without injection; classifier-normal causal
intervention with matched-norm random-direction, sign-reversed, and adjacent-layer
controls; a second model size; cross-task transfer.

`NOTES.md` carries the decision log — what was rejected, what broke, and what ten times
the compute would buy.

## Licence

MIT.
