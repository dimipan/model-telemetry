# Gain-Gated Telemetry for Reasoning Stalls

You take an off-the-shelf reasoning model, watch it solve a math problem step by step, and
at each step measure two simple things and multiply them together. That is the whole method.

The two things:

- **Progress** — did this step resolve anything new? GSM8K ships the reference solution's
  arithmetic baked in (`<<48/2=24>>`), so you have a checklist of the intermediate results a
  correct solution computes. As the model reasons, you tick boxes off. Progress is: did this
  step tick a new box?
- **Recurrence** — is the model's internal state repeating? You grab the hidden-state vector
  at each step and measure how similar it is to the last few. High similarity means the
  representation is circling.

The method is: **count recurrence only when nothing got resolved.** `recurrence · [progress = 0]`.
Multiply "am I looping" by "and I resolved nothing." No training, no classifier, no learned
direction in activation space. Two measurements and a gate.

## Why the gate

Recurrence on its own is a bad stall detector. Reasoning models revisit similar states all
the time for good reasons — that is not stalling, that is thinking. Repetition only matters
when nothing is being resolved. So progress is the gate, and recurrence is what the gate
lets through. That combination is the thing this repo tests.

Worked example — the model solves a problem whose reference intermediates are `{24, 18, 42}`:

```
step  model writes                       resolved?   progress   recurrence   gated
1     "48 / 2 = 24"                       24 ✓        1          0.41         0     progress, ignore
2     "24 − 6 = 18"                       18 ✓        1          0.55         0     progress, ignore
3     "let me double-check: 48/2 = 24"    —           0          0.93         0.93  ← stall
4     "yes, that's 24, consistent"        —           0          0.91         0.91  ← stall
5     "18 + 24 = 42, answer is 42"        42 ✓        1          0.60         0     progress, ignore
```

Steps 1, 2, 5 have high recurrence too, but progress happened, so the gate zeroes them.
Only steps 3–4 — repeating while resolving nothing — light up.

## Where the idea comes from

The gate is borrowed from Acquisition-State Telemetry (AST), which watches an external
process acquire information and reports whether it is still making progress. Here the same
idea is turned inward, onto the model's own reasoning.

| | AST | this repo |
|---|---|---|
| watches | an external process (dialogue, robot, estimator) | the model's own reasoning |
| evidence | what a task binding reports, black-box | the model's hidden states, white-box |
| progress | task categories resolved | reference-path steps covered |
| recurrence | evidence returning in representation space | hidden states repeating step to step |

Same rule — recurrence counts as a stall only when nothing is being resolved — moved from
the outside of a system to the inside.

## What gets tested

You cannot trust the gate just because it sounds right, so you inject known stalls and check
whether it catches them better than recurrence-alone or progress-alone.

- **Exact-copy.** Duplicate steps verbatim. A sanity check on the pipeline; simple word
  overlap is expected to win here. Not the real test.
- **Verification.** Insert steps that restate an already-solved value ("Checking once more:
  48/2 = 24."). Repetition with zero progress — enforced when the trace is written, so an
  inserted step can never accidentally resolve something. This is the real test.

The claim holds only if the gated signal beats **both** of its parts, and only if it holds
across problem-level splits. If it does not, the evaluation says so plainly.

## Results

*Placeholders until `04_evaluate.py` runs.* Reported as mean ± sd over 10 problem-level
splits, plus a count of how many splits the gate won. The verdict is the per-split win
count, not the average. The script prints which of these is true:

- Gated beats both parts in most splits → the idea holds.
- Progress alone wins → recurrence adds nothing.
- Recurrence alone wins → the gate is throwing away signal.
- Word-overlap or step-length wins → the result is a surface artefact, not a stall signal.
- The embedding-only control matches the chosen layer → the signal is not internal.

Each is a real outcome. Whichever is true is what gets written up.

## Layout

```text
telemetry.py         the two measurements + the gate (numpy only)
modeling.py          model load, dtype resolution

01_generate.py       run the model on GSM8K, save reasoning traces (text only)
02_inject.py         add exact-copy and verification stalls, with exact labels
03_extract.py        re-run the traces to pull hidden states at each step
probe_recurrence.py  check the cosine signal is usable before trusting results
04_evaluate.py       the whole experiment, one table

requirements.txt     dependencies
NOTES.md             decision log: what was tried, what broke
tests/smoke_test.py  plumbing check, no GPU
```

Two details that keep the result honest: extraction is a separate pass that runs *after*
injection, so edited traces never reuse an unedited trace's hidden states; and splits are by
problem, so no problem appears in both the fitting and test halves.

## Run it

```bash
pip install -r requirements.txt
python tests/smoke_test.py                 # no GPU, no downloads

python 01_generate.py --n 300 --seed 0
python 02_inject.py

python 03_extract.py --in data/injected_traces.jsonl --out data/acts_probe.npz \
                     --condition clean --limit 40
python probe_recurrence.py                 # decides raw vs centered cosine; log it in NOTES.md

python 03_extract.py --in data/injected_traces.jsonl --out data/acts_injected.npz
python 04_evaluate.py --out results/raw.csv
```

The probe exists because hidden states can be so similar to each other that cosine sits at
0.98–0.999 for everything, in which case recurrence carries no signal. That risk is invisible
in the results table, so it is checked first.

## Limitations

- The GSM8K checklist is one valid solution path, not the only correct reasoning. A model can
  be right while taking a different route, which scores as no progress and can look like a
  stall. `02_inject.py` prints the coverage distribution; if coverage is low everywhere, that
  is itself the finding.
- The progress scorer is deliberately conservative — it under-counts rather than over-counts,
  which is the safe direction for a stall detector.
- Injected stalls may not match how real overthinking looks. Natural stalls are out of scope
  for now.
- This is association, not causation, on one small model. Results may not transfer.

**Next (not started):** natural stalls without injection, a causal test, a second model size,
other tasks.

MIT.
