# Gain-Gated Telemetry for Reasoning Stalls

Acquisition-State Telemetry (AST) watches a process from the outside and reports whether it
is still resolving its task or just spinning. This repo tests whether the same idea works
*inside* a model — on its reasoning, not an external task.

The question: **when a reasoning model repeats itself internally without making progress, does
that show up in its hidden states?**

## The signal

Two measurements per reasoning step, multiplied:

- **progress** — did this step resolve a new value from the reference solution? (GSM8K ships
  its arithmetic as `<<48/2=24>>`, so there is a checklist to tick off.)
- **recurrence** — is the hidden state repeating? Cosine similarity to the last few steps.

```
stall signal = recurrence · [progress = 0]
```

Recurrence alone is useless because models repeat states for good reasons. It only means a stall
when nothing is being resolved. Progress is the gate, recurrence is what it lets through.

*(AST measures recurrence against an accumulated per-category evidence subspace. A reasoning
trace is one sequence of steps, so here recurrence is just similarity to recent steps — cosine.)*

| | AST | this repo |
|---|---|---|
| watches | an external process (dialogue, robot, estimator) | the model's own reasoning |
| evidence | what a task binding reports, black-box | the model's hidden states, white-box |
| progress | task categories resolved | reference-path steps covered |
| recurrence | evidence returning in representation space | hidden states repeating step to step |

Same rule — recurrence counts as a stall only when nothing is being resolved — moved from
the outside of a system to the inside.

## What the pipeline does, step by step

```
1  modeling.py           load an open reasoning model (DeepSeek from HuggingFace) — we need its internals
2  01_generate.py        run it on GSM8K, save the reasoning traces (text only)
3  02_inject.py          plant known stalls: insert steps that repeat an already-solved value
4  03_extract.py         re-run the edited traces, pull hidden states at every step
5  probe_recurrence.py   check the cosine signal isn't dead (hidden states can be ~identical)
6  04_evaluate.py        does the gate beat recurrence-alone, progress-alone, and cheap baselines?
```

Worked example — the model solves a problem whose reference intermediates are `{24, 18, 42}`:

```
step  model writes                       resolved?   progress   recurrence   gated
1     "48 / 2 = 24"                       24 ✓        1          0.41         0     progress, ignore
2     "24 − 6 = 18"                       18 ✓        1          0.55         0     progress, ignore
3     "let me double-check: 48/2 = 24"    —           0          0.93         0.93  ← stall
4     "yes, that's 24, consistent"        —           0          0.91         0.91  ← stall
5     "18 + 24 = 42, answer is 42"        42 ✓        1          0.60         0     progress, ignore
```

## Result

Placeholder until `04_evaluate.py` runs. The verdict is one line the script prints: does the
gate beat both its ingredients, across problem-level splits, without a cheap baseline matching
it.

## Honest scope

- The stalls are **synthetic**. This tests detection of deliberate
  repetition, not real overthinking in the wild.
- Reading model internals is a whole field with heavier machinery. This is minimal on
  purpose: two measurements and a gate, nothing trained — testing whether even that survives
  proper baselines.
- One small model, may not transfer.

**Next:** natural stalls without injection; a second model; a causal check.

## Layout

```
telemetry.py         the two measurements + the gate (numpy)
modeling.py          model load
01_generate.py       traces from GSM8K
02_inject.py         planted stalls with exact labels
03_extract.py        hidden states per step
probe_recurrence.py  signal sanity check
04_evaluate.py       the comparison, one table
tests/smoke_test.py  plumbing, no GPU
NOTES.md             what was tried, what broke
```

```bash
pip install -r requirements.txt
python tests/smoke_test.py
python 01_generate.py --n 300 --seed 0
python 02_inject.py
python 03_extract.py --in data/injected_traces.jsonl --out data/acts.npz
python probe_recurrence.py
python 04_evaluate.py --out results/raw.csv
```

MIT.
