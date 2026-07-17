"""
telemetry.py — reference-path progress, latent recurrence, and the three arms.

Pure numpy. No torch, no model. This is the only module that defines the
signals under test.

    RecurrenceOnly_t      = r_t
    GainOnly_t            = -g_t
    GatedRecurrence_t     = r_t * 1[n_new_t == 0]         <- proposed signal
    GatedRecurrenceSoft_t = r_t * exp(-lambda * n_new_t)  <- robustness check

All four are PRESPECIFIED to point the same way: HIGHER = MORE STALL-LIKE.
Their orientation is never fitted. Fitting it would dissolve the prespecified
ablation the README claims. 04_evaluate.py enforces this; token entropy and
step length are the only signals whose sign is chosen on train, and they are
declared as fitted nuisance baselines.

The gain-gated recurrence measure is adapted from Acquisition-State Telemetry,
where it was developed to monitor external evidence-acquisition processes.
Here it is applied to a model-internal reasoning substrate.

TERMINOLOGY. upsilon measures REFERENCE-INTERMEDIATE COVERAGE, not unrestricted
"knowledge gain". GSM8K's inline annotations define one valid solution path.
See README limitations.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants. Not tuned per-condition; that is the whole point.
# ---------------------------------------------------------------------------
RECURRENCE_WINDOW = 8
WARMUP_STEPS = 1
NUM_TOL_REL = 1e-6
SOFT_GATE_LAMBDA = 1.0

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANNOT_RE = re.compile(r"<<([^<>]+?)>>")

# A step's CLAIMED RESULT: the value to the right of an explicit result marker.
# The marker list is deliberately broad — "8 times 3 is 24" is one of the most
# common phrasings a model produces, and dropping bare "is" would throw away real
# results and crater reference coverage for the wrong reason. Breadth here is what
# lets the FALLBACK be conservative (see claimed_result).
_RESULT_RE = re.compile(
    r"(?:"
    r"="                                             # 48/2 = 24
    r"|equals?|equalling|equaling"                   # equals 24
    r"|giv(?:e|es|ing)"                              # giving 24   <- -ing forms matter
    r"|yield(?:s|ing|ed)?"                           # yields 24
    r"|produc(?:e|es|ing|ed)"                        # produces 24
    r"|return(?:s|ing|ed)?"
    r"|leav(?:e|es|ing)|left"                        # leaves 24 / that left 24
    r"|com(?:e|es|ing)\s+(?:out\s+)?to"             # comes out to 24
    r"|amount(?:s|ing)?\s+to"
    r"|result(?:s|ing)?\s+in"
    r"|(?:result|total|answer|sum|product|difference|quotient)\s+(?:is|=)"
    r"|therefore[,:\s]+|thus[,:\s]+|so[,:\s]+"
    r"|is"                                           # "8 times 3 is 24"
    r")\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
# NOTE: "are" is deliberately ABSENT. "there are 8 boxes and 3 items each" would
# otherwise claim a result of 8 — a GIVEN, not a computed value. That is precisely
# the false-progress path this rule exists to close.


# ---------------------------------------------------------------------------
# 1. Reference intermediates
# ---------------------------------------------------------------------------
def parse_reference_intermediates(gold_solution: str, dedupe: bool = True) -> list[dict]:
    """Reference calculation path from GSM8K's <<expr=value>> annotations.

    dedupe=True (DEFAULT, and the v0.1 policy): M is the set of DISTINCT
    reference-result VALUES, in first-occurrence order.

    Why: numeric matching cannot tell two identical result values apart. If the
    reference path produces 24 twice, one '24' in the trace would resolve both
    slots at once and coverage would silently overcount. Distinct-value coverage
    is the only policy this scorer can honour. Ordered-slot coverage would be
    more faithful but assumes the model walks the gold path in gold order.
    """
    parsed: list[dict] = []
    for raw in _ANNOT_RE.findall(gold_solution):
        if "=" not in raw:
            continue
        expr, _, val = raw.rpartition("=")
        v = _to_float(val.strip())
        if v is None:
            continue
        parsed.append({"expr": expr.strip(), "value": v, "raw": raw.strip()})

    if not dedupe:
        return parsed

    seen, unique = set(), []
    for item in parsed:
        key = round(item["value"], 10)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def numbers_in(text: str) -> list[float]:
    vals = []
    for m in _NUM_RE.findall(text or ""):
        v = _to_float(m)
        if v is not None:
            vals.append(v)
    return vals


def claimed_result(text: str) -> float | None:
    """The value a step CLAIMS to have produced, or None if the step claims none.

    Rule:
      1. RHS of the LAST explicit result marker in the step. (Broad marker list.)
      2. Else, if the step contains EXACTLY ONE number, that number.
      3. Else None — multiple numbers with no result marker are AMBIGUOUS, and
         guessing is how false progress gets manufactured.

    Rule 3 is the point. An earlier version fell back to "the last number
    anywhere", which meant a setup step like "there are 8 boxes and 3 items each"
    claimed a result of 3 — so a reference intermediate whose value happened to
    be 3 was marked resolved on a GIVEN that was never computed. The operand path
    was narrowed by the marker rule but not closed; rule 3 closes it.

    This is deliberately conservative and will LOWER reference coverage. That is
    the correct trade: undercounting progress opens the gate too often (a
    false-positive stall), whereas overcounting progress closes the gate on real
    stalls and silently destroys the thing being measured.
    """
    m = _RESULT_RE.findall(text or "")
    if m:
        return _to_float(m[-1])
    nums = numbers_in(text)
    if len(nums) == 1:
        return nums[0]
    return None


def _matches(target: float, candidate: float | None) -> bool:
    if candidate is None:
        return False
    tol = max(NUM_TOL_REL * abs(target), NUM_TOL_REL)
    return abs(candidate - target) <= tol


_EXPR_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _operands_of(expr: str) -> list[float]:
    """Numeric operands in a reference expression, e.g. '3.5*8' -> [3.5, 8].

    Magnitudes only. GSM8K expressions like '52-8-28' use '-' as the subtraction
    OPERATOR; attaching it to the operand ('-8') would then fail to match the '8'
    that actually surfaces in the step text ("subtract 8 and 28 from 52").
    """
    out = []
    for m in _EXPR_NUM_RE.findall(expr or ""):
        v = _to_float(m)
        if v is not None:
            out.append(v)
    return out


def _operands_present(operands: Sequence[float], step_nums: Sequence[float]) -> bool:
    """True if every operand of the reference expression appears in the step.

    This is the OPERAND-MATCH rule (segmentation Option C). DeepSeek-R1 often
    DESCRIBES a calculation in its <think> block ("I multiply 8 by 3.5 to find
    Micah's distance") and only writes the RESULT (28) in the post-answer summary
    we discard. Result-only matching therefore undercounts genuine progress:
    measured coverage was 0.32 think-only vs 0.77 with the summary. Crediting an
    intermediate when both its operands are present in a reasoning step recovers
    coverage to ~0.67 WITHOUT reintroducing the summary (and its duplication).

    Faithfulness argument: a step that states both operands of a reference
    operation has, in substance, performed that operation. The reference
    intermediate is about whether the computation happened, not whether a
    particular result string was emitted.

    Requires >= 2 operands so a lone given ("Amber ran 8 miles") cannot resolve
    anything on its own.
    """
    if len(operands) < 2:
        return False
    for o in operands:
        tol = max(NUM_TOL_REL * abs(o), NUM_TOL_REL)
        if not any(abs(o - n) <= tol for n in step_nums):
            return False
    return True


# ---------------------------------------------------------------------------
# 2. Reference-path progress
# ---------------------------------------------------------------------------
def reference_progress(
    step_texts: Sequence[str],
    intermediates: Sequence[dict],
    match: str = "operand",
    result_only_steps: Sequence[bool] | None = None,
) -> dict:
    """Monotone coverage of the reference calculation path.

    match:
      "result"   — resolve only when a step's CLAIMED RESULT equals the value.
                   Strict; undercounts when the model defers writing results.
      "operand"  — ALSO resolve when both operands of the reference expression
                   appear in a step (Option C, the default). See _operands_present.

    result_only_steps : optional per-step bool mask. Where True, that step is
      scored RESULT-ONLY even under match="operand". This exists to close a real
      leak found at n=300:

        A verification template restating "8/2 = 4" contains the numbers 8, 2, 4.
        Under operand-match the incidental pair {8, 4} satisfies the operands of an
        UNRELATED reference expression "8+4=12" — so the injected step resolved 12
        by number cross-talk, breaking the no-gain guarantee. Injected steps
        restate a value; they must be credited ONLY on the value they claim, never
        on incidental operand co-occurrence. 02_inject marks injected steps here.

    Monotone: once resolved, stays resolved.
    """
    T = len(step_texts)
    M = len(intermediates)
    if M == 0:
        return {
            "upsilon": np.zeros(T),
            "gain": np.zeros(T),
            "n_new": np.zeros(T, dtype=np.int64),
        }

    use_operand = match == "operand"
    ops = [_operands_of(it["expr"]) for it in intermediates] if use_operand else None
    ro_mask = list(result_only_steps) if result_only_steps is not None else [False] * T

    resolved = np.zeros(M, dtype=bool)
    upsilon = np.zeros(T, dtype=np.float64)
    n_new = np.zeros(T, dtype=np.int64)

    for t, text in enumerate(step_texts):
        res = claimed_result(text)
        step_operand = use_operand and not ro_mask[t]     # injected steps: result-only
        step_nums = numbers_in(text) if step_operand else None
        newly = 0
        for i, item in enumerate(intermediates):
            if resolved[i]:
                continue
            hit = _matches(item["value"], res)
            if not hit and step_operand:
                hit = _operands_present(ops[i], step_nums)
            if hit:
                resolved[i] = True
                newly += 1
        n_new[t] = newly
        upsilon[t] = resolved.sum() / M

    return {
        "upsilon": upsilon,
        "gain": np.diff(upsilon, prepend=0.0),
        "n_new": n_new,
    }


# ---------------------------------------------------------------------------
# 3. Latent recurrence
# ---------------------------------------------------------------------------
def latent_recurrence(H, window: int = RECURRENCE_WINDOW, mu=None) -> np.ndarray:
    """r_t = max_{j in [t-window, t-1]} cosine(h_t - mu, h_j - mu).

    `mu` is an optional per-layer mean, FITTED ON TRAIN CLEAN STEPS ONLY.

    Why centering is offered: transformer residual streams are strongly
    anisotropic. A few rogue dimensions dominate the norm, so raw cosine between
    ANY two hidden states can sit at 0.98-0.999. In that regime r_t varies but
    carries almost nothing, and every downstream AUPRC is noise wearing a
    result's clothes. Mean-centering is the standard mitigation.

    Run probe_recurrence.py BEFORE the full study and let the observed spread
    decide. Log the decision in NOTES.md.
    """
    H = np.asarray(H, dtype=np.float64)
    if mu is not None:
        H = H - np.asarray(mu, dtype=np.float64)[None, :]

    T = H.shape[0]
    r = np.zeros(T, dtype=np.float64)
    if T < 2:
        return r

    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    Hn = H / norms

    for t in range(1, T):
        lo = max(0, t - window)
        r[t] = float((Hn[lo:t] @ Hn[t]).max())
    return r


# ---------------------------------------------------------------------------
# 4. The arms. Orientation is FIXED, never fitted.
# ---------------------------------------------------------------------------
def recurrence_only(r, gain, n_new):
    return np.asarray(r, dtype=np.float64)


def gain_only(r, gain, n_new):
    """Absence of reference-path progress. Negated so higher = more stall-like."""
    return -np.asarray(gain, dtype=np.float64)


def gated_recurrence(r, gain, n_new):
    """PRIMARY ARM. Recurrence, gated by absence of reference-path progress.

    Binary gate. Reference intermediates change state discretely, and with
    |M| ~ 3-6 a multiplicative (1 - g_t) gate sits in [0.83, 1.0] — it barely
    gates anything and collapses to RecurrenceOnly. Binary is both the natural
    and the explainable choice.
    """
    gate = (np.asarray(n_new, dtype=np.int64) == 0).astype(np.float64)
    return np.asarray(r, dtype=np.float64) * gate


def gated_recurrence_soft(r, gain, n_new):
    """SECONDARY. r_t * exp(-lambda * n_new_t). Robustness check on the gate."""
    n = np.asarray(n_new, dtype=np.float64)
    return np.asarray(r, dtype=np.float64) * np.exp(-SOFT_GATE_LAMBDA * n)


ARMS = {
    "recurrence_only": recurrence_only,
    "gain_only": gain_only,
    "gated_recurrence": gated_recurrence,
    "gated_recurrence_soft": gated_recurrence_soft,
}

# PRESPECIFIED sign (+1 = higher is more stall-like). Never fitted.
FIXED_ORIENTATION = {
    "recurrence_only": 1.0,
    "gain_only": 1.0,
    "gated_recurrence": 1.0,
    "gated_recurrence_soft": 1.0,
    "lexical_ngram": 1.0,
}

# Sign chosen on TRAIN. Declared in the results table as fitted nuisance baselines.
FITTED_ORIENTATION = ("token_entropy", "step_length", "history_size")


# ---------------------------------------------------------------------------
# 5. Text baselines
# ---------------------------------------------------------------------------
def _word_ngrams(text: str, n: int = 3) -> set:
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def lexical_recurrence(step_texts, n: int = 3, window: int = RECURRENCE_WINDOW) -> np.ndarray:
    """Max 3-gram containment against the previous `window` steps.

    Decides whether the latent signal does anything beyond copy detection.
    Expected to win on exact-copy stalls.
    """
    grams = [_word_ngrams(s, n) for s in step_texts]
    out = np.zeros(len(step_texts), dtype=np.float64)
    for t in range(1, len(step_texts)):
        gt = grams[t]
        if not gt:
            continue
        best = 0.0
        for j in range(max(0, t - window), t):
            if grams[j]:
                best = max(best, len(gt & grams[j]) / len(gt))
        out[t] = best
    return out


def step_length(step_texts) -> np.ndarray:
    """Trivial-cue check. If competitive, the templates are a length artefact and
    the result is worthless. 04_evaluate.py says so out loud."""
    return np.array([len(re.findall(r"\w+", s or "")) for s in step_texts], dtype=np.float64)


def history_size(T: int, window: int = RECURRENCE_WINDOW) -> np.ndarray:
    """POSITION CONFOUND GUARD. min(t, W) — the number of candidates r_t maximises over.

    r_t = max over [t-W, t-1], so the number of comparisons GROWS with t until it
    saturates at W. Meanwhile verification stalls are only inserted after at least
    two original steps, so positives are structurally LATER than many negatives.
    Recurrence could therefore rise for a purely combinatorial reason — more draws,
    higher maximum — with no representational content at all.

    step_length cannot detect this. This baseline is matched to the estimator's own
    candidate count, so if it is competitive with the gated arm, the "signal" is
    position, not geometry.
    """
    return np.array([min(t, window) for t in range(T)], dtype=np.float64)


# Depth 0 of hidden_states is the EMBEDDING OUTPUT, not a transformer layer.
# It is kept as a NEGATIVE CONTROL and excluded from headline layer selection: a
# signal that already exists at the embedding reflects token identity, punctuation
# or template style, not model computation — which is the whole point of the study.
EMBEDDING_DEPTH = 0


# ---------------------------------------------------------------------------
# 6. Canonical rendering (shared by 01_generate and 03_extract)
# ---------------------------------------------------------------------------
STEP_OPEN, STEP_CLOSE = "<step>", "</step>\n"


def render_steps(steps: Sequence[str]) -> tuple[str, list]:
    """Rebuild a CANONICAL response and record each step's character span.

    Extraction runs over this canonicalised trace, NOT over the raw generated
    response. Clean and injected traces are therefore rendered by one identical
    rule. Untagged preamble and the <answer> block are dropped. Both forms are
    saved (raw_response / canonical_response).

    The span ends at the last character of the step TEXT, before </step>, so the
    boundary token is the step's own final token — not the closing tag.
    """
    parts, spans, cursor = [], [], 0
    for s in steps:
        chunk = STEP_OPEN + s + STEP_CLOSE
        spans.append((cursor + len(STEP_OPEN), cursor + len(STEP_OPEN) + len(s)))
        parts.append(chunk)
        cursor += len(chunk)
    return "".join(parts), spans


def eval_mask(T: int, warmup: int = WARMUP_STEPS) -> np.ndarray:
    """Steps eligible for SCORING. Warmup steps still contribute history."""
    m = np.ones(T, dtype=bool)
    m[:warmup] = False
    return m