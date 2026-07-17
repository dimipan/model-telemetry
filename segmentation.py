"""
segmentation.py — turn a raw model response into reasoning steps.

Single source of truth, imported by 01_generate.py, resegment.py and the tests.

WHY THIS EXISTS (empirical, from the n=20 sanity run)
-----------------------------------------------------
DeepSeek-R1-Distill models ignore <step> formatting instructions almost entirely
(0/20 emitted the tags) and instead produce their trained house style:

    <think>
    ...free-form reasoning, one logical move per sentence...
    </think>

    **Step 1:** ... polished restatement ...
    **Final Answer:** \\boxed{N}

Two consequences drove the design:

1. Whole-response paragraph splitting produced FRANKENSTEIN traces — the <think>
   summary paragraphs followed by the post-</think> restatement of the SAME
   reasoning. That duplicate content would poison a RECURRENCE study specifically:
   r_t would fire on the model restating its own answer, not on a stall.

2. The genuine reasoning trajectory — including the "Wait, let me check..." moves
   that ARE stalling — lives inside <think>. That is what we want to measure.

So: segment the <think> block, sentence-split. The polished post-answer summary
is discarded. ~81% of GSM8K reference values remain reachable inside <think>
(measured on the sanity run: 67% appear in both regions, 14% in <think> only).

DeepSeek's chat template auto-opens <think>, so a raw response may BEGIN mid-block
with no opening tag. If there is no </think> at all (a runaway generation that hit
the token cap still reasoning), the whole output is treated as the reasoning block;
such traces are typically incorrect and excluded by the correct-only injection
default anyway.
"""

from __future__ import annotations

import re

# Legacy: honoured if a model ever does emit <step> tags (none did on the sanity run).
STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)

MIN_SENT_CHARS = 4          # drop stray fragments like a lone "So."


def reasoning_block(raw: str) -> str:
    """The <think> reasoning span, stripped of tags and the post-answer summary."""
    if "</think>" in raw:
        block = raw.split("</think>", 1)[0]
    else:
        block = raw                       # runaway: whole output is reasoning
    return block.replace("<think>", "").strip()


def _sentence_split(text: str) -> list[str]:
    # Protect decimals and inline LaTeX-ish dollar spans from the splitter.
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\\$*\"'])", text)
    return [s.strip() for s in sents if len(s.strip()) >= MIN_SENT_CHARS]


def segment(raw: str) -> tuple[list[str], str]:
    """Return (steps, method).

    method is one of {"step_tags", "think_sentences", "raw_sentences"} and is
    recorded per trace so the segmentation regime is auditable, not hidden behind
    a single boolean.
    """
    tagged = [s.strip() for s in STEP_RE.findall(raw) if s.strip()]
    if len(tagged) >= 2:
        return tagged, "step_tags"

    block = reasoning_block(raw)
    sents = _sentence_split(block)
    if len(sents) >= 2:
        method = "think_sentences" if "</think>" in raw else "raw_sentences"
        return sents, method

    # last resort: whole response, sentence-split
    return _sentence_split(raw) or [raw.strip()], "raw_sentences"
