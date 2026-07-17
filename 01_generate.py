#!/usr/bin/env python3
"""
01_generate.py — generate step-delimited GSM8K reasoning traces.

Pass A. Text only. No hidden states. Activation extraction happens later, under
teacher forcing (03_extract.py), so original and injected traces traverse
identical extraction code.

Resumable: re-running appends only the problems not already in the output file.
A 300-problem run should not be lost to one interruption.

Usage
-----
    python 01_generate.py --n 300 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import urllib.request

from modeling import DEFAULT_MODEL, load
from segmentation import segment
from telemetry import render_steps

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
CACHE = pathlib.Path("data/gsm8k_test.jsonl")

# The n=20 sanity run showed R1-Distill ignores <step> formatting entirely and
# reasons in its native <think> style. We stopped fighting that: steps are derived
# from the <think> reasoning block (see segmentation.py), and the prompt just asks
# for clear stepwise work and a final boxed/stated answer.
PROMPT_TEMPLATE = """Solve the following problem, reasoning one step at a time. \
Show each calculation explicitly and state its numeric result. \
Give the final numeric answer at the end.

Problem: {question}
"""

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def problem_id(question: str) -> str:
    """Stable across seeds and shuffles. A positional index is not."""
    return "gsm8k_" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]


def sample_seed(base_seed: int, pid: str) -> int:
    """Deterministic PER-PROBLEM seed.

    Seeding once before the loop breaks resumability: a resumed run skips
    completed problems WITHOUT consuming their random draws, so the first
    unfinished problem meets a different RNG state than it would have in an
    uninterrupted run. The recorded global seed then cannot reproduce that
    output. Per-problem seeding makes resumed and uninterrupted runs identical
    problem by problem.
    """
    suffix = int(pid.rsplit("_", 1)[-1][:8], 16)
    return (base_seed ^ suffix) % (2 ** 31 - 1)


def load_gsm8k(n: int, seed: int) -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading GSM8K test split -> {CACHE}")
        urllib.request.urlretrieve(GSM8K_TEST_URL, CACHE)

    rows = [json.loads(l) for l in CACHE.read_text().splitlines() if l.strip()]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]

    return [
        {
            "problem_id": problem_id(r["question"].strip()),
            "question": r["question"].strip(),
            "gold_solution": r["answer"],
            "gold_answer": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for r in rows
    ]


# segmentation lives in segmentation.py so 01_generate, resegment and the tests
# all share one definition.


def extract_answer(response: str) -> str | None:
    for rx in (BOXED_RE, ANSWER_RE):
        m = rx.findall(response)
        if m:
            nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", m[-1])
            if nums:
                return nums[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default="data/original_traces.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    args = ap.parse_args()

    import torch
    from transformers import set_seed

    problems = load_gsm8k(args.n, args.seed)   # shuffle only; generation is
                                               # seeded per problem, below

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["problem_id"])
        print(f"[resume] {len(done)} problems already generated; skipping them")

    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"[data] {len(todo)} problems to generate ({len(problems)} requested)")
    if not todo:
        return

    tok, model = load(args.model, args.device, args.dtype)

    gen_params = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "seed": args.seed,
        "dtype": str(next(model.parameters()).dtype),
    }

    n_fallback = n_correct = 0
    with outp.open("a") as fh:
        for i, p in enumerate(todo):
            seed_i = sample_seed(args.seed, p["problem_id"])
            set_seed(seed_i)

            msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=p["question"])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tok.eos_token_id,
                )
            new_ids = out[0][enc["input_ids"].shape[1] :]
            raw = tok.decode(new_ids, skip_special_tokens=True)

            steps, method = segment(raw)
            canonical, _ = render_steps(steps)
            pred = extract_answer(raw)
            correct = is_correct(pred, p["gold_answer"])

            n_fallback += int(method not in ("step_tags", "think_sentences"))
            n_correct += int(correct)

            fh.write(json.dumps({
                **p,
                "prompt": prompt,
                "raw_response": raw,
                "canonical_response": canonical,
                "response_token_ids": new_ids.tolist(),
                "steps": steps,
                "segmentation_method": method,
                "n_steps": len(steps),
                "pred_answer": pred,
                "correct": correct,
                "sample_seed": seed_i,
                "gen_params": gen_params,
            }) + "\n")
            fh.flush()

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(todo)} | acc {n_correct/(i+1):.2f} "
                      f"| non-think seg {n_fallback/(i+1):.2f}", flush=True)

    print(f"\n[done] {outp}")
    print(f"  accuracy               : {n_correct/len(todo):.3f}")
    print(f"  non-think segmentation : {n_fallback/len(todo):.3f}   <- want this LOW")
    print("  (think_sentences is the intended regime; raw_sentences means no")
    print("   </think> was found — usually a runaway generation.)")


if __name__ == "__main__":
    main()