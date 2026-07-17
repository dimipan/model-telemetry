"""
modeling.py — model loading, with a dtype that is resolved rather than assumed.

Kept separate from telemetry.py so the signal definitions stay torch-free and
testable on a laptop.
"""

from __future__ import annotations

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def resolve_dtype(prefer: str, device: str):
    """Pick a working dtype for the REQUESTED device.

    Two bugs this closes:
      1. bf16 was hard-coded. Pre-Ampere cards (T4) do not support it the way
         newer ones do.
      2. An earlier version asked whether CUDA existed ANYWHERE on the machine,
         not whether the caller asked for it — so `--device cpu` on a GPU box
         still selected fp16 for a CPU model.
    """
    import torch

    if prefer == "float16":
        return torch.float16
    if prefer == "bfloat16":
        return torch.bfloat16
    if prefer == "float32":
        return torch.float32

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load(model_name: str = DEFAULT_MODEL, device: str = "cuda", dtype: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = resolve_dtype(dtype, device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dt,
        device_map={"": device},   # explicit whole-model map; more robust than "cuda"
    )
    model.eval()
    print(f"[model] {model_name} | dtype={dt} | device={device}")
    return tok, model
