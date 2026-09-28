"""Stage 2 — model loader.

Loads Qwen2.5-Omni-3B (thinker-only) in float16 with eager attention.
Follows the AGENTS.md recipe: load to CPU first, then `.to("mps")` —
`device_map={"": "mps"}` triggers a SIGSEGV in `copy_cast_kernel_mps` on
this PyTorch/torchvision/MPS combo.
"""
from __future__ import annotations

import torch


# Single source of truth for the model id. Both the spike script and the
# scorer import from here.
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"


def pick_device() -> torch.device:
    """Return MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """Load the processor + thinker-only model and move it to MPS.

    Returns:
        processor — HuggingFace processor for audio + text tokenisation.
        model — Qwen2_5OmniThinkerForConditionalGeneration in eval() mode
                on the chosen device, dtype=float16.
        device — torch.device the model lives on (so the scorer can cast
                tensors to the same place).
    """
    from transformers import (
        Qwen2_5OmniThinkerForConditionalGeneration,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,  # CPU-then-move, see module docstring.
    )
    model.eval()

    device = pick_device()
    model.to(device)
    return processor, model, device
