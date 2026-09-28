"""Qwen2.5-Omni THINKER-only load test — bypasses the talker to cut memory and
sidesteps the SDPA path that crashed in the full-model load.
"""
import os, sys, traceback
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import torch

print("start", flush=True)

# Use the THINKER-only class — the user's request. The thinker class wraps
# audio encoder + LLM but skips the talker / code2wav, which is what we don't
# need for yes/no on audio. Memory should be ~30–40% smaller than full model.
try:
    from transformers import (
        Qwen2_5OmniThinkerForConditionalGeneration,
        AutoProcessor,
    )
    print("imports ok", flush=True)

    model_id = "Qwen/Qwen2.5-Omni-3B"
    print(f"loading processor for {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("processor loaded", flush=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype = torch.float16
    print(f"loading THINKER-only on {device} dtype={dtype} attn=eager...", flush=True)
    sys.stdout.flush()
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,  # load to CPU first to avoid the device_map MPS copy bug
    )
    print(f"thinker loaded to CPU; param_count={sum(p.numel() for p in model.parameters())/1e9:.2f}B",
          flush=True)
    print("moving to MPS piece by piece...", flush=True)
    model.to(device)
    print("model moved to MPS", flush=True)
    print("DONE", flush=True)
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
