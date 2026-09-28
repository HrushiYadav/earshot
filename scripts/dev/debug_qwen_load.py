"""Minimal Qwen2.5-Omni load test — capture any error during from_pretrained."""
import os, sys, traceback
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
import torch

print("start", flush=True)
sys.stdout.flush()

try:
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    print("imports ok", flush=True)
    model_id = "Qwen/Qwen2.5-Omni-3B"
    print(f"loading processor for {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("processor loaded", flush=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype = torch.float16
    print(f"loading model on {device} dtype={dtype}...", flush=True)
    sys.stdout.flush()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True,
    )
    print("model loaded", flush=True)
    try:
        if hasattr(model, "disable_talker"):
            model.disable_talker()
            print("talker disabled", flush=True)
    except Exception as e:
        print(f"disable_talker failed: {e}", flush=True)
    print("DONE", flush=True)
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
