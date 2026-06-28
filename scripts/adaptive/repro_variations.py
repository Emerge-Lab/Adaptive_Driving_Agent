"""Test variations of the failing call to narrow down the bug."""
import os
import sys
import torch
import torch.nn as nn

device = torch.device("cuda")
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}", flush=True)

co_d = 256
co_h = 4
co_layers = 2
co_T = 201

def build_encoder():
    layer = nn.TransformerEncoderLayer(
        d_model=co_d, nhead=co_h, dim_feedforward=co_d * 2,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    ).to(device).eval()
    return nn.TransformerEncoder(layer, num_layers=co_layers).to(device).eval()

def try_call(label, B, dtype, with_mask, is_causal):
    torch.cuda.empty_cache()
    enc = build_encoder()
    src = torch.randn(B, co_T, co_d, device=device, dtype=dtype)
    enc = enc.to(dtype)
    if with_mask:
        mask = torch.triu(torch.full((co_T, co_T), float("-inf"), device=device, dtype=dtype), diagonal=1)
    else:
        mask = None
    try:
        with torch.no_grad():
            out = enc(src, mask=mask, is_causal=is_causal)
        torch.cuda.synchronize()
        print(f"PASS  {label}", flush=True)
        return True
    except Exception as e:
        print(f"FAIL  {label}  -- {str(e)[:100]}", flush=True)
        return False

case = os.environ.get("CASE", "")
B = int(os.environ.get("B", "16384"))

if case == "default":
    try_call(f"B={B} fp32 causal-mask is_causal=True", B, torch.float32, True, True)
elif case == "no_mask":
    try_call(f"B={B} fp32 no-mask is_causal=False", B, torch.float32, False, False)
elif case == "is_causal_false":
    try_call(f"B={B} fp32 causal-mask is_causal=False", B, torch.float32, True, False)
elif case == "bf16":
    try_call(f"B={B} bf16 causal-mask is_causal=True", B, torch.bfloat16, True, True)
elif case == "no_mask_is_causal":
    try_call(f"B={B} fp32 no-mask is_causal=True", B, torch.float32, False, True)
else:
    print(f"unknown CASE={case}")
    sys.exit(2)
