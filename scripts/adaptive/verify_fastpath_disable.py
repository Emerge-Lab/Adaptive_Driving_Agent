"""Verify the principled fix: torch.backends.mha.set_fastpath_enabled(False).

Two tests:
1. Correctness — at small B where the fastpath doesn't crash, compare outputs
   between fastpath ON vs OFF. Should be numerically equivalent for causal mask.
2. Crash-avoidance — at B=16384 (the broken size), confirm the fix doesn't
   crash and produces a valid output.
"""
import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(0)

co_d, co_h, co_T = 256, 4, 201

def build():
    layer = nn.TransformerEncoderLayer(
        d_model=co_d, nhead=co_h, dim_feedforward=co_d * 2,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    ).to(device).eval()
    return nn.TransformerEncoder(layer, num_layers=2).to(device).eval()

# ---- Correctness check (small B) ----
print("== CORRECTNESS TEST: fastpath ON vs OFF, B=512 ==")
B_small = 512
enc = build()
src_s = torch.randn(B_small, co_T, co_d, device=device, dtype=torch.float32)
mask = torch.triu(torch.full((co_T, co_T), float("-inf"), device=device), diagonal=1)

torch.backends.mha.set_fastpath_enabled(True)
with torch.no_grad():
    out_fastpath = enc(src_s, mask=mask, is_causal=True)

torch.backends.mha.set_fastpath_enabled(False)
with torch.no_grad():
    out_slowpath = enc(src_s, mask=mask, is_causal=True)

max_diff = (out_fastpath - out_slowpath).abs().max().item()
print(f"max|fastpath - slowpath| = {max_diff:.6e}")
if max_diff < 1e-3:
    print("CORRECTNESS_PASSED: fastpath and slowpath produce numerically equivalent output")
else:
    print(f"CORRECTNESS_FAILED: diff too large ({max_diff})")

# ---- Crash-avoidance check (B=16384) ----
print()
print("== CRASH-AVOIDANCE TEST: fastpath OFF, B=16384 ==")
B_big = 16384
src_b = torch.randn(B_big, co_T, co_d, device=device, dtype=torch.float32)

torch.backends.mha.set_fastpath_enabled(False)
try:
    with torch.no_grad():
        out_big = enc(src_b, mask=mask, is_causal=True)
    torch.cuda.synchronize()
    print(f"CRASH_AVOID_PASSED: B={B_big} works with fastpath disabled. "
          f"out shape={tuple(out_big.shape)} min={out_big.min().item():.3f} max={out_big.max().item():.3f}")
except Exception as e:
    print(f"CRASH_AVOID_FAILED: {str(e)[:150]}")
