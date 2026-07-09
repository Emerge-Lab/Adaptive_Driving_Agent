"""Verify whether mask=None, is_causal=True actually produces a causal mask.

If yes: outputs should match the explicit-causal-mask path.
If no: outputs differ -> the 'fix' silently breaks causality.

Test at a B where neither path crashes, so we can compare outputs.
"""
import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(0)

co_d = 256
co_h = 4
co_T = 201
B = 512   # small enough that explicit-mask fastpath doesn't crash

# Build encoder with deterministic init for reproducibility
layer = nn.TransformerEncoderLayer(
    d_model=co_d, nhead=co_h, dim_feedforward=co_d * 2,
    dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
).to(device).eval()
enc = nn.TransformerEncoder(layer, num_layers=2).to(device).eval()

src = torch.randn(B, co_T, co_d, device=device, dtype=torch.float32)
causal_mask = torch.triu(torch.full((co_T, co_T), float("-inf"), device=device), diagonal=1)

with torch.no_grad():
    out_explicit = enc(src, mask=causal_mask, is_causal=True)
    out_implicit = enc(src, mask=None, is_causal=True)
    out_implicit_nocausal = enc(src, mask=None, is_causal=False)
    out_nomask = enc(src, mask=None)

# Compare outputs
def diff(a, b):
    return (a - b).abs().max().item()

print(f"max|explicit - implicit_is_causal|        = {diff(out_explicit, out_implicit):.6e}")
print(f"max|explicit - implicit_NOT_causal|       = {diff(out_explicit, out_implicit_nocausal):.6e}")
print(f"max|explicit - no_mask_no_is_causal_flag| = {diff(out_explicit, out_nomask):.6e}")
print()

# Reasoning:
# - If is_causal=True with mask=None ACTUALLY does causal attention: explicit == implicit_is_causal.
# - If it silently does non-causal: implicit_is_causal == implicit_NOT_causal == no_mask.

eq_implicit = diff(out_explicit, out_implicit) < 1e-4
eq_nocausal = diff(out_implicit, out_implicit_nocausal) < 1e-4

if eq_implicit:
    print("PASS: mask=None + is_causal=True IS equivalent to explicit causal mask.")
elif eq_nocausal:
    print("FAIL: mask=None + is_causal=True silently does NON-CAUSAL attention.")
else:
    print(f"WEIRD: implicit_is_causal != explicit AND != implicit_NOT_causal")
