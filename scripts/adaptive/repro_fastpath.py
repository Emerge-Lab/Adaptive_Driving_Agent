"""Standalone reproducer for the hidden_size != 256 / co-player crash.

We mimic the exact failing call: co-player nn.TransformerEncoderLayer fastpath
forward with B=16384 (the size produced by bs=32 * per-worker batch 512), T=201,
D=256, h=4, AFTER first allocating an ego-shaped buffer (n, 804, h_ego).

We toggle SDPA backends individually to identify which kernel inside the C++
fastpath faults.
"""
import os
import sys
import torch
import torch.nn as nn

device = torch.device("cuda")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)

# Mimic ego buffer allocation (pufferl evaluate() line 615)
# bs=32, ego_agents_per_batch large
ego_hidden = int(os.environ.get("EGO_HIDDEN", "64"))
ego_n = int(os.environ.get("EGO_N", "512"))           # ego_agents_per_batch ish
ego_horizon = 804
ego_buf = torch.zeros(ego_n, ego_horizon, ego_hidden, device=device, dtype=torch.float32)
print(f"alloc ego buf: ({ego_n}, {ego_horizon}, {ego_hidden}) = {ego_buf.numel()*4/1e6:.1f} MB",
      flush=True)

# Build co-player TransformerEncoder (matching co-player config)
co_d = 256
co_h = 4
co_layers = 2
co_T = 201
co_B = int(os.environ.get("CO_B", "16384"))            # default = full bs=32 case

layer = nn.TransformerEncoderLayer(
    d_model=co_d, nhead=co_h, dim_feedforward=co_d * 2,
    dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
).to(device).eval()
encoder = nn.TransformerEncoder(layer, num_layers=co_layers).to(device).eval()
print(f"built encoder d={co_d} h={co_h} layers={co_layers}", flush=True)

# Build inputs matching the failing co-player call
src = torch.randn(co_B, co_T, co_d, device=device, dtype=torch.float32)
mask = torch.triu(torch.full((co_T, co_T), float("-inf"), device=device), diagonal=1)
print(f"src: {tuple(src.shape)} mask: {tuple(mask.shape)} (B={co_B})", flush=True)

# Toggle SDPA kernels per env var so we can identify which path faults
sdpa_mode = os.environ.get("SDPA_MODE", "default")
torch.backends.cuda.enable_flash_sdp(sdpa_mode in ("default", "flash"))
torch.backends.cuda.enable_mem_efficient_sdp(sdpa_mode in ("default", "memeff"))
torch.backends.cuda.enable_math_sdp(sdpa_mode in ("default", "math"))
print(f"SDPA mode: {sdpa_mode}  "
      f"flash={torch.backends.cuda.flash_sdp_enabled()}  "
      f"memeff={torch.backends.cuda.mem_efficient_sdp_enabled()}  "
      f"math={torch.backends.cuda.math_sdp_enabled()}", flush=True)

print(f"calling encoder fwd...", flush=True)
with torch.no_grad():
    out = encoder(src, mask=mask, is_causal=True)
print(f"OK out: {tuple(out.shape)} min={out.min().item():.3f} max={out.max().item():.3f}",
      flush=True)
print("REPRO_PASSED", flush=True)
