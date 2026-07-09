"""Microbenchmark: forward time for nn.TransformerEncoder with fastpath ON vs OFF.

We can only compare at sizes where the fastpath doesn't crash, so we benchmark
across a range of B and report the slowdown ratio. The actual production B for
the failing case is 16384, but the fastpath crashes there. We extrapolate from
the trend at B=512, 1024, 2048, 4096, 8192, 12288.
"""
import time
import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(0)

co_d, co_h, co_T, co_L = 256, 4, 201, 2

def build():
    layer = nn.TransformerEncoderLayer(
        d_model=co_d, nhead=co_h, dim_feedforward=co_d * 2,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    ).to(device).eval()
    return nn.TransformerEncoder(layer, num_layers=co_L).to(device).eval()

def bench(B, fastpath_enabled, n_warmup=3, n_iter=10):
    torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
    enc = build()
    src = torch.randn(B, co_T, co_d, device=device, dtype=torch.float32)
    mask = torch.triu(torch.full((co_T, co_T), float("-inf"), device=device), diagonal=1)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = enc(src, mask=mask, is_causal=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = enc(src, mask=mask, is_causal=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000  # ms/call

print(f"{'B':>6} {'fastpath ms':>12} {'slowpath ms':>12} {'slow/fast':>10}")
print("-" * 50)
for B in [512, 1024, 2048, 4096, 8192, 12288]:
    try:
        t_fast = bench(B, True)
    except Exception as e:
        t_fast = float("nan")
        print(f"B={B}  fast CRASH: {str(e)[:60]}", flush=True)
        continue
    try:
        t_slow = bench(B, False)
    except Exception as e:
        t_slow = float("nan")
    ratio = t_slow / t_fast if t_fast > 0 else float("nan")
    print(f"{B:>6} {t_fast:>12.2f} {t_slow:>12.2f} {ratio:>9.2f}x", flush=True)

# Also benchmark slowpath at the production B=16384 (where fastpath crashes)
print()
print("== production B=16384 (fastpath crashes; slow only) ==")
torch.cuda.empty_cache()
try:
    t_slow_prod = bench(16384, False, n_iter=5)
    print(f"B=16384  slowpath: {t_slow_prod:.2f} ms/call")
except Exception as e:
    print(f"B=16384  slowpath: FAILED: {str(e)[:100]}")
