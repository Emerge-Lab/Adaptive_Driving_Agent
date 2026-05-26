"""Analyze the parity_probe.pkl: find positions where train-path logprob
differs from eval-path logprob (mb_logprobs), and characterize the divergence."""
import pickle
import sys

import numpy as np
import torch


def main():
    path = "/scratch/mmk9418/projects/Adaptive_Driving_Agent/logs/parity_probe.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)

    print("=== keys + shapes ===")
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"  {k:18s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:18s}  type={type(v).__name__}")

    mb_logprobs = d["mb_logprobs"]
    newlogprob = d["newlogprob"]
    ratio = d["ratio"]
    mb_removed = d["mb_removed"]
    mb_terminals = d["mb_terminals"]
    mb_truncations = d["mb_truncations"]

    # Position the diagonal where divergence is biggest
    logratio = newlogprob - mb_logprobs
    abs_logratio = logratio.abs()
    flat_idx = abs_logratio.argmax()
    if abs_logratio.dim() == 2:
        b, t = (flat_idx // abs_logratio.shape[1]).item(), (flat_idx % abs_logratio.shape[1]).item()
    elif abs_logratio.dim() == 1:
        b, t = 0, flat_idx.item()
    else:
        b = (flat_idx // (abs_logratio.numel() // abs_logratio.shape[0])).item()
        t = (flat_idx % (abs_logratio.numel() // abs_logratio.shape[0])).item()
    print(f"\n=== biggest divergence at (b={b}, t={t}) ===")
    print(f"  mb_logprobs={mb_logprobs[b, t].item():.6f}")
    print(f"  newlogprob ={newlogprob[b, t].item():.6f}")
    print(f"  logratio   ={logratio[b, t].item():.6f}")
    print(f"  ratio      ={ratio[b, t].item():.6f}")
    print(f"  mb_removed[b, t]   = {bool(mb_removed[b, t].item())}")
    print(f"  mb_terminals[b, t] = {float(mb_terminals[b, t].item())}")
    print(f"  mb_truncations[b, t] = {float(mb_truncations[b, t].item())}")

    # Trajectory for batch row b around t
    lo = max(0, t - 5)
    hi = min(mb_removed.shape[1], t + 5)
    print(f"\n  Around (b={b}, t={t}):")
    print(f"  {'t':>4}  {'rem':>3}  {'trm':>3}  {'trc':>3}  {'logp_old':>10}  {'logp_new':>10}  {'ratio':>10}")
    for tt in range(lo, hi):
        print(f"  {tt:>4}  {int(mb_removed[b, tt]):>3}  {int(mb_terminals[b, tt]):>3}  {int(mb_truncations[b, tt]):>3}  {mb_logprobs[b, tt].item():>10.4f}  {newlogprob[b, tt].item():>10.4f}  {ratio[b, tt].item():>10.4f}")

    # Sample: top-10 divergences AT ACTIVE positions (mb_removed=0)
    print("\n=== Top 10 active-sample divergences ===")
    active = ~mb_removed.bool()
    al = abs_logratio.clone()
    al[~active] = -1.0
    topk = torch.topk(al.flatten(), 10)
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        b, t = idx // al.shape[1], idx % al.shape[1]
        # Look back: how many limbo slots in [0..t-1]?
        prior_limbo = mb_removed[b, :t].sum().item()
        prior_term = mb_terminals[b, :t].sum().item()
        prior_trunc = mb_truncations[b, :t].sum().item()
        print(f"  #{rank}: (b={b}, t={t}) |Δlogp|={val:.3f}  ratio={ratio[b, t].item():.4f}  prior_limbo={int(prior_limbo)}  prior_term={int(prior_term)}  prior_trunc={int(prior_trunc)}")

    # Stats: divergence as function of prior_limbo count
    print("\n=== Divergence stats vs. number of limbo slots in prior context ===")
    B, T = mb_removed.shape
    prior_limbo_per_t = torch.cumsum(mb_removed.float(), dim=1) - mb_removed.float()  # (B, T)
    buckets = [0, 1, 5, 10, 25, 50, 100, 200]
    for lo_b, hi_b in zip(buckets[:-1], buckets[1:]):
        mask = active & (prior_limbo_per_t >= lo_b) & (prior_limbo_per_t < hi_b)
        if not mask.any():
            continue
        a = abs_logratio[mask]
        r = ratio[mask]
        print(f"  prior_limbo in [{lo_b},{hi_b}):  n={int(mask.sum())}  |Δlogp|: mean={a.mean():.4f}  max={a.max():.4f}  ratio: min={r.min():.4f}  max={r.max():.4f}")


if __name__ == "__main__":
    main()
