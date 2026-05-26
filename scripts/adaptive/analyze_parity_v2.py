"""Compare eval-path logits vs train-path logits position by position
on batch row 0 of the first PPO minibatch."""
import pickle

import numpy as np
import torch
import torch.nn.functional as F


def main():
    path = "/scratch/mmk9418/projects/Adaptive_Driving_Agent/logs/parity_probe_v2.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)

    print("=== keys + shapes ===")
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(f"  {k:20s}  shape={tuple(v.shape)}  dtype={v.dtype}")

    lt = d["logits_train_b0"].float()  # (T, A)
    le = d["logits_eval_b0"].float()   # (T, A)
    actions = d["mb_actions_b0"]       # (T, 1) or (T,)
    mb_lp = d["mb_logprobs_b0"]
    new_lp = d["newlogprob_b0"]
    rem = d["mb_removed_b0"]
    trm = d["mb_terminals_b0"]
    trc = d["mb_truncations_b0"]

    T, A = lt.shape

    # Per-step max abs logit diff
    diff = (lt - le).abs()
    max_per_t = diff.max(dim=1).values
    print(f"\n=== Per-step max |logit_train - logit_eval| ===")
    print(f"  global  mean={max_per_t.mean():.4f}  median={max_per_t.median():.4f}  min={max_per_t.min():.4f}  max={max_per_t.max():.4f}")

    # Per-step log_softmax diff (action distribution divergence)
    lp_train = F.log_softmax(lt, dim=-1)
    lp_eval = F.log_softmax(le, dim=-1)
    # KL(eval || train)
    p_eval = lp_eval.exp()
    kl_per_t = (p_eval * (lp_eval - lp_train)).sum(dim=-1)

    # First 10 positions, full diagnostic
    print("\n=== First 20 positions: per-step stats ===")
    print(f"  {'t':>4}  {'rem':>3}  {'trm':>3}  {'trc':>3}  {'max|Δlogit|':>11}  {'KL_eval||train':>14}  {'logp_old':>10}  {'logp_new':>10}")
    for t in range(min(20, T)):
        print(f"  {t:>4}  {int(rem[t]):>3}  {int(trm[t]):>3}  {int(trc[t]):>3}  {max_per_t[t].item():>11.4f}  {kl_per_t[t].item():>14.4f}  {mb_lp[t].item():>10.4f}  {new_lp[t].item():>10.4f}")

    # Find positions where divergence is HIGH
    print("\n=== Highest divergence positions (max|Δlogit|) ===")
    topk = torch.topk(max_per_t, 15)
    print(f"  {'rank':>4}  {'t':>4}  {'rem':>3}  {'trm':>3}  {'trc':>3}  {'max|Δlogit|':>11}  {'KL_e||t':>10}")
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        print(f"  {rank:>4}  {idx:>4}  {int(rem[idx]):>3}  {int(trm[idx]):>3}  {int(trc[idx]):>3}  {val:>11.4f}  {kl_per_t[idx].item():>10.4f}")

    # First position where divergence starts (sorted by t)
    THRESHOLD = 0.1
    bad_positions = torch.where(max_per_t > THRESHOLD)[0]
    if len(bad_positions) > 0:
        print(f"\n=== First 5 positions where max|Δlogit| > {THRESHOLD} ===")
        for t in bad_positions[:5].tolist():
            # show context
            prior_rem = int(rem[:t].sum().item())
            prior_trm = int(trm[:t].sum().item())
            prior_trc = int(trc[:t].sum().item())
            print(f"  t={t}  rem={int(rem[t])}  trm={int(trm[t])}  trc={int(trc[t])}  max|Δlogit|={max_per_t[t]:.4f}  prior(rem={prior_rem}, trm={prior_trm}, trc={prior_trc})")


if __name__ == "__main__":
    main()
