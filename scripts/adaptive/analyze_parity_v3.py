"""Compute logprobs from train-logits, eval-replay-logits, and original
mb_logprobs side by side, to localize the compile-vs-uncompile divergence."""
import pickle

import numpy as np
import torch
import torch.nn.functional as F


def main():
    path = "/scratch/mmk9418/projects/Adaptive_Driving_Agent/logs/parity_probe_v2.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)

    lt = d["logits_train_b0"].float()  # (T, A) — from compiled train forward
    le = d["logits_eval_b0"].float()   # (T, A) — from UNCOMPILED eval-path replay
    actions = d["mb_actions_b0"]       # (T, 1)
    mb_lp = d["mb_logprobs_b0"]        # from compiled eval at rollout time
    new_lp = d["newlogprob_b0"]        # from compiled train at PPO time

    T, A = lt.shape

    # Compute log-softmax & gather action logprob
    lp_train = F.log_softmax(lt, dim=-1)
    lp_eval_replay = F.log_softmax(le, dim=-1)
    a = actions.view(-1).long()
    logp_from_lt = lp_train.gather(1, a.unsqueeze(1)).squeeze(1)       # logp of action from train logits
    logp_from_le = lp_eval_replay.gather(1, a.unsqueeze(1)).squeeze(1) # logp of action from uncompiled eval replay

    diff_train_vs_replay = (logp_from_lt - logp_from_le).abs()
    diff_train_vs_recorded = (logp_from_lt - mb_lp).abs()
    diff_replay_vs_recorded = (logp_from_le - mb_lp).abs()

    print(f"=== Per-step diffs ===")
    print(f"  |logp_from_train - logp_from_uncompiled_eval_replay|:")
    print(f"    mean={diff_train_vs_replay.mean():.4f}  max={diff_train_vs_replay.max():.4f}")
    print(f"  |logp_from_train - mb_lp_recorded_at_rollout|:")
    print(f"    mean={diff_train_vs_recorded.mean():.4f}  max={diff_train_vs_recorded.max():.4f}")
    print(f"  |logp_from_uncompiled_eval_replay - mb_lp_recorded_at_rollout|:")
    print(f"    mean={diff_replay_vs_recorded.mean():.4f}  max={diff_replay_vs_recorded.max():.4f}")

    # First 20 positions
    print(f"\n=== First 20 positions ===")
    print(f"  {'t':>4}  {'rem':>3}  {'logp_train':>10}  {'logp_eval_replay':>15}  {'mb_lp':>8}  {'newlp(stored)':>13}")
    rem = d["mb_removed_b0"]
    for t in range(20):
        print(f"  {t:>4}  {int(rem[t]):>3}  {logp_from_lt[t].item():>10.4f}  {logp_from_le[t].item():>15.4f}  {mb_lp[t].item():>8.4f}  {new_lp[t].item():>13.4f}")


if __name__ == "__main__":
    main()
