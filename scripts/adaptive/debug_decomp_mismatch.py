"""Debug: track per-step rewards AND env-side trial_R_* accumulators side by
side for ONE specific (rollout, agent) and find where they diverge.

Usage:
  python scripts/adaptive/debug_decomp_mismatch.py --wid 9gc19bcy --k 4 \
      --seed 42 --iter 114 --num-maps 50 --num-rollouts 4 --target-agent 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/scratch/mmk9418/projects/Adaptive_Driving_Agent")
sys.path.insert(0, str(REPO_ROOT))

import pufferlib
import pufferlib.pufferl as pufferl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--num-maps", type=int, default=50)
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--target-agent", type=int, default=0,
                    help="Which agent slot to trace.")
    ap.add_argument("--scenario-length", type=int, default=201)
    args = ap.parse_args()

    import os
    os.environ["PUFFER_TRANSFORMER_LEGACY_EVAL"] = "1"

    horizon = args.k * args.scenario_length
    ckpt = (REPO_ROOT / "experiments" /
            f"puffer_adaptive_drive_{args.wid}" /
            f"model_puffer_adaptive_drive_{args.iter:06d}.pt")
    assert ckpt.exists(), f"ckpt {ckpt} not found"

    argv = [
        "eval",
        "puffer_adaptive_drive",
        "--load", str(ckpt),
        "--seed", str(args.seed),
        "--eval.human-replay-num-rollouts", str(args.num_rollouts),
        "--eval.human-replay-num-agents", str(args.num_maps),
        "--env.num-maps", str(args.num_maps),
        "--env.num-agents", str(args.num_maps),
        "--env.use-all-maps", "True",
        "--env.k-scenarios", str(args.k),
        "--env.scenario-length", str(args.scenario_length),
        "--train.horizon", str(horizon),
        "--env.goal-behavior", "3",
        "--env.conditioning.type", "none",
        "--env.reward-vehicle-collision", "-0.5",
        "--env.reward-offroad-collision", "-0.5",
        "--env.reward-lane-align", "0.05",
    ]
    sys.argv = argv

    config = pufferl.parse_args()
    pufferl.set_seed(config["seed"])
    vecenv = pufferl.make_vecenv(config, evaluation=True)
    policy = pufferl.load_policy(config, vecenv)
    policy.eval()
    device = config["train"]["device"]

    puffer_env = vecenv.driver_env if hasattr(vecenv, "driver_env") else vecenv
    target = args.target_agent

    print(f"\n=== TRACING agent={target}, rollouts={args.num_rollouts}, k={args.k} ===\n")

    for r in range(args.num_rollouts):
        obs, _ = vecenv.reset()
        # Init state for transformer
        state = dict(
            transformer_context=torch.zeros(args.num_maps, policy.horizon, policy.hidden_size, device=device),
            transformer_position=torch.zeros(1, dtype=torch.long, device=device),
        )

        print(f"\n--- rollout {r} ---")
        # Manual per-step tracking for target agent
        py_trial_R = 0.0   # Python's per-step sum for current trial
        current_trial = 0
        max_steps = args.k * args.scenario_length

        for t in range(max_steps):
            with torch.no_grad():
                ob = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)

            obs, rewards, dones, truncs, info_list = vecenv.step(action_np)

            r_step = float(rewards[target])
            te = bool(puffer_env.trial_ended_this_step[target])
            c_acc_goal      = float(puffer_env.trial_R_goal[target])
            c_acc_collision = float(puffer_env.trial_R_collision[target])
            c_acc_offroad   = float(puffer_env.trial_R_offroad[target])
            c_acc_lane      = float(puffer_env.trial_R_lane[target])

            py_trial_R += r_step

            if r_step != 0.0 or te:
                tag = " <-- TRIAL END" if te else ""
                print(f"  step={t:>3} trial={current_trial} r={r_step:+8.4f}"
                      f"  py_sum={py_trial_R:+8.4f}"
                      f"  C: g={c_acc_goal:+7.3f} c={c_acc_collision:+7.3f} "
                      f"o={c_acc_offroad:+7.3f} l={c_acc_lane:+7.3f}"
                      f" sum={c_acc_goal+c_acc_collision+c_acc_offroad+c_acc_lane:+7.3f}"
                      f"{tag}")

            if te:
                c_sum = c_acc_goal + c_acc_collision + c_acc_offroad + c_acc_lane
                diff = py_trial_R - c_sum
                marker = " *** DIFF" if abs(diff) > 0.01 else ""
                print(f"      → TRIAL {current_trial} END:  py_sum={py_trial_R:+.4f}  "
                      f"c_sum={c_sum:+.4f}  diff={diff:+.4f}{marker}")
                current_trial += 1
                py_trial_R = 0.0
                if current_trial >= args.k:
                    break


if __name__ == "__main__":
    main()
