"""Counterfactual cache rollout (Path 3 step 4).

For a given adaptive checkpoint, run N k=2 rollouts twice:
  - Condition A (preserved): K/V cache carries over from s_0 -> s_1 normally.
  - Condition B (zeroed):    K/V cache is zeroed at start of s_1, plus
                              transformer_position reset to 0.

Both conditions use identical env seeds + map sequences so per-agent scores
are paired. Then compare ada_delta_score = score(s_1) - score(s_0).

If `mean(score_s1 | preserved) > mean(score_s1 | zeroed)` significantly,
the cache contributes to behavior in s_1. Combined with the attention
probe, this isolates "policy uses cache info" from "policy attends but
gets nothing useful".

Usage:
  python scripts/counterfactual_cache.py <checkpoint.pt> \
      --coplayer experiments/puffer_drive_<id>.pt \
      --rollouts 30 [--gpu 7] [--map-rand] [--out /tmp/cf.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/ADA")

from pufferlib.pufferl import load_env
import pufferlib.pytorch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--coplayer", required=True)
    p.add_argument("--gpu", default="7")
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--scen-len", type=int, default=201)
    p.add_argument("--map-dir", default="resources/drive/binaries/nuplan_201")
    p.add_argument("--num-maps", type=int, default=4999)
    p.add_argument("--map-rand", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--goal-reward-threshold", type=float, default=0.5)
    return p.parse_args()


def build_env(args, seed):
    env_kwargs = dict(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        num_agents=64,
        num_ego_agents=32,
        scenario_length=args.scen_len,
        k_scenarios=args.k,
        dynamics_model="classic",
        map_rand_per_scenario=bool(args.map_rand),
        co_player_enabled=True,
        external_co_player_actions=False,
        co_player_policy=dict(
            policy_path=args.coplayer,
            architecture="Transformer",
            input_size=128, hidden_size=256,
            transformer=dict(
                input_size=256, hidden_size=256, num_layers=2,
                num_heads=4, horizon=args.scen_len, dropout=0.0,
            ),
            conditioning=dict(
                type="all",
                collision_weight_lb=-2, collision_weight_ub=0,
                offroad_weight_lb=-2, offroad_weight_ub=0,
                goal_weight_lb=0, goal_weight_ub=1,
                entropy_weight_lb=0, entropy_weight_ub=0.10,
                discount_weight_lb=0.8, discount_weight_ub=1,
            ),
        ),
        conditioning=dict(type="none"),
        render_mode=0,
        goal_behavior=2,
    )
    render_args = {
        "env": env_kwargs,
        "vec": {"num_envs": 1, "backend": "Serial"},
        "package": "ocean",
    }
    return load_env("puffer_adaptive_drive", render_args)


def load_policy(ckpt, driver, device, k, scen_len):
    from pufferlib.ocean import torch as ocean_torch
    base = ocean_torch.Drive(driver, input_size=128, hidden_size=256)
    policy = ocean_torch.Transformer(
        driver, base,
        input_size=256, hidden_size=256, num_layers=2,
        num_heads=4, horizon=k * scen_len, dropout=0.0,
    ).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def rollout(env, policy, device, args, condition, base_seed):
    """One k-scenario rollout. Returns per-(agent, scenario) success bool array
    of shape (k, num_ego_agents) and the per-agent rewards-history matrix."""
    driver = env.driver_env
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    obs, _ = env.reset()
    state = {}
    num_egos = len(driver.ego_ids)
    success = np.zeros((args.k, num_egos), dtype=bool)
    T = args.k * args.scen_len
    for t in range(T):
        scen = t // args.scen_len
        # Condition B: zero the cache at start of every scenario > 0.
        if condition == "zeroed" and scen > 0 and t % args.scen_len == 0:
            if "k_cache" in state:
                state["k_cache"] = [torch.zeros_like(c) for c in state["k_cache"]]
                state["v_cache"] = [torch.zeros_like(c) for c in state["v_cache"]]
                state["transformer_position"] = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad():
            ego_obs = obs[driver.ego_ids]
            ob_t = torch.as_tensor(ego_obs).to(device)
            logits, _ = policy.forward_eval(ob_t, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(num_egos, -1)
        obs, rewards, dones, truncs, info = env.step(action_np)
        # Stop-on-goal mode: a +reward_goal spike (~1.0) at any step in a
        # scenario means the agent reached its goal that scenario.
        rewards_arr = np.asarray(rewards).reshape(-1)
        ego_rewards = rewards_arr[driver.ego_ids]
        success[scen] |= (ego_rewards > args.goal_reward_threshold)
    return success


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    print(f"checkpoint: {args.checkpoint}")
    print(f"coplayer:   {args.coplayer}")
    print(f"k={args.k} scen_len={args.scen_len} rollouts={args.rollouts} map_rand={bool(args.map_rand)}")

    env = build_env(args, seed=args.seed_base)
    driver = env.driver_env
    print(f"num_egos = {len(driver.ego_ids)}")
    policy = load_policy(args.checkpoint, driver, device, args.k, args.scen_len)
    print(f"policy horizon = {policy.horizon}")

    results = {"preserved": [], "zeroed": []}
    for r in range(args.rollouts):
        seed = args.seed_base + r
        for cond in ("preserved", "zeroed"):
            succ = rollout(env, policy, device, args, cond, seed)
            results[cond].append(succ)
            print(f"  rollout {r:3d} cond={cond:9s} "
                  f"s0_rate={succ[0].mean():.3f} s1_rate={succ[1].mean():.3f} "
                  f"ada_delta={succ[1].mean() - succ[0].mean():+.3f}")
    env.close()

    # Aggregate.
    A = np.stack(results["preserved"])  # (R, k, num_egos)
    B = np.stack(results["zeroed"])
    R, k, n = A.shape

    # Save raw data first so a stats-bug doesn't lose the rollouts.
    if args.out:
        out = {
            "checkpoint": args.checkpoint,
            "coplayer": args.coplayer,
            "rollouts": int(R),
            "num_egos": int(n),
            "k": int(k),
            "scen_len": int(args.scen_len),
            "map_rand": bool(args.map_rand),
            "preserved_success": A.tolist(),
            "zeroed_success": B.tolist(),
        }
        with open(args.out, "w") as fh:
            json.dump(out, fh)
        print(f"saved raw → {args.out}")

    print()
    print(f"=== summary over {R} rollouts × {n} agents per rollout ===")
    for cond, arr in [("preserved (cache kept)", A), ("zeroed    (cache wiped)", B)]:
        s0 = arr[:, 0].mean()
        s1 = arr[:, 1].mean()
        delta = s1 - s0
        # std across rollouts of per-rollout means
        s0_std = arr[:, 0].mean(axis=-1).std(ddof=1)
        s1_std = arr[:, 1].mean(axis=-1).std(ddof=1)
        print(f"  {cond}:  s0={s0:.4f}±{s0_std:.4f}  s1={s1:.4f}±{s1_std:.4f}  ada_delta={delta:+.4f}")

    # Paired (per-agent) lift from cache. Cast bool -> int for arithmetic.
    s1_A = A[:, 1].reshape(-1).astype(np.int32)
    s1_B = B[:, 1].reshape(-1).astype(np.int32)
    paired_lift = s1_A.mean() - s1_B.mean()
    paired_lift_std = (s1_A - s1_B).std(ddof=1) / np.sqrt(len(s1_A))
    print(f"  paired s1 lift (preserved - zeroed): {paired_lift:+.4f} (sem={paired_lift_std:.4f})")
    print(f"    n_paired={len(s1_A)}  (R*n_egos)")
    if paired_lift_std > 0:
        z = paired_lift / paired_lift_std
        print(f"    z-score = {z:+.2f}")

    # Conditional recovery: P(succ s1 | fail s0) under each condition.
    print()
    print("=== conditional recovery: P(succ s1 | fail s0) ===")
    for cond, arr in [("preserved", A), ("zeroed", B)]:
        f0 = (~arr[:, 0]).reshape(-1)
        s1 = arr[:, 1].reshape(-1)
        if f0.sum() > 0:
            p = s1[f0].mean()
            print(f"  {cond}:  {p:.4f}  (n_failed_s0={int(f0.sum())})")
        else:
            print(f"  {cond}:  N/A (no failures in s_0)")

if __name__ == "__main__":
    main()
