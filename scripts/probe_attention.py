"""Cache-attention probe (Path 3 diagnostic).

Loads an adaptive checkpoint, runs one episode against a co-player on a
single env, and at every step records the per-(layer, head) attention
weights from the current query to every position in the K/V cache. The
output is enough to answer:

  Q1: How much attention does the policy in scenario k put on positions
      written during scenario 0..k-1?
  Q2: Does that attention vary by layer / head?

For each layer × head we save a (T, horizon) matrix where T is the number
of steps in the episode (= k_scenarios * scenario_length). Each row sums
to 1 (a softmax over the visible cache positions; entries past the
current write position are 0 due to causal masking).

Heatmap visualization is written separately by `visualize_attention.py`.

Usage:
  python scripts/probe_attention.py <checkpoint.pt> [--out /tmp/probe.npz]
                                                    [--gpu 0]
                                                    [--coplayer <id>]
                                                    [--k 2] [--scen-len 201]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/ADA")

from pufferlib.pufferl import load_env
from pufferlib.ocean.drive.rollout import RenderContext, RenderView


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="path to model_*.pt")
    p.add_argument("--out", default="/tmp/probe_attention.npz")
    p.add_argument("--gpu", default="0")
    p.add_argument(
        "--coplayer", default="experiments/puffer_drive_ocd1syvg.pt", help="co-player checkpoint to pair against"
    )
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--scen-len", type=int, default=201)
    p.add_argument("--map-dir", default="resources/drive/binaries/nuplan_201")
    p.add_argument("--num-maps", type=int, default=4999)
    p.add_argument("--map-rand", action="store_true", help="enable map_rand_per_scenario (matches training setup)")
    p.add_argument("--ego-agent", type=int, default=0, help="which ego agent's attention to record (single int)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_env(args):
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
        external_co_player_actions=False,  # simpler: workers run partner inline
        co_player_policy=dict(
            policy_path=args.coplayer,
            architecture="Transformer",
            input_size=128,
            hidden_size=256,
            transformer=dict(
                input_size=256,
                hidden_size=256,
                num_layers=2,
                num_heads=4,
                horizon=args.scen_len,
                dropout=0.0,
            ),
            conditioning=dict(
                type="all",
                collision_weight_lb=-2,
                collision_weight_ub=0,
                offroad_weight_lb=-2,
                offroad_weight_ub=0,
                goal_weight_lb=0,
                goal_weight_ub=1,
                entropy_weight_lb=0,
                entropy_weight_ub=0.10,
                discount_weight_lb=0.8,
                discount_weight_ub=1,
            ),
        ),
        conditioning=dict(type="none"),
        render_mode=0,
        goal_behavior=2,  # stop-on-goal so success is unambiguous per scenario
    )
    render_args = {
        "env": env_kwargs,
        "vec": {"num_envs": 1, "backend": "Serial"},
        "package": "ocean",
    }
    return load_env("puffer_adaptive_drive", render_args)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"checkpoint:  {args.checkpoint}")
    print(f"co-player:   {args.coplayer}")
    print(f"k={args.k}  scen_len={args.scen_len}  map_rand={bool(args.map_rand)}")
    assert os.path.exists(args.checkpoint), args.checkpoint

    print("[1/4] building env…")
    env = build_env(args)
    driver = env.driver_env
    print(f"  num_agents={driver.num_agents}  ego_ids[:5]={list(driver.ego_ids[:5])}")

    print("[2/4] loading policy…")
    # Mirror pufferl.load_policy: instantiate Drive base + Transformer wrapper,
    # then load state_dict. The checkpoint is a pure state_dict (saved via
    # save_checkpoint -> uncompiled_policy.state_dict()).
    from pufferlib.ocean import torch as ocean_torch

    base = ocean_torch.Drive(driver, input_size=128, hidden_size=256)
    horizon_eff = args.k * args.scen_len  # adaptive case: episode_length
    policy = ocean_torch.Transformer(
        driver,
        base,
        input_size=256,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        horizon=horizon_eff,
        dropout=0.0,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    print(
        f"  policy class={type(policy).__name__}  horizon={policy.horizon}  layers={policy.num_layers}  heads={policy.num_heads}"
    )

    print("[3/4] running probed rollout…")
    obs, _ = env.reset()
    state = {"_probe_attention": True, "_attn_weights": []}

    T = args.k * args.scen_len
    total_agents = driver.num_agents
    rewards_history = np.zeros((T, total_agents), dtype=np.float32)
    dones_history = np.zeros((T, total_agents), dtype=bool)

    for t in range(T):
        if t % 50 == 0:
            print(f"  step {t}/{T}  cached_attn_records={len(state['_attn_weights'])}")
        with torch.no_grad():
            ego_obs = obs[driver.ego_ids]
            ob_t = torch.as_tensor(ego_obs).to(device)
            logits, _ = policy.forward_eval(ob_t, state)
            import pufferlib.pytorch

            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(len(driver.ego_ids), -1)
        obs, rewards, dones, truncs, info = env.step(action_np)
        rewards_history[t] = np.asarray(rewards).reshape(-1)
        dones_history[t] = np.asarray(dones).reshape(-1)

    env.close()

    print(f"[4/4] aggregating {len(state['_attn_weights'])} attention records…")
    # state["_attn_weights"] is a list of dicts. Each dict was appended once
    # per (step, layer). For each (layer, head) build a (T, horizon) matrix
    # where row t = attention from query at step t to every cache position.
    # Records are in the order written, so for L layers we have L records
    # per step. We extract for the chosen ego agent.
    num_layers = max(r["layer"] for r in state["_attn_weights"]) + 1
    horizon = state["_attn_weights"][0]["weights"].shape[-1]
    num_heads = state["_attn_weights"][0]["weights"].shape[1]
    num_egos = state["_attn_weights"][0]["weights"].shape[0]
    ego = args.ego_agent
    assert 0 <= ego < num_egos, f"ego_agent {ego} out of range [0, {num_egos})"

    print(f"  layers={num_layers} heads={num_heads} horizon={horizon} ego_idx={ego}")

    # (layer, head, T, horizon)
    attn_lhth = np.zeros((num_layers, num_heads, T, horizon), dtype=np.float32)
    # Each step records L layers in order, so step t starts at index t * num_layers
    for t in range(T):
        for li in range(num_layers):
            rec = state["_attn_weights"][t * num_layers + li]
            assert rec["layer"] == li, f"layer ordering mismatch at step {t}: expected {li} got {rec['layer']}"
            # weights: (B, H, 1, horizon). Take ego agent, drop query dim.
            w = rec["weights"][ego, :, 0, :].numpy()  # (H, horizon)
            attn_lhth[li, :, t, :] = w

    # Sanity: each row should sum to ~1 (softmax over visible positions). On
    # rows where no positions are visible (none yet — shouldn't happen since
    # we always write the current slot before attending), sum could be 0.
    row_sums = attn_lhth.sum(axis=-1)  # (L, H, T)
    print(f"  row_sum stats: min={row_sums.min():.3f} max={row_sums.max():.3f} mean={row_sums.mean():.3f}")

    np.savez_compressed(
        args.out,
        attn_lhth=attn_lhth,
        rewards_history=rewards_history,
        dones_history=dones_history,
        ego_ids=np.asarray(driver.ego_ids),
        ego_recorded=ego,
        num_layers=num_layers,
        num_heads=num_heads,
        horizon=horizon,
        T=T,
        k=args.k,
        scen_len=args.scen_len,
        map_rand=bool(args.map_rand),
        checkpoint=args.checkpoint,
        coplayer=args.coplayer,
    )
    print(f"  saved → {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    # Quick numerical summary: how much attention does s_1 put on s_0 positions?
    if args.k > 1:
        print()
        print("=== cross-scenario attention summary ===")
        for k_idx in range(1, args.k):
            q_start = k_idx * args.scen_len
            q_end = (k_idx + 1) * args.scen_len
            past_end = k_idx * args.scen_len  # everything before s_k
            for li in range(num_layers):
                for h in range(num_heads):
                    block = attn_lhth[li, h, q_start:q_end, :past_end]  # (S, past_end)
                    cross = block.sum(axis=-1).mean()  # mean over s_k query steps
                    print(f"  s_{k_idx} → past   layer={li} head={h}  mean attention mass on past slots = {cross:.4f}")


if __name__ == "__main__":
    main()
