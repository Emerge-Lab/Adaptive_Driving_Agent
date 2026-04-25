"""Standalone benchmark for co-player Transformer forward_eval.

Mirrors the policy construction in pufferlib/vector.py:840-960 with the
co-player config from the user's command (conditioning=all, classic dynamics,
discrete actions, Transformer 2L/4H/256).

Usage:
    python scripts/profile_co_player.py \
        --checkpoint experiments/puffer_drive_b6big5j1.pt \
        --num-co-players 512 --horizon 91 --warmup 5 --iters 50
"""

import argparse
import os
import time
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch

import pufferlib.models
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive


def build_co_player_policy(
    *,
    dynamics_model: str = "classic",
    action_type: str = "discrete",
    condition_type: str = "all",
    input_size: int = 256,
    hidden_size: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    horizon: int = 91,
    base_input_size: int = 128,
    base_hidden_size: int = 256,
):
    reward_conditioned = condition_type in ("reward", "all")
    entropy_conditioned = condition_type in ("entropy", "all")
    discount_conditioned = condition_type in ("discount", "all")

    if action_type == "discrete":
        if dynamics_model == "classic":
            single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
        elif dynamics_model == "jerk":
            single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
        else:
            raise ValueError(dynamics_model)
    else:
        single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    ego_features = {
        "classic": binding.EGO_FEATURES_CLASSIC,
        "jerk": binding.EGO_FEATURES_JERK,
    }[dynamics_model]
    conditioning_dims = (
        (3 if reward_conditioned else 0)
        + (1 if entropy_conditioned else 0)
        + (1 if discount_conditioned else 0)
    )
    ego_features += conditioning_dims
    max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
    max_partner_objects = binding.MAX_AGENTS - 1
    partner_features = binding.PARTNER_FEATURES
    road_features = binding.ROAD_FEATURES
    num_obs = ego_features + max_partner_objects * partner_features + max_road_objects * road_features
    single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(num_obs,), dtype=np.float32)

    co_player_env = SimpleNamespace(
        single_action_space=single_action_space,
        single_observation_space=single_observation_space,
        reward_conditioned=reward_conditioned,
        entropy_conditioned=entropy_conditioned,
        discount_conditioned=discount_conditioned,
        dynamics_model=dynamics_model,
        max_partner_objects=max_partner_objects,
        partner_features=partner_features,
        max_road_objects=max_road_objects,
        road_features=road_features,
    )

    base_policy = Drive(co_player_env, input_size=base_input_size, hidden_size=base_hidden_size)
    policy = pufferlib.models.TransformerWrapper(
        co_player_env,
        base_policy,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        horizon=horizon,
        dropout=0.0,
    )
    return policy, num_obs


def _make_state(num_co_players: int, horizon: int, hidden_size: int, device, dtype):
    return dict(
        transformer_context=torch.zeros(num_co_players, horizon, hidden_size, device=device, dtype=dtype),
        transformer_position=torch.zeros(1, dtype=torch.long, device=device),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/puffer_drive_b6big5j1.pt")
    parser.add_argument("--num-co-players", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=91)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads (CPU only)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    policy, num_obs = build_co_player_policy(horizon=args.horizon)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(state_dict, strict=True)
    policy = policy.to(device).to(dtype).eval()

    print(
        f"Co-player policy: B={args.num_co_players} obs_dim={num_obs} "
        f"horizon={args.horizon} hidden=256 layers=2 heads=4 device={device} dtype={dtype} threads={args.threads}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    obs_np = rng.standard_normal((args.num_co_players, num_obs), dtype=np.float32)
    # Last feature of each road object is categorical [0, 7); fill with valid ints.
    ego_features = num_obs - (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES - binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    road_start = ego_features + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
    road_view = obs_np[:, road_start:].reshape(args.num_co_players, binding.MAX_ROAD_SEGMENT_OBSERVATIONS, binding.ROAD_FEATURES)
    road_view[:, :, -1] = rng.integers(0, 7, size=road_view.shape[:2])
    obs = torch.from_numpy(obs_np).to(device).to(dtype)
    state = _make_state(args.num_co_players, args.horizon, 256, device, dtype)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            policy.forward_eval(obs, state)
        _sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            policy.forward_eval(obs, state)
        _sync()
        elapsed = time.perf_counter() - t0

    per_call_ms = 1000.0 * elapsed / args.iters
    print(
        f"forward_eval: {per_call_ms:.2f} ms/call over {args.iters} iters "
        f"(total {elapsed:.2f} s); projected per-episode (182 steps): {per_call_ms * 182 / 1000:.2f} s",
        flush=True,
    )

    # Decomposition: time the encoder vs the rest of forward_eval.
    base = policy.policy
    with torch.inference_mode():
        for _ in range(3):
            base.encode_observations(obs)
        _sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            base.encode_observations(obs)
        _sync()
        enc_elapsed = time.perf_counter() - t0
    enc_ms = 1000.0 * enc_elapsed / args.iters
    print(
        f"encode_observations alone: {enc_ms:.2f} ms/call "
        f"({enc_ms / per_call_ms * 100:.1f}% of forward_eval)",
        flush=True,
    )


if __name__ == "__main__":
    main()
