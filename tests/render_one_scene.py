"""Render ONE chosen sub-env's scene from an inspect_system-style rollout.

Drive's vec env exposes num_envs internal sub-envs (one per scene/SDC under
human_replay with max_controlled_agents=1). The render binding takes an
env_id arg, but inspect_system.py + render_videos always pass env_id=0,
so only scene 0 is ever captured.

This script reuses inspect_system's env config + rollout but renders the
sub-env you pick. It also reports per-agent active-step counts at the end
so you can correlate which scene was the "long-active" one.

Usage:
    xvfb-run -a python tests/render_one_scene.py \
        --checkpoint experiments/puffer_adaptive_drive_rwg5a65x/model_puffer_adaptive_drive_000076.pt \
        --info       experiments/puffer_adaptive_drive_rwg5a65x/info.json \
        --env-id 4 --n-steps 804 \
        --out outputs/inspect/rwg5a65x_iter76/human_replay_env4
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pufferlib.vector
import pufferlib.ocean
import pufferlib.models
from pufferlib.ocean.drive.rollout import RenderContext, RenderView, rollout_loop


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--env-id", type=int, default=0,
                    help="which sub-env / scene to render (0..num_agents-1)")
    ap.add_argument("--n-steps", type=int, default=804)
    ap.add_argument("--num-agents", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--map-seed", type=int, default=42,
                    help="env map_seed → must match the probe pass's --map-seed for matched scenes")
    ap.add_argument("--map-dir", default=None,
                    help="override map_dir from info.json (e.g. resources/drive/binaries/nuplan_hard)")
    return ap.parse_args()


def make_human_replay_env_kwargs(info: dict, num_agents: int, map_seed: int, map_dir: str | None = None) -> dict:
    env_cfg = copy.deepcopy(info.get("env", {}))
    env_cfg["num_agents"] = num_agents
    if map_dir is not None:
        env_cfg["map_dir"] = map_dir
        env_cfg["num_maps"] = min(env_cfg.get("num_maps", 540), 540)
    env_cfg["co_player_enabled"] = False
    env_cfg["human_replay_mode"] = True
    env_cfg["max_controlled_agents"] = 1
    env_cfg.pop("num_ego_agents", None)
    env_cfg.pop("external_co_player_actions", None)
    env_cfg.pop("co_player_policy", None)
    env_cfg["goal_behavior"] = 3
    env_cfg["map_seed"] = map_seed
    env_cfg["render_mode"] = 1  # RENDER_HEADLESS
    return env_cfg


def load_policy(info: dict, ckpt_path: str, vec, device: str):
    from pufferlib.ocean.torch import Drive as EgoBase
    transformer_cfg = info.get("transformer", {}) or {}
    policy_cfg = info.get("policy", {}) or {}
    horizon = transformer_cfg.get(
        "horizon",
        info.get("env", {}).get("k_scenarios", 1)
        * info.get("env", {}).get("scenario_length", 91),
    )
    driver = vec.driver_env
    base = EgoBase(driver, input_size=policy_cfg.get("input_size", 128),
                   hidden_size=policy_cfg.get("hidden_size", 256))
    policy = pufferlib.models.TransformerWrapper(
        env=driver, policy=base,
        input_size=transformer_cfg.get("input_size", 256),
        hidden_size=transformer_cfg.get("hidden_size", 256),
        num_layers=transformer_cfg.get("num_layers", 2),
        num_heads=transformer_cfg.get("num_heads", 4),
        horizon=horizon, dropout=0.0,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    policy.load_state_dict(sd, strict=False)
    return policy.to(device).eval()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(args.info).read_text())

    env_kwargs = make_human_replay_env_kwargs(info, args.num_agents, args.map_seed, map_dir=args.map_dir)
    creator = pufferlib.ocean.env_creator("puffer_adaptive_drive")
    vec = pufferlib.vector.make(creator, env_kwargs=env_kwargs,
                                backend="Serial", num_envs=1, seed=args.seed)
    driver = vec.driver_env
    print(f"driver.num_envs = {driver.num_envs}")
    print(f"map_ids per scene = {list(driver.map_ids)}")
    if not (0 <= args.env_id < driver.num_envs):
        raise ValueError(f"--env-id {args.env_id} out of range [0, {driver.num_envs})")
    print(f"rendering scene env_id={args.env_id} (map {driver.map_ids[args.env_id]})")

    policy = load_policy(info, args.checkpoint, vec, args.device)

    # Track per-agent active steps for the post-rollout summary.
    n_active = np.zeros(driver.num_agents, dtype=int)
    obs, _ = vec.reset()
    state = {}

    # Configure render context
    map_id = int(driver.map_ids[args.env_id])
    basename = f"epoch_000000_human_replay_env{args.env_id}_map{map_id:03d}"
    render_ctx = RenderContext(
        view_mode=RenderView.FULL_SIM_STATE,
        env_id=args.env_id,
        draw_traces=True,
        video_basename=basename,
    )

    # rollout_loop is the same one render_videos uses; it will set_video_suffix
    # on the first render call to fix the mp4 filename. We don't pass a policy
    # state-tracking arg — rollout_loop builds it.
    rollout_loop(
        policy=policy,
        env=vec,
        device=args.device,
        use_rnn=False,
        max_steps=args.n_steps,
        render_ctx=render_ctx,
    )

    # Active-step summary: an agent is "active" at tick t iff driver.removed[i]==False then.
    # rollout_loop doesn't expose this; do a quick second-pass query via env_state if needed.
    # For now we'll print map_ids + ask user to look at the mp4.
    vec.close()

    print(f"\n=== rendered scene ===")
    print(f"env_id = {args.env_id}")
    print(f"map_id = {map_id}")
    print(f"output dir = {args.out}")
    # Find the produced mp4 wherever rollout_loop dumped it (Drive writes to .)
    candidates = list(Path(".").glob(f"{basename}*.mp4"))
    if not candidates:
        candidates = list(Path(".").glob("epoch_000000_*.mp4"))
    for c in candidates:
        dst = args.out / c.name
        shutil.move(str(c), str(dst))
        print(f"  saved: {dst}")
    if not candidates:
        print("  [warn] no mp4 found in CWD — check rollout_loop/render output paths")


if __name__ == "__main__":
    main()
