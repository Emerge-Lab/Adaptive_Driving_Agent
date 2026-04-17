#!/usr/bin/env python
"""Quick test script for Python-based headless rendering.

Usage:
    # Single render with co-player
    python render_test.py --model-path experiments/puffer_drive_XXXX.pt --co-player-path experiments/puffer_drive_YYYY.pt

    # Human replay mode
    python render_test.py --model-path experiments/puffer_drive_XXXX.pt --human-replay

    # Multiple renders with different maps
    python render_test.py --model-path experiments/puffer_drive_XXXX.pt --human-replay --num-renders 50
"""

import argparse
import os
import sys
import glob
import shutil
import copy

import torch

# Add pufferlib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.drive.rollout import RenderContext, RenderView, rollout_loop


def render_single(args, config, render_idx, seed):
    """Render a single video with a specific seed (for map randomization)."""
    # Deep copy config so we can modify it
    cfg = copy.deepcopy(config)
    # Use the seed for map selection
    cfg["env"]["map_seed"] = seed

    print(f"\n{'='*60}")
    print(f"Render {render_idx + 1}/{args.num_renders} (map_seed={seed})")
    print(f"{'='*60}")

    # Create environment with this seed
    print(f"Creating environment...")
    vecenv = load_env(args.env_name, cfg)
    driver = vecenv.driver_env

    print(f"  population_play: {driver.population_play}")
    print(f"  num_agents: {driver.num_agents}")

    # Load policy (reuse if possible, but for simplicity recreate each time)
    print(f"Loading policy...")
    cfg["load_model_path"] = args.model_path
    policy = load_policy(cfg, vecenv, args.env_name)
    policy.eval()

    # Determine if using RNN
    use_rnn = hasattr(policy, 'lstm') or hasattr(policy, 'transformer') or hasattr(policy, 'horizon')

    # Calculate max_steps
    k_scenarios = args.k_scenarios
    scenario_length = args.scenario_length
    max_steps = args.max_steps if args.max_steps is not None else k_scenarios * scenario_length

    print(f"Starting render (max_steps={max_steps})...")

    view_suffix = {0: "_sim_state", 1: "_bev", 2: "_persp"}.get(args.view_mode, "")

    render_ctx = RenderContext(
        view_mode=args.view_mode,
        env_id=0,
        draw_traces=True,
        video_suffix=view_suffix,
    )

    # Reset with specific seed to get consistent map
    vecenv.reset(seed=seed)

    rollout_loop(
        policy=policy,
        env=vecenv,
        device=args.device,
        use_rnn=use_rnn,
        max_steps=max_steps,
        render_ctx=render_ctx,
    )

    vecenv.close()

    # Generate proper video filename
    model_id = os.path.basename(os.path.dirname(args.model_path)).replace("puffer_adaptive_drive_", "")
    if not model_id or model_id == os.path.basename(args.model_path):
        model_id = os.path.splitext(os.path.basename(args.model_path))[0]

    if args.human_replay:
        video_prefix = f"adaptive_{model_id}_human_replay_k{k_scenarios}_map{render_idx:03d}"
    else:
        coplayer_id = os.path.splitext(os.path.basename(args.co_player_path))[0].replace("puffer_drive_", "")
        video_prefix = f"adaptive_{model_id}_coplayer_{coplayer_id}_k{k_scenarios}_map{render_idx:03d}"

    view_name = {0: "sim_state", 1: "bev", 2: "persp"}.get(args.view_mode, "unknown")

    # Move and rename generated videos
    os.makedirs(args.output_dir, exist_ok=True)

    video_files = glob.glob("*.mp4")
    for video_file in video_files:
        new_name = f"{video_prefix}_{view_name}.mp4"
        target = os.path.join(args.output_dir, new_name)
        shutil.move(video_file, target)
        print(f"Saved: {target}")

    if not video_files:
        print("Warning: No video files generated!")

    return target if video_files else None


def main():
    parser = argparse.ArgumentParser(description="Test Python-based headless rendering")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained adaptive agent checkpoint")
    parser.add_argument("--co-player-path", type=str, default=None, help="Path to co-player policy checkpoint (not needed with --human-replay)")
    parser.add_argument("--env-name", type=str, default="puffer_adaptive_drive", help="Environment name")
    parser.add_argument("--output-dir", type=str, default="./render_output", help="Output directory for videos")
    parser.add_argument("--view-mode", type=int, default=0, choices=[0, 1, 2],
                        help="View mode: 0=sim_state, 1=bev, 2=agent_persp")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps to render (default: k_scenarios * scenario_length)")
    parser.add_argument("--k-scenarios", type=int, default=2, help="Number of scenarios per episode (for adaptive agents)")
    parser.add_argument("--scenario-length", type=int, default=91, help="Steps per scenario")
    parser.add_argument("--human-replay", action="store_true", help="Human replay mode (no co-players, use human trajectories)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for policy inference")
    parser.add_argument("--num-maps", type=int, default=None, help="Number of maps in pool (default: max(100, num_renders))")
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries/nuplan", help="Map directory")
    parser.add_argument("--num-renders", type=int, default=1, help="Number of renders with different random maps")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed for map randomization")
    # Co-player conditioning
    parser.add_argument("--co-player-arch", type=str, default="Transformer", help="Co-player architecture")
    parser.add_argument("--co-player-cond-type", type=str, default="all", help="Co-player conditioning type")
    parser.add_argument("--co-player-entropy-lb", type=float, default=0.0, help="Co-player entropy weight lb")
    parser.add_argument("--co-player-entropy-ub", type=float, default=0.1, help="Co-player entropy weight ub")
    parser.add_argument("--co-player-discount-lb", type=float, default=0.8, help="Co-player discount weight lb")
    parser.add_argument("--co-player-discount-ub", type=float, default=1.0, help="Co-player discount weight ub")
    args = parser.parse_args()

    # Validate args
    if not args.human_replay and args.co_player_path is None:
        parser.error("--co-player-path is required unless using --human-replay mode")

    # Set num_maps to at least num_renders so each render can get a different map
    if args.num_maps is None:
        args.num_maps = max(100, args.num_renders)

    # Save original sys.argv and replace with minimal args for load_config
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]

    print(f"Loading config for {args.env_name}...")
    config = load_config(args.env_name)

    # Restore original sys.argv
    sys.argv = original_argv

    # Override for rendering
    config["env"]["render_mode"] = 1  # RENDER_HEADLESS
    config["env"]["map_dir"] = args.map_dir
    config["env"]["num_maps"] = args.num_maps
    config["env"]["num_agents"] = 64
    config["env"]["num_ego_agents"] = 32
    config["env"]["k_scenarios"] = args.k_scenarios
    config["env"]["scenario_length"] = args.scenario_length
    config["vec"] = {"backend": "Serial", "num_envs": 1}

    # Set up co-player or human replay mode
    if args.human_replay:
        config["env"]["human_replay_mode"] = True
        config["env"]["co_player_enabled"] = False
        config["env"]["max_controlled_agents"] = 1
    else:
        config["env"]["co_player_enabled"] = True
        config["env"]["co_player_policy"]["policy_path"] = args.co_player_path
        config["env"]["co_player_policy"]["architecture"] = args.co_player_arch
        config["env"]["co_player_policy"]["conditioning"]["type"] = args.co_player_cond_type
        config["env"]["co_player_policy"]["conditioning"]["entropy_weight_lb"] = args.co_player_entropy_lb
        config["env"]["co_player_policy"]["conditioning"]["entropy_weight_ub"] = args.co_player_entropy_ub
        config["env"]["co_player_policy"]["conditioning"]["discount_weight_lb"] = args.co_player_discount_lb
        config["env"]["co_player_policy"]["conditioning"]["discount_weight_ub"] = args.co_player_discount_ub

    print(f"Configuration:")
    print(f"  map_dir: {args.map_dir}")
    print(f"  num_maps: {args.num_maps}")
    print(f"  k_scenarios: {args.k_scenarios}")
    print(f"  scenario_length: {args.scenario_length}")
    print(f"  human_replay: {args.human_replay}")
    print(f"  num_renders: {args.num_renders}")
    if not args.human_replay:
        print(f"  co_player_path: {args.co_player_path}")
        print(f"  co_player_arch: {args.co_player_arch}")

    # Render multiple videos with different seeds
    saved_videos = []
    for i in range(args.num_renders):
        seed = args.start_seed + i
        video_path = render_single(args, config, i, seed)
        if video_path:
            saved_videos.append(video_path)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(saved_videos)} videos in {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
