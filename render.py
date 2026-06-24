#!/usr/bin/env python
"""Unified Python rendering CLI for PufferDrive.

Replaces the old C `./visualize` binary, `render_test.py`, and the rendering
side of `evaluate_human_logs.py`. Works with both LSTM and Transformer policies
on both nuPlan and WOMD datasets.

Modes
-----
  Baseline (no co-players):
      python render.py --model-path X.pt
  With a frozen co-player population:
      python render.py --model-path adaptive.pt --co-player-path coplayer.pt
  Human replay (one ego, others follow logged trajectories):
      python render.py --model-path X.pt --human-replay

Architecture is auto-detected from the checkpoint state-dict, but can be
overridden with --policy-architecture.
"""

import argparse
import copy
import glob
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.drive.rollout import RenderContext, RenderView, rollout_loop


VIEW_MODE_BY_NAME = {
    "sim_state": RenderView.FULL_SIM_STATE,
    "bev": RenderView.BEV_AGENT_OBS,
    "persp": RenderView.AGENT_PERSPECTIVE,
}

VIEW_NAME = {
    RenderView.FULL_SIM_STATE: "sim_state",
    RenderView.BEV_AGENT_OBS: "bev",
    RenderView.AGENT_PERSPECTIVE: "persp",
}


def model_id_from_path(path):
    """Short, readable id derived from the checkpoint filename or its run dir."""
    fname = os.path.splitext(os.path.basename(path))[0]
    # If the file lives inside a run directory like puffer_drive_<run_id>/,
    # prefer that (handles intermediate model_*.pt files).
    parent = os.path.basename(os.path.dirname(path) or "")
    candidate = parent if parent.startswith("puffer_") else fname
    return candidate.replace("puffer_adaptive_drive_", "").replace("puffer_drive_", "")


def run_dir_for(model_path):
    """Locate the experiment directory that owns this checkpoint."""
    parent = os.path.dirname(model_path)
    parent_name = os.path.basename(parent or "")
    if parent_name.startswith("puffer_"):
        return parent
    # File is `<root>/puffer_..._<run>.pt`: matching run dir sits next to it.
    fname = os.path.splitext(os.path.basename(model_path))[0]
    if fname.startswith("puffer_"):
        candidate = os.path.join(parent, fname)
        if os.path.isdir(candidate):
            return candidate
    return None


def default_output_dir(model_path):
    """Default to <run_dir>/renders so artifacts live with the experiment."""
    run_dir = run_dir_for(model_path)
    if run_dir:
        return os.path.join(run_dir, "renders")
    return os.path.join("./render_output", model_id_from_path(model_path))


def detect_architecture(model_path, device="cpu"):
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if "positional_embedding" in state_dict:
        return "Transformer"
    if "lstm.weight_ih_l0" in state_dict:
        return "Recurrent"
    return None


def build_config(args):
    """Build the env/vec/policy config dict for one render."""
    if args.adaptive or args.k_scenarios > 1 or args.co_player_path is not None:
        env_name = "puffer_adaptive_drive"
    else:
        env_name = "puffer_drive"

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        config = load_config(env_name)
    finally:
        sys.argv = saved_argv

    arch = args.policy_architecture or detect_architecture(args.model_path) or "Recurrent"
    config["policy_architecture"] = arch
    config["rnn_name"] = arch
    config["use_rnn"] = True
    config["load_model_path"] = args.model_path
    config["vec"] = {"backend": "Serial", "num_envs": 1}

    config["env"]["render_mode"] = 1  # RENDER_HEADLESS
    config["env"]["map_dir"] = args.map_dir
    config["env"]["num_maps"] = args.num_maps
    config["env"]["num_agents"] = args.num_agents
    config["env"]["num_ego_agents"] = args.num_ego_agents
    config["env"]["k_scenarios"] = args.k_scenarios
    config["env"]["scenario_length"] = args.scenario_length
    if args.goal_behavior is not None:
        config["env"]["goal_behavior"] = int(args.goal_behavior)
    # Under gb=3: max_trials_per_episode and per_trial_timeout are derived
    # from k_scenarios + scenario_length in AdaptiveDrivingAgent.__init__.
    # No separate CLI knobs.

    if args.human_replay:
        if env_name == "puffer_adaptive_drive":
            config["env"]["human_replay_mode"] = True
        config["env"]["co_player_enabled"] = False
        config["env"]["max_controlled_agents"] = 1
    elif args.co_player_path is not None:
        config["env"]["co_player_enabled"] = True
        cpp = config["env"].setdefault("co_player_policy", {})
        cpp["policy_path"] = args.co_player_path
        cpp["architecture"] = args.co_player_architecture or detect_architecture(args.co_player_path) or "Recurrent"
        cpp_cond = cpp.setdefault("conditioning", {})
        cpp_cond["type"] = args.co_player_conditioning_type
        cpp_cond["collision_weight_lb"] = args.co_player_collision_weight_lb
        cpp_cond["collision_weight_ub"] = args.co_player_collision_weight_ub
        cpp_cond["offroad_weight_lb"] = args.co_player_offroad_weight_lb
        cpp_cond["offroad_weight_ub"] = args.co_player_offroad_weight_ub
        cpp_cond["goal_weight_lb"] = args.co_player_goal_weight_lb
        cpp_cond["goal_weight_ub"] = args.co_player_goal_weight_ub
        cpp_cond["entropy_weight_lb"] = args.co_player_entropy_weight_lb
        cpp_cond["entropy_weight_ub"] = args.co_player_entropy_weight_ub
        cpp_cond["discount_weight_lb"] = args.co_player_discount_weight_lb
        cpp_cond["discount_weight_ub"] = args.co_player_discount_weight_ub
    else:
        config["env"]["co_player_enabled"] = False

    cond = config["env"].setdefault("conditioning", {})
    cond["type"] = args.conditioning_type
    cond["collision_weight_lb"] = args.collision_weight_lb
    cond["collision_weight_ub"] = args.collision_weight_ub
    cond["offroad_weight_lb"] = args.offroad_weight_lb
    cond["offroad_weight_ub"] = args.offroad_weight_ub
    cond["goal_weight_lb"] = args.goal_weight_lb
    cond["goal_weight_ub"] = args.goal_weight_ub
    cond["entropy_weight_lb"] = args.entropy_weight_lb
    cond["entropy_weight_ub"] = args.entropy_weight_ub
    cond["discount_weight_lb"] = args.discount_weight_lb
    cond["discount_weight_ub"] = args.discount_weight_ub

    config["train"]["device"] = args.device
    return env_name, config


def mode_tag(args):
    if args.human_replay:
        return "human_replay"
    if args.co_player_path is not None:
        return "coplayer"
    return "baseline"


def render_one(env_name, base_config, view_modes, render_idx, seed, args):
    """One render = (one map seed × all view modes), one rollout per view."""
    print(f"\n[Render {render_idx + 1}/{args.num_renders}] map_seed={seed}")
    cfg = copy.deepcopy(base_config)
    cfg["env"]["map_seed"] = seed

    vecenv = load_env(env_name, cfg)
    try:
        policy = load_policy(cfg, vecenv, env_name)
        policy.eval()

        # Pull the actual map id loaded — beats relying on the seed only
        map_ids = getattr(vecenv.driver_env, "map_ids", None)
        map_id = int(map_ids[0]) if map_ids is not None and len(map_ids) > 0 else seed

        model_id = model_id_from_path(args.model_path)
        if args.co_player_path is not None and not args.human_replay:
            mode = f"vs_{model_id_from_path(args.co_player_path)}"
        else:
            mode = mode_tag(args)
        coplayer_part = ""

        # Default max_steps = full episode budget = k_scenarios * scenario_length
        # under both trial and non-trial modes. Under gb=3 the auto-link makes
        # max_trials * per_trial_timeout identical.
        if args.max_steps is not None:
            max_steps = args.max_steps
        else:
            max_steps = args.k_scenarios * args.scenario_length
        os.makedirs(args.output_dir, exist_ok=True)
        saved = []

        for view_mode in view_modes:
            view = VIEW_NAME[view_mode]
            basename = f"{model_id}_{mode}{coplayer_part}_k{args.k_scenarios}_map{map_id:03d}_{view}"
            vecenv.reset(seed=seed)
            rollout_loop(
                policy=policy,
                env=vecenv,
                device=args.device,
                use_rnn=True,
                max_steps=max_steps,
                render_ctx=RenderContext(
                    view_mode=view_mode,
                    env_id=0,
                    draw_traces=True,
                    video_basename=basename,
                ),
            )
            src = f"{basename}.mp4"
            if os.path.exists(src):
                target = os.path.join(args.output_dir, src)
                shutil.move(src, target)
                print(f"  saved {target}")
                saved.append(target)
            else:
                print(f"  WARNING: expected {src} not produced")
    finally:
        vecenv.close()

    return saved


def main():
    p = argparse.ArgumentParser(description="Unified Python rendering for PufferDrive")
    p.add_argument("--model-path", required=True, help="Trained ego policy checkpoint (.pt)")
    p.add_argument(
        "--co-player-path", default=None, help="Frozen co-player policy (.pt). Omit for baseline / human-replay."
    )
    p.add_argument("--human-replay", action="store_true", help="Render in human-replay mode (one ego, others = log)")
    p.add_argument("--adaptive", action="store_true", help="Force puffer_adaptive_drive env even when k_scenarios=1")

    p.add_argument(
        "--policy-architecture",
        choices=["Recurrent", "Transformer"],
        default=None,
        help="Override ego architecture (auto-detected from checkpoint if omitted)",
    )
    p.add_argument(
        "--co-player-architecture",
        choices=["Recurrent", "Transformer"],
        default=None,
        help="Override co-player architecture",
    )

    p.add_argument(
        "--map-dir",
        default="resources/drive/binaries/training",
        help="Map binary directory (e.g. resources/drive/binaries/nuplan)",
    )
    p.add_argument("--num-maps", type=int, default=None, help="Map pool size (default: max(100, num_renders))")
    p.add_argument("--num-renders", type=int, default=1, help="Number of independent renders (different map seeds)")
    p.add_argument("--start-seed", type=int, default=1)
    p.add_argument(
        "--seed-stride",
        type=int,
        default=1009,
        help="Stride between consecutive map seeds. Spaced apart so adjacent renders pick different maps.",
    )
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--num-ego-agents", type=int, default=32)

    p.add_argument("--k-scenarios", type=int, default=2, help="Number of scenarios per episode (adaptive)")
    p.add_argument("--scenario-length", type=int, default=91)
    p.add_argument(
        "--goal-behavior",
        type=int,
        default=None,
        help="Goal behavior: 0=RESPAWN, 1=GENERATE_NEW, 2=STOP, 3=TRIAL "
        "(under TRIAL: k_scenarios = #trials, scenario_length = per-trial timeout). "
        "Defaults to whatever the checkpoint was trained with (ini default 0).",
    )
    p.add_argument(
        "--max-steps", type=int, default=None, help="Steps per render (default: k_scenarios * scenario_length)"
    )

    p.add_argument("--view-mode", choices=["sim_state", "bev", "persp", "all"], default="sim_state")
    p.add_argument("--output-dir", default=None, help="Where to write mp4s (default: <experiment_dir>/renders)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--conditioning-type", choices=["none", "reward", "entropy", "discount", "all"], default="none")
    p.add_argument("--collision-weight-lb", type=float, default=-3.0)
    p.add_argument("--collision-weight-ub", type=float, default=-3.0)
    p.add_argument("--offroad-weight-lb", type=float, default=-1.0)
    p.add_argument("--offroad-weight-ub", type=float, default=-1.0)
    p.add_argument("--goal-weight-lb", type=float, default=1.0)
    p.add_argument("--goal-weight-ub", type=float, default=1.0)
    p.add_argument("--entropy-weight-lb", type=float, default=0.001)
    p.add_argument("--entropy-weight-ub", type=float, default=0.001)
    p.add_argument("--discount-weight-lb", type=float, default=0.98)
    p.add_argument("--discount-weight-ub", type=float, default=0.98)

    p.add_argument(
        "--co-player-conditioning-type", choices=["none", "reward", "entropy", "discount", "all"], default="all"
    )
    p.add_argument("--co-player-collision-weight-lb", type=float, default=-1.0)
    p.add_argument("--co-player-collision-weight-ub", type=float, default=0.0)
    p.add_argument("--co-player-offroad-weight-lb", type=float, default=-0.4)
    p.add_argument("--co-player-offroad-weight-ub", type=float, default=0.0)
    p.add_argument("--co-player-goal-weight-lb", type=float, default=0.0)
    p.add_argument("--co-player-goal-weight-ub", type=float, default=1.0)
    p.add_argument("--co-player-entropy-weight-lb", type=float, default=0.0)
    p.add_argument("--co-player-entropy-weight-ub", type=float, default=0.1)
    p.add_argument("--co-player-discount-weight-lb", type=float, default=0.8)
    p.add_argument("--co-player-discount-weight-ub", type=float, default=1.0)

    args = p.parse_args()

    if args.num_maps is None:
        args.num_maps = max(100, args.num_renders)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model_path)

    if args.view_mode == "all":
        view_modes = list(VIEW_MODE_BY_NAME.values())
    else:
        view_modes = [VIEW_MODE_BY_NAME[args.view_mode]]

    env_name, config = build_config(args)

    print(f"Env: {env_name}")
    print(f"Architecture: {config['policy_architecture']}")
    print(f"Map dir: {args.map_dir}, num_maps: {args.num_maps}, k_scenarios: {args.k_scenarios}")
    print(f"Mode: {'human_replay' if args.human_replay else ('coplayer' if args.co_player_path else 'baseline')}")
    print(f"Views: {[vm.name for vm in view_modes]}")

    saved = []
    for i in range(args.num_renders):
        seed = args.start_seed + i * args.seed_stride
        saved.extend(render_one(env_name, config, view_modes, i, seed, args))

    print(f"\nDone. {len(saved)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
