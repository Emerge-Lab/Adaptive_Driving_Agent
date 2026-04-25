import os
import sys
import glob
import shutil
import subprocess
import json


def run_human_replay_eval_in_subprocess(config, logger, global_step):
    """Run human replay evaluation in a subprocess and log metrics to wandb.

    Routes through `pufferl eval --eval.human-replay-eval True` for both adaptive
    and non-adaptive agents. The subprocess uses HumanReplayEvaluator, which
    handles both architectures (LSTM, Transformer) and both agent types.
    """
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print("No model files found for human replay evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        env_config = config.get("env_config", {})
        eval_config = config.get("eval", {})
        conditioning = env_config.get("conditioning", {})
        conditioning_type = conditioning.get("type", "none")
        # Resolve map_dir on the parent so the child can't silently fall back to the ini default
        map_dir = eval_config.get("map_dir") or env_config.get("map_dir")

        cmd = [
            sys.executable, "-m", "pufferlib.pufferl", "eval", config["env"],
            "--load-model-path", latest_cpt,
            "--eval.wosac-realism-eval", "False",
            "--eval.human-replay-eval", "True",
            "--eval.human-replay-num-agents", str(eval_config.get("human_replay_num_agents", 64)),
            "--eval.human-replay-num-maps", str(eval_config.get("human_replay_num_maps", 100)),
            "--eval.human-replay-num-rollouts", str(eval_config.get("human_replay_num_rollouts", 100)),
            "--eval.human-replay-control-mode", str(eval_config.get("human_replay_control_mode", "control_vehicles")),
            *(["--eval.map-dir", str(map_dir)] if map_dir else []),
            "--eval.num-maps", str(eval_config.get("num_maps", 20)),
            "--env.conditioning.type", conditioning_type,
            "--env.conditioning.collision-weight-lb", str(conditioning.get("collision_weight_lb", -3.0)),
            "--env.conditioning.collision-weight-ub", str(conditioning.get("collision_weight_ub", -3.0)),
            "--env.conditioning.offroad-weight-lb", str(conditioning.get("offroad_weight_lb", -1.0)),
            "--env.conditioning.offroad-weight-ub", str(conditioning.get("offroad_weight_ub", -1.0)),
            "--env.conditioning.goal-weight-lb", str(conditioning.get("goal_weight_lb", 1.0)),
            "--env.conditioning.goal-weight-ub", str(conditioning.get("goal_weight_ub", 1.0)),
            "--env.conditioning.entropy-weight-lb", str(conditioning.get("entropy_weight_lb", 0.001)),
            "--env.conditioning.entropy-weight-ub", str(conditioning.get("entropy_weight_ub", 0.001)),
            "--env.conditioning.discount-weight-lb", str(conditioning.get("discount_weight_lb", 0.98)),
            "--env.conditioning.discount-weight-ub", str(conditioning.get("discount_weight_ub", 0.98)),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"Human replay evaluation failed (exit {result.returncode}): {result.stderr}")
            return

        stdout = result.stdout
        if "HUMAN_REPLAY_METRICS_START" not in stdout or "HUMAN_REPLAY_METRICS_END" not in stdout:
            return

        start = stdout.find("HUMAN_REPLAY_METRICS_START") + len("HUMAN_REPLAY_METRICS_START")
        end = stdout.find("HUMAN_REPLAY_METRICS_END")
        metrics = json.loads(stdout[start:end].strip())

        if not (hasattr(logger, "wandb") and logger.wandb):
            return

        log_data = {
            "eval/human_replay_collision_rate": metrics.get("collision_rate", 0),
            "eval/human_replay_offroad_rate": metrics.get("offroad_rate", 0),
            "eval/human_replay_completion_rate": metrics.get("completion_rate", 0),
            "eval/human_replay_score": metrics.get("score", 0),
        }
        for k, v in metrics.items():
            if k.startswith("ada_delta_") or k.startswith("first_scenario_") or k.startswith("last_scenario_"):
                log_data[f"eval/human_replay_{k}"] = v
        logger.wandb.log(log_data, step=global_step)

    except subprocess.TimeoutExpired:
        print("Human replay evaluation timed out")
    except Exception as e:
        print(f"Failed to run human replay evaluation: {e}")


def run_wosac_eval_in_subprocess(config, logger, global_step):
    """Run WOSAC realism evaluation in a subprocess and log metrics to wandb."""
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print("No model files found for WOSAC evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        env_config = config.get("env_config", {})
        eval_config = config.get("eval", {})
        # Forward training map_dir so eval doesn't silently fall back to ini default
        map_dir = eval_config.get("map_dir") or env_config.get("map_dir")

        cmd = [
            sys.executable, "-m", "pufferlib.pufferl", "eval", config["env"],
            "--load-model-path", latest_cpt,
            "--eval.wosac-realism-eval", "True",
            "--eval.wosac-num-agents", str(eval_config.get("wosac_num_agents", 256)),
            "--eval.wosac-init-mode", str(eval_config.get("wosac_init_mode", "create_all_valid")),
            "--eval.wosac-control-mode", str(eval_config.get("wosac_control_mode", "control_wosac")),
            "--eval.wosac-init-steps", str(eval_config.get("wosac_init_steps", 10)),
            "--eval.wosac-goal-behavior", str(eval_config.get("wosac_goal_behavior", 2)),
            "--eval.wosac-goal-radius", str(eval_config.get("wosac_goal_radius", 2.0)),
            "--eval.wosac-sanity-check", str(eval_config.get("wosac_sanity_check", False)),
            "--eval.wosac-aggregate-results", str(eval_config.get("wosac_aggregate_results", True)),
            *(["--eval.map-dir", str(map_dir)] if map_dir else []),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"WOSAC evaluation failed (exit {result.returncode}): {result.stderr}")
            stderr_lower = result.stderr.lower()
            if "out of memory" in stderr_lower:
                print("GPU OOM during WOSAC eval; skipping.")
            return

        stdout = result.stdout
        if "WOSAC_METRICS_START" not in stdout or "WOSAC_METRICS_END" not in stdout:
            return

        start = stdout.find("WOSAC_METRICS_START") + len("WOSAC_METRICS_START")
        end = stdout.find("WOSAC_METRICS_END")
        metrics = json.loads(stdout[start:end].strip())

        if hasattr(logger, "wandb") and logger.wandb:
            logger.wandb.log({
                "eval/wosac_realism_meta_score": metrics["realism_meta_score"],
                "eval/wosac_ade": metrics["ade"],
                "eval/wosac_min_ade": metrics["min_ade"],
                "eval/wosac_total_num_agents": metrics["total_num_agents"],
            }, step=global_step)

    except subprocess.TimeoutExpired:
        print("WOSAC evaluation timed out after 600 seconds")
    except Exception as e:
        print(f"Failed to run WOSAC evaluation: {type(e).__name__}: {e}")


_VIEW_NAMES = {0: "sim_state", 1: "bev", 2: "persp"}


def render_videos(config, policy, logger, epoch, global_step, device="cuda", human_replay=False):
    """Generate and log videos via Python rollout (works with any policy architecture).

    Policy inference is in PyTorch; rendering goes through the C bindings
    (`vec_render`, `vec_set_video_suffix`). Saves under
    <data_dir>/<env>_<run_id>/renders/ with names that include the map_id,
    mode, and view so multiple modes on the same map don't collide.

    Args:
        config: PuffeRL flat train config.
        policy: PyTorch policy (LSTM or Transformer wrapper).
        logger: Logger with run_id and optional .wandb.
        epoch: Current training epoch.
        global_step: Current global step (for wandb step alignment).
        device: Inference device.
        human_replay: If True, render in human-replay mode (1 ego, others = logs).
    """
    import copy
    import torch
    from pufferlib.pufferl import load_env
    from pufferlib.ocean.drive.rollout import RenderContext, RenderView, rollout_loop

    try:
        run_id = logger.run_id
        env_name = config.get("env", "drive")
        model_dir = os.path.join(config["data_dir"], f"{env_name}_{run_id}")
        video_output_dir = os.path.join(model_dir, "renders")
        os.makedirs(video_output_dir, exist_ok=True)

        view_modes = config.get("render_view_modes", [RenderView.FULL_SIM_STATE])
        if isinstance(view_modes, int):
            view_modes = [view_modes]

        env_kwargs = copy.deepcopy(config.get("env_config", {}))
        env_kwargs["render_mode"] = 1  # RENDER_HEADLESS

        if human_replay:
            env_kwargs["co_player_enabled"] = False
            env_kwargs["max_controlled_agents"] = 1
            if "adaptive" in env_name:
                env_kwargs["human_replay_mode"] = True

        render_args = {
            "env": env_kwargs,
            "vec": config.get("vec", {"num_envs": 1, "backend": "Serial"}),
            "package": config.get("package", "ocean"),
        }

        use_rnn = config.get("use_rnn", False)
        episode_length = env_kwargs.get("scenario_length", 91)
        k_scenarios = env_kwargs.get("k_scenarios", 1)
        if k_scenarios > 1:
            episode_length = k_scenarios * episode_length

        mode = "human_replay" if human_replay else (
            "coplayer" if env_kwargs.get("co_player_enabled") else "baseline"
        )
        videos_to_log_world = []
        videos_to_log_agent = []

        for view_mode in view_modes:
            render_env = load_env(env_name, render_args)
            try:
                map_ids = getattr(render_env.driver_env, "map_ids", None)
                map_id = int(map_ids[0]) if map_ids is not None and len(map_ids) > 0 else 0
                view = _VIEW_NAMES.get(int(view_mode), "view")
                basename = f"epoch_{epoch:06d}_{mode}_k{k_scenarios}_map{map_id:03d}_{view}"

                policy.eval()
                rollout_loop(
                    policy=policy,
                    env=render_env,
                    device=device,
                    use_rnn=use_rnn,
                    max_steps=episode_length,
                    render_ctx=RenderContext(
                        view_mode=view_mode,
                        env_id=0,
                        draw_traces=True,
                        video_basename=basename,
                    ),
                )
            finally:
                render_env.close()

            src = f"{basename}.mp4"
            if not os.path.exists(src):
                print(f"render: expected {src} not produced")
                continue

            target_path = os.path.join(video_output_dir, src)
            shutil.move(src, target_path)

            if hasattr(logger, "wandb") and logger.wandb:
                import wandb
                if view == "sim_state":
                    videos_to_log_world.append(wandb.Video(target_path, format="mp4"))
                else:
                    videos_to_log_agent.append(wandb.Video(target_path, format="mp4"))

        if hasattr(logger, "wandb") and logger.wandb and (videos_to_log_world or videos_to_log_agent):
            payload = {}
            world_key = "eval/human_replay_world_view" if human_replay else "render/world_state"
            agent_key = "eval/human_replay_agent_view" if human_replay else "render/agent_view"
            if videos_to_log_world:
                payload[world_key] = videos_to_log_world
            if videos_to_log_agent:
                payload[agent_key] = videos_to_log_agent
            logger.wandb.log(payload, step=global_step)

    except Exception as e:
        print(f"Failed to render videos: {e}")
        import traceback
        traceback.print_exc()
