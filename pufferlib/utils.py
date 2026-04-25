import os
import sys
import glob
import shutil
import subprocess
import json


def run_human_replay_eval_in_subprocess(config, logger, global_step):
    """
    Run human replay evaluation in a subprocess and log metrics to wandb.

    For adaptive agents, this runs evaluate_human_logs.py with --human-replay flag.
    For non-adaptive agents, this runs pufferl eval with human-replay-eval flag.
    """
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print("No model files found for human replay evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        # Check if this is an adaptive driving agent
        # config["env"] is the env name string (e.g., "puffer_adaptive_drive")
        env_name = config.get("env", "")
        is_adaptive = "adaptive" in env_name

        # Get nested config sections
        env_config = config.get("env_config", {})
        eval_config = config.get("eval", {})

        # Get conditioning config for passing to eval subprocess
        conditioning = env_config.get("conditioning", {})
        conditioning_type = conditioning.get("type", "none")

        print(f"[Human Replay Eval] env_name={env_name}, is_adaptive={is_adaptive}")
        print(f"[Human Replay Eval] Using model: {latest_cpt}")
        print(f"[Human Replay Eval] conditioning_type={conditioning_type}")

        if is_adaptive:
            # Use evaluate_human_logs.py for adaptive agents with human replay
            # Get architecture from config (determines Recurrent vs Transformer)
            # Check both policy_architecture and rnn_name for compatibility
            rnn_name = config.get("policy_architecture", config.get("rnn_name", "Recurrent"))

            cmd = [
                sys.executable,
                "evaluate_human_logs.py",
                "--policy-path",
                latest_cpt,
                "--rnn-name",
                rnn_name,
                "--adaptive-driving-agent",
                "1",
                "--k-scenarios",
                str(env_config.get("k_scenarios", 1)),
                "--num-agents",
                str(eval_config.get("human_replay_num_agents", 32)),
                "--num-maps",
                str(eval_config.get("human_replay_num_maps", 100)),
                "--map-dir",
                str(eval_config.get("map_dir", env_config.get("map_dir", "resources/drive/binaries/training"))),
                "--num-rollouts",
                str(eval_config.get("human_replay_num_rollouts", 100)),
                "--dynamics-model",
                str(env_config.get("dynamics_model", "classic")),
                "--human-replay",  # Enable human replay mode
                "--max-controlled-agents",
                "1",
                "--output",
                "/tmp/human_replay_eval.json",
                # Pass conditioning settings
                "--conditioning-type",
                conditioning_type,
                "--collision-weight-lb",
                str(conditioning.get("collision_weight_lb", -3.0)),
                "--collision-weight-ub",
                str(conditioning.get("collision_weight_ub", -3.0)),
                "--offroad-weight-lb",
                str(conditioning.get("offroad_weight_lb", -1.0)),
                "--offroad-weight-ub",
                str(conditioning.get("offroad_weight_ub", -1.0)),
                "--goal-weight-lb",
                str(conditioning.get("goal_weight_lb", 1.0)),
                "--goal-weight-ub",
                str(conditioning.get("goal_weight_ub", 1.0)),
                "--entropy-weight-lb",
                str(conditioning.get("entropy_weight_lb", 0.001)),
                "--entropy-weight-ub",
                str(conditioning.get("entropy_weight_ub", 0.001)),
                "--discount-weight-lb",
                str(conditioning.get("discount_weight_lb", 0.98)),
                "--discount-weight-ub",
                str(conditioning.get("discount_weight_ub", 0.98)),
            ]
            print(f"[Human Replay Eval] Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

            if result.returncode == 0:
                # Read metrics from output JSON file
                try:
                    with open("/tmp/human_replay_eval.json", "r") as f:
                        human_replay_metrics = json.load(f)

                    # Log to wandb if available
                    if hasattr(logger, "wandb") and logger.wandb:
                        log_data = {
                            "eval/human_replay_collision_rate": human_replay_metrics.get("collision_rate", 0),
                            "eval/human_replay_offroad_rate": human_replay_metrics.get("offroad_rate", 0),
                            "eval/human_replay_completion_rate": human_replay_metrics.get("completion_rate", 0),
                            "eval/human_replay_score": human_replay_metrics.get("score", 0),
                        }
                        # Add adaptive delta metrics if available (difference between last and first scenario)
                        if "ada_delta_score" in human_replay_metrics:
                            # All delta metrics
                            delta_metrics = [
                                "ada_delta_score",
                                "ada_delta_collision_rate",
                                "ada_delta_offroad_rate",
                                "ada_delta_completion_rate",
                                "ada_delta_episode_return",
                                "ada_delta_perf",
                                "ada_delta_dnf_rate",
                                "ada_delta_num_goals_reached",
                            ]
                            for metric in delta_metrics:
                                if metric in human_replay_metrics:
                                    log_data[f"eval/human_replay_{metric}"] = human_replay_metrics[metric]

                            # First and last scenario metrics
                            scenario_metrics = [
                                "first_scenario_score",
                                "first_scenario_collision_rate",
                                "first_scenario_offroad_rate",
                                "first_scenario_episode_return",
                                "last_scenario_score",
                                "last_scenario_collision_rate",
                                "last_scenario_offroad_rate",
                                "last_scenario_episode_return",
                            ]
                            for metric in scenario_metrics:
                                if metric in human_replay_metrics:
                                    log_data[f"eval/human_replay_{metric}"] = human_replay_metrics[metric]

                        logger.wandb.log(log_data, step=global_step)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Failed to read human replay metrics: {e}")
            else:
                print(f"Human replay evaluation failed with exit code {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
        else:
            # Non-adaptive: use original pufferl eval path
            cmd = [
                sys.executable,
                "-m",
                "pufferlib.pufferl",
                "eval",
                config["env"],
                "--load-model-path",
                latest_cpt,
                "--eval.wosac-realism-eval",
                "False",
                "--eval.human-replay-eval",
                "True",
                "--eval.human-replay-num-agents",
                str(eval_config.get("human_replay_num_agents", 64)),
                "--eval.human-replay-control-mode",
                str(eval_config.get("human_replay_control_mode", "control_sdc_only")),
                # Forward eval map_dir/num_maps so the subprocess does not fall back to ini defaults
                "--eval.map-dir",
                str(eval_config.get("map_dir", env_config.get("map_dir", "resources/drive/binaries/training"))),
                "--eval.num-maps",
                str(eval_config.get("num_maps", 20)),
                # Pass conditioning settings
                "--env.conditioning.type",
                conditioning_type,
                "--env.conditioning.collision-weight-lb",
                str(conditioning.get("collision_weight_lb", -3.0)),
                "--env.conditioning.collision-weight-ub",
                str(conditioning.get("collision_weight_ub", -3.0)),
                "--env.conditioning.offroad-weight-lb",
                str(conditioning.get("offroad_weight_lb", -1.0)),
                "--env.conditioning.offroad-weight-ub",
                str(conditioning.get("offroad_weight_ub", -1.0)),
                "--env.conditioning.goal-weight-lb",
                str(conditioning.get("goal_weight_lb", 1.0)),
                "--env.conditioning.goal-weight-ub",
                str(conditioning.get("goal_weight_ub", 1.0)),
                "--env.conditioning.entropy-weight-lb",
                str(conditioning.get("entropy_weight_lb", 0.001)),
                "--env.conditioning.entropy-weight-ub",
                str(conditioning.get("entropy_weight_ub", 0.001)),
                "--env.conditioning.discount-weight-lb",
                str(conditioning.get("discount_weight_lb", 0.98)),
                "--env.conditioning.discount-weight-ub",
                str(conditioning.get("discount_weight_ub", 0.98)),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

            if result.returncode == 0:
                # Extract JSON from stdout between markers
                stdout = result.stdout
                if "HUMAN_REPLAY_METRICS_START" in stdout and "HUMAN_REPLAY_METRICS_END" in stdout:
                    start = stdout.find("HUMAN_REPLAY_METRICS_START") + len("HUMAN_REPLAY_METRICS_START")
                    end = stdout.find("HUMAN_REPLAY_METRICS_END")
                    json_str = stdout[start:end].strip()
                    human_replay_metrics = json.loads(json_str)

                    # Log to wandb if available
                    if hasattr(logger, "wandb") and logger.wandb:
                        logger.wandb.log(
                            {
                                "eval/human_replay_collision_rate": human_replay_metrics["collision_rate"],
                                "eval/human_replay_offroad_rate": human_replay_metrics["offroad_rate"],
                                "eval/human_replay_completion_rate": human_replay_metrics["completion_rate"],
                            },
                            step=global_step,
                        )
            else:
                print(f"Human replay evaluation failed with exit code {result.returncode}: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Human replay evaluation timed out")
    except Exception as e:
        print(f"Failed to run human replay evaluation: {e}")


def run_wosac_eval_in_subprocess(config, logger, global_step):
    """
    Run WOSAC evaluation in a subprocess and log metrics to wandb.

    Args:
        config: Configuration dictionary containing data_dir, env, and wosac settings
        logger: Logger object with run_id and optional wandb attribute
        epoch: Current training epoch
        global_step: Current global training step

    Returns:
        None. Prints error messages if evaluation fails.
    """
    try:
        run_id = logger.run_id
        model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")
        model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))

        if not model_files:
            print("No model files found for WOSAC evaluation")
            return

        latest_cpt = max(model_files, key=os.path.getctime)

        # Prepare evaluation command
        eval_config = config.get("eval", {})
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--load-model-path",
            latest_cpt,
            "--eval.wosac-realism-eval",
            "True",
            "--eval.wosac-num-agents",
            str(eval_config.get("wosac_num_agents", 256)),
            "--eval.wosac-init-mode",
            str(eval_config.get("wosac_init_mode", "create_all_valid")),
            "--eval.wosac-control-mode",
            str(eval_config.get("wosac_control_mode", "control_wosac")),
            "--eval.wosac-init-steps",
            str(eval_config.get("wosac_init_steps", 10)),
            "--eval.wosac-goal-behavior",
            str(eval_config.get("wosac_goal_behavior", 2)),
            "--eval.wosac-goal-radius",
            str(eval_config.get("wosac_goal_radius", 2.0)),
            "--eval.wosac-sanity-check",
            str(eval_config.get("wosac_sanity_check", False)),
            "--eval.wosac-aggregate-results",
            str(eval_config.get("wosac_aggregate_results", True)),
        ]

        # Run WOSAC evaluation in subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())

        if result.returncode == 0:
            # Extract JSON from stdout between markers
            stdout = result.stdout
            if "WOSAC_METRICS_START" in stdout and "WOSAC_METRICS_END" in stdout:
                start = stdout.find("WOSAC_METRICS_START") + len("WOSAC_METRICS_START")
                end = stdout.find("WOSAC_METRICS_END")
                json_str = stdout[start:end].strip()
                wosac_metrics = json.loads(json_str)

                # Log to wandb if available
                if hasattr(logger, "wandb") and logger.wandb:
                    logger.wandb.log(
                        {
                            "eval/wosac_realism_meta_score": wosac_metrics["realism_meta_score"],
                            "eval/wosac_ade": wosac_metrics["ade"],
                            "eval/wosac_min_ade": wosac_metrics["min_ade"],
                            "eval/wosac_total_num_agents": wosac_metrics["total_num_agents"],
                        },
                        step=global_step,
                    )
        else:
            print(f"WOSAC evaluation failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")

            # Check for memory issues
            stderr_lower = result.stderr.lower()
            if "out of memory" in stderr_lower or "cuda out of memory" in stderr_lower:
                print("GPU out of memory. Skipping this WOSAC evaluation.")

    except subprocess.TimeoutExpired:
        print("WOSAC evaluation timed out after 600 seconds")
    except MemoryError as e:
        print(f"WOSAC evaluation ran out of memory. Skipping this evaluation: {e}")
    except Exception as e:
        print(f"Failed to run WOSAC evaluation: {type(e).__name__}: {e}")


def render_videos_python(config, policy, logger, epoch, global_step, device="cuda"):
    """
    Generate and log training videos using Python-based rendering.

    This function works with ANY policy architecture (LSTM, Transformer, etc.)
    because policy inference happens in Python/PyTorch, not in C.

    Args:
        config: Configuration dictionary containing env settings
        policy: The policy to render (PyTorch model)
        logger: Logger object with run_id and optional wandb attribute
        epoch: Current training epoch
        global_step: Current global training step
        device: Device for policy inference (default: "cuda")

    Returns:
        None. Prints error messages if rendering fails.
    """
    import copy
    import glob
    import torch
    from pufferlib.pufferl import load_env
    from pufferlib.ocean.drive.rollout import RenderContext, RenderView, rollout_loop

    try:
        print("[Python Render] ========== STARTING PYTHON-BASED RENDERING ==========")
        run_id = logger.run_id
        # config["env"] is the env name string in PuffeRL's train_config
        env_name = config.get("env", "drive")
        model_dir = os.path.join(config["data_dir"], f"{env_name}_{run_id}")
        video_output_dir = os.path.join(model_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # Get render settings from config
        view_modes = config.get("render_view_modes", [RenderView.FULL_SIM_STATE])
        if isinstance(view_modes, int):
            view_modes = [view_modes]

        # Create render config for load_env
        # load_env expects: args["env"] = env kwargs dict, args["vec"] = vec kwargs, args["package"] = package name
        # PuffeRL's config has: config["env"] = env name, config["env_config"] = env kwargs
        # Use deep copy to avoid modifying original config
        env_kwargs = copy.deepcopy(config.get("env_config", {}))
        env_kwargs["render_mode"] = 1  # RENDER_HEADLESS

        # Debug: Print conditioning and co-player settings
        conditioning = env_kwargs.get("conditioning", {})
        co_player_policy = env_kwargs.get("co_player_policy", {})
        co_player_enabled = env_kwargs.get("co_player_enabled", False)

        print(f"[render] env_name: {env_name}")
        print(f"[render] co_player_enabled: {co_player_enabled}")
        print(f"[render] conditioning type: {conditioning.get('type', 'none')}")
        if co_player_enabled:
            print(f"[render] co_player_policy path: {co_player_policy.get('policy_path', 'NOT SET')}")
            print(f"[render] co_player_policy architecture: {co_player_policy.get('architecture', 'NOT SET')}")
            co_player_cond = co_player_policy.get("conditioning", {})
            print(f"[render] co_player conditioning type: {co_player_cond.get('type', 'none')}")

        render_args = {
            "env": env_kwargs,
            "vec": config.get("vec", {"num_envs": 1, "backend": "Serial"}),
            "package": config.get("package", "ocean"),
        }

        # Determine if using RNN/Transformer
        # Check policy type directly from config or infer from policy
        use_rnn = config.get("use_rnn", False)
        rnn_name = config.get("rnn_name") or config.get("policy_architecture", "Recurrent")
        print(f"[render] use_rnn: {use_rnn}, rnn_name: {rnn_name}")

        # Get episode length from env_config
        episode_length = env_kwargs.get("scenario_length", 91)
        # For adaptive agents, episode_length = k_scenarios * scenario_length
        k_scenarios = env_kwargs.get("k_scenarios", 1)
        if k_scenarios > 1:
            episode_length = k_scenarios * episode_length
        print(f"[render] episode_length: {episode_length}, k_scenarios: {k_scenarios}")

        videos_to_log_world = []
        videos_to_log_agent = []

        for view_mode in view_modes:
            view_suffix = {
                RenderView.FULL_SIM_STATE: "_sim_state",
                RenderView.BEV_AGENT_OBS: "_bev",
                RenderView.AGENT_PERSPECTIVE: "_persp",
            }.get(view_mode, "")

            # Create render environment
            render_env = load_env(env_name, render_args)

            # Debug: Print render environment state
            driver = render_env.driver_env
            print(f"[render] render_env created successfully")
            print(f"[render] population_play: {driver.population_play}")
            print(f"[render] num_agents: {driver.num_agents}")
            if driver.population_play:
                print(f"[render] num_ego_agents: {driver.num_ego_agents}")
                print(f"[render] num_co_players: {driver.num_co_players}")
                print(f"[render] co_player_policy loaded: {driver.co_player_policy is not None}")
            print(f"[render] reward_conditioned: {driver.reward_conditioned}")
            print(f"[render] entropy_conditioned: {driver.entropy_conditioned}")
            print(f"[render] discount_conditioned: {driver.discount_conditioned}")
            print(f"[render] observation_space shape: {render_env.observation_space.shape}")

            try:
                policy.eval()
                print(f"[Python Render] Starting rollout_loop with max_steps={episode_length}, view_mode={view_mode}")
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
                        video_suffix=view_suffix,
                    ),
                )
                print(f"[Python Render] rollout_loop completed successfully")
            finally:
                render_env.close()

        # Collect generated videos (written to cwd by C code)
        video_files = glob.glob("*.mp4")
        print(f"[Python Render] Found {len(video_files)} video files: {video_files}")
        for video_file in video_files:
            target_filename = f"epoch_{epoch:06d}_{os.path.basename(video_file)}"
            target_path = os.path.join(video_output_dir, target_filename)
            shutil.move(video_file, target_path)

            if hasattr(logger, "wandb") and logger.wandb:
                import wandb
                if "_sim_state" in target_filename or "topdown" in target_filename:
                    videos_to_log_world.append(wandb.Video(target_path, format="mp4"))
                else:
                    videos_to_log_agent.append(wandb.Video(target_path, format="mp4"))

        # Log videos to wandb
        if hasattr(logger, "wandb") and logger.wandb and (videos_to_log_world or videos_to_log_agent):
            payload = {}
            if videos_to_log_world:
                payload["render/world_state"] = videos_to_log_world
            if videos_to_log_agent:
                payload["render/agent_view"] = videos_to_log_agent
            logger.wandb.log(payload, step=global_step)

        print(f"Python-based rendering completed for epoch {epoch}")

    except Exception as e:
        print(f"Failed to render videos with Python: {e}")
        import traceback
        traceback.print_exc()


def render_videos(config, vecenv, logger, epoch, global_step, bin_path):
    """
    Generate and log training videos using C-based rendering.

    Args:
        config: Configuration dictionary containing data_dir, env, and render settings
        vecenv: Vectorized environment with driver_env attribute
        logger: Logger object with run_id and optional wandb attribute
        epoch: Current training epoch
        global_step: Current global training step
        bin_path: Path to the exported .bin model weights file

    Returns:
        None. Prints error messages if rendering fails.
    """
    if not os.path.exists(bin_path):
        print(f"Binary weights file does not exist: {bin_path}")
        return

    run_id = logger.run_id
    model_dir = os.path.join(config["data_dir"], f"{config['env']}_{run_id}")

    # Now call the C rendering function
    try:
        # Create output directory for videos
        video_output_dir = os.path.join(model_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        # Copy the binary weights to the expected location
        expected_weights_path = "resources/drive/puffer_drive_weights.bin"
        os.makedirs(os.path.dirname(expected_weights_path), exist_ok=True)
        shutil.copy2(bin_path, expected_weights_path)

        # TODO: Fix memory leaks so that this is not needed
        # Suppress AddressSanitizer exit code (temp)
        env_vars = os.environ.copy()
        env_vars["ASAN_OPTIONS"] = "exitcode=0"

        # Detect if this is an adaptive agent
        env_name = config.get("env", "")
        is_adaptive = "adaptive" in env_name

        # Select correct INI file based on agent type
        if is_adaptive:
            ini_file = "pufferlib/config/ocean/adaptive.ini"
        else:
            ini_file = "pufferlib/config/ocean/drive.ini"

        # Base command with only visualization flags (env config comes from INI)
        base_cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", "./visualize"]

        # Pass the correct INI file
        base_cmd.extend(["--ini-file", ini_file])

        # Get env config for k_scenarios and co-player settings
        env_config = config.get("env_config", {})

        # Pass conditioning type to visualize binary
        conditioning = env_config.get("conditioning", {})
        conditioning_type = conditioning.get("type", "none")
        base_cmd.extend(["--conditioning-type", conditioning_type])

        # Pass k_scenarios for adaptive agents (longer videos)
        k_scenarios = env_config.get("k_scenarios", 1)
        if k_scenarios > 1:
            base_cmd.extend(["--k-scenarios", str(k_scenarios)])

        # Pass co-player policy if population play is enabled
        co_player_enabled = env_config.get("co_player_enabled", False)
        if co_player_enabled:
            co_player_path = f"resources/drive/{config['env']}_co_player.bin"
            if os.path.exists(co_player_path):
                base_cmd.extend(["--co-player-policy", co_player_path])

        # Visualization config flags only
        if config.get("show_grid", False):
            base_cmd.append("--show-grid")
        if config.get("obs_only", False):
            base_cmd.append("--obs-only")
        if config.get("show_lasers", False):
            base_cmd.append("--lasers")
        if config.get("show_human_logs", False):
            base_cmd.append("--show-human-logs")
        if config.get("zoom_in", False):
            base_cmd.append("--zoom-in")

        # Frame skip for rendering performance
        frame_skip = config.get("frame_skip", 1)
        if frame_skip > 1:
            base_cmd.extend(["--frame-skip", str(frame_skip)])

        # View mode
        view_mode = config.get("view_mode", "both")
        base_cmd.extend(["--view", view_mode])

        # Get num_maps if available
        env_cfg = getattr(vecenv, "driver_env", None)
        if env_cfg is not None and getattr(env_cfg, "num_maps", None):
            base_cmd.extend(["--num-maps", str(env_cfg.num_maps)])

        # Handle single or multiple map rendering
        render_maps = config.get("render_map", None)
        if render_maps is None:
            render_maps = [None]
        elif isinstance(render_maps, (str, os.PathLike)):
            render_maps = [render_maps]
        else:
            # Ensure list-like
            render_maps = list(render_maps)

        # Collect videos to log as lists so W&B shows all in the same step
        videos_to_log_world = []
        videos_to_log_agent = []

        for i, map_path in enumerate(render_maps):
            cmd = list(base_cmd)  # copy
            if map_path is not None and os.path.exists(map_path):
                cmd.extend(["--map-name", str(map_path)])

            # Output paths (overwrite each iteration; then moved/renamed)
            cmd.extend(["--output-topdown", "resources/drive/output_topdown.mp4"])
            cmd.extend(["--output-agent", "resources/drive/output_agent.mp4"])

            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=600, env=env_vars)

            vids_exist = os.path.exists("resources/drive/output_topdown.mp4") and os.path.exists(
                "resources/drive/output_agent.mp4"
            )

            if result.returncode == 0 or (result.returncode == 1 and vids_exist):
                videos = [
                    (
                        "resources/drive/output_topdown.mp4",
                        f"epoch_{epoch:06d}_map{i:02d}_topdown.mp4" if map_path else f"epoch_{epoch:06d}_topdown.mp4",
                    ),
                    (
                        "resources/drive/output_agent.mp4",
                        f"epoch_{epoch:06d}_map{i:02d}_agent.mp4" if map_path else f"epoch_{epoch:06d}_agent.mp4",
                    ),
                ]

                for source_vid, target_filename in videos:
                    if os.path.exists(source_vid):
                        target_path = os.path.join(video_output_dir, target_filename)
                        shutil.move(source_vid, target_path)
                        # Accumulate for a single wandb.log call
                        if hasattr(logger, "wandb") and logger.wandb:
                            import wandb

                            if "topdown" in target_filename:
                                videos_to_log_world.append(wandb.Video(target_path, format="mp4"))
                            else:
                                videos_to_log_agent.append(wandb.Video(target_path, format="mp4"))
                    else:
                        print(f"Video generation completed but {source_vid} not found")
            else:
                print(f"C rendering failed (map index {i}) with exit code {result.returncode}: {result.stdout}")

        # Log all videos at once so W&B keeps all of them under the same step
        if hasattr(logger, "wandb") and logger.wandb and (videos_to_log_world or videos_to_log_agent):
            payload = {}
            if videos_to_log_world:
                payload["render/world_state"] = videos_to_log_world
            if videos_to_log_agent:
                payload["render/agent_view"] = videos_to_log_agent
            logger.wandb.log(payload, step=global_step)

    except subprocess.TimeoutExpired:
        print("C rendering timed out")
    except Exception as e:
        print(f"Failed to generate GIF: {e}")

    finally:
        # Clean up bin weights file
        if os.path.exists(expected_weights_path):
            os.remove(expected_weights_path)


def render_human_replay_videos(config, policy_bin_path, output_dir, num_maps=5, logger=None, global_step=0):
    """
    Render videos for human replay evaluation (1 ego agent + human log trajectories).

    In this mode, only one agent is policy-controlled (the ego), while all other agents
    follow their logged human trajectories (rendered in GOLD).

    Args:
        config: Configuration dictionary with env settings
        policy_bin_path: Path to the policy weights .bin file
        output_dir: Directory to save output videos
        num_maps: Number of maps to render
        logger: Optional logger with wandb attribute for logging
        global_step: Current training step for wandb logging

    Returns:
        List of output video paths
    """
    if not os.path.exists(policy_bin_path):
        print(f"Policy weights file does not exist: {policy_bin_path}")
        return []

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Copy the binary weights to the expected location
        expected_weights_path = "resources/drive/puffer_drive_weights.bin"
        os.makedirs(os.path.dirname(expected_weights_path), exist_ok=True)
        shutil.copy2(policy_bin_path, expected_weights_path)

        env_vars = os.environ.copy()
        env_vars["ASAN_OPTIONS"] = "exitcode=0"

        # Get env config
        env_config = config.get("env_config", config.get("env", {}))
        k_scenarios = env_config.get("k_scenarios", env_config.get("k-scenarios", 1))
        map_dir = env_config.get("map_dir", env_config.get("map-dir", None))
        conditioning = env_config.get("conditioning", {})
        conditioning_type = conditioning.get("type", "none")

        # Select correct INI file based on env name
        env_name = config.get("env", "")
        is_adaptive = "adaptive" in env_name
        ini_file = "pufferlib/config/ocean/adaptive.ini" if is_adaptive else "pufferlib/config/ocean/drive.ini"

        # Build command for human replay rendering
        cmd = [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1280x720x24",
            "./visualize",
            "--ini-file",
            ini_file,
            "--policy-name",
            expected_weights_path,
            "--max-controlled-agents",
            "1",  # Only 1 ego agent
            "--k-scenarios",
            str(k_scenarios),
            "--num-maps",
            str(num_maps),
            "--log-trajectories",  # Show human trajectory logs
            "--zoom-in",
            "--view",
            "both",
            "--output-topdown",
            "resources/drive/output_topdown.mp4",
            "--output-agent",
            "resources/drive/output_agent.mp4",
            "--conditioning-type",
            conditioning_type,
        ]

        # Add map_dir override if specified (for NuPlan or other datasets)
        if map_dir:
            cmd.extend(["--map-dir", map_dir])

        output_videos = []
        videos_to_log_world = []
        videos_to_log_agent = []

        print(f"[Human Replay Render] Starting render for {num_maps} maps, map_dir={map_dir}", flush=True)
        print(f"[Human Replay Render] Command: {' '.join(cmd)}", flush=True)

        for map_idx in range(num_maps):
            print(f"[Human Replay Render] Rendering map {map_idx}...", flush=True)
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=600, env=env_vars)
            print(f"[Human Replay Render] Return code: {result.returncode}", flush=True)
            if result.stdout:
                print(f"[Human Replay Render] stdout: {result.stdout[:500]}", flush=True)
            if result.stderr:
                print(f"[Human Replay Render] stderr: {result.stderr[:500]}", flush=True)

            vids_exist = os.path.exists("resources/drive/output_topdown.mp4") and os.path.exists(
                "resources/drive/output_agent.mp4"
            )
            print(f"[Human Replay Render] Videos exist: {vids_exist}", flush=True)

            if result.returncode == 0 or (result.returncode == 1 and vids_exist):
                videos = [
                    ("resources/drive/output_topdown.mp4", f"human_replay_map{map_idx:02d}_topdown.mp4"),
                    ("resources/drive/output_agent.mp4", f"human_replay_map{map_idx:02d}_agent.mp4"),
                ]

                for source_vid, target_filename in videos:
                    if os.path.exists(source_vid):
                        target_path = os.path.join(output_dir, target_filename)
                        shutil.move(source_vid, target_path)
                        output_videos.append(target_path)

                        if logger and hasattr(logger, "wandb") and logger.wandb:
                            import wandb

                            if "topdown" in target_filename:
                                videos_to_log_world.append(wandb.Video(target_path, format="mp4"))
                            else:
                                videos_to_log_agent.append(wandb.Video(target_path, format="mp4"))
            else:
                print(f"Human replay rendering failed for map {map_idx}: {result.stderr}")

        # Log to wandb
        if logger and hasattr(logger, "wandb") and logger.wandb and (videos_to_log_world or videos_to_log_agent):
            payload = {}
            if videos_to_log_world:
                payload["eval/human_replay_world_view"] = videos_to_log_world
            if videos_to_log_agent:
                payload["eval/human_replay_agent_view"] = videos_to_log_agent
            logger.wandb.log(payload, step=global_step)

        return output_videos

    except subprocess.TimeoutExpired:
        print("Human replay rendering timed out")
        return []
    except Exception as e:
        print(f"Failed to render human replay videos: {e}")
        return []
    finally:
        if os.path.exists(expected_weights_path):
            os.remove(expected_weights_path)
