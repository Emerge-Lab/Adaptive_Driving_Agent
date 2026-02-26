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

        print(f"[Human Replay Eval] env_name={env_name}, is_adaptive={is_adaptive}")
        print(f"[Human Replay Eval] Using model: {latest_cpt}")

        if is_adaptive:
            # Use evaluate_human_logs.py for adaptive agents with human replay
            cmd = [
                sys.executable,
                "evaluate_human_logs.py",
                "--policy-path",
                latest_cpt,
                "--policy-architecture",
                config.get("policy_architecture", "Transformer"),
                "--adaptive-driving-agent",
                "1",
                "--k-scenarios",
                str(env_config.get("k_scenarios", 2)),
                "--num-agents",
                str(eval_config.get("human_replay_num_agents", 32)),
                "--num-maps",
                str(eval_config.get("human_replay_num_maps", 100)),
                "--num-rollouts",
                str(eval_config.get("human_replay_num_rollouts", 100)),
                "--dynamics-model",
                str(env_config.get("dynamics_model", "classic")),
                "--human-replay",  # Enable human replay mode
                "--max-controlled-agents",
                "1",
                "--output",
                "/tmp/human_replay_eval.json",
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

        # Prepare evaluation command
        eval_config = config.get("eval", {})
        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            config["env"],
            "--eval.wosac-realism-eval",
            "True",
            "--eval.wosac-batch-size",
            str(eval_config.get("wosac_batch_size", 32)),
            "--eval.wosac-target-scenarios",
            str(eval_config.get("wosac_target_scenarios", 64)),
            "--eval.wosac-scenario-pool-size",
            str(eval_config.get("wosac_scenario_pool_size", 10_000)),
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
        ]

        if not model_files:
            print("No model files found for WOSAC evaluation. Running WOSAC with random policy.")
        elif len(model_files) > 0:
            latest_cpt = max(model_files, key=os.path.getctime)
            cmd.extend(["--load-model-path", latest_cpt])

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
                            "eval/realism_meta_score_std": wosac_metrics["realism_meta_score_std"],
                            "eval/wosac_kinematic_metrics": wosac_metrics["kinematic_metrics"],
                            "eval/wosac_interactive_metrics": wosac_metrics["interactive_metrics"],
                            "eval/wosac_map_based_metrics": wosac_metrics["map_based_metrics"],
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
