import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import pufferlib
import pufferlib.vector
from pufferlib.ocean import env_creator
from pufferlib.ocean.torch import Drive, Recurrent

import matplotlib.pyplot as plt
import numpy as np


def plot_adaptive_metrics(first_metrics, last_metrics, delta_metrics, output_path):
    """
    Plot adaptive metrics showing first scenario (0-shot), last scenario, and delta improvement.
    """
    # Metrics to plot
    metrics_to_plot = {
        "score": "Score",
        "collision_rate": "Collision Rate",
        "offroad_rate": "Offroad Rate",
        "episode_return": "Episode Return",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]

        first_val = first_metrics[metric_key]
        last_val = last_metrics[metric_key]
        delta_key = f"ada_delta_{metric_key}"
        delta_pct = delta_metrics[delta_key]

        # Create bar chart
        x = np.arange(2)
        bars = ax.bar(x, [first_val, last_val], width=0.6, alpha=0.8)

        # Color bars based on improvement
        # For collision/offroad, decrease is good (green), increase is bad (red)
        # For score/return, increase is good (green), decrease is bad (red)
        if metric_key in ["collision_rate", "offroad_rate"]:
            bars[0].set_color("gray")
            bars[1].set_color("green" if delta_pct < 0 else "red")
        else:
            bars[0].set_color("gray")
            bars[1].set_color("green" if delta_pct > 0 else "red")

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, [first_val, last_val])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Add delta percentage annotation
        mid_x = 0.5
        mid_y = max(first_val, last_val) * 0.5
        arrow_props = dict(
            arrowstyle="->",
            lw=2,
            color="green"
            if (delta_pct > 0 and metric_key in ["score", "episode_return"])
            or (delta_pct < 0 and metric_key in ["collision_rate", "offroad_rate"])
            else "red",
        )

        ax.annotate(
            f"{delta_pct:+.1f}%",
            xy=(1, last_val),
            xytext=(mid_x, mid_y),
            fontsize=14,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=arrow_props,
        )

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(["First Scenario\n(0-shot)", "Last Scenario\n(Adapted)"], fontsize=11)
        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_name}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path.replace(".json", "_adaptive_metrics.png"), dpi=300, bbox_inches="tight")
    print(f"\nAdaptive metrics plot saved to {output_path.replace('.json', '_adaptive_metrics.png')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32, help="Max parallel rollouts per batch")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument(
        "--condition-type",
        type=str,
        default="none",
        choices=["none", "reward", "entropy", "discount", "all"],
        help="Conditioning type (none, reward, entropy, discount, all)",
    )
    parser.add_argument("--output", type=str, default="eval_human_logs.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-controlled-agents", type=int, default=-1
    )  ## needs to be 1 if you want human logs, -1 if you want Self Play
    parser.add_argument("--adaptive-driving-agent", type=int, default=0, help="Enable adaptive driving agent")
    parser.add_argument("--k-scenarios", type=int, default=1, help="Number of scenarios (default 1 for non-adaptive)")
    parser.add_argument("--dynamics-model", type=str, default="classic")
    parser.add_argument(
        "--human-replay",
        action="store_true",
        help="Enable human replay mode (other agents follow logged trajectories instead of neural network co-players)",
    )
    args = parser.parse_args()

    num_batches = (args.num_rollouts + args.batch_size - 1) // args.batch_size

    print(f"Evaluation Configuration:")
    print(f"  Policy: {args.policy_path}")
    print(f"  Conditioning: {args.condition_type}")
    print(f"  Num maps: {args.num_maps}")
    print(f"  Total rollouts: {args.num_rollouts}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {num_batches}")
    print(f"  Num agents per env: {args.num_agents}")
    print(f"  Adaptive agent: {bool(args.adaptive_driving_agent)}")
    print(f"  K scenarios: {args.k_scenarios}")
    print(f"  Human replay mode: {args.human_replay}")
    print(f"  Output: {args.output}")
    print(f"  Dynamics Model: {args.dynamics_model}")

    # Load policy
    print("Loading policy...")
    env_name = "puffer_adaptive_drive" if args.adaptive_driving_agent else "puffer_drive"
    make_env = env_creator(env_name)
    # Build temp env kwargs
    temp_env_kwargs = {
        "num_agents": 64,
        "num_maps": args.num_maps,
        "scenario_length": 91,
        "adaptive_driving_agent": args.adaptive_driving_agent,
        "k_scenarios": args.k_scenarios,
        "dynamics_model": args.dynamics_model,
    }
    # Human replay mode: disable co-players, use human trajectories
    if args.human_replay:
        temp_env_kwargs["human_replay_mode"] = True
        temp_env_kwargs["max_controlled_agents"] = 1
        # Use control_vehicles with max_controlled_agents=1 to control one agent
        # and let others follow human trajectories (expert replay)
        temp_env_kwargs["control_mode"] = "control_vehicles"

    # For adaptive agents, report metrics for all scenarios (not just 0-shot)
    if args.adaptive_driving_agent:
        temp_env_kwargs["report_all_scenarios"] = True

    temp_env = make_env(**temp_env_kwargs)

    base_policy = Drive(temp_env, input_size=64, hidden_size=256)
    policy = Recurrent(temp_env, base_policy, input_size=256, hidden_size=256).to(args.device)
    state_dict = torch.load(args.policy_path, map_location=args.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    temp_env.close()
    print("Policy loaded successfully\n")

    # Run evaluation in batches
    all_returns = []
    all_scenario_metrics = []  # Track metrics per scenario for adaptive agents
    all_metrics = []

    env_kwargs = {
        "num_agents": args.num_agents,
        "num_maps": args.num_maps,
        "max_controlled_agents": args.max_controlled_agents,
        "report_interval": 1,
        "scenario_length": 91,
        "adaptive_driving_agent": args.adaptive_driving_agent,
        "k_scenarios": args.k_scenarios,
        "dynamics_model": args.dynamics_model,
    }

    # Human replay mode: disable co-players, use human trajectories
    if args.human_replay:
        env_kwargs["human_replay_mode"] = True
        env_kwargs["max_controlled_agents"] = args.max_controlled_agents if args.max_controlled_agents > 0 else 1
        # Use control_vehicles with max_controlled_agents to control N agents
        # and let others follow human trajectories (expert replay)
        env_kwargs["control_mode"] = "control_vehicles"

    # For adaptive agents, report metrics for all scenarios (not just 0-shot)
    if args.adaptive_driving_agent:
        env_kwargs["report_all_scenarios"] = True

    print("Running evaluation...")
    for batch_idx in range(num_batches):
        batch_rollouts = min(args.batch_size, args.num_rollouts - batch_idx * args.batch_size)

        print(f"Batch {batch_idx + 1}/{num_batches} ({batch_rollouts} rollouts)")

        # Find largest valid num_workers (divisor of batch_rollouts)
        max_workers = min(args.num_workers, batch_rollouts)
        while batch_rollouts % max_workers != 0:
            max_workers -= 1

        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=env_kwargs,
            backend=pufferlib.vector.Multiprocessing,
            num_envs=batch_rollouts,
            num_workers=max_workers,
        )

        obs, _ = vecenv.reset()
        total_agents = obs.shape[0]

        state = {
            "lstm_h": torch.zeros(total_agents, policy.hidden_size, device=args.device),
            "lstm_c": torch.zeros(total_agents, policy.hidden_size, device=args.device),
        }

        batch_infos = []
        total_reward = np.zeros(total_agents)
        scenario_infos = []  # Track infos per scenario

        with torch.no_grad():
            # Run through all scenarios (1 for non-adaptive, k for adaptive)
            for scenario in range(args.k_scenarios):
                scenario_info_list = []
                desc = f"  Scenario {scenario + 1}/{args.k_scenarios}" if args.k_scenarios > 1 else "  Steps"
                for t in tqdm(range(91), desc=desc, ncols=80, leave=False):
                    obs_t = torch.as_tensor(obs, device=args.device)
                    logits, _ = policy.forward_eval(obs_t, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits)

                    obs, reward, done, trunc, info = vecenv.step(action.cpu().numpy())
                    total_reward += reward

                    if info:
                        valid_infos = [inf for inf in info if "score" in inf]
                        batch_infos.extend(valid_infos)
                        scenario_info_list.extend(valid_infos)

                # Store per-scenario infos
                if args.adaptive_driving_agent:
                    scenario_infos.append(scenario_info_list)

        vecenv.close()

        # Aggregate batch metrics
        num_infos = len(batch_infos) or 1
        batch_metrics = {
            "score": sum(info.get("score", 0) for info in batch_infos) / num_infos,
            "collision_rate": sum(info.get("collision_rate", 0) for info in batch_infos) / num_infos,
            "offroad_rate": sum(info.get("offroad_rate", 0) for info in batch_infos) / num_infos,
            "completion_rate": sum(info.get("completion_rate", 0) for info in batch_infos) / num_infos,
            "dnf_rate": sum(info.get("dnf_rate", 0) for info in batch_infos) / num_infos,
            "avg_collisions_per_agent": sum(info.get("avg_collisions_per_agent", 0) for info in batch_infos)
            / num_infos,
            "avg_offroad_per_agent": sum(info.get("avg_offroad_per_agent", 0) for info in batch_infos) / num_infos,
        }

        rollout_rewards = total_reward.reshape(batch_rollouts, args.num_agents)
        rollout_returns = rollout_rewards.mean(axis=1)

        all_returns.extend(rollout_returns.tolist())
        all_metrics.append(batch_metrics)

        # Store scenario-specific metrics for adaptive agents
        if args.adaptive_driving_agent:
            batch_scenario_metrics = []
            for scenario_info_list in scenario_infos:
                num_scenario_infos = len(scenario_info_list) or 1
                scenario_metrics = {
                    "score": sum(info.get("score", 0) for info in scenario_info_list) / num_scenario_infos,
                    "collision_rate": sum(info.get("collision_rate", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "offroad_rate": sum(info.get("offroad_rate", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "completion_rate": sum(info.get("completion_rate", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "dnf_rate": sum(info.get("dnf_rate", 0) for info in scenario_info_list) / num_scenario_infos,
                    "num_goals_reached": sum(info.get("num_goals_reached", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "lane_alignment_rate": sum(info.get("lane_alignment_rate", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "avg_displacement_error": sum(info.get("avg_displacement_error", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                    "episode_return": sum(info.get("episode_return", 0) for info in scenario_info_list)
                    / num_scenario_infos,
                }
                # Compute perf (score without collision before goal)
                scenario_metrics["perf"] = scenario_metrics["score"]
                batch_scenario_metrics.append(scenario_metrics)

            all_scenario_metrics.append(batch_scenario_metrics)

    # Aggregate across all batches
    all_returns = np.array(all_returns)

    # Divide by k_scenarios since we accumulated across all scenarios
    all_returns = all_returns / args.k_scenarios

    metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    results = {
        "avg_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "se_return": float(np.std(all_returns) / np.sqrt(len(all_returns))),  # Standard error
        **{k: float(v) for k, v in metrics.items()},
    }

    # Compute adaptive delta metrics from scenario metrics
    first_scenario_metrics = None
    last_scenario_metrics = None

    if args.adaptive_driving_agent and len(all_scenario_metrics) > 0:
        # Aggregate scenario metrics across all batches
        # all_scenario_metrics is a list of [batch][scenario] metrics
        # We need to compute average for each scenario across all batches

        aggregated_scenario_metrics = []
        for scenario_idx in range(args.k_scenarios):
            scenario_metrics_list = [batch[scenario_idx] for batch in all_scenario_metrics]

            # Average each metric across batches for this scenario
            avg_scenario_metrics = {}
            for key in scenario_metrics_list[0].keys():
                avg_scenario_metrics[key] = np.mean([m[key] for m in scenario_metrics_list])

            aggregated_scenario_metrics.append(avg_scenario_metrics)

        # Get first and last scenario metrics
        first_scenario_metrics = aggregated_scenario_metrics[0]
        last_scenario_metrics = aggregated_scenario_metrics[-1]

        # Helper function to compute delta percentage
        def compute_delta_percent(first_val, last_val):
            return (last_val - first_val) 

        # Compute all delta metrics
        results["ada_delta_completion_rate"] = compute_delta_percent(
            first_scenario_metrics["completion_rate"], last_scenario_metrics["completion_rate"]
        )
        results["ada_delta_score"] = compute_delta_percent(
            first_scenario_metrics["score"], last_scenario_metrics["score"]
        )
        results["ada_delta_perf"] = compute_delta_percent(first_scenario_metrics["perf"], last_scenario_metrics["perf"])
        results["ada_delta_collision_rate"] = compute_delta_percent(
            first_scenario_metrics["collision_rate"], last_scenario_metrics["collision_rate"]
        )
        results["ada_delta_offroad_rate"] = compute_delta_percent(
            first_scenario_metrics["offroad_rate"], last_scenario_metrics["offroad_rate"]
        )
        results["ada_delta_num_goals_reached"] = compute_delta_percent(
            first_scenario_metrics["num_goals_reached"], last_scenario_metrics["num_goals_reached"]
        )
        results["ada_delta_dnf_rate"] = compute_delta_percent(
            first_scenario_metrics["dnf_rate"], last_scenario_metrics["dnf_rate"]
        )
        results["ada_delta_lane_alignment_rate"] = compute_delta_percent(
            first_scenario_metrics["lane_alignment_rate"], last_scenario_metrics["lane_alignment_rate"]
        )
        results["ada_delta_avg_displacement_error"] = compute_delta_percent(
            first_scenario_metrics["avg_displacement_error"], last_scenario_metrics["avg_displacement_error"]
        )
        results["ada_delta_episode_return"] = compute_delta_percent(
            first_scenario_metrics["episode_return"], last_scenario_metrics["episode_return"]
        )

        # Store first and last scenario values for reporting
        results["first_scenario_score"] = float(first_scenario_metrics["score"])
        results["first_scenario_collision_rate"] = float(first_scenario_metrics["collision_rate"])
        results["first_scenario_offroad_rate"] = float(first_scenario_metrics["offroad_rate"])
        results["first_scenario_episode_return"] = float(first_scenario_metrics["episode_return"])

        results["last_scenario_score"] = float(last_scenario_metrics["score"])
        results["last_scenario_collision_rate"] = float(last_scenario_metrics["collision_rate"])
        results["last_scenario_offroad_rate"] = float(last_scenario_metrics["offroad_rate"])
        results["last_scenario_episode_return"] = float(last_scenario_metrics["episode_return"])

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  Return: {results['avg_return']:.2f} ± {results['se_return']:.2f} (SE)")
    print(f"  Score: {results['score']:.3f}")
    print(f"  Completion: {results['completion_rate']:.3f}")
    print(f"  Collision: {results['collision_rate']:.3f}")
    print(f"  Collision per agent: {results['avg_collisions_per_agent']:.3f}")
    print(f"  Offroad: {results['offroad_rate']:.3f}")

    if args.adaptive_driving_agent:
        print(f"\n0-Shot Performance (First Scenario):")
        print(f"  Score: {results['first_scenario_score']:.3f}")
        print(f"  Collision: {results['first_scenario_collision_rate']:.3f}")
        print(f"  Offroad: {results['first_scenario_offroad_rate']:.3f}")
        print(f"  Return: {results['first_scenario_episode_return']:.2f}")

        print(f"\nAdapted Performance (Last Scenario):")
        print(f"  Score: {results['last_scenario_score']:.3f}")
        print(f"  Collision: {results['last_scenario_collision_rate']:.3f}")
        print(f"  Offroad: {results['last_scenario_offroad_rate']:.3f}")
        print(f"  Return: {results['last_scenario_episode_return']:.2f}")

        print(f"\nAdaptive Metrics (Delta %):")
        print(f"  Score: {results['ada_delta_score']:.2f}%")
        print(f"  Collision rate: {results['ada_delta_collision_rate']:.2f}%")
        print(f"  Offroad rate: {results['ada_delta_offroad_rate']:.2f}%")
        print(f"  Episode return: {results['ada_delta_episode_return']:.2f}%")

        # Generate visualization
        plot_adaptive_metrics(first_scenario_metrics, last_scenario_metrics, results, args.output)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
