import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import pufferlib
import pufferlib.vector
from pufferlib.ocean import env_creator
from pufferlib.ocean.torch import Drive, Recurrent, Transformer
from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator
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
    print("Beginning human evaluations using HumanReplayEvaluator")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--policy-architecture", type=str, default="Recurrent")
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument("--output", type=str, default="eval_human_logs.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-controlled-agents", type=int, default=1)
    parser.add_argument("--adaptive-driving-agent", type=int, default=0)
    parser.add_argument("--k-scenarios", type=int, default=1)
    parser.add_argument("--dynamics-model", type=str, default="classic")
    parser.add_argument("--human-replay", action="store_true")
    args_parsed = parser.parse_args()

    print(f"Evaluation Configuration:")
    print(f"  Policy: {args_parsed.policy_path}")
    print(f"  Policy Architecture: {args_parsed.policy_architecture}")
    print(f"  Num maps: {args_parsed.num_maps}")
    print(f"  Total rollouts: {args_parsed.num_rollouts}")
    print(f"  Num agents per env: {args_parsed.num_agents}")
    print(f"  Adaptive agent: {bool(args_parsed.adaptive_driving_agent)}")
    print(f"  K scenarios: {args_parsed.k_scenarios}")
    print(f"  Dynamics Model: {args_parsed.dynamics_model}")
    print(f"  Output: {args_parsed.output}\n")

    # Build args dict in the format expected by HumanReplayEvaluator
    env_name = "puffer_adaptive_drive" if args_parsed.adaptive_driving_agent else "puffer_drive"
    make_env = env_creator(env_name)

    scenario_length = 91
    context_length = args_parsed.k_scenarios * scenario_length

    args = {
        "train": {
            "device": args_parsed.device,
            "use_rnn": args_parsed.policy_architecture == "Recurrent",
            "policy_architecture": args_parsed.policy_architecture,
            "context_window": context_length,
        },
        "env": {
            "num_agents": args_parsed.num_agents,
            "num_maps": args_parsed.num_maps,
            "scenario_length": scenario_length,
            "adaptive_driving_agent": args_parsed.adaptive_driving_agent,
            "k_scenarios": args_parsed.k_scenarios,
            "dynamics_model": args_parsed.dynamics_model,
            "max_controlled_agents": args_parsed.max_controlled_agents,
            "report_interval": 1,
            "control_mode": "control_vehicles",
            "episode_length": scenario_length,
            "report_all_scenarios": args_parsed.adaptive_driving_agent,
            "dynamics_model": "classic",
            "reward_vehicle_collision": -0.5,
            "reward_offroad_collision": -0.5,
            "reward_goal": 1.0,
            "reward_goal_post_respawn": 0.25,
        },
        "vec": {
            "backend": "PufferEnv",
            "num_envs": 1,
        },
        "eval": {
            "human_replay_control_mode": "control_vehicles",
        },
    }

    if args_parsed.human_replay:
        args["env"]["human_replay_mode"] = True

    # Load policy once
    print("Loading policy...")
    temp_env = make_env(**args["env"])

    if args_parsed.policy_architecture == "Recurrent":
        base_policy = Drive(temp_env, input_size=64, hidden_size=256)
        policy = Recurrent(temp_env, base_policy, input_size=256, hidden_size=256).to(args_parsed.device)
    elif args_parsed.policy_architecture == "Transformer":
        base_policy = Drive(temp_env, input_size=128, hidden_size=256)
        policy = Transformer(
            temp_env,
            base_policy,
            input_size=256,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            context_length=context_length,
        ).to(args_parsed.device)

    state_dict = torch.load(args_parsed.policy_path, map_location=args_parsed.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    temp_env.close()
    print("Policy loaded successfully\n")

    # Create evaluator
    from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

    evaluator = HumanReplayEvaluator(args)

    # Run multiple rollouts
    print(f"Running {args_parsed.num_rollouts} rollouts...")
    all_results = []

    for rollout_idx in tqdm(range(args_parsed.num_rollouts), desc="Rollouts"):
        # Create fresh env for each rollout
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=args["env"],
            backend=pufferlib.vector.Serial,
            num_envs=1,
        )

        # Run single rollout
        results = evaluator.rollout(args, vecenv, policy)
        all_results.append(results)

        vecenv.close()

    # Aggregate results
    print("\nAggregating results...")
    aggregated = {}

    # Get all metric keys from first result
    all_keys = list(all_results[0].keys())
    metric_keys = [k for k in all_keys if not k.startswith("ada_delta")]
    delta_keys = [k for k in all_keys if k.startswith("ada_delta")]

    # Average regular metrics
    for key in metric_keys:
        values = [r.get(key, 0) for r in all_results]
        aggregated[key] = float(np.mean(values))

    # Average delta metrics if present
    if delta_keys:
        for key in delta_keys:
            values = [r.get(key, 0) for r in all_results]
            aggregated[key] = float(np.mean(values))

        # Derive last scenario metrics from first + delta
        # Extract first scenario metrics
        first_scenario_keys = [k for k in metric_keys if k not in ["n"]]

        # Map metric names to their delta counterparts
        metric_to_delta = {
            "score": "ada_delta_score",
            "collision_rate": "ada_delta_collision_rate",
            "offroad_rate": "ada_delta_offroad_rate",
            "completion_rate": "ada_delta_completion_rate",
            "episode_return": "ada_delta_episode_return",
            "dnf_rate": "ada_delta_dnf_rate",
            "lane_alignment_rate": "ada_delta_lane_alignment_rate",
        }

        # Store first scenario metrics
        for metric_name in metric_to_delta.keys():
            if metric_name in aggregated:
                aggregated[f"first_scenario_{metric_name}"] = aggregated[metric_name]

        # Compute last scenario metrics: last = first + delta
        for metric_name, delta_key in metric_to_delta.items():
            if metric_name in aggregated and delta_key in aggregated:
                aggregated[f"last_scenario_{metric_name}"] = aggregated[metric_name] + aggregated[delta_key]

    # Save results
    with open(args_parsed.output, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print results
    if args_parsed.adaptive_driving_agent and delta_keys:
        print(f"\n0-Shot Performance (First Scenario):")
        print(f"  Score: {aggregated.get('first_scenario_score', float('nan')):.3f}")
        print(f"  Collision: {aggregated.get('first_scenario_collision_rate', float('nan')):.3f}")
        print(f"  Offroad: {aggregated.get('first_scenario_offroad_rate', float('nan')):.3f}")
        print(f"  Return: {aggregated.get('first_scenario_episode_return', float('nan')):.2f}")

        print(f"\nAdapted Performance (Last Scenario):")
        print(f"  Score: {aggregated.get('last_scenario_score', float('nan')):.3f}")
        print(f"  Collision: {aggregated.get('last_scenario_collision_rate', float('nan')):.3f}")
        print(f"  Offroad: {aggregated.get('last_scenario_offroad_rate', float('nan')):.3f}")
        print(f"  Return: {aggregated.get('last_scenario_episode_return', float('nan')):.2f}")

        print(f"\nAdaptive Metrics (Delta):")
        print(f"  Score: {aggregated.get('ada_delta_score', float('nan')):.4f}")
        print(f"  Collision rate: {aggregated.get('ada_delta_collision_rate', float('nan')):.4f}")
        print(f"  Offroad rate: {aggregated.get('ada_delta_offroad_rate', float('nan')):.4f}")
        print(f"  Episode return: {aggregated.get('ada_delta_episode_return', float('nan')):.4f}")

    print(f"\nSaved to {args_parsed.output}")
    import sys

    sys.exit(0)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
