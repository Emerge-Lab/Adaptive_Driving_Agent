import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import pufferlib
import pufferlib.vector
from pufferlib.ocean import env_creator
from pufferlib.ocean.torch import Drive, Recurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32, help="Max parallel rollouts per batch")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument("--condition-type", type=str, default="none",
                        choices=["none", "reward", "entropy", "discount", "all"],
                        help="Conditioning type (none, reward, entropy, discount, all)")
    parser.add_argument("--output", type=str, default="eval_human_logs.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-controlled-agents", type=int, default=-1)
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
    print(f"  Output: {args.output}\n")

    # Load policy
    print("Loading policy...")
    make_env = env_creator("puffer_drive")
    temp_env = make_env(num_agents=64, num_maps=args.num_maps, scenario_length = 91, condition_type = args.condition_type);

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
    all_metrics = []

    env_kwargs = {
        "num_agents": args.num_agents,
        "num_maps": args.num_maps,
        "max_controlled_agents": args.max_controlled_agents,
        "report_interval": 1,
        "scenario_length": 91,
        "condition_type": args.condition_type,
    }

    print("Running evaluation...")
    for batch_idx in range(num_batches):
        batch_rollouts = min(args.batch_size, args.num_rollouts - batch_idx * args.batch_size)

        print(f"Batch {batch_idx + 1}/{num_batches} ({batch_rollouts} rollouts)")

        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=env_kwargs,
            backend=pufferlib.vector.Multiprocessing,
            num_envs=batch_rollouts,
            num_workers=min(args.num_workers, batch_rollouts),
        )

        obs, _ = vecenv.reset()
        total_agents = obs.shape[0]

        state = {
            "lstm_h": torch.zeros(total_agents, policy.hidden_size, device=args.device),
            "lstm_c": torch.zeros(total_agents, policy.hidden_size, device=args.device),
        }

        batch_infos = []
        total_reward = np.zeros(total_agents)

        with torch.no_grad():
            for t in tqdm(range(91), desc="  Steps", ncols=80, leave=False):
                obs_t = torch.as_tensor(obs, device=args.device)
                logits, _ = policy.forward_eval(obs_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)

                obs, reward, done, trunc, info = vecenv.step(action.cpu().numpy())
                total_reward += reward

                if info:
                    batch_infos.extend(info)

        vecenv.close()

        # Aggregate batch metrics
        num_infos = len(batch_infos) or 1
        batch_metrics = {
            "score": sum(info.get("score", 0) for info in batch_infos) / num_infos,
            "collision_rate": sum(info.get("collision_rate", 0) for info in batch_infos) / num_infos,
            "offroad_rate": sum(info.get("offroad_rate", 0) for info in batch_infos) / num_infos,
            "completion_rate": sum(info.get("completion_rate", 0) for info in batch_infos) / num_infos,
            "dnf_rate": sum(info.get("dnf_rate", 0) for info in batch_infos) / num_infos,
            "avg_collisions_per_agent": sum(info.get("avg_collisions_per_agent", 0) for info in batch_infos) / num_infos,
            "avg_offroad_per_agent": sum(info.get("avg_offroad_per_agent", 0) for info in batch_infos) / num_infos,
        }

        rollout_rewards = total_reward.reshape(batch_rollouts, args.num_agents)
        rollout_returns = rollout_rewards.mean(axis=1)

        all_returns.extend(rollout_returns.tolist())
        all_metrics.append(batch_metrics)

    # Aggregate across all batches
    all_returns = np.array(all_returns)
    metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}

    results = {
        "avg_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        **{k: float(v) for k, v in metrics.items()}
    }

    # with open(args.output, "w") as f:
    #     json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Score: {results['score']:.3f}")
    print(f"  Completion: {results['completion_rate']:.3f}")
    print(f"  Collision: {results['collision_rate']:.3f}")
    print(f"  Collision per agent: {results['avg_collisions_per_agent']:.3f}")
    print(f"  Offroad: {results['offroad_rate']:.3f}")
    # print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
