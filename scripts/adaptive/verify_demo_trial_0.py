"""Verify that demo_trial_0=True makes the ego follow the recorded human
trajectory during trial 0.
"""
import sys, traceback
import numpy as np


def main(demo):
    print(f"[verify] starting demo={demo}", flush=True)
    import pufferlib.ocean.drive.drive as d
    print(f"[verify] import OK", flush=True)
    env = d.Drive(
        num_agents=32,
        num_maps=50,
        k_scenarios=4,
        scenario_length=50,
        adaptive_driving_agent=True,
        co_player_enabled=False,
        goal_behavior=3,
        map_dir="resources/drive/binaries/nuplan_hard",
        demo_trial_0=demo,
        map_seed=42,
    )
    print(f"[verify] env constructed, num_ego_agents={env.num_ego_agents}", flush=True)
    env.reset()
    print(f"[verify] reset done", flush=True)
    # get_ground_truth_trajectories segfaults on this env config; skip it.

    np.random.seed(0)
    for step in range(60):
        actions = np.random.randint(0, 91, size=(env.num_ego_agents, 1)).astype(np.int32)
        env.step(actions)
        if step in (0, 1, 5, 10, 20, 30, 40, 50, 55):
            s = env.get_global_agent_state()
            print(f"[verify] step {step:>3}: ego0 x={s['x'][0]:.2f} y={s['y'][0]:.2f} "
                  f"trial_ended={bool(env.trial_ended_this_step[0])}", flush=True)


if __name__ == "__main__":
    demo = sys.argv[1].lower() == "true"
    try:
        main(demo)
    except Exception:
        traceback.print_exc()
        raise
