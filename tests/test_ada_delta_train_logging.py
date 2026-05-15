"""Contract: training-time wandb logs include ada_delta_trial_K_minus_0
under goal_behavior=3 for every K in 1..max_trials_per_episode-1.

Pre-refactor, those keys only appeared in eval-time HumanReplayEvaluator
output (every 40 epochs). Now they appear every `report_interval` ticks
during training, so adaptation can be tracked live in wandb.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(k, scenario_length=10, goal_radius=200.0):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=4,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=3,
        max_trials_per_episode=k,
        per_trial_timeout=scenario_length,
        goal_radius=goal_radius,
        report_interval=10,
    )


def _step(env):
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    env.step(actions)
    return env.observations


def test_trial_k_score_keys_emitted():
    """vec_log emits trial_0_score..trial_{N_TRIAL_K_SLOTS-1}_score under gb=3."""
    from pufferlib.ocean.drive import binding

    env = _make_env(k=4)
    env.reset(seed=42)
    log = None
    for _ in range(200):
        _step(env)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            break
    assert log and log.get("n", 0) > 0, "no episode emission within 200 steps"
    for k in range(8):  # N_TRIAL_K_SLOTS
        assert f"trial_{k}_score" in log, f"missing trial_{k}_score in log keys: {sorted(log.keys())}"
    env.close()


def test_ada_delta_keys_injected_in_training_step():
    """After env.step(), the info dict at report_interval boundaries
    contains ada_delta_trial_K_minus_0 keys for K in 1..max_trials-1."""
    from pufferlib.ocean.drive import binding

    env = _make_env(k=4, scenario_length=10)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    info_with_deltas = None
    for _ in range(200):
        _, _, _, _, info_list = env.step(actions)
        for info in info_list:
            if isinstance(info, dict) and any(k.startswith("ada_delta_trial_") for k in info):
                info_with_deltas = info
                break
        if info_with_deltas:
            break
    assert info_with_deltas is not None, "no ada_delta_trial_K_minus_0 keys emitted in 200 steps"
    for k in (1, 2, 3):
        assert f"ada_delta_trial_{k}_minus_0" in info_with_deltas, (
            f"missing ada_delta_trial_{k}_minus_0 in info: "
            f"{sorted(k for k in info_with_deltas if k.startswith('ada_delta'))}"
        )
    # Trial 0 delta would always be 0; we don't emit it.
    assert "ada_delta_trial_0_minus_0" not in info_with_deltas
    env.close()


def test_ada_delta_value_matches_trial_score_subtraction():
    """ada_delta_trial_K_minus_0 == trial_K_score - trial_0_score."""
    from pufferlib.ocean.drive import binding

    env = _make_env(k=4, scenario_length=10, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    for _ in range(200):
        _, _, _, _, info_list = env.step(actions)
        for info in info_list:
            if isinstance(info, dict) and "trial_0_score" in info and "ada_delta_trial_1_minus_0" in info:
                expected = info["trial_1_score"] - info["trial_0_score"]
                assert abs(info["ada_delta_trial_1_minus_0"] - expected) < 1e-6
                env.close()
                return
    env.close()
    raise AssertionError("never emitted a log with both trial_0_score and ada_delta_trial_1_minus_0")


def test_non_trial_modes_no_trial_delta_keys():
    """gb=0/1/2: ada_delta_trial_* keys must NOT appear (only ada_delta_<metric> from scenario boundaries)."""
    from pufferlib.ocean.drive import Drive

    env = Drive(
        num_agents=4,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=20,
        ini_file=INI,
        goal_behavior=0,
        report_interval=10,
    )
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    for _ in range(100):
        _, _, _, _, info_list = env.step(actions)
        for info in info_list:
            if isinstance(info, dict):
                bad = [k for k in info if k.startswith("ada_delta_trial_")]
                assert not bad, f"gb=0 leaked trial-delta keys: {bad}"
    env.close()


if __name__ == "__main__":
    test_trial_k_score_keys_emitted()
    test_ada_delta_keys_injected_in_training_step()
    test_ada_delta_value_matches_trial_score_subtraction()
    test_non_trial_modes_no_trial_delta_keys()
    print("test_ada_delta_train_logging: PASS")
