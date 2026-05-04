"""M4 verification: GOAL_TRIAL exposes per-trial Log fields via vec_log path.

Direct env->log writes from c_step's GOAL_TRIAL block; vec_log aggregates
across envs. Python sees the new keys in info dicts.

Tests:
  1. goal_behavior in {0,1,2}: the new fields are exposed but stay zero.
  2. goal_behavior=3 with tiny timeout: n_trials_completed grows over
     time; trial_mean_length matches per_trial_timeout (since all
     trials timeout, no goals reached); trial_goal_reach_rate = 0.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(goal_behavior, max_trials=2, per_trial_timeout=None, num_agents=4, scenario_length=91):
    from pufferlib.ocean.drive import Drive

    kwargs = dict(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=goal_behavior,
        max_trials_per_episode=max_trials,
    )
    if per_trial_timeout is not None:
        kwargs["per_trial_timeout"] = per_trial_timeout
    return Drive(**kwargs)


def _drain_logs(env, num_steps):
    """Step env num_steps times; collect any info dicts emitted."""
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    all_infos = []
    for _ in range(num_steps):
        _, _, _, _, info = env.step(actions)
        if info:
            all_infos.extend(info if isinstance(info, list) else [info])
    return all_infos


def test_log_fields_exist_under_all_goal_behaviors():
    """The new keys (n_trials_completed, trial_mean_length, trial_goal_reach_rate)
    should be exposed under every goal_behavior. Under non-TRIAL they're zero."""
    for gb in [0, 1, 2]:
        env = _make_env(goal_behavior=gb, scenario_length=91)
        env.reset(seed=42)
        infos = _drain_logs(env, 200)
        # Find any info that has the new keys
        keys_seen = set()
        for d in infos:
            if isinstance(d, dict):
                keys_seen.update(d.keys())
        for k in ["n_trials_completed", "n_trials_goal_reached", "n_trials_timed_out",
                  "trial_mean_length", "trial_goal_reach_rate"]:
            assert k in keys_seen, f"gb={gb}: key {k} missing from info dicts (saw: {sorted(keys_seen)[:20]})"
        # Under non-TRIAL, all trial fields should be zero
        for d in infos:
            if isinstance(d, dict) and "n_trials_completed" in d:
                assert d["n_trials_completed"] == 0, f"gb={gb}: n_trials_completed={d['n_trials_completed']} (expected 0)"
                assert d["n_trials_goal_reached"] == 0
                assert d["n_trials_timed_out"] == 0
        print(f"  ok: gb={gb}: trial keys exposed and zero")
        env.close()


def test_log_fields_increment_under_goal_trial():
    """gb=3 with timeout=5: trials accumulate, all timeout (no goals reached)."""
    TIMEOUT = 5
    env = _make_env(goal_behavior=3, max_trials=2, per_trial_timeout=TIMEOUT, scenario_length=200)
    env.reset(seed=42)
    infos = _drain_logs(env, 100)

    final_n_completed = 0
    final_goal_reached = 0
    final_timed_out = 0
    final_trial_mean_length = 0
    final_goal_reach_rate = 0
    for d in infos:
        if isinstance(d, dict) and "n_trials_completed" in d and d["n_trials_completed"] > 0:
            final_n_completed = d["n_trials_completed"]
            final_goal_reached = d["n_trials_goal_reached"]
            final_timed_out = d["n_trials_timed_out"]
            final_trial_mean_length = d["trial_mean_length"]
            final_goal_reach_rate = d["trial_goal_reach_rate"]

    assert final_n_completed > 0, f"gb=3: n_trials_completed never grew (final={final_n_completed})"
    # Random policy in 100 steps with timeout=5 unlikely to reach a goal
    assert final_timed_out == final_n_completed, (
        f"expected all trials to timeout (random policy, tiny timeout). "
        f"got n_completed={final_n_completed}, n_timed_out={final_timed_out}"
    )
    assert final_goal_reached == 0, f"unexpected goal reach: {final_goal_reached}"
    # mean trial length should be very close to TIMEOUT (every trial takes
    # exactly TIMEOUT ticks before timing out; small variation possible)
    assert abs(final_trial_mean_length - TIMEOUT) < 1.5, (
        f"trial_mean_length={final_trial_mean_length} not near TIMEOUT={TIMEOUT}"
    )
    assert final_goal_reach_rate == 0, f"goal_reach_rate should be 0 (no goals reached)"
    print(f"  ok: gb=3: n_completed={final_n_completed} all timeout; mean_length={final_trial_mean_length:.1f} ≈ {TIMEOUT}")
    env.close()


def _run_all():
    test_log_fields_exist_under_all_goal_behaviors()
    test_log_fields_increment_under_goal_trial()
    print("\ntest_trial_log_fields: PASS")


if __name__ == "__main__":
    _run_all()
