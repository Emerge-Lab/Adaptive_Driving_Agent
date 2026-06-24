"""Integration test for the GOAL_TRIAL plumbing fix:
trial-boundary events should set `truncations[i] = 1` (so pufferl's GAE sees
them as bootstrap-stop), but leave `terminals[i] = 0` (so KV-cache persists
across trial boundaries for the adaptive ego's in-context adaptation).

This test instantiates a real Drive env in goal_behavior=3 mode, drives it
with a noop policy long enough to hit trial boundaries via per_trial_timeout,
and asserts the resulting (terminals, truncations, trial_ended_this_step)
co-fire pattern.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(goal_behavior, max_trials=2, per_trial_timeout=5, num_agents=4, scenario_length=200):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=goal_behavior,
        max_trials_per_episode=max_trials,
        per_trial_timeout=per_trial_timeout,
    )


def _zero_step(env):
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    return env.step(actions)


def test_trial_boundary_sets_truncations_only():
    """Under GOAL_TRIAL with a per-trial timeout of 5 ticks: at every 5th
    step (the timeout boundary), truncations should fire (per agent) while
    terminals stays zero — until trial_count reaches max_trials_per_episode,
    where terminals fires alongside truncations.

    Plumbing path: C writes trial_ended_this_step[i]=1 in c_step; Drive.step()
    mirrors it onto truncations[i]=1 immediately after binding.vec_step.
    """
    env = _make_env(goal_behavior=3, max_trials=2, per_trial_timeout=5, scenario_length=200)
    env.reset(seed=42)
    fired_trial_no_term = 0
    fired_both = 0
    for t in range(1, 20):
        _zero_step(env)
        te = np.asarray(env.trial_ended_this_step).reshape(-1)
        trunc = np.asarray(env.truncations).reshape(-1)
        term = np.asarray(env.terminals).reshape(-1)
        if te.any():
            # Every trial-end step must mirror onto truncations
            assert (te.astype(bool) <= trunc.astype(bool)).all(), (
                f"step {t}: trial_ended but truncations not set: te={te.tolist()} trunc={trunc.tolist()}"
            )
            if not term.any():
                fired_trial_no_term += 1
            else:
                fired_both += 1
    assert fired_trial_no_term > 0, "Expected at least one trial boundary with truncations only (not terminals)"
    assert fired_both > 0, "Expected at least one episode boundary with both terminals and truncations"
    env.close()


def test_truncations_clear_each_step():
    """The trial-boundary trunc mirror is per-step, not sticky. The step
    immediately after a trial-end should NOT carry truncations forward."""
    env = _make_env(goal_behavior=3, max_trials=3, per_trial_timeout=5, scenario_length=200)
    env.reset(seed=42)
    last_trunc_step = -10
    for t in range(1, 30):
        _zero_step(env)
        te = np.asarray(env.trial_ended_this_step).reshape(-1)
        trunc = np.asarray(env.truncations).reshape(-1)
        if te.any():
            last_trunc_step = t
        elif t == last_trunc_step + 1:
            assert not trunc.any(), (
                f"step {t}: trunc carried over from trial-end step {last_trunc_step}: {trunc.tolist()}"
            )
    env.close()


def test_non_trial_modes_unchanged():
    """goal_behavior in {0,1,2}: truncations must stay zero across many steps
    (no Python-side scenario boundary trigger because the env has no k_eff
    curriculum enabled at default config). Confirms the fix didn't change
    behavior outside GOAL_TRIAL."""
    for gb in (0, 1, 2):
        env = _make_env(goal_behavior=gb)
        env.reset(seed=42)
        for t in range(1, 40):
            _zero_step(env)
            trunc = np.asarray(env.truncations).reshape(-1)
            assert not trunc.any(), f"gb={gb} step {t}: spurious truncation fired: {trunc.tolist()}"
        env.close()


if __name__ == "__main__":
    test_trial_boundary_sets_truncations_only()
    test_truncations_clear_each_step()
    test_non_trial_modes_unchanged()
    print("All GAE-decoupling integration tests passed.")
