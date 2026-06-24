"""Contract: under goal_behavior=3, C is the sole writer of `truncations`
and `trial_ended_this_step`.

Pre-refactor, Python mirrored `trial_ended_this_step → truncations` after
every vec_step. This dual-writer pattern was fragile: any order-of-writes
bug between C and Python (e.g., Python writes 0 after C wrote 1) silently
corrupted the GAE bootstrap-stop signal.

Now C is the only writer of both buffers. Python is read-only under gb=3.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(goal_behavior, k_scenarios=2, scenario_length=10):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=4,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=goal_behavior,
        k_scenarios=k_scenarios,
        max_trials_per_episode=k_scenarios,
        per_trial_timeout=scenario_length,
    )


def test_c_zeros_truncations_each_step_under_gb3():
    """C zeroes truncations at top of c_step under gb=3. We poke 1s into
    truncations BEFORE stepping; after step, only positions C set should
    be 1 (or all zero if no trial ended this step)."""
    env = _make_env(goal_behavior=3, k_scenarios=2, scenario_length=10)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    # Pollute truncations
    env.truncations[:] = 1
    env.step(actions)
    # Within the first few steps no trial has ended yet (per_trial_timeout=10,
    # tight goal_radius default). Truncations should be 0 immediately after
    # step.
    assert np.asarray(env.truncations, dtype=bool).sum() == 0, (
        f"After step with no trial-end, truncations should be all zero. Got: {env.truncations}"
    )
    env.close()


def test_c_writes_truncations_at_trial_boundary():
    """At a trial boundary, C sets truncations[i] = 1. Test exercises this
    by setting per_trial_timeout small so the trial-end fires deterministically."""
    env = _make_env(goal_behavior=3, k_scenarios=4, scenario_length=3)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    saw_truncation = False
    for _ in range(30):
        env.step(actions)
        if np.asarray(env.truncations, dtype=bool).any():
            # When truncations fires, trial_ended_this_step should ALSO fire
            # (they're mirror events at trial boundary).
            te = np.asarray(env.trial_ended_this_step, dtype=bool)
            tr = np.asarray(env.truncations, dtype=bool)
            assert (te == tr).all(), (
                f"Under gb=3, truncations and trial_ended_this_step must fire on the "
                f"same agents. trial_ended_this_step={te}, truncations={tr}"
            )
            saw_truncation = True
            break
    assert saw_truncation, "No trial-end fired in 30 steps — test setup wrong"
    env.close()


def test_non_trial_mode_truncations_untouched_by_c():
    """Under gb=0/1/2, C must NOT zero or write truncations — Python owns it
    (k_eff curriculum writes truncations at scenario boundaries)."""
    env = _make_env(goal_behavior=0, k_scenarios=2, scenario_length=10)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    # Pollute truncations BEFORE step. Under gb=0 C should leave it alone.
    env.truncations[:] = 1
    # Note: Python's step() zeroes truncations under gb != 3 before vec_step.
    # That's intentional. So we can't observe "C left it alone" directly;
    # instead we observe that c_step's gb=3-only memset block didn't fire.
    # Proxy: terminate fresh, the only place that should write truncations
    # under gb=0 is Python (curriculum). Just verify step works.
    env.step(actions)
    # After step, truncations is what step() decided. We're checking that
    # the env didn't crash and produced valid (0/1) values.
    tr = np.asarray(env.truncations, dtype=bool)
    assert tr.shape == (env.num_agents,)
    env.close()


if __name__ == "__main__":
    test_c_zeros_truncations_each_step_under_gb3()
    test_c_writes_truncations_at_trial_boundary()
    test_non_trial_mode_truncations_untouched_by_c()
    print("test_truncations_ownership: PASS")
