"""Fix #1: standard metrics (score / episode_length / episode_return /
offroad_rate / collision_rate / dnf_rate / completion_rate / lane_alignment_rate)
must populate under goal_behavior=GOAL_TRIAL=3, not stay at zero.

Previously: `add_log` was suppressed under GOAL_TRIAL (M4 commit) because the
scenario_length early-return in c_step is gated off for trial mode. Only the
trial-specific log fields (n_trials_completed, trial_mean_length, etc.) made
it into env->log.

Now: when an agent's episode ends (trial_count >= max_trials_per_episode),
add_log_one_agent aggregates that agent's per-episode metrics into env->log
and resets per-agent state. vec_log picks them up the next time total_n
crosses num_agents.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(goal_behavior, max_trials=2, per_trial_timeout=10, num_agents=8, scenario_length=200):
    from pufferlib.ocean.drive import Drive

    # report_interval is set very high so drive.py's internal vec_log call
    # (drive.py:1072, fires every report_interval ticks) does NOT consume the
    # log before this test gets a chance to read it. With report_interval=1
    # (default) drive.py drains the log after every step.
    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=goal_behavior,
        max_trials_per_episode=max_trials,
        per_trial_timeout=per_trial_timeout,
        report_interval=10000,
    )


def _drain_until_log(env, max_steps=400):
    """Step with zero actions until vec_log returns a non-empty dict (i.e.
    we've seen enough episodes to emit). Returns the final log dict + step count."""
    from pufferlib.ocean.drive import binding

    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    log = {}
    for step in range(max_steps):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            return log, step
    return log, max_steps


def test_standard_metrics_populate_under_goal_trial():
    """Under goal_behavior=3 with a tight per_trial_timeout, episodes finish
    fast and vec_log should report non-zero standard metrics (specifically
    episode_length, since every trial logs at least 1 step).
    """
    env = _make_env(goal_behavior=3, max_trials=2, per_trial_timeout=8, num_agents=8)
    env.reset(seed=42)
    log, step_count = _drain_until_log(env, max_steps=300)
    assert log and log.get("n", 0) > 0, f"vec_log emitted empty/zero log after {step_count} steps: {log}"
    # Episode length is the most reliable non-zero — every trial increments it.
    assert log.get("episode_length", 0) > 0, (
        f"episode_length should be > 0 under GOAL_TRIAL (step_count={step_count}): {log}"
    )
    # Trial-specific metrics still work (regression guard)
    assert log.get("n_trials_completed", 0) > 0, f"n_trials_completed should be > 0: {log}"
    assert "trial_mean_length" in log, f"trial_mean_length key missing: {log}"
    # Score is allowed to be 0 (depends on goal-reach), but the KEY must exist
    assert "score" in log, f"score key missing: {log}"
    assert "collision_rate" in log, f"collision_rate key missing: {log}"
    assert "offroad_rate" in log, f"offroad_rate key missing: {log}"
    assert "dnf_rate" in log, f"dnf_rate key missing: {log}"
    env.close()


def test_non_trial_modes_unchanged():
    """gb=0/1/2 keep emitting the standard metrics as before — confirm the
    helper-add didn't break the existing add_log path."""
    for gb in (0, 1, 2):
        env = _make_env(goal_behavior=gb, num_agents=8, scenario_length=20)
        env.reset(seed=42)
        log, _ = _drain_until_log(env, max_steps=200)
        assert log and log.get("n", 0) > 0, f"gb={gb}: log empty"
        assert "episode_length" in log
        assert "score" in log
        assert "collision_rate" in log
        # trial-specific keys: present but all zero under non-trial modes
        assert log.get("n_trials_completed", 0) == 0, (
            f"gb={gb}: trial counter should stay zero, got {log.get('n_trials_completed')}"
        )
        env.close()


def test_per_agent_logs_reset_after_episode():
    """Under GOAL_TRIAL, after an agent's episode ends, its per-agent log
    fields (env->logs[i]) must reset so the next episode starts clean.
    Otherwise episode_length would compound across episodes for that agent.

    Indirect probe: run TWO consecutive episode budgets and confirm the
    second vec_log emission has the same scale of metrics as the first
    (within a tolerance). If per-agent reset were missing, the second
    emission would have ~2x larger episode_length.
    """
    env = _make_env(goal_behavior=3, max_trials=2, per_trial_timeout=8, num_agents=8)
    env.reset(seed=42)
    log1, _ = _drain_until_log(env, max_steps=200)
    assert log1 and log1.get("episode_length", 0) > 0
    el1 = log1["episode_length"]
    log2, _ = _drain_until_log(env, max_steps=200)
    assert log2 and log2.get("episode_length", 0) > 0
    el2 = log2["episode_length"]
    # If reset were broken, el2 would be ~el1 + episode budget. With reset,
    # el2 ≈ el1 (give a generous 2x tolerance for variance from goal-reach
    # timing differences).
    assert el2 < 2 * el1, f"episode_length doubled across emissions ({el1} → {el2}); per-agent reset likely broken"
    env.close()


if __name__ == "__main__":
    test_standard_metrics_populate_under_goal_trial()
    test_non_trial_modes_unchanged()
    test_per_agent_logs_reset_after_episode()
    print("All trial standard-metrics tests passed.")
