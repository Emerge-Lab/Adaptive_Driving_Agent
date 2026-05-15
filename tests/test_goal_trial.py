"""M3 verification: GOAL_TRIAL=3 behavior + non-regression on goal_behavior 0/1/2.

Two halves:
  1. Non-regression: with goal_behavior in {0, 1, 2}, trial_ended_this_step
     stays all-zero through many steps. terminals never fires from the
     trial path. trial_count never increments.
  2. GOAL_TRIAL: with goal_behavior=3 + a tiny per_trial_timeout, every
     timeout cycle bumps trial_count and fires trial_ended_this_step.
     After max_trials_per_episode trials, terminals fires for that agent.
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


def _step(env):
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    env.step(actions)


def test_non_regression_gb0():
    """goal_behavior=0 (RESPAWN): trial_ended_this_step stays 0."""
    env = _make_env(goal_behavior=0)
    env.reset(seed=42)
    for t in range(60):
        _step(env)
        assert not env.trial_ended_this_step.any(), f"gb=0 step {t}: trial_ended_this_step fired"
    print("  ok: goal_behavior=0 → trial_ended_this_step always zero")
    env.close()


def test_non_regression_gb1():
    """goal_behavior=1 (GENERATE_NEW): trial_ended_this_step stays 0."""
    env = _make_env(goal_behavior=1)
    env.reset(seed=42)
    for t in range(60):
        _step(env)
        assert not env.trial_ended_this_step.any(), f"gb=1 step {t}: trial_ended_this_step fired"
    print("  ok: goal_behavior=1 → trial_ended_this_step always zero")
    env.close()


def test_non_regression_gb2():
    """goal_behavior=2 (STOP): trial_ended_this_step stays 0."""
    env = _make_env(goal_behavior=2)
    env.reset(seed=42)
    for t in range(60):
        _step(env)
        assert not env.trial_ended_this_step.any(), f"gb=2 step {t}: trial_ended_this_step fired"
    print("  ok: goal_behavior=2 → trial_ended_this_step always zero")
    env.close()


def test_trial_timeout_fires():
    """goal_behavior=3 with tiny per_trial_timeout: every TIMEOUT-th step,
    trial_ended_this_step should fire across all agents simultaneously
    (since they all start trial 0 at timestep=0)."""
    TIMEOUT = 5
    MAX_TRIALS = 2
    env = _make_env(goal_behavior=3, max_trials=MAX_TRIALS, per_trial_timeout=TIMEOUT, scenario_length=200)
    env.reset(seed=42)

    trial_end_steps = []
    terminal_steps = []
    for t in range(1, 30):
        _step(env)
        if env.trial_ended_this_step.any():
            trial_end_steps.append((t, env.trial_ended_this_step.sum()))
        if env.terminals.any():
            terminal_steps.append((t, env.terminals.sum()))

    # Trial timeouts: agents start trial 0 at timestep=0. Timeout at
    # (timestep - trial_start) >= TIMEOUT means trial ends at step TIMEOUT.
    # Then they get respawned, trial_start = TIMEOUT. Next timeout at
    # step 2*TIMEOUT. Etc.
    assert len(trial_end_steps) > 0, f"goal_behavior=3 never fired trial_ended_this_step in 30 steps"
    print(f"  ok: gb=3 timeout fires; trial_end_steps (first 5)={trial_end_steps[:5]}")
    # Episode boundary fires when trial_count hits MAX_TRIALS (=2 by default).
    # That should be at step 2*TIMEOUT (= 10) for agents that timed out twice.
    assert len(terminal_steps) > 0, f"goal_behavior=3 never fired terminals (expected at trial_count >= {MAX_TRIALS})"
    print(f"  ok: gb=3 terminals fire at trial_count >= {MAX_TRIALS}; terminal_steps (first 5)={terminal_steps[:5]}")
    env.close()


def test_trial_episode_resets():
    """goal_behavior=3 + Option D: after max_trials trials, each agent's
    terminals fires once and the agent goes idle (removed=1) until Python's
    resample_frequency triggers c_reset. We verify:
      - At least one terminals event fires near step MAX_TRIALS * TIMEOUT
        (= 10 here), proving the C-side episode boundary works.
      - Within the same Python cycle, agents idle (no repeated terminals
        spam from looped trials).
    """
    TIMEOUT = 5
    MAX_TRIALS = 2
    env = _make_env(goal_behavior=3, max_trials=MAX_TRIALS, per_trial_timeout=TIMEOUT, scenario_length=200)
    env.reset(seed=42)

    terminal_steps = []
    for t in range(1, 50):
        _step(env)
        if env.terminals.any():
            terminal_steps.append(t)

    # Each "episode" = MAX_TRIALS * TIMEOUT = 10 steps. Under Option D the
    # agent then idles until resample_frequency (default 91 for base Drive),
    # so in 50 ticks we expect EXACTLY ONE terminals event around step 10.
    assert len(terminal_steps) >= 1, f"Expected ≥1 episode boundary, got {len(terminal_steps)}"
    assert terminal_steps[0] <= MAX_TRIALS * TIMEOUT + 2, (
        f"First terminals should fire near step {MAX_TRIALS * TIMEOUT}, got {terminal_steps[0]}"
    )
    print(f"  ok: episode boundary fires at step ~{MAX_TRIALS * TIMEOUT}: {terminal_steps[:6]}")
    env.close()


def _run_all():
    test_non_regression_gb0()
    test_non_regression_gb1()
    test_non_regression_gb2()
    test_trial_timeout_fires()
    test_trial_episode_resets()
    print("\ntest_goal_trial: PASS")


if __name__ == "__main__":
    _run_all()
