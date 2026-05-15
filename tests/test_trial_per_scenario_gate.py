"""Fix #2: under goal_behavior=GOAL_TRIAL=3, the per-scenario logic block
(drive.py:1095) that fires at `tick % scenario_length == 0` is skipped.

Why: trial boundaries are variable-length and driven by trial_ended_this_step,
not by `tick % scenario_length`. Firing partner-reset / map-rotation /
scenario-metric-aggregation at fixed-time scenario boundaries would land
mid-trial.

This test asserts:
  1. Under GOAL_TRIAL, info dicts emitted by step() do NOT contain `scenario_X_*`
     keys (since the per-scenario block is skipped).
  2. Under non-trial modes (gb=0/1/2), `scenario_X_*` keys still appear.
  3. Under GOAL_TRIAL, self.current_scenario stays at 0 (no per-scenario
     advancement) — confirming the gate works.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/adaptive.ini"


def _make_adaptive(k, goal_behavior, scenario_length=50):
    from pufferlib.ocean.drive.adaptive import AdaptiveDrivingAgent

    return AdaptiveDrivingAgent(
        num_agents=8,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        k_scenarios=k,
        dynamics_model="classic",
        goal_behavior=goal_behavior,
        max_trials_per_episode=2,
        per_trial_timeout=0,
        co_player_enabled=False,
    )


def _step_for(env, n_steps):
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    infos = []
    for _ in range(n_steps):
        _, _, _, _, step_infos = env.step(actions)
        infos.extend(step_infos if isinstance(step_infos, list) else [step_infos])
    return infos


def _has_scenario_keys(infos):
    """True if any emitted info dict contains a scenario_X_* key."""
    for info in infos:
        if not isinstance(info, dict):
            continue
        for k in info:
            if isinstance(k, str) and k.startswith("scenario_") and "_" in k[9:]:
                return True
    return False


def test_per_scenario_skipped_under_goal_trial():
    """Under GOAL_TRIAL, no scenario_X_* keys emitted, current_scenario stays 0."""
    env = _make_adaptive(k=2, goal_behavior=3, scenario_length=50)
    env.reset(seed=42)
    initial_scenario = env.current_scenario
    # Step past the scenario_length boundary (would normally trigger per-scenario block)
    infos = _step_for(env, n_steps=120)
    assert env.current_scenario == initial_scenario, (
        f"Under GOAL_TRIAL, current_scenario should not advance "
        f"(initial={initial_scenario}, after 120 steps={env.current_scenario})"
    )
    assert not _has_scenario_keys(infos), (
        f"Under GOAL_TRIAL, no scenario_X_* keys should be emitted: "
        f"saw infos with keys {[list(i.keys()) for i in infos if isinstance(i, dict)][:3]}"
    )
    env.close()


def test_per_scenario_runs_under_non_trial():
    """Under gb=0, the per-scenario block still fires — current_scenario
    advances and scenario_X_* keys appear."""
    env = _make_adaptive(k=2, goal_behavior=0, scenario_length=50)
    env.reset(seed=42)
    initial_scenario = env.current_scenario
    infos = _step_for(env, n_steps=120)
    # current_scenario should have advanced at least once across 50-tick boundaries
    # NOTE: it wraps mod k_scenarios, so after 2 boundaries (100 ticks) it's 0 again.
    # The key assertion: at least one scenario_X_* key was emitted.
    assert _has_scenario_keys(infos), (
        "Under gb=0, expected scenario_X_* keys to be emitted at scenario_length boundary"
    )
    env.close()


def test_resample_frequency_still_fires_under_goal_trial():
    """Under GOAL_TRIAL, the resample_frequency block (map rotation) still
    runs at tick % resample_frequency == 0 — the auto-link sets
    resample_frequency = k_scenarios * scenario_length, so it fires at
    the worst-case episode budget. Test by checking that tick wraps to 0
    after resample_frequency ticks."""
    env = _make_adaptive(k=2, goal_behavior=3, scenario_length=50)
    env.reset(seed=42)
    # resample_frequency = k_scenarios * scenario_length = 100
    _step_for(env, n_steps=99)
    assert env.tick == 99, f"tick should be 99, got {env.tick}"
    _step_for(env, n_steps=1)
    # After 100 ticks, tick wraps to 0 (drive.py:1186)
    assert env.tick == 0, f"After resample_frequency boundary, tick should reset to 0, got {env.tick}"
    env.close()


if __name__ == "__main__":
    test_per_scenario_skipped_under_goal_trial()
    test_per_scenario_runs_under_non_trial()
    test_resample_frequency_still_fires_under_goal_trial()
    print("All per-scenario-gate tests passed.")
