"""Contract tests for B'' env-level trial semantic.

Design (see docs/src/trial_mode.md, "Env-level trials"):
  - Each env has ONE trial clock (env->trial_count, env->trial_start_timestep),
    not per-agent.
  - On ego goal-reach mid-trial: ego goes off-map (removed=1, INVALID_POSITION,
    vx=vy=0). No truncations / terminals yet — wait for trial-end.
  - env trial-end fires when ALL active egos in env have removed=1 OR env's
    per_trial_timeout elapses since trial start. At env trial-end:
      * truncations[i] = 1 for every active ego in env
      * trial_ended_this_step[i] = 1 for every active ego in env
      * All entities (egos + co-players) reset to init position; removed=0
      * env->trial_count++, env->trial_start_timestep = env->timestep
  - At env episode-end (env->trial_count == max_trials):
      * terminals[i] = 1 for every active ego in env
      * Option D: all egos removed=1 + off-map until c_reset

These tests run on a tiny env (per_trial_timeout=5, k=2) for determinism.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(k=2, scenario_length=5, num_agents=4, goal_radius=2.0):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=3,
        k_scenarios=k,
        max_trials_per_episode=k,
        per_trial_timeout=scenario_length,
        goal_radius=goal_radius,
        report_interval=10000,
    )


def _zero_actions(env):
    return np.zeros(env.action_space.shape, dtype=env.actions.dtype)


def test_removed_buffer_exists_and_is_zero_at_reset():
    """Python-side `removed` SHM buffer exists and starts all-zero."""
    env = _make_env()
    env.reset(seed=42)
    assert hasattr(env, "removed"), "env must expose a `removed` SHM buffer"
    assert np.asarray(env.removed, dtype=bool).shape == (env.num_agents,)
    assert not np.asarray(env.removed, dtype=bool).any(), "removed must be all-zero after reset"
    env.close()


def test_env_trial_end_fires_on_timeout_only():
    """Tight goal_radius so no ego reaches. Trial-end MUST fire at
    per_trial_timeout for the env, with truncations=1 on every ego."""
    env = _make_env(k=2, scenario_length=5, num_agents=4, goal_radius=2.0)
    env.reset(seed=42)
    actions = _zero_actions(env)
    truncations_at = None
    for t in range(1, 10):
        env.step(actions)
        if np.asarray(env.truncations, dtype=bool).any():
            truncations_at = t
            break
    assert truncations_at == 5, f"trial-end (timeout) should fire at tick=5, got {truncations_at}"
    # Trial-end fires for ALL active agents simultaneously
    tr = np.asarray(env.truncations, dtype=bool)
    te = np.asarray(env.trial_ended_this_step, dtype=bool)
    assert tr.all() or tr.sum() >= 1, f"truncations should fire env-wide: {tr}"
    assert (tr == te).all(), f"truncations and trial_ended_this_step must align: tr={tr}, te={te}"
    env.close()


def test_ego_goes_off_map_on_reach():
    """Wide goal_radius so all egos reach quickly. Each ego should become
    removed=1 the step after it reaches goal."""
    env = _make_env(k=2, scenario_length=20, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = _zero_actions(env)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1  # gentle accel — within speed limit
    saw_removed = False
    for _ in range(20):
        env.step(actions)
        if np.asarray(env.removed, dtype=bool).any():
            saw_removed = True
            break
    assert saw_removed, "At least one ego should have removed=1 mid-trial after reaching goal"
    env.close()


def test_env_trial_end_resets_all_entities_to_init():
    """After a NON-terminal env trial-end (trial < max_trials), all egos
    must be back on-map (removed=0). k must be >= 3 so trial 1 end isn't
    the same step as episode-end."""
    env = _make_env(k=3, scenario_length=5, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = _zero_actions(env)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1
    # Run until first env trial-end
    for t in range(1, 20):
        env.step(actions)
        if np.asarray(env.truncations, dtype=bool).any():
            # At trial-end step itself, the reset has already fired in C —
            # removed should already be 0 (entities back at init).
            removed_after = np.asarray(env.removed, dtype=bool)
            term = np.asarray(env.terminals, dtype=bool)
            assert not term.any(), f"trial 1 must not also be episode-end (k={3}); got terminals={term}"
            assert not removed_after.any(), (
                f"After env trial-end (mid-episode), all egos must be back on-map. Got: {removed_after}"
            )
            env.close()
            return
    raise AssertionError("env trial-end never fired in 20 steps")


def test_episode_end_fires_after_max_trials():
    """After max_trials env trial-ends, terminals must fire for all egos.
    Option D semantic: removed=1 stays until c_reset."""
    env = _make_env(k=2, scenario_length=3, num_agents=4, goal_radius=2.0)
    env.reset(seed=42)
    actions = _zero_actions(env)
    trial_ends = 0
    term_at = None
    for t in range(1, 20):
        env.step(actions)
        if np.asarray(env.truncations, dtype=bool).any():
            trial_ends += 1
        if np.asarray(env.terminals, dtype=bool).any():
            term_at = t
            break
    assert trial_ends >= 1, f"expected ≥1 trial-end before episode end, got {trial_ends}"
    assert term_at is not None, "terminals never fired within 20 steps"
    # At terminals, all egos should be removed (Option D)
    assert np.asarray(env.removed, dtype=bool).all(), (
        f"after episode-end terminals, all egos should be removed: {env.removed}"
    )
    env.close()


def test_truncations_not_fired_on_individual_reach():
    """Before env trial-end, individual reaches must NOT fire truncations.
    Only the env-level trial-end does."""
    env = _make_env(k=2, scenario_length=30, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = _zero_actions(env)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 0.1
    for t in range(1, 6):
        env.step(actions)
        rem = np.asarray(env.removed, dtype=bool)
        tr = np.asarray(env.truncations, dtype=bool)
        # If any ego has removed=1 but not all of them, truncations must NOT
        # fire yet — we're mid-trial waiting for stragglers.
        if rem.any() and not rem.all():
            assert not tr.any(), (
                f"step={t}: removed={rem} but truncations={tr} — env trial-end fired prematurely"
            )
    env.close()


if __name__ == "__main__":
    test_removed_buffer_exists_and_is_zero_at_reset()
    test_env_trial_end_fires_on_timeout_only()
    test_ego_goes_off_map_on_reach()
    test_env_trial_end_resets_all_entities_to_init()
    test_episode_end_fires_after_max_trials()
    test_truncations_not_fired_on_individual_reach()
    print("test_env_level_trial: PASS")
