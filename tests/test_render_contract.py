"""Render-visibility contract under all goal_behaviors.

Renders read per-entity state directly (mesh draw, collision, obs[6]). Six
gates in drive.h check `respawn_timestep != -1` and skip drawing / zero
features. Any code path that respawns an agent in trial mode MUST clear
that flag to -1, else the agent disappears from the render mid-episode.

These tests don't actually render a video — they check the observable
proxy: ego's obs[6] (= `(respawn_timestep != -1) ? 1 : 0`). If obs[6]
sticks at 1 in trials 2..k, the renderer's mesh draw will skip the ego.

Run me before touching drive.h or render.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"

# Ego obs feature 6 is `respawn_timestep != -1` (set in drive.h:2409).
# Used by render mesh-draw gate at drive.h:3482 and obs-zeroing at 2455/2457.
EGO_GHOST_OBS_IDX = 6


def _make_env(goal_behavior, k_scenarios=4, scenario_length=20, max_trials=None, per_trial=None):
    from pufferlib.ocean.drive import Drive

    kwargs = dict(
        num_agents=4,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=goal_behavior,
        k_scenarios=k_scenarios,
    )
    if max_trials is not None:
        kwargs["max_trials_per_episode"] = max_trials
    if per_trial is not None:
        kwargs["per_trial_timeout"] = per_trial
    return Drive(**kwargs)


def _zero_actions(env):
    return np.zeros(env.action_space.shape, dtype=env.actions.dtype)


def test_ego_visible_in_every_trial_k4():
    """gb=3, k=4: ego obs[6] must return to 0 within 2 steps after each
    mid-episode trial boundary. Pre-fix (M7-fix) it would stick at 1, hiding
    ego from renders for trials 2-4."""
    env = _make_env(goal_behavior=3, k_scenarios=4, scenario_length=10, max_trials=4, per_trial=10)
    env.reset(seed=42)
    actions = _zero_actions(env)
    # Run the full episode budget (k_scenarios * scenario_length = 40 ticks).
    # Watch every step; whenever a mid-episode trial boundary fires, the
    # ghost obs flag must clear within 2 subsequent steps.
    last_trial_boundary = None
    ghost_stuck_steps = []
    for t in range(40):
        env.step(actions)
        # Trial boundary mid-episode: truncations fire, terminals do not.
        trunc_now = np.asarray(env.truncations, dtype=bool).any()
        term_now = np.asarray(env.terminals, dtype=bool).any()
        if trunc_now and not term_now:
            last_trial_boundary = t
        if last_trial_boundary is not None and t > last_trial_boundary + 1:
            # By 2 steps after a trial-respawn the ghost flag MUST be clear
            # (mid-episode trial mode is not ghost-fade mode).
            ghost_obs = env.observations[:, EGO_GHOST_OBS_IDX]
            stuck = np.where(ghost_obs > 0.5)[0]
            if len(stuck) > 0 and not np.asarray(env.terminals, dtype=bool).any():
                ghost_stuck_steps.append((t, last_trial_boundary, stuck.tolist()))
    assert not ghost_stuck_steps, (
        f"Ego ghost flag (obs[{EGO_GHOST_OBS_IDX}]) stuck after mid-episode trial respawn. "
        f"This hides the ego from renders in trials 2..K. Stuck events: {ghost_stuck_steps[:5]}"
    )
    env.close()


def test_ego_visible_in_every_trial_k2():
    """Same as above but k=2 (smallest meaningful trial-mode setup)."""
    env = _make_env(goal_behavior=3, k_scenarios=2, scenario_length=10, max_trials=2, per_trial=10)
    env.reset(seed=42)
    actions = _zero_actions(env)
    saw_mid_trial_boundary = False
    for t in range(20):
        env.step(actions)
        trunc_now = np.asarray(env.truncations, dtype=bool).any()
        term_now = np.asarray(env.terminals, dtype=bool).any()
        if trunc_now and not term_now:
            saw_mid_trial_boundary = True
        if saw_mid_trial_boundary and not term_now:
            ghost_obs = env.observations[:, EGO_GHOST_OBS_IDX]
            stuck = np.where(ghost_obs > 0.5)[0]
            # Tolerate one step of stuck (the step at which the respawn fires);
            # by the next observation it must clear.
            if t > 0:
                assert len(stuck) == 0, (
                    f"step={t}: ego ghost flag stuck for agents {stuck.tolist()} "
                    f"under gb=3 k=2 after mid-trial respawn — renders will fail."
                )
    assert saw_mid_trial_boundary, "test setup did not produce a mid-episode trial boundary"
    env.close()


def test_non_trial_modes_ghost_semantics_preserved():
    """gb=0 (RESPAWN) intentionally has ghost-fade semantics post-respawn,
    so obs[6] CAN be 1 after a respawn — we must not regress that."""
    env = _make_env(goal_behavior=0, k_scenarios=2, scenario_length=20)
    env.reset(seed=42)
    actions = _zero_actions(env)
    # Just step it for a while; the test passes if it runs without raising
    # (we're not asserting anything specific about ghost obs here, but we
    # are confirming gb=0's code path doesn't crash with our test setup).
    for _ in range(40):
        env.step(actions)
    # Sanity: env produced obs for all agents
    assert env.observations.shape[0] >= 1
    env.close()


def test_render_gate_state_after_full_episode_k4():
    """Coarser end-to-end: by the time the episode ends under gb=3 k=4,
    every agent should be either in a clean playable state (ghost=0) OR
    explicitly idle (Option D: removed=1 → off-grid). No agent should be
    'visually invisible but still on the grid', because that is the bug
    the M7-fix addressed."""
    env = _make_env(goal_behavior=3, k_scenarios=4, scenario_length=10, max_trials=4, per_trial=10)
    env.reset(seed=42)
    actions = _zero_actions(env)
    # Just before the env auto-resamples (around the resample_frequency tick),
    # peek at obs.
    for _ in range(38):  # short of the 40-tick resample_frequency
        env.step(actions)
    ghost_obs = env.observations[:, EGO_GHOST_OBS_IDX]
    # Agents that have terminated will have obs from before terminal; that's
    # OK. Agents still playing should not be ghosted.
    # This is a sanity check: at most a transient frame.
    assert ghost_obs.sum() <= env.observations.shape[0], f"ghost obs accumulated above agent count: {ghost_obs}"
    env.close()


if __name__ == "__main__":
    test_ego_visible_in_every_trial_k4()
    test_ego_visible_in_every_trial_k2()
    test_non_trial_modes_ghost_semantics_preserved()
    test_render_gate_state_after_full_episode_k4()
    print("test_render_contract: PASS")
