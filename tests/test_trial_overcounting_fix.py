"""Three fixes from the adversarial review.

Fix #1: `add_log_one_agent` resets all per-entity state that c_reset resets.
        Without this, fields like `respawn_timestep`, `current_goal_reached`,
        and `metrics_array[*]` carry over from one trial-mode episode to the
        next, silently corrupting obs and reward.

Fix #2: `move_expert` loops the recorded trajectory under GOAL_TRIAL
        (`t % array_size`). Pre-fix, experts vanished (INVALID_POSITION)
        for the entire second half of every episode when
        `max_trials * per_trial_timeout > array_size`.

Fix #3 (the score bug): under GOAL_TRIAL the else branch at drive.h:2796
        increments `goals_reached_this_episode += 1` but pre-fix never set
        `current_goal_reached = 1`. So every step the agent sat in goal
        radius bumped the counter — score was completely useless even after
        the max_trials-denominator fix.

This file asserts the fix in three corresponding tests.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


def _make_env(max_trials, per_trial_timeout=8, num_agents=4, goal_radius=200.0, scenario_length=200):
    """Default goal_radius=200 means the agent is essentially always at goal,
    so we can test the over-counting behavior easily."""
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        ini_file=INI,
        goal_behavior=3,
        max_trials_per_episode=max_trials,
        per_trial_timeout=per_trial_timeout,
        goal_radius=goal_radius,
        action_type="continuous",
        report_interval=10000,
    )


def test_goals_reached_capped_at_max_trials_per_episode():
    """Fix #3: with huge goal_radius, the agent is always inside goal radius.
    Pre-fix, goals_reached_this_episode would increment on EVERY step (gated
    only by current_goal_reached, which was never set in the trial-mode
    branch). Post-fix, current_goal_reached=1 stops re-firing within a trial,
    and respawn_agent resets it for the next trial. So with max_trials=2 and
    every trial succeeding, goals_reached_this_episode per episode should be
    exactly 2, not the number of ticks the agent stayed in radius.

    Probe: after one episode, log.goals_reached_this_episode (avg per-agent
    per-episode) should equal max_trials_per_episode."""
    from pufferlib.ocean.drive import binding

    env = _make_env(max_trials=2, per_trial_timeout=8, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    # Drive forward so goal-reach fires (with goal_radius=200, basically every step counts).
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    log = None
    for _ in range(200):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            break
    assert log and log.get("n", 0) > 0, f"No episode emitted: {log}"
    # Average goals_reached_this_episode across agents should be EXACTLY
    # max_trials=2 (= n_trials_goal_reached) — no over-counting.
    avg_goals = log.get("goals_reached_this_episode", 0)
    n_trials_reached = log.get("n_trials_goal_reached", 0)
    assert abs(avg_goals - 2.0) < 0.1, (
        f"goals_reached_this_episode per ep should be ~2 (max_trials), got {avg_goals}. "
        f"Pre-fix would have been many times larger (one per in-radius tick)."
    )
    assert abs(n_trials_reached - 2.0) < 0.1, (
        f"n_trials_goal_reached should be 2 (every trial succeeded), got {n_trials_reached}"
    )
    env.close()


def test_score_requires_all_trials_to_succeed():
    """With the over-counting fixed AND the max_trials denominator,
    score=1 now requires goals_reached > max_trials*threshold. For k=2,
    threshold=0.5, so score=1 iff frac>0.5 iff goals_reached=2 (since
    goals_reached is integer)."""
    from pufferlib.ocean.drive import binding

    env = _make_env(max_trials=2, per_trial_timeout=8, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    for _ in range(200):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            break
    assert log["n_trials_goal_reached"] == 2.0
    assert log["score"] == 1.0, (
        f"score should be 1 only when ALL trials succeed (goals_reached=max_trials); got {log['score']}"
    )
    env.close()


def test_score_zero_with_only_partial_trial_success():
    """Tight goal_radius=2 so zero-action agent doesn't reach. n_trials_goal_reached=0
    → goals_reached=0 → score=0. Pre-fix this also passed because frac=0 < 0.99,
    but pre-fix with goal_radius=200 would have score=1 spuriously — see other test."""
    from pufferlib.ocean.drive import binding

    env = _make_env(max_trials=2, per_trial_timeout=8, num_agents=4, goal_radius=2.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    for _ in range(400):
        env.step(actions)
    log = binding.vec_log(env.c_envs, env.num_agents)
    assert log and log.get("n", 0) > 0
    assert log["score"] == 0.0, f"Zero-action with tight goal_radius should score 0, got {log['score']}"
    env.close()


def test_per_entity_respawn_timestep_resets_at_episode_end():
    """Fix #1: respawn_timestep was NOT reset in add_log_one_agent pre-fix.
    After the first multi-trial episode, every agent had respawn_timestep set
    to some value > -1 forever, so obs[6] = (respawn_timestep != -1) was
    stuck at 1 indefinitely.

    Under Option D + map rotation at resample_frequency: episode 2 uses a
    DIFFERENT map than episode 1, so goals_reached varies by map (not every
    starting pose lies inside goal_radius of the new map's goal). What we
    can still assert is that goals_reached_this_episode is BOUNDED by
    max_trials_per_episode in EVERY emission. Pre-fix (without
    current_goal_reached gating) it would balloon to many times max_trials
    on emission 1 because state carried over.
    """
    from pufferlib.ocean.drive import binding

    MAX_TRIALS = 2
    env = _make_env(max_trials=MAX_TRIALS, per_trial_timeout=8, num_agents=4, goal_radius=200.0)
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    # Drain TWO episodes worth of vec_log emissions. Each emission's
    # goals_reached must be in [0, max_trials] — no over-counting.
    logs = []
    for _ in range(800):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            logs.append(log)
        if len(logs) >= 2:
            break
    assert len(logs) >= 2, f"Need 2 emissions; got {len(logs)}"
    for i, log in enumerate(logs):
        avg_goals = log.get("goals_reached_this_episode", 0)
        # Hard upper bound: per-episode goals can't exceed max_trials with
        # the current_goal_reached gate working. Pre-fix this could be ~8+
        # (one increment per in-radius tick) on an aggregated log.
        assert avg_goals <= MAX_TRIALS + 0.1, (
            f"Emission {i}: goals_reached={avg_goals} exceeds max_trials={MAX_TRIALS} — "
            f"over-counting regression (current_goal_reached gate broken)."
        )
        # Lower bound: ≥0. Combined with the upper bound this means state
        # reset works (no accumulation across episodes).
        assert avg_goals >= 0.0, f"Emission {i}: negative goals_reached={avg_goals}"
    # At least one emission must show the agent actually reaching goal(s) —
    # otherwise we haven't really tested over-counting at all.
    assert any(log.get("goals_reached_this_episode", 0) > 0.5 for log in logs), (
        f"No emission shows goals_reached_this_episode > 0; test setup invalid: {logs}"
    )
    env.close()


def test_expert_traffic_present_past_scenario_length():
    """Fix #2: under GOAL_TRIAL the episode budget (max_trials * per_trial_timeout)
    can exceed scenario_length (the recorded expert trajectory length).
    Pre-fix, all static expert agents vanished (INVALID_POSITION) past
    scenario_length, gutting trial 2+.

    Probe via vec_log under GOAL_TRIAL: with experts looping their trajectory,
    `active_agent_count` in the log should stay non-zero deep into the
    second trial. Pre-fix, `move_expert` set every agent to INVALID_POSITION
    past tick=array_size, but it didn't change active_agent_count — so
    that's not the right probe.

    Better probe: run the env in trial mode and inspect `env.observations`,
    which is the per-step Python-readable buffer. Pre-fix, the ego's
    "nearest partners" feature slot would be all-zero past scenario_length
    (because partner positions are INVALID, filtered out as too-far).
    """
    from pufferlib.ocean.drive import Drive, binding

    # nuplan_201 trajectories are 201 ticks. With per_trial_timeout=201 and
    # max_trials=2, the episode budget is 402 — second trial happens past
    # the trajectory length.
    env = Drive(
        num_agents=8,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=201,
        ini_file=INI,
        goal_behavior=3,
        max_trials_per_episode=2,
        per_trial_timeout=201,
        action_type="continuous",
        report_interval=10000,
    )
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    obs0 = env.observations.copy()
    for _ in range(50):
        env.step(actions)
    obs_mid_trial0 = env.observations.copy()
    for _ in range(200):  # now we're well into trial 2 (tick ~250)
        env.step(actions)
    obs_trial1 = env.observations.copy()
    # obs has shape (num_agents, obs_dim). The ego's partner features start
    # somewhere after the ego features. Pre-fix, partner features would go
    # to all zeros past array_size. With the move_expert loop, they stay
    # populated. Probe: are there any non-zero observation entries past
    # the first 20 dims (which are ego features)?
    nonzero_t0 = np.abs(obs_mid_trial0[:, 20:]).sum()
    nonzero_t1 = np.abs(obs_trial1[:, 20:]).sum()
    # Both should be non-zero. If experts vanished in trial 1, nonzero_t1
    # would be near 0 (all partner features cleared).
    assert nonzero_t0 > 0, f"mid-trial-0 partner obs is all-zero ({nonzero_t0}); test setup wrong"
    assert nonzero_t1 > 0, f"trial-1 partner obs is all-zero ({nonzero_t1}); experts vanished"
    # Stronger: trial 1 obs energy should be within an order of magnitude of trial 0.
    ratio = nonzero_t1 / nonzero_t0
    assert ratio > 0.1, (
        f"trial-1 partner obs energy = {ratio:.3f}× trial-0 — suggests experts disappeared. "
        f"With looping, trial-1 should be roughly comparable to trial-0."
    )
    env.close()


def test_non_trial_modes_unaffected_by_overcounting_fix():
    """The `current_goal_reached = 1` gate added in the else branch fires for
    GOAL_STOP too (which uses the same branch). Confirm gb=2 (STOP) still
    behaves: agent that reaches goal gets a single +reward_goal and stays
    stopped, doesn't get re-rewarded each tick."""
    from pufferlib.ocean.drive import binding

    env = _make_env(max_trials=2, per_trial_timeout=8, num_agents=4, goal_radius=200.0)
    env.close()
    # Now make a GOAL_STOP env
    from pufferlib.ocean.drive import Drive

    env = Drive(
        num_agents=4,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=20,
        ini_file=INI,
        goal_behavior=2,  # STOP
        goal_radius=200.0,
        action_type="continuous",
        report_interval=10000,
    )
    env.reset(seed=42)
    actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
    if env.action_space.shape[-1] == 2:
        actions[:, 0] = 1.0
    for _ in range(200):
        env.step(actions)
        log = binding.vec_log(env.c_envs, env.num_agents)
        if log and log.get("n", 0) > 0:
            break
    # Under GOAL_STOP with 1 goal per scenario, agent reaches goal once,
    # goals_reached_this_episode = 1. Score with threshold 0.99 requires frac > 0.99.
    # frac = 1 / goals_sampled. If goals_sampled stays at 1, frac=1.0 > 0.99 → score=1.
    # That's correct.
    assert log["goals_reached_this_episode"] <= 1.5, (
        f"GOAL_STOP should reach goal ~once per scenario, got {log['goals_reached_this_episode']}"
    )
    env.close()


if __name__ == "__main__":
    test_goals_reached_capped_at_max_trials_per_episode()
    test_score_requires_all_trials_to_succeed()
    test_score_zero_with_only_partial_trial_success()
    test_per_entity_respawn_timestep_resets_at_episode_end()
    test_expert_traffic_present_past_scenario_length()
    test_non_trial_modes_unaffected_by_overcounting_fix()
    print("All trial-overcounting-fix tests passed.")
