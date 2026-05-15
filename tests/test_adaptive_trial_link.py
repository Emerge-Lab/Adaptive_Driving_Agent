"""Verifies that under goal_behavior=GOAL_TRIAL (=3), AdaptiveDrivingAgent
automatically links max_trials_per_episode → k_scenarios and per_trial_timeout
→ scenario_length, so that a launcher passing only `--env.k-scenarios K` and
`--env.goal-behavior 3` gets k trials per episode out of the box.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"


def _make_adaptive(k, goal_behavior=3, scenario_length=201, **extra):
    """Mimic the puffer_adaptive_drive env_creator kwarg flow."""
    from pufferlib.ocean.drive.adaptive import AdaptiveDrivingAgent

    kwargs = dict(
        num_agents=8,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=scenario_length,
        k_scenarios=k,
        dynamics_model="classic",
        goal_behavior=goal_behavior,
        # ini defaults that the puffer arg parser would inject:
        max_trials_per_episode=2,
        per_trial_timeout=0,
        co_player_enabled=False,
    )
    kwargs.update(extra)
    return AdaptiveDrivingAgent(**kwargs)


def test_auto_link_k2():
    env = _make_adaptive(k=2)
    assert env.max_trials_per_episode == 2, f"k=2: expected max_trials=2, got {env.max_trials_per_episode}"
    assert env.per_trial_timeout == 201, f"k=2: expected per_trial_timeout=201, got {env.per_trial_timeout}"
    env.close()


def test_auto_link_k3():
    env = _make_adaptive(k=3)
    assert env.max_trials_per_episode == 3, f"k=3: expected max_trials=3, got {env.max_trials_per_episode}"
    assert env.per_trial_timeout == 201, f"k=3: expected per_trial_timeout=201, got {env.per_trial_timeout}"
    env.close()


def test_auto_link_k4():
    env = _make_adaptive(k=4)
    assert env.max_trials_per_episode == 4, f"k=4: expected max_trials=4, got {env.max_trials_per_episode}"
    assert env.per_trial_timeout == 201, f"k=4: expected per_trial_timeout=201, got {env.per_trial_timeout}"
    env.close()


def test_no_link_when_goal_behavior_not_3():
    """Under goal_behavior in {0,1,2}, max_trials_per_episode stays at the
    INI default (2) regardless of k_scenarios — trial mode is off."""
    env = _make_adaptive(k=3, goal_behavior=0)
    assert env.max_trials_per_episode == 2, (
        f"gb=0 + k=3: max_trials should stay at INI default 2, got {env.max_trials_per_episode}"
    )
    env.close()


def test_user_override_ignored_under_gb3():
    """Option A invariant: under gb=3, max_trials_per_episode is ALWAYS
    k_scenarios. Any explicit override is silently replaced. To get a
    different trial count, change k_scenarios."""
    env = _make_adaptive(k=3, max_trials_per_episode=5)
    assert env.max_trials_per_episode == 3, (
        f"under gb=3, max_trials_per_episode must equal k_scenarios (=3); got {env.max_trials_per_episode}"
    )
    env.close()


def test_per_trial_timeout_override_ignored_under_gb3():
    """Same invariant for per_trial_timeout — always scenario_length under gb=3."""
    env = _make_adaptive(k=2, per_trial_timeout=50)
    assert env.per_trial_timeout == env.scenario_length, (
        f"under gb=3, per_trial_timeout must equal scenario_length; got {env.per_trial_timeout}"
    )
    env.close()


if __name__ == "__main__":
    test_auto_link_k2()
    test_auto_link_k3()
    test_auto_link_k4()
    test_no_link_when_goal_behavior_not_3()
    test_user_override_ignored_under_gb3()
    test_per_trial_timeout_override_ignored_under_gb3()
    print("All adaptive trial-link tests passed.")
