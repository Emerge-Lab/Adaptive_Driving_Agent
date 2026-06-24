"""M5 verification: HumanReplayEvaluator branches on goal_behavior.

Under goal_behavior=3, the rollout loop iterates trials (using
trial_ended_this_step) and emits trial_X_score + ada_delta_trial_K_minus_0
keys. Under goal_behavior in {0,1,2} the existing scenario-mode is
preserved.

Uses a deterministic stub policy that always outputs action 0 — fine for
testing the eval pipeline plumbing.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


class _StubPolicy(nn.Module):
    """Minimal policy: forward_eval returns action-0 logits + zero value."""

    def __init__(self, num_actions, hidden_size=16, horizon=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        # Tag so HumanReplayEvaluator picks the transformer state-dict shape.
        self.transformer = nn.Identity()
        self.num_actions = num_actions

    def forward_eval(self, obs, state):
        B = obs.shape[0]
        logits = torch.zeros(B, self.num_actions, device=obs.device)
        logits[:, 0] = 1.0  # always pick action 0
        value = torch.zeros(B, device=obs.device)
        return logits, value


def _make_drive_env(goal_behavior, max_trials=2, per_trial_timeout=5, num_agents=4):
    from pufferlib.ocean.drive import Drive

    return Drive(
        num_agents=num_agents,
        map_dir=MAP_DIR,
        num_maps=10,
        scenario_length=20,
        ini_file=INI,
        goal_behavior=goal_behavior,
        max_trials_per_episode=max_trials,
        per_trial_timeout=per_trial_timeout,
    )


def _make_args(goal_behavior, max_trials=2, per_trial_timeout=5, num_rollouts=2):
    return {
        "env": {
            "goal_behavior": goal_behavior,
            "max_trials_per_episode": max_trials,
            "per_trial_timeout": per_trial_timeout,
            "k_scenarios": 1,
            "scenario_length": 20,
            "init_steps": 0,
        },
        "train": {"device": "cpu"},
        "eval": {"human_replay_num_rollouts": num_rollouts, "recovery_goal_reward_threshold": 0.5},
    }


def test_trial_mode_emits_trial_keys():
    """gb=3: rollout() returns trial_0_score, trial_1_score, ada_delta_trial_1_minus_0."""
    from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

    env = _make_drive_env(goal_behavior=3, max_trials=2, per_trial_timeout=5)
    env.reset(seed=42)
    args = _make_args(goal_behavior=3, max_trials=2, per_trial_timeout=5, num_rollouts=2)
    evaluator = HumanReplayEvaluator(args)
    policy = _StubPolicy(
        num_actions=env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    )

    out = evaluator.rollout(args, env, policy)

    # Trial mode keys
    assert "trial_0_score" in out, f"missing trial_0_score; keys={sorted(out.keys())[:30]}"
    assert "trial_1_score" in out, f"missing trial_1_score"
    assert "ada_delta_trial_1_minus_0" in out, f"missing ada_delta_trial_1_minus_0"
    # per_agent records use 't' prefix in trial mode
    assert "per_agent_success_log" in out
    if out["per_agent_success_log"]:
        keys = set(out["per_agent_success_log"][0].keys())
        assert "t0" in keys and "t1" in keys, f"trial-mode records should have t0, t1; got {keys}"
        assert "s0" not in keys, f"trial-mode records should NOT have s0; got {keys}"
    print(
        f"  ok: gb=3 trial mode → trial_0_score={out['trial_0_score']:.3f} "
        f"trial_1_score={out['trial_1_score']:.3f} "
        f"ada_delta_trial_1_minus_0={out['ada_delta_trial_1_minus_0']:.3f}"
    )
    env.close()


def test_trial_mode_auto_link_k4():
    """Regression: when AdaptiveDrivingAgent auto-links max_trials_per_episode
    from k_scenarios=4 under gb=3, the evaluator must detect the actual env
    value (4) rather than blindly reading the INI default (2) from args. If
    it falls back to 2, trial_2_score / trial_3_score / ada_delta_trial_3_minus_0
    are missing and the user only sees trial1 vs trial0."""
    from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

    # Build a stub env that mimics the auto-link result: driver_env exposes
    # max_trials_per_episode=4 even though the args dict still says 2.
    env = _make_drive_env(goal_behavior=3, max_trials=4, per_trial_timeout=5)
    env.reset(seed=42)
    # args says max_trials=2 (the INI default) + k_scenarios=4. The auto-link
    # in AdaptiveDrivingAgent would have set the env's actual max_trials to 4;
    # the evaluator must arrive at the same conclusion from args (k_scenarios)
    # since `driver_env` only exists on vec-env wrappers, not raw Drive.
    args = _make_args(goal_behavior=3, max_trials=2, per_trial_timeout=5, num_rollouts=2)
    args["env"]["k_scenarios"] = 4  # this is what would have driven the auto-link

    evaluator = HumanReplayEvaluator(args)
    policy = _StubPolicy(
        num_actions=env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    )

    out = evaluator.rollout(args, env, policy)

    # All 4 trial scores must appear (not just 0 and 1)
    for k in range(4):
        assert f"trial_{k}_score" in out, (
            f"missing trial_{k}_score with auto-linked max_trials=4. "
            f"keys present: {sorted(k for k in out.keys() if k.startswith('trial_'))[:10]}"
        )
    # ada_delta_trial_{1..3}_minus_0 must all appear
    for k in range(1, 4):
        assert f"ada_delta_trial_{k}_minus_0" in out, f"missing ada_delta_trial_{k}_minus_0"
    # per_agent records should have t0..t3
    if out["per_agent_success_log"]:
        keys = set(out["per_agent_success_log"][0].keys())
        for k in range(4):
            assert f"t{k}" in keys, f"trial-mode records should have t{k}; got {keys}"
    print(
        f"  ok: gb=3 auto-link → trial_0..3 scores all logged; "
        f"deltas: "
        f"d1={out['ada_delta_trial_1_minus_0']:.3f} "
        f"d2={out['ada_delta_trial_2_minus_0']:.3f} "
        f"d3={out['ada_delta_trial_3_minus_0']:.3f}"
    )
    env.close()


def test_scenario_mode_preserved():
    """gb=0: scenario-mode unchanged. per_agent_success_log uses 's' prefix."""
    from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

    env = _make_drive_env(goal_behavior=0, max_trials=2, per_trial_timeout=5)
    env.reset(seed=42)
    args = _make_args(goal_behavior=0, num_rollouts=2)
    args["env"]["k_scenarios"] = 2
    evaluator = HumanReplayEvaluator(args)
    policy = _StubPolicy(
        num_actions=env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    )

    out = evaluator.rollout(args, env, policy)

    # Scenario mode: trial keys absent
    assert "trial_0_score" not in out, f"gb=0 should NOT emit trial_0_score"
    assert "ada_delta_trial_1_minus_0" not in out
    if out["per_agent_success_log"]:
        keys = set(out["per_agent_success_log"][0].keys())
        assert "s0" in keys, f"scenario-mode records should have s0; got {keys}"
        assert "t0" not in keys, f"scenario-mode records should NOT have t0; got {keys}"
    print(f"  ok: gb=0 scenario mode preserved (per_agent records use s0/s1)")
    env.close()


def _run_all():
    test_trial_mode_emits_trial_keys()
    test_scenario_mode_preserved()
    print("\ntest_evaluator_trial_mode: PASS")


if __name__ == "__main__":
    _run_all()
