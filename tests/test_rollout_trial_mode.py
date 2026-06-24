"""Fix #3: rollout_loop under GOAL_TRIAL.

Asserts:
  1. max_steps default: under non-trial mode, defaults to scenario_length;
     under GOAL_TRIAL, defaults to max_trials * per_trial_timeout (so the
     whole multi-trial episode is rendered, not just trial 0).
  2. Break condition: under non-trial, breaks on truncs.all(); under
     GOAL_TRIAL, breaks on terminals.all() (truncs fires on every trial
     boundary now, so truncs.all() would cut the render too early).
  3. Trial bookkeeping: under GOAL_TRIAL, rollout_loop populates
     _trial_starts list in the info dict — each entry is (step, agent, new_idx).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MAP_DIR = "resources/drive/binaries/nuplan_201"
INI = "pufferlib/config/ocean/drive.ini"


class _DummyPolicy:
    """Minimal policy that satisfies rollout_loop.forward_eval contract.
    Returns zero-mean Normal actions for continuous, zero logits for discrete."""

    def __init__(self, action_shape, continuous=False, device="cpu"):
        self.action_shape = action_shape
        self.continuous = continuous
        self.device = device

    def forward_eval(self, obs, state):
        if self.continuous:
            B = obs.shape[0]
            dim = self.action_shape[-1]
            loc = torch.zeros(B, dim, device=self.device)
            scale = torch.ones(B, dim, device=self.device) * 1e-6  # deterministic zero actions
            dist = torch.distributions.Normal(loc, scale)
            value = torch.zeros(B, device=self.device)
            return dist, value
        B = obs.shape[0]
        n_actions = self.action_shape[-1] if len(self.action_shape) > 1 else 5
        logits = torch.zeros(B, n_actions, device=self.device)
        value = torch.zeros(B, device=self.device)
        return logits, value

    def eval(self):
        return self


def _make_drive_env(goal_behavior, max_trials=2, per_trial_timeout=10, scenario_length=200, num_agents=4):
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
        action_type="continuous",
        report_interval=10000,
    )


def test_max_steps_default_under_goal_trial():
    """rollout_loop with max_steps=None under GOAL_TRIAL should use
    max_trials * per_trial_timeout, not scenario_length."""
    from pufferlib.ocean.drive.rollout import rollout_loop

    env = _make_drive_env(goal_behavior=3, max_trials=3, per_trial_timeout=8, scenario_length=200)
    policy = _DummyPolicy(env.action_space.shape, continuous=True)
    info = rollout_loop(policy, env, device="cpu", use_rnn=True, max_steps=None)
    # Expected max_steps = 3 * 8 = 24. The render should have run at least
    # until a few trials elapsed (each trial ~ 8 ticks). Confirm by reading
    # the final trial_idx from info.
    trial_info = next((i for i in info if isinstance(i, dict) and "_final_trial_idx" in i), None)
    assert trial_info is not None, f"Expected _final_trial_idx in info under GOAL_TRIAL: {info}"
    # All agents should have seen multiple trials end (with timeout=8, max_steps=24, expect ~3 trials).
    final_idx = trial_info["_final_trial_idx"]
    assert max(final_idx) >= 2, f"Expected final trial_idx >= 2 (saw multiple trials), got {final_idx}"
    env.close()


def test_max_steps_default_under_non_trial():
    """Under non-trial, max_steps defaults to scenario_length (preserves prior behavior)."""
    from pufferlib.ocean.drive.rollout import rollout_loop

    env = _make_drive_env(goal_behavior=0, scenario_length=30)
    policy = _DummyPolicy(env.action_space.shape, continuous=True)
    info = rollout_loop(policy, env, device="cpu", use_rnn=True, max_steps=None)
    # No _trial_starts emitted in non-trial mode
    has_trial_info = any(isinstance(i, dict) and "_final_trial_idx" in i for i in info)
    assert not has_trial_info, f"Expected no trial info under gb=0, got: {info}"
    env.close()


def test_trial_starts_populated_under_goal_trial():
    """_trial_starts list should record (step, agent, new_idx) tuples for every
    trial boundary observed during the render."""
    from pufferlib.ocean.drive.rollout import rollout_loop

    env = _make_drive_env(goal_behavior=3, max_trials=3, per_trial_timeout=5, scenario_length=200, num_agents=4)
    policy = _DummyPolicy(env.action_space.shape, continuous=True)
    info = rollout_loop(policy, env, device="cpu", use_rnn=True, max_steps=None)
    trial_info = next((i for i in info if isinstance(i, dict) and "_trial_starts" in i), None)
    assert trial_info is not None, f"Expected _trial_starts in info: {info}"
    starts = trial_info["_trial_starts"]
    # With per_trial_timeout=5 and zero-action policy, every 5 ticks all agents
    # time out together. So _trial_starts should have entries at step 5, 10, 15.
    assert len(starts) > 0, f"_trial_starts is empty: {starts}"
    # Each entry is (step, agent_idx, new_trial_idx). Steps should be monotonic.
    steps = [s[0] for s in starts]
    assert steps == sorted(steps), f"trial start steps not monotonic: {steps}"
    env.close()


if __name__ == "__main__":
    test_max_steps_default_under_goal_trial()
    test_max_steps_default_under_non_trial()
    test_trial_starts_populated_under_goal_trial()
    print("All rollout-trial-mode tests passed.")
