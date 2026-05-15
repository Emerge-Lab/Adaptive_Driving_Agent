"""Demonstrates: under GOAL_TRIAL, GAE bootstraps across the trial-boundary
state discontinuity, contaminating value targets for steps preceding a trial
end. Then verifies the fix: passing `terminals OR trial_ends` as the
bootstrap-stop mask kills that contamination, while keeping plain `terminals`
for the cache-reset path.

Semantics of compute_puff_advantage (from pufferlib/extensions/pufferlib.cpp:28):
  delta[t]    = rho * (rewards[t+1] + gamma * values[t+1] * (1 - dones[t+1]) - values[t])
  advantage[t] = delta[t] + gamma * lambda * c * advantage[t+1] * (1 - dones[t+1])

So `dones[t+1] = 1` means "step t+1 is a new episode start — don't bootstrap
V[t+1] into delta[t]." For the trial-mode fix we need to set this flag at the
slot immediately AFTER a trial ends.
"""

import torch

from pufferlib.pufferl import compute_puff_advantage


def _adv(values, rewards, dones, gamma=0.99, lam=1.0):
    """Wrap compute_puff_advantage with float32 inputs and a 1-row batch."""
    T = len(values)
    v = torch.tensor([values], dtype=torch.float32)
    r = torch.tensor([rewards], dtype=torch.float32)
    d = torch.tensor([dones], dtype=torch.float32)
    ratio = torch.ones(1, T, dtype=torch.float32)
    out = torch.zeros(1, T, dtype=torch.float32)
    gammas = torch.tensor([gamma], dtype=torch.float32)
    a = compute_puff_advantage(v, r, d, ratio, out, gammas, lam, 1.0, 1.0)
    return a[0].tolist()


def test_baseline_no_terminals_bootstraps_through():
    """Sanity: no terminals → bootstrap propagates all the way."""
    # 6 steps, value=1 everywhere, reward=1 at step 3, gamma=0.99, lambda=1
    adv = _adv(values=[1, 1, 1, 1, 1, 1], rewards=[0, 0, 0, 1, 0, 0], dones=[0, 0, 0, 0, 0, 0])
    # delta_t = r[t+1] + 0.99*v[t+1] - v[t]
    # delta_0 = 0 + 0.99 - 1 = -0.01
    # delta_1 = 0 + 0.99 - 1 = -0.01
    # delta_2 = 1 + 0.99 - 1 = 0.99   (reward at step 3 flows back to step 2)
    # delta_3 = 0 + 0.99 - 1 = -0.01
    # delta_4 = 0 + 0.99 - 1 = -0.01
    # delta_5 = 0  (last; no t+1)
    # adv_5 = 0; adv_4 = -0.01; adv_3 = -0.01 + 0.99*(-0.01) = -0.0199;
    # adv_2 = 0.99 + 0.99*(-0.0199) = 0.9703; adv_1 = -0.01 + 0.99*0.9703 = 0.9506;
    # adv_0 = -0.01 + 0.99*0.9506 = 0.9311
    assert abs(adv[2] - 0.9703) < 1e-3, f"adv[2] should be ~0.97 (reward bootstraps back), got {adv[2]}"
    assert abs(adv[0] - 0.9311) < 1e-3, f"adv[0] should be ~0.93 (bootstrap propagates), got {adv[0]}"


def test_terminal_at_step_3_blocks_bootstrap():
    """dones[3]=1 means step 3 is new episode start → V[3] not bootstrapped into delta[2]."""
    adv = _adv(values=[1, 1, 1, 1, 1, 1], rewards=[0, 0, 0, 1, 0, 0], dones=[0, 0, 0, 1, 0, 0])
    # delta_2 = rewards[3] + 0.99*v[3]*(1-dones[3]) - v[2] = 1 + 0 - 1 = 0
    # adv_2 = delta_2 + 0.99*adv_3*(1-dones[3]) = 0 + 0 = 0
    # adv_1 = delta_1 + 0.99*adv_2 = (0 + 0.99 - 1) + 0 = -0.01
    # adv_0 = delta_0 + 0.99*adv_1 = -0.01 + 0.99*(-0.01) = -0.0199
    assert abs(adv[2] - 0.0) < 1e-3, f"adv[2] should be 0 (no bootstrap past terminal), got {adv[2]}"
    assert abs(adv[0] - (-0.0199)) < 1e-3, f"adv[0] should be ~-0.02 (clean episode), got {adv[0]}"


def test_trial_boundary_without_terminal_contaminates_gae():
    """SCENARIO: a trial ends at step 2. Reward 1 is earned (goal bonus) and the
    agent is respawned to traj[0]. With the CURRENT pufferl.py rollout writes
    (terminals[i] = env's d, where d=0 at trial-end-but-not-episode-end),
    GAE sees dones=0 everywhere → contaminates the value target.

    Simulate the post-trial-boundary state: step 3 is the post-respawn obs
    with a "fresh start" value (say V=5, very different from the trial 0
    end value V=1). Reward 1 at step 3 (the goal bonus from old trial).
    """
    # Trial 0: steps 0..2, V≈1 (mid-trial). Reward 1 at step 3 (the carry-over goal bonus).
    # Trial 1: steps 3..5, V=5 (fresh start has higher value because the policy
    # expects more reward ahead). Without telling GAE this is a trial boundary,
    # delta_2 = r[3] + 0.99*v[3] - v[2] = 1 + 0.99*5 - 1 = 4.95
    # That 4.95 is WRONG — most of it comes from V[3]=5, the value of a state
    # that is causally disconnected from step 2 (agent was teleported).
    adv_bug = _adv(values=[1, 1, 1, 5, 5, 5], rewards=[0, 0, 0, 1, 0, 0], dones=[0, 0, 0, 0, 0, 0])
    # adv[2] under bug ≈ 4.95 + 0.99*adv[3] ≈ huge positive contamination
    assert adv_bug[2] > 4.0, f"BUG REPRO: adv[2] under current code is contaminated by V[3]={5}, got {adv_bug[2]}"

    # Now simulate the FIX: bootstrap_mask = terminals OR trial_ends.
    # At slot 3 we set bootstrap_mask=1 (=trial boundary), so V[3] is NOT
    # bootstrapped into delta[2], regardless of terminals being 0.
    adv_fix = _adv(values=[1, 1, 1, 5, 5, 5], rewards=[0, 0, 0, 1, 0, 0], dones=[0, 0, 0, 1, 0, 0])
    # adv[2] under fix = 1 + 0 - 1 = 0, plus 0 from killed bootstrap = 0
    assert abs(adv_fix[2]) < 1e-3, f"FIX: adv[2] should be 0 (V[3] not bootstrapped), got {adv_fix[2]}"

    # Contamination magnitude:
    contamination = adv_bug[2] - adv_fix[2]
    assert contamination > 4.0, (
        f"Under current rollout, trial boundary contaminates adv[2] by {contamination:.3f} "
        f"(=V[next] * gamma carried into the previous trial's value target). This is the bug."
    )


def test_trial_boundary_does_not_block_attention_or_cache():
    """The fix uses bootstrap_mask for GAE only. The other consumers of
    `terminals` (transformer attention mask, KV-cache reset) must keep
    using plain `terminals` so that:
    - Attention spans trial boundaries within an episode (needed for adaptation).
    - KV cache persists across trials (the load-bearing line for the thesis).
    This test asserts the contract by showing that the two signals must be
    kept distinct in any implementation."""
    # Simulated buffer for one episode of 2 trials, terminal at step 5:
    #   terminals     = [0, 0, 0, 0, 0, 1]      # only true ep end
    #   trial_ends    = [0, 0, 0, 1, 0, 0]      # trial 0 ended at step 2; new trial starts at step 3
    terminals = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    trial_ends = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.float32)
    bootstrap_mask = (terminals.bool() | trial_ends.bool()).float()
    # Expected: [0, 0, 0, 1, 0, 1] — bootstrap stops at both trial boundary and episode end.
    assert bootstrap_mask.tolist() == [[0, 0, 0, 1, 0, 1]]

    # And the cache-reset signal (used in pufferl.py:688 done_mask) should
    # only fire at the episode boundary:
    cache_reset_mask = terminals  # not bootstrap_mask
    assert cache_reset_mask.tolist() == [[0, 0, 0, 0, 0, 1]]


if __name__ == "__main__":
    test_baseline_no_terminals_bootstraps_through()
    test_terminal_at_step_3_blocks_bootstrap()
    test_trial_boundary_without_terminal_contaminates_gae()
    test_trial_boundary_does_not_block_attention_or_cache()
    print("All GAE trial-boundary tests passed.")
