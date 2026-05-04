"""Train/eval PE consistency under multi-episode-per-row segments.

The motivating scenario: under the trial-redesign, a single segment row
contains MULTIPLE episodes (terminals fire at multiple slots). At
training time, the segment is processed by `forward()`; at rollout
time, by `forward_eval()` step-by-step with the cache reset at every
episode boundary (per pufferl.py:686-715).

For the SAME logical step within episode-N of the segment, the PE
indexing must match: forward_eval sees `pe[pos_within_episode]` (because
pos resets to 0 on every cache reset); forward must also see
`pe[pos_within_episode]` (via compute_pos_within_episode).

This test:
1. Runs forward_eval step-by-step over a length-T sequence, manually
   resetting state at the boundaries we care about (simulates pufferl).
2. Runs forward() over the entire sequence as a single (1, T) batch
   with `terminals` set at the same boundaries.
3. Asserts the two paths produce equivalent outputs.

Run: `python tests/test_pe_train_eval_consistency.py`
"""

import os
import sys
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pufferlib.models  # noqa: E402


class _DummyEncoder(nn.Module):
    """Same minimal stand-in used by test_transformer_kv_cache.py."""

    def __init__(self, obs_dim, hidden_size, num_actions=3):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.value_fn = nn.Linear(hidden_size, 1)
        self.is_continuous = False
        self.atn_dim = [num_actions]

    def encode_observations(self, observations, state=None):
        return torch.tanh(self.encoder(observations))

    def decode_actions(self, hidden):
        return self.actor(hidden), self.value_fn(hidden).squeeze(-1)


def _make_wrapper(horizon=8, hidden_size=16, obs_dim=8, num_layers=1, num_heads=2, seed=0):
    torch.manual_seed(seed)
    env = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(-1, 1, (obs_dim,), dtype=np.float32),
        single_action_space=gymnasium.spaces.Discrete(3),
        is_dict_obs=False,
        emulated=None,
    )
    inner = _DummyEncoder(obs_dim=obs_dim, hidden_size=hidden_size)
    wrapper = pufferlib.models.TransformerWrapper(
        env=env,
        policy=inner,
        input_size=hidden_size,
        hidden_size=hidden_size,
        horizon=horizon,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
        use_checkpointing=False,
    )
    wrapper.eval()
    return wrapper


def test_single_episode_per_row():
    """No mid-row terminals → forward and step-by-step forward_eval match."""
    horizon = 8
    T = 6  # less than horizon
    B = 1

    wrapper = _make_wrapper(horizon=horizon, seed=42)
    rng = torch.Generator().manual_seed(7)
    obs_seq = torch.randn(B, T, 8, generator=rng)  # (B, T, obs_dim)

    # Path A: forward_eval step-by-step
    state = {}
    eval_logits = []
    eval_values = []
    with torch.inference_mode():
        for t in range(T):
            logits, value = wrapper.forward_eval(obs_seq[:, t, :], state)
            eval_logits.append(logits)
            eval_values.append(value)

    # Path B: forward() over the whole sequence
    train_state = {
        "transformer_position": None,
        "transformer_context": None,
        "terminals": torch.zeros(B, T),  # all zeros = single episode
    }
    with torch.inference_mode():
        train_logits, train_values = wrapper(obs_seq, train_state)

    # Compare
    # forward output: train_logits (B*T, num_actions), train_values (B, T)
    # forward_eval output: eval_logits[t] (B, num_actions), eval_values[t] (B,)
    train_logits_BT = train_logits.view(B, T, -1)
    for t in range(T):
        diff_l = (train_logits_BT[:, t] - eval_logits[t]).abs().max().item()
        diff_v = (train_values[:, t] - eval_values[t]).abs().max().item()
        assert diff_l < 1e-4, f"single-ep step {t}: logits diff {diff_l:.2e}"
        assert diff_v < 1e-4, f"single-ep step {t}: values diff {diff_v:.2e}"
    print(f"  ok: single-episode-per-row, T={T} → train=eval bit-close")


def test_multi_episode_per_row():
    """Mid-row terminal at slot 2 → forward (with per-episode-reset PE)
    must match forward_eval after a manual cache reset at slot 3."""
    horizon = 8
    T = 6
    B = 1
    boundary = 2  # terminal at slot 2 → episode 1 starts at slot 3

    wrapper = _make_wrapper(horizon=horizon, seed=42)
    rng = torch.Generator().manual_seed(7)
    obs_seq = torch.randn(B, T, 8, generator=rng)

    # Path A: forward_eval, reset state after slot==boundary
    state = {}
    eval_logits = []
    eval_values = []
    with torch.inference_mode():
        for t in range(T):
            logits, value = wrapper.forward_eval(obs_seq[:, t, :], state)
            eval_logits.append(logits)
            eval_values.append(value)
            if t == boundary:
                # Simulate pufferl's done-handling: pos→0, cache rows zeroed.
                # This is what happens at an episode boundary in the rollout.
                state["transformer_position"] = torch.zeros(1, dtype=torch.long)
                kc = state.get("k_cache")
                vc = state.get("v_cache")
                if kc is not None:
                    for c in kc:
                        c.zero_()
                if vc is not None:
                    for c in vc:
                        c.zero_()

    # Path B: forward() with terminals[boundary]=1
    terminals = torch.zeros(B, T)
    terminals[0, boundary] = 1.0
    train_state = {
        "transformer_position": None,
        "transformer_context": None,
        "terminals": terminals,
    }
    with torch.inference_mode():
        train_logits, train_values = wrapper(obs_seq, train_state)

    # Compare per-slot. Episode 0 (slots 0..2): both paths see fresh
    # cache + pe[0..2]. Episode 1 (slots 3..5): forward_eval sees fresh
    # cache + pe[0..2]; forward should also see pe[0..2] via the
    # per-episode reset.
    # forward output: train_logits (B*T, num_actions), train_values (B, T)
    # forward_eval output: eval_logits[t] (B, num_actions), eval_values[t] (B,)
    train_logits_BT = train_logits.view(B, T, -1)
    for t in range(T):
        diff_l = (train_logits_BT[:, t] - eval_logits[t]).abs().max().item()
        diff_v = (train_values[:, t] - eval_values[t]).abs().max().item()
        # Episode mask in forward additionally blocks cross-episode
        # attention, which forward_eval naturally has post-reset (cache
        # is empty then refilled). So they should match.
        assert diff_l < 1e-4, f"multi-ep step {t}: logits diff {diff_l:.2e}"
        assert diff_v < 1e-4, f"multi-ep step {t}: values diff {diff_v:.2e}"
    print(f"  ok: multi-episode-per-row, T={T}, boundary at slot {boundary} → train=eval bit-close")


def test_multi_episode_three_episodes():
    """3 episodes per row: terminals at slots 1, 4."""
    horizon = 8
    T = 6
    B = 1
    boundaries = [1, 4]  # ep0 = {0,1}, ep1 = {2,3,4}, ep2 = {5}

    wrapper = _make_wrapper(horizon=horizon, seed=42)
    rng = torch.Generator().manual_seed(7)
    obs_seq = torch.randn(B, T, 8, generator=rng)

    # Path A: forward_eval with manual resets at boundaries
    state = {}
    eval_logits = []
    eval_values = []
    with torch.inference_mode():
        for t in range(T):
            logits, value = wrapper.forward_eval(obs_seq[:, t, :], state)
            eval_logits.append(logits)
            eval_values.append(value)
            if t in boundaries:
                state["transformer_position"] = torch.zeros(1, dtype=torch.long)
                kc = state.get("k_cache")
                vc = state.get("v_cache")
                if kc is not None:
                    for c in kc:
                        c.zero_()
                if vc is not None:
                    for c in vc:
                        c.zero_()

    # Path B: forward() with terminals at the same boundaries
    terminals = torch.zeros(B, T)
    for b in boundaries:
        terminals[0, b] = 1.0
    train_state = {
        "transformer_position": None,
        "transformer_context": None,
        "terminals": terminals,
    }
    with torch.inference_mode():
        train_logits, train_values = wrapper(obs_seq, train_state)

    # forward output: train_logits (B*T, num_actions), train_values (B, T)
    # forward_eval output: eval_logits[t] (B, num_actions), eval_values[t] (B,)
    train_logits_BT = train_logits.view(B, T, -1)
    for t in range(T):
        diff_l = (train_logits_BT[:, t] - eval_logits[t]).abs().max().item()
        diff_v = (train_values[:, t] - eval_values[t]).abs().max().item()
        assert diff_l < 1e-4, f"3-ep step {t}: logits diff {diff_l:.2e}"
        assert diff_v < 1e-4, f"3-ep step {t}: values diff {diff_v:.2e}"
    print(f"  ok: 3 episodes per row, boundaries at {boundaries} → train=eval bit-close")


def _run_all():
    test_single_episode_per_row()
    test_multi_episode_per_row()
    test_multi_episode_three_episodes()
    print("\ntest_pe_train_eval_consistency: PASS")


if __name__ == "__main__":
    _run_all()
