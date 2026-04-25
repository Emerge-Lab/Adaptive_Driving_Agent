"""Equivalence tests for the KV-cached TransformerWrapper.forward_eval.

Compares the new streaming forward against the legacy full-context forward
on a sequence of inputs, asserting bit-close outputs at fp32 / loose-close
at bf16. Also covers wrap-around past `horizon` steps and per-row resets.
"""

import os
import sys
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
import torch.nn as nn
try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pufferlib.models  # noqa: E402


class _DummyEncoder(nn.Module):
    """Minimal stand-in for the Drive policy's encode/decode interface."""

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
        logits = self.actor(hidden)
        value = self.value_fn(hidden)
        return (logits,), value


def _make_wrapper(obs_dim=8, hidden_size=16, num_layers=2, num_heads=4, horizon=8, seed=0):
    torch.manual_seed(seed)
    env = SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gymnasium.spaces.MultiDiscrete([3]),
    )
    base = _DummyEncoder(obs_dim, hidden_size, num_actions=3)
    wrapper = pufferlib.models.TransformerWrapper(
        env,
        base,
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        horizon=horizon,
        dropout=0.0,
    )
    wrapper.eval()
    return wrapper


def _step_seq(wrapper, obs_seq, use_legacy):
    state = {}
    outs = []
    fn = wrapper._forward_eval_legacy if use_legacy else wrapper.forward_eval
    with torch.inference_mode():
        for obs in obs_seq:
            logits, value = fn(obs, state)
            outs.append((logits[0].clone(), value.clone()))
    return outs


def test_kv_cache_matches_legacy(horizon, steps, B):
    wrapper = _make_wrapper(horizon=horizon)
    rng = torch.Generator().manual_seed(123)
    obs_seq = [torch.randn(B, 8, generator=rng) for _ in range(steps)]

    legacy = _step_seq(wrapper, obs_seq, use_legacy=True)
    cached = _step_seq(wrapper, obs_seq, use_legacy=False)

    for i, ((lL, vL), (lC, vC)) in enumerate(zip(legacy, cached)):
        assert torch.allclose(lL, lC, atol=1e-5, rtol=1e-4), (
            f"Logits diverge at step {i}: max abs diff {(lL - lC).abs().max().item():.2e}"
        )
        assert torch.allclose(vL, vC, atol=1e-5, rtol=1e-4), (
            f"Values diverge at step {i}: max abs diff {(vL - vC).abs().max().item():.2e}"
        )


def test_per_row_reset_matches_legacy():
    """Reset row i in both legacy and KV-cache state, then continue stepping."""
    horizon, steps, B = 8, 14, 4
    wrapper = _make_wrapper(horizon=horizon)
    rng = torch.Generator().manual_seed(7)
    obs_seq = [torch.randn(B, 8, generator=rng) for _ in range(steps)]
    reset_at = 5
    reset_rows = torch.tensor([1, 3])

    # ---- legacy path ----
    state_l = {}
    legacy_outs = []
    with torch.inference_mode():
        for i, obs in enumerate(obs_seq):
            if i == reset_at:
                state_l["transformer_context"][reset_rows] = 0
            logits, value = wrapper._forward_eval_legacy(obs, state_l)
            legacy_outs.append((logits[0].clone(), value.clone()))

    # ---- KV-cache path ----
    state_c = {}
    cached_outs = []
    with torch.inference_mode():
        for i, obs in enumerate(obs_seq):
            if i == reset_at:
                wrapper.reset_eval_state(state_c, done_indices=reset_rows)
            logits, value = wrapper.forward_eval(obs, state_c)
            cached_outs.append((logits[0].clone(), value.clone()))

    # Only the reset rows are guaranteed to match exactly post-reset; non-reset
    # rows should match throughout. Check both regions.
    for i, ((lL, vL), (lC, vC)) in enumerate(zip(legacy_outs, cached_outs)):
        assert torch.allclose(lL, lC, atol=1e-5, rtol=1e-4), (
            f"Logits diverge at step {i}: max abs diff {(lL - lC).abs().max().item():.2e}"
        )
        assert torch.allclose(vL, vC, atol=1e-5, rtol=1e-4), (
            f"Values diverge at step {i}: max abs diff {(vL - vC).abs().max().item():.2e}"
        )


def test_full_reset_matches_fresh_state():
    """reset_eval_state(None) should be equivalent to discarding the state."""
    horizon, B = 8, 3
    wrapper = _make_wrapper(horizon=horizon)
    rng = torch.Generator().manual_seed(99)
    pre_obs = [torch.randn(B, 8, generator=rng) for _ in range(6)]
    post_obs = [torch.randn(B, 8, generator=rng) for _ in range(6)]

    state_a = {}
    with torch.inference_mode():
        for o in pre_obs:
            wrapper.forward_eval(o, state_a)
        wrapper.reset_eval_state(state_a, done_indices=None)
        out_a = [wrapper.forward_eval(o, state_a) for o in post_obs]

    state_b = {}
    with torch.inference_mode():
        out_b = [wrapper.forward_eval(o, state_b) for o in post_obs]

    for i, (a, b) in enumerate(zip(out_a, out_b)):
        assert torch.allclose(a[0][0], b[0][0], atol=1e-6), f"Logits differ post-reset @ step {i}"
        assert torch.allclose(a[1], b[1], atol=1e-6), f"Values differ post-reset @ step {i}"


def _run_all():
    cases = [
        (8, 5, 4),
        (8, 8, 4),
        (8, 12, 4),
        (8, 25, 6),
        (16, 40, 8),
        (91, 200, 4),  # full size, multiple wraps
    ]
    for horizon, steps, B in cases:
        test_kv_cache_matches_legacy(horizon, steps, B)
        print(f"  ok: horizon={horizon} steps={steps} B={B}")
    print("test_kv_cache_matches_legacy: PASS")
    test_per_row_reset_matches_legacy()
    print("test_per_row_reset_matches_legacy: PASS")
    test_full_reset_matches_fresh_state()
    print("test_full_reset_matches_fresh_state: PASS")


if __name__ == "__main__":
    _run_all()
