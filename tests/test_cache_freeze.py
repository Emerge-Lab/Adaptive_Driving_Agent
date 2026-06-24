"""Contract: KV cache slots written while an agent is off-map (removed=1) are
excluded from attention via per-agent garbage_mask.

The attention math is verified directly: we drive the transformer's
forward_eval with a synthetic state dict, mark some slots as garbage,
and assert that the attention weights at those slots are zero (after
softmax).

We use the _probe_attention path in models.py which captures attention
weights per layer in state["_attn_weights"].
"""

import os

# Without this, _USE_LEGACY_EVAL defaults True and the streaming KV path
# (which owns garbage_mask) is bypassed — the test would silently no-op.
os.environ.setdefault("PUFFER_TRANSFORMER_LEGACY_EVAL", "0")
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _MinimalPolicy(nn.Module):
    """Stub that satisfies TransformerWrapper's policy contract:
    encode_observations(obs, state) -> (B, hidden) and
    decode_actions(hidden) -> (logits, values)."""

    def __init__(self, obs_dim, hidden, n_actions):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, hidden)
        self.decoder_a = nn.Linear(hidden, n_actions)
        self.decoder_v = nn.Linear(hidden, 1)
        self.is_continuous = False

    def encode_observations(self, obs, state=None):
        return self.encoder(obs)

    def decode_actions(self, hidden):
        return self.decoder_a(hidden), self.decoder_v(hidden).squeeze(-1)


class _StubEnv:
    """Just exposes single_observation_space.shape — the only thing
    TransformerWrapper.__init__ reads from env."""

    def __init__(self, obs_dim):
        from gymnasium import spaces

        self.single_observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,))


def _make_wrapper(batch_size=4, horizon=8, hidden=16, n_heads=2):
    from pufferlib.models import TransformerWrapper

    env = _StubEnv(obs_dim=hidden)
    policy = _MinimalPolicy(obs_dim=hidden, hidden=hidden, n_actions=3)
    return TransformerWrapper(
        env=env,
        policy=policy,
        horizon=horizon,
        num_layers=2,
        num_heads=n_heads,
        input_size=hidden,
        hidden_size=hidden,
    )


def test_garbage_mask_excludes_slots_from_attention():
    """If garbage_mask[a, k] = True, the softmax weight at slot k for agent a
    must be 0 after the next forward."""
    torch.manual_seed(0)
    B, T, H = 4, 8, 16
    wrapper = _make_wrapper(batch_size=B, horizon=T, hidden=H, n_heads=2)
    wrapper.eval()

    state = wrapper.init_eval_state(batch_size=B, device="cpu", dtype=torch.float32)
    state["_probe_attention"] = True

    # Step 5 times so cache fills slots 0..4 for all agents. Agent 0 has
    # `removed=True` at steps 2 and 3 — slots 2, 3 should be marked garbage.
    for step in range(5):
        obs = torch.randn(B, H)
        removed = torch.zeros(B, dtype=torch.bool)
        if step in (2, 3):
            removed[0] = True
        state["removed"] = removed
        state["_attn_weights"] = []  # reset per step
        with torch.no_grad():
            wrapper.forward_eval(obs, state)

    # After step 4: garbage_mask[0, 2] and [0, 3] should be True
    gm = state["garbage_mask"]
    assert gm[0, 2].item() and gm[0, 3].item(), f"garbage slots not marked for agent 0: {gm[0]}"
    # Other agents: nothing marked
    assert not gm[1:].any().item(), f"non-removed agents should have empty garbage_mask: {gm[1:]}"

    # Now step once more (no removed) and inspect attention weights for agent 0
    state["removed"] = torch.zeros(B, dtype=torch.bool)
    state["_attn_weights"] = []
    obs = torch.randn(B, H)
    with torch.no_grad():
        wrapper.forward_eval(obs, state)

    # Per-layer attention weights are (B, H, 1, horizon).
    # Agent 0's slots 2 and 3 must have zero weight (masked out by garbage_mask).
    for layer_rec in state["_attn_weights"]:
        w = layer_rec["weights"]  # (B, H, 1, horizon)
        assert w[0, :, 0, 2].abs().max().item() < 1e-6, (
            f"layer {layer_rec['layer']}: slot 2 weight nonzero for agent 0: {w[0, :, 0, 2]}"
        )
        assert w[0, :, 0, 3].abs().max().item() < 1e-6, f"layer {layer_rec['layer']}: slot 3 weight nonzero for agent 0"
        # Sanity: other agents' slots 2, 3 should still get nonzero weight
        assert w[1, :, 0, 2].abs().max().item() > 1e-6, "agent 1 slot 2 should NOT be masked"


def test_garbage_mask_clears_on_full_reset():
    """reset_eval_state(state, done_indices=None) must zero garbage_mask."""
    B, T, H = 4, 8, 16
    wrapper = _make_wrapper(batch_size=B, horizon=T, hidden=H, n_heads=2)
    state = wrapper.init_eval_state(batch_size=B, device="cpu", dtype=torch.float32)
    state["garbage_mask"][:] = True
    wrapper.reset_eval_state(state, done_indices=None)
    assert not state["garbage_mask"].any().item(), "garbage_mask should be zeroed after full reset"


def test_garbage_mask_clears_per_agent_on_partial_reset():
    """reset_eval_state(state, done_indices=[a]) must zero garbage_mask[a]
    but leave other agents untouched."""
    B, T, H = 4, 8, 16
    wrapper = _make_wrapper(batch_size=B, horizon=T, hidden=H, n_heads=2)
    state = wrapper.init_eval_state(batch_size=B, device="cpu", dtype=torch.float32)
    state["garbage_mask"][:] = True
    wrapper.reset_eval_state(state, done_indices=torch.tensor([1, 3]))
    assert state["garbage_mask"][0].all().item(), "agent 0 garbage_mask should be unchanged"
    assert not state["garbage_mask"][1].any().item(), "agent 1 garbage_mask should be cleared"
    assert state["garbage_mask"][2].all().item(), "agent 2 garbage_mask should be unchanged"
    assert not state["garbage_mask"][3].any().item(), "agent 3 garbage_mask should be cleared"


def test_no_garbage_mask_no_regression():
    """Without `removed` in state, forward_eval must still work — model
    creates a fresh garbage_mask (all False), so attention is unchanged
    from the pre-fix behavior."""
    torch.manual_seed(0)
    B, T, H = 4, 8, 16
    wrapper = _make_wrapper(batch_size=B, horizon=T, hidden=H, n_heads=2)
    wrapper.eval()
    state = wrapper.init_eval_state(batch_size=B, device="cpu", dtype=torch.float32)
    # No state["removed"] key
    for _ in range(3):
        obs = torch.randn(B, H)
        with torch.no_grad():
            wrapper.forward_eval(obs, state)
    # garbage_mask stays all-False
    assert not state["garbage_mask"].any().item(), "without removed signal, no slots should be marked garbage"


if __name__ == "__main__":
    test_garbage_mask_excludes_slots_from_attention()
    test_garbage_mask_clears_on_full_reset()
    test_garbage_mask_clears_per_agent_on_partial_reset()
    test_no_garbage_mask_no_regression()
    print("test_cache_freeze: PASS")
